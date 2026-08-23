"""Build, fetch, extend and render a task slice.

The orchestration layer: entry-point resolution → SQL-graph walk → ranking →
persistence, and separately, a stored slice → view → budget → payload.

Two decisions here are worth stating because they are what make a slice
*persistent* rather than a one-shot query result.

**Building and rendering are separate.** A slice is walked once and stored
whole; a render decides a fidelity and a budget over the stored members. So
the same slice can be read as two hundred cards now and six full sources in a
minute without re-walking, and the budget is never allowed to decide what the
slice *contains* — only what this response *shows*, which is why every drop is
recoverable by re-rendering rather than by rebuilding.

**Extension resumes; it does not rebuild.** :func:`extend_slice` reloads the
stored frontier — the members whose own edges were never expanded — and walks
from exactly those, at exactly the distance they already carry. Members
already in the slice are never re-discovered, and their distances (shortest
path from the *original* seeds) survive untouched. The result reports
``resumed_from`` so a caller can verify that, rather than take it on faith.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.persistence.models import GraphNode
from repowise.core.slices import views
from repowise.core.slices.budget import BudgetReport, fit_members, payload_tokens
from repowise.core.slices.entry_points import (
    count_query_hits,
    query_terms,
    resolve_entry_points,
)
from repowise.core.slices.errors import RepositoryNotIndexedError
from repowise.core.slices.models import (
    DOWNSTREAM,
    UPSTREAM,
    SliceEdge,
    SliceMember,
    SliceRecord,
    WalkPolicy,
)
from repowise.core.slices.ranking import rank_members, ranking_contract
from repowise.core.slices.store import SliceStore, new_slice_id
from repowise.core.slices.walk import WalkResume, expand_seeds, walk

#: Default ceiling on entry points nominated from a task query. Five is enough
#: for a real task to reach its subject from more than one angle and few
#: enough that a vague query does not seed the walk from half the repo.
DEFAULT_ENTRY_POINT_LIMIT = 5

#: Slice edges carried on a render when the caller asks for the subgraph.
_EDGE_CAP = 80

#: External dependencies named on a render. Both cuts are disclosed.
_EXTERNAL_CAP = 25

#: SQLite parameter-limit-safe batch for the per-render presence check.
_PRESENCE_BATCH = 400


def annotate_query_hits(members: list[SliceMember], task: str) -> None:
    """Give every member the task-term signal the ranking's ``query`` term reads.

    Seeds already carry it from their candidacy scoring; reached members would
    otherwise score zero on it, so a file the task names by name would rank
    below one it never mentions purely because the walk reached the second
    first.
    """
    terms = query_terms(task)
    if not terms:
        return
    for member in members:
        member.query_hits = max(
            member.query_hits,
            count_query_hits(terms, member.name, member.file_path, member.node_id),
        )


async def _present_nodes(session: AsyncSession, repo_id: str, node_ids: list[str]) -> set[str]:
    """Which of *node_ids* the index still has.

    One batched read per render, because a stored slice outlives the index it
    was cut from. Re-indexing rewrites every graph row; a member whose node
    vanished must come back labelled rather than come back looking current.
    """
    present: set[str] = set()
    for start in range(0, len(node_ids), _PRESENCE_BATCH):
        chunk = node_ids[start : start + _PRESENCE_BATCH]
        rows = await session.execute(
            select(GraphNode.node_id).where(
                GraphNode.repository_id == repo_id,
                GraphNode.node_id.in_(chunk),
            )
        )
        present.update(rows.scalars().all())
    return present


async def _assert_indexed(session: AsyncSession, repo_id: str) -> None:
    count = await session.scalar(
        select(func.count()).select_from(GraphNode).where(GraphNode.repository_id == repo_id)
    )
    if not count:
        raise RepositoryNotIndexedError(repo_id)


async def build_slice(
    session: AsyncSession,
    *,
    repo_id: str,
    repo_path: str,
    task: str,
    store: SliceStore,
    entry_points: list[str] | None = None,
    policy: WalkPolicy | None = None,
    exclude_spec: Any = None,
    entry_point_limit: int = DEFAULT_ENTRY_POINT_LIMIT,
) -> SliceRecord:
    """Walk a new slice for *task* and persist it under a fresh id."""
    await _assert_indexed(session, repo_id)
    effective = (policy or WalkPolicy()).clamped()

    candidates = await resolve_entry_points(
        session,
        repo_id,
        task=task,
        entry_points=entry_points,
        limit=max(1, entry_point_limit),
        include_tests=effective.include_tests,
        exclude_spec=exclude_spec,
    )
    seed_members = await expand_seeds(
        session, repo_id, candidates, effective, revision=1, exclude_spec=exclude_spec
    )

    result = await walk(
        session,
        repo_id,
        seed_members=seed_members,
        policy=effective,
        exclude_spec=exclude_spec,
        revision=1,
    )

    all_members = seed_members + result.members
    annotate_query_hits(all_members, task)
    members = rank_members(all_members)
    record = SliceRecord(
        slice_id=new_slice_id(),
        repository_id=repo_id,
        repo_path=str(repo_path),
        task=task,
        policy=effective,
        seeds=[c.node_id for c in candidates],
        members=members,
        edges=result.edges,
        external_dependencies=result.externals,
        revision=1,
        member_cap_hit=result.member_cap_hit,
    )
    store.save(
        record,
        event={
            "kind": "build",
            "task": task,
            "seeds": record.seeds,
            "entry_points_requested": entry_points or [],
            "members_added": len(members),
            "rounds": result.rounds,
            "frontier_queries": result.queries,
        },
    )
    record.events = [
        {"seq": 1, "kind": "build", "at": record.created_at, "members_added": len(members)}
    ]
    return record


def load_slice(store: SliceStore, slice_id: str) -> SliceRecord:
    """Re-fetch a stored slice. Raises ``SliceNotFoundError`` for an unknown id."""
    return store.load(slice_id)


async def extend_slice(
    session: AsyncSession,
    *,
    store: SliceStore,
    slice_id: str,
    extra_downstream: int = 1,
    extra_upstream: int = 0,
    entry_points: list[str] | None = None,
    task_addendum: str | None = None,
    exclude_spec: Any = None,
    max_members: int | None = None,
) -> tuple[SliceRecord, dict[str, Any]]:
    """Grow a stored slice from its frontier. Returns ``(record, extension)``.

    ``extension`` is the audit trail: how many frontier nodes were resumed
    from, how many rounds ran, how many members were added, and how many
    frontier queries it cost. A caller that wants to prove the extension did
    not re-walk reads ``resumed_from`` and ``re_walked``.
    """
    record = store.load(slice_id)
    await _assert_indexed(session, record.repository_id)

    revision = record.revision + 1
    base_policy = record.policy
    step = WalkPolicy(
        downstream_depth=max(0, extra_downstream),
        upstream_depth=max(0, extra_upstream),
        include_tests=base_policy.include_tests,
        seed_symbol_fanout=base_policy.seed_symbol_fanout,
        max_members=max_members or base_policy.max_members,
        min_edge_confidence=base_policy.min_edge_confidence,
    ).clamped()

    new_seeds: list[SliceMember] = []
    new_seed_ids: list[str] = []
    if entry_points or task_addendum:
        candidates = await resolve_entry_points(
            session,
            record.repository_id,
            task=task_addendum or record.task,
            entry_points=entry_points,
            limit=DEFAULT_ENTRY_POINT_LIMIT,
            include_tests=step.include_tests,
            exclude_spec=exclude_spec,
        )
        existing = {m.node_id for m in record.members}
        candidates = [c for c in candidates if c.node_id not in existing]
        if candidates:
            new_seeds = await expand_seeds(
                session,
                record.repository_id,
                candidates,
                step,
                revision=revision,
                exclude_spec=exclude_spec,
            )
            new_seeds = [m for m in new_seeds if m.node_id not in existing]
            new_seed_ids = [m.node_id for m in new_seeds]

    resume = WalkResume(
        visited={m.node_id for m in record.members},
        down_frontier=record.frontier(DOWNSTREAM),
        up_frontier=record.frontier(UPSTREAM),
        layers={m.node_id: m.layer for m in record.members},
    )

    result = await walk(
        session,
        record.repository_id,
        seed_members=new_seeds,
        policy=step,
        resume=resume,
        exclude_spec=exclude_spec,
        revision=revision,
        existing_count=len(record.members),
    )

    # A member whose edges this walk expanded is no longer where the walk
    # stops. Leaving the flag set would make the next extension re-expand it
    # and pay for edges it already has.
    for member in record.members:
        if member.node_id in result.expanded.get(DOWNSTREAM, set()):
            member.frontier_down = False
        if member.node_id in result.expanded.get(UPSTREAM, set()):
            member.frontier_up = False

    added = new_seeds + result.members
    annotate_query_hits(added, task_addendum or record.task)
    record.members = rank_members(record.members + added)
    record.edges = _merge_edges(record.edges, result.edges)
    record.external_dependencies = sorted(set(record.external_dependencies) | set(result.externals))
    record.revision = revision
    record.member_cap_hit = record.member_cap_hit or result.member_cap_hit
    record.seeds = list(dict.fromkeys(record.seeds + new_seed_ids))
    record.policy = WalkPolicy(
        downstream_depth=base_policy.downstream_depth + step.downstream_depth,
        upstream_depth=base_policy.upstream_depth + step.upstream_depth,
        include_tests=base_policy.include_tests,
        seed_symbol_fanout=base_policy.seed_symbol_fanout,
        max_members=step.max_members,
        min_edge_confidence=base_policy.min_edge_confidence,
    )

    extension = {
        "revision": revision,
        "members_added": len(added),
        "seeds_added": new_seed_ids,
        "resumed_from": result.resumed_from,
        "rounds": result.rounds,
        "frontier_queries": result.queries,
        "re_walked": False,
        "member_cap_hit": result.member_cap_hit,
        "extended_at": time.time(),
    }
    store.save(record, event={"kind": "extend", **extension})
    record.events.append({"seq": len(record.events) + 1, "kind": "extend", **extension})
    return record, extension


def _merge_edges(existing: list[SliceEdge], new: list[SliceEdge]) -> list[SliceEdge]:
    merged: dict[tuple[str, str, str], SliceEdge] = {
        (e.source, e.target, e.edge_type): e for e in existing
    }
    for edge in new:
        merged.setdefault((edge.source, edge.target, edge.edge_type), edge)
    return sorted(merged.values(), key=lambda e: (e.source, e.target, e.edge_type))


async def render_slice(
    session: AsyncSession,
    record: SliceRecord,
    *,
    view: str = "skeleton",
    budget_tokens: int = 6000,
    max_source_lines: int = views.DEFAULT_MAX_SOURCE_LINES,
    repo_root: Path | str | None = None,
    include_edges: bool = False,
    extension: dict[str, Any] | None = None,
    on_budget: Callable[[BudgetReport], None] | None = None,
) -> dict[str, Any]:
    """Serialise a stored slice at one fidelity, inside one token budget.

    ``on_budget`` receives the *complete* :class:`BudgetReport` — including
    every drop, not the capped list the response itemises. The MCP layer uses
    it to push the full set into the omission store, so a slice that drops
    three hundred members still discloses a bounded list on the wire and stays
    recoverable in full.
    """
    resolved_view = views.normalize_view(view)
    root = repo_root if repo_root is not None else (record.repo_path or None)

    ctx = await views.prepare(
        session,
        record.repository_id,
        record.members,
        view=resolved_view,
        repo_root=root,
        max_source_lines=max_source_lines,
    )
    present = await _present_nodes(
        session, record.repository_id, [m.node_id for m in record.members]
    )

    priced: list[tuple[SliceMember, dict[str, Any]]] = []
    stale: list[str] = []
    for member in sorted(record.members, key=lambda m: m.rank):
        payload = views.render_member(member, resolved_view, ctx)
        if member.node_id not in present:
            stale.append(member.node_id)
            payload["stale"] = True
            payload["stale_reason"] = (
                f"no longer in the index — removed or renamed since this slice was "
                f"cut at revision {member.added_revision}"
            )
        priced.append((member, payload))

    envelope: dict[str, Any] = {
        "status": "ok",
        "slice_id": record.slice_id,
        "task": record.task,
        "view": resolved_view,
        "summary": record.summary(),
        "seeds": list(record.seeds),
        "ranking": ranking_contract(),
        "externals": record.external_dependencies[:_EXTERNAL_CAP],
        "history": record.events[-5:],
    }
    if len(record.external_dependencies) > _EXTERNAL_CAP:
        envelope["externals_truncated"] = len(record.external_dependencies) - _EXTERNAL_CAP
    if extension is not None:
        envelope["extension"] = extension
    if include_edges:
        # Ranked before it is cut: the strongest relations survive, ties broken
        # on the endpoints so two renders of one slice never disagree. An
        # alphabetical cut would hand back whichever edges sort first, which is
        # not the same question as which edges matter.
        ranked_edges = sorted(
            record.edges, key=lambda e: (-e.confidence, e.source, e.target, e.edge_type)
        )
        envelope["edges"] = [e.to_dict() for e in ranked_edges[:_EDGE_CAP]]
        envelope["edges_ranked_by"] = "confidence"
        if len(ranked_edges) > _EDGE_CAP:
            envelope["edges_truncated"] = len(ranked_edges) - _EDGE_CAP
    if record.member_cap_hit:
        envelope["walk_truncated"] = (
            f"the walk stopped at max_members={record.policy.max_members}; "
            f"narrow the task or lower the depth for a complete subgraph"
        )
    if stale:
        # A slice outlives the index it was cut from, which is the point — but
        # a member the index no longer has must be handed back labelled, not
        # handed back as if it were still there. Each one carries ``stale`` on
        # its own payload; this is the summary the reader sees first.
        envelope["index_drift"] = (
            f"{len(stale)} of {len(record.members)} member(s) are no longer in the "
            f"index and are marked `stale`. The repository was re-indexed after this "
            f"slice was cut; rebuild it with build_task_slice to pick up the new shape."
        )
        envelope["summary"]["stale_members"] = len(stale)

    envelope_tokens = payload_tokens(envelope)
    members, report = fit_members(
        priced,
        view=resolved_view,
        budget_tokens=budget_tokens,
        envelope_tokens=envelope_tokens,
    )
    if on_budget is not None:
        on_budget(report)
    envelope["members"] = members
    envelope["budget"] = report.to_dict()
    envelope["next"] = _next_hint(record, report)
    return envelope


def _next_hint(record: SliceRecord, report: BudgetReport) -> str:
    if report.truncated:
        return (
            f"{len(report.dropped)} member(s) were not shown. Re-fetch "
            f"get_task_slice(slice_id='{record.slice_id}') with a larger budget_tokens "
            f"or view='card'."
        )
    frontier = len(record.frontier(DOWNSTREAM)) + len(record.frontier(UPSTREAM))
    if frontier:
        return (
            f"{frontier} member(s) sit on the walk frontier. "
            f"extend_task_slice(slice_id='{record.slice_id}') grows the slice from "
            f"there without re-walking."
        )
    return f"Slice {record.slice_id} is fully expanded at its current policy."
