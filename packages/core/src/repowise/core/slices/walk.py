"""The SQL-graph walk that turns entry points into a slice.

Substrate: ``graph_nodes`` and ``graph_edges``, the rows ingestion already
writes. No second graph engine, no in-memory rehydration of the whole
NetworkX graph — a task slice touches tens of nodes out of tens of thousands,
so the walk is a handful of indexed frontier queries and rehydrating the graph
to answer them would cost more than the whole slice.

Two rules the graph imposes, both from
:mod:`repowise.core.ingestion.models`:

* **The graph has two layers and they do not mix.** Files reach files through
  ``FILE_DEPENDENCY_EDGE_TYPES``; symbols reach symbols through
  ``SYMBOL_USE_EDGE_TYPES``; nothing points from a symbol back to a file. So
  the walk runs one BFS per layer and the frontier is grouped by layer on
  every round. Crossing layers happens once, at seeding
  (:func:`expand_seeds`), through the containment edges — which is the only
  bridge that exists.
* **Direction is asymmetric on purpose.** Downstream (what the seed reaches)
  is the code the task will read and change; upstream (what reaches the seed)
  is the ring it must not break. Walking both to the same depth buries the
  first in the second on any central node.

Resumability is why members carry ``frontier_down`` / ``frontier_up``. A node
discovered on the last round never had its own edges expanded, so it is where
the walk stopped. :func:`walk` accepts exactly those nodes back as a
:class:`WalkResume` and continues from them — extension adds members without
re-walking anything already walked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.exclusion import is_excluded
from repowise.core.ids import is_external
from repowise.core.ingestion.models import (
    CONTAINMENT_EDGE_TYPES,
    FILE_DEPENDENCY_EDGE_TYPES,
    SYMBOL_USE_EDGE_TYPES,
)
from repowise.core.persistence.models import GraphEdge, GraphNode
from repowise.core.slices.models import (
    DOWNSTREAM,
    UPSTREAM,
    EntryCandidate,
    SliceEdge,
    SliceMember,
    WalkPolicy,
)
from repowise.core.slices.nodes import is_test_node, node_path

logger = logging.getLogger(__name__)

#: Frontier chunk size for ``IN`` clauses. Matches the cap the dependency
#: tools use so a wide frontier stays portable to SQLite builds with the
#: older 999-variable limit.
_FRONTIER_CHUNK = 400

_EDGE_TYPES_BY_LAYER: dict[str, tuple[str, ...]] = {
    "file": tuple(sorted(FILE_DEPENDENCY_EDGE_TYPES)),
    "symbol": tuple(sorted(SYMBOL_USE_EDGE_TYPES)),
}


@dataclass
class WalkResume:
    """Where a previous walk stopped, so the next one starts there.

    ``visited`` is every node already in the slice: without it an extension
    re-adds members it already has, and their ``distance`` would be rewritten
    to whatever the resumed walk found rather than the shortest path from the
    original seeds.
    """

    visited: set[str] = field(default_factory=set)
    down_frontier: dict[str, int] = field(default_factory=dict)
    up_frontier: dict[str, int] = field(default_factory=dict)
    layers: dict[str, str] = field(default_factory=dict)


@dataclass
class WalkResult:
    """Everything one walk produced — new members only, never a re-report."""

    members: list[SliceMember] = field(default_factory=list)
    edges: list[SliceEdge] = field(default_factory=list)
    externals: list[str] = field(default_factory=list)
    #: Nodes whose edges this walk expanded. The caller clears their frontier
    #: flags: a node that has been expanded is no longer where the walk stops.
    expanded: dict[str, set[str]] = field(default_factory=dict)
    rounds: dict[str, int] = field(default_factory=dict)
    resumed_from: dict[str, int] = field(default_factory=dict)
    member_cap_hit: bool = False
    #: Frontier queries issued. Reported so a caller can prove an extension
    #: resumed rather than re-walked.
    queries: int = 0


@dataclass
class _Evidence:
    """What one direction's walk learned about one node."""

    distance: int
    edge_types: set[str] = field(default_factory=set)
    reached_from: list[str] = field(default_factory=list)


@dataclass
class _Discovery:
    node: GraphNode
    distance: int
    reference_count: int = 0
    max_confidence: float = 0.0
    edge_types: set[str] = field(default_factory=set)
    # Kept per direction, not merged: a node found both downstream and
    # upstream has two different stories about how it got here, and a single
    # merged set makes each reason claim edges the other direction supplied.
    by_direction: dict[str, _Evidence] = field(default_factory=dict)

    @property
    def directions(self) -> set[str]:
        return set(self.by_direction)


def _chunks(values: list[str]) -> list[list[str]]:
    return [values[i : i + _FRONTIER_CHUNK] for i in range(0, len(values), _FRONTIER_CHUNK)]


def member_from_node(
    node: GraphNode,
    *,
    distance: int,
    is_seed: bool = False,
    revision: int = 1,
) -> SliceMember:
    """Build a member from a persisted graph node row."""
    return SliceMember(
        node_id=node.node_id,
        node_type=node.node_type,
        layer=node.node_type,
        file_path=node_path(node),
        distance=distance,
        is_seed=is_seed,
        name=node.name,
        kind=node.kind,
        signature=node.signature,
        start_line=node.start_line,
        end_line=node.end_line,
        language=node.language or "",
        is_test=bool(node.is_test),
        pagerank=float(node.pagerank or 0.0),
        added_revision=revision,
    )


async def expand_seeds(
    session: AsyncSession,
    repo_id: str,
    candidates: list[EntryCandidate],
    policy: WalkPolicy,
    *,
    revision: int = 1,
    exclude_spec: Any = None,
) -> list[SliceMember]:
    """Turn entry candidates into seed members on both graph layers.

    A symbol candidate also starts the file layer from the file that defines
    it.  That supporting file is *not* promoted into a semantic seed and does
    not fan back out into sibling symbols: the caller named one exact call
    path, not every declaration beside it. A caller-named file candidate does
    seed up to ``seed_symbol_fanout`` of the symbols it defines, taken in
    pagerank order. This is the one place the walk crosses layers, through the
    containment edges (``defines`` / ``has_method``) that
    :mod:`repowise.core.ingestion.models` names as the only bridge.
    """
    seeds: dict[str, SliceMember] = {}
    wanted_paths: list[str] = []
    explicit_file_seeds: set[str] = set()

    for candidate in candidates:
        row = await session.execute(
            select(GraphNode).where(
                GraphNode.repository_id == repo_id,
                GraphNode.node_id == candidate.node_id,
            )
        )
        node = row.scalar_one_or_none()
        if node is None or is_external(node.node_id):
            continue
        member = member_from_node(node, distance=0, is_seed=True, revision=revision)
        member.seed_score = candidate.score
        member.query_hits = len(candidate.matched_terms)
        why = "; ".join(candidate.reasons[:3]) or "requested entry point"
        member.reasons.append(f"entry point ({why})")
        seeds[member.node_id] = member
        if node.node_type == "symbol":
            wanted_paths.append(member.file_path)
        elif node.node_type == "file":
            explicit_file_seeds.add(member.node_id)

    # A symbol seed drags in its own file, so the file layer has somewhere to
    # start. Without this a symbol-only query produces a slice with no file
    # members at all and the file-dependency layer is never walked.
    for path in wanted_paths:
        if path in seeds or is_excluded(path, exclude_spec):
            continue
        row = await session.execute(
            select(GraphNode).where(
                GraphNode.repository_id == repo_id,
                GraphNode.node_id == path,
                GraphNode.node_type == "file",
            )
        )
        node = row.scalar_one_or_none()
        if node is None:
            continue
        member = member_from_node(node, distance=0, is_seed=False, revision=revision)
        member.reasons.append("contains the exact seed symbol; starts the file dependency layer")
        seeds[member.node_id] = member

    if policy.seed_symbol_fanout and explicit_file_seeds:
        # Only files the caller/task actually selected fan out. A file added to
        # support an exact symbol must not manufacture sibling seeds.
        file_seed_ids = sorted(explicit_file_seeds)
        for chunk in _chunks(file_seed_ids):
            rows = await session.execute(
                select(GraphEdge, GraphNode)
                .join(
                    GraphNode,
                    and_(
                        GraphNode.repository_id == GraphEdge.repository_id,
                        GraphNode.node_id == GraphEdge.target_node_id,
                    ),
                )
                .where(
                    GraphEdge.repository_id == repo_id,
                    GraphEdge.source_node_id.in_(chunk),
                    GraphEdge.edge_type.in_(tuple(sorted(CONTAINMENT_EDGE_TYPES))),
                    GraphNode.node_type == "symbol",
                )
            )
            by_file: dict[str, list[GraphNode]] = {}
            for edge, node in rows.all():
                by_file.setdefault(edge.source_node_id, []).append(node)
            for source, nodes in by_file.items():
                nodes.sort(key=lambda n: (-float(n.pagerank or 0.0), n.node_id))
                for node in nodes[: policy.seed_symbol_fanout]:
                    if node.node_id in seeds or is_external(node.node_id):
                        continue
                    if not policy.include_tests and is_test_node(node):
                        continue
                    member = member_from_node(node, distance=0, is_seed=True, revision=revision)
                    member.reasons.append(f"defined by seed file {source}")
                    seeds[member.node_id] = member

    return list(seeds.values())


async def _frontier_round(
    session: AsyncSession,
    repo_id: str,
    frontier: dict[str, int],
    layers: dict[str, str],
    *,
    inbound: bool,
    policy: WalkPolicy,
    exclude_spec: Any,
    visited: set[str],
    discoveries: dict[str, _Discovery],
    externals: set[str],
    edges_out: list[SliceEdge],
) -> tuple[dict[str, int], int]:
    """One BFS round. Returns the next frontier and the query count."""
    by_layer: dict[str, list[str]] = {}
    for node_id in frontier:
        by_layer.setdefault(layers.get(node_id, "file"), []).append(node_id)

    direction = UPSTREAM if inbound else DOWNSTREAM
    next_frontier: dict[str, int] = {}
    queries = 0

    for layer, node_ids in by_layer.items():
        edge_types = _EDGE_TYPES_BY_LAYER.get(layer)
        if not edge_types:
            continue
        near_column = GraphEdge.target_node_id if inbound else GraphEdge.source_node_id
        far_column = GraphEdge.source_node_id if inbound else GraphEdge.target_node_id

        for chunk in _chunks(sorted(node_ids)):
            conditions = [
                GraphEdge.repository_id == repo_id,
                near_column.in_(chunk),
                GraphEdge.edge_type.in_(edge_types),
                GraphNode.node_type == layer,
            ]
            if not policy.include_tests:
                conditions.append(GraphNode.is_test.is_(False))
            if policy.min_edge_confidence > 0.0:
                conditions.append(GraphEdge.confidence >= policy.min_edge_confidence)

            rows = await session.execute(
                select(GraphEdge, GraphNode)
                .join(
                    GraphNode,
                    and_(
                        GraphNode.repository_id == GraphEdge.repository_id,
                        GraphNode.node_id == far_column,
                    ),
                )
                .where(*conditions)
            )
            queries += 1
            # Sorted in Python rather than SQL for the reason the dependency
            # tools record: an ORDER BY here flips SQLite onto the unique
            # (repo, source, target, type) index and scans every repo edge.
            batch = sorted(
                rows.all(),
                key=lambda pair: (
                    pair[0].source_node_id,
                    pair[0].target_node_id,
                    pair[0].edge_type,
                ),
            )
            for edge, node in batch:
                near = edge.target_node_id if inbound else edge.source_node_id
                far = edge.source_node_id if inbound else edge.target_node_id
                distance = frontier[near] + 1

                edges_out.append(
                    SliceEdge(
                        source=edge.source_node_id,
                        target=edge.target_node_id,
                        edge_type=edge.edge_type,
                        confidence=float(edge.confidence or 0.0),
                        direction=direction,
                    )
                )

                if is_external(far):
                    externals.add(far)
                    continue
                if far in visited or is_excluded(node_path(node), exclude_spec):
                    continue
                if not policy.include_tests and is_test_node(node):
                    continue

                disc = discoveries.get(far)
                if disc is None:
                    disc = _Discovery(node=node, distance=distance)
                    discoveries[far] = disc
                disc.distance = min(disc.distance, distance)
                disc.reference_count += 1
                disc.max_confidence = max(disc.max_confidence, float(edge.confidence or 0.0))
                disc.edge_types.add(edge.edge_type)

                evidence = disc.by_direction.get(direction)
                if evidence is None:
                    evidence = _Evidence(distance=distance)
                    disc.by_direction[direction] = evidence
                evidence.distance = min(evidence.distance, distance)
                evidence.edge_types.add(edge.edge_type)
                if near not in evidence.reached_from:
                    evidence.reached_from.append(near)

                next_frontier[far] = min(next_frontier.get(far, distance), distance)
                layers[far] = node.node_type

    return next_frontier, queries


def _reason_for(evidence: _Evidence, direction: str) -> str:
    kinds = ", ".join(sorted(evidence.edge_types)) or "depends on"
    anchors = ", ".join(evidence.reached_from[:2])
    extra = len(evidence.reached_from) - 2
    more = f" (+{extra} more)" if extra > 0 else ""
    if direction == UPSTREAM:
        return f"{kinds} → {anchors}{more}; reaches the slice at distance {evidence.distance}"
    return f"{kinds} from {anchors}{more}; reached by the slice at distance {evidence.distance}"


async def walk(
    session: AsyncSession,
    repo_id: str,
    *,
    seed_members: list[SliceMember],
    policy: WalkPolicy,
    resume: WalkResume | None = None,
    exclude_spec: Any = None,
    revision: int = 1,
    existing_count: int = 0,
) -> WalkResult:
    """Walk the SQL graph from the seeds (or from a stored frontier).

    ``resume`` carries a previous walk's stopping point. When it is given the
    seeds are *additions*, not a restart: everything in ``resume.visited``
    stays out of the result, and the frontiers it names are expanded at the
    distance they already had.
    """
    policy = policy.clamped()
    result = WalkResult()

    visited: set[str] = set(resume.visited) if resume else set()
    layers: dict[str, str] = dict(resume.layers) if resume else {}
    down: dict[str, int] = dict(resume.down_frontier) if resume else {}
    up: dict[str, int] = dict(resume.up_frontier) if resume else {}

    for member in seed_members:
        visited.add(member.node_id)
        layers[member.node_id] = member.layer
        down[member.node_id] = member.distance
        up[member.node_id] = member.distance

    result.resumed_from = {
        DOWNSTREAM: len(resume.down_frontier) if resume else 0,
        UPSTREAM: len(resume.up_frontier) if resume else 0,
    }

    discoveries: dict[str, _Discovery] = {}
    externals: set[str] = set()
    edges_out: list[SliceEdge] = []
    expanded: dict[str, set[str]] = {DOWNSTREAM: set(), UPSTREAM: set()}
    room = max(0, policy.max_members - existing_count - len(seed_members))

    for direction, frontier, depth in (
        (DOWNSTREAM, down, policy.downstream_depth),
        (UPSTREAM, up, policy.upstream_depth),
    ):
        rounds = 0
        current = dict(frontier)
        # Per-direction, because a node the downstream walk already has must
        # not stop the upstream walk from recording that it also reaches the
        # slice from above — the two facts are merged in ``discoveries``.
        seen_here = set(visited)
        for _ in range(depth):
            if not current or len(discoveries) >= room:
                break
            expanded[direction].update(current)
            current, queries = await _frontier_round(
                session,
                repo_id,
                current,
                layers,
                inbound=(direction == UPSTREAM),
                policy=policy,
                exclude_spec=exclude_spec,
                visited=seen_here,
                discoveries=discoveries,
                externals=externals,
                edges_out=edges_out,
            )
            result.queries += queries
            rounds += 1
            seen_here.update(current)
            if len(discoveries) >= room:
                result.member_cap_hit = True
                break
        result.rounds[direction] = rounds

    found_down = {n for n, d in discoveries.items() if DOWNSTREAM in d.directions}
    found_up = {n for n, d in discoveries.items() if UPSTREAM in d.directions}
    unexpanded_down = found_down - expanded[DOWNSTREAM]
    unexpanded_up = found_up - expanded[UPSTREAM]

    ordered = sorted(discoveries.items(), key=lambda kv: (kv[1].distance, kv[0]))
    if len(ordered) > room:
        ordered = ordered[:room]
        result.member_cap_hit = True
    if room == 0 and (down or up) and (policy.downstream_depth or policy.upstream_depth):
        # Nothing was even attempted: the slice was already at ``max_members``
        # before this walk began. Saying so is the difference between "the
        # graph ends here" and "your ceiling does".
        result.member_cap_hit = True

    kept_ids = {node_id for node_id, _ in ordered}
    for node_id, disc in ordered:
        member = member_from_node(disc.node, distance=disc.distance, revision=revision)
        member.reference_count = disc.reference_count
        member.max_confidence = disc.max_confidence
        member.edge_types = set(disc.edge_types)
        member.frontier_down = node_id in unexpanded_down
        member.frontier_up = node_id in unexpanded_up
        for direction in sorted(disc.by_direction):
            member.reasons.append(_reason_for(disc.by_direction[direction], direction))
        result.members.append(member)

    # A seed is a frontier in a direction it was never expanded in — a depth-0
    # walk expands nothing, and that is exactly the slice an extension has to
    # be able to grow from.
    seed_ids = {m.node_id for m in seed_members}
    for member in seed_members:
        member.frontier_down = member.node_id not in expanded[DOWNSTREAM]
        member.frontier_up = member.node_id not in expanded[UPSTREAM]

    keep = kept_ids | seed_ids | visited
    result.edges = [
        e
        for e in _dedupe_edges(edges_out)
        if (e.source in keep or is_external(e.source))
        and (e.target in keep or is_external(e.target))
    ]
    result.externals = sorted(externals)
    result.expanded = expanded
    return result


def _dedupe_edges(edges: list[SliceEdge]) -> list[SliceEdge]:
    seen: dict[tuple[str, str, str], SliceEdge] = {}
    for edge in edges:
        seen.setdefault((edge.source, edge.target, edge.edge_type), edge)
    return sorted(seen.values(), key=lambda e: (e.source, e.target, e.edge_type))
