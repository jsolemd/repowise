"""Workspace routing for the source-search lane.

The coordinator is a single-repository engine: every store it opens belongs to
one repo, and its ranking and confidence are calibrated per corpus. A
workspace serves several repos, so this module is where the difference lives —
a scoped call routes to that repo's coordinator, a federated call composes
several coordinators' finished answers. Nothing a coordinator decided is
recomputed or reweighted here; per-repo calibration is a gate-protected
contract, and this layer's whole job is to compose it honestly.

Fusion policy — envelope-ranked repo blocks. Per-repo fused scores are
rank-derived (RRF), so comparing them across corpora manufactures precision
that is not there: every repo's top hit scores alike by construction. Repos
are ranked instead by what IS comparable across corpora — the confidence
class each envelope asserts, the owner's own subject evidence, and dense
cosine under the one shared embedder — and the merged list presents the
winning repo's rows first with their internal order intact, then the rest in
envelope order.

The window is not the winner's block cut at ``limit``. A federated answer
that fills every slot from one repo has silently answered a workspace
question with a repository answer, so each remaining answering repo keeps a
reserved tail slot for its top row — the same bargain
:func:`~repowise.server.mcp_server.tool_search._append_symbol_backed` strikes
for files the page retrievers structurally cannot see.

Confidence composes downward, never upward: the federated class is the
winner's own, except when two repos are each confident of different owners —
that disagreement IS uncertainty, and the response says ``caution`` and lists
the competitors rather than picking one silently. When every repo declines,
the workspace declines; federation must not turn four abstentions into an
answer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

log = logging.getLogger(__name__)

_CONF_ORDER = {"confident": 2, "caution": 1, "no_match": 0}
_CAUTION = "caution"
_NO_MATCH = "no_match"


def _tag_scoped(response: dict[str, Any], alias: str) -> dict[str, Any]:
    """Stamp one repo's identity onto a coordinator response, in place."""
    for row in response.get("results") or []:
        row["repo"] = alias
    for cand in response.get("candidates") or []:
        cand["repo"] = alias
    owner = response.get("selected_owner")
    if owner:
        owner["repo"] = alias
    meta = response.setdefault("_meta", {})
    meta.setdefault("source_search", {})["answered_from"] = alias
    return response


def _owner_evidence(env: dict[str, Any]) -> dict[str, Any]:
    """The evidence dict belonging to the claim this envelope actually makes.

    Coverage is read from the selected owner and nowhere else, because that is
    the coordinator's own rule: evidence from another candidate cannot make
    *owner* trustworthy. An envelope with no owner asserts no ownership at all
    — its top row is the only thing it offers, and reading that keeps the key
    total instead of special-cased.
    """
    owner = env.get("selected_owner") or {}
    evidence = owner.get("evidence")
    if isinstance(evidence, dict):
        return evidence
    return ((env.get("results") or [{}])[0].get("evidence")) or {}


def _coverage(evidence: dict[str, Any], key: str) -> float:
    """One coverage fraction, or 0.0 where the envelope carries no such measure.

    Absent is deliberately not a penalty and not a bonus. An evidence dict
    without the coverage keys is the coordinator's real shape for "retrieval
    succeeded but the co-location evidence did not" — a state it already
    answers with ``caution``. Mapping it to 0.0 makes these keys inert when no
    leg measured a subject, so the ordering degrades to exactly what it was
    before this signal existed rather than to an accident of which corpus
    reported more fields.
    """
    value = evidence.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _envelope_strength(env: dict[str, Any]) -> tuple[int, int, int, float, float, float]:
    """What one repo's answer asserts, in cross-corpus-comparable terms only.

    Ordered the way the coordinator's own confidence gate composes the same
    evidence, because the honest cross-repo question is which owner best
    satisfies that gate — not which corpus the embedder happened to place
    nearer.

    1. ``confidence`` — the class the repo already committed to.
    2. ``exact_name`` — the query named the artifact; nothing outranks it.
    3. ``content_concept_coverage > 0`` — whether the owner carries any of the
       query's concepts in itself. A3 reads this as a boolean gate, not a
       magnitude: a filename may corroborate a subject the candidate already
       carries, but it may never constitute one, and a candidate whose entire
       case is its path cannot own the answer for the workspace either.
    4. ``concept_coverage`` — how much of the query's subject the owner
       matches at all. Coverage keeps A3's permissive reading here, so a file
       honestly named for its topic still counts for everything it is worth.
    5. ``content_concept_coverage`` again, now as a magnitude, to separate
       owners that cover the subject equally by how much of it they carry
       themselves.
    6. ``dense_cosine`` — the last resort it always was. It is the one signal
       that is numerically comparable across corpora and the one that says
       least about ownership, which is why it now runs after the evidence the
       coordinator actually gates on rather than in front of it.
    """
    conf = _CONF_ORDER.get(env.get("confidence"), 0)
    cosines: list[float] = []
    exact = 0
    owner = env.get("selected_owner") or {}
    for evidence in (
        owner.get("evidence") or {},
        ((env.get("results") or [{}])[0].get("evidence") or {}),
    ):
        if evidence.get("exact_name"):
            exact = 1
        cos = evidence.get("dense_cosine")
        if isinstance(cos, (int, float)):
            cosines.append(float(cos))
    claim = _owner_evidence(env)
    subject = _coverage(claim, "concept_coverage")
    carried = _coverage(claim, "content_concept_coverage")
    return (
        conf,
        exact,
        1 if carried > 0.0 else 0,
        subject,
        carried,
        max(cosines, default=-1.0),
    )


def _order_key(item: tuple[str, dict[str, Any]]) -> tuple:
    alias, env = item
    conf, exact, carries, subject, carried, cosine = _envelope_strength(env)
    return (-conf, -exact, -carries, -subject, -carried, -cosine, alias)


def _federated_window(
    blocks: list[tuple[str, list[dict[str, Any]]]], limit: int
) -> list[dict[str, Any]]:
    """Merge ranked repo blocks into one window that no repo can starve.

    The winner keeps the head and its internal order; every other answering
    repo is owed one tail slot for its top row, and the rows those slots
    displace are the weakest end of what was already there. Nothing about
    owner selection or within-repo order is touched — this decides only who
    is *visible*.

    When ``limit`` is at least the number of answering repos, every one of
    them is seated. Below that the window cannot pay everyone, and the debt
    is settled in envelope order from the weakest end: the winner never loses
    its top row, and the repos whose envelopes asserted least give way first.
    """
    ordered = [row for _, rows in blocks for row in rows]
    if limit <= 0:
        return []
    if len(ordered) <= limit:
        return ordered
    reserved = [rows[0] for _, rows in blocks[1:] if rows]
    if not reserved:
        return ordered[:limit]

    # Reserving a slot shrinks the head, which can un-seat a row that was
    # paying for its own reservation, so the head length is a fixpoint rather
    # than one subtraction. It only ever shrinks and never past one, so this
    # settles in at most one pass per reserved row. Rows are compared by
    # identity: two repos can return dicts that compare equal and still be two
    # different answers.
    head_len = limit
    while True:
        seated = {id(row) for row in ordered[:head_len]}
        owed = [row for row in reserved if id(row) not in seated]
        shrunk = max(1, limit - len(owed))
        if shrunk >= head_len:
            break
        head_len = shrunk
    return ordered[:head_len] + owed[: limit - head_len]


async def workspace_source_search(
    query: str,
    *,
    limit: int,
    mode: str,
    repo: str | None,
    registry: Any,
    build_meta: Callable[[], dict[str, Any]],
) -> dict[str, Any] | None:
    """Serve one source-search call in workspace mode, or ``None`` to fall
    through to the stock path (the same fail-soft contract as single-repo:
    a missing lane must degrade to the search that existed before it, never
    to an error the stock path would not have produced).

    ``repo=None`` scopes to the workspace's default repo and ``repo="all"``
    federates — the same resolution the wiki lane applies, so the two lanes
    cannot disagree about what an argument means.
    """
    from repowise.server.source_search_wiring import context_coordinator

    resolved = registry.resolve_repo_param(repo)

    if isinstance(resolved, str):
        ctx = await registry.get(resolved)
        coordinator = await context_coordinator(ctx)
        if coordinator is None:
            return None
        response = await coordinator.search(
            query, limit=limit, mode=mode, base_meta=build_meta()
        )
        return _tag_scoped(response, resolved)

    unavailable: dict[str, str] = {}
    members: list[tuple[str, Any]] = []
    for alias in resolved:
        try:
            members.append((alias, await registry.get(alias)))
        except Exception as exc:
            unavailable[alias] = f"context unavailable: {type(exc).__name__}"
    coordinators = await asyncio.gather(
        *(context_coordinator(ctx) for _, ctx in members)
    )
    live: list[tuple[str, Any]] = []
    for (alias, _), coordinator in zip(members, coordinators):
        if coordinator is None:
            unavailable[alias] = "source index unavailable"
        else:
            live.append((alias, coordinator))
    if not live:
        return None

    searches = await asyncio.gather(
        *(c.search(query, limit=limit, mode=mode, base_meta={}) for _, c in live),
        return_exceptions=True,
    )
    envelopes: list[tuple[str, dict[str, Any]]] = []
    for (alias, _), result in zip(live, searches):
        if isinstance(result, BaseException):
            log.warning(
                "source-search: federated leg %s failed: %r", alias, result
            )
            unavailable[alias] = f"search failed: {type(result).__name__}"
        else:
            envelopes.append((alias, result))
    if not envelopes:
        return None

    ranked = sorted(envelopes, key=_order_key)
    winner_alias, winner = ranked[0]

    confident = [
        (alias, env["selected_owner"])
        for alias, env in envelopes
        if env.get("confidence") == "confident" and env.get("selected_owner")
    ]
    conflict = len({alias for alias, _ in confident}) >= 2
    confidence = _CAUTION if conflict else (winner.get("confidence") or _CAUTION)

    blocks: list[tuple[str, list[dict[str, Any]]]] = []
    for alias, env in ranked:
        rows = env.get("results") or []
        for row in rows:
            row["repo"] = alias
        blocks.append((alias, list(rows)))
    results = _federated_window(blocks, limit)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for alias, env in ranked:
        for cand in env.get("candidates") or []:
            path = cand.get("path")
            if not path or (alias, path) in seen:
                continue
            seen.add((alias, path))
            candidates.append({"path": path, "repo": alias})
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break

    owner = winner.get("selected_owner")
    if owner:
        owner = dict(owner)
        owner["repo"] = winner_alias

    meta = build_meta()
    per_repo: dict[str, Any] = {}
    for alias, env in envelopes:
        per_repo[alias] = (env.get("_meta") or {}).get("source_search") or {}
    for alias, reason in unavailable.items():
        per_repo[alias] = {"unavailable": reason}
    meta["source_search"] = {
        "federated": True,
        "repo_order": [alias for alias, _ in ranked],
        "repos": per_repo,
    }

    response: dict[str, Any] = {
        "results": results,
        "mode": winner.get("mode", mode),
        "confidence": confidence,
        "selected_owner": owner,
        "_meta": meta,
    }
    if candidates:
        response["candidates"] = candidates
    if conflict:
        response["competing_owners"] = [
            {"repo": alias, "file": so.get("file"), "reason": so.get("reason")}
            for alias, so in confident
        ]
        response["note"] = (
            f"{len(confident)} repositories each returned a confident owner for "
            "this query. The federated confidence is caution until the query "
            "names one (repo=<alias>); competing_owners lists them all."
        )
    elif confidence == _NO_MATCH:
        response["note"] = (
            f"No indexed match for {query!r} in any workspace repository. The "
            "results (if any) are nearest neighbours, not evidence — "
            "_meta.source_search.repos names each corpus and commit consulted."
        )
    return response
