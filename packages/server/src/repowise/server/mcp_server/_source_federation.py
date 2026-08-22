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
class each envelope asserts, exact-name evidence, and dense cosine under the
one shared embedder — and the merged list presents the winning repo's rows
first with their internal order intact, then the rest in envelope order.

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


def _envelope_strength(env: dict[str, Any]) -> tuple[int, int, float]:
    """What one repo's answer asserts, in cross-corpus-comparable terms only."""
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
    return (conf, exact, max(cosines, default=-1.0))


def _order_key(item: tuple[str, dict[str, Any]]) -> tuple:
    alias, env = item
    conf, exact, cosine = _envelope_strength(env)
    return (-conf, -exact, -cosine, alias)


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

    results: list[dict[str, Any]] = []
    for alias, env in ranked:
        for row in env.get("results") or []:
            row["repo"] = alias
            results.append(row)
    results = results[:limit]

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
