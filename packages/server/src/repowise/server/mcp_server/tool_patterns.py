"""MCP tool: find_patterns — six named structural queries over the symbol graph.

Each pattern is a stated predicate, and the response carries it. That is
the whole point of naming them: an agent asking for ``hub_functions``
must get back the rule that produced the list, or it will read the list
against whatever "hub" means to it, which is not what ran.

Calling with no pattern — or with a name that is not one of the six —
returns the catalogue rather than an empty match list, because an empty
list is indistinguishable from "this repository has none".
"""

from __future__ import annotations

import time
from typing import Any

from repowise.core.analysis.patterns import (
    PATTERN_NAMES,
    SymbolGraph,
    UnknownPatternError,
    catalogue,
    run_pattern,
)
from repowise.core.persistence.crud.graph import get_all_graph_edges, get_all_graph_nodes
from repowise.core.persistence.database import get_session
from repowise.core.registry import mcp_tool_registry as mcp
from repowise.server.mcp_server._helpers import (
    _get_exclude_spec,
    _get_repo,
    _resolve_repo_context,
    _unsupported_repo_all,
    is_excluded,
)
from repowise.server.mcp_server._meta import build_meta as _build_meta

#: Matches per response. Each one carries a location block plus its
#: evidence, so this is where the transport budget lands.
_MAX_MATCHES = 30

#: Parameters each pattern accepts from the wire. Anything not listed is
#: reported in ``ignored_arguments`` rather than silently dropped: a
#: threshold that did not apply changes the answer, and the caller has to
#: be able to see that it did not apply.
_ACCEPTED: dict[str, frozenset[str]] = {
    "duplicate_signatures": frozenset({"min_files", "include_tests"}),
    "orphan_exports": frozenset({"include_tests", "min_lines"}),
    "hub_functions": frozenset({"min_callers", "percentile", "include_tests"}),
    "isolated_siblings": frozenset({"min_siblings", "include_tests"}),
    "reuse_candidates": frozenset({"min_directories", "include_tests"}),
    "bridge_functions": frozenset({"include_tests"}),
}


async def _load_graph(ctx: Any) -> tuple[SymbolGraph, Any]:
    """Build the pattern view from the persisted index for one repository."""
    exclude_spec = _get_exclude_spec(ctx.path)
    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)
        nodes = await get_all_graph_nodes(session, repository.id)
        edges = await get_all_graph_edges(session, repository.id)

    kept = [n for n in nodes if not is_excluded(_node_path(n), exclude_spec)]
    keep_ids = {n["node_id"] for n in kept}
    kept_edges = [
        e for e in edges if e["source_node_id"] in keep_ids and e["target_node_id"] in keep_ids
    ]
    return SymbolGraph.from_rows(kept, kept_edges), repository


def _node_path(node: dict[str, Any]) -> str:
    return str(node.get("file_path") or node.get("node_id") or "")


def _catalogue_response(reason: str, started: float, repository: Any = None) -> dict:
    return {
        "patterns": catalogue(),
        "available_patterns": list(PATTERN_NAMES),
        "note": reason,
        "_meta": _build_meta(
            timing_ms=(time.perf_counter() - started) * 1000, repository=repository
        ),
    }


@mcp.tool(default=False)
async def find_patterns(
    pattern: str | None = None,
    repo: str | None = None,
    limit: int = 20,
    include_tests: bool = False,
    min_files: int | None = None,
    min_lines: int | None = None,
    min_callers: int | None = None,
    min_siblings: int | None = None,
    min_directories: int | None = None,
    percentile: float | None = None,
) -> dict:
    """Named structural queries over the symbol graph, each with its definition.

    Call with no pattern to see the catalogue: six names, and for each one
    the question it answers, the exact predicate, how matches are ranked,
    and the adjacent surface it is most often confused with.

    The six:
      duplicate_signatures — the same callable declared in more than one file.
      orphan_exports       — public symbols nothing in the index uses.
      hub_functions        — widest inbound blast radius.
      isolated_siblings    — symbols that share a scope but touch none of it.
      reuse_candidates     — already called from several directories.
      bridge_functions     — the only path between two otherwise unlinked files.

    Args:
        pattern: one of the six names. Omit for the catalogue.
        repo: usually omitted.
        limit: max matches returned (clamped to 30).
        include_tests: include symbols defined in test files.
        min_files: duplicate_signatures — distinct files a group needs (default 2).
        min_lines: orphan_exports — smallest symbol to report (default 1).
        min_callers: hub_functions — absolute inbound floor (default 5).
        min_siblings: isolated_siblings — scope-mates needed (default 2).
        min_directories: reuse_candidates — external directories needed (default 2).
        percentile: hub_functions — inbound-degree percentile (default 95).
    """
    started = time.perf_counter()
    if repo == "all":
        return _unsupported_repo_all("find_patterns")

    if not pattern:
        return _catalogue_response(
            "No pattern requested. Pick one of the six names and call again.", started
        )
    if pattern not in _ACCEPTED:
        return _catalogue_response(
            f"{pattern!r} is not one of the six patterns. The catalogue is below; an "
            "empty match list would have been indistinguishable from 'none found'.",
            started,
        )

    requested_limit = limit
    limit = min(max(limit, 1), _MAX_MATCHES)

    supplied = {
        "include_tests": include_tests,
        "min_files": min_files,
        "min_lines": min_lines,
        "min_callers": min_callers,
        "min_siblings": min_siblings,
        "min_directories": min_directories,
        "percentile": percentile,
    }
    accepted = _ACCEPTED[pattern]
    params = {k: v for k, v in supplied.items() if v is not None and k in accepted}
    ignored = [
        {
            "argument": name,
            "reason": f"{name} does not apply to {pattern}; it was not used.",
        }
        for name, value in supplied.items()
        if value is not None and name not in accepted and name != "include_tests"
    ]

    ctx = await _resolve_repo_context(repo)
    graph, repository = await _load_graph(ctx)

    try:
        result = run_pattern(pattern, graph, **params)
    except UnknownPatternError:  # pragma: no cover — guarded above
        return _catalogue_response(f"{pattern!r} is not a known pattern.", started, repository)

    payload = result.as_dict(limit=limit)
    payload["available_patterns"] = list(PATTERN_NAMES)
    if not graph.symbols:
        payload["no_results_reason"] = (
            "This repository has no indexed symbols, so no pattern could match. This is "
            "not a finding about the code — run an index first."
        )
    if requested_limit > _MAX_MATCHES:
        payload["limit_note"] = (
            f"Requested limit={requested_limit} exceeded the transport budget and was "
            f"clamped to {_MAX_MATCHES}."
        )
    if ignored:
        payload["ignored_arguments"] = ignored
    payload["_meta"] = _build_meta(
        timing_ms=(time.perf_counter() - started) * 1000,
        repository=repository,
        targets=sorted({m.get("file", "") for m in payload["matches"] if m.get("file")}),
    )
    return payload
