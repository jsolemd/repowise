"""MCP Tool: get_execution_flows — trace how the codebase executes.

Hybrid approach: reads persisted entry-point scores from community_meta_json,
then recomputes BFS call-path traces on demand from stored call edges. This
avoids a dedicated execution_flows table while keeping the expensive scoring
off the hot path.
"""

from __future__ import annotations

import time
from itertools import pairwise
from typing import Any

from repowise.core.persistence.crud import (
    get_graph_node,
    get_top_entry_points,
)
from repowise.core.persistence.database import get_session
from repowise.core.persistence.models import GraphNode
from repowise.core.registry import mcp_tool_registry as mcp
from repowise.server.mcp_server._flow_ids import (
    ParsedFlowId,
    index_generation,
    mint_flow_id,
    parse_flow_id,
)
from repowise.server.mcp_server._graph_utils import (
    bfs_trace,
    resolve_trace_communities,
)
from repowise.server.mcp_server._graph_utils import (
    entry_point_score as _ep_score,
)
from repowise.server.mcp_server._helpers import (
    _get_exclude_spec,
    _get_repo,
    _resolve_repo_context,
    _unsupported_repo_all,
    filter_embedded_path_ids,
    is_excluded,
)
from repowise.server.mcp_server._meta import build_meta as _build_meta


def _step(index: int, node: GraphNode | None, node_id: str, via: str | None) -> dict[str, Any]:
    """One hop of a flow, with everything needed to open it.

    A trace is a list of node ids, which is enough to rank a flow and not
    enough to read one: the caller still has to resolve every hop to a file and
    a line range before it can look at any of it. A drill returns the resolved
    chain so that round trip does not exist.

    A node the graph no longer holds still gets a step — its id, and nulls
    where the detail would be. Dropping it would renumber the chain and quietly
    change which hop calls which.
    """
    step: dict[str, Any] = {
        "index": index,
        "node_id": node_id,
        "name": (node.name if node else None) or node_id.split("::")[-1],
        "file": node.file_path if node else None,
        "kind": node.kind if node else None,
        "resolved": node is not None,
    }
    if node is not None:
        if node.qualified_name:
            step["qualified_name"] = node.qualified_name
        if node.start_line is not None:
            step["start_line"] = node.start_line
        if node.end_line is not None:
            step["end_line"] = node.end_line
        if node.language:
            step["language"] = node.language
        step["is_test"] = bool(node.is_test)
    if via:
        # How the edge into this step was resolved — the hop's provenance, so a
        # heuristic link is not read as a parsed one.
        step["via"] = via
    return step


@mcp.tool(
    default=False,
    surface_order=230,
    trust_kind="structural",
    artifact_type="call_path",
    presentation="call_path",
    evidence_basis="inferred",
)
async def get_execution_flows(
    top_n: int = 10,
    max_depth: int = 8,
    entry_point: str | None = None,
    repo: str | None = None,
    flow_id: str | None = None,
    depth: int | None = None,
) -> dict:
    """Show how the codebase executes: top entry points and their call traces.

    Returns scored entry points with BFS call-path traces showing which
    functions are called in sequence and whether the flow crosses
    community boundaries.

    Every flow carries a ``flow_id`` — a stable handle for "the flow from this
    entry point, in this index generation". Pass one back as ``flow_id`` to
    re-request that flow and get its resolved step chain (file, symbol, lines
    per hop) instead of a list of node ids. The handle is safe to keep across
    sessions; if the index has been rebuilt since it was minted, the response
    still answers and says so via ``generation_changed``.

    Args:
        top_n: Number of top entry points to trace (default 10).
        max_depth: Max trace depth per flow (default 8).
        entry_point: Trace from a specific symbol (overrides top_n scoring).
        repo: Usually omitted.
        flow_id: A ``flow_id`` from an earlier response. Returns that one flow
            with its full step chain. Overrides ``top_n`` and ``entry_point``.
        depth: Trace depth for a ``flow_id`` drill (defaults to ``max_depth``).
    """
    if repo == "all":
        return _unsupported_repo_all("get_execution_flows")
    ctx = await _resolve_repo_context(repo)

    t0 = time.perf_counter()

    # Bound parameters
    top_n = max(1, min(top_n, 50))
    max_depth = max(1, min(max_depth, 20))

    requested: ParsedFlowId | None = None
    if flow_id:
        requested = parse_flow_id(flow_id)
        if requested is None:
            return {
                "flow_id": flow_id,
                "error": (
                    f"Not a flow id: {flow_id!r}. Expected the "
                    "'flow:<generation>:<entry point>' form returned by this tool."
                ),
                "_meta": _build_meta(timing_ms=(time.perf_counter() - t0) * 1000),
            }
        entry_point = requested.entry_point
        max_depth = max(1, min(depth if depth is not None else max_depth, 20))

    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)
        repo_id = repository.id
        generation = index_generation(repository)

        # Determine entry points
        entry_nodes: list[tuple[GraphNode, float]] = []

        if entry_point:
            # Trace from a specific symbol
            node = await get_graph_node(session, repo_id, entry_point)
            if node is None:
                missing: dict[str, Any] = {
                    "entry_point": entry_point,
                    "error": f"Symbol not found: {entry_point!r}",
                    "_meta": _build_meta(timing_ms=(time.perf_counter() - t0) * 1000),
                }
                if requested is not None:
                    # A handle that resolved to nothing is the one case where a
                    # caller most needs to know the index moved: the entry point
                    # it names may simply have been renamed away.
                    missing["flow_id"] = flow_id
                    missing["generation_changed"] = requested.generation != generation
                    missing["requested_generation"] = requested.generation
                    missing["current_generation"] = generation
                return missing
            entry_nodes = [(node, _ep_score(node))]
        else:
            # Top-N scored entry points from DB
            top_nodes = await get_top_entry_points(session, repo_id, min_score=0.0, limit=top_n)
            for n in top_nodes:
                entry_nodes.append((n, _ep_score(n)))

        exclude_spec = _get_exclude_spec(ctx.path)
        if exclude_spec:
            entry_nodes = [
                (n, s)
                for (n, s) in entry_nodes
                if not is_excluded(n.file_path or n.node_id, exclude_spec)
            ]

        if not entry_nodes:
            return {
                "total_entry_points": 0,
                "flows": [],
                "_meta": _build_meta(timing_ms=(time.perf_counter() - t0) * 1000),
            }

        # BFS trace from each entry point
        node_cache: dict[str, GraphNode] = {}
        flows: list[dict[str, Any]] = []

        for ep_node, ep_score in entry_nodes:
            hop_origins: dict[tuple[str, str], str] = {}
            termination: dict[str, Any] = {}
            trace = await bfs_trace(
                session,
                repo_id,
                ep_node.node_id,
                max_depth,
                node_cache,
                hop_origins,
                termination,
            )
            # Drop excluded files reached downstream so they don't leak via the
            # trace (entry-point filtering above doesn't cover BFS descendants).
            if exclude_spec:
                filtered = filter_embedded_path_ids(trace, exclude_spec)
                # The walk classified the node it actually stopped at. When
                # filtering drops that node, the trace we publish ends earlier
                # and it ends there because of the exclude spec — reporting the
                # walk's reason would assert the new last node calls nothing.
                if filtered and filtered[-1] != trace[-1]:
                    termination["reason"] = "excluded_target"
                    termination["detail"] = {}
                trace = filtered

            communities_visited, crosses = await resolve_trace_communities(
                session, repo_id, trace, node_cache
            )

            flow: dict[str, Any] = {
                "flow_id": mint_flow_id(generation, ep_node.node_id),
                "entry_point": ep_node.node_id,
                "entry_point_name": ep_node.name or ep_node.node_id.split("::")[-1],
                "entry_point_score": round(ep_score, 3),
                "trace": trace,
                "depth": len(trace) - 1,
                "crosses_community": crosses,
                "communities_visited": communities_visited,
                "termination": termination.get("reason"),
            }
            if termination.get("detail"):
                flow["termination_detail"] = termination["detail"]
            # Which strategy produced each hop, aligned to `trace` pairwise.
            # Omitted when no hop has one, so an older index shows no field
            # rather than a list of nulls.
            via = [hop_origins.get(pair) for pair in pairwise(trace)]
            if any(via):
                flow["trace_via"] = via
            if requested is not None:
                # The drill's whole point: hops resolved to files and lines, so
                # the caller reads the chain instead of re-querying every node.
                origins = [None, *via]
                flow["steps"] = [
                    _step(i, node_cache.get(node_id), node_id, origins[i])
                    for i, node_id in enumerate(trace)
                ]
            flows.append(flow)

    # Score descending, then entry point — the score alone is not a total order.
    # ``get_top_entry_points`` sorts on it too and leaves ties in whatever order
    # the rows arrived, so two flows of equal score could swap places between
    # identical calls. A flow id is only worth having if the list carrying it is
    # reproducible.
    flows.sort(key=lambda f: (-f["entry_point_score"], f["entry_point"]))

    result: dict[str, Any] = {
        "total_entry_points": len(flows),
        "flows": flows,
        "index_generation": generation,
        "_meta": _build_meta(
            timing_ms=(time.perf_counter() - t0) * 1000,
            hint=(
                "Pass a flow_id back to drill it into a resolved step chain. "
                "Use get_context(include=['callers','callees']) on any trace node for detail."
            ),
        ),
    }
    if requested is not None:
        result["flow_id"] = flow_id
        result["requested_generation"] = requested.generation
        result["generation_changed"] = requested.generation != generation
        if result["generation_changed"]:
            result["generation_hint"] = (
                "This flow_id was minted from an earlier index generation. The entry "
                "point still resolves and the trace below is freshly walked, but the "
                "call graph may have changed since the handle was issued."
            )
    return result
