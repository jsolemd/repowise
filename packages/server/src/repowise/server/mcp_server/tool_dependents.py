"""MCP Tool: get_dependents — complete paginated inbound graph traversal."""

from __future__ import annotations

from collections import Counter
from typing import Any

from repowise.core.persistence.database import get_session
from repowise.core.registry import mcp_tool_registry as mcp
from repowise.server.mcp_server._dependency_queries import (
    DependentHit,
    GraphTargetResolution,
    find_inbound_dependents,
    graph_node_path,
    resolve_graph_target,
)
from repowise.server.mcp_server._helpers import (
    _get_exclude_spec,
    _get_repo,
    _resolve_repo_context,
    _unsupported_repo_all,
    is_excluded,
)

_MAX_DEPTH = 8
_DEFAULT_LIMIT = 25
_MAX_LIMIT = 100


def _resolution_failure(resolution: GraphTargetResolution) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": resolution.status,
        "target": {"query": resolution.query, "node": None, "node_type": None},
        "dependents": [],
        "total": 0,
    }
    if resolution.status == "ambiguous":
        result.update(
            {
                "match_count": len(resolution.candidates),
                "candidates": resolution.candidates,
                "candidates_truncated": resolution.candidates_truncated,
                "explanation": (
                    f"Symbol '{resolution.query}' is ambiguous; choose one candidate "
                    "symbol_id and retry."
                ),
            }
        )
    elif resolution.status == "graph_missing":
        result.update(
            {
                "candidates": resolution.candidates,
                "explanation": (
                    f"Symbol '{resolution.query}' exists in the symbol index but has no "
                    "dependency-graph node. Run repowise update."
                ),
            }
        )
    else:
        result["explanation"] = f"Target '{resolution.query}' not found in graph"
    return result


def _serialize_hit(hit: DependentHit) -> dict[str, Any]:
    node = hit.node
    return {
        "node": node.node_id,
        "node_type": node.node_type,
        "file": graph_node_path(node),
        "name": node.name,
        "kind": node.kind,
        "distance": hit.distance,
        "reference_count": hit.reference_count,
        "pagerank": node.pagerank or 0.0,
        "is_test": bool(node.is_test),
        "edge_types": sorted(hit.edge_types),
        "depends_on": sorted(hit.depends_on),
    }


@mcp.tool(default=False)
async def get_dependents(
    target: str,
    repo: str | None = None,
    depth: int = 1,
    include_tests: bool = False,
    offset: int = 0,
    limit: int = _DEFAULT_LIMIT,
) -> dict[str, Any]:
    """List complete inbound dependents of a file or symbol, one page at a time.

    File targets follow canonical file-dependency edges; symbol targets follow
    canonical symbol-use edges. Bare symbol names resolve when unique and
    ambiguous names return candidates. Traversal is complete through depth 1-8:
    pagination limits only the returned page, while ``total`` describes every
    matching dependent. Tests are excluded by default and are not used as
    hidden transitive bridges.

    Args:
        target: File path, symbol id, or unambiguous symbol name.
        repo: Repository path, name, or ID.
        depth: Inbound traversal depth (1-8, default 1).
        include_tests: Include test nodes and traversal through them.
        offset: Zero-based result offset.
        limit: Page size (1-100, default 25).
    """
    if repo == "all":
        return _unsupported_repo_all("get_dependents")

    effective_depth = max(1, min(depth, _MAX_DEPTH))
    effective_offset = max(0, offset)
    effective_limit = max(1, min(limit, _MAX_LIMIT))
    ctx = await _resolve_repo_context(repo)
    exclude_spec = _get_exclude_spec(ctx.path)
    if is_excluded(target, exclude_spec):
        return {
            "error": f"'{target}' is excluded by exclude_patterns.",
            "dependents": [],
            "total": 0,
        }

    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)
        resolution = await resolve_graph_target(session, repository.id, target)
        if resolution.status != "resolved":
            return _resolution_failure(resolution)

        target_node = resolution.node
        assert target_node is not None
        if target_node.node_type not in {"file", "symbol"}:
            return {
                "status": "invalid_target_type",
                "target": {
                    "query": target,
                    "node": target_node.node_id,
                    "node_type": target_node.node_type,
                },
                "dependents": [],
                "total": 0,
                "explanation": "get_dependents accepts file or symbol targets.",
            }
        if is_excluded(graph_node_path(target_node), exclude_spec):
            return {
                "error": f"'{graph_node_path(target_node)}' is excluded by exclude_patterns.",
                "dependents": [],
                "total": 0,
            }

        hits, edge_types = await find_inbound_dependents(
            session,
            repository.id,
            target_node,
            depth=effective_depth,
            include_tests=include_tests,
            exclude_spec=exclude_spec,
        )

    # GraphEdge rows are pair/type-aggregated. Rank by the number of distinct
    # persisted relations into the preceding BFS frontier, then centrality.
    hits.sort(
        key=lambda hit: (
            -hit.reference_count,
            -(hit.node.pagerank or 0.0),
            hit.distance,
            hit.node.node_id,
        )
    )
    total = len(hits)
    page_hits = hits[effective_offset : effective_offset + effective_limit]
    returned = len(page_hits)
    has_more = effective_offset + returned < total
    counts_by_depth = Counter(hit.distance for hit in hits)

    result: dict[str, Any] = {
        "status": "found",
        "target": {
            "query": target,
            "node": target_node.node_id,
            "node_type": target_node.node_type,
            "file": graph_node_path(target_node),
            "matched_by": resolution.matched_by,
        },
        "depth": effective_depth,
        "include_tests": include_tests,
        "edge_types": list(edge_types),
        "dependents": [_serialize_hit(hit) for hit in page_hits],
        "total": total,
        "counts_by_depth": {
            str(distance): counts_by_depth.get(distance, 0)
            for distance in range(1, effective_depth + 1)
        },
        "pagination": {
            "offset": effective_offset,
            "limit": effective_limit,
            "returned": returned,
            "has_more": has_more,
            "next_offset": effective_offset + returned if has_more else None,
        },
        "ranking": ["reference_count", "pagerank"],
        "reference_count_unit": "distinct_graph_relations",
    }
    requested = {"depth": depth, "offset": offset, "limit": limit}
    applied = {
        "depth": effective_depth,
        "offset": effective_offset,
        "limit": effective_limit,
    }
    if requested != applied:
        result["requested"] = requested
    return result
