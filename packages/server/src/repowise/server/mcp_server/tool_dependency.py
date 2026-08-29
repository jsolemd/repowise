"""MCP Tool: get_dependency_path — file dependencies or pure call chains."""

from __future__ import annotations

import json
from itertools import islice
from typing import Any

from sqlalchemy import select

from repowise.core.ingestion.models import TEMPORAL_EDGE_TYPES
from repowise.core.persistence.database import get_session
from repowise.core.persistence.models import GitMetadata, GraphEdge, GraphNode
from repowise.core.registry import mcp_tool_registry as mcp
from repowise.server.mcp_server._dependency_queries import (
    GraphTargetResolution,
    graph_node_path,
    resolve_graph_target,
)
from repowise.server.mcp_server._graph_utils import build_visual_context
from repowise.server.mcp_server._helpers import (
    _get_exclude_spec,
    _get_repo,
    _resolve_repo_context,
    _unsupported_repo_all,
    attach_ignored_arguments,
    filter_graph_nodes,
    is_excluded,
    resolve_enum_argument,
)

_PATH_MODES = frozenset({"files", "calls"})
_CALL_EDGE_TYPES = ("calls", "references")
_MAX_PATHS = 5


def _resolution_payload(resolution: GraphTargetResolution) -> dict[str, Any]:
    node = resolution.node
    return {
        "query": resolution.query,
        "node": node.node_id if node is not None else None,
        "node_type": node.node_type if node is not None else None,
        "matched_by": resolution.matched_by,
    }


def _resolution_failure(
    resolution: GraphTargetResolution,
    *,
    endpoint: str,
    mode: str,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": resolution.status,
        "mode": mode,
        "path": [],
        "paths": [],
        "distance": -1,
        "endpoint": endpoint,
        "query": resolution.query,
    }
    if resolution.status == "ambiguous":
        base.update(
            {
                "match_count": len(resolution.candidates),
                "candidates": resolution.candidates,
                "candidates_truncated": resolution.candidates_truncated,
                "explanation": (
                    f"{endpoint.title()} symbol '{resolution.query}' is ambiguous; "
                    "choose one candidate symbol_id and retry."
                ),
            }
        )
    elif resolution.status == "graph_missing":
        base.update(
            {
                "candidates": resolution.candidates,
                "explanation": (
                    f"{endpoint.title()} symbol '{resolution.query}' exists in the symbol "
                    "index but has no dependency-graph node. Run repowise update."
                ),
            }
        )
    else:
        base["explanation"] = f"{endpoint.title()} node '{resolution.query}' not found in graph"
    return base


def _path_with_relationships(
    path: list[str],
    relations: dict[tuple[str, str], set[str]],
) -> list[dict[str, str]]:
    """Preserve the legacy path shape while choosing relation labels stably."""
    result: list[dict[str, str]] = []
    relation_priority = {"calls": 0, "references": 1}
    for index, node in enumerate(path):
        relationship = ""
        if index < len(path) - 1:
            edge_types = relations[(node, path[index + 1])]
            relationship = min(edge_types, key=lambda edge: (relation_priority.get(edge, 2), edge))
        result.append({"node": node, "relationship": relationship})
    return result


async def _attach_co_change_signal(
    result: dict[str, Any],
    *,
    session_factory: Any,
    repository_id: str,
    source: str,
    target: str,
) -> None:
    """Attach the legacy file-level coupling fallback when it exists."""
    async with get_session(session_factory) as session:
        src_res = await session.execute(
            select(GitMetadata).where(
                GitMetadata.repository_id == repository_id,
                GitMetadata.file_path == source,
            )
        )
        src_meta = src_res.scalar_one_or_none()
    if not src_meta or not src_meta.co_change_partners_json:
        return
    for partner in json.loads(src_meta.co_change_partners_json):
        if partner.get("file_path", "") != target:
            continue
        result["co_change_signal"] = {
            "co_change_count": partner.get("co_change_count", partner.get("count", 0)),
            "last_co_change": partner.get("last_co_change"),
            "note": (
                "No import dependency, but these files co-change frequently — "
                "likely logical coupling."
            ),
        }
        return


@mcp.tool(default=False, surface_order=220, trust_kind="structural")
async def get_dependency_path(
    source: str,
    target: str,
    repo: str | None = None,
    mode: str = "files",
    limit_paths: int = 1,
) -> dict[str, Any]:
    """Find shortest file dependencies or calls-only symbol chains.

    ``mode="files"`` preserves the existing dependency traversal. Use
    ``mode="calls"`` for symbol-level chains containing only ``calls`` and
    ``references`` edges. Bare symbol names resolve when unique; ambiguous
    names return candidates. Up to five distinct shortest chains can be
    requested without changing the legacy top-level ``path`` result.

    Args:
        source: Source file path, symbol id, or unambiguous symbol name.
        target: Target file path, symbol id, or unambiguous symbol name.
        repo: Repository path, name, or ID.
        mode: ``files`` (default) or pure symbol-level ``calls`` traversal.
        limit_paths: Number of distinct shortest chains to return (1-5).
    """
    if repo == "all":
        return _unsupported_repo_all("get_dependency_path")

    ignored: list[dict[str, Any]] = []
    resolved_mode = (
        resolve_enum_argument(
            mode,
            _PATH_MODES,
            argument="mode",
            ignored=ignored,
        )
        or "files"
    )
    effective_limit = max(1, min(limit_paths, _MAX_PATHS))

    ctx = await _resolve_repo_context(repo)
    exclude_spec = _get_exclude_spec(ctx.path)
    for path in (source, target):
        if is_excluded(path, exclude_spec):
            result: dict[str, Any] = {
                "error": f"'{path}' is excluded by exclude_patterns.",
                "mode": resolved_mode,
                "path": [],
                "paths": [],
                "distance": -1,
            }
            attach_ignored_arguments(result, ignored)
            return result

    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)
        repository_id = repository.id
        source_resolution = await resolve_graph_target(session, repository_id, source)
        target_resolution = await resolve_graph_target(session, repository_id, target)

        for endpoint, resolution in (
            ("source", source_resolution),
            ("target", target_resolution),
        ):
            if resolution.status != "resolved":
                result = _resolution_failure(
                    resolution,
                    endpoint=endpoint,
                    mode=resolved_mode,
                )
                attach_ignored_arguments(result, ignored)
                return result

        source_node = source_resolution.node
        target_node = target_resolution.node
        assert source_node is not None and target_node is not None

        if resolved_mode == "calls":
            for endpoint, node in (("source", source_node), ("target", target_node)):
                if node.node_type != "symbol":
                    result = {
                        "status": "invalid_target_type",
                        "mode": resolved_mode,
                        "path": [],
                        "paths": [],
                        "distance": -1,
                        "endpoint": endpoint,
                        "query": source if endpoint == "source" else target,
                        "resolved_node": node.node_id,
                        "explanation": (
                            "mode='calls' requires symbol endpoints. Pass a symbol id "
                            "(path::Name) or an unambiguous symbol name."
                        ),
                    }
                    attach_ignored_arguments(result, ignored)
                    return result

        for node in (source_node, target_node):
            if is_excluded(graph_node_path(node), exclude_spec):
                result = {
                    "error": f"'{graph_node_path(node)}' is excluded by exclude_patterns.",
                    "mode": resolved_mode,
                    "path": [],
                    "paths": [],
                    "distance": -1,
                }
                attach_ignored_arguments(result, ignored)
                return result

        edge_stmt = select(GraphEdge).where(GraphEdge.repository_id == repository_id)
        node_stmt = select(GraphNode).where(GraphNode.repository_id == repository_id)
        if resolved_mode == "calls":
            edge_stmt = edge_stmt.where(GraphEdge.edge_type.in_(_CALL_EDGE_TYPES))
            node_stmt = node_stmt.where(GraphNode.node_type == "symbol")
        else:
            # ``co_changes`` is historical correlation, not a code dependency.
            # Containment remains for compatibility: it is the bridge from a
            # file to the symbol it defines in the existing traversal.
            edge_stmt = edge_stmt.where(GraphEdge.edge_type.notin_(TEMPORAL_EDGE_TYPES))

        edge_result = await session.execute(
            edge_stmt.order_by(
                GraphEdge.source_node_id,
                GraphEdge.target_node_id,
                GraphEdge.edge_type,
            )
        )
        node_result = await session.execute(node_stmt.order_by(GraphNode.node_id))
        edges = list(edge_result.scalars().all())
        nodes = filter_graph_nodes(list(node_result.scalars().all()), exclude_spec)

    try:
        import networkx as nx
    except ImportError:
        result = {
            "mode": resolved_mode,
            "path": [],
            "paths": [],
            "distance": -1,
            "explanation": "networkx not available for path queries",
        }
        attach_ignored_arguments(result, ignored)
        return result

    allowed = {node.node_id for node in nodes}
    graph: nx.DiGraph[str] = nx.DiGraph()
    graph.add_nodes_from(sorted(allowed))
    relations: dict[tuple[str, str], set[str]] = {}
    for edge in edges:
        if edge.source_node_id not in allowed or edge.target_node_id not in allowed:
            continue
        graph.add_edge(edge.source_node_id, edge.target_node_id)
        relations.setdefault((edge.source_node_id, edge.target_node_id), set()).add(edge.edge_type)

    resolved_source = source_node.node_id
    resolved_target = target_node.node_id
    resolved = {
        "source": _resolution_payload(source_resolution),
        "target": _resolution_payload(target_resolution),
    }
    if resolved_source not in graph or resolved_target not in graph:
        missing = "source" if resolved_source not in graph else "target"
        missing_node = resolved_source if missing == "source" else resolved_target
        result = {
            "status": "not_found",
            "mode": resolved_mode,
            "path": [],
            "paths": [],
            "distance": -1,
            "resolved": resolved,
            "explanation": f"{missing.title()} node '{missing_node}' not found in graph",
        }
        attach_ignored_arguments(result, ignored)
        return result

    try:
        raw_paths = list(
            islice(
                nx.all_shortest_paths(graph, resolved_source, resolved_target),
                effective_limit + 1,
            )
        )
    except nx.NetworkXNoPath:
        result = {
            "status": "no_path",
            "mode": resolved_mode,
            "path": [],
            "paths": [],
            "distance": -1,
            "resolved": resolved,
            "limit_paths": effective_limit,
            "explanation": "No direct dependency path found",
            "visual_context": build_visual_context(
                graph,
                resolved_source,
                resolved_target,
                nodes,
                nx,
            ),
        }
        if (
            resolved_mode == "files"
            and source_node.node_type == "file"
            and target_node.node_type == "file"
        ):
            await _attach_co_change_signal(
                result,
                session_factory=ctx.session_factory,
                repository_id=repository_id,
                source=resolved_source,
                target=resolved_target,
            )
        attach_ignored_arguments(result, ignored)
        return result

    paths_truncated = len(raw_paths) > effective_limit
    raw_paths = raw_paths[:effective_limit]
    paths = [
        {
            "path": _path_with_relationships(path, relations),
            "distance": len(path) - 1,
        }
        for path in raw_paths
    ]
    first = paths[0]
    result = {
        "status": "found",
        "mode": resolved_mode,
        # Backward-compatible single-path fields remain the first chain.
        "path": first["path"],
        "distance": first["distance"],
        "paths": paths,
        "paths_truncated": paths_truncated,
        "limit_paths": effective_limit,
        "resolved": resolved,
        "explanation": (
            f"Shortest path from {resolved_source} to {resolved_target} has "
            f"{first['distance']} hops"
        ),
    }
    if limit_paths != effective_limit:
        result["limit_paths_requested"] = limit_paths
    attach_ignored_arguments(result, ignored)
    return result
