"""Shared SQL queries for dependency-path and inbound-dependent MCP tools.

The graph has two distinct layers. File nodes are joined by the named
``FILE_DEPENDENCY_EDGE_TYPES`` view; symbol nodes are joined by
``SYMBOL_USE_EDGE_TYPES``. Keeping target resolution and inbound traversal in
one module prevents the two agent tools from disagreeing about which layer a
target belongs to or silently bounding a transitive result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.ingestion.models import (
    FILE_DEPENDENCY_EDGE_TYPES,
    SYMBOL_USE_EDGE_TYPES,
)
from repowise.core.persistence.models import GraphEdge, GraphNode
from repowise.server.mcp_server._helpers import is_excluded
from repowise.server.mcp_server._symbol_lookup import resolve_symbol_match

# Keep IN clauses portable to SQLite builds with the older 999-variable cap.
_FRONTIER_CHUNK = 400


@dataclass(frozen=True)
class GraphTargetResolution:
    """Resolved graph target, or enough structured evidence to recover."""

    query: str
    status: str
    node: GraphNode | None = None
    matched_by: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    candidates_truncated: bool = False


@dataclass
class DependentHit:
    """One inbound node discovered at its shortest distance from the target."""

    node: GraphNode
    distance: int
    reference_count: int = 0
    edge_types: set[str] = field(default_factory=set)
    depends_on: set[str] = field(default_factory=set)


def graph_node_path(node: GraphNode) -> str:
    """Return the repository-relative path represented by *node*."""
    if node.node_type == "file":
        return node.node_id
    return node.file_path or node.node_id.split("::", 1)[0]


async def resolve_graph_target(
    session: AsyncSession,
    repo_id: str,
    query: str,
) -> GraphTargetResolution:
    """Resolve an exact graph node id or an unambiguous symbol name.

    Exact graph ids win, which preserves file-path behavior and avoids sending
    a path through symbol-name heuristics. Name-form symbols reuse the shared
    MCP symbol resolver; a multi-match result remains ambiguous and is never
    collapsed to whichever row happened to arrive first.
    """
    exact = await session.execute(
        select(GraphNode).where(
            GraphNode.repository_id == repo_id,
            GraphNode.node_id == query,
        )
    )
    node = exact.scalar_one_or_none()
    if node is not None:
        return GraphTargetResolution(
            query=query,
            status="resolved",
            node=node,
            matched_by="node_id",
        )

    match = await resolve_symbol_match(session, repo_id, query)
    if match.status == "ambiguous":
        return GraphTargetResolution(
            query=query,
            status="ambiguous",
            matched_by=match.rung,
            candidates=match.candidates(),
            candidates_truncated=match.truncated,
        )
    if match.status == "not_found":
        return GraphTargetResolution(query=query, status="not_found")

    symbol_id = match.rows[0].symbol_id
    result = await session.execute(
        select(GraphNode).where(
            GraphNode.repository_id == repo_id,
            GraphNode.node_id == symbol_id,
        )
    )
    node = result.scalar_one_or_none()
    if node is None:
        # The wiki-symbol row can outlive a partial graph rebuild. Treat that
        # as an honest miss on this surface instead of returning a node the
        # traversal cannot possibly walk.
        return GraphTargetResolution(
            query=query,
            status="graph_missing",
            matched_by=match.rung,
            candidates=match.candidates(),
        )
    return GraphTargetResolution(
        query=query,
        status="resolved",
        node=node,
        matched_by=match.rung,
    )


def dependent_edge_types(node_type: str) -> tuple[str, ...]:
    """Return the canonical dependency view for one graph layer."""
    if node_type == "file":
        return tuple(sorted(FILE_DEPENDENCY_EDGE_TYPES))
    if node_type == "symbol":
        return tuple(sorted(SYMBOL_USE_EDGE_TYPES))
    return ()


def _chunks(values: list[str]) -> list[list[str]]:
    return [values[i : i + _FRONTIER_CHUNK] for i in range(0, len(values), _FRONTIER_CHUNK)]


async def find_inbound_dependents(
    session: AsyncSession,
    repo_id: str,
    target: GraphNode,
    *,
    depth: int,
    include_tests: bool,
    exclude_spec: Any = None,
) -> tuple[list[DependentHit], tuple[str, ...]]:
    """Walk every inbound dependency through *depth*, with no row cap.

    Each breadth-first round queries all edges whose target is in the current
    frontier. Wide frontiers are chunked only for SQL parameter portability;
    every chunk is consumed before the next depth begins. Tests filtered out
    here are neither returned nor used as hidden bridges to production nodes.

    ``reference_count`` is the number of distinct persisted graph relations
    from a newly discovered node into the preceding frontier. GraphEdge is
    unique per ``(source, target, edge_type)``, so this is intentionally not
    presented as a source-occurrence/call-site count.
    """
    edge_types = dependent_edge_types(target.node_type)
    if not edge_types or depth <= 0:
        return [], edge_types

    visited: set[str] = {target.node_id}
    frontier: list[str] = [target.node_id]
    discovered: dict[str, DependentHit] = {}

    for distance in range(1, depth + 1):
        candidates: dict[str, DependentHit] = {}
        for chunk in _chunks(frontier):
            conditions = [
                GraphEdge.repository_id == repo_id,
                GraphEdge.target_node_id.in_(chunk),
                GraphEdge.edge_type.in_(edge_types),
                GraphNode.node_type == target.node_type,
            ]
            if not include_tests:
                conditions.append(GraphNode.is_test.is_(False))

            rows = await session.execute(
                select(GraphEdge, GraphNode)
                .join(
                    GraphNode,
                    and_(
                        GraphNode.repository_id == GraphEdge.repository_id,
                        GraphNode.node_id == GraphEdge.source_node_id,
                    ),
                )
                .where(*conditions)
            )
            # Ordering this in SQL makes SQLite choose the unique
            # (repo, source, target, type) index and scan every repo edge. The
            # unordered predicate uses ix_graph_edges_repo_target — the index
            # that owns inbound queries — and this small frontier-local sort
            # supplies deterministic processing without changing that plan.
            batch_rows = list(rows.all())
            batch_rows.sort(
                key=lambda pair: (
                    pair[0].source_node_id,
                    pair[0].target_node_id,
                    pair[0].edge_type,
                )
            )
            for edge, node in batch_rows:
                if node.node_id in visited or is_excluded(graph_node_path(node), exclude_spec):
                    continue
                hit = candidates.get(node.node_id)
                if hit is None:
                    hit = DependentHit(node=node, distance=distance)
                    candidates[node.node_id] = hit
                hit.reference_count += 1
                hit.edge_types.add(edge.edge_type)
                hit.depends_on.add(edge.target_node_id)

        if not candidates:
            break

        frontier = sorted(candidates)
        visited.update(frontier)
        discovered.update(candidates)

    return list(discovered.values()), edge_types
