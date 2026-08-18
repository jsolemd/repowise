"""Orientation numbers for ``get_overview``: how big, what languages, what shape.

``get_overview`` is the first call an agent makes on a repository it does not
know, and it answered every question about that repository except its size. An
agent that wanted to know whether it was looking at four hundred files or forty
thousand, or whether this was a Python codebase with some YAML or a TypeScript
one with some Python, had to spend a second and third call finding out — which
is the cost this exists to remove.

Three aggregates, all computed in the database rather than by loading rows: a
first-call tool must not pay to materialise nine thousand nodes to count them.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from repowise.core.persistence.models import GraphEdge, GraphNode

__all__ = ["EXTERNAL_LANGUAGE", "MAX_EDGE_TYPES", "MAX_LANGUAGES", "build_repo_stats"]

#: Languages listed individually before the tail is folded into one entry. Long
#: enough that a polyglot repository still reads as polyglot, short enough that
#: the block stays glanceable.
MAX_LANGUAGES = 12

#: Edge types listed individually. The graph has under ten in practice; the cap
#: is here so a future edge kind cannot turn this block into a wall.
MAX_EDGE_TYPES = 12

#: The language stamped on nodes that stand for a third-party import rather than
#: a file in the tree. Counted separately: folding them into the distribution
#: would report dependencies as source files and overstate the repository.
EXTERNAL_LANGUAGE = "external"


async def _counts_by(session: Any, column: Any, model: Any, repository_id: str) -> dict[str, int]:
    """``{value: count}`` for *column*, grouped in the database."""
    result = await session.execute(
        select(column, func.count()).where(model.repository_id == repository_id).group_by(column)
    )
    return {str(value or ""): int(count) for value, count in result.all()}


def _fold_tail(counts: dict[str, int], limit: int) -> list[dict[str, Any]]:
    """The *limit* largest entries, biggest first, with the rest as ``other``.

    Ties break on the name so two runs against one index agree — the same
    reason the flow list sorts on more than its score.
    """
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    head = [{"name": name, "count": count} for name, count in ordered[:limit]]
    tail = ordered[limit:]
    if tail:
        head.append({"name": "other", "count": sum(count for _, count in tail)})
    return head


async def build_repo_stats(session: Any, repository: Any) -> dict[str, Any]:
    """Size and shape of the indexed graph, for a caller with no map yet.

    Describes **the index**, not the working tree: it counts what has been
    parsed and persisted, which is what every other number in this response is
    also drawn from. A file the index never walked is absent here for the same
    reason it is absent from ``key_modules``.
    """
    node_types = await _counts_by(session, GraphNode.node_type, GraphNode, repository.id)
    edge_types = await _counts_by(session, GraphEdge.edge_type, GraphEdge, repository.id)

    languages_result = await session.execute(
        select(GraphNode.language, func.count())
        .where(
            GraphNode.repository_id == repository.id,
            GraphNode.node_type == "file",
        )
        .group_by(GraphNode.language)
    )
    languages = {str(lang or ""): int(count) for lang, count in languages_result.all()}
    external = languages.pop(EXTERNAL_LANGUAGE, 0)
    indexed_files = sum(languages.values())

    stats: dict[str, Any] = {
        "graph": {
            "nodes": {"total": sum(node_types.values()), "by_type": node_types},
            "edges": {
                "total": sum(edge_types.values()),
                "by_type": _fold_tail(edge_types, MAX_EDGE_TYPES),
            },
        },
        "indexed_files": indexed_files,
        "files_by_language": [
            {
                "language": entry["name"],
                "files": entry["count"],
                # Share of the indexed tree, so "mostly Python" is readable
                # without the caller doing the division.
                "share": round(entry["count"] / indexed_files, 3) if indexed_files else 0.0,
            }
            for entry in _fold_tail(languages, MAX_LANGUAGES)
        ],
    }
    if external:
        stats["external_dependency_nodes"] = external
    return stats
