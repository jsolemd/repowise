"""get_overview's orientation block: how big, what languages, what shape."""

from __future__ import annotations

import pytest

from repowise.core.persistence.models import GraphEdge, GraphNode
from repowise.server.mcp_server._repo_stats import (
    MAX_LANGUAGES,
    _fold_tail,
    build_repo_stats,
)

# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


def test_entries_come_back_biggest_first():
    assert _fold_tail({"a": 1, "b": 9, "c": 5}, 10) == [
        {"name": "b", "count": 9},
        {"name": "c", "count": 5},
        {"name": "a", "count": 1},
    ]


def test_the_tail_folds_into_one_entry():
    folded = _fold_tail({"a": 9, "b": 8, "c": 3, "d": 2}, 2)
    assert folded == [
        {"name": "a", "count": 9},
        {"name": "b", "count": 8},
        {"name": "other", "count": 5},
    ]


def test_nothing_is_folded_when_everything_fits():
    assert all(entry["name"] != "other" for entry in _fold_tail({"a": 1, "b": 2}, 5))


def test_equal_counts_order_by_name():
    """Two runs against one index must agree, ties included."""
    assert [e["name"] for e in _fold_tail({"z": 4, "a": 4, "m": 4}, 10)] == ["a", "m", "z"]


def test_an_empty_index_folds_to_nothing():
    assert _fold_tail({}, 5) == []


# ---------------------------------------------------------------------------
# The stats themselves
# ---------------------------------------------------------------------------


@pytest.fixture
async def stats_graph(session, repo_id):
    """A small graph with a known shape, including an external import node."""
    files = [
        ("a.py", "python"),
        ("b.py", "python"),
        ("c.py", "python"),
        ("d.ts", "typescript"),
        ("e.yaml", "yaml"),
        ("requests", "external"),
    ]
    for index, (path, language) in enumerate(files):
        session.add(
            GraphNode(
                id=f"st-f{index}",
                repository_id=repo_id,
                node_id=path,
                node_type="file",
                language=language,
            )
        )
    for index in range(4):
        session.add(
            GraphNode(
                id=f"st-s{index}",
                repository_id=repo_id,
                node_id=f"a.py::sym{index}",
                node_type="symbol",
                language="python",
            )
        )
    # One edge per (source, target, type): the table is unique on that triple,
    # so a fixture that repeats a pair is describing a graph that cannot exist.
    edges = [("calls", 3), ("imports", 2)]
    counter = 0
    for edge_type, count in edges:
        for target in range(count):
            session.add(
                GraphEdge(
                    id=f"st-e{counter}",
                    repository_id=repo_id,
                    source_node_id="a.py",
                    target_node_id=f"target{target}.py",
                    edge_type=edge_type,
                )
            )
            counter += 1
    await session.commit()
    return repo_id


class _Repo:
    def __init__(self, repo_id: str) -> None:
        self.id = repo_id


async def test_nodes_and_edges_are_counted_by_type(session, stats_graph):
    stats = await build_repo_stats(session, _Repo(stats_graph))

    assert stats["graph"]["nodes"]["total"] == 10
    assert stats["graph"]["nodes"]["by_type"] == {"file": 6, "symbol": 4}
    assert stats["graph"]["edges"]["total"] == 5
    assert stats["graph"]["edges"]["by_type"] == [
        {"name": "calls", "count": 3},
        {"name": "imports", "count": 2},
    ]


async def test_the_language_distribution_carries_shares(session, stats_graph):
    stats = await build_repo_stats(session, _Repo(stats_graph))

    assert stats["indexed_files"] == 5  # the external node is not a file
    assert stats["files_by_language"][0] == {"language": "python", "files": 3, "share": 0.6}
    assert {entry["language"] for entry in stats["files_by_language"]} == {
        "python",
        "typescript",
        "yaml",
    }


async def test_third_party_imports_are_not_counted_as_source_files(session, stats_graph):
    """Counting a dependency as a file overstates the tree it is imported into."""
    stats = await build_repo_stats(session, _Repo(stats_graph))

    assert "external" not in {entry["language"] for entry in stats["files_by_language"]}
    assert stats["external_dependency_nodes"] == 1
    assert sum(entry["files"] for entry in stats["files_by_language"]) == stats["indexed_files"]


async def test_shares_are_of_the_indexed_tree(session, stats_graph):
    stats = await build_repo_stats(session, _Repo(stats_graph))
    assert sum(entry["share"] for entry in stats["files_by_language"]) == pytest.approx(1.0)


async def test_an_empty_repository_reports_zeroes_rather_than_dividing_by_them(session, repo_id):
    stats = await build_repo_stats(session, _Repo(repo_id))
    assert stats["indexed_files"] == 0
    assert stats["files_by_language"] == []
    assert stats["graph"]["nodes"]["total"] == 0
    assert "external_dependency_nodes" not in stats


async def test_a_polyglot_repository_still_reads_as_polyglot(session, repo_id):
    """More languages than the cap fold, but the cap is not narrow."""
    for index in range(MAX_LANGUAGES + 3):
        session.add(
            GraphNode(
                id=f"poly-{index}",
                repository_id=repo_id,
                node_id=f"f{index}.x",
                node_type="file",
                language=f"lang{index:02d}",
            )
        )
    await session.commit()

    stats = await build_repo_stats(session, _Repo(repo_id))
    assert len(stats["files_by_language"]) == MAX_LANGUAGES + 1
    assert stats["files_by_language"][-1]["language"] == "other"
    assert stats["files_by_language"][-1]["files"] == 3


# ---------------------------------------------------------------------------
# In the payload
# ---------------------------------------------------------------------------


async def test_get_overview_carries_the_block(setup_mcp):
    from repowise.server.mcp_server.tool_overview import get_overview

    result = await get_overview()
    stats = result["repo_stats"]
    assert stats["graph"]["nodes"]["total"] >= 1
    assert "files_by_language" in stats
    assert "indexed_files" in stats
