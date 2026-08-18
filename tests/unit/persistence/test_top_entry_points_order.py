"""``get_top_entry_points`` must be a total order, not just a sorted one.

Entry-point scores tie routinely — two symbols at 0.88 is the normal case on a
real index, not a contrived one — and the query behind this read has no
``ORDER BY``, so a tied pair came back in whatever order the rows happened to
arrive and could swap between two identical calls.

That is invisible until something downstream hands these out under a stable
identifier. The MCP flows tool mints one handle per entry point, and a handle is
only worth having if the list carrying it is reproducible; a caller diffing two
responses would otherwise see flows move for no reason at all.
"""

from __future__ import annotations

import json

from repowise.core.persistence.crud import get_top_entry_points
from repowise.core.persistence.models import GraphNode
from tests.unit.persistence.helpers import insert_repo


def _symbol(repo_id: str, node_id: str, score: float) -> GraphNode:
    return GraphNode(
        repository_id=repo_id,
        node_id=node_id,
        node_type="symbol",
        community_meta_json=json.dumps({"entry_point_score": score}),
    )


async def _seed(session, scored: list[tuple[str, float]]) -> str:
    repo = await insert_repo(session)
    session.add_all([_symbol(repo.id, node_id, score) for node_id, score in scored])
    await session.commit()
    return repo.id


async def test_tied_scores_come_back_in_a_fixed_order(async_session) -> None:
    """Inserted in one order, returned in another — and always the same one."""
    repo_id = await _seed(
        async_session,
        [("z/last.py::run", 0.88), ("a/first.py::run", 0.88), ("m/middle.py::run", 0.88)],
    )

    orders = set()
    for _ in range(5):
        nodes = await get_top_entry_points(async_session, repo_id, min_score=0.0, limit=10)
        orders.add(tuple(n.node_id for n in nodes))

    assert len(orders) == 1, f"tied scores reordered between calls: {orders}"
    assert list(orders.pop()) == ["a/first.py::run", "m/middle.py::run", "z/last.py::run"]


async def test_score_still_decides_before_the_tie_break(async_session) -> None:
    """The tie-break is a tie-break — it must not outrank the score."""
    repo_id = await _seed(
        async_session,
        [("a/low.py::run", 0.10), ("z/high.py::run", 0.90), ("m/mid.py::run", 0.50)],
    )

    nodes = await get_top_entry_points(async_session, repo_id, min_score=0.0, limit=10)
    assert [n.node_id for n in nodes] == ["z/high.py::run", "m/mid.py::run", "a/low.py::run"]


async def test_the_limit_cut_is_reproducible_across_a_tie(async_session) -> None:
    """A tie straddling the cut must not change *which* entries survive it."""
    repo_id = await _seed(
        async_session,
        [(f"pkg/{letter}.py::run", 0.5) for letter in "edcba"],
    )

    first = await get_top_entry_points(async_session, repo_id, min_score=0.0, limit=2)
    second = await get_top_entry_points(async_session, repo_id, min_score=0.0, limit=2)

    assert [n.node_id for n in first] == [n.node_id for n in second]
    assert [n.node_id for n in first] == ["pkg/a.py::run", "pkg/b.py::run"]


async def test_the_score_floor_still_filters(async_session) -> None:
    repo_id = await _seed(
        async_session, [("a.py::keep", 0.60), ("b.py::drop", 0.10)]
    )

    nodes = await get_top_entry_points(async_session, repo_id, min_score=0.3, limit=10)
    assert [n.node_id for n in nodes] == ["a.py::keep"]
