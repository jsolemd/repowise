"""Wide reachability frontiers must fit SQLite's real bind-variable limit."""

from __future__ import annotations

import json
import sqlite3

import pytest

from repowise.core.analysis.test_reachability import (
    _edges_from,
    _edges_into,
    imported_names_by_test,
)
from repowise.core.analysis.test_reachability import (
    tests_reaching_by_tier as reaching,
)
from repowise.core.persistence.models import GraphEdge, GraphNode
from tests.unit.persistence.helpers import insert_repo


@pytest.fixture
async def limited_session(async_session):
    # aiosqlite owns the connection thread. Change the actual SQLite limit on
    # that thread; a mock execute-count guard would not reproduce the failure.
    connection = await async_session.connection()
    raw = await connection.get_raw_connection()
    driver = raw.driver_connection
    previous = await driver._execute(
        driver._conn.setlimit, sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999
    )
    try:
        yield async_session
    finally:
        await driver._execute(driver._conn.setlimit, sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)


async def _edge(session, repo_id, source, target, kind, *, origin=None, names=()):
    session.add(
        GraphEdge(
            repository_id=repo_id,
            source_node_id=source,
            target_node_id=target,
            edge_type=kind,
            resolution_origin=origin,
            imported_names_json=json.dumps(names),
        )
    )


@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
async def test_wide_edge_reads_keep_all_pairs_filters_and_distinctness(limited_session, direction):
    session = limited_session
    repo = await insert_repo(session)
    other = await insert_repo(session, name="other", local_path="/tmp/other-repo")
    paths = [f"src/item_{n}.py" for n in range(1200)]
    expected = set()
    for n in (0, 700, 1199):
        source, target = (
            (f"caller_{n}", paths[n]) if direction == "incoming" else (paths[n], f"symbol_{n}")
        )
        expected.add((source, target))
        await _edge(session, repo.id, source, target, "calls")
        # DISTINCT spans edge types and repeated input paths.
        await _edge(session, repo.id, source, target, "reads")
        await _edge(session, other.id, "foreign", target, "calls")
    await _edge(session, repo.id, paths[-1], paths[-1], "imports")
    if direction == "incoming":
        await _edge(session, repo.id, "unreliable", paths[-1], "calls", origin="global_unique")
    await session.flush()
    requested = [*paths, paths[0], paths[700]]

    if direction == "incoming":
        rows = await _edges_into(
            session, repo.id, requested, ["calls", "reads"], frozenset({"global_unique"})
        )
    else:
        rows = await _edges_from(session, repo.id, requested, ["calls", "reads"])

    assert set(rows) == expected
    assert len(rows) == len(expected)


async def test_wide_import_name_reads_keep_late_batches_and_union_names(limited_session):
    session = limited_session
    repo = await insert_repo(session)
    paths = [f"src/item_{n}.py" for n in range(1200)]
    for n in (0, 700, 1199):
        await _edge(session, repo.id, "tests/test_reader.py", paths[n], "imports", names=["alpha"])
        await _edge(session, repo.id, "tests/test_reader.py", paths[n], "reads", names=["beta"])
        await _edge(session, repo.id, "src/not_a_test.py", paths[n], "imports", names=["noise"])
    await session.flush()

    result = await imported_names_by_test(session, repo.id, paths, {"tests/test_reader.py"})

    assert result == {
        paths[n]: {"tests/test_reader.py": frozenset({"alpha", "beta"})} for n in (0, 700, 1199)
    }


async def test_one_file_with_many_symbols_still_finds_its_tests(limited_session):
    session = limited_session
    repo = await insert_repo(session)
    production = "src/large.py"
    test = "tests/test_large.py"
    session.add_all(
        [
            GraphNode(repository_id=repo.id, node_id=production, node_type="file", is_test=False),
            GraphNode(repository_id=repo.id, node_id=test, node_type="file", is_test=True),
        ]
    )
    for n in range(1200):
        await _edge(session, repo.id, production, f"{production}::f{n}", "defines")
        if n % 50 == 49:
            await session.flush()
    await _edge(session, repo.id, test, f"{test}::test_it", "defines")
    await _edge(session, repo.id, f"{test}::test_it", f"{production}::f1199", "calls")
    await session.flush()

    result = await reaching(session, repo.id, [production])

    assert result[production].tests == [test]
    assert result[production].via == "call-graph"
    assert result[production].total == 1
