"""Transactional wiring from authoritative symbol writes to the source outbox."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
from sqlalchemy import select

from repowise.core.ingestion import ASTParser, FileTraverser
from repowise.core.persistence.database import (
    create_engine,
    create_session_factory,
    resolve_db_url,
)
from repowise.core.persistence.models import SourceIndexUpdate
from repowise.core.pipeline.incremental import persist_incremental_index
from repowise.core.source_search import SOURCE_SEARCH_ENV
from repowise.core.source_search.outbox import (
    enqueue_incremental_update,
    suppress_incremental_paths,
)


class _Graph:
    def __init__(self, path: str) -> None:
        self._graph = nx.DiGraph()
        self._graph.add_node(path, node_type="file")

    def graph(self):
        return self._graph

    def pagerank(self):
        return {}


def _parsed(repo: Path, path: str):
    info = next(item for item in FileTraverser(repo).traverse() if item.path == path)
    return ASTParser().parse_file(info, (repo / path).read_bytes())


async def _rows(repo: Path) -> list[SourceIndexUpdate]:
    engine = create_engine(resolve_db_url(repo))
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            return list(
                (
                    await session.execute(
                        select(SourceIndexUpdate).order_by(SourceIndexUpdate.sequence)
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()


async def test_incremental_symbol_commit_enqueues_a_ready_change(tmp_path, monkeypatch):
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def run():\n    return 1\n")
    parsed = _parsed(repo, "src/app.py")
    diff = SimpleNamespace(
        path="src/app.py",
        status="modified",
        old_path=None,
        new_parsed=parsed,
    )

    await persist_incremental_index(
        repo,
        _Graph("src/app.py"),
        {},
        None,
        None,
        ["src/app.py"],
        current_graph_file_paths={"src/app.py"},
        file_diffs=[diff],
        parsed_files=[parsed],
    )

    rows = await _rows(repo)
    assert len(rows) == 1
    assert rows[0].state == "pending"
    assert rows[0].upstream_ready is True
    assert json.loads(rows[0].change_set_json)[0]["parse_state"] == "parsed"


async def test_symbol_failure_commits_a_blocked_change_instead_of_false_ready(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def run():\n    return 1\n")
    parsed = _parsed(repo, "src/app.py")
    diff = SimpleNamespace(
        path="src/app.py",
        status="modified",
        old_path=None,
        new_parsed=parsed,
    )

    from repowise.core.pipeline import persist as persist_module

    async def _fail_symbols(*args, **kwargs):
        raise RuntimeError("symbol transaction input rejected")

    monkeypatch.setattr(persist_module, "persist_incremental_symbols", _fail_symbols)
    await persist_incremental_index(
        repo,
        _Graph("src/app.py"),
        {},
        None,
        None,
        ["src/app.py"],
        current_graph_file_paths={"src/app.py"},
        file_diffs=[diff],
        parsed_files=[parsed],
    )

    rows = await _rows(repo)
    assert len(rows) == 1
    assert rows[0].state == "blocked"
    assert rows[0].upstream_ready is False
    assert "symbol transaction input rejected" in (rows[0].last_error or "")


async def test_watch_fast_lane_suppresses_only_paths_it_already_captured(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "fast.py").write_text("def fast():\n    return 1\n")
    (repo / "src" / "commit.py").write_text("def committed():\n    return 2\n")

    from repowise.core.persistence import get_session, init_db, upsert_repository

    engine = create_engine(resolve_db_url(repo))
    try:
        await init_db(engine)
        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            repository = await upsert_repository(
                session, name="repo", local_path=str(repo)
            )
            diffs = [
                SimpleNamespace(path="src/fast.py", status="modified", old_path=None),
                SimpleNamespace(path="src/commit.py", status="modified", old_path=None),
            ]
            with suppress_incremental_paths({"src/fast.py"}):
                await enqueue_incremental_update(
                    session,
                    repository.id,
                    repo,
                    file_diffs=diffs,
                    parsed_files=[],
                )
    finally:
        await engine.dispose()

    rows = await _rows(repo)
    assert len(rows) == 1
    changes = json.loads(rows[0].change_set_json)
    assert [change["path"] for change in changes] == ["src/commit.py"]


def test_older_full_repo_parse_cannot_overwrite_a_newer_save(tmp_path):
    from repowise.core.pipeline.persist import _changed_file_symbols

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    path = repo / "src" / "app.py"
    path.write_text("def before():\n    return 1\n")
    parsed = _parsed(repo, "src/app.py")
    path.write_text("def after():\n    return 2\n")

    reconcile_paths, symbols = _changed_file_symbols([parsed], ["src/app.py"])

    assert reconcile_paths == []
    assert symbols == []
