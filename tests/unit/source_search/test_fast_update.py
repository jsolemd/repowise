"""Saved-file capture for the watcher source fast lane."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from repowise.core.ingestion import ASTParser, FileTraverser
from repowise.core.persistence.crud import reconcile_symbols_for_files
from repowise.core.persistence.database import (
    create_engine,
    create_session_factory,
    get_session,
    init_db,
    resolve_db_url,
)
from repowise.core.persistence.models import Repository, SourceIndexUpdate, WikiSymbol
from repowise.core.source_search.fast_update import capture_source_changes


async def _seed(repo: Path) -> None:
    engine = create_engine(resolve_db_url(repo))
    try:
        await init_db(engine)
        factory = create_session_factory(engine)
        info = FileTraverser(repo).file_info_for_path("src/app.py")
        assert info is not None
        parsed = ASTParser().parse_file(info, (repo / "src" / "app.py").read_bytes())
        for symbol in parsed.symbols:
            if not getattr(symbol, "file_path", None):
                symbol.file_path = parsed.file_info.path
        async with get_session(factory) as session:
            repository = Repository(id="r1", name="repo", local_path=str(repo))
            session.add(repository)
            await session.flush()
            await reconcile_symbols_for_files(
                session,
                repository.id,
                ["src/app.py"],
                parsed.symbols,
            )
    finally:
        await engine.dispose()


async def _stored(repo: Path) -> tuple[list[WikiSymbol], list[SourceIndexUpdate]]:
    engine = create_engine(resolve_db_url(repo))
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            symbols = list((await session.execute(select(WikiSymbol))).scalars().all())
            updates = list(
                (
                    await session.execute(
                        select(SourceIndexUpdate).order_by(SourceIndexUpdate.sequence)
                    )
                )
                .scalars()
                .all()
            )
            return symbols, updates
    finally:
        await engine.dispose()


@pytest.fixture
async def fast_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def old_name():\n    return 1\n")
    await _seed(repo)
    return repo


async def test_saved_file_symbols_and_outbox_commit_together(fast_repo: Path) -> None:
    (fast_repo / "src" / "app.py").write_text("def new_name():\n    return 2\n")

    result = await capture_source_changes(fast_repo, {"src/app.py"})

    symbols, updates = await _stored(fast_repo)
    assert result.parsed == 1
    assert [symbol.name for symbol in symbols] == ["new_name"]
    assert len(updates) == 1
    change = json.loads(updates[0].change_set_json)[0]
    assert change["path"] == "src/app.py"
    assert change["parse_state"] == "parsed"
    assert change["content_hash"]


async def test_deleted_file_prunes_symbols_and_records_a_tombstone(fast_repo: Path) -> None:
    (fast_repo / "src" / "app.py").unlink()

    result = await capture_source_changes(fast_repo, ["src/app.py"])

    symbols, updates = await _stored(fast_repo)
    assert result.removed == 1
    assert symbols == []
    change = json.loads(updates[0].change_set_json)[0]
    assert change["status"] == "deleted"
    assert change["parse_state"] == "deleted"


async def test_parser_failure_retains_last_good_symbols(fast_repo: Path, monkeypatch) -> None:
    (fast_repo / "src" / "app.py").write_text("def replacement():\n    return 3\n")

    def _fail_parse(*_args, **_kwargs):
        raise RuntimeError("synthetic parser crash")

    monkeypatch.setattr(ASTParser, "parse_file", _fail_parse)
    result = await capture_source_changes(fast_repo, ["src/app.py"])

    symbols, updates = await _stored(fast_repo)
    assert result.retained_stale == 1
    assert [symbol.name for symbol in symbols] == ["old_name"]
    assert json.loads(updates[0].change_set_json)[0]["parse_state"] == "failed"


async def test_outbox_failure_rolls_back_symbol_replacement(fast_repo: Path, monkeypatch) -> None:
    from repowise.core.source_search import outbox

    (fast_repo / "src" / "app.py").write_text("def uncommitted_name():\n    return 4\n")

    async def _fail_enqueue(*_args, **_kwargs):
        raise RuntimeError("outbox rejected")

    monkeypatch.setattr(outbox, "enqueue_incremental_update", _fail_enqueue)
    with pytest.raises(RuntimeError, match="outbox rejected"):
        await capture_source_changes(fast_repo, ["src/app.py"])

    symbols, updates = await _stored(fast_repo)
    assert [symbol.name for symbol in symbols] == ["old_name"]
    assert updates == []
