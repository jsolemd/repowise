"""A parser witness certifies committed, complete SQL symbol coverage."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest
from sqlalchemy import select, update

from repowise.core.ingestion import ASTParser, FileTraverser
from repowise.core.ingestion.parse_cache import parser_fingerprint
from repowise.core.persistence.crud import reconcile_symbols_for_files
from repowise.core.persistence.models import Repository, WikiSymbol
from repowise.core.persistence.parser_state import (
    complete_symbol_parser_refresh,
    stored_symbol_parser,
    symbol_parser_refresh_required,
)
from repowise.core.pipeline.persist import persist_incremental_symbols
from tests.unit.persistence.helpers import insert_repo


def _parse(root):
    return [
        ASTParser().parse_file(info, Path(info.abs_path).read_bytes())
        for info in FileTraverser(root).traverse()
    ]


async def _seed(session, root):
    repo = await insert_repo(session)
    (root / "a.py").write_text('def alpha():\n    """Fresh documentation."""\n    return 1\n')
    (root / "b.py").write_text("def beta():\n    return 2\n")
    parsed = _parse(root)
    for pf in parsed:
        for symbol in pf.symbols:
            symbol.file_path = pf.file_info.path
    await reconcile_symbols_for_files(
        session,
        repo.id,
        [pf.file_info.path for pf in parsed],
        [symbol for pf in parsed for symbol in pf.symbols],
    )
    await session.execute(update(WikiSymbol).values(docstring=None))
    await session.execute(
        update(Repository)
        .where(Repository.id == repo.id)
        .values(symbols_parser_fingerprint="old-parser")
    )
    await session.commit()
    return repo.id, parsed


@pytest.mark.parametrize("failure", ["omitted", "unreadable", "raced", "prune-refused"])
async def test_partial_refresh_cannot_stamp_or_overwrite_last_good_transaction(
    async_session, tmp_path, monkeypatch, failure
):
    repo_id, parsed = await _seed(async_session, tmp_path)
    if failure == "omitted":
        parsed = [pf for pf in parsed if pf.file_info.path != "b.py"]
    elif failure == "raced":
        (tmp_path / "b.py").write_text("def changed_after_parse():\n    return 3\n")
    elif failure == "unreadable":
        original = Path.read_bytes

        def read(path):
            if path == tmp_path / "b.py":
                raise PermissionError("temporarily unreadable")
            return original(path)

        monkeypatch.setattr(Path, "read_bytes", read)

    refresh = await persist_incremental_symbols(async_session, repo_id, parsed, [])
    assert refresh is not None
    with pytest.raises(RuntimeError, match="Symbol parser refresh incomplete"):
        await complete_symbol_parser_refresh(
            async_session,
            repo_id,
            refresh,
            required_paths={"a.py", "b.py"},
            prune_ready=failure != "prune-refused",
        )
    await async_session.rollback()
    assert await stored_symbol_parser(async_session, repo_id) == "old-parser"
    assert (
        await async_session.execute(select(WikiSymbol.docstring).where(WikiSymbol.name == "alpha"))
    ).scalar_one() is None


async def test_retained_sql_path_missing_from_traversal_prevents_stamp(async_session, tmp_path):
    repo_id, parsed = await _seed(async_session, tmp_path)
    parsed = [pf for pf in parsed if pf.file_info.path == "a.py"]
    refresh = await persist_incremental_symbols(async_session, repo_id, parsed, [])
    with pytest.raises(RuntimeError, match=r"b\.py"):
        await complete_symbol_parser_refresh(
            async_session,
            repo_id,
            refresh,
            required_paths={"a.py"},
        )
    await async_session.rollback()
    assert await stored_symbol_parser(async_session, repo_id) == "old-parser"


async def test_empty_successful_parse_counts_as_refreshed(async_session, tmp_path):
    repo_id, _ = await _seed(async_session, tmp_path)
    (tmp_path / "b.py").write_text("# the old symbol was removed\n")
    refresh = await persist_incremental_symbols(async_session, repo_id, _parse(tmp_path), [])
    await complete_symbol_parser_refresh(
        async_session, repo_id, refresh, required_paths={"a.py", "b.py"}
    )
    await async_session.commit()
    assert await stored_symbol_parser(async_session, repo_id) == parser_fingerprint()
    assert (
        await async_session.execute(select(WikiSymbol).where(WikiSymbol.name == "beta"))
    ).scalar_one_or_none() is None
    assert await persist_incremental_symbols(async_session, repo_id, _parse(tmp_path), []) is None


async def test_full_refresh_invalidates_old_resume_completions_atomically(async_session, tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from repowise.core.persistence.models import PipelineJob
    from repowise.core.pipeline.resume import ResumeController, ResumeLedger, ResumePhase

    repo_id, parsed = await _seed(async_session, tmp_path)
    # More than the native list_jobs page size must be invalidated, while
    # unrelated jobs remain valid.
    for index in range(103):
        async_session.add(
            PipelineJob(
                repository_id=repo_id, phase="analysis", state="completed", id=f"old-{index}"
            )
        )
    async_session.add(PipelineJob(repository_id=repo_id, phase="index", state="completed"))
    async_session.add(PipelineJob(repository_id=repo_id, phase="unrelated", state="completed"))
    await async_session.commit()
    factory = async_sessionmaker(async_session.bind, expire_on_commit=False)
    controller = ResumeController(factory, repo_id, resume=True)
    assert not await controller.can_skip(ResumePhase.INDEX)
    refresh = await persist_incremental_symbols(async_session, repo_id, parsed, [])
    await complete_symbol_parser_refresh(
        async_session, repo_id, refresh, required_paths={"a.py", "b.py"}
    )
    await async_session.commit()
    await ResumeLedger(factory, repo_id).mark_completed(ResumePhase.INDEX)
    assert not await controller.can_skip(ResumePhase.ANALYSIS)
    restarted = ResumeController(factory, repo_id, resume=True)
    assert await restarted.can_skip(ResumePhase.INDEX)
    assert not await restarted.can_skip(ResumePhase.ANALYSIS)
    rows = list((await async_session.execute(select(PipelineJob))).scalars())
    assert all(row.state == "cancelled" for row in rows if row.id.startswith("old-"))
    assert next(row for row in rows if row.phase == "unrelated").state == "completed"


@pytest.mark.parametrize("omit_file", [False, True], ids=["complete", "partial-parse"])
async def test_resume_checkpoint_witness_and_receipt_require_full_traversal(
    async_session, tmp_path, monkeypatch, omit_file
):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from repowise.core.ingestion import GraphBuilder
    from repowise.core.persistence.models import SourceIndexUpdate
    from repowise.core.pipeline.resume import ResumeController

    repository = await insert_repo(async_session, local_path=str(tmp_path))
    repo_id = repository.id
    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n")
    (tmp_path / "b.py").write_text("def beta():\n    return 2\n")
    infos = list(FileTraverser(tmp_path).traverse())
    parsed = _parse(tmp_path)
    if omit_file:
        parsed = [pf for pf in parsed if pf.file_info.path != "b.py"]
    graph = GraphBuilder(tmp_path)
    for pf in parsed:
        graph.add_file(pf)
    graph.build()
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")
    factory = async_sessionmaker(async_session.bind, expire_on_commit=False)
    controller = ResumeController(factory, repo_id, resume=False)
    await controller.checkpoint_index(
        parsed_files=parsed, file_infos=infos, graph_builder=graph, git_metadata_list=[]
    )
    assert controller.index_persisted is not omit_file
    assert await stored_symbol_parser(async_session, repo_id) == (
        None if omit_file else parser_fingerprint()
    )
    receipts = list((await async_session.execute(select(SourceIndexUpdate))).scalars())
    assert len(receipts) == (0 if omit_file else 1)
    if receipts:
        assert receipts[0].mode == "full"
        assert receipts[0].upstream_ready


async def test_missing_store_inspection_does_not_create_database(tmp_path):
    assert not await symbol_parser_refresh_required(tmp_path)
    assert not (tmp_path / ".repowise").exists()


@pytest.mark.parametrize("configured", [False, True], ids=["explicit", "environment"])
async def test_missing_external_sqlite_is_not_created(tmp_path, monkeypatch, configured):
    database = tmp_path / "external.db"
    url = f"sqlite+aiosqlite:///{database}"
    if configured:
        monkeypatch.setenv("REPOWISE_DB_URL", url)
    assert not await symbol_parser_refresh_required(tmp_path, db_url=None if configured else url)
    assert not database.exists()
    assert not (tmp_path / ".repowise").exists()


async def test_memory_sqlite_is_still_inspected(tmp_path):
    assert await symbol_parser_refresh_required(tmp_path, db_url="sqlite+aiosqlite:///:memory:")


async def test_moved_single_repository_uses_source_loader_fallback(tmp_path):
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        get_session,
        init_db,
        resolve_db_url,
    )

    engine = create_engine(resolve_db_url(tmp_path))
    try:
        await init_db(engine)
        async with get_session(create_session_factory(engine)) as session:
            session.add(Repository(name="moved", local_path=str(tmp_path / "old-location")))
        assert await symbol_parser_refresh_required(tmp_path)
        async with get_session(create_session_factory(engine)) as session:
            await session.execute(
                update(Repository).values(symbols_parser_fingerprint=parser_fingerprint())
            )
        assert not await symbol_parser_refresh_required(tmp_path)
    finally:
        await engine.dispose()


async def test_metadata_only_caller_does_not_attempt_or_certify_parse(async_session):
    repository = await insert_repo(async_session)
    assert await persist_incremental_symbols(async_session, repository.id, None, []) is None
    assert await stored_symbol_parser(async_session, repository.id) is None


async def test_blocked_receipt_promotion_rolls_back_with_symbol_witness(async_session, tmp_path):
    from repowise.core.persistence.models import SourceIndexUpdate
    from repowise.core.source_search.outbox import enqueue_full_update

    repo_id, parsed = await _seed(async_session, tmp_path)
    blocked = await enqueue_full_update(
        async_session,
        repo_id,
        tmp_path,
        parsed_files=parsed,
        upstream_ready=False,
        upstream_error="legacy failure",
    )
    generation = blocked.generation_id
    await async_session.commit()
    refresh = await persist_incremental_symbols(async_session, repo_id, parsed, [])
    await complete_symbol_parser_refresh(
        async_session, repo_id, refresh, required_paths={"a.py", "b.py"}
    )
    reopened = await enqueue_full_update(
        async_session,
        repo_id,
        tmp_path,
        parsed_files=parsed,
        upstream_refreshed=True,
    )
    assert reopened.generation_id == generation
    assert reopened.state == "pending"
    await async_session.rollback()
    assert await stored_symbol_parser(async_session, repo_id) == "old-parser"
    assert (
        await async_session.execute(
            select(
                SourceIndexUpdate.state,
                SourceIndexUpdate.upstream_ready,
                SourceIndexUpdate.last_error,
            )
        )
    ).one() == ("blocked", False, "legacy failure")


async def test_legacy_column_is_read_without_migration_then_bootstrapped(tmp_path):
    from sqlalchemy import text

    from repowise.core.persistence.database import create_engine, init_db, resolve_db_url

    engine = create_engine(resolve_db_url(tmp_path))
    try:
        await init_db(engine)
        async with engine.begin() as conn:
            await conn.execute(
                text("ALTER TABLE repositories DROP COLUMN symbols_parser_fingerprint")
            )
        assert await symbol_parser_refresh_required(tmp_path)
        async with engine.connect() as conn:
            columns = await conn.execute(text("PRAGMA table_info(repositories)"))
            assert "symbols_parser_fingerprint" not in {row[1] for row in columns}
        await init_db(engine)
        async with engine.connect() as conn:
            columns = await conn.execute(text("PRAGMA table_info(repositories)"))
            assert "symbols_parser_fingerprint" in {row[1] for row in columns}
    finally:
        await engine.dispose()


def test_fork_migration_has_one_head_and_postgresql_additive_ddl():
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parents[3] / "packages/core"
    config = Config()
    config.set_main_option("script_location", str(root / "alembic"))
    assert ScriptDirectory.from_config(config).get_heads() == ["solemd_0001"]
    path = root / "alembic/versions/solemd_0001_symbol_parser.py"
    spec = importlib.util.spec_from_file_location("symbol_parser_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        migration.upgrade()
        migration.downgrade()
    ddl = output.getvalue()
    assert "ADD COLUMN symbols_parser_fingerprint VARCHAR(64)" in ddl
    assert "DROP COLUMN symbols_parser_fingerprint" in ddl
