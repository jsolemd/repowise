"""Concurrency and publication contracts at the agent's read boundary."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import event

from repowise.core.persistence import database
from repowise.core.persistence.models import Repository, SourceIndexUpdate
from repowise.core.persistence.vector_store.lancedb_store import LanceDBVectorStore
from repowise.core.providers.embedding.base import MockEmbedder
from repowise.core.source_search import status as status_module
from repowise.core.source_search.generation import GenerationRef
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    default_manifest_path,
    write_manifest,
)
from repowise.core.source_search.vector_store import SourceChunkVectorStore
from repowise.server import source_search_wiring as wiring


@pytest.fixture
async def queue_database(tmp_path):
    engine = database.create_engine(database.resolve_db_url(tmp_path))
    await database.init_db(engine)
    factory = database.create_session_factory(engine)
    async with database.get_session(factory) as session:
        session.add(Repository(id="repo", name="repo", local_path=str(tmp_path)))
        await session.flush()
        session.add(
            SourceIndexUpdate(
                repository_id="repo",
                generation_id="active",
                state="published",
                sequence=1,
                dedupe_key="active",
                parser_fingerprint="parser",
            )
        )
    try:
        yield engine, factory
    finally:
        await engine.dispose()


async def test_empty_queue_skips_error_lookup(tmp_path, queue_database, monkeypatch):
    engine, _ = queue_database
    statements = []
    event.listen(
        engine.sync_engine,
        "before_cursor_execute",
        lambda conn, cursor, statement, *args: statements.append(statement),
    )
    # The standalone path owns and disposes this engine; count its real SQL.
    monkeypatch.setattr(database, "create_engine", lambda *args: engine)
    snapshot = await status_module._source_update_snapshot(
        tmp_path,
        None,
        active_sequence=1,
        active_generation_id="active",
    )
    assert snapshot.active.generation_id == "active"
    assert snapshot.counts == {}
    assert snapshot.last_error is None
    assert len(statements) == 3  # repository, active generation, grouped queue


async def test_status_borrows_factory_and_reads_new_queue_state(
    tmp_path,
    queue_database,
    monkeypatch,
):
    engine, factory = queue_database
    create = Mock(side_effect=AssertionError("must use the host's engine"))
    disposals = []
    event.listen(engine.sync_engine, "engine_disposed", lambda engine: disposals.append(engine))
    monkeypatch.setattr(database, "create_engine", create)
    # No manifest means sequence zero: every queued generation is outstanding.
    first = await status_module.inspect_source_index(
        tmp_path,
        verify_stores=False,
        session_factory=factory,
    )
    async with database.get_session(factory) as session:
        session.add(
            SourceIndexUpdate(
                repository_id="repo",
                generation_id="blocked",
                state="blocked",
                sequence=2,
                dedupe_key="blocked",
                parser_fingerprint="parser",
                last_error="embedding unavailable",
            )
        )
    second = await status_module.inspect_source_index(
        tmp_path,
        verify_stores=False,
        session_factory=factory,
    )
    assert first.blocked_updates == 0
    assert second.blocked_updates == 1
    assert second.last_error == "embedding unavailable"
    create.assert_not_called()
    assert disposals == []


async def test_search_status_uses_its_repository_factory(tmp_path, monkeypatch):
    inspect = AsyncMock(side_effect=RuntimeError("controlled status failure"))
    monkeypatch.setattr(status_module, "inspect_source_index", inspect)
    factory = object()
    coordinator = wiring._StatusCoordinator(
        SimpleNamespace(search=AsyncMock(return_value={"results": []})),
        tmp_path,
        None,
        None,
        session_factory=factory,
    )
    response = await coordinator.search("owner")
    assert inspect.await_args.kwargs["session_factory"] is factory
    assert response["_meta"]["source_search"]["status"] == "unknown"


@pytest.mark.parametrize("cancelled", [False, True])
async def test_borrowed_status_returns_connection_on_failure(
    tmp_path,
    queue_database,
    monkeypatch,
    cancelled,
):
    engine, factory = queue_database
    connections, returned, disposals = [], [], []
    event.listen(engine.sync_engine, "checkout", lambda *args: connections.append(args[1]))
    event.listen(engine.sync_engine, "checkin", lambda *args: returned.append(args[1]))
    event.listen(engine.sync_engine, "engine_disposed", lambda engine: disposals.append(engine))

    def fail(*args):
        if cancelled:
            raise asyncio.CancelledError()
        raise RuntimeError("queue read failed")

    event.listen(engine.sync_engine, "before_cursor_execute", fail)
    create = Mock(side_effect=AssertionError("must use the host's engine"))
    monkeypatch.setattr(database, "create_engine", create)
    try:
        call = status_module.inspect_source_index(
            tmp_path,
            verify_stores=False,
            session_factory=factory,
        )
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await call
        else:
            result = await call
            assert result.state == "inconsistent"
            assert result.integrity_findings[0].component == status_module.COMPONENT_QUEUE
        assert connections and returned == connections
        assert disposals == []
        create.assert_not_called()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", fail)


async def test_standalone_status_disposes_engine_on_failure(tmp_path, queue_database, monkeypatch):
    engine, _ = queue_database
    disposals = []
    event.listen(engine.sync_engine, "engine_disposed", lambda engine: disposals.append(engine))
    monkeypatch.setattr(database, "create_engine", lambda *args: engine)
    monkeypatch.setattr(
        database, "create_session_factory", Mock(side_effect=RuntimeError("factory"))
    )
    result = await status_module.inspect_source_index(tmp_path, verify_stores=False)
    assert result.state == "inconsistent"
    assert disposals == [engine.sync_engine]


@pytest.mark.parametrize("store_type", [SourceChunkVectorStore, LanceDBVectorStore])
async def test_concurrent_initialization_waits_for_the_tables(tmp_path, monkeypatch, store_type):
    lancedb = pytest.importorskip("lancedb")
    entered, release = asyncio.Event(), asyncio.Event()
    table = SimpleNamespace(schema=AsyncMock(return_value=SimpleNamespace(names=[])))

    async def open_table(name):
        entered.set()
        await release.wait()
        return table

    db = SimpleNamespace(open_table=open_table, table_names=AsyncMock(return_value=["wiki_pages"]))
    connect = AsyncMock(return_value=db)
    monkeypatch.setattr(lancedb, "connect_async", connect)
    store = store_type(str(tmp_path), MockEmbedder())
    first = asyncio.create_task(store._ensure_connected())
    await entered.wait()
    second = asyncio.create_task(store._ensure_connected())
    await asyncio.sleep(0)
    assert not second.done()
    assert store._db is None
    release.set()
    await asyncio.gather(first, second)
    assert store._table is table
    assert connect.await_count == 1
    await store.close()


async def test_failed_initialization_can_retry(tmp_path, monkeypatch):
    lancedb = pytest.importorskip("lancedb")
    table = SimpleNamespace(schema=AsyncMock(return_value=SimpleNamespace(names=[])))
    opener = AsyncMock(side_effect=[RuntimeError("open failed"), table, table])
    monkeypatch.setattr(
        lancedb, "connect_async", AsyncMock(return_value=SimpleNamespace(open_table=opener))
    )
    store = SourceChunkVectorStore(str(tmp_path), MockEmbedder())
    with pytest.raises(RuntimeError, match="open failed"):
        await store._ensure_connected()
    assert store._db is None
    await store._ensure_connected()
    assert store._table is table
    await store.close()


async def test_response_keeps_its_read_generation_across_publication(tmp_path, monkeypatch):
    before = GenerationRef("before", 1)
    after = SourceIndexManifest(
        recipe_fingerprint="recipe",
        corpus_hash="new-corpus",
        symbol_chunks=1,
        file_window_chunks=0,
        files_covered=1,
        indexed_commit="new-commit",
        built_at="2026-09-05T00:00:00Z",
        embedder=EmbedderIdentity("mock", "mock", 8),
        generation_id="after",
        generation_sequence=2,
    )

    async def search(*args, **kwargs):
        write_manifest(default_manifest_path(tmp_path), after)
        return {"results": [], "_meta": {"source_search": {"indexed_commit": "old-commit"}}}

    monkeypatch.setattr(
        status_module,
        "_source_update_snapshot",
        AsyncMock(return_value=status_module._SourceUpdateSnapshot(None, {}, 0, None)),
    )
    monkeypatch.setattr(
        status_module, "working_tree_candidates", lambda *args, **kwargs: ({}, None)
    )
    fts = SimpleNamespace(generation=before, indexed_among=lambda paths: set())
    coordinator = wiring._StatusCoordinator(
        SimpleNamespace(search=search), tmp_path, SimpleNamespace(), fts
    )
    response = await coordinator.search("owner")
    meta = response["_meta"]["source_search"]
    assert meta["generation_id"] == "before"
    assert meta["generation_sequence"] == 1
    assert meta["published_generation_id"] == "after"
    assert meta["indexed_commit"] == "old-commit"
    assert meta["status"] == "stale"
    assert meta["working_tree"]["checked"] is False
    assert meta["working_tree"]["unavailable_reason"] == "reader_generation_changed"


async def test_cancelled_borrower_releases_its_retired_reader(monkeypatch):
    monkeypatch.setattr(wiring, "_retired", [])
    monkeypatch.setattr(wiring, "_borrowers", {})
    entered = asyncio.Event()
    coordinator = SimpleNamespace(close=AsyncMock())

    async def query():
        async with wiring.coordinator_lease(coordinator):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(query())
    await entered.wait()
    wiring._retire(coordinator)
    await wiring._sweep_retired()
    coordinator.close.assert_not_awaited()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    coordinator.close.assert_awaited_once()
    assert wiring._borrowers == {}
    assert wiring._retired == []


@pytest.mark.parametrize("fts_fails", [False, True])
async def test_failed_query_records_the_failure_and_read_generation(tmp_path, fts_fails):
    from repowise.core.source_search.coordinator import SourceSearchCoordinator

    events = []

    def fail(*args, **kwargs):
        raise RuntimeError("controlled retrieval failure")

    write_manifest(
        default_manifest_path(tmp_path),
        SourceIndexManifest(
            recipe_fingerprint="recipe",
            corpus_hash="stable-corpus",
            symbol_chunks=1,
            file_window_chunks=0,
            files_covered=1,
            indexed_commit="commit",
            built_at="2026-09-05T00:00:00Z",
            embedder=EmbedderIdentity("mock", "mock", 8),
            generation_id="read-generation",
            generation_sequence=3,
        ),
    )
    coordinator = SourceSearchCoordinator(
        repo_path=tmp_path,
        embedder=SimpleNamespace(embed=AsyncMock(side_effect=fail)),
        source_vectors=SimpleNamespace(),
        source_fts=SimpleNamespace(
            query=fail if fts_fails else lambda *args, **kwargs: [],
            active_file_paths=lambda: set(),
            term_file_evidence=lambda terms: {},
        ),
        query_log=SimpleNamespace(append=lambda event: events.append(event.to_dict())),
    )
    response = await coordinator.search("owner")
    assert response.get("status", "ok") == events[0]["status"] == ("error" if fts_fails else "ok")
    assert events[0]["error_code"] == response.get("error", {}).get("code")
    assert len(events[0]["failed_legs"]) == (2 if fts_fails else 1)
    assert events[0]["generation"] == "read-generation"
