"""Server cleanup follows committed page authority and each repository's stores."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from repowise.core.persistence.crud import (
    get_generation_job,
    upsert_generation_job,
    upsert_repository,
)
from repowise.core.persistence.database import create_engine, init_db
from repowise.core.persistence.models import Page
from repowise.core.persistence.search import FullTextSearch
from repowise.core.persistence.vector_store import InMemoryVectorStore
from repowise.core.pipeline.cleanup_debt import load_cleanup_debt, record_cleanup_debt
from repowise.core.pipeline.persist import tombstone_file_pages_outside_scope
from repowise.core.providers.embedding.base import MockEmbedder
from repowise.core.update_lock import (
    read_update_lock,
    release_update_lock,
    try_acquire_update_lock,
)
from repowise.server.job_executor import execute_job

PAGE_ID = "file_page:src/main.py"


class _CommitFailureSession(AsyncSession):
    async def commit(self):
        if self.info.pop("fail_page_commit", False):
            raise RuntimeError("page commit failed")
        await super().commit()


async def _catalog(root):
    (root / ".repowise").mkdir(parents=True)
    engine = create_engine(f"sqlite+aiosqlite:///{root / '.repowise/wiki.db'}")
    await init_db(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=_CommitFailureSession)
    fts = FullTextSearch(engine)
    await fts.ensure_index()
    vectors = InMemoryVectorStore(MockEmbedder())
    async with factory() as session:
        repo = await upsert_repository(session, name=root.name, local_path=str(root))
        now = datetime.now(UTC)
        session.add(Page(
            id=PAGE_ID, repository_id=repo.id, page_type="file_page",
            title="quartzprobe", content="quartzprobe implementation",
            target_path="src/main.py", source_hash="x" * 64,
            model_name="mock", provider_name="mock", created_at=now, updated_at=now,
        ))
        await session.commit()
    await fts.index(PAGE_ID, "quartzprobe", "quartzprobe implementation", target_path="src/main.py")
    await vectors.embed_and_upsert(PAGE_ID, "quartzprobe", {"title": "quartzprobe"})
    state = SimpleNamespace(session_factory=factory, fts=fts, vector_store=vectors)
    return SimpleNamespace(
        root=root, engine=engine, factory=factory, fts=fts, vectors=vectors, repo=repo, state=state,
        retained_paths=set(), fail_commit=False, sweeps=[],
    )


@pytest.fixture
async def catalog(tmp_path, monkeypatch):
    import repowise.server.job_executor as executor

    cat = await _catalog(tmp_path / "target")
    monkeypatch.setenv("REPOWISE_TOOLS_NO_GENERATIVE", "1")
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "0")
    monkeypatch.setenv("REPOWISE_INDEX_INFORMATION_FLOOR", "0")
    cat.pipeline = AsyncMock(return_value=SimpleNamespace(
        generated_pages=[], parsed_files=[], file_count=1, symbol_count=2,
    ))
    monkeypatch.setattr(executor, "run_pipeline", cat.pipeline)

    async def persist_subset(_result, session, repo_id):
        assert read_update_lock(cat.root) is not None
        ids = await tombstone_file_pages_outside_scope(session, repo_id, cat.retained_paths)
        cat.sweeps.append(ids)
        session.info["fail_page_commit"] = cat.fail_commit
        return ids

    monkeypatch.setattr(executor, "persist_pipeline_result", persist_subset)
    monkeypatch.setattr(executor, "_stamp_last_sync_commit", Mock())
    monkeypatch.setattr(
        "repowise.server.search_helpers.resolve_repo_vector_store",
        AsyncMock(return_value=cat.vectors),
    )
    monkeypatch.setattr(
        "repowise.server.source_search_lifecycle.reconcile_server_source_index", AsyncMock(),
    )
    try:
        yield cat
    finally:
        await cat.vectors.close()
        await cat.engine.dispose()


async def _job(cat, state=None):
    async with cat.factory() as session:
        job = await upsert_generation_job(session, repository_id=cat.repo.id, config={"mode": "index_only"})
        await session.commit()
    await execute_job(job.id, state or cat.state, session_factory_override=cat.factory)
    async with cat.factory() as session:
        return await get_generation_job(session, job.id)


async def _freshness(cat):
    async with cat.factory() as session:
        return (await session.get(Page, PAGE_ID)).freshness_status


async def test_failed_sql_commit_preserves_both_search_indexes(catalog):
    catalog.fail_commit = True
    job = await _job(catalog)
    assert job.status == "failed"
    assert "page commit failed" in job.error_message
    assert await _freshness(catalog) == "fresh"
    assert await catalog.vectors.list_page_ids() == {PAGE_ID}
    assert await catalog.fts.list_indexed_ids() == {PAGE_ID}
    assert load_cleanup_debt(catalog.root) == {"fts": set(), "vectors": set()}
    assert read_update_lock(catalog.root) is None


@pytest.mark.parametrize("kind", ["fts", "vectors"])
async def test_failed_cleanup_retries_without_new_retirees(catalog, monkeypatch, kind):
    store = getattr(catalog, kind)
    delete = store.delete_many
    monkeypatch.setattr(store, "delete_many", AsyncMock(side_effect=OSError(f"{kind} unavailable")))
    job = await _job(catalog)
    assert job.status == "failed"
    assert await _freshness(catalog) == "tombstone"
    other = "vectors" if kind == "fts" else "fts"
    assert load_cleanup_debt(catalog.root) == {kind: {PAGE_ID}, other: set()}
    monkeypatch.setattr(store, "delete_many", delete)
    assert (await _job(catalog)).status == "completed"
    assert catalog.sweeps == [[PAGE_ID], []]
    assert await catalog.vectors.list_page_ids() == set()
    assert await catalog.fts.list_indexed_ids() == set()
    assert load_cleanup_debt(catalog.root) == {"fts": set(), "vectors": set()}
    assert read_update_lock(catalog.root) is None


async def test_cancelled_cleanup_keeps_both_intents_for_retry(catalog, monkeypatch):
    import asyncio

    delete = catalog.fts.delete_many
    monkeypatch.setattr(catalog.fts, "delete_many", AsyncMock(side_effect=asyncio.CancelledError))
    assert (await _job(catalog)).status == "cancelled"
    assert load_cleanup_debt(catalog.root) == {"fts": {PAGE_ID}, "vectors": {PAGE_ID}}
    assert read_update_lock(catalog.root) is None
    monkeypatch.setattr(catalog.fts, "delete_many", delete)
    assert (await _job(catalog)).status == "completed"
    assert load_cleanup_debt(catalog.root) == {"fts": set(), "vectors": set()}


async def test_restored_page_invalidates_old_cleanup_intent(catalog):
    catalog.retained_paths = {"src/main.py"}
    for kind in ("fts", "vectors"):
        record_cleanup_debt(catalog.root, kind, {PAGE_ID})
    assert (await _job(catalog)).status == "completed"
    assert await _freshness(catalog) == "fresh"
    assert await catalog.vectors.list_page_ids() == {PAGE_ID}
    assert await catalog.fts.list_indexed_ids() == {PAGE_ID}
    assert load_cleanup_debt(catalog.root) == {"fts": set(), "vectors": set()}


@pytest.mark.parametrize("cached", [True, False])
async def test_workspace_cleanup_cannot_delete_primary_repo_fts(catalog, tmp_path, cached):
    primary = await _catalog(tmp_path / "primary")
    primary.state.workspace_fts = {catalog.repo.id: catalog.fts} if cached else {}
    try:
        assert (await _job(catalog, primary.state)).status == "completed"
        assert await primary.fts.list_indexed_ids() == {PAGE_ID}
        assert await _freshness(primary) == "fresh"
        assert await catalog.fts.list_indexed_ids() == set()
        assert await _freshness(catalog) == "tombstone"
    finally:
        await primary.vectors.close()
        await primary.engine.dispose()


@pytest.mark.parametrize("mode", ["busy", "unverified"])
async def test_job_without_verified_update_ownership_does_not_mutate_index(catalog, monkeypatch, mode):
    from repowise.core import update_lock

    for kind in ("fts", "vectors"):
        record_cleanup_debt(catalog.root, kind, {PAGE_ID})
    if mode == "busy":
        assert try_acquire_update_lock(catalog.root, "other-writer") is None
    else:
        monkeypatch.setattr(update_lock.os, "link", Mock(side_effect=OSError("read-only")))
    owner = read_update_lock(catalog.root)
    try:
        job = await _job(catalog)
        assert job.status == "failed"
        assert ("already running" if mode == "busy" else "could not be verified") in job.error_message
        catalog.pipeline.assert_not_awaited()
        assert await _freshness(catalog) == "fresh"
        assert await catalog.fts.list_indexed_ids() == {PAGE_ID}
        assert await catalog.vectors.list_page_ids() == {PAGE_ID}
        assert load_cleanup_debt(catalog.root) == {"fts": {PAGE_ID}, "vectors": {PAGE_ID}}
        assert read_update_lock(catalog.root) == owner
    finally:
        if mode == "busy":
            release_update_lock(catalog.root)


async def test_cleanup_excludes_native_writer_until_both_indexes_finish(catalog, monkeypatch):
    original_delete = catalog.vectors.delete_many

    async def competing_writer(ids):
        # This is after SQL authority was read, exactly where a revived page's
        # vector could otherwise be erased by the retiring job.
        owner = try_acquire_update_lock(catalog.root, "revive-page")
        assert owner is not None, "another writer entered between authority read and deletion"
        await original_delete(ids)

    monkeypatch.setattr(catalog.vectors, "delete_many", competing_writer)
    assert (await _job(catalog)).status == "completed"
    assert read_update_lock(catalog.root) is None


async def test_memory_fallback_cannot_clear_persisted_vector_debt(catalog, monkeypatch):
    import sys

    from repowise.core.persistence.vector_store import LanceDBVectorStore
    from repowise.server.search_helpers import _build_repo_vector_store

    lance_path = catalog.root / ".repowise" / "lancedb"
    durable = LanceDBVectorStore(str(lance_path), MockEmbedder())
    await durable.embed_and_upsert(PAGE_ID, "quartzprobe", {"title": "quartzprobe"})
    await durable.close()

    # Exercise the real builder's optional-dependency fallback while a real
    # persisted table still contains the retired page.
    with monkeypatch.context() as unavailable:
        unavailable.setitem(sys.modules, "lancedb", None)
        fallback = _build_repo_vector_store(catalog.root, MockEmbedder(), create=True)
        assert isinstance(fallback, InMemoryVectorStore)
        unavailable.setattr(
            "repowise.server.search_helpers.resolve_repo_vector_store",
            AsyncMock(return_value=fallback),
        )
        job = await _job(catalog)
        assert job.status == "failed"
        assert "persisted page vectors are unavailable" in job.error_message
        assert load_cleanup_debt(catalog.root) == {"fts": set(), "vectors": {PAGE_ID}}
        await fallback.close()

    durable = LanceDBVectorStore(str(lance_path), MockEmbedder())
    try:
        assert await durable.list_page_ids() == {PAGE_ID}
        monkeypatch.setattr(
            "repowise.server.search_helpers.resolve_repo_vector_store",
            AsyncMock(return_value=durable),
        )
        assert (await _job(catalog)).status == "completed"
        assert await durable.list_page_ids() == set()
        assert load_cleanup_debt(catalog.root) == {"fts": set(), "vectors": set()}
    finally:
        await durable.close()
