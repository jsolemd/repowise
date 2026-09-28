"""Deletion retries recheck committed page authority before touching an index."""

from unittest.mock import Mock

import pytest
from sqlalchemy import event
from sqlalchemy.exc import SQLAlchemyError

from repowise.core.persistence.crud import upsert_page
from repowise.core.pipeline.cleanup_debt import (
    exclude_live_cleanup_ids,
    load_cleanup_debt,
    record_cleanup_debt,
)
from tests.unit.persistence.helpers import insert_repo, make_page_kwargs


@pytest.mark.parametrize("live_status", ["fresh", "stale", "expired"])
async def test_committed_live_ids_cancel_both_debts(
    async_engine,
    async_session,
    tmp_path,
    live_status,
):
    repo = await insert_repo(async_session)
    for page_id, status in [("restored", live_status), ("gone", "tombstone")]:
        await upsert_page(
            async_session,
            **make_page_kwargs(
                repo.id,
                page_id=page_id,
                freshness_status=status,
            ),
        )
    await async_session.commit()
    ids = {"restored", "gone", "swept"}
    for kind in ("fts", "vectors"):
        record_cleanup_debt(tmp_path, kind, ids)

    assert await exclude_live_cleanup_ids(tmp_path, async_engine, ids) == {"gone", "swept"}
    assert load_cleanup_debt(tmp_path) == {
        "fts": {"gone", "swept"},
        "vectors": {"gone", "swept"},
    }
    assert ids == {"restored", "gone", "swept"}


async def test_debt_authority_reads_are_bounded(async_engine, tmp_path):
    sizes = []

    def capture(conn, cursor, statement, parameters, context, many):
        sizes.append(len(parameters))

    event.listen(async_engine.sync_engine, "before_cursor_execute", capture)
    ids = {f"swept-{n}" for n in range(1001)}
    try:
        assert await exclude_live_cleanup_ids(tmp_path, async_engine, ids) == ids
    finally:
        event.remove(async_engine.sync_engine, "before_cursor_execute", capture)
    assert len(sizes) == 3 and max(sizes) <= 501  # 500 IDs plus tombstone status.


async def test_failed_authority_read_leaves_both_debts(async_engine, tmp_path):
    for kind in ("fts", "vectors"):
        record_cleanup_debt(tmp_path, kind, {"restore"})

    def fail(*args):
        raise SQLAlchemyError("unavailable")

    event.listen(async_engine.sync_engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(SQLAlchemyError, match="unavailable"):
            await exclude_live_cleanup_ids(tmp_path, async_engine, {"restore"})
    finally:
        event.remove(async_engine.sync_engine, "before_cursor_execute", fail)
    assert load_cleanup_debt(tmp_path) == {"fts": {"restore"}, "vectors": {"restore"}}


async def test_empty_cleanup_never_opens_authority(tmp_path):
    engine = Mock()
    assert await exclude_live_cleanup_ids(tmp_path, engine, set()) == set()
    engine.connect.assert_not_called()


async def test_hostless_incremental_does_not_restore_cleared_debt(tmp_path):
    from types import SimpleNamespace

    import networkx as nx

    from repowise.cli._repo_session import open_repo_db
    from repowise.core.persistence.database import get_session
    from repowise.core.persistence.search import FullTextSearch
    from repowise.core.pipeline.incremental import persist_incremental_index

    (tmp_path / "restored.py").write_text("value = 1\n")
    page_id = "file_page:restored.py"
    engine, factory, repo_id = await open_repo_db(tmp_path, repo_name="test")
    try:
        async with get_session(factory) as session:
            await upsert_page(
                session,
                **make_page_kwargs(
                    repo_id,
                    page_id=page_id,
                    target_path="restored.py",
                ),
            )
        fts = FullTextSearch(engine)
        await fts.index(page_id, "Restored", "xylophone", target_path="restored.py")
        for kind in ("fts", "vectors"):
            record_cleanup_debt(tmp_path, kind, {page_id})
        graph = nx.DiGraph()
        graph.add_node("restored.py", node_type="file")

        await persist_incremental_index(
            tmp_path,
            SimpleNamespace(graph=lambda: graph),
            {},
            None,
            None,
            [],
            log=lambda message: None,
            parsed_files=[],
            vector_store=None,
        )

        assert load_cleanup_debt(tmp_path) == {"fts": set(), "vectors": set()}
        assert [hit.page_id for hit in await fts.search("xylophone")] == [page_id]
    finally:
        await engine.dispose()


async def test_full_index_records_both_debts_before_fts_failure(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from sqlalchemy import delete

    from repowise.cli._repo_session import open_repo_db
    from repowise.cli.commands.update_cmd.incremental import cleanup_retired_page_vectors
    from repowise.core.persistence.database import get_session
    from repowise.core.persistence.models import Page
    from repowise.core.persistence.search import FullTextSearch
    from repowise.core.pipeline import full_index, persist

    page_id = "module_page:retired"
    engine, factory, repo_id = await open_repo_db(tmp_path, repo_name="test")
    try:
        async with get_session(factory) as session:
            await upsert_page(session, **make_page_kwargs(repo_id, page_id=page_id))

        async def sweep(_result, session, _repo_id, **kwargs):
            await session.execute(delete(Page).where(Page.id == page_id))
            return [page_id]

        monkeypatch.setattr(
            "repowise.core.pipeline.run_pipeline",
            AsyncMock(
                return_value=SimpleNamespace(repo_name="test"),
            ),
        )
        monkeypatch.setattr(persist, "persist_pipeline_result", sweep)
        monkeypatch.setattr(
            FullTextSearch,
            "delete_many",
            AsyncMock(
                side_effect=OSError("FTS unavailable"),
            ),
        )

        with pytest.raises(OSError, match="FTS unavailable"):
            await full_index.index_repo_full(tmp_path)

        async with get_session(factory) as session:
            assert await session.get(Page, page_id) is None
        assert load_cleanup_debt(tmp_path) == {"fts": {page_id}, "vectors": {page_id}}
        store = SimpleNamespace(delete_many=AsyncMock())
        await asyncio.to_thread(cleanup_retired_page_vectors, tmp_path, [], vector_store=store)
        store.delete_many.assert_awaited_once_with([page_id])
        assert load_cleanup_debt(tmp_path) == {"fts": {page_id}, "vectors": set()}
    finally:
        await engine.dispose()
