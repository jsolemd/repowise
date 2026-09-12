from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from repowise.docs.graph import (
    GraphSyncAction,
    delete_library_metadata,
    get_docs_graph_sync_status,
    reconcile_docs_graph_sync,
    sync_library_metadata,
)
from repowise.docs.library.models import LibraryFile, LibraryState, LibraryStatus


def _library_state() -> LibraryState:
    now = datetime.now(UTC)
    return LibraryState(
        library_id="/mantinedev/mantine",
        repo="mantinedev/mantine",
        name="Mantine",
        description="React components library",
        docs_path="docs",
        branch="master",
        include_patterns=[],
        exclude_patterns=[],
        priority=4,
        status=LibraryStatus.READY,
        current_sha="abcdef123456",
        indexed_at=now,
        freshness_checked_at=now,
        error_message=None,
        chunk_count=12,
        file_count=2,
        created_at=now,
        updated_at=now,
    )


def _library_files() -> list[LibraryFile]:
    now = datetime.now(UTC)
    return [
        LibraryFile(
            library_id="/mantinedev/mantine",
            file_path="docs/button.mdx",
            content_hash="hash-1",
            chunk_count=3,
            indexed_at=now,
            created_at=now,
            updated_at=now,
        ),
        LibraryFile(
            library_id="/mantinedev/mantine",
            file_path="docs/input.mdx",
            content_hash="hash-2",
            chunk_count=4,
            indexed_at=now,
            created_at=now,
            updated_at=now,
        ),
    ]


@pytest.mark.asyncio
async def test_sync_library_metadata_noops_when_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("repowise.docs.graph.docs_graph_enabled", lambda: False)
    monkeypatch.setattr("repowise.docs.graph._get_client", AsyncMock(return_value=None))

    assert await sync_library_metadata(_library_state(), _library_files()) is False


@pytest.mark.asyncio
async def test_sync_library_metadata_writes_library_and_files(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("repowise.docs.graph.docs_graph_enabled", lambda: True)
    fake_client = AsyncMock()
    fake_client.verify_connectivity.return_value = True
    persist_status = AsyncMock()
    monkeypatch.setattr("repowise.docs.graph._get_client", AsyncMock(return_value=fake_client))
    monkeypatch.setattr("repowise.docs.graph._ensure_schema", AsyncMock())
    monkeypatch.setattr("repowise.docs.graph.update_library_status", persist_status)

    assert await sync_library_metadata(_library_state(), _library_files()) is True

    assert fake_client.execute_write.await_count == 3
    library_params = fake_client.execute_write.await_args_list[0].args[1]
    assert library_params["freshness_checked_at"] is not None
    persist_status.assert_awaited_once()
    assert persist_status.await_args.kwargs["graph_synced_at"] is not None
    assert persist_status.await_args.kwargs["graph_sync_error"] == ""
    status = get_docs_graph_sync_status()
    assert status["state"] == "ok"
    assert status["last_library_id"] == "/mantinedev/mantine"


@pytest.mark.asyncio
async def test_delete_library_metadata_uses_neo4j_when_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("repowise.docs.graph.docs_graph_enabled", lambda: True)
    fake_client = AsyncMock()
    fake_client.verify_connectivity.return_value = True
    monkeypatch.setattr("repowise.docs.graph._get_client", AsyncMock(return_value=fake_client))
    monkeypatch.setattr("repowise.docs.graph._ensure_schema", AsyncMock())

    assert await delete_library_metadata("/mantinedev/mantine") is True

    fake_client.execute_write.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_library_metadata_queues_replay_on_neo4j_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("repowise.docs.graph.docs_graph_enabled", lambda: True)
    fake_client = AsyncMock()
    fake_client.verify_connectivity.return_value = False
    queue_job = AsyncMock()
    persist_status = AsyncMock()
    monkeypatch.setattr("repowise.docs.graph._get_client", AsyncMock(return_value=fake_client))
    monkeypatch.setattr("repowise.docs.graph._safe_enqueue_graph_sync_job", queue_job)
    monkeypatch.setattr("repowise.docs.graph.update_library_status", persist_status)

    assert await sync_library_metadata(_library_state(), _library_files()) is False

    queue_job.assert_awaited_once_with("/mantinedev/mantine", GraphSyncAction.UPSERT)
    assert persist_status.await_args.kwargs["graph_sync_error"] == "neo4j unavailable"


@pytest.mark.asyncio
async def test_delete_library_metadata_queues_replay_on_neo4j_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("repowise.docs.graph.docs_graph_enabled", lambda: True)
    fake_client = AsyncMock()
    fake_client.verify_connectivity.return_value = False
    queue_job = AsyncMock()
    monkeypatch.setattr("repowise.docs.graph._get_client", AsyncMock(return_value=fake_client))
    monkeypatch.setattr("repowise.docs.graph._safe_enqueue_graph_sync_job", queue_job)

    assert await delete_library_metadata("/mantinedev/mantine") is False

    queue_job.assert_awaited_once_with("/mantinedev/mantine", GraphSyncAction.DELETE)


@pytest.mark.asyncio
async def test_reconcile_docs_graph_sync_marks_missing_library_metadata_pending(
    monkeypatch: pytest.MonkeyPatch,
):
    library = _library_state().model_copy(
        update={
            "graph_synced_at": datetime.now(UTC),
            "graph_sync_error": "",
        }
    )
    fake_client = AsyncMock()
    fake_client.verify_connectivity.return_value = True
    mark_pending = AsyncMock()
    enqueue = AsyncMock()

    monkeypatch.setattr("repowise.docs.graph.docs_graph_enabled", lambda: True)
    monkeypatch.setattr("repowise.docs.graph._get_client", AsyncMock(return_value=fake_client))
    monkeypatch.setattr("repowise.docs.graph._ensure_schema", AsyncMock())
    monkeypatch.setattr("repowise.docs.graph.list_libraries", AsyncMock(return_value=[library]))
    monkeypatch.setattr(
        "repowise.docs.graph._fetch_docs_graph_file_counts", AsyncMock(return_value={})
    )
    monkeypatch.setattr("repowise.docs.graph._list_graph_sync_jobs", AsyncMock(return_value={}))
    monkeypatch.setattr("repowise.docs.graph.mark_graph_sync_pending", mark_pending)
    monkeypatch.setattr("repowise.docs.graph.enqueue_graph_sync_job", enqueue)

    summary = await reconcile_docs_graph_sync()

    assert summary == {
        "eligible_libraries": 1,
        "gap_libraries": 1,
        "jobs_enqueued": 1,
    }
    mark_pending.assert_awaited_once_with(
        "/mantinedev/mantine",
        error="graph metadata reconciliation pending",
    )
    enqueue.assert_awaited_once_with("/mantinedev/mantine", GraphSyncAction.UPSERT)


@pytest.mark.asyncio
async def test_reconcile_docs_graph_sync_avoids_rewriting_existing_pending_upserts(
    monkeypatch: pytest.MonkeyPatch,
):
    library = _library_state().model_copy(
        update={
            "graph_synced_at": None,
            "graph_sync_error": "graph metadata reconciliation pending",
        }
    )
    fake_client = AsyncMock()
    fake_client.verify_connectivity.return_value = True
    existing_job = type(
        "ExistingJob",
        (),
        {"action": GraphSyncAction.UPSERT},
    )()
    mark_pending = AsyncMock()
    enqueue = AsyncMock()

    monkeypatch.setattr("repowise.docs.graph.docs_graph_enabled", lambda: True)
    monkeypatch.setattr("repowise.docs.graph._get_client", AsyncMock(return_value=fake_client))
    monkeypatch.setattr("repowise.docs.graph._ensure_schema", AsyncMock())
    monkeypatch.setattr("repowise.docs.graph.list_libraries", AsyncMock(return_value=[library]))
    monkeypatch.setattr(
        "repowise.docs.graph._fetch_docs_graph_file_counts", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        "repowise.docs.graph._list_graph_sync_jobs",
        AsyncMock(return_value={"/mantinedev/mantine": existing_job}),
    )
    monkeypatch.setattr("repowise.docs.graph.mark_graph_sync_pending", mark_pending)
    monkeypatch.setattr("repowise.docs.graph.enqueue_graph_sync_job", enqueue)

    summary = await reconcile_docs_graph_sync()

    assert summary == {
        "eligible_libraries": 1,
        "gap_libraries": 1,
        "jobs_enqueued": 0,
    }
    mark_pending.assert_not_awaited()
    enqueue.assert_not_awaited()
