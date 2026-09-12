"""Tests for docs registry bootstrap and sync behavior."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from repowise.docs.bootstrap import find_libraries_yaml, sync_library_registry
from repowise.docs.library.models import (
    IndexJob,
    JobEnqueueDisposition,
    JobEnqueueResult,
    JobType,
    LibrariesConfig,
    LibraryConfig,
    LibrarySourceType,
    LibraryState,
    LibraryStatus,
)


def _config(repo: str, *, priority: int = 5) -> LibraryConfig:
    return LibraryConfig(
        repo=repo,
        name=repo.split("/")[-1],
        description=f"Docs for {repo}",
        priority=priority,
    )


def _state(
    repo: str,
    *,
    status: LibraryStatus = LibraryStatus.PENDING,
    priority: int = 5,
    **overrides,
) -> LibraryState:
    data = {
        "library_id": f"/{repo}",
        "repo": repo,
        "name": repo.split("/")[-1],
        "description": f"Docs for {repo}",
        "docs_path": "",
        "branch": "main",
        "include_patterns": [],
        "exclude_patterns": [],
        "priority": priority,
        "status": status,
        "current_sha": None,
        "indexed_at": None,
        "error_message": None,
        "chunk_count": 0,
        "file_count": 0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    data.update(overrides)
    return LibraryState(**data)


def test_find_libraries_yaml_prefers_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    override = tmp_path / "libraries.override.yaml"
    override.write_text("libraries: []\n", encoding="utf-8")

    monkeypatch.setenv("DOC_SEARCH_LIBRARIES_YAML", str(override))
    assert find_libraries_yaml() == override


@pytest.mark.asyncio
async def test_sync_library_registry_adopts_existing_chunks_and_queues_missing_jobs(
    monkeypatch: pytest.MonkeyPatch,
):
    config = LibrariesConfig(
        libraries=[_config("mantinedev/mantine", priority=4), _config("framer/motion", priority=6)]
    )

    upserted_states = {
        "/mantinedev/mantine": _state("mantinedev/mantine", priority=4),
        "/framer/motion": _state("framer/motion", priority=6),
    }

    monkeypatch.setattr(
        "repowise.docs.bootstrap.load_libraries_config", lambda yaml_path=None: config
    )
    monkeypatch.setattr("repowise.docs.bootstrap.db.list_libraries", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "repowise.docs.bootstrap.db.upsert_library",
        AsyncMock(side_effect=lambda lib_config: upserted_states[lib_config.library_id]),
    )
    monkeypatch.setattr(
        "repowise.docs.bootstrap.get_chunk_count",
        AsyncMock(side_effect=lambda library_id: 12 if library_id == "/mantinedev/mantine" else 0),
    )
    mock_update = AsyncMock()
    mock_enqueue = AsyncMock(
        side_effect=[
            JobEnqueueResult(
                disposition=JobEnqueueDisposition.QUEUED,
                job=IndexJob(
                    id="job-mantine",
                    library_id="/mantinedev/mantine",
                    job_type=JobType.INCREMENTAL,
                    priority=6,
                ),
            ),
            JobEnqueueResult(
                disposition=JobEnqueueDisposition.QUEUED,
                job=IndexJob(
                    id="job-motion",
                    library_id="/framer/motion",
                    job_type=JobType.FULL,
                    priority=6,
                ),
            ),
        ]
    )
    monkeypatch.setattr("repowise.docs.bootstrap.db.update_library_status", mock_update)
    monkeypatch.setattr("repowise.docs.bootstrap.db.enqueue_job", mock_enqueue)

    summary = await sync_library_registry(queue_missing_jobs=True)

    assert summary == {
        "configured": 2,
        "upserted": 2,
        "new_libraries": 2,
        "adopted_from_qdrant": 1,
        "jobs_queued": 2,
        "metadata_reconciliation_requested": 1,
        "legacy_aliases_migrated": 0,
        "legacy_chunks_relabelled": 0,
        "legacy_chunks_deleted": 0,
        "queue_missing_jobs": True,
        "registry_previously_empty": True,
    }
    mock_update.assert_awaited_once()
    assert mock_update.await_args.kwargs["next_freshness_check_at"] is not None
    assert mock_enqueue.await_args_list[0].args == ("/mantinedev/mantine", JobType.INCREMENTAL)
    assert mock_enqueue.await_args_list[0].kwargs == {"priority": 6}
    assert mock_enqueue.await_args_list[1].args == ("/framer/motion", JobType.FULL)
    assert mock_enqueue.await_args_list[1].kwargs == {"priority": 6}


@pytest.mark.asyncio
async def test_sync_library_registry_leaves_existing_libraries_alone(
    monkeypatch: pytest.MonkeyPatch,
):
    config = LibrariesConfig(libraries=[_config("mantinedev/mantine", priority=4)])
    existing_state = _state(
        "mantinedev/mantine",
        status=LibraryStatus.READY,
        priority=4,
        current_sha="abc123def456",
        indexed_at=datetime.utcnow(),
        chunk_count=12,
        file_count=3,
    )

    monkeypatch.setattr(
        "repowise.docs.bootstrap.load_libraries_config", lambda yaml_path=None: config
    )
    monkeypatch.setattr(
        "repowise.docs.bootstrap.db.list_libraries", AsyncMock(return_value=[existing_state])
    )
    monkeypatch.setattr(
        "repowise.docs.bootstrap.db.upsert_library", AsyncMock(return_value=existing_state)
    )
    mock_get_chunk_count = AsyncMock()
    mock_update = AsyncMock()
    mock_enqueue = AsyncMock()
    monkeypatch.setattr("repowise.docs.bootstrap.get_chunk_count", mock_get_chunk_count)
    monkeypatch.setattr("repowise.docs.bootstrap.db.update_library_status", mock_update)
    monkeypatch.setattr("repowise.docs.bootstrap.db.enqueue_job", mock_enqueue)

    summary = await sync_library_registry(queue_missing_jobs=True)

    assert summary == {
        "configured": 1,
        "upserted": 1,
        "new_libraries": 0,
        "adopted_from_qdrant": 0,
        "jobs_queued": 0,
        "metadata_reconciliation_requested": 0,
        "legacy_aliases_migrated": 0,
        "legacy_chunks_relabelled": 0,
        "legacy_chunks_deleted": 0,
        "queue_missing_jobs": True,
        "registry_previously_empty": False,
    }
    mock_get_chunk_count.assert_not_awaited()
    mock_update.assert_not_awaited()
    mock_enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_library_registry_reconciles_existing_ready_library_metadata_gap(
    monkeypatch: pytest.MonkeyPatch,
):
    config = LibrariesConfig(
        libraries=[_config("opensearch-project/documentation-website", priority=4)]
    )
    existing_state = _state(
        "opensearch-project/documentation-website",
        status=LibraryStatus.READY,
        priority=4,
        current_sha="b706b80cc6765c29d262f230c714bfdf86e817e2",
        indexed_at=datetime.utcnow(),
        chunk_count=0,
        file_count=0,
    )

    mock_enqueue = AsyncMock(
        return_value=JobEnqueueResult(
            disposition=JobEnqueueDisposition.QUEUED,
            job=IndexJob(
                id="job-opensearch-reconcile",
                library_id=existing_state.library_id,
                job_type=JobType.INCREMENTAL,
                priority=6,
            ),
        )
    )

    monkeypatch.setattr(
        "repowise.docs.bootstrap.load_libraries_config", lambda yaml_path=None: config
    )
    monkeypatch.setattr(
        "repowise.docs.bootstrap.db.list_libraries", AsyncMock(return_value=[existing_state])
    )
    monkeypatch.setattr(
        "repowise.docs.bootstrap.db.upsert_library", AsyncMock(return_value=existing_state)
    )
    monkeypatch.setattr("repowise.docs.bootstrap.get_chunk_count", AsyncMock())
    monkeypatch.setattr("repowise.docs.bootstrap.db.update_library_status", AsyncMock())
    monkeypatch.setattr("repowise.docs.bootstrap.db.enqueue_job", mock_enqueue)

    summary = await sync_library_registry(queue_missing_jobs=True)

    assert summary == {
        "configured": 1,
        "upserted": 1,
        "new_libraries": 0,
        "adopted_from_qdrant": 0,
        "jobs_queued": 1,
        "metadata_reconciliation_requested": 1,
        "legacy_aliases_migrated": 0,
        "legacy_chunks_relabelled": 0,
        "legacy_chunks_deleted": 0,
        "queue_missing_jobs": True,
        "registry_previously_empty": False,
    }
    mock_enqueue.assert_awaited_once_with(
        existing_state.library_id,
        JobType.INCREMENTAL,
        priority=6,
    )


@pytest.mark.asyncio
async def test_sync_library_registry_migrates_legacy_library_ids(monkeypatch: pytest.MonkeyPatch):
    config = LibrariesConfig(
        libraries=[
            LibraryConfig(
                source_type=LibrarySourceType.SNAPSHOT,
                library_id="/codeatlas/cosmograph",
                legacy_ids=["/jsolemd/cosmograph-docs"],
                name="Cosmograph",
                description="Cosmograph docs",
            )
        ]
    )
    legacy_state = _state("jsolemd/cosmograph-docs", status=LibraryStatus.READY)
    migrated_state = LibraryState(
        source_type=LibrarySourceType.SNAPSHOT,
        library_id="/codeatlas/cosmograph",
        repo="",
        name="Cosmograph",
        description="Cosmograph docs",
        docs_path="",
        branch="main",
        include_patterns=[],
        exclude_patterns=[],
        priority=5,
        status=LibraryStatus.READY,
        current_sha="legacy-tree-sha",
        indexed_at=datetime.utcnow(),
        error_message=None,
        chunk_count=72,
        file_count=24,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    mock_upsert = AsyncMock(return_value=migrated_state)
    mock_migrate_alias = AsyncMock(return_value=migrated_state)
    mock_get_chunk_count = AsyncMock(
        side_effect=lambda library_id: 72 if library_id == "/jsolemd/cosmograph-docs" else 0
    )
    mock_relabel_chunks = AsyncMock(return_value=72)
    mock_delete_legacy_chunks = AsyncMock(return_value=0)
    mock_update = AsyncMock()
    mock_enqueue = AsyncMock()

    monkeypatch.setattr(
        "repowise.docs.bootstrap.load_libraries_config", lambda yaml_path=None: config
    )
    monkeypatch.setattr(
        "repowise.docs.bootstrap.db.list_libraries", AsyncMock(return_value=[legacy_state])
    )
    monkeypatch.setattr("repowise.docs.bootstrap.db.upsert_library", mock_upsert)
    monkeypatch.setattr("repowise.docs.bootstrap.db.migrate_library_alias", mock_migrate_alias)
    monkeypatch.setattr("repowise.docs.bootstrap.get_chunk_count", mock_get_chunk_count)
    monkeypatch.setattr(
        "repowise.docs.bootstrap.db.get_snapshot_state", AsyncMock(return_value=None)
    )
    monkeypatch.setattr("repowise.docs.bootstrap.relabel_library_chunks", mock_relabel_chunks)
    monkeypatch.setattr("repowise.docs.bootstrap.delete_by_library", mock_delete_legacy_chunks)
    monkeypatch.setattr("repowise.docs.bootstrap.db.update_library_status", mock_update)
    monkeypatch.setattr("repowise.docs.bootstrap.db.enqueue_job", mock_enqueue)

    summary = await sync_library_registry(queue_missing_jobs=True)

    assert summary == {
        "configured": 1,
        "upserted": 1,
        "new_libraries": 0,
        "adopted_from_qdrant": 0,
        "jobs_queued": 0,
        "metadata_reconciliation_requested": 0,
        "legacy_aliases_migrated": 1,
        "legacy_chunks_relabelled": 72,
        "legacy_chunks_deleted": 0,
        "queue_missing_jobs": True,
        "registry_previously_empty": False,
    }
    mock_upsert.assert_awaited_once()
    mock_migrate_alias.assert_awaited_once()
    mock_relabel_chunks.assert_awaited_once_with(
        "/jsolemd/cosmograph-docs", "/codeatlas/cosmograph"
    )
    mock_update.assert_not_awaited()
    mock_enqueue.assert_not_awaited()
