"""Tests for docs runtime bootstrap orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import repowise.docs.jobs.single_writer as single_writer
import repowise.docs.runtime as docs_runtime


@pytest.fixture(autouse=True)
def reset_docs_runtime_state() -> None:
    def _reset() -> None:
        docs_runtime._STATE.initialized = False
        docs_runtime._STATE.background_started = False
        docs_runtime._STATE.worker_id = None
        docs_runtime._STATE.bootstrap_summary = None
        docs_runtime._STATE.lock = None
        docs_runtime._STATE.background_declined_reason = None
        single_writer._lock_connection = None

    _reset()
    yield
    _reset()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ensure_docs_runtime_bootstraps_snapshot_sources_before_background_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_summary = {"configured": 4, "jobs_queued": 0}
    snapshot_summary = {"attempted": 1, "published": 1, "failed": 0}

    # This process claims the writer role; the lock itself is covered in
    # tests/doc_search/test_single_writer.py.
    monkeypatch.setattr(docs_runtime, "acquire_background_lock", AsyncMock(return_value=True))
    monkeypatch.setattr(docs_runtime, "release_background_lock", AsyncMock())
    monkeypatch.setattr(docs_runtime, "init_pool", AsyncMock())
    monkeypatch.setattr(docs_runtime, "initialize_health_checker", AsyncMock())
    monkeypatch.setattr(
        docs_runtime,
        "sync_library_registry",
        AsyncMock(return_value=registry_summary),
    )
    monkeypatch.setattr(
        docs_runtime,
        "list_libraries",
        AsyncMock(return_value=["library-state"]),
    )
    monkeypatch.setattr(
        docs_runtime,
        "bootstrap_missing_snapshot_libraries",
        AsyncMock(return_value=snapshot_summary),
    )
    monkeypatch.setattr(docs_runtime, "start_worker", AsyncMock(return_value="worker-1"))
    monkeypatch.setattr(docs_runtime, "start_scheduler", AsyncMock())
    monkeypatch.setattr(docs_runtime, "start_recovery_task", AsyncMock())
    await docs_runtime.ensure_docs_runtime(start_background=True)

    docs_runtime.sync_library_registry.assert_awaited_once_with(queue_missing_jobs=True)
    docs_runtime.list_libraries.assert_awaited_once()
    docs_runtime.bootstrap_missing_snapshot_libraries.assert_awaited_once_with(["library-state"])
    assert docs_runtime._STATE.bootstrap_summary == {
        "registry": registry_summary,
        "snapshot_bootstrap": snapshot_summary,
    }
    assert docs_runtime._STATE.worker_id == "worker-1"


@pytest.mark.parametrize(
    "stage",
    [
        "sync_library_registry",
        "bootstrap_missing_snapshot_libraries",
        "start_worker",
        "start_scheduler",
        "start_recovery_task",
    ],
)
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
@pytest.mark.asyncio
async def test_failed_writer_start_releases_ownership(monkeypatch, stage, failure):
    names = [
        "init_pool",
        "initialize_health_checker",
        "acquire_background_lock",
        "release_background_lock",
        "sync_library_registry",
        "list_libraries",
        "bootstrap_missing_snapshot_libraries",
        "start_worker",
        "start_scheduler",
        "start_recovery_task",
        "stop_worker",
        "stop_scheduler",
        "stop_recovery_task",
    ]
    for name in names:
        monkeypatch.setattr(docs_runtime, name, AsyncMock(return_value=True))
    getattr(docs_runtime, stage).side_effect = failure("startup interrupted")

    with pytest.raises(failure):
        await docs_runtime.ensure_docs_runtime(start_background=True)

    docs_runtime.release_background_lock.assert_awaited_once()
    assert not docs_runtime._STATE.background_started
    assert docs_runtime._STATE.worker_id is None
    assert docs_runtime._STATE.bootstrap_summary is None
    if stage in {"start_scheduler", "start_recovery_task"}:
        docs_runtime.stop_worker.assert_awaited_once()
    if stage == "start_recovery_task":
        docs_runtime.stop_scheduler.assert_awaited_once()
