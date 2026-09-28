"""Tests for APScheduler background jobs."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from repowise.core.persistence import crud
from repowise.core.persistence.database import get_session
from repowise.core.persistence.models import GenerationJob
from repowise.server.scheduler import _inspect_repository, setup_scheduler


async def test_polling_fallback_launches_persisted_job(session_factory, tmp_path) -> None:
    """A diverged repository should launch the sync job that was persisted."""
    repo_path = tmp_path / "repo"
    state_path = repo_path / ".repowise" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps({"last_sync_commit": "old-sha"}), encoding="utf-8")

    async with get_session(session_factory) as session:
        repo = await crud.upsert_repository(
            session,
            name="test-repo",
            local_path=str(repo_path),
        )
        repo_id = repo.id

    app_state = SimpleNamespace(background_tasks=set())
    scheduler = setup_scheduler(session_factory, app_state=app_state)
    polling_job = next(job for job in scheduler.get_jobs() if job.id == "polling_fallback")
    head_result = SimpleNamespace(returncode=0, stdout="new-sha\n")
    execute_job = AsyncMock()

    with (
        patch("subprocess.run", return_value=head_result),
        patch("repowise.server.job_executor.execute_job", execute_job),
    ):
        await polling_job.func()
        if app_state.background_tasks:
            await asyncio.gather(*app_state.background_tasks)

    execute_job.assert_awaited_once()
    job_id, launched_state = execute_job.await_args.args
    assert launched_state is app_state
    assert execute_job.await_args.kwargs["session_factory_override"] is session_factory

    async with get_session(session_factory) as session:
        result = await session.execute(
            select(GenerationJob).where(GenerationJob.repository_id == repo_id)
        )
        persisted_job = result.scalar_one()

    assert job_id == persisted_job.id
    assert persisted_job.status == "pending"
    assert json.loads(persisted_job.config_json) == {
        "mode": "sync",
        "trigger": "polling_fallback",
        "before": "old-sha",
        "after": "new-sha",
    }


async def test_polling_fallback_offloads_repository_inspection(session_factory, tmp_path) -> None:
    """The scheduler must not inspect state or run Git on the event loop."""
    async with get_session(session_factory) as session:
        await crud.upsert_repository(
            session,
            name="test-repo",
            local_path=str(tmp_path),
        )

    scheduler = setup_scheduler(session_factory)
    polling_job = next(job for job in scheduler.get_jobs() if job.id == "polling_fallback")
    to_thread = AsyncMock(return_value=None)

    with patch("repowise.server.scheduler.asyncio.to_thread", to_thread):
        await polling_job.func()

    to_thread.assert_awaited_once_with(_inspect_repository, str(tmp_path))


async def test_scheduled_job_cancelled_before_start_gets_terminal_status(
    session_factory, tmp_path,
):
    from repowise.server.services.job_queue import shutdown_job_tasks

    async with get_session(session_factory) as session:
        await crud.upsert_repository(session, name="repo", local_path=str(tmp_path))
    app_state = SimpleNamespace(background_tasks=set())
    scheduler = setup_scheduler(session_factory, app_state=app_state)
    polling_job = next(job for job in scheduler.get_jobs() if job.id == "polling_fallback")
    executor = AsyncMock()
    shutdown = []

    def stop_before_start(*args, **kwargs):
        # The shutdown task is runnable before the executor gets its first
        # turn, but launch_job_task registers the executor before yielding.
        shutdown.append(asyncio.create_task(shutdown_job_tasks(app_state)))
        return executor(*args, **kwargs)

    with (
        patch("repowise.server.scheduler._inspect_repository", return_value=("old", "new")),
        patch("repowise.server.job_executor.execute_job", stop_before_start),
    ):
        await asyncio.create_task(polling_job.func())
        await asyncio.gather(*shutdown)

    executor.assert_not_awaited()
    assert app_state.background_tasks == set()
    assert app_state.job_tasks == {}
    async with get_session(session_factory) as session:
        job = (await session.scalars(select(GenerationJob))).one()
        assert job.status == "cancelled"
        assert job.error_message == "Server shutdown"


@pytest.mark.parametrize("callback_id", ["staleness_check", "polling_fallback"])
async def test_scheduler_callback_cannot_enter_sql_after_shutdown(session_factory, callback_id):
    app_state = SimpleNamespace(background_tasks=set(), job_shutdown_requested=True)
    scheduler = setup_scheduler(session_factory, app_state=app_state)
    callback = next(job for job in scheduler.get_jobs() if job.id == callback_id)
    with patch("repowise.core.persistence.database.get_session") as open_session:
        await callback.func()
    open_session.assert_not_called()
    assert app_state.background_tasks == set()
