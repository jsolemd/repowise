"""Shared REST/MCP background-job queueing."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy import select

from repowise.core.persistence import crud
from repowise.core.persistence.models import GenerationJob
from repowise.server.services.job_queue import queue_index_only_job


async def _repo(session, tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    return await crud.upsert_repository(
        session,
        name="repo",
        local_path=str(path),
    )


def _runtime():
    return SimpleNamespace(background_tasks=set(), job_tasks={})


async def test_queue_index_only_job_persists_auditable_non_generative_mode(
    session,
    session_factory,
    tmp_path,
):
    repo = await _repo(session, tmp_path)
    await session.commit()
    executor = AsyncMock()

    result = await queue_index_only_job(
        app_state=_runtime(),
        session_factory=session_factory,
        repository_id=repo.id,
        force=True,
        executor=executor,
    )
    await asyncio.sleep(0)

    job = await session.get(GenerationJob, result.job_id)
    assert result.status == "accepted"
    assert result.existing is False
    assert job is not None
    assert json.loads(job.config_json) == {
        "mode": "index_only",
        "force": True,
        "generate_docs": False,
    }
    executor.assert_awaited_once()


async def test_queue_index_only_job_reuses_active_job_without_launching(
    session,
    session_factory,
    tmp_path,
):
    repo = await _repo(session, tmp_path)
    active = await crud.upsert_generation_job(
        session,
        repository_id=repo.id,
        status="running",
        config={"mode": "sync"},
    )
    await session.commit()
    executor = AsyncMock()

    result = await queue_index_only_job(
        app_state=_runtime(),
        session_factory=session_factory,
        repository_id=repo.id,
        force=False,
        executor=executor,
    )

    rows = list((await session.execute(select(GenerationJob))).scalars().all())
    assert result.status == "already_running"
    assert result.job_id == active.id
    assert result.job_state == "running"
    assert result.existing is True
    assert len(rows) == 1
    executor.assert_not_awaited()


async def test_concurrent_index_only_requests_create_one_job(
    session,
    session_factory,
    tmp_path,
):
    repo = await _repo(session, tmp_path)
    await session.commit()
    release = asyncio.Event()

    async def executor(*_args, **_kwargs):
        await release.wait()

    runtime = _runtime()
    first, second = await asyncio.gather(
        queue_index_only_job(
            app_state=runtime,
            session_factory=session_factory,
            repository_id=repo.id,
            force=True,
            executor=executor,
        ),
        queue_index_only_job(
            app_state=runtime,
            session_factory=session_factory,
            repository_id=repo.id,
            force=True,
            executor=executor,
        ),
    )
    release.set()
    await asyncio.sleep(0)

    rows = list((await session.execute(select(GenerationJob))).scalars().all())
    assert {first.status, second.status} == {"accepted", "already_running"}
    assert first.job_id == second.job_id
    assert len(rows) == 1
