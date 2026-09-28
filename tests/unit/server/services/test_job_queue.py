"""Shared REST/MCP background-job queueing."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from repowise.core.persistence import crud
from repowise.core.persistence.models import GenerationJob
from repowise.server.services import job_queue as job_queue_module
from repowise.server.services.job_queue import queue_index_only_job, repository_job_lock


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
        "bypass_current_noop": True,
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


async def test_force_changes_only_the_recorded_request_flag(
    session,
    session_factory,
    tmp_path,
):
    """A forced index-only job is the same work an unforced one queues.

    Pins the claim the payload and docs make: ``force`` is a request-side
    no-op bypass, not a rebuild mode. If someone later teaches the executor a
    real forced mode, the two configs stop being interchangeable and this
    fails — which is the point at which the cost/scope wording has to change
    with it.
    """
    repo = await _repo(session, tmp_path)
    await session.commit()

    forced = await queue_index_only_job(
        app_state=_runtime(),
        session_factory=session_factory,
        repository_id=repo.id,
        force=True,
        executor=AsyncMock(),
    )
    await asyncio.sleep(0)
    forced_job = await session.get(GenerationJob, forced.job_id)
    assert forced_job is not None
    forced_config = json.loads(forced_job.config_json)

    # Retire it so the next request is not deduplicated into this one.
    forced_job.status = "completed"
    await session.commit()

    plain = await queue_index_only_job(
        app_state=_runtime(),
        session_factory=session_factory,
        repository_id=repo.id,
        force=False,
        executor=AsyncMock(),
    )
    await asyncio.sleep(0)
    plain_job = await session.get(GenerationJob, plain.job_id)
    assert plain_job is not None
    plain_config = json.loads(plain_job.config_json)

    assert forced.force is True
    assert plain.force is False
    assert forced_config["bypass_current_noop"] is True
    assert plain_config["bypass_current_noop"] is False
    # Nothing else about the queued work differs.
    assert {k: v for k, v in forced_config.items() if k != "bypass_current_noop"} == {
        k: v for k, v in plain_config.items() if k != "bypass_current_noop"
    }
    # And no key named "force" is persisted for a later reader to mistake for
    # an executor rebuild switch.
    assert "force" not in forced_config
    assert "force" not in plain_config


class _Factory:
    """An ordinary session-factory stand-in: hashable and weak-referenceable."""

    def __init__(self, label: str) -> None:
        self.label = label


class _SlottedFactory:
    """A session-factory stand-in that cannot be weak-referenced."""

    __slots__ = ("label",)

    def __init__(self, label: str) -> None:
        self.label = label


class _EqualEverythingFactory:
    """A stand-in that claims equality with every other factory."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EqualEverythingFactory)

    def __hash__(self) -> int:
        return 0


def test_repository_job_lock_is_one_object_per_factory_and_repository():
    factory_a = _Factory("a")
    factory_b = _Factory("b")

    lock = repository_job_lock(factory_a, "repo-1")

    assert repository_job_lock(factory_a, "repo-1") is lock
    assert repository_job_lock(factory_a, "repo-2") is not lock
    assert repository_job_lock(factory_b, "repo-1") is not lock


def test_repository_job_lock_table_releases_a_collected_session_factory():
    """The lock table must not outlive the database it locks.

    Keyed on ``id(session_factory)`` alone this could not hold: the entry
    survived its factory and the table only ever grew, one entry per
    (factory, repo) pair ever seen, for the life of the process.
    """
    gc.collect()
    factory = _Factory("throwaway")
    key = id(factory)
    repository_job_lock(factory, "repo-1")
    assert key in job_queue_module._queue_locks

    ref = weakref.ref(factory)
    del factory
    gc.collect()

    assert ref() is None
    assert key not in job_queue_module._queue_locks


def test_repository_job_lock_refuses_a_stale_entry_at_a_reused_address():
    """CPython hands a collected object's address to the next allocation.

    Seeding the table at a live factory's address reproduces exactly that: an
    ``id()``-keyed table would hand this factory the dead pair's lock, so a
    REST caller and an MCP caller on genuinely different databases could
    serialise against each other — or, worse, a re-registered repo could
    inherit a lock some other code path still believes it holds.
    """
    factory = _Factory("live")
    stale = job_queue_module._FactoryLocks(locks={"repo-1": asyncio.Lock()})
    job_queue_module._queue_locks[id(factory)] = stale
    try:
        lock = repository_job_lock(factory, "repo-1")

        assert lock is not stale.locks["repo-1"]
        assert repository_job_lock(factory, "repo-1") is lock
    finally:
        job_queue_module._queue_locks.pop(id(factory), None)


def test_repository_job_lock_serves_a_factory_that_is_not_weak_referenceable():
    factory = _SlottedFactory("stand-in")
    with pytest.raises(TypeError):
        weakref.ref(factory)

    try:
        lock = repository_job_lock(factory, "repo-1")

        assert repository_job_lock(factory, "repo-1") is lock
        assert repository_job_lock(factory, "repo-2") is not lock
    finally:
        job_queue_module._queue_locks.pop(id(factory), None)


def test_repository_job_lock_serves_a_factory_that_is_not_hashable():
    """A ``SimpleNamespace`` stand-in is neither hashable nor weak-referenceable.

    A lock table is the wrong place to impose either requirement on a session
    factory, so this pins that the lookup asks for neither.
    """
    factory = SimpleNamespace(label="unhashable")
    with pytest.raises(TypeError):
        hash(factory)

    try:
        lock = repository_job_lock(factory, "repo-1")

        assert repository_job_lock(factory, "repo-1") is lock
        assert repository_job_lock(factory, "repo-2") is not lock
    finally:
        job_queue_module._queue_locks.pop(id(factory), None)


def test_repository_job_lock_separates_factories_that_compare_equal():
    """Two equal-but-distinct factories address different databases."""
    factory_a = _EqualEverythingFactory()
    factory_b = _EqualEverythingFactory()
    assert factory_a == factory_b

    assert repository_job_lock(factory_a, "repo-1") is not repository_job_lock(factory_b, "repo-1")


@pytest.mark.parametrize("outcome", [
    "completed", "crashed", "cancelled", "launch_failed", "construction_failed",
])
async def test_job_lease_outlives_terminal_persistence(
    session, session_factory, tmp_path, monkeypatch, outcome,
):
    repo = await _repo(session, tmp_path)
    await session.commit()
    runtime = _runtime()
    released = asyncio.Event()
    observed = []

    @asynccontextmanager
    async def lease():
        try:
            yield
        finally:
            async with session_factory() as fresh:
                job = (await fresh.scalars(select(GenerationJob))).one()
                observed.append(job.status)
            released.set()

    async def execute(job_id, *_args, **_kwargs):
        if outcome == "crashed":
            raise RuntimeError("pipeline failed")
        async with session_factory() as fresh:
            await crud.update_job_status(fresh, job_id, "completed")
            await fresh.commit()

    if outcome == "launch_failed":
        create_task = asyncio.create_task

        def fail_once(coro, **kwargs):
            if kwargs.get("name", "").startswith("job-"):
                monkeypatch.setattr(asyncio, "create_task", create_task)
                raise RuntimeError("task launch failed")
            return create_task(coro, **kwargs)

        monkeypatch.setattr(asyncio, "create_task", fail_once)
    elif outcome == "construction_failed":
        def execute(*_args, **_kwargs):
            raise RuntimeError("executor construction failed")

    queued = await queue_index_only_job(
        app_state=runtime, session_factory=session_factory,
        repository_id=repo.id, force=False, executor=execute, job_lifespan=lease(),
    )
    assert not released.is_set()
    if outcome == "cancelled":
        # Cancel before the executor's first instruction: its own finally
        # cannot run, so the launch owner's terminal callback owns the lease.
        runtime.job_tasks[queued.job_id].cancel()
    await asyncio.wait_for(released.wait(), timeout=5)
    while runtime.background_tasks:
        await asyncio.gather(*runtime.background_tasks, return_exceptions=True)
        await asyncio.sleep(0)

    assert observed == [{
        "completed": "completed", "crashed": "failed",
        "cancelled": "cancelled", "launch_failed": "failed", "construction_failed": "failed",
    }[outcome]]
    assert runtime.job_tasks == {}


@pytest.mark.parametrize("outcome", ["already_running", "queue_failed", "queue_cancelled"])
async def test_unaccepted_job_returns_its_runtime_lease(
    session, session_factory, tmp_path, monkeypatch, outcome,
):
    repo = await _repo(session, tmp_path)
    await crud.upsert_generation_job(session, repository_id=repo.id, status="running")
    await session.commit()
    returned = []

    @asynccontextmanager
    async def lease():
        try:
            yield
        finally:
            returned.append(True)

    if outcome != "already_running":
        error = RuntimeError if outcome == "queue_failed" else asyncio.CancelledError
        monkeypatch.setattr(crud, "get_repository", AsyncMock(side_effect=error("lookup failed")))

    request = queue_index_only_job(
        app_state=_runtime(), session_factory=session_factory,
        repository_id=repo.id, force=False, executor=AsyncMock(), job_lifespan=lease(),
    )
    if outcome == "already_running":
        assert (await request).status == "already_running"
    else:
        with pytest.raises(error):
            await request
    assert returned == [True]


async def test_job_completion_callback_error_is_observed(session, session_factory, tmp_path, caplog):
    repo = await _repo(session, tmp_path)
    job = await crud.upsert_generation_job(session, repository_id=repo.id, status="pending")
    await session.commit()
    runtime = _runtime()
    release = AsyncMock(side_effect=RuntimeError("close failed"))

    job_queue_module.launch_job_task(
        app_state=runtime, job_id=job.id, session_factory=session_factory,
        executor=AsyncMock(), on_finished=release,
    )
    while runtime.background_tasks:
        await asyncio.gather(*runtime.background_tasks, return_exceptions=True)
        await asyncio.sleep(0)

    release.assert_awaited_once()
    assert "job_resource_release_failed" in caplog.text
