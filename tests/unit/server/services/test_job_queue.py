"""Shared REST/MCP background-job queueing."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref
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
