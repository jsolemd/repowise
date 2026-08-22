"""Shared background-job launch and index-only queueing."""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import select

from repowise.core.persistence import crud
from repowise.core.persistence.database import get_session
from repowise.core.persistence.models import GenerationJob

logger = logging.getLogger(__name__)

JobExecutor = Callable[..., Coroutine[Any, Any, None]]


@dataclass(frozen=True, slots=True)
class IndexJobQueueResult:
    """Outcome of one deduplicated index-only queue request.

    ``force`` echoes the caller's request flag and nothing else. It selects no
    rebuild mode: an accepted forced request queues an ordinary ``index_only``
    job, identical in every field to the one a ``force=False`` request queues.
    The single behaviour it buys is bypassing the caller-side "already current"
    no-op, which is why it is persisted under the name ``bypass_current_noop``
    rather than ``force`` — a key named ``force`` invites a future reader to
    treat it as a rebuild switch it has never been.
    """

    status: Literal["accepted", "already_running"]
    job_id: str
    job_state: str
    force: bool
    existing: bool


@dataclass(slots=True)
class _FactoryLocks:
    """One session factory's repository locks, plus proof of whose they are.

    ``ref`` is a weak reference to the owning factory; ``owner`` holds it
    strongly instead for the rare factory that cannot be weak-referenced. Both
    exist to answer one question — is the object at this address still the
    factory these locks were made for?
    """

    locks: dict[str, asyncio.Lock]
    ref: weakref.ref | None = None
    owner: Any | None = None

    def owns(self, session_factory: Any) -> bool:
        """Identity, deliberately not equality.

        A session factory is free to define ``__eq__``/``__hash__``, and two
        equal-but-distinct factories address different databases, so equality
        would merge locks that must stay apart. ``is`` cannot.
        """
        if self.ref is not None:
            return self.ref() is session_factory
        return self.owner is session_factory


# One lock per (session factory, repository) pair. Every in-process caller that
# creates a job for a repository takes it around its own check-then-insert: the
# REST routes in ``routers.repos`` via ``_repository_job_lock``, and MCP
# index-only queueing via :func:`queue_index_only_job`. Without a lock those
# callers share, the durable ``pending|running`` guard is only advisory —
# under SQLite WAL snapshot isolation two sessions both see no active job and
# both insert. That durable check remains authoritative across restarts and
# for work this process did not start.
#
# The address is still the lookup key, but it is no longer trusted as the
# identity: ``_FactoryLocks.owns`` re-checks that the caller's factory is the
# one an entry was made for, and a weakref callback removes the entry when that
# factory is collected. Keying on ``id()`` alone was wrong in both directions —
# CPython reuses an address once its owner is collected, so a workspace repo
# whose factory is deregistered and re-registered could be handed a dead pair's
# lock, and nothing ever removed an entry, so the table grew for the life of
# the process with one entry per (factory, repo) pair ever seen. The address is
# kept as the key rather than the object because a session factory need be
# neither hashable nor weak-referenceable, and a lock table is the wrong place
# to impose either requirement.
_queue_locks: dict[int, _FactoryLocks] = {}


def _drop_factory_locks(key: int, entry: _FactoryLocks) -> None:
    """Evict *entry* only if it is still the table's entry for *key*."""
    if _queue_locks.get(key) is entry:
        del _queue_locks[key]


def repository_job_lock(session_factory: Any, repository_id: str) -> asyncio.Lock:
    """Return the process-wide launch lock for one repository database.

    Callers that create a job must hold this across their active-job SELECT,
    their INSERT, and the COMMIT that publishes it. Resolving it from the same
    session factory the caller will query is what makes REST and MCP share one
    lock rather than two that never see each other.
    """
    key = id(session_factory)
    entry = _queue_locks.get(key)
    if entry is None or not entry.owns(session_factory):
        entry = _FactoryLocks(locks={})
        try:
            entry.ref = weakref.ref(
                session_factory,
                lambda _dead, key=key, entry=entry: _drop_factory_locks(key, entry),
            )
        except TypeError:
            # Not weak-referenceable. Hold it strongly rather than fail the job
            # being queued: an object that stays alive cannot have its address
            # recycled, so the key stays correct. Bounded by the number of such
            # factories, not by (factory, repo) pairs.
            entry.owner = session_factory
        _queue_locks[key] = entry
    return entry.locks.setdefault(repository_id, asyncio.Lock())


def launch_job_task(
    *,
    app_state: Any,
    job_id: str,
    session_factory: Any,
    executor: JobExecutor,
) -> None:
    """Launch a background job and keep its task strongly referenced.

    This is the transport-neutral body formerly owned by the repository REST
    router. The router still resolves its request-scoped session factory and
    passes its patchable ``execute_job`` binding, so existing REST behavior and
    tests remain unchanged; standalone MCP supplies the same runtime fields.
    """

    async def _mark_terminal(status: str, reason: str) -> None:
        try:
            from repowise.core.persistence.crud import update_job_status

            async with get_session(session_factory) as session:
                await update_job_status(
                    session,
                    job_id,
                    status,
                    error_message=reason[:500],
                )
        except Exception:
            logger.exception("fallback_job_failure_record_failed", extra={"job_id": job_id})

    bg_tasks = getattr(app_state, "background_tasks", None)
    if bg_tasks is None:
        bg_tasks = set()
        app_state.background_tasks = bg_tasks

    try:
        task: asyncio.Task[None] = asyncio.create_task(
            executor(job_id, app_state, session_factory_override=session_factory),
            name=f"job-{job_id}",
        )
    except Exception as exc:
        logger.exception("create_task_failed", extra={"job_id": job_id})
        task = asyncio.create_task(
            _mark_terminal("failed", f"Failed to launch background task: {exc}")
        )
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)
        return

    bg_tasks.add(task)

    def _fire_and_track(coro: Coroutine[Any, Any, None]) -> None:
        short_task: asyncio.Task[None] = asyncio.create_task(coro)
        bg_tasks.add(short_task)
        short_task.add_done_callback(bg_tasks.discard)

    job_tasks = getattr(app_state, "job_tasks", None)
    if job_tasks is None:
        job_tasks = {}
        app_state.job_tasks = job_tasks
    job_tasks[job_id] = task

    def _on_done(done: asyncio.Task[Any]) -> None:
        bg_tasks.discard(done)
        job_tasks.pop(job_id, None)
        if done.cancelled():
            _fire_and_track(_mark_terminal("cancelled", "Cancelled by user"))
            return
        exc = done.exception()
        if exc is not None:
            logger.error("background_job_failed", exc_info=exc)
            _fire_and_track(_mark_terminal("failed", f"Background task crashed: {exc}"))

    task.add_done_callback(_on_done)


async def queue_index_only_job(
    *,
    app_state: Any,
    session_factory: Any,
    repository_id: str,
    force: bool,
    executor: JobExecutor,
) -> IndexJobQueueResult:
    """Deduplicate, persist, and launch one non-generative index-only job.

    A direct source-store reconcile is intentionally not used when HEAD may be
    ahead: it would combine stale persisted symbol bounds with current file
    bytes. The established ``index_only`` executor refreshes upstream parsing
    and SQL state before publishing the derived source stores.

    ``force`` does not reach the executor and is not meant to. The job this
    queues under ``force=True`` is the same incremental ``index_only`` run as
    under ``force=False``; the flag's entire effect is upstream, letting a
    caller queue that run even when it has already judged the index current.
    It is persisted as ``bypass_current_noop`` so the status tool can report
    an accurate reason a job exists, and so no future reader can mistake a
    recorded request flag for an executor rebuild mode.
    """

    lock = repository_job_lock(session_factory, repository_id)
    async with lock:
        async with get_session(session_factory) as session:
            repository = await crud.get_repository(session, repository_id)
            if repository is None:
                raise LookupError(f"Repository not found: {repository_id}")

            active = (
                (
                    await session.execute(
                        select(GenerationJob)
                        .where(
                            GenerationJob.repository_id == repository_id,
                            GenerationJob.status.in_(["pending", "running"]),
                        )
                        .order_by(GenerationJob.created_at.desc())
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            if active is not None:
                return IndexJobQueueResult(
                    status="already_running",
                    job_id=active.id,
                    job_state=active.status,
                    force=force,
                    existing=True,
                )

            job = await crud.upsert_generation_job(
                session,
                repository_id=repository_id,
                status="pending",
                config={
                    "mode": "index_only",
                    # This records request semantics, not an executor rebuild
                    # mode: force only bypasses the status-tool current no-op.
                    "bypass_current_noop": force,
                    "generate_docs": False,
                },
            )
            # The executor opens a separate session, so flush is insufficient
            # under SQLite WAL isolation.
            await session.commit()
            job_id = job.id

        launch_job_task(
            app_state=app_state,
            job_id=job_id,
            session_factory=session_factory,
            executor=executor,
        )
        return IndexJobQueueResult(
            status="accepted",
            job_id=job_id,
            job_state="pending",
            force=force,
            existing=False,
        )


__all__ = [
    "IndexJobQueueResult",
    "launch_job_task",
    "queue_index_only_job",
    "repository_job_lock",
]
