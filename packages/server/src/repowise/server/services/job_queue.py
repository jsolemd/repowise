"""Shared background-job launch and index-only queueing."""

from __future__ import annotations

import asyncio
import logging
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
    """Outcome of one deduplicated index-only queue request."""

    status: Literal["accepted", "already_running"]
    job_id: str
    job_state: str
    force: bool
    existing: bool


# One lock per repository/session-factory pair closes the check-then-insert
# race for every in-process caller that uses :func:`repository_job_lock`.
# The durable active-row check remains authoritative across restarts and
# pre-existing work.
_queue_locks: dict[tuple[int, str], asyncio.Lock] = {}


def repository_job_lock(session_factory: Any, repository_id: str) -> asyncio.Lock:
    """Return the process-wide launch lock for one repository database."""
    return _queue_locks.setdefault((id(session_factory), repository_id), asyncio.Lock())


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
