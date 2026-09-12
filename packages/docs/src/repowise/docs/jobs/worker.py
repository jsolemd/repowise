"""Job queue worker for doc-search.

Provides a background worker that:
1. Polls for pending jobs
2. Claims and processes them atomically
3. Maintains heartbeat during processing
4. Handles stale job reclamation
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from typing import TYPE_CHECKING

from repowise.docs.config import get_settings
from repowise.docs.db import (
    claim_job,
    complete_job,
    fail_job,
    heartbeat_job,
    reclaim_stale_jobs,
)
from repowise.docs.library.models import JobType

if TYPE_CHECKING:
    from repowise.docs.library.models import IndexJob

logger = logging.getLogger(__name__)

# Worker state
_worker_id: str | None = None
_shutdown_event: asyncio.Event | None = None
_worker_task: asyncio.Task | None = None
_stale_reclaim_task: asyncio.Task | None = None

# Timing configuration
POLL_INTERVAL_SEC = 5  # How often to poll for jobs
HEARTBEAT_INTERVAL_SEC = 30  # How often to heartbeat
STALE_CHECK_INTERVAL_SEC = 60  # How often to check for stale jobs


async def _dependencies_ready() -> bool:
    """Return True when the docs worker dependencies are ready for job claims."""
    try:
        from repowise.docs.server.health import get_dependency_status

        health = await get_dependency_status()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Docs dependency health unavailable: %s", exc)
        return False

    return all(health.get(key) == "ok" for key in ("qdrant", "tei", "database"))


async def _heartbeat_loop(job_id: str, stop_event: asyncio.Event) -> None:
    """
    Maintain heartbeat for a running job.

    Runs in the background while job is being processed.
    Updates heartbeat_at every HEARTBEAT_INTERVAL_SEC.
    """
    while not stop_event.is_set():
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
            if stop_event.is_set():
                break

            success = await heartbeat_job(job_id)
            if not success:
                logger.warning(f"Heartbeat failed for job {job_id} - job may have been reclaimed")
                break
            logger.debug(f"Heartbeat for job {job_id}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Heartbeat error for job {job_id}: {e}")


async def _process_job(job: IndexJob) -> None:
    """
    Process a single job.

    Runs index_library() with heartbeat maintenance.
    """
    from repowise.docs.pipeline import index_library

    logger.info(f"Processing job {job.id} for library {job.library_id} (type={job.job_type})")

    # Start heartbeat loop
    heartbeat_stop = asyncio.Event()
    heartbeat_task = asyncio.create_task(_heartbeat_loop(job.id, heartbeat_stop))

    try:
        # Determine force flag from job type
        force = job.job_type in (JobType.FORCE, JobType.FULL)

        # Run the indexing pipeline
        result = await index_library(job.library_id, force=force)

        # Mark job as completed
        completed = await complete_job(job.id)
        if not completed:
            # Job may have been reclaimed by stale reclaimer - library status already
            # updated by index_library(), so just log a warning
            logger.warning(
                f"Job {job.id} completion failed (job may have been reclaimed). "
                f"Library {job.library_id} status was updated directly."
            )
        else:
            logger.info(
                f"Job {job.id} completed: {result.get('status', 'unknown')} "
                f"(chunks={result.get('total_chunks', 0)})"
            )

    except Exception as e:
        logger.error(f"Job {job.id} failed: {e}")
        failed = await fail_job(job.id, str(e))
        if not failed:
            logger.warning(
                f"Job {job.id} fail update failed (job may have been reclaimed). "
                f"Library {job.library_id} error status was updated directly."
            )

    finally:
        # Stop heartbeat
        heartbeat_stop.set()
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task


async def _worker_loop(worker_id: str, shutdown_event: asyncio.Event) -> None:
    """
    Main worker loop.

    Polls for jobs, claims them atomically, and processes them.
    """
    logger.info(f"Worker {worker_id} starting")

    while not shutdown_event.is_set():
        try:
            if not await _dependencies_ready():
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=POLL_INTERVAL_SEC,
                    )
                    break
                except TimeoutError:
                    continue

            # Try to claim a job
            job = await claim_job(worker_id)

            if job:
                await _process_job(job)
            else:
                # No jobs available, wait before polling again
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=POLL_INTERVAL_SEC,
                    )

        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} cancelled")
            break
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            # Brief pause before retrying
            await asyncio.sleep(1)

    logger.info(f"Worker {worker_id} stopped")


async def _stale_reclaim_loop(shutdown_event: asyncio.Event) -> None:
    """
    Periodically reclaim stale jobs.

    Jobs that haven't heartbeated within stale_timeout_sec are reset to pending.
    """
    settings = get_settings()
    logger.info(
        f"Stale job reclaimer starting (timeout={settings.stale_timeout_sec}s, "
        f"check_interval={STALE_CHECK_INTERVAL_SEC}s)"
    )

    while not shutdown_event.is_set():
        try:
            # Wait for check interval or shutdown
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=STALE_CHECK_INTERVAL_SEC,
                )
                break  # Shutdown requested
            except TimeoutError:
                pass  # Normal timeout, do the check

            # Reclaim stale jobs
            reclaimed_ids = await reclaim_stale_jobs(settings.stale_timeout_sec)
            if reclaimed_ids:
                logger.warning(f"Reclaimed {len(reclaimed_ids)} stale jobs: {reclaimed_ids}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Stale reclaim error: {e}")

    logger.info("Stale job reclaimer stopped")


async def start_worker(worker_id: str | None = None) -> str:
    """
    Start the background worker.

    Args:
        worker_id: Optional worker identifier. Auto-generated if not provided.

    Returns:
        The worker ID being used.

    Raises:
        RuntimeError: If worker is already running.
    """
    global _worker_id, _shutdown_event, _worker_task, _stale_reclaim_task

    if _worker_task is not None and not _worker_task.done():
        raise RuntimeError("Worker is already running")

    _worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
    _shutdown_event = asyncio.Event()
    settings = get_settings()
    reclaimed_ids = await reclaim_stale_jobs(settings.stale_timeout_sec)
    if reclaimed_ids:
        logger.warning(
            "Reclaimed %d stale jobs on worker startup: %s", len(reclaimed_ids), reclaimed_ids
        )

    # Start worker loop
    _worker_task = asyncio.create_task(
        _worker_loop(_worker_id, _shutdown_event),
        name=f"doc-search-worker-{_worker_id}",
    )

    # Start stale job reclaimer
    _stale_reclaim_task = asyncio.create_task(
        _stale_reclaim_loop(_shutdown_event),
        name="doc-search-stale-reclaimer",
    )

    logger.info(f"Started worker {_worker_id}")
    return _worker_id


async def stop_worker(timeout: float = 10.0) -> None:
    """
    Stop the background worker gracefully.

    Args:
        timeout: Maximum seconds to wait for worker to stop.
    """
    global _worker_id, _shutdown_event, _worker_task, _stale_reclaim_task

    if _shutdown_event is None:
        return  # Not running

    logger.info(f"Stopping worker {_worker_id}...")

    # Signal shutdown
    _shutdown_event.set()

    # Wait for tasks to complete
    tasks = [t for t in [_worker_task, _stale_reclaim_task] if t is not None]
    if tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(f"Worker shutdown timed out after {timeout}s, cancelling tasks")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # Reset state
    _worker_id = None
    _shutdown_event = None
    _worker_task = None
    _stale_reclaim_task = None

    logger.info("Worker stopped")


def is_worker_running() -> bool:
    """Check if the worker is currently running."""
    return _worker_task is not None and not _worker_task.done()


def get_worker_id() -> str | None:
    """Get the current worker ID, or None if not running."""
    return _worker_id
