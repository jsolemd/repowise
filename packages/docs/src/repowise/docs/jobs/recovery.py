"""Library status recovery for doc-search.

Provides a background task that detects and recovers orphaned libraries
that are stuck in INDEXING status with no active jobs.

This can happen when:
1. Worker crashes after setting library to INDEXING but before completing
2. Job is reclaimed by stale reclaimer but library status not reset
3. Database transaction fails partially (job updated, library not)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from repowise.docs.db import get_connection

logger = logging.getLogger(__name__)

# Recovery task state
_recovery_task: asyncio.Task | None = None
_shutdown_event: asyncio.Event | None = None

# Configuration
RECOVERY_CHECK_INTERVAL_SEC = 300  # Check every 5 minutes
ORPHAN_THRESHOLD_SEC = 3600  # Libraries stuck for >1 hour are considered orphaned


async def recover_orphaned_libraries(
    threshold_sec: int = ORPHAN_THRESHOLD_SEC,
) -> list[str]:
    """
    Find libraries stuck in INDEXING with no active jobs and reset them.

    A library is considered orphaned if:
    - Status is INDEXING
    - No pending or running job exists for that library
    - Library has been in INDEXING for longer than threshold_sec

    Args:
        threshold_sec: Minimum time in INDEXING status before recovery (default: 1 hour)

    Returns:
        List of recovered library IDs
    """
    async with get_connection() as conn:
        # Find orphaned libraries in a single query
        rows = await conn.fetch(
            """
            UPDATE doc_search.libraries
            SET status = 'error',
                error_message = COALESCE(
                    NULLIF(error_message, ''),
                    'Orphaned: stuck in indexing with no active job (auto-recovered at '
                    || TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS') || ')'
                ),
                updated_at = NOW()
            WHERE status = 'indexing'
              AND updated_at < NOW() - INTERVAL '1 second' * $1
              AND NOT EXISTS (
                SELECT 1 FROM doc_search.index_jobs
                WHERE index_jobs.library_id = libraries.library_id
                  AND index_jobs.status IN ('pending', 'running')
              )
            RETURNING library_id
            """,
            threshold_sec,
        )
        return [row["library_id"] for row in rows]


async def _recovery_loop(shutdown_event: asyncio.Event) -> None:
    """
    Periodically check for and recover orphaned libraries.

    Runs every RECOVERY_CHECK_INTERVAL_SEC and resets any orphaned libraries.
    """
    logger.info(
        f"Library recovery task starting (check_interval={RECOVERY_CHECK_INTERVAL_SEC}s, "
        f"orphan_threshold={ORPHAN_THRESHOLD_SEC}s)"
    )

    while not shutdown_event.is_set():
        try:
            # Wait for check interval or shutdown
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=RECOVERY_CHECK_INTERVAL_SEC,
                )
                break  # Shutdown requested
            except TimeoutError:
                pass  # Normal timeout, do the check

            # Recover orphaned libraries
            recovered_ids = await recover_orphaned_libraries()
            if recovered_ids:
                logger.warning(
                    f"Recovered {len(recovered_ids)} orphaned libraries: {recovered_ids}"
                )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Library recovery error: {e}")

    logger.info("Library recovery task stopped")


async def start_recovery_task() -> None:
    """
    Start the background recovery task.

    Raises:
        RuntimeError: If recovery task is already running.
    """
    global _recovery_task, _shutdown_event

    if _recovery_task is not None and not _recovery_task.done():
        raise RuntimeError("Recovery task is already running")

    _shutdown_event = asyncio.Event()
    _recovery_task = asyncio.create_task(
        _recovery_loop(_shutdown_event),
        name="doc-search-recovery",
    )

    logger.info("Started library recovery task")


async def stop_recovery_task(timeout: float = 5.0) -> None:
    """
    Stop the background recovery task gracefully.

    Args:
        timeout: Maximum seconds to wait for task to stop.
    """
    global _recovery_task, _shutdown_event

    if _shutdown_event is None:
        return  # Not running

    logger.info("Stopping library recovery task...")

    # Signal shutdown
    _shutdown_event.set()

    # Wait for task to complete
    if _recovery_task is not None:
        try:
            await asyncio.wait_for(_recovery_task, timeout=timeout)
        except TimeoutError:
            logger.warning(f"Recovery task shutdown timed out after {timeout}s, cancelling")
            _recovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _recovery_task

    # Reset state
    _recovery_task = None
    _shutdown_event = None

    logger.info("Library recovery task stopped")


def is_recovery_running() -> bool:
    """Check if the recovery task is currently running."""
    return _recovery_task is not None and not _recovery_task.done()
