"""Index-job persistence for doc-search."""

from __future__ import annotations

import logging

import asyncpg

from repowise.docs.db_runtime import get_connection
from repowise.docs.library.models import (
    IndexJob,
    JobEnqueueDisposition,
    JobEnqueueResult,
    JobStatus,
    JobType,
)

logger = logging.getLogger(__name__)

_JOB_TYPE_STRENGTH = {
    JobType.INCREMENTAL: 0,
    JobType.FULL: 1,
    JobType.FORCE: 2,
}


def _row_to_index_job(row: asyncpg.Record) -> IndexJob:
    """Convert a database row to IndexJob."""
    return IndexJob(
        id=str(row["id"]),
        library_id=row["library_id"],
        job_type=JobType(row["job_type"]),
        priority=row["priority"],
        status=JobStatus(row["status"]),
        worker_id=row["worker_id"],
        claimed_at=row["claimed_at"],
        heartbeat_at=row["heartbeat_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        error_message=row["error_message"],
        files_processed=row["files_processed"],
        files_total=row["files_total"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def enqueue_job(
    library_id: str,
    job_type: JobType = JobType.INCREMENTAL,
    priority: int = 5,
) -> JobEnqueueResult:
    """
    Enqueue a job for a library.

    Returns a typed result describing whether the request created, coalesced,
    or upgraded the effective pending work for the library.
    """
    async with get_connection() as conn:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO doc_search.index_jobs (library_id, job_type, priority)
                VALUES ($1, $2, $3)
                RETURNING *
                """,
                library_id,
                job_type.value,
                priority,
            )
            if row:
                return JobEnqueueResult(
                    disposition=JobEnqueueDisposition.QUEUED,
                    job=_row_to_index_job(row),
                )
        except asyncpg.UniqueViolationError:
            pending_row = await conn.fetchrow(
                """
                SELECT *
                FROM doc_search.index_jobs
                WHERE library_id = $1 AND status = 'pending'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                library_id,
            )
            if pending_row is None:
                running_row = await conn.fetchrow(
                    """
                    SELECT *
                    FROM doc_search.index_jobs
                    WHERE library_id = $1 AND status = 'running'
                    ORDER BY started_at DESC NULLS LAST, created_at DESC
                    LIMIT 1
                    """,
                    library_id,
                )
                if running_row is not None:
                    logger.debug("Job request coalesced into running job for %s", library_id)
                    return JobEnqueueResult(
                        disposition=JobEnqueueDisposition.COALESCED,
                        job=_row_to_index_job(running_row),
                    )
                raise

            pending_job = _row_to_index_job(pending_row)
            next_job_type = pending_job.job_type
            if _JOB_TYPE_STRENGTH[job_type] > _JOB_TYPE_STRENGTH[pending_job.job_type]:
                next_job_type = job_type
            next_priority = min(pending_job.priority, priority)
            upgraded = (
                next_job_type != pending_job.job_type or next_priority != pending_job.priority
            )

            if upgraded:
                updated_row = await conn.fetchrow(
                    """
                    UPDATE doc_search.index_jobs
                    SET job_type = $2, priority = $3
                    WHERE id = $1
                    RETURNING *
                    """,
                    pending_job.id,
                    next_job_type.value,
                    next_priority,
                )
                if updated_row is None:
                    raise RuntimeError(f"Failed to upgrade pending job for {library_id}") from None
                logger.info(
                    "Upgraded pending job for %s to type=%s priority=%d",
                    library_id,
                    next_job_type.value,
                    next_priority,
                )
                return JobEnqueueResult(
                    disposition=JobEnqueueDisposition.UPGRADED,
                    job=_row_to_index_job(updated_row),
                )

            logger.debug("Job coalesced for %s", library_id)
            return JobEnqueueResult(
                disposition=JobEnqueueDisposition.COALESCED,
                job=pending_job,
            )

    raise RuntimeError(f"Failed to enqueue job for {library_id}")


async def claim_job(worker_id: str) -> IndexJob | None:
    """
    Claim the next available job.

    Uses FOR UPDATE SKIP LOCKED for safe concurrent claiming.
    """
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            UPDATE doc_search.index_jobs
            SET status = 'running',
                worker_id = $1,
                claimed_at = NOW(),
                heartbeat_at = NOW(),
                started_at = NOW()
            WHERE id = (
                SELECT id FROM doc_search.index_jobs
                WHERE status = 'pending'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM doc_search.index_jobs AS running
                      WHERE running.library_id = doc_search.index_jobs.library_id
                        AND running.status = 'running'
                  )
                ORDER BY priority, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """,
            worker_id,
        )
        if row:
            return _row_to_index_job(row)
    return None


async def heartbeat_job(job_id: str, files_processed: int | None = None) -> bool:
    """Update job heartbeat. Returns True if job still exists and is running."""
    async with get_connection() as conn:
        result = await conn.execute(
            """
            UPDATE doc_search.index_jobs
            SET heartbeat_at = NOW(),
                files_processed = COALESCE($2, files_processed)
            WHERE id = $1 AND status = 'running'
            """,
            job_id,
            files_processed,
        )
        return result == "UPDATE 1"


async def complete_job(job_id: str) -> bool:
    """Mark a job as completed."""
    async with get_connection() as conn:
        result = await conn.execute(
            """
            UPDATE doc_search.index_jobs
            SET status = 'completed', completed_at = NOW()
            WHERE id = $1 AND status = 'running'
            """,
            job_id,
        )
        return result == "UPDATE 1"


async def fail_job(job_id: str, error_message: str) -> bool:
    """Mark a job as failed."""
    async with get_connection() as conn:
        result = await conn.execute(
            """
            UPDATE doc_search.index_jobs
            SET status = 'failed', completed_at = NOW(), error_message = $2
            WHERE id = $1 AND status = 'running'
            """,
            job_id,
            error_message,
        )
        return result == "UPDATE 1"


async def reclaim_stale_jobs(stale_timeout_sec: int) -> list[str]:
    """
    Reclaim jobs that haven't heartbeated within the timeout.

    Running jobs are reclaimed in two ways:
    - reset to `pending` when no pending job for the same library exists
    - marked `failed` when a newer pending job for the same library already exists

    Returns list of stale job IDs that were reclaimed by either path.
    """
    async with get_connection() as conn:
        async with conn.transaction():
            retry_rows = await conn.fetch(
                """
                UPDATE doc_search.index_jobs AS running
                SET status = 'pending',
                    worker_id = NULL,
                    claimed_at = NULL,
                    heartbeat_at = NULL,
                    started_at = NULL
                WHERE running.status = 'running'
                  AND running.heartbeat_at < NOW() - INTERVAL '1 second' * $1
                  AND NOT EXISTS (
                      SELECT 1
                      FROM doc_search.index_jobs AS pending
                      WHERE pending.library_id = running.library_id
                        AND pending.status = 'pending'
                  )
                RETURNING running.id
                """,
                stale_timeout_sec,
            )

            superseded_rows = await conn.fetch(
                """
                UPDATE doc_search.index_jobs AS running
                SET status = 'failed',
                    completed_at = NOW(),
                    error_message = $2
                WHERE running.status = 'running'
                  AND running.heartbeat_at < NOW() - INTERVAL '1 second' * $1
                  AND EXISTS (
                      SELECT 1
                      FROM doc_search.index_jobs AS pending
                      WHERE pending.library_id = running.library_id
                        AND pending.status = 'pending'
                  )
                RETURNING running.id
                """,
                stale_timeout_sec,
                "Stale running job superseded by existing pending job during recovery",
            )

        return [str(row["id"]) for row in [*retry_rows, *superseded_rows]]


async def get_job(job_id: str) -> IndexJob | None:
    """Get a job by ID."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM doc_search.index_jobs WHERE id = $1",
            job_id,
        )
        if row:
            return _row_to_index_job(row)
    return None


async def list_jobs(
    library_id: str | None = None,
    status: JobStatus | None = None,
    limit: int = 100,
) -> list[IndexJob]:
    """List jobs with optional filters."""
    conditions = []
    values = []
    idx = 1

    if library_id is not None:
        conditions.append(f"library_id = ${idx}")
        values.append(library_id)
        idx += 1
    if status is not None:
        conditions.append(f"status = ${idx}")
        values.append(status.value)
        idx += 1

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    values.append(limit)

    async with get_connection() as conn:
        rows = await conn.fetch(
            f"""
            SELECT * FROM doc_search.index_jobs
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${idx}
            """,
            *values,
        )
        return [_row_to_index_job(row) for row in rows]
