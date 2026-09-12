"""Queue primitives for durable docs graph sync jobs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from repowise.docs.db import get_connection

QUEUE_SCHEMA_READY = False
QUEUE_SCHEMA_LOCK: asyncio.Lock | None = None
GRAPH_SYNC_BASE_RETRY_SECONDS = 30
GRAPH_SYNC_MAX_RETRY_SECONDS = 15 * 60


class GraphSyncAction(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"


@dataclass
class GraphSyncJob:
    library_id: str
    action: GraphSyncAction
    status: str
    attempts: int
    next_attempt_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime


def get_queue_lock() -> asyncio.Lock:
    global QUEUE_SCHEMA_LOCK
    if QUEUE_SCHEMA_LOCK is None:
        QUEUE_SCHEMA_LOCK = asyncio.Lock()
    return QUEUE_SCHEMA_LOCK


async def ensure_queue_schema() -> None:
    global QUEUE_SCHEMA_READY
    if QUEUE_SCHEMA_READY:
        return

    async with get_queue_lock():
        if QUEUE_SCHEMA_READY:
            return
        async with get_connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_search.graph_sync_jobs (
                    library_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL
                        CHECK (action IN ('upsert', 'delete')),
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'running', 'failed')),
                    attempts INT NOT NULL DEFAULT 0,
                    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_sync_jobs_claim
                    ON doc_search.graph_sync_jobs (status, next_attempt_at, updated_at)
                """
            )
        QUEUE_SCHEMA_READY = True


def row_to_graph_sync_job(row) -> GraphSyncJob:
    return GraphSyncJob(
        library_id=row["library_id"],
        action=GraphSyncAction(row["action"]),
        status=row["status"],
        attempts=row["attempts"],
        next_attempt_at=row["next_attempt_at"],
        last_error=row["last_error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def graph_retry_delay_seconds(attempts: int) -> int:
    return min(
        GRAPH_SYNC_MAX_RETRY_SECONDS, GRAPH_SYNC_BASE_RETRY_SECONDS * (2 ** max(0, attempts))
    )


async def enqueue_graph_sync_job(library_id: str, action: GraphSyncAction) -> GraphSyncJob:
    await ensure_queue_schema()
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO doc_search.graph_sync_jobs (
                library_id,
                action,
                status,
                attempts,
                next_attempt_at,
                last_error
            )
            VALUES ($1, $2, 'pending', 0, NOW(), NULL)
            ON CONFLICT (library_id) DO UPDATE
            SET action = EXCLUDED.action,
                status = 'pending',
                next_attempt_at = NOW(),
                last_error = NULL,
                updated_at = NOW()
            RETURNING *
            """,
            library_id,
            action.value,
        )
    return row_to_graph_sync_job(row)


async def claim_graph_sync_job() -> GraphSyncJob | None:
    await ensure_queue_schema()
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            UPDATE doc_search.graph_sync_jobs
            SET status = 'running',
                updated_at = NOW()
            WHERE library_id = (
                SELECT library_id
                FROM doc_search.graph_sync_jobs
                WHERE status IN ('pending', 'failed')
                  AND next_attempt_at <= NOW()
                ORDER BY
                    CASE action WHEN 'delete' THEN 0 ELSE 1 END,
                    updated_at,
                    created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING *
            """
        )
    return row_to_graph_sync_job(row) if row else None


async def list_graph_sync_jobs() -> dict[str, GraphSyncJob]:
    await ensure_queue_schema()
    async with get_connection() as conn:
        rows = await conn.fetch("SELECT * FROM doc_search.graph_sync_jobs")
    return {row["library_id"]: row_to_graph_sync_job(row) for row in rows}


async def complete_graph_sync_job(library_id: str) -> None:
    await ensure_queue_schema()
    async with get_connection() as conn:
        await conn.execute(
            "DELETE FROM doc_search.graph_sync_jobs WHERE library_id = $1",
            library_id,
        )


async def fail_graph_sync_job(library_id: str, error: str, attempts: int) -> None:
    await ensure_queue_schema()
    delay_seconds = graph_retry_delay_seconds(attempts)
    async with get_connection() as conn:
        await conn.execute(
            """
            UPDATE doc_search.graph_sync_jobs
            SET status = 'failed',
                attempts = attempts + 1,
                next_attempt_at = NOW() + INTERVAL '1 second' * $2,
                last_error = $3,
                updated_at = NOW()
            WHERE library_id = $1
            """,
            library_id,
            delay_seconds,
            error,
        )
