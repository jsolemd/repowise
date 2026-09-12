"""Connection-pool and schema-compatibility helpers for doc-search."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def _ensure_schema_compatibility(pool: asyncpg.Pool) -> None:
    """Apply idempotent queue/index compatibility updates for live deployments.

    All DDL and backfill statements are idempotent (IF NOT EXISTS / WHERE guards)
    and batched into a single round-trip for startup efficiency.
    """
    async with pool.acquire() as conn:
        await conn.execute("""
            -- Legacy index cleanup
            DROP INDEX IF EXISTS doc_search.idx_jobs_coalesce;

            -- Column additions (all IF NOT EXISTS — safe to re-run)
            ALTER TABLE doc_search.libraries ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'git';
            ALTER TABLE doc_search.libraries ADD COLUMN IF NOT EXISTS source_subpath TEXT DEFAULT '';
            ALTER TABLE doc_search.libraries ADD COLUMN IF NOT EXISTS freshness_checked_at TIMESTAMPTZ;
            ALTER TABLE doc_search.libraries ADD COLUMN IF NOT EXISTS next_freshness_check_at TIMESTAMPTZ;
            ALTER TABLE doc_search.libraries ADD COLUMN IF NOT EXISTS last_freshness_state TEXT
                CHECK (last_freshness_state IN ('fresh', 'stale', 'unknown'));
            ALTER TABLE doc_search.libraries ADD COLUMN IF NOT EXISTS last_remote_sha TEXT;
            ALTER TABLE doc_search.libraries ADD COLUMN IF NOT EXISTS last_freshness_error TEXT;
            ALTER TABLE doc_search.libraries ADD COLUMN IF NOT EXISTS graph_synced_at TIMESTAMPTZ;
            ALTER TABLE doc_search.libraries ADD COLUMN IF NOT EXISTS graph_sync_error TEXT;
            ALTER TABLE doc_search.libraries ALTER COLUMN repo SET DEFAULT '';

            -- Backfill NULLs (guarded WHERE clauses — no-op when already populated)
            UPDATE doc_search.libraries SET source_type = 'git'
                WHERE source_type IS NULL OR source_type = '';
            UPDATE doc_search.libraries SET source_subpath = ''
                WHERE source_subpath IS NULL;
            UPDATE doc_search.libraries
                SET freshness_checked_at = COALESCE(indexed_at, updated_at, created_at)
                WHERE status = 'ready' AND freshness_checked_at IS NULL;
            UPDATE doc_search.libraries
                SET last_freshness_state = COALESCE(last_freshness_state, 'unknown')
                WHERE status = 'ready' AND last_freshness_state IS NULL;

            -- Indexes (all IF NOT EXISTS)
            CREATE INDEX IF NOT EXISTS idx_libraries_freshness_checked_at
                ON doc_search.libraries (freshness_checked_at);
            CREATE INDEX IF NOT EXISTS idx_libraries_next_freshness_check_at
                ON doc_search.libraries (next_freshness_check_at);
            CREATE INDEX IF NOT EXISTS idx_libraries_source_type
                ON doc_search.libraries (source_type);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_pending_coalesce
                ON doc_search.index_jobs (library_id) WHERE status = 'pending';

            -- Snapshot tables (IF NOT EXISTS)
            CREATE TABLE IF NOT EXISTS doc_search.snapshot_states (
                library_id TEXT PRIMARY KEY
                    REFERENCES doc_search.libraries(library_id) ON DELETE CASCADE,
                source_ref TEXT NOT NULL,
                manifest_hash TEXT NOT NULL,
                file_count INT NOT NULL DEFAULT 0,
                published_at TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_snapshot_states_published_at
                ON doc_search.snapshot_states (published_at);

            CREATE TABLE IF NOT EXISTS doc_search.snapshot_files (
                library_id TEXT NOT NULL
                    REFERENCES doc_search.libraries(library_id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_url TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (library_id, file_path)
            );
            CREATE INDEX IF NOT EXISTS idx_snapshot_files_library
                ON doc_search.snapshot_files (library_id);

            -- Index for claim_job correlated subquery (running jobs per library)
            CREATE INDEX IF NOT EXISTS idx_jobs_running_library
                ON doc_search.index_jobs (library_id) WHERE status = 'running';

            -- Triggers for snapshot tables (match other doc_search tables)
            CREATE OR REPLACE TRIGGER snapshot_states_updated_at
                BEFORE UPDATE ON doc_search.snapshot_states
                FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();
            CREATE OR REPLACE TRIGGER snapshot_files_updated_at
                BEFORE UPDATE ON doc_search.snapshot_files
                FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();
        """)


async def init_pool() -> asyncpg.Pool:
    """Initialize the connection pool."""
    global _pool
    if _pool is not None:
        return _pool

    async with _pool_lock:
        if _pool is not None:
            return _pool

        settings = get_settings()
        if not settings.postgres_dsn:
            raise ValueError("postgres_dsn (or DATABASE_URL) must be set")

        pool = await asyncpg.create_pool(
            settings.postgres_dsn,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        await _ensure_schema_compatibility(pool)
        _pool = pool
        logger.info("Database connection pool initialized")
        return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    async with _pool_lock:
        pool = _pool
        _pool = None

    if pool is not None:
        await pool.close()
        logger.info("Database connection pool closed")


async def get_pool() -> asyncpg.Pool:
    """Get the connection pool, initializing if needed."""
    if _pool is None:
        return await init_pool()
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncIterator[asyncpg.Connection]:
    """Get a connection from the pool."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn
