"""Cluster-wide single-owner guard for the docs background runtime.

The docs writer now lives inside the ``repowise-docs`` host itself. Two
replicas of that container would each start a job worker, a freshness
scheduler, and a recovery loop against the same ``doc_search`` schema. The job
queue's ``FOR UPDATE SKIP LOCKED`` claim already protects an individual job
from being run twice, but the surrounding loops are not safe under
concurrency: two schedulers double-enqueue freshness work, two recovery sweeps
reclaim each other's in-flight jobs, and two registry syncs race on library
bootstrap.

A PostgreSQL session-level advisory lock makes "exactly one background owner"
an invariant of the database rather than of the deployment, with no new table,
no new dependency, and no cleanup path on crash (the lock dies with the
session).

The lock is deliberately taken on a **dedicated** connection opened outside the
shared pool: asyncpg runs ``pg_advisory_unlock_all()`` as part of its
connection reset when a pooled connection is released, so a pooled connection
physically cannot hold this lock past the end of one ``acquire()`` block.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg

from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)

#: Fixed 64-bit key for ``pg_try_advisory_lock``. Derived once from
#: ``blake2b(b"repowise.docs.background_runtime", digest_size=8)`` read as a signed
#: big-endian int, then frozen as a literal so the key is greppable and can
#: never drift with a hash-implementation change. Never reuse it for another
#: advisory lock in this database.
DOCS_BACKGROUND_LOCK_KEY = -6896715891042309208

Connector = Callable[..., Awaitable[Any]]

_lock_connection: Any | None = None


def holds_background_lock() -> bool:
    """Whether this process currently owns the docs background runtime."""
    return _lock_connection is not None


async def acquire_background_lock(*, connect: Connector | None = None) -> bool:
    """Try to become the single docs background owner.

    Returns ``True`` when this process holds the lock (including when it
    already held it), ``False`` when another process owns it. A ``False``
    result is not an error: the caller is expected to keep serving reads and
    skip the background runtime.
    """
    global _lock_connection
    if _lock_connection is not None:
        return True

    settings = get_settings()
    dsn = settings.postgres_dsn
    if not dsn:
        raise ValueError("postgres_dsn (or DATABASE_URL) must be set")

    connector: Connector = connect or asyncpg.connect
    conn = await connector(dsn)
    try:
        acquired = bool(
            await conn.fetchval("SELECT pg_try_advisory_lock($1)", DOCS_BACKGROUND_LOCK_KEY)
        )
    except BaseException:
        await _close_quietly(conn)
        raise

    if not acquired:
        await _close_quietly(conn)
        return False

    _lock_connection = conn
    logger.info(
        "Acquired the docs background-runtime advisory lock (key=%d)", DOCS_BACKGROUND_LOCK_KEY
    )
    return True


async def release_background_lock() -> None:
    """Release the advisory lock and close its dedicated connection."""
    global _lock_connection
    conn, _lock_connection = _lock_connection, None
    if conn is None:
        return

    try:
        await conn.fetchval("SELECT pg_advisory_unlock($1)", DOCS_BACKGROUND_LOCK_KEY)
    except Exception as exc:  # pragma: no cover - unlock is best effort
        # Closing the session below drops the lock regardless, so a failure
        # here is loggable but never fatal.
        logger.warning("Failed to unlock the docs background-runtime lock: %s", exc)
    finally:
        await _close_quietly(conn)
        logger.info("Released the docs background-runtime advisory lock")


async def _close_quietly(conn: Any) -> None:
    try:
        await conn.close()
    except Exception as exc:  # pragma: no cover - teardown is best effort
        logger.warning("Failed to close the docs background-runtime lock connection: %s", exc)


__all__ = [
    "DOCS_BACKGROUND_LOCK_KEY",
    "acquire_background_lock",
    "holds_background_lock",
    "release_background_lock",
]
