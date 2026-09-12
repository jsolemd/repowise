"""Transport-neutral docs runtime with explicit reader/writer ownership."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass

import anyio

from repowise.docs.bootstrap import sync_library_registry
from repowise.docs.db import close_pool, init_pool, list_libraries
from repowise.docs.indexer import close_qdrant_client
from repowise.docs.indexer.embedder import close_http_client as close_docs_embedding_http_client
from repowise.docs.jobs import (
    acquire_background_lock,
    release_background_lock,
    start_recovery_task,
    start_scheduler,
    start_worker,
    stop_recovery_task,
    stop_scheduler,
    stop_worker,
)
from repowise.docs.library.remote import close_http_client as close_docs_remote_http_client
from repowise.docs.scrapers.snapshot_runtime import bootstrap_missing_snapshot_libraries
from repowise.docs.server.health import (
    close_health_checker,
    initialize_health_checker,
)

logger = logging.getLogger(__name__)


@dataclass
class _DocsRuntimeState:
    initialized: bool = False
    background_started: bool = False
    worker_id: str | None = None
    bootstrap_summary: dict[str, object] | None = None
    lock: asyncio.Lock | None = None
    #: Set when this process asked for the background runtime but another
    #: process already owns the advisory lock. Reads still work; the writer
    #: loops are deliberately not started.
    background_declined_reason: str | None = None


_STATE = _DocsRuntimeState()


def _get_lock() -> asyncio.Lock:
    if _STATE.lock is None:
        _STATE.lock = asyncio.Lock()
    return _STATE.lock


async def ensure_docs_runtime(*, start_background: bool = False) -> None:
    """Initialize the shared docs runtime once for both MCP and HTTP usage.

    ``start_background=True`` makes this process the docs writer: registry
    sync, snapshot bootstrap, job worker, freshness scheduler, and recovery
    loop. That role is exclusive and is claimed through a Postgres advisory
    lock, so a second replica of the same container degrades to a reader
    instead of double-running the writer loops.
    """
    async with _get_lock():
        if not _STATE.initialized:
            try:
                await init_pool()
                await initialize_health_checker()
                _STATE.initialized = True
            except BaseException:
                with anyio.CancelScope(shield=True):
                    await close_health_checker()
                    await close_pool()
                    _STATE.initialized = False
                raise

        if start_background and not _STATE.background_started:
            # Claim the writer role before any writer-side work. Registry sync
            # and snapshot bootstrap enqueue jobs and mutate libraries, so they
            # belong to the single owner just as much as the loops below do.
            if not await acquire_background_lock():
                _STATE.background_declined_reason = (
                    "another process holds the docs background-runtime advisory lock"
                )
                logger.warning(
                    "Docs background runtime not started: %s. This process serves reads only.",
                    _STATE.background_declined_reason,
                )
                return
            _STATE.background_declined_reason = None

            resources = AsyncExitStack()
            resources.push_async_callback(release_background_lock)
            try:
                registry_summary = await sync_library_registry(queue_missing_jobs=True)
                snapshot_bootstrap = await bootstrap_missing_snapshot_libraries(
                    await list_libraries()
                )
                _STATE.bootstrap_summary = {
                    "registry": registry_summary,
                    "snapshot_bootstrap": snapshot_bootstrap,
                }
                logger.info("Docs registry sync summary: %s", registry_summary)
                logger.info("Snapshot bootstrap summary: %s", snapshot_bootstrap)

                _STATE.worker_id = await start_worker()
                resources.push_async_callback(stop_worker)
                await start_scheduler()
                resources.push_async_callback(stop_scheduler)
                await start_recovery_task()
                resources.push_async_callback(stop_recovery_task)
                _STATE.background_started = True
                resources.pop_all()  # close_docs_runtime owns the running loops
            except BaseException:
                # Cancellation can interrupt bootstrap before any worker starts.
                # Return ownership and every acquired loop before allowing retry.
                with anyio.CancelScope(shield=True):
                    try:
                        await resources.aclose()
                    finally:
                        _STATE.worker_id = None
                        _STATE.background_started = False
                        _STATE.bootstrap_summary = None
                raise


async def close_docs_runtime() -> None:
    """Tear down the shared docs runtime."""
    async with _get_lock():
        if _STATE.background_started:
            await stop_recovery_task()
            await stop_scheduler()
            await stop_worker()
            _STATE.background_started = False
            _STATE.worker_id = None

        # Always attempt the release: a start that failed part-way already gave
        # the lock back, and this is a no-op when the lock was never held.
        await release_background_lock()
        _STATE.background_declined_reason = None

        if _STATE.initialized:
            await close_health_checker()
            await close_docs_embedding_http_client()
            await close_docs_remote_http_client()
            close_qdrant_client()
            await close_pool()
            _STATE.initialized = False
            _STATE.bootstrap_summary = None
            logger.info("Docs runtime closed")


def docs_runtime_accepts_reads(*, status: dict[str, object] | None = None) -> bool:
    """Return True when read-only docs navigation is available."""
    if status is None:
        return _STATE.initialized
    return bool(status.get("initialized") and status.get("health_status_code") == 200)


def docs_runtime_accepts_mutations(*, status: dict[str, object] | None = None) -> bool:
    """Return True when queued docs mutations can be safely accepted."""
    if status is None:
        return _STATE.initialized and _STATE.background_started
    return bool(
        status.get("initialized")
        and status.get("background_started")
        and status.get("health_status_code") == 200
    )
