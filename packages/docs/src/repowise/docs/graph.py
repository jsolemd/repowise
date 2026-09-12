"""Neo4j metadata sync for indexed documentation libraries."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime

from repowise.docs.db import (
    get_library,
    get_library_files,
    list_libraries,
    mark_graph_sync_pending,
    update_library_status,
)
from repowise.docs.graph_queue import (
    GraphSyncAction,
    GraphSyncJob,
    enqueue_graph_sync_job,
)
from repowise.docs.graph_queue import (
    claim_graph_sync_job as _claim_graph_sync_job,
)
from repowise.docs.graph_queue import (
    complete_graph_sync_job as _complete_graph_sync_job,
)
from repowise.docs.graph_queue import (
    ensure_queue_schema as _ensure_queue_schema,
)
from repowise.docs.graph_queue import (
    fail_graph_sync_job as _fail_graph_sync_job,
)
from repowise.docs.graph_queue import (
    list_graph_sync_jobs as _list_graph_sync_jobs,
)
from repowise.docs.graph_runtime import (
    SYNC_STATE as _SYNC_STATE,
)
from repowise.docs.graph_runtime import (
    docs_graph_enabled as _docs_graph_enabled,
)
from repowise.docs.graph_runtime import (
    ensure_docs_graph_schema as _ensure_schema,
)
from repowise.docs.graph_runtime import (
    get_docs_graph_client as _get_client,
)
from repowise.docs.graph_runtime import (
    record_sync_failure as _record_sync_failure,
)
from repowise.docs.graph_runtime import (
    record_sync_success as _record_sync_success,
)
from repowise.docs.graph_storage import (
    fetch_docs_graph_file_counts as _fetch_docs_graph_file_counts,
)
from repowise.docs.graph_storage import (
    file_rows as _file_rows,
)
from repowise.docs.library.models import LibraryFile, LibraryState, LibraryStatus

logger = logging.getLogger(__name__)

_GRAPH_SYNC_TASK: asyncio.Task | None = None
_GRAPH_SYNC_STOP_EVENT: asyncio.Event | None = None
_LAST_GRAPH_RECONCILE_AT: datetime | None = None
_GRAPH_SYNC_POLL_INTERVAL_SECONDS = 30
_GRAPH_SYNC_RECONCILE_INTERVAL_SECONDS = 10 * 60
_GRAPH_SYNC_PENDING_REASON = "graph metadata reconciliation pending"


async def _persist_graph_status(
    library_id: str,
    *,
    graph_synced_at: datetime | None = None,
    graph_sync_error: str | None = None,
) -> None:
    try:
        await update_library_status(
            library_id,
            graph_synced_at=graph_synced_at,
            graph_sync_error=graph_sync_error,
        )
    except Exception:
        return


def docs_graph_enabled() -> bool:
    """Return True when docs metadata sync to Neo4j is enabled."""
    return _docs_graph_enabled()


def get_docs_graph_sync_status() -> dict[str, object]:
    """Expose the latest docs->Neo4j sync state for health/readiness surfaces."""
    enabled = docs_graph_enabled()
    state = "disabled"
    if enabled:
        state = "error" if _SYNC_STATE.last_error else "ok"
        if _SYNC_STATE.last_attempt_at is None:
            state = "idle"
    return {
        "enabled": enabled,
        "state": state,
        "last_action": _SYNC_STATE.last_action,
        "last_library_id": _SYNC_STATE.last_library_id,
        "last_file_count": _SYNC_STATE.last_file_count,
        "last_attempt_at": (
            _SYNC_STATE.last_attempt_at.isoformat() if _SYNC_STATE.last_attempt_at else None
        ),
        "last_success_at": (
            _SYNC_STATE.last_success_at.isoformat() if _SYNC_STATE.last_success_at else None
        ),
        "last_error": _SYNC_STATE.last_error,
        "relay_running": _GRAPH_SYNC_TASK is not None and not _GRAPH_SYNC_TASK.done(),
    }


async def _safe_enqueue_graph_sync_job(library_id: str, action: GraphSyncAction) -> None:
    try:
        await enqueue_graph_sync_job(library_id, action)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Could not queue docs graph sync for %s (%s): %s",
            library_id,
            action.value,
            exc,
        )


async def _drain_graph_sync_job(job: GraphSyncJob) -> None:
    try:
        if job.action == GraphSyncAction.DELETE:
            success = await delete_library_metadata(job.library_id, enqueue_on_failure=False)
        else:
            library = await get_library(job.library_id)
            if library is None:
                success = await delete_library_metadata(job.library_id, enqueue_on_failure=False)
            else:
                files = await get_library_files(job.library_id)
                success = await sync_library_metadata(
                    library,
                    files,
                    enqueue_on_failure=False,
                )
        if success:
            await _complete_graph_sync_job(job.library_id)
            return
        error = _SYNC_STATE.last_error or f"{job.action.value} sync failed"
    except Exception as exc:  # pragma: no cover - defensive
        error = str(exc)
        logger.warning("Docs graph queue job failed for %s: %s", job.library_id, exc)

    await _fail_graph_sync_job(job.library_id, error, job.attempts)


async def _graph_sync_loop(stop_event: asyncio.Event) -> None:
    global _LAST_GRAPH_RECONCILE_AT
    logger.info("Docs graph relay started")
    while not stop_event.is_set():
        try:
            job = await _claim_graph_sync_job()
            if job is None:
                now = datetime.now(UTC)
                if (
                    _LAST_GRAPH_RECONCILE_AT is None
                    or (now - _LAST_GRAPH_RECONCILE_AT).total_seconds()
                    >= _GRAPH_SYNC_RECONCILE_INTERVAL_SECONDS
                ):
                    await reconcile_docs_graph_sync()
                    _LAST_GRAPH_RECONCILE_AT = now
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=_GRAPH_SYNC_POLL_INTERVAL_SECONDS,
                    )
                    break
                except TimeoutError:
                    continue
            await _drain_graph_sync_job(job)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Docs graph relay error: %s", exc)
            await asyncio.sleep(1)
    logger.info("Docs graph relay stopped")


async def start_docs_graph_sync_task() -> None:
    global _GRAPH_SYNC_TASK, _GRAPH_SYNC_STOP_EVENT
    if not docs_graph_enabled():
        return
    if _GRAPH_SYNC_TASK is not None and not _GRAPH_SYNC_TASK.done():
        return
    await _ensure_queue_schema()
    _GRAPH_SYNC_STOP_EVENT = asyncio.Event()
    _GRAPH_SYNC_TASK = asyncio.create_task(
        _graph_sync_loop(_GRAPH_SYNC_STOP_EVENT),
        name="doc-search-graph-sync",
    )


async def stop_docs_graph_sync_task(timeout: float = 5.0) -> None:
    global _GRAPH_SYNC_TASK, _GRAPH_SYNC_STOP_EVENT, _LAST_GRAPH_RECONCILE_AT
    if _GRAPH_SYNC_STOP_EVENT is not None:
        _GRAPH_SYNC_STOP_EVENT.set()
    if _GRAPH_SYNC_TASK is not None:
        try:
            await asyncio.wait_for(_GRAPH_SYNC_TASK, timeout=timeout)
        except TimeoutError:
            _GRAPH_SYNC_TASK.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _GRAPH_SYNC_TASK
    _GRAPH_SYNC_TASK = None
    _GRAPH_SYNC_STOP_EVENT = None
    _LAST_GRAPH_RECONCILE_AT = None


def _needs_graph_reconciliation(
    library: LibraryState,
    *,
    graph_file_count: int | None,
) -> bool:
    return (
        library.status == LibraryStatus.READY
        and library.file_count > 0
        and (
            library.graph_synced_at is None
            or bool(library.graph_sync_error)
            or graph_file_count is None
            or graph_file_count != library.file_count
        )
    )


async def reconcile_docs_graph_sync() -> dict[str, int]:
    """Queue graph replays for ready libraries whose Neo4j metadata is missing or stale."""
    if not docs_graph_enabled():
        return {"eligible_libraries": 0, "gap_libraries": 0, "jobs_enqueued": 0}

    try:
        client = await _get_client()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Docs graph reconciliation unavailable: %s", exc)
        return {"eligible_libraries": 0, "gap_libraries": 0, "jobs_enqueued": 0}

    if client is None or not await client.verify_connectivity():
        return {"eligible_libraries": 0, "gap_libraries": 0, "jobs_enqueued": 0}

    await _ensure_schema(client)
    libraries = [
        lib for lib in await list_libraries(status=LibraryStatus.READY) if lib.file_count > 0
    ]
    if not libraries:
        return {"eligible_libraries": 0, "gap_libraries": 0, "jobs_enqueued": 0}

    graph_file_counts = await _fetch_docs_graph_file_counts(client)
    queued_jobs = await _list_graph_sync_jobs()
    gap_libraries = 0
    jobs_enqueued = 0

    for library in libraries:
        if not _needs_graph_reconciliation(
            library,
            graph_file_count=graph_file_counts.get(library.library_id),
        ):
            continue
        gap_libraries += 1
        if (
            library.graph_synced_at is not None
            or library.graph_sync_error != _GRAPH_SYNC_PENDING_REASON
        ):
            await mark_graph_sync_pending(
                library.library_id,
                error=_GRAPH_SYNC_PENDING_REASON,
            )
        existing_job = queued_jobs.get(library.library_id)
        if existing_job is None or existing_job.action != GraphSyncAction.UPSERT:
            await enqueue_graph_sync_job(library.library_id, GraphSyncAction.UPSERT)
            jobs_enqueued += 1

    return {
        "eligible_libraries": len(libraries),
        "gap_libraries": gap_libraries,
        "jobs_enqueued": jobs_enqueued,
    }


async def sync_library_metadata(
    library: LibraryState,
    files: list[LibraryFile],
    *,
    enqueue_on_failure: bool = True,
) -> bool:
    """Upsert library/file metadata into Neo4j for mixed code+docs navigation."""
    try:
        client = await _get_client()
    except Exception as exc:
        _record_sync_failure(
            action="sync",
            library_id=library.library_id,
            error=f"client init failed: {exc}",
        )
        await _persist_graph_status(
            library.library_id,
            graph_sync_error=f"client init failed: {exc}",
        )
        logger.warning("Docs graph sync unavailable for %s: %s", library.library_id, exc)
        if enqueue_on_failure:
            await _safe_enqueue_graph_sync_job(library.library_id, GraphSyncAction.UPSERT)
        return False
    if client is None:
        if enqueue_on_failure:
            await _safe_enqueue_graph_sync_job(library.library_id, GraphSyncAction.UPSERT)
        return False
    if not await client.verify_connectivity():
        _record_sync_failure(
            action="sync",
            library_id=library.library_id,
            error="neo4j unavailable",
        )
        await _persist_graph_status(
            library.library_id,
            graph_sync_error="neo4j unavailable",
        )
        logger.warning("Skipping docs graph sync for %s; Neo4j unavailable", library.library_id)
        if enqueue_on_failure:
            await _safe_enqueue_graph_sync_job(library.library_id, GraphSyncAction.UPSERT)
        return False

    try:
        await _ensure_schema(client)

        await client.execute_write(
            """
            MERGE (lib:DocLibrary {library_id: $library_id})
            SET lib.source_type = $source_type,
                lib.repo = $repo,
                lib.source_subpath = $source_subpath,
                lib.name = $name,
                lib.description = $description,
                lib.docs_path = $docs_path,
                lib.branch = $branch,
                lib.priority = $priority,
                lib.status = $status,
                lib.current_sha = $current_sha,
                lib.indexed_at = $indexed_at,
                lib.freshness_checked_at = $freshness_checked_at,
                lib.chunk_count = $chunk_count,
                lib.file_count = $file_count,
                lib.updated_at = $updated_at,
                lib.source = 'doc_search'
            """,
            {
                "library_id": library.library_id,
                "source_type": library.source_type.value,
                "repo": library.repo,
                "source_subpath": library.source_subpath,
                "name": library.name,
                "description": library.description,
                "docs_path": library.docs_path,
                "branch": library.branch,
                "priority": library.priority,
                "status": library.status.value,
                "current_sha": library.current_sha,
                "indexed_at": library.indexed_at.isoformat() if library.indexed_at else None,
                "freshness_checked_at": (
                    library.freshness_checked_at.isoformat()
                    if library.freshness_checked_at
                    else None
                ),
                "chunk_count": library.chunk_count,
                "file_count": library.file_count,
                "updated_at": library.updated_at.isoformat(),
            },
        )

        rows = _file_rows(files)
        if rows:
            await client.execute_write(
                """
                MATCH (lib:DocLibrary {library_id: $library_id})
                UNWIND $files AS file
                MERGE (doc:DocFile {library_id: $library_id, file_path: file.file_path})
                SET doc.content_hash = file.content_hash,
                    doc.chunk_count = file.chunk_count,
                    doc.indexed_at = file.indexed_at,
                    doc.updated_at = file.updated_at,
                    doc.source = 'doc_search'
                MERGE (lib)-[:HAS_FILE]->(doc)
                """,
                {
                    "library_id": library.library_id,
                    "files": rows,
                },
            )

        await client.execute_write(
            """
            MATCH (lib:DocLibrary {library_id: $library_id})-[:HAS_FILE]->(doc:DocFile {library_id: $library_id})
            WHERE NOT doc.file_path IN $paths
            DETACH DELETE doc
            """,
            {
                "library_id": library.library_id,
                "paths": [row["file_path"] for row in rows],
            },
        )
    except Exception as exc:
        _record_sync_failure(
            action="sync",
            library_id=library.library_id,
            error=str(exc),
        )
        await _persist_graph_status(
            library.library_id,
            graph_sync_error=str(exc),
        )
        logger.warning("Docs graph sync failed for %s: %s", library.library_id, exc)
        if enqueue_on_failure:
            await _safe_enqueue_graph_sync_job(library.library_id, GraphSyncAction.UPSERT)
        return False

    _record_sync_success(
        action="sync",
        library_id=library.library_id,
        file_count=len(files),
    )
    await _persist_graph_status(
        library.library_id,
        graph_synced_at=datetime.now(UTC),
        graph_sync_error="",
    )
    logger.info(
        "Synced docs graph metadata for %s (%d files)",
        library.library_id,
        len(rows),
    )
    return True


async def delete_library_metadata(
    library_id: str,
    *,
    enqueue_on_failure: bool = True,
) -> bool:
    """Remove one library and its doc-file nodes from Neo4j."""
    try:
        client = await _get_client()
    except Exception as exc:
        _record_sync_failure(
            action="delete",
            library_id=library_id,
            error=f"client init failed: {exc}",
        )
        logger.warning("Docs graph delete unavailable for %s: %s", library_id, exc)
        if enqueue_on_failure:
            await _safe_enqueue_graph_sync_job(library_id, GraphSyncAction.DELETE)
        return False
    if client is None:
        if enqueue_on_failure:
            await _safe_enqueue_graph_sync_job(library_id, GraphSyncAction.DELETE)
        return False
    if not await client.verify_connectivity():
        _record_sync_failure(
            action="delete",
            library_id=library_id,
            error="neo4j unavailable",
        )
        logger.warning("Skipping docs graph delete for %s; Neo4j unavailable", library_id)
        if enqueue_on_failure:
            await _safe_enqueue_graph_sync_job(library_id, GraphSyncAction.DELETE)
        return False
    try:
        await _ensure_schema(client)
        await client.execute_write(
            "MATCH (lib:DocLibrary {library_id: $library_id}) DETACH DELETE lib",
            {"library_id": library_id},
        )
    except Exception as exc:
        _record_sync_failure(
            action="delete",
            library_id=library_id,
            error=str(exc),
        )
        logger.warning("Docs graph delete failed for %s: %s", library_id, exc)
        if enqueue_on_failure:
            await _safe_enqueue_graph_sync_job(library_id, GraphSyncAction.DELETE)
        return False

    _record_sync_success(action="delete", library_id=library_id, file_count=0)
    return True
