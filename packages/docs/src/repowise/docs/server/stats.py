"""Shared operator statistics for the integrated documentation-search subsystem."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from repowise.docs.config import get_settings
from repowise.docs.db import list_jobs
from repowise.docs.db import list_libraries as db_list_libraries
from repowise.docs.graph_status import get_docs_graph_sync_status
from repowise.docs.indexer.embedder import get_cache_stats
from repowise.docs.jobs import (
    JobStatus,
    JobType,
    get_scheduler_status,
    get_unreachable_libraries,
    get_worker_id,
    is_recovery_running,
    is_scheduler_running,
    is_worker_running,
)
from repowise.docs.library.models import (
    IndexJob,
    LibrarySourceType,
    LibraryState,
    LibraryStatus,
)

logger = logging.getLogger("doc-search")


def summarize_libraries(libraries: list[LibraryState]) -> dict[str, int | str]:
    """Summarize registry state for operator surfaces."""
    ready_count = sum(1 for lib in libraries if lib.status == LibraryStatus.READY)
    metadata_gap_count = sum(
        1
        for lib in libraries
        if lib.status == LibraryStatus.READY and (not lib.current_sha or lib.file_count <= 0)
    )
    graph_gap_count = sum(
        1
        for lib in libraries
        if lib.status == LibraryStatus.READY
        and lib.file_count > 0
        and (lib.graph_synced_at is None or bool(lib.graph_sync_error))
    )

    registry_status = "ok"
    if not libraries:
        registry_status = "empty"
    elif ready_count == 0 or metadata_gap_count > 0:
        registry_status = "warming"

    return {
        "status": registry_status,
        "total_libraries": len(libraries),
        "ready_libraries": ready_count,
        "pending_libraries": sum(1 for lib in libraries if lib.status == LibraryStatus.PENDING),
        "indexing_libraries": sum(1 for lib in libraries if lib.status == LibraryStatus.INDEXING),
        "error_libraries": sum(1 for lib in libraries if lib.status == LibraryStatus.ERROR),
        "metadata_gap_libraries": metadata_gap_count,
        "graph_gap_libraries": graph_gap_count,
    }


def serialize_library_state(lib: LibraryState) -> dict[str, object]:
    """Convert a library state into a compact operator-facing payload."""
    return {
        "library_id": lib.library_id,
        "name": lib.name,
        "repo": lib.repo,
        "source_type": lib.source_type.value
        if isinstance(lib.source_type, LibrarySourceType)
        else str(lib.source_type),
        "status": lib.status.value if isinstance(lib.status, LibraryStatus) else str(lib.status),
        "branch": lib.branch,
        "docs_path": lib.docs_path,
        "source_subpath": lib.source_subpath,
        "file_count": int(lib.file_count or 0),
        "chunk_count": int(lib.chunk_count or 0),
        "indexed_at": lib.indexed_at.isoformat() if lib.indexed_at else None,
        "graph_synced_at": lib.graph_synced_at.isoformat() if lib.graph_synced_at else None,
        "last_freshness_state": lib.last_freshness_state,
        "graph_sync_error": lib.graph_sync_error,
        "error_message": lib.error_message,
        "priority": int(lib.priority) if lib.priority is not None else None,
        "current_sha": lib.current_sha,
        "freshness_checked_at": lib.freshness_checked_at.isoformat()
        if lib.freshness_checked_at
        else None,
        "next_freshness_check_at": lib.next_freshness_check_at.isoformat()
        if lib.next_freshness_check_at
        else None,
        "last_remote_sha": lib.last_remote_sha,
        "last_freshness_error": lib.last_freshness_error,
    }


def serialize_job(job: IndexJob) -> dict[str, object]:
    """Convert an index job into the operator-facing job payload."""
    return {
        "id": job.id,
        "library_id": job.library_id,
        "job_type": job.job_type.value if isinstance(job.job_type, JobType) else str(job.job_type),
        "priority": job.priority,
        "status": job.status.value if isinstance(job.status, JobStatus) else str(job.status),
        "worker_id": job.worker_id,
        "files_processed": job.files_processed,
        "files_total": job.files_total,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "claimed_at": job.claimed_at.isoformat() if job.claimed_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "heartbeat_at": job.heartbeat_at.isoformat() if job.heartbeat_at else None,
    }


def count_library_inventory(libraries: list[LibraryState]) -> dict[str, int]:
    """Count indexed files, chunks, and source backings across the registry."""
    return {
        "total_files": sum(
            int(lib.file_count or 0) for lib in libraries if lib.status == LibraryStatus.READY
        ),
        "total_chunks": sum(
            int(lib.chunk_count or 0) for lib in libraries if lib.status == LibraryStatus.READY
        ),
        "git_libraries": sum(1 for lib in libraries if lib.source_type == LibrarySourceType.GIT),
        "snapshot_libraries": sum(
            1 for lib in libraries if lib.source_type == LibrarySourceType.SNAPSHOT
        ),
    }


def build_library_inventory_payload(libraries: list[LibraryState]) -> dict[str, object]:
    """Return the shared ``{summary, inventory, libraries}`` registry view."""
    return {
        "summary": summarize_libraries(libraries),
        "inventory": count_library_inventory(libraries),
        "libraries": [serialize_library_state(lib) for lib in libraries],
    }


async def build_docs_library_inventory() -> dict[str, object]:
    """Return a lightweight inventory for polling and operator views."""
    return build_library_inventory_payload(await db_list_libraries())


async def _default_get_health_status() -> tuple[dict, int]:
    """Read health lazily.

    ``repowise.docs.server.health`` imports :func:`summarize_libraries` from this
    module at import time, so a module-level import back into it would be a
    cycle. Callers that already hold the real function inject it instead.
    """
    from repowise.docs.server.health import get_health_status

    return await get_health_status()


def _job_completion_sort_key(job: IndexJob) -> tuple[int, float]:
    """Order by completion time without tripping over None or naive datetimes."""
    completed = getattr(job, "completed_at", None)
    if completed is None:
        return (0, 0.0)
    try:
        return (1, completed.timestamp())
    except (AttributeError, OSError, OverflowError, ValueError):
        return (0, 0.0)


async def build_docs_libraries_payload(
    *,
    list_libraries_fn=db_list_libraries,
    list_jobs_fn=list_jobs,
    get_scheduler_status_fn=get_scheduler_status,
    get_unreachable_libraries_fn=get_unreachable_libraries,
    is_worker_running_fn=is_worker_running,
    get_worker_id_fn=get_worker_id,
    get_health_status_fn=_default_get_health_status,
    failed_job_limit: int = 10,
) -> dict[str, object]:
    """Build the documentation-library index served at ``/docs/libraries``.

    One read that answers "what is indexed, how fresh is it, and what is the
    queue doing" -- the state that was previously only reachable through raw
    SQL, ``/readyz``, or the MCP tool.
    """
    libraries_state = await list_libraries_fn()
    payload = build_library_inventory_payload(libraries_state)

    try:
        all_jobs = await list_jobs_fn(limit=1000)
        pending_jobs = [serialize_job(job) for job in all_jobs if job.status == JobStatus.PENDING]
        running_jobs = [serialize_job(job) for job in all_jobs if job.status == JobStatus.RUNNING]
        failed_jobs = sorted(
            (job for job in all_jobs if job.status == JobStatus.FAILED),
            key=_job_completion_sort_key,
            reverse=True,
        )
        recent_failed = [serialize_job(job) for job in failed_jobs[:failed_job_limit]]
    except Exception as exc:
        logger.warning("Could not list docs jobs for the library index: %s", exc)
        pending_jobs = []
        running_jobs = []
        recent_failed = []

    try:
        health, _status_code = await get_health_status_fn()
        health_status = str(health.get("status", "unknown"))
        runtime = dict(health.get("runtime") or {})
        dependencies = {
            key: str(health.get(key, "starting")) for key in ("qdrant", "tei", "database")
        }
    except Exception as exc:
        logger.warning("Could not read docs health for the library index: %s", exc)
        health_status = "unknown"
        runtime = {}
        dependencies = {"qdrant": "unknown", "tei": "unknown", "database": "unknown"}

    payload.update(
        {
            "scheduler": get_scheduler_status_fn(),
            "unreachable_libraries": dict(get_unreachable_libraries_fn()),
            "worker": {"running": is_worker_running_fn(), "id": get_worker_id_fn()},
            "runtime": runtime,
            "dependencies": dependencies,
            "health_status": health_status,
            "jobs": {
                "pending": pending_jobs,
                "running": running_jobs,
                "recent_failed": recent_failed,
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }
    )
    return payload


async def build_docs_stats_payload(
    *,
    list_libraries_fn=db_list_libraries,
    list_jobs_fn=list_jobs,
    get_chunk_count_fn=None,
    get_settings_fn=get_settings,
    is_scheduler_running_fn=is_scheduler_running,
    get_scheduler_status_fn=get_scheduler_status,
    is_worker_running_fn=is_worker_running,
    get_worker_id_fn=get_worker_id,
    is_recovery_running_fn=is_recovery_running,
    get_docs_graph_sync_status_fn=get_docs_graph_sync_status,
    get_cache_stats_fn=get_cache_stats,
) -> dict[str, object]:
    """Build the full docs admin stats payload."""
    _ = get_chunk_count_fn
    libraries_state = await list_libraries_fn()
    inventory_payload = {
        "summary": summarize_libraries(libraries_state),
        "inventory": {
            "total_files": sum(
                int(lib.file_count or 0)
                for lib in libraries_state
                if lib.status == LibraryStatus.READY
            ),
            "total_chunks": sum(
                int(lib.chunk_count or 0)
                for lib in libraries_state
                if lib.status == LibraryStatus.READY
            ),
            "git_libraries": sum(
                1 for lib in libraries_state if lib.source_type == LibrarySourceType.GIT
            ),
            "snapshot_libraries": sum(
                1 for lib in libraries_state if lib.source_type == LibrarySourceType.SNAPSHOT
            ),
        },
        "libraries": [serialize_library_state(lib) for lib in libraries_state],
    }
    summary = dict(inventory_payload["summary"])
    summary["total"] = summary["total_libraries"]
    summary["ready"] = summary["ready_libraries"]
    summary["pending"] = summary["pending_libraries"]
    summary["indexing"] = summary["indexing_libraries"]
    summary["error"] = summary["error_libraries"]
    inventory = inventory_payload["inventory"]
    libraries = inventory_payload["libraries"]

    try:
        all_jobs = await list_jobs_fn(limit=1000)
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        pending_jobs = sum(1 for job in all_jobs if job.status == JobStatus.PENDING)
        running_jobs = sum(1 for job in all_jobs if job.status == JobStatus.RUNNING)
        completed_today = sum(
            1
            for job in all_jobs
            if job.status == JobStatus.COMPLETED
            and job.completed_at
            and job.completed_at >= today_start
        )
        failed_today = sum(
            1
            for job in all_jobs
            if job.status == JobStatus.FAILED
            and job.completed_at
            and job.completed_at >= today_start
        )
    except Exception as exc:
        logger.warning("Could not get docs job counts: %s", exc)
        pending_jobs = 0
        running_jobs = 0
        completed_today = 0
        failed_today = 0

    settings = get_settings_fn()
    check_interval = getattr(settings, "freshness_check_interval", 6 * 60 * 60)

    return {
        "libraries": summary,
        "inventory": inventory,
        "libraryList": libraries,
        "scheduler": {
            "running": is_scheduler_running_fn(),
            "checkIntervalHours": check_interval // 3600,
        },
        "schedulerStatus": get_scheduler_status_fn(),
        "jobs": {
            "pending": pending_jobs,
            "running": running_jobs,
            "completedToday": completed_today,
            "failedToday": failed_today,
        },
        "worker": {"running": is_worker_running_fn(), "id": get_worker_id_fn()},
        "recovery": {"running": is_recovery_running_fn()},
        "docsGraph": get_docs_graph_sync_status_fn(),
        "embeddingCache": get_cache_stats_fn(),
    }
