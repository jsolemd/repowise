"""HTTP admin route handlers for the integrated documentation-search subsystem."""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from repowise.docs.config import get_settings
from repowise.docs.db import get_job, list_jobs
from repowise.docs.db import list_libraries as db_list_libraries
from repowise.docs.graph_status import get_docs_graph_sync_status
from repowise.docs.indexer import get_chunk_count
from repowise.docs.indexer.embedder import get_cache_stats
from repowise.docs.jobs import (
    JobStatus,
    JobType,
    enqueue_job,
    get_scheduler_status,
    get_unreachable_libraries,
    get_worker_id,
    is_recovery_running,
    is_scheduler_running,
    is_worker_running,
)
from repowise.docs.library.models import JobEnqueueDisposition
from repowise.docs.server.health import get_health_status
from repowise.docs.server.stats import (
    build_docs_libraries_payload,
    build_docs_stats_payload,
    serialize_job,
)

logger = logging.getLogger("doc-search")


async def index_handler(request: Request) -> Response:
    library_id = request.query_params.get("library")
    if not library_id:
        return JSONResponse(
            {"error": "Missing required 'library' query parameter"}, status_code=400
        )

    type_str = request.query_params.get("type", "incremental").lower()
    try:
        job_type = JobType(type_str)
    except ValueError:
        return JSONResponse(
            {"error": f"Invalid job type: {type_str}. Use: incremental, full, force"},
            status_code=400,
        )

    priority_str = request.query_params.get("priority", "5")
    try:
        priority = int(priority_str)
        if not 1 <= priority <= 10:
            raise ValueError("Priority must be 1-10")
    except ValueError as exc:
        return JSONResponse({"error": f"Invalid priority: {exc}"}, status_code=400)

    result = await enqueue_job(library_id, job_type, priority)
    if result.disposition == JobEnqueueDisposition.QUEUED:
        logger.info(
            "Enqueued job %s for %s (type=%s)",
            result.job.id,
            library_id,
            result.job.job_type.value,
        )
        return JSONResponse(
            {
                "job_id": result.job.id,
                "library_id": library_id,
                "job_type": result.job.job_type.value,
                "priority": result.job.priority,
                "status": result.disposition.value,
            },
            status_code=201,
        )

    logger.info(
        "Job %s for %s (%s)",
        result.disposition.value,
        library_id,
        result.job.id,
    )
    return JSONResponse(
        {
            "job_id": result.job.id,
            "library_id": library_id,
            "job_type": result.job.job_type.value,
            "priority": result.job.priority,
            "status": result.disposition.value,
        },
        status_code=200,
    )


async def list_jobs_handler(request: Request) -> Response:
    library_id = request.query_params.get("library")

    status_str = request.query_params.get("status")
    status = None
    if status_str:
        try:
            status = JobStatus(status_str.lower())
        except ValueError:
            return JSONResponse(
                {
                    "error": f"Invalid status: {status_str}. Use: pending, running, completed, failed"
                },
                status_code=400,
            )

    limit_str = request.query_params.get("limit", "100")
    try:
        limit = int(limit_str)
        if not 1 <= limit <= 1000:
            raise ValueError("Limit must be 1-1000")
    except ValueError as exc:
        return JSONResponse({"error": f"Invalid limit: {exc}"}, status_code=400)

    jobs = await list_jobs(library_id=library_id, status=status, limit=limit)
    return JSONResponse(
        {
            "jobs": [
                {
                    "id": job.id,
                    "library_id": job.library_id,
                    "job_type": job.job_type.value,
                    "priority": job.priority,
                    "status": job.status.value,
                    "worker_id": job.worker_id,
                    "files_processed": job.files_processed,
                    "files_total": job.files_total,
                    "error_message": job.error_message,
                    "created_at": job.created_at.isoformat() if job.created_at else None,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                }
                for job in jobs
            ],
            "count": len(jobs),
            "worker": {"running": is_worker_running(), "id": get_worker_id()},
            "scheduler": {"running": is_scheduler_running()},
        }
    )


async def get_job_handler(request: Request) -> Response:
    job_id = request.path_params["job_id"]
    job = await get_job(job_id)
    if not job:
        return JSONResponse({"error": f"Job not found: {job_id}"}, status_code=404)

    return JSONResponse(serialize_job(job))


async def stats_handler(request: Request) -> Response:
    return JSONResponse(
        await build_docs_stats_payload(
            list_libraries_fn=db_list_libraries,
            list_jobs_fn=list_jobs,
            get_chunk_count_fn=get_chunk_count,
            get_settings_fn=get_settings,
            is_scheduler_running_fn=is_scheduler_running,
            get_scheduler_status_fn=get_scheduler_status,
            is_worker_running_fn=is_worker_running,
            get_worker_id_fn=get_worker_id,
            is_recovery_running_fn=is_recovery_running,
            get_docs_graph_sync_status_fn=get_docs_graph_sync_status,
            get_cache_stats_fn=get_cache_stats,
        )
    )


async def libraries_handler(request: Request) -> Response:
    """Serve the documentation-library index: registry, freshness, and queue."""
    return JSONResponse(
        await build_docs_libraries_payload(
            list_libraries_fn=db_list_libraries,
            list_jobs_fn=list_jobs,
            get_scheduler_status_fn=get_scheduler_status,
            get_unreachable_libraries_fn=get_unreachable_libraries,
            is_worker_running_fn=is_worker_running,
            get_worker_id_fn=get_worker_id,
            get_health_status_fn=get_health_status,
        )
    )


async def health_handler(request: Request) -> Response:
    health, status_code = await get_health_status()
    return JSONResponse(health, status_code=status_code)


def build_admin_routes() -> list[Route]:
    return [
        Route("/health", health_handler, methods=["GET"]),
        Route("/stats", stats_handler, methods=["GET"]),
        Route("/libraries", libraries_handler, methods=["GET"]),
        Route("/action/index", index_handler, methods=["POST"]),
        Route("/action/jobs", list_jobs_handler, methods=["GET"]),
        Route("/action/jobs/{job_id}", get_job_handler, methods=["GET"]),
    ]
