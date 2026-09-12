"""Internal HTTP worker for RepoWise documentation.

This process owns the documentation lane end to end: reads, the four library
mutations, and the background runtime (job worker, freshness scheduler,
recovery loop). There is no separate writer to forward to and no proxy mode;
``/docs/action/index`` enqueues work here, behind the fail-closed mutation
auth in :mod:`repowise.docs.transport.auth`.

Exactly one process may own the background runtime. That is enforced by a
Postgres advisory lock in :mod:`repowise.docs.jobs.single_writer`, not by
configuration, so a second replica degrades to reads instead of double-running
the writer loops.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from repowise.docs.config import Settings, get_settings
from repowise.docs.runtime import close_docs_runtime, ensure_docs_runtime
from repowise.docs.server import routes as admin_routes
from repowise.docs.server.executor import DocToolExecutor
from repowise.docs.server.health import get_health_status
from repowise.docs.server.mcp_tools import MUTATION_TOOL_NAMES
from repowise.docs.transport.auth import authorize_mutation_request

logger = logging.getLogger("doc-search-http")

#: Reported wherever the old reader/proxy split used to surface a mode.
RUNTIME_MODE = "native"


def create_app(*, settings: Settings | Any | None = None) -> Starlette:
    """Build RepoWise's internal docs worker; public MCP belongs to RepoWise."""
    resolved_settings = settings or get_settings()
    executor = DocToolExecutor(settings=resolved_settings)

    async def livez(_request: Request) -> Response:
        return JSONResponse({"status": "ok", "service": "repowise-docs"})

    async def operational_health(_request: Request) -> Response:
        health, status_code = await get_health_status(require_background=True)
        payload = dict(health)
        payload.update(
            {
                "status": "ok" if status_code == 200 else "degraded",
                "service": "repowise-docs",
                "runtime_mode": RUNTIME_MODE,
                # Retained as a stable key for operators and dashboards that
                # curl this endpoint: the writer is this process. "native" is
                # deliberately neither "ok" nor "error" so a consumer still
                # comparing against the proxy-era values fails loudly.
                "writer": {"status": RUNTIME_MODE},
                "mutation_tools": {
                    "mode": RUNTIME_MODE,
                    "tools": sorted(MUTATION_TOOL_NAMES),
                },
            }
        )
        return JSONResponse(payload, status_code=status_code)

    async def stats(request: Request) -> Response:
        response = await admin_routes.stats_handler(request)
        payload = json.loads(response.body)
        payload["runtimeMode"] = RUNTIME_MODE
        payload["writer"] = {"status": RUNTIME_MODE}
        return JSONResponse(payload, status_code=response.status_code)

    async def index(request: Request) -> Response:
        """Enqueue an indexing job -- state-changing, so authorize first."""
        refusal = authorize_mutation_request(request, resolved_settings)
        if refusal is not None:
            return refusal
        return await admin_routes.index_handler(request)

    async def tool(request: Request) -> Response:
        """Run the same validated executor for the MCP and dashboard clients."""
        name = request.path_params["name"]
        if name not in executor.schemas:
            return JSONResponse({"error": "Unknown documentation tool"}, status_code=404)
        if name in MUTATION_TOOL_NAMES or name == "export_doc_bundle":
            refusal = authorize_mutation_request(request, resolved_settings)
            if refusal is not None:
                return refusal
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            return JSONResponse({"error": "Expected application/json"}, status_code=415)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                return JSONResponse({"error": "Tool arguments exceed 64 KiB"}, status_code=413)
        try:
            arguments = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return JSONResponse({"error": "Invalid JSON arguments"}, status_code=400)
        result = await executor.execute(name, arguments)
        content = "\n".join(block.text for block in result)
        try:
            payload = json.loads(content)
        except ValueError:
            payload = {"output": "markdown", "content": content}
        return JSONResponse(payload)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        await ensure_docs_runtime(start_background=True)
        try:
            app.state.runtime_mode = RUNTIME_MODE
            yield
        finally:
            await close_docs_runtime()

    routes = [
        Route("/livez", livez, methods=["GET"]),
        Route("/readyz", operational_health, methods=["GET"]),
        Route("/health", operational_health, methods=["GET"]),
        Route("/docs/health", operational_health, methods=["GET"]),
        Route("/docs/stats", stats, methods=["GET"]),
        Route("/docs/libraries", admin_routes.libraries_handler, methods=["GET"]),
        Route("/docs/tools/{name}", tool, methods=["POST"]),
        Route("/docs/action/index", index, methods=["POST"]),
        Route("/docs/action/jobs", admin_routes.list_jobs_handler, methods=["GET"]),
        Route(
            "/docs/action/jobs/{job_id}",
            admin_routes.get_job_handler,
            methods=["GET"],
        ),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    return app


def _log_mutation_auth_policy(settings: Settings | Any) -> None:
    """State the mutation-auth posture once at startup, loudly if it is open."""
    if getattr(settings, "webhook_token", ""):
        logger.info("Docs action endpoints require the X-Doc-Search-Token header.")
    elif getattr(settings, "allow_unauthenticated", False):
        logger.warning(
            "Docs action endpoints accept unauthenticated mutations "
            "(DOC_SEARCH_ALLOW_UNAUTHENTICATED set). Only safe on a trusted, "
            "non-exposed network. Set DOC_SEARCH_WEBHOOK_TOKEN to lock down."
        )
    else:
        logger.warning(
            "No DOC_SEARCH_WEBHOOK_TOKEN set: /docs/action/index will reject "
            "requests (fail closed). Set DOC_SEARCH_WEBHOOK_TOKEN, or "
            "DOC_SEARCH_ALLOW_UNAUTHENTICATED=1 on a trusted network."
        )


def main() -> None:
    """Run the documentation host."""
    settings = get_settings()
    logger.info("Starting docs HTTP server on %s:%d", settings.host, settings.port)
    _log_mutation_auth_policy(settings)
    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


__all__ = ["RUNTIME_MODE", "create_app", "main"]


if __name__ == "__main__":
    main()
