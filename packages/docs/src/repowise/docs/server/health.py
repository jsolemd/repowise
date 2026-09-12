"""Health checking helpers for the integrated documentation-search subsystem."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from repowise.docs.config import get_settings
from repowise.docs.db import get_connection
from repowise.docs.db import list_libraries as db_list_libraries
from repowise.docs.graph_status import get_docs_graph_sync_status
from repowise.docs.jobs import (
    get_scheduler_status,
    get_unreachable_libraries,
    is_recovery_running,
    is_scheduler_running,
    is_worker_running,
)
from repowise.docs.server.stats import summarize_libraries

logger = logging.getLogger("doc-search")


class HealthChecker:
    """Health checker for external dependencies."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._http_client: httpx.AsyncClient | None = None
        self._cached_health: dict[bool, dict] = {}
        self._last_health_check_monotonic: dict[bool, float] = {}
        self._health_cache_ttl_seconds = 5.0
        self._cached_dependency_health: dict[str, dict] | None = None
        self._last_dependency_check_monotonic = 0.0
        self._dependency_cache_ttl_seconds = 30.0

    async def initialize(self) -> None:
        """Initialize the health-check HTTP client."""
        self._http_client = httpx.AsyncClient(timeout=5.0)
        logger.info(
            "Doc-search server initializing with Qdrant=%s, TEI=%s, Supabase=%s",
            self.settings.qdrant_host,
            self.settings.tei_host,
            self.settings.supabase_url,
        )

    async def close(self) -> None:
        """Close the health-check HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def check_qdrant(self) -> dict:
        """Check Qdrant health."""
        if not self._http_client:
            return {"status": "error", "error": "HTTP client not initialized"}

        try:
            response = await self._http_client.get(f"{self.settings.qdrant_host}/readyz")
            if response.status_code == 200:
                return {"status": "ok"}
            return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def check_tei(self) -> dict:
        """Check TEI health."""
        if not self._http_client:
            return {"status": "error", "error": "HTTP client not initialized"}

        try:
            response = await self._http_client.get(f"{self.settings.tei_host}/health")
            if response.status_code == 200:
                return {"status": "ok"}
            return {"status": "error", "error": f"HTTP {response.status_code}"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def check_database(self) -> dict:
        """Check PostgreSQL availability for the docs job/control plane."""
        try:
            async with get_connection() as conn:
                value = await conn.fetchval("SELECT 1")
            if value == 1:
                return {"status": "ok"}
            return {"status": "error", "error": f"Unexpected result: {value!r}"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    async def get_dependency_health(self) -> dict[str, dict]:
        """Get just the external dependency health with a longer TTL cache."""
        now = time.monotonic()
        if (
            self._cached_dependency_health is not None
            and now - self._last_dependency_check_monotonic < self._dependency_cache_ttl_seconds
        ):
            return {key: dict(value) for key, value in self._cached_dependency_health.items()}

        qdrant, tei, database = await asyncio.gather(
            self.check_qdrant(),
            self.check_tei(),
            self.check_database(),
        )
        result = {
            "qdrant": qdrant,
            "tei": tei,
            "database": database,
        }
        self._cached_dependency_health = result
        self._last_dependency_check_monotonic = now
        return {key: dict(value) for key, value in result.items()}

    async def get_health(self, *, require_background: bool = True) -> dict:
        """Get combined health for the writer, or for a reads-only caller."""
        now = time.monotonic()
        if (
            require_background in self._cached_health
            and now - self._last_health_check_monotonic.get(require_background, 0.0)
            < self._health_cache_ttl_seconds
        ):
            return dict(self._cached_health[require_background])

        dependency_health = await self.get_dependency_health()
        qdrant = dependency_health["qdrant"]
        tei = dependency_health["tei"]
        database = dependency_health["database"]
        registry: dict[str, object]
        if database["status"] != "ok":
            registry = {"status": "error", "error": "database unavailable"}
            graph_gap_count = 0
        else:
            try:
                libraries = await db_list_libraries()
                registry = summarize_libraries(libraries)
                graph_gap_count = int(registry.get("graph_gap_libraries") or 0)
            except Exception as exc:
                registry = {"status": "error", "error": str(exc)}
                graph_gap_count = 0
        docs_graph = get_docs_graph_sync_status()
        if require_background:
            runtime = {
                "worker": "ok" if is_worker_running() else "stopped",
                "scheduler": "ok" if is_scheduler_running() else "stopped",
                "recovery": "ok" if is_recovery_running() else "stopped",
            }
            runtime_ok = all(value == "ok" for value in runtime.values())
        else:
            runtime = {
                "worker": "not_required",
                "scheduler": "not_required",
                "recovery": "not_required",
            }
            runtime_ok = True
        docs_graph["graph_gap_libraries"] = graph_gap_count
        docs_graph_ok = not docs_graph["enabled"] or (
            docs_graph["state"] in {"ok", "idle"} and graph_gap_count == 0
        )

        all_ok = (
            qdrant["status"] == "ok"
            and tei["status"] == "ok"
            and database["status"] == "ok"
            and registry["status"] == "ok"
            and runtime_ok
            and docs_graph_ok
        )

        result: dict = {
            "status": "ok" if all_ok else "degraded",
            "qdrant": qdrant["status"],
            "tei": tei["status"],
            "database": database["status"],
            "runtime_mode": "native" if require_background else "reads_only",
            "runtime": runtime,
            "details": {
                "qdrant": qdrant,
                "tei": tei,
                "database": database,
                "registry": registry,
                "docs_graph": docs_graph,
                "scheduler": get_scheduler_status(),
            },
        }

        unreachable = get_unreachable_libraries()
        if unreachable:
            result["unreachable_libraries"] = {
                lib_id: {"consecutive_failures": count} for lib_id, count in unreachable.items()
            }

        self._cached_health[require_background] = result
        self._last_health_check_monotonic[require_background] = now
        return result


_health_checker: HealthChecker | None = None


async def initialize_health_checker() -> None:
    """Create and initialize the global health checker."""
    global _health_checker
    _health_checker = HealthChecker()
    await _health_checker.initialize()


async def close_health_checker() -> None:
    """Close and clear the global health checker."""
    global _health_checker
    if _health_checker is not None:
        await _health_checker.close()
        _health_checker = None


async def get_health_status(*, require_background: bool = True) -> tuple[dict, int]:
    """Return the current health payload and matching HTTP status code."""
    if _health_checker is None:
        return {"status": "starting"}, 503

    health = await _health_checker.get_health(require_background=require_background)
    status_code = 200 if health["status"] == "ok" else 503
    return health, status_code


async def get_dependency_status() -> dict[str, str]:
    """Return dependency-only statuses for low-churn readiness checks."""
    if _health_checker is None:
        return {"qdrant": "starting", "tei": "starting", "database": "starting"}

    health = await _health_checker.get_dependency_health()
    return {key: value["status"] for key, value in health.items()}
