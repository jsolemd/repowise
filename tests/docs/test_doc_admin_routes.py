"""Tests for docs admin HTTP routes mounted under the unified server."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from repowise.docs.library.models import (
    IndexJob,
    JobStatus,
    JobType,
    LibraryState,
    LibraryStatus,
)
from repowise.docs.server.routes import build_admin_routes


@pytest.fixture
def client() -> TestClient:
    return TestClient(Starlette(routes=build_admin_routes()), raise_server_exceptions=False)


def test_health_route_returns_current_health(monkeypatch, client: TestClient) -> None:
    monkeypatch.setattr(
        "repowise.docs.server.routes.get_health_status",
        AsyncMock(return_value=({"status": "ok", "database": "ok"}, 200)),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_index_route_requires_library_query_param(client: TestClient) -> None:
    response = client.post("/action/index")

    assert response.status_code == 400
    assert "library" in response.json()["error"]


def test_jobs_route_reports_runtime_state(monkeypatch, client: TestClient) -> None:
    monkeypatch.setattr("repowise.docs.server.routes.list_jobs", AsyncMock(return_value=[]))
    monkeypatch.setattr("repowise.docs.server.routes.is_worker_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.routes.get_worker_id", lambda: "worker-1")
    monkeypatch.setattr("repowise.docs.server.routes.is_scheduler_running", lambda: True)

    response = client.get("/action/jobs")

    assert response.status_code == 200
    assert response.json() == {
        "jobs": [],
        "count": 0,
        "worker": {"running": True, "id": "worker-1"},
        "scheduler": {"running": True},
    }


def test_stats_route_includes_healthful_runtime_summary(monkeypatch, client: TestClient) -> None:
    monkeypatch.setattr("repowise.docs.server.routes.db_list_libraries", AsyncMock(return_value=[]))
    monkeypatch.setattr("repowise.docs.server.routes.get_chunk_count", AsyncMock(return_value=0))
    monkeypatch.setattr("repowise.docs.server.routes.list_jobs", AsyncMock(return_value=[]))
    monkeypatch.setattr("repowise.docs.server.routes.is_scheduler_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.routes.is_worker_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.routes.get_worker_id", lambda: "worker-1")
    monkeypatch.setattr("repowise.docs.server.routes.is_recovery_running", lambda: True)
    monkeypatch.setattr(
        "repowise.docs.server.routes.get_docs_graph_sync_status",
        lambda: {"enabled": True, "state": "ok", "relay_running": True},
    )
    monkeypatch.setattr(
        "repowise.docs.server.routes.get_scheduler_status",
        lambda: {
            "running": True,
            "last_started_at": "2026-04-01T00:00:00+00:00",
            "last_completed_at": "2026-04-01T00:10:00+00:00",
            "summary": {"checked": 3, "enqueued": 1},
        },
    )
    monkeypatch.setattr(
        "repowise.docs.server.routes.get_cache_stats",
        lambda: {"size": 1, "hits": 2, "misses": 3, "max_size": 256},
    )
    monkeypatch.setattr(
        "repowise.docs.server.routes.get_settings",
        lambda: type("Settings", (), {"freshness_check_interval": 3600})(),
    )

    response = client.get("/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["libraries"]["total"] == 0
    assert payload["scheduler"] == {"running": True, "checkIntervalHours": 1}
    assert payload["schedulerStatus"] == {
        "running": True,
        "last_started_at": "2026-04-01T00:00:00+00:00",
        "last_completed_at": "2026-04-01T00:10:00+00:00",
        "summary": {"checked": 3, "enqueued": 1},
    }
    assert payload["worker"] == {"running": True, "id": "worker-1"}
    assert payload["recovery"] == {"running": True}
    assert payload["docsGraph"] == {"enabled": True, "state": "ok", "relay_running": True}


def _library(**overrides) -> LibraryState:
    """A ready git-backed library, overridable field by field."""
    base = {
        "library_id": "/react-hook-form/react-hook-form",
        "repo": "react-hook-form/react-hook-form",
        "name": "React Hook Form",
        "docs_path": "",
        "branch": "master",
        "priority": 6,
        "status": LibraryStatus.READY,
        "current_sha": "a" * 40,
        "indexed_at": datetime(2026, 4, 1, tzinfo=UTC),
        "freshness_checked_at": datetime(2026, 4, 2, tzinfo=UTC),
        "next_freshness_check_at": datetime(2026, 4, 3, tzinfo=UTC),
        "last_freshness_state": "fresh",
        "last_remote_sha": "b" * 40,
        "last_freshness_error": None,
        "file_count": 12,
        "chunk_count": 340,
    }
    base.update(overrides)
    return LibraryState(**base)


def _job(**overrides) -> IndexJob:
    base = {
        "id": "job-1",
        "library_id": "/react-hook-form/react-hook-form",
        "job_type": JobType.INCREMENTAL,
        "priority": 6,
        "status": JobStatus.PENDING,
    }
    base.update(overrides)
    return IndexJob(**base)


def _patch_libraries_route(
    monkeypatch,
    *,
    libraries: list[LibraryState],
    jobs: object = (),
    health: tuple[dict, int] = (
        {
            "status": "ok",
            "qdrant": "ok",
            "tei": "ok",
            "database": "ok",
            "runtime": {"worker": "ok", "scheduler": "ok", "recovery": "ok"},
        },
        200,
    ),
    unreachable: dict[str, int] | None = None,
) -> None:
    monkeypatch.setattr(
        "repowise.docs.server.routes.db_list_libraries", AsyncMock(return_value=libraries)
    )
    list_jobs_mock = jobs if isinstance(jobs, AsyncMock) else AsyncMock(return_value=list(jobs))
    monkeypatch.setattr("repowise.docs.server.routes.list_jobs", list_jobs_mock)
    monkeypatch.setattr(
        "repowise.docs.server.routes.get_health_status", AsyncMock(return_value=health)
    )
    monkeypatch.setattr(
        "repowise.docs.server.routes.get_unreachable_libraries", lambda: dict(unreachable or {})
    )
    monkeypatch.setattr("repowise.docs.server.routes.is_worker_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.routes.get_worker_id", lambda: "worker-1")
    monkeypatch.setattr(
        "repowise.docs.server.routes.get_scheduler_status",
        lambda: {
            "running": True,
            "last_started_at": "2026-04-01T00:00:00+00:00",
            "last_completed_at": "2026-04-01T00:10:00+00:00",
            "summary": {"checked": 3, "enqueued": 1},
        },
    )


def test_libraries_route_returns_every_freshness_field(monkeypatch, client: TestClient) -> None:
    """The dashboard reads freshness state that /docs/stats used to drop."""
    _patch_libraries_route(
        monkeypatch,
        libraries=[_library()],
        jobs=[
            _job(),
            _job(id="job-2", status=JobStatus.RUNNING, worker_id="worker-1"),
            _job(
                id="job-old",
                status=JobStatus.FAILED,
                error_message="boom",
                completed_at=datetime(2026, 3, 1, tzinfo=UTC),
            ),
            _job(
                id="job-new",
                status=JobStatus.FAILED,
                error_message="boom again",
                completed_at=datetime(2026, 3, 9, tzinfo=UTC),
            ),
        ],
        unreachable={"/dead/repo": 4},
    )

    response = client.get("/libraries")

    assert response.status_code == 200
    payload = response.json()
    library = payload["libraries"][0]
    assert library["priority"] == 6
    assert library["current_sha"] == "a" * 40
    assert library["last_remote_sha"] == "b" * 40
    assert library["freshness_checked_at"] == "2026-04-02T00:00:00+00:00"
    assert library["next_freshness_check_at"] == "2026-04-03T00:00:00+00:00"
    assert library["last_freshness_state"] == "fresh"
    assert library["last_freshness_error"] is None
    assert payload["summary"]["ready_libraries"] == 1
    assert payload["inventory"]["total_chunks"] == 340
    assert payload["scheduler"]["running"] is True
    assert payload["worker"] == {"running": True, "id": "worker-1"}
    assert payload["unreachable_libraries"] == {"/dead/repo": 4}
    assert payload["runtime"] == {"worker": "ok", "scheduler": "ok", "recovery": "ok"}
    assert payload["dependencies"] == {"qdrant": "ok", "tei": "ok", "database": "ok"}
    assert payload["health_status"] == "ok"
    assert [job["id"] for job in payload["jobs"]["pending"]] == ["job-1"]
    assert [job["id"] for job in payload["jobs"]["running"]] == ["job-2"]
    # Newest failure first, and the job payload carries the queue-timing fields.
    assert [job["id"] for job in payload["jobs"]["recent_failed"]] == ["job-new", "job-old"]
    assert payload["jobs"]["recent_failed"][0]["heartbeat_at"] is None
    assert payload["generated_at"].endswith("+00:00")


def test_libraries_route_surfaces_unknown_freshness_state(monkeypatch, client: TestClient) -> None:
    """'unknown' is the state that hid a five-months-stale library: never blank."""
    _patch_libraries_route(
        monkeypatch,
        libraries=[
            _library(
                last_freshness_state="unknown",
                last_freshness_error="Branch main not found",
                current_sha=None,
                freshness_checked_at=None,
                next_freshness_check_at=None,
            )
        ],
    )

    payload = client.get("/libraries").json()

    library = payload["libraries"][0]
    assert library["last_freshness_state"] == "unknown"
    assert library["last_freshness_error"] == "Branch main not found"
    assert library["current_sha"] is None
    assert library["freshness_checked_at"] is None
    assert library["next_freshness_check_at"] is None
    # A ready library with no sha is a metadata gap, not a healthy row.
    assert payload["summary"]["metadata_gap_libraries"] == 1


def test_libraries_route_degrades_when_job_listing_fails(monkeypatch, client: TestClient) -> None:
    """A broken queue read must not take the library index down with it."""
    _patch_libraries_route(
        monkeypatch,
        libraries=[_library()],
        jobs=AsyncMock(side_effect=RuntimeError("job table gone")),
        health=({"status": "starting"}, 503),
    )

    response = client.get("/libraries")

    assert response.status_code == 200
    payload = response.json()
    assert payload["jobs"] == {"pending": [], "running": [], "recent_failed": []}
    assert payload["libraries"][0]["name"] == "React Hook Form"
    assert payload["health_status"] == "starting"
    assert payload["runtime"] == {}
    assert payload["dependencies"] == {
        "qdrant": "starting",
        "tei": "starting",
        "database": "starting",
    }
