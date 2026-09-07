"""``/api/docs-libraries`` — the proxy in front of the documentation service.

The router owns no data, so the only things worth asserting are the two that
can break independently of the docs service: that a successful payload reaches
the caller unchanged, and that an unreachable service becomes a 502 with a
readable sentence rather than a 500 or a hung request.

The app here is built locally rather than through ``conftest``'s fixture: this
router needs no database, no FTS index and no vector store, and mounting it in
the shared factory would make every other server test carry it.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from repowise.server.routers import docs_libraries

#: One realistic response, trimmed to the fields the dashboard reads. Kept
#: verbatim in the assertion so a proxy that reshapes the body fails here.
_PAYLOAD = {
    "summary": {
        "status": "ok",
        "total_libraries": 2,
        "ready_libraries": 1,
        "pending_libraries": 1,
        "indexing_libraries": 0,
        "error_libraries": 0,
        "metadata_gap_libraries": 0,
        "graph_gap_libraries": 0,
    },
    "inventory": {
        "total_files": 120,
        "total_chunks": 4500,
        "git_libraries": 2,
        "snapshot_libraries": 0,
    },
    "libraries": [
        {
            "library_id": "react-hook-form",
            "name": "React Hook Form",
            "repo": "react-hook-form/react-hook-form",
            "status": "ready",
            "last_freshness_state": "unknown",
            "current_sha": "a1b2c3d4e5f6",
            "last_remote_sha": None,
        }
    ],
    "scheduler": {
        "running": True,
        "last_started_at": None,
        "last_completed_at": None,
        "summary": {},
    },
    "unreachable_libraries": {},
    "worker": {"running": True, "id": "worker-1"},
    "runtime": {"worker": "ok", "scheduler": "ok"},
    "dependencies": {"qdrant": "ok", "tei": "ok", "database": "ok"},
    "health_status": "ok",
    "jobs": {"pending": [], "running": [], "recent_failed": []},
    "generated_at": "2026-09-07T12:00:00+00:00",
}


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(docs_libraries.router)
    return app


async def _get(app: FastAPI) -> httpx.Response:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/api/docs-libraries")


class _StubClient:
    """Stands in for ``httpx.AsyncClient`` as an async context manager."""

    def __init__(self, handler) -> None:
        self._handler = handler

    async def __aenter__(self) -> _StubClient:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def get(self, url: str) -> httpx.Response:
        return self._handler(url)


@pytest.fixture
def docs_url(monkeypatch: pytest.MonkeyPatch) -> str:
    """Point the router at a stand-in host, proving the env var is honoured."""
    monkeypatch.setenv("REPOWISE_DOCS_URL", "http://docs.test:9999")
    return "http://docs.test:9999"


async def test_returns_payload_from_docs_service(
    monkeypatch: pytest.MonkeyPatch, docs_url: str
) -> None:
    requested: list[str] = []

    def handler(url: str) -> httpx.Response:
        requested.append(url)
        return httpx.Response(200, json=_PAYLOAD, request=httpx.Request("GET", url))

    monkeypatch.setattr(docs_libraries.httpx, "AsyncClient", lambda **_: _StubClient(handler))

    resp = await _get(_app())

    assert resp.status_code == 200
    assert resp.json() == _PAYLOAD
    assert requested == [f"{docs_url}/docs/libraries"]


async def test_docs_url_accepts_a_trailing_slash(monkeypatch, docs_url):
    monkeypatch.setenv("REPOWISE_DOCS_URL", f"{docs_url}/")

    def handler(url: str) -> httpx.Response:
        assert url == f"{docs_url}/docs/libraries"
        return httpx.Response(200, json=_PAYLOAD, request=httpx.Request("GET", url))

    monkeypatch.setattr(docs_libraries.httpx, "AsyncClient", lambda **_: _StubClient(handler))
    assert (await _get(_app())).status_code == 200


@pytest.mark.parametrize("body", ["<html>Unavailable</html>", "null", "[]"])
async def test_invalid_inventory_returns_502(monkeypatch, body):
    def handler(url: str) -> httpx.Response:
        return httpx.Response(200, text=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(docs_libraries.httpx, "AsyncClient", lambda **_: _StubClient(handler))
    response = await _get(_app())
    assert response.status_code == 502
    assert response.json()["detail"] == "The documentation service returned an invalid inventory."


async def test_timeout_returns_502_and_closes_client(monkeypatch):
    closed = []

    class TimeoutClient(_StubClient):
        async def __aexit__(self, *exc_info):
            closed.append(True)
            return False

    def handler(url: str) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=httpx.Request("GET", url))

    def client(**kwargs):
        assert kwargs["timeout"] == 10.0
        return TimeoutClient(handler)

    monkeypatch.setattr(docs_libraries.httpx, "AsyncClient", client)
    assert (await _get(_app())).status_code == 502
    assert closed == [True]


async def test_returns_502_when_docs_service_unreachable(
    monkeypatch: pytest.MonkeyPatch, docs_url: str
) -> None:
    def handler(url: str) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(docs_libraries.httpx, "AsyncClient", lambda **_: _StubClient(handler))

    resp = await _get(_app())

    assert resp.status_code == 502
    assert resp.json()["detail"] == "Couldn't reach the documentation service."


async def test_returns_502_when_docs_service_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 500 from the docs service is the same story to the reader as a dead port."""

    def handler(url: str) -> httpx.Response:
        return httpx.Response(500, text="boom", request=httpx.Request("GET", url))

    monkeypatch.setattr(docs_libraries.httpx, "AsyncClient", lambda **_: _StubClient(handler))

    resp = await _get(_app())

    assert resp.status_code == 502
