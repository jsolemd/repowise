from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from repowise.docs.indexer import embedder
from repowise.docs.library.models import LibraryStatus
from repowise.docs.server.health import HealthChecker


class _FakeAsyncClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, list[str]]]] = []
        self.is_closed = False

    async def post(self, url: str, json: dict[str, list[str]]) -> httpx.Response:
        self.calls.append((url, json))
        response = self._responses.pop(0)
        return response

    async def aclose(self) -> None:
        self.is_closed = True


@pytest.mark.asyncio
async def test_embed_texts_reuses_shared_client(monkeypatch):
    responses = [
        httpx.Response(
            200,
            json=[[0.1, 0.2]],
            request=httpx.Request("POST", "http://tei/embed"),
        ),
        httpx.Response(
            200,
            json=[[0.3, 0.4]],
            request=httpx.Request("POST", "http://tei/embed"),
        ),
    ]
    created: list[_FakeAsyncClient] = []

    def _client_factory(*args, **kwargs):
        client = _FakeAsyncClient(responses)
        created.append(client)
        return client

    monkeypatch.setattr(embedder.httpx, "AsyncClient", _client_factory)
    monkeypatch.setattr(
        embedder,
        "get_settings",
        lambda: SimpleNamespace(
            tei_host="http://tei", embedding_batch_size=32, chunk_max_tokens=800
        ),
    )

    await embedder.close_http_client()

    assert await embedder.embed_texts(["alpha"]) == [[0.1, 0.2]]
    assert await embedder.embed_texts(["beta"]) == [[0.3, 0.4]]
    assert len(created) == 1
    assert created[0].calls == [
        ("http://tei/embed", {"inputs": ["alpha"]}),
        ("http://tei/embed", {"inputs": ["beta"]}),
    ]

    await embedder.close_http_client()


@pytest.mark.asyncio
async def test_embed_texts_retries_transient_tei_errors(monkeypatch):
    responses = [
        httpx.Response(
            503,
            json={"error": "busy"},
            request=httpx.Request("POST", "http://tei/embed"),
        ),
        httpx.Response(
            200,
            json=[[0.5, 0.6]],
            request=httpx.Request("POST", "http://tei/embed"),
        ),
    ]
    created: list[_FakeAsyncClient] = []

    def _client_factory(*args, **kwargs):
        client = _FakeAsyncClient(responses)
        created.append(client)
        return client

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(embedder.httpx, "AsyncClient", _client_factory)
    monkeypatch.setattr(embedder.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(
        embedder,
        "get_settings",
        lambda: SimpleNamespace(
            tei_host="http://tei", embedding_batch_size=32, chunk_max_tokens=800
        ),
    )

    await embedder.close_http_client()

    assert await embedder.embed_texts(["gamma"]) == [[0.5, 0.6]]
    assert len(created) == 1
    assert len(created[0].calls) == 2

    await embedder.close_http_client()


@pytest.mark.asyncio
async def test_embed_texts_trims_oversized_inputs_before_post(monkeypatch):
    responses = [
        httpx.Response(
            200,
            json=[[0.1, 0.2]],
            request=httpx.Request("POST", "http://tei/embed"),
        ),
    ]
    created: list[_FakeAsyncClient] = []

    def _client_factory(*args, **kwargs):
        client = _FakeAsyncClient(responses)
        created.append(client)
        return client

    monkeypatch.setattr(embedder.httpx, "AsyncClient", _client_factory)
    monkeypatch.setattr(
        embedder,
        "get_settings",
        lambda: SimpleNamespace(tei_host="http://tei", embedding_batch_size=32, chunk_max_tokens=4),
    )

    await embedder.close_http_client()

    oversized = "a" * 600

    await embedder.embed_texts([oversized])

    assert len(created) == 1
    assert created[0].calls == [
        ("http://tei/embed", {"inputs": ["a" * 512]}),
    ]

    await embedder.close_http_client()


@pytest.mark.asyncio
async def test_health_checker_caches_dependency_checks(monkeypatch):
    checker = HealthChecker()
    checker._health_cache_ttl_seconds = 60.0

    qdrant = AsyncMock(return_value={"status": "ok"})
    tei = AsyncMock(return_value={"status": "ok"})
    database = AsyncMock(return_value={"status": "ok"})

    monkeypatch.setattr(checker, "check_qdrant", qdrant)
    monkeypatch.setattr(checker, "check_tei", tei)
    monkeypatch.setattr(checker, "check_database", database)
    monkeypatch.setattr("repowise.docs.server.health.db_list_libraries", AsyncMock(return_value=[]))
    monkeypatch.setattr("repowise.docs.server.health.is_worker_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.is_scheduler_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.is_recovery_running", lambda: True)
    monkeypatch.setattr(
        "repowise.docs.server.health.get_unreachable_libraries",
        lambda: {},
    )

    first = await checker.get_health()
    second = await checker.get_health()

    assert first == second
    assert qdrant.await_count == 1
    assert tei.await_count == 1
    assert database.await_count == 1


@pytest.mark.asyncio
async def test_health_checker_dependency_cache_is_reused_independently(monkeypatch):
    checker = HealthChecker()
    checker._dependency_cache_ttl_seconds = 60.0

    qdrant = AsyncMock(return_value={"status": "ok"})
    tei = AsyncMock(return_value={"status": "ok"})
    database = AsyncMock(return_value={"status": "ok"})

    monkeypatch.setattr(checker, "check_qdrant", qdrant)
    monkeypatch.setattr(checker, "check_tei", tei)
    monkeypatch.setattr(checker, "check_database", database)

    first = await checker.get_dependency_health()
    second = await checker.get_dependency_health()

    assert first == second
    assert qdrant.await_count == 1
    assert tei.await_count == 1
    assert database.await_count == 1


@pytest.mark.asyncio
async def test_health_checker_degrades_when_docs_registry_is_empty(monkeypatch):
    checker = HealthChecker()
    monkeypatch.setattr(checker, "check_qdrant", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(checker, "check_tei", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(checker, "check_database", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr("repowise.docs.server.health.db_list_libraries", AsyncMock(return_value=[]))
    monkeypatch.setattr("repowise.docs.server.health.is_worker_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.is_scheduler_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.is_recovery_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.get_unreachable_libraries", lambda: {})
    monkeypatch.setattr(
        "repowise.docs.server.health.get_docs_graph_sync_status",
        lambda: {"enabled": False, "state": "idle"},
    )

    health = await checker.get_health()

    assert health["status"] == "degraded"
    assert health["details"]["registry"] == {
        "status": "empty",
        "total_libraries": 0,
        "ready_libraries": 0,
        "pending_libraries": 0,
        "indexing_libraries": 0,
        "error_libraries": 0,
        "metadata_gap_libraries": 0,
        "graph_gap_libraries": 0,
    }


@pytest.mark.asyncio
async def test_health_checker_reports_ok_when_registry_has_ready_library(monkeypatch):
    checker = HealthChecker()
    monkeypatch.setattr(checker, "check_qdrant", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(checker, "check_tei", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(checker, "check_database", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(
        "repowise.docs.server.health.db_list_libraries",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    status=LibraryStatus.READY,
                    current_sha="sha",
                    file_count=3,
                    graph_synced_at=datetime.now(UTC),
                    graph_sync_error="",
                ),
                SimpleNamespace(
                    status=LibraryStatus.PENDING,
                    current_sha=None,
                    file_count=0,
                    graph_synced_at=None,
                    graph_sync_error=None,
                ),
            ]
        ),
    )
    monkeypatch.setattr("repowise.docs.server.health.is_worker_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.is_scheduler_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.is_recovery_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.get_unreachable_libraries", lambda: {})
    monkeypatch.setattr(
        "repowise.docs.server.health.get_docs_graph_sync_status",
        lambda: {"enabled": False, "state": "idle"},
    )

    health = await checker.get_health()

    assert health["status"] == "ok"
    assert health["details"]["registry"] == {
        "status": "ok",
        "total_libraries": 2,
        "ready_libraries": 1,
        "pending_libraries": 1,
        "indexing_libraries": 0,
        "error_libraries": 0,
        "metadata_gap_libraries": 0,
        "graph_gap_libraries": 0,
    }


@pytest.mark.asyncio
async def test_health_checker_degrades_when_docs_graph_has_ready_library_gaps(monkeypatch):
    checker = HealthChecker()
    monkeypatch.setattr(checker, "check_qdrant", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(checker, "check_tei", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(checker, "check_database", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(
        "repowise.docs.server.health.db_list_libraries",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    status=LibraryStatus.READY,
                    current_sha="sha",
                    file_count=4,
                    graph_synced_at=None,
                    graph_sync_error="graph metadata reconciliation pending",
                ),
            ]
        ),
    )
    monkeypatch.setattr("repowise.docs.server.health.is_worker_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.is_scheduler_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.is_recovery_running", lambda: True)
    monkeypatch.setattr("repowise.docs.server.health.get_unreachable_libraries", lambda: {})
    monkeypatch.setattr(
        "repowise.docs.server.health.get_docs_graph_sync_status",
        lambda: {"enabled": True, "state": "idle"},
    )

    health = await checker.get_health()

    assert health["status"] == "degraded"
    assert health["details"]["registry"]["graph_gap_libraries"] == 1
