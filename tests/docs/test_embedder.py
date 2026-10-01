from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from repowise.docs.chunking.models import ChunkType, DocChunk
from repowise.docs.indexer import embedder, mutations
from repowise.docs.library.models import LibraryStatus
from repowise.docs.server.health import HealthChecker


def _settings(**overrides):
    values = {
        "embedder_url": "http://llama",
        "embedder_socket": "",
        "embedding_model": "embeddinggemma",
        "embedding_batch_size": 32,
        "embedding_dimensions": 2,
        "qdrant_collection": "docs",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _response(path: str, payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, content=json.dumps(payload), request=httpx.Request("POST", f"http://llama{path}")
    )


def _embeddings(*vectors: list[float]) -> httpx.Response:
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    return _response("/v1/embeddings", {"data": data})


class _FakeServer:
    """Answers /props itself and replays queued responses for every other call."""

    def __init__(self, responses: list[httpx.Response], *, n_ctx: int = 2048) -> None:
        self._responses = list(responses)
        self._n_ctx = n_ctx
        self.calls: list[tuple[str, dict]] = []
        self.is_closed = False

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        self.calls.append((path, kwargs))
        if path == "/props":
            return _response(path, {"default_generation_settings": {"n_ctx": self._n_ctx}})
        return self._responses.pop(0)

    async def aclose(self) -> None:
        self.is_closed = True

    def posted(self, path: str) -> list[dict]:
        return [kwargs["json"] for called, kwargs in self.calls if called == path]


@pytest.fixture(autouse=True)
async def _fresh_embedder(monkeypatch):
    monkeypatch.setattr(embedder, "_token_budget", None)
    monkeypatch.setattr(embedder, "_query_cache", embedder.OrderedDict())
    await embedder.close_http_client()
    yield
    await embedder.close_http_client()


def _install(monkeypatch, server: _FakeServer, **settings) -> list[dict]:
    created: list[dict] = []

    def _client_factory(*args, **kwargs):
        created.append(kwargs)
        return server

    monkeypatch.setattr(embedder.httpx, "AsyncClient", _client_factory)
    monkeypatch.setattr(embedder, "get_settings", lambda: _settings(**settings))
    return created


@pytest.mark.asyncio
async def test_roles_are_prompted_and_the_client_is_shared(monkeypatch):
    server = _FakeServer([_embeddings([0.1, 0.2]), _embeddings([0.3, 0.4])])
    created = _install(monkeypatch, server)

    assert await embedder.embed_documents(["alpha"], ["Guide > Install"]) == [[0.1, 0.2]]
    assert await embedder.embed_query("beta") == [0.3, 0.4]
    assert await embedder.embed_query("beta") == [0.3, 0.4]  # served from the cache

    assert len(created) == 1
    assert created[0]["base_url"] == "http://llama"
    assert created[0]["transport"] is None
    assert server.posted("/v1/embeddings") == [
        {"model": "embeddinggemma", "input": ["title: Guide > Install | text: alpha"]},
        {"model": "embeddinggemma", "input": ["task: search result | query: beta"]},
    ]
    assert [path for path, _ in server.calls].count("/props") == 1


@pytest.mark.asyncio
async def test_untitled_documents_use_the_none_title(monkeypatch):
    server = _FakeServer([_embeddings([0.1, 0.2])])
    _install(monkeypatch, server)

    await embedder.embed_documents(["alpha"])

    assert server.posted("/v1/embeddings")[0]["input"] == ["title: none | text: alpha"]


@pytest.mark.asyncio
async def test_socket_setting_routes_requests_through_the_unix_socket(monkeypatch):
    server = _FakeServer([])
    created = _install(monkeypatch, server, embedder_socket="/run/llama-server/llama-server.sock")
    transports: list[dict] = []
    monkeypatch.setattr(
        embedder.httpx, "AsyncHTTPTransport", lambda **kwargs: transports.append(kwargs) or "uds"
    )

    embedder._get_http_client()

    assert transports[0]["uds"] == "/run/llama-server/llama-server.sock"
    assert created[0]["transport"] == "uds"


@pytest.mark.asyncio
async def test_transient_server_errors_are_retried(monkeypatch):
    server = _FakeServer(
        [_response("/v1/embeddings", {"error": "loading"}, status=503), _embeddings([0.5, 0.6])]
    )
    _install(monkeypatch, server)
    monkeypatch.setattr(embedder.asyncio, "sleep", AsyncMock())

    assert await embedder.embed_documents(["gamma"]) == [[0.5, 0.6]]
    assert len(server.posted("/v1/embeddings")) == 2


@pytest.mark.asyncio
async def test_inputs_longer_than_the_context_are_cut_with_the_server_tokenizer(monkeypatch):
    # n_ctx 40 leaves a 32-token budget; the long input tokenizes to 50 tokens.
    long_text = "x" * 200
    server = _FakeServer(
        [
            _response("/tokenize", {"tokens": list(range(50))}),
            _response("/detokenize", {"content": "cut"}),
            _embeddings([0.1, 0.2], [0.3, 0.4]),
        ],
        n_ctx=40,
    )
    _install(monkeypatch, server)

    await embedder.embed_documents(["short", long_text])

    assert server.posted("/tokenize") == [
        {"model": "embeddinggemma", "content": "title: none | text: " + long_text}
    ]
    assert server.posted("/detokenize") == [{"model": "embeddinggemma", "tokens": list(range(32))}]
    assert server.posted("/v1/embeddings")[0]["input"] == ["title: none | text: short", "cut"]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [0, -1])
async def test_nonpositive_batch_size_is_rejected_before_http(monkeypatch, batch_size):
    server = _FakeServer([])
    _install(monkeypatch, server)
    with pytest.raises(ValueError, match="batch size must be positive"):
        await embedder.embed_documents(["alpha"], batch_size=batch_size)
    assert server.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"data": []},
        {"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        {"data": [{"index": i, "embedding": [0.1, 0.2]} for i in range(3)]},
        {"error": "bad response"},
        [[0.1, 0.2], [0.3, 0.4]],
        {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},
        {"data": [{"index": 0, "embedding": [0.1, 0.2]}, {"index": 0, "embedding": [0.3, 0.4]}]},
        {"data": [{"index": 1, "embedding": [0.1, 0.2]}, {"index": 2, "embedding": [0.3, 0.4]}]},
        {
            "data": [
                {"index": 0.0, "embedding": [0.1, 0.2]},
                {"index": 1.0, "embedding": [0.3, 0.4]},
            ]
        },
        {
            "data": [
                {"index": False, "embedding": [0.1, 0.2]},
                {"index": True, "embedding": [0.3, 0.4]},
            ]
        },
        {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [0.2]}]},
        {"data": [{"index": 0, "embedding": [True, 0.2]}, {"index": 1, "embedding": [0.3, 0.4]}]},
        {"data": [{"index": 0, "embedding": ["a", 0.2]}, {"index": 1, "embedding": [0.3, 0.4]}]},
        {
            "data": [
                {"index": 0, "embedding": [float("inf"), 0.2]},
                {"index": 1, "embedding": [0.3, 0.4]},
            ]
        },
    ],
)
async def test_invalid_embedding_batch_cannot_publish_partial_chunks(monkeypatch, payload):
    # Use the real embedding and mutation path so a provider mismatch cannot
    # be hidden by zip truncation before the pipeline stamps a file current.
    server = _FakeServer([_response("/v1/embeddings", payload)])
    _install(monkeypatch, server)
    monkeypatch.setattr(mutations, "get_settings", lambda: _settings())
    chunks = [
        DocChunk(
            library_id="/test/docs",
            file_path="api.md",
            commit_sha="test",
            chunk_type=ChunkType.DOC,
            content=content,
            breadcrumb=[],
            section_anchor=content,
            source_url="",
        )
        for content in ("first", "second")
    ]
    qdrant = MagicMock()

    with pytest.raises(ValueError, match="llama-server"):
        await mutations.upsert_chunks(chunks, qdrant)

    qdrant.upsert.assert_not_called()
    assert len(server.posted("/v1/embeddings")) == 1  # Malformed successes are not retried.


@pytest.mark.asyncio
async def test_embeddings_are_returned_in_input_order(monkeypatch):
    server = _FakeServer(
        [
            _response(
                "/v1/embeddings",
                {
                    "data": [
                        {"index": 1, "embedding": [0.3, 0.4]},
                        {"index": 0, "embedding": [0.1, 0.2]},
                    ]
                },
            )
        ]
    )
    _install(monkeypatch, server)

    assert await embedder.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_transport_retry_does_not_close_another_requests_client(monkeypatch):
    active = asyncio.Event()
    retry_finished = asyncio.Event()

    class ConcurrentClient:
        is_closed = False
        attempts = 0

        async def request(self, method, path, **kwargs):
            if path == "/props":
                return _response(path, {"default_generation_settings": {"n_ctx": 2048}})
            if kwargs["json"]["input"] == ["title: none | text: slow"]:
                active.set()
                await retry_finished.wait()
                assert not self.is_closed, "retry closed a concurrent request's pool"
            else:
                await active.wait()
                self.attempts += 1
                if self.attempts == 1:
                    raise httpx.ReadTimeout("transient read failure")
                retry_finished.set()
            return _embeddings([0.1, 0.2])

        async def aclose(self):
            self.is_closed = True

    client = ConcurrentClient()
    monkeypatch.setattr(embedder, "_http_client", client)
    monkeypatch.setattr(embedder, "_get_http_client", lambda: client)
    monkeypatch.setattr(embedder, "get_settings", lambda: _settings())
    monkeypatch.setattr(embedder.asyncio, "sleep", AsyncMock())

    assert await asyncio.gather(
        embedder.embed_documents(["slow"]), embedder.embed_documents(["retry"])
    ) == [[[0.1, 0.2]], [[0.1, 0.2]]]
    assert client.attempts == 2
    assert not client.is_closed
    await embedder.close_http_client()
    assert client.is_closed


@pytest.mark.asyncio
async def test_health_checker_caches_dependency_checks(monkeypatch):
    checker = HealthChecker()
    checker._health_cache_ttl_seconds = 60.0

    qdrant = AsyncMock(return_value={"status": "ok"})
    embedding = AsyncMock(return_value={"status": "ok"})
    database = AsyncMock(return_value={"status": "ok"})

    monkeypatch.setattr(checker, "check_qdrant", qdrant)
    monkeypatch.setattr(checker, "check_embedder", embedding)
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
    assert embedding.await_count == 1
    assert database.await_count == 1


@pytest.mark.asyncio
async def test_health_checker_dependency_cache_is_reused_independently(monkeypatch):
    checker = HealthChecker()
    checker._dependency_cache_ttl_seconds = 60.0

    qdrant = AsyncMock(return_value={"status": "ok"})
    embedding = AsyncMock(return_value={"status": "ok"})
    database = AsyncMock(return_value={"status": "ok"})

    monkeypatch.setattr(checker, "check_qdrant", qdrant)
    monkeypatch.setattr(checker, "check_embedder", embedding)
    monkeypatch.setattr(checker, "check_database", database)

    first = await checker.get_dependency_health()
    second = await checker.get_dependency_health()

    assert first == second
    assert qdrant.await_count == 1
    assert embedding.await_count == 1
    assert database.await_count == 1


@pytest.mark.asyncio
async def test_health_checker_degrades_when_docs_registry_is_empty(monkeypatch):
    checker = HealthChecker()
    monkeypatch.setattr(checker, "check_qdrant", AsyncMock(return_value={"status": "ok"}))
    monkeypatch.setattr(checker, "check_embedder", AsyncMock(return_value={"status": "ok"}))
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
    monkeypatch.setattr(checker, "check_embedder", AsyncMock(return_value={"status": "ok"}))
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
    monkeypatch.setattr(checker, "check_embedder", AsyncMock(return_value={"status": "ok"}))
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


@pytest.mark.asyncio
async def test_reembed_all_replaces_only_dense_vectors_page_by_page(monkeypatch):
    pages = [
        (
            [
                SimpleNamespace(id="a", payload={"content": "alpha", "breadcrumb_text": "Guide"}),
                SimpleNamespace(id="b", payload={"content": "beta", "breadcrumb_text": ""}),
            ],
            "next",
        ),
        ([SimpleNamespace(id="c", payload={"content": "gamma"})], None),
    ]
    qdrant = MagicMock()
    qdrant.scroll = AsyncMock(side_effect=pages)
    qdrant.update_vectors = AsyncMock()
    embed = AsyncMock(side_effect=[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6]]])
    monkeypatch.setattr(mutations, "embed_documents", embed)
    monkeypatch.setattr(mutations, "get_settings", lambda: _settings())

    assert await mutations.reembed_all(qdrant) == 3

    assert [call.args for call in embed.await_args_list] == [
        (["alpha", "beta"], ["Guide", None]),
        (["gamma"], [None]),
    ]
    assert [call.kwargs["offset"] for call in qdrant.scroll.await_args_list] == [None, "next"]
    updated = [
        (point.id, point.vector)
        for call in qdrant.update_vectors.await_args_list
        for point in call.kwargs["points"]
    ]
    assert updated == [
        ("a", {"dense": [0.1, 0.2]}),
        ("b", {"dense": [0.3, 0.4]}),
        ("c", {"dense": [0.5, 0.6]}),
    ]
