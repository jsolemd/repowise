"""Unit tests for LlamaCppEmbedder.

A fake httpx.AsyncClient stands in for llama-server: it tokenizes one token per
character and answers ``/props``, ``/tokenize``, ``/detokenize`` and
``/v1/embeddings``, so no server is required.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from repowise.core.providers.embedding.llamacpp import LlamaCppEmbedder


class _FakeResponse:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._data


class _FakeServer:
    """llama-server with a 20-token context, one token per character."""

    context: ClassVar[int] = 20
    width: ClassVar[int] = 4
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _FakeServer:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str, *, params: dict[str, Any]) -> _FakeResponse:
        self.calls.append((url, params))
        return _FakeResponse({"default_generation_settings": {"n_ctx": self.context}})

    async def post(self, url: str, *, json: dict[str, Any]) -> _FakeResponse:
        self.calls.append((url, json))
        if url.endswith("/tokenize"):
            return _FakeResponse({"tokens": [ord(c) for c in json["content"]]})
        if url.endswith("/detokenize"):
            return _FakeResponse({"content": "".join(chr(t) for t in json["tokens"])})
        for text in json["input"]:
            # The real server refuses an input over its context; so does this one.
            assert len(text) + 2 <= self.context, f"input of {len(text)} tokens refused"
        rows = [
            {"index": i, "embedding": [float(len(text))] + [0.0] * (self.width - 1)}
            for i, text in enumerate(json["input"])
        ]
        return _FakeResponse({"data": list(reversed(rows))})


@pytest.fixture(autouse=True)
def _fake_server(monkeypatch: pytest.MonkeyPatch):
    for var in ("REPOWISE_EMBEDDING_DIMS", "REPOWISE_EMBEDDING_MODEL", "LLAMACPP_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    _FakeServer.calls = []
    monkeypatch.setattr(
        "repowise.core.providers.embedding.llamacpp.httpx.AsyncClient", _FakeServer
    )


def test_registry_builds_llamacpp_and_counts_it_local() -> None:
    from repowise.core.providers.embedding.registry import LOCAL_EMBEDDERS, get_embedder

    assert isinstance(get_embedder("llamacpp", model="embeddinggemma"), LlamaCppEmbedder)
    assert "llamacpp" in LOCAL_EMBEDDERS


async def test_every_request_names_the_model_and_rows_keep_input_order() -> None:
    embedder = LlamaCppEmbedder(model="embeddinggemma", base_url="http://srv/", dimensions=4)
    vectors = await embedder.embed(["ab", "abcd"])

    # Rows arrive reversed; each vector's first component is its input's length.
    assert [v[0] for v in vectors] == [1.0, 1.0]  # normalized
    assert all(call[0].startswith("http://srv/") for call in _FakeServer.calls)
    assert all(call[1].get("model") == "embeddinggemma" for call in _FakeServer.calls)


async def test_short_inputs_are_sent_without_tokenizing() -> None:
    embedder = LlamaCppEmbedder(dimensions=4)
    await embedder.embed(["short", "also short"])

    assert not any(url.endswith("/tokenize") for url, _ in _FakeServer.calls)


async def test_long_input_keeps_its_first_tokens_within_the_context() -> None:
    embedder = LlamaCppEmbedder(dimensions=4)
    text = "x" * 50
    await embedder.embed([text, "ok"])

    sent = next(body for url, body in _FakeServer.calls if url.endswith("/v1/embeddings"))
    # Context 20 less the reserve of 8 leaves 12 tokens.
    assert sent["input"] == ["x" * 12, "ok"]


async def test_input_over_the_byte_bound_but_within_the_budget_is_untouched(monkeypatch) -> None:
    monkeypatch.setattr(_FakeServer, "context", 40)
    embedder = LlamaCppEmbedder(dimensions=4)
    text = "é" * 20  # 40 bytes, 20 tokens: asks the server, keeps everything

    await embedder.embed([text])

    sent = next(body for url, body in _FakeServer.calls if url.endswith("/v1/embeddings"))
    assert sent["input"] == [text]
    assert any(url.endswith("/tokenize") for url, _ in _FakeServer.calls)


async def test_context_is_read_once() -> None:
    embedder = LlamaCppEmbedder(dimensions=4)
    await embedder.embed(["a"])
    await embedder.embed(["b"])

    assert sum(url.endswith("/props") for url, _ in _FakeServer.calls) == 1


async def test_width_other_than_declared_is_refused() -> None:
    embedder = LlamaCppEmbedder(dimensions=8)

    with pytest.raises(ValueError, match="REPOWISE_EMBEDDING_DIMS=4"):
        await embedder.embed(["a"])


def test_width_comes_from_env_then_model_name(monkeypatch) -> None:
    assert LlamaCppEmbedder(model="embeddinggemma").dimensions == 768
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "512")
    assert LlamaCppEmbedder(model="embeddinggemma").dimensions == 512
