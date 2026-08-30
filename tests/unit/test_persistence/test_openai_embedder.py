"""Unit tests for OpenAIEmbedder.

All tests mock openai.OpenAI — no real API calls are made.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai", reason="openai SDK not installed")

from repowise.core.providers.embedding.openai import OpenAIEmbedder, _token_counter_for
from repowise.core.source_search.chunks import TRUNCATION_MARKER

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key required"):
        OpenAIEmbedder(api_key=None)


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = OpenAIEmbedder()
    assert emb._api_key == "sk-test"


def test_dimensions_small():
    emb = OpenAIEmbedder(api_key="k", model="text-embedding-3-small")
    assert emb.dimensions == 1536


def test_dimensions_large():
    emb = OpenAIEmbedder(api_key="k", model="text-embedding-3-large")
    assert emb.dimensions == 3072


def test_dimensions_unknown_model_defaults_to_1536():
    emb = OpenAIEmbedder(api_key="k", model="some-future-model")
    assert emb.dimensions == 1536


def test_dimensions_explicit_override_wins_for_unknown_model():
    emb = OpenAIEmbedder(api_key="k", model="local-embedder", dimensions=2560)
    assert emb.dimensions == 2560


def test_dimensions_from_env(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "2560")
    emb = OpenAIEmbedder(api_key="k", model="local-embedder")
    assert emb.dimensions == 2560


def test_explicit_dimensions_beats_env(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "2560")
    emb = OpenAIEmbedder(api_key="k", model="local-embedder", dimensions=1024)
    assert emb.dimensions == 1024


def test_env_overrides_known_model_width(monkeypatch):
    # Precedence holds for a stock model too: the env wins over the table.
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "512")
    emb = OpenAIEmbedder(api_key="k", model="text-embedding-3-small")
    assert emb.dimensions == 512


@pytest.mark.parametrize("bad", [0, -5, True])
def test_invalid_dimensions_raises(bad):
    with pytest.raises(ValueError, match="dimensions must be a positive integer"):
        OpenAIEmbedder(api_key="k", model="local-embedder", dimensions=bad)


def test_malformed_env_raises_the_same_message(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "abc")
    with pytest.raises(ValueError, match="dimensions must be a positive integer"):
        OpenAIEmbedder(api_key="k", model="local-embedder")


def test_timeout_from_shared_env(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDING_TIMEOUT", "180")
    assert OpenAIEmbedder(api_key="k", model="local-embedder")._timeout == 180.0


def test_provider_env_beats_shared_env(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDING_TIMEOUT", "180")
    monkeypatch.setenv("OPENAI_EMBEDDING_TIMEOUT", "45")
    assert OpenAIEmbedder(api_key="k", model="local-embedder")._timeout == 45.0


def test_explicit_timeout_beats_env(monkeypatch):
    monkeypatch.setenv("REPOWISE_EMBEDDING_TIMEOUT", "180")
    assert OpenAIEmbedder(api_key="k", model="local-embedder", timeout=5)._timeout == 5.0


def test_hosted_default_is_unchanged(monkeypatch):
    # Pins the value, not the wiring: the env knob exists so this can stay put,
    # since it also bounds the retried query path.
    monkeypatch.delenv("REPOWISE_EMBEDDING_TIMEOUT", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_TIMEOUT", raising=False)
    assert OpenAIEmbedder(api_key="k")._timeout == 10.0


@pytest.mark.parametrize("bad", ["abc", "30s", "0", "-5", "inf", "nan"])
def test_a_malformed_env_value_falls_back_instead_of_breaking_the_run(monkeypatch, bad):
    # build_embedder turns any construction error into a keyless 8-wide store,
    # so raising here would make a typo silently destroy retrieval for setups
    # that worked before the variable was honoured at all.
    monkeypatch.setenv("REPOWISE_EMBEDDING_TIMEOUT", bad)
    assert OpenAIEmbedder(api_key="k", model="local-embedder")._timeout == 10.0


@pytest.mark.parametrize("bad", [0, -5, float("inf"), float("nan"), True, "30"])
def test_an_invalid_explicit_timeout_raises(bad):
    with pytest.raises(ValueError, match="timeout must be a positive number"):
        OpenAIEmbedder(api_key="k", model="local-embedder", timeout=bad)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _make_mock_embedding(values: list[float]) -> MagicMock:
    item = MagicMock()
    item.embedding = values
    return item


def _make_mock_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [_make_mock_embedding(v) for v in vectors]
    return response


async def test_embed_empty_returns_empty():
    emb = OpenAIEmbedder(api_key="k")
    result = await emb.embed([])
    assert result == []


async def test_embed_returns_normalized_vectors():
    # dimensions=3 so the declared width matches the 3-wide fake vector.
    raw = [1.0, 0.0, 0.0]
    emb = OpenAIEmbedder(api_key="k", dimensions=3)

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.return_value = _make_mock_response([raw])
        result = await emb.embed(["hello"])

    assert len(result) == 1
    norm = math.sqrt(sum(x * x for x in result[0]))
    assert abs(norm - 1.0) < 1e-6


async def test_embed_batch_returns_correct_count():
    # dimensions=2 so the declared width matches the 2-wide fake vectors.
    texts = ["a", "b", "c"]
    raw_vecs = [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]]
    emb = OpenAIEmbedder(api_key="k", dimensions=2)

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.return_value = _make_mock_response(raw_vecs)
        result = await emb.embed(texts)

    assert len(result) == 3


async def test_embed_passes_model_and_input():
    # dimensions=2 so the fake 2-wide response passes the width guard.
    emb = OpenAIEmbedder(api_key="k", model="text-embedding-3-large", dimensions=2)
    captured: list = []

    def fake_create(**kwargs):
        captured.append({"model": kwargs["model"], "input": kwargs["input"]})
        return _make_mock_response([[1.0, 0.0]])

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.side_effect = fake_create
        await emb.embed(["test text"])

    assert captured[0]["model"] == "text-embedding-3-large"
    assert captured[0]["input"] == ["test text"]


# ---------------------------------------------------------------------------
# Forwarding the width to the API
# ---------------------------------------------------------------------------


async def _capture_create_kwargs(emb: OpenAIEmbedder) -> dict:
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        # Return a vector of the declared width so the new width guard doesn't fire.
        return _make_mock_response([[1.0] + [0.0] * (emb.dimensions - 1)])

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.side_effect = fake_create
        await emb.embed(["text"])
    return captured


# The behaviour is model-agnostic: an override is declared *and* requested from
# the API for any model; the model names below are only illustrative inputs.
@pytest.mark.parametrize(
    "model",
    ["text-embedding-3-small", "text-embedding-ada-002", "local-embedder"],
)
async def test_overridden_width_is_declared_and_sent(monkeypatch, model):
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "512")
    emb = OpenAIEmbedder(api_key="k", model=model)
    assert emb.dimensions == 512
    kwargs = await _capture_create_kwargs(emb)
    assert kwargs["dimensions"] == 512


async def test_default_request_omits_dimensions():
    # No override → the request stays byte-identical: no dimensions key, so a
    # server that would reject the parameter is never sent it.
    kwargs = await _capture_create_kwargs(
        OpenAIEmbedder(api_key="k", model="text-embedding-3-small")
    )
    assert "dimensions" not in kwargs


# ---------------------------------------------------------------------------
# Width verification — the check added in fix/embedder-width-verification
# ---------------------------------------------------------------------------


async def test_embed_raises_when_api_returns_wrong_width_from_dims_table():
    # Declared via _DIMS (user never chose the number). The error message should
    # mention _DIMS so the user knows where to look.
    emb = OpenAIEmbedder(api_key="k", model="text-embedding-3-small")
    assert emb.dimensions == 1536  # from _DIMS table

    with patch("openai.OpenAI") as mock_client:
        # Fake API silently ignores 'dimensions' and returns its native 3-wide vector.
        mock_client.return_value.embeddings.create.return_value = _make_mock_response(
            [[1.0, 0.0, 0.0]]
        )
        with pytest.raises(ValueError, match="1536") as exc_info:
            await emb.embed(["hello"])

    msg = str(exc_info.value)
    assert "3" in msg  # actual width named
    assert "_DIMS" in msg  # points at the table, not the user


async def test_embed_raises_when_api_returns_wrong_width_with_user_override(monkeypatch):
    # Declared via REPOWISE_EMBEDDING_DIMS (user explicitly set it). The error
    # message should reference REPOWISE_EMBEDDING_DIMS so they know what to change.
    monkeypatch.setenv("REPOWISE_EMBEDDING_DIMS", "512")
    emb = OpenAIEmbedder(api_key="k", model="text-embedding-3-small")
    assert emb.dimensions == 512  # from env override

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.return_value = _make_mock_response(
            [[1.0, 0.0, 0.0]]
        )
        with pytest.raises(ValueError, match="512") as exc_info:
            await emb.embed(["hello"])

    msg = str(exc_info.value)
    assert "3" in msg
    assert "REPOWISE_EMBEDDING_DIMS" in msg


async def test_embed_width_check_is_skipped_for_empty_response():
    # embed([]) short-circuits before the API call and returns []; the guard
    # must not fire on the empty list itself.
    emb = OpenAIEmbedder(api_key="k")
    assert await emb.embed([]) == []


# ---------------------------------------------------------------------------
# Per-input cap (F34) — the request the endpoint would refuse
# ---------------------------------------------------------------------------
#
# The batch writers cap on the way in (``iter_embed_chunks``), but three paths
# reach an embedder without them: the single-item ``embed_and_upsert``, the
# source-search query leg, and the source indexer. OpenAI does not truncate an
# oversized input — it rejects the whole request, so one long page took its
# batch down with it and the loss showed up only as missing vectors.


async def _capture_input(emb: OpenAIEmbedder, texts: list[str]) -> list[str]:
    captured: dict = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        width = emb.dimensions
        return _make_mock_response([[1.0] + [0.0] * (width - 1) for _ in kwargs["input"]])

    with patch("openai.OpenAI") as mock_client:
        mock_client.return_value.embeddings.create.side_effect = fake_create
        await emb.embed(texts)
    return list(captured["input"])


async def test_an_oversized_input_is_shortened_before_the_http_call():
    from repowise.core.persistence.vector_store._base import EMBED_TEXT_MAX_TOKENS

    emb = OpenAIEmbedder(api_key="k", dimensions=2)
    # Ordinary prose/source, so the real tokenizer path runs: ~4 characters to
    # a token puts 200,000 characters an order of magnitude over the budget.
    text = "the quick brown fox jumps over the lazy dog " * 5000
    sent = await _capture_input(emb, [text])

    assert len(sent[0]) < len(text)
    counter = _token_counter_for(emb._model)
    assert counter is not None, "tiktoken should be resolvable in the test env"
    assert counter(sent[0]) <= EMBED_TEXT_MAX_TOKENS


async def test_a_text_within_the_cap_is_sent_byte_for_byte():
    emb = OpenAIEmbedder(api_key="k", dimensions=2)
    sent = await _capture_input(emb, ["hello", "world"])
    assert sent == ["hello", "world"]


async def test_a_token_dense_text_the_character_cap_would_pass_is_cut(monkeypatch):
    """The bug the character cap cannot see.

    10,000 characters is a third of ``EMBED_TEXT_MAX_CHARS`` and would sail
    through it. At four tokens to the character it is 40,000 tokens, which the
    endpoint refuses outright.
    """
    monkeypatch.setattr(
        "repowise.core.providers.embedding.openai._token_counter_for",
        lambda model: lambda text: len(text) * 4,
    )
    emb = OpenAIEmbedder(api_key="k", dimensions=2)
    text = "zz " * 3_334  # short words, so the quadratic-run guard stays out of it

    sent = await _capture_input(emb, [text])
    assert len(sent[0]) < len(text)
    body = sent[0].replace(TRUNCATION_MARKER, "")
    assert len(body) * 4 <= 8_000


async def test_a_long_unbroken_run_falls_back_to_the_character_cap(monkeypatch):
    """The tokenizer is quadratic in the length of one unbroken word.

    Measured on cl100k: a 30,000-character run costs 200 ms *per input, per
    request*. Text like that is generated data, and the character cap is what
    it gets instead — a cap that binds, at a cost that does not.
    """
    from repowise.core.persistence.vector_store._base import EMBED_TEXT_MAX_CHARS

    calls: list[int] = []

    def counted(model):
        def _count(text: str) -> int:
            calls.append(len(text))
            return len(text) // 4

        return _count

    monkeypatch.setattr("repowise.core.providers.embedding.openai._token_counter_for", counted)
    emb = OpenAIEmbedder(api_key="k", dimensions=2)

    sent = await _capture_input(emb, ["x" * 40_000])
    assert calls == [], "the tokenizer must not see a text it would choke on"
    assert len(sent[0]) == int(EMBED_TEXT_MAX_CHARS * 0.68) + len(TRUNCATION_MARKER) + int(
        EMBED_TEXT_MAX_CHARS * 0.28
    )


async def test_the_cut_is_reported_with_the_basis_that_measured_it(capsys):
    """Debug level, and structlog's own stream — hence capsys, not caplog.

    The basis is the part worth logging: "truncated" alone cannot distinguish
    a real overrun from a character cap that guessed wrong.
    """
    emb = OpenAIEmbedder(api_key="k", dimensions=2)
    text = "the quick brown fox jumps over the lazy dog " * 5000

    await _capture_input(emb, [text])

    out = capsys.readouterr().out
    assert "openai_embed_input_truncated" in out
    assert "cap_basis=tokens" in out
    assert "truncated=1" in out
