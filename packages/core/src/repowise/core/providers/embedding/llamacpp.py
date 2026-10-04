"""llama.cpp's ``llama-server`` as a repowise embedding provider.

The server speaks the OpenAI embeddings API at ``/v1/embeddings``. What this
adapter adds is the context limit. llama-server refuses an input longer than
the model's context rather than truncating it, and for an encoder such as
embeddinggemma that context is the 2,048 tokens it was trained on, which the
server will not raise; one refused input fails its whole batch. Ollama and
sentence-transformers keep an over-long input's first tokens, and so does this
adapter. It counts and cuts with the server's own tokenizer (``/tokenize`` and
``/detokenize``), so the cut matches the tokens the model will see rather than
a proxy tokenizer's estimate, and the server still wraps the kept text in the
model's special tokens.

A router-mode server serves several models and routes each request by its
``model`` field, so every call names the model, and the context is read from
that model's ``/props``.

A request the server cannot serve is retried briefly when the failure is the
kind that passes (a 503 while a model loads, a read timeout). A connection the
server does not accept is not retried: on loopback a live server accepts at
once, even mid-load, so a refused or slow connect means nothing is serving, and
the connect waits seconds rather than the full request timeout. If it still
fails, the host's ensure command is asked to bring the server back (see
:mod:`~repowise.core.providers.embedding.outage`), and the request is retried
once if it reports the server serving. Otherwise the call raises
:class:`~repowise.core.providers.embedding.outage.EmbedderUnavailableError`,
whose message says why semantic search is off and how to restore it, and the
current MCP request is marked keyword-only.
"""

from __future__ import annotations

import asyncio
import math
import os

import httpx
import structlog

from repowise.core.providers.embedding.base import resolve_embedding_timeout
from repowise.core.providers.embedding.ollama import _infer_dimensions
from repowise.core.providers.embedding.outage import (
    EmbedderUnavailableError,
    ensure_embedder,
    note_keyword_only,
)

log = structlog.get_logger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8080"  # llama-server's own default
_DEFAULT_MODEL = "embeddinggemma"
_DEFAULT_TIMEOUT = 60.0
#: Seconds to wait for the server to accept a connection, within the request
#: timeout. Past it the server is down or wedged, and recovery is the next step.
_CONNECT_TIMEOUT_S = 2.0
#: Tokens held back from the context: the special tokens the server adds around
#: every input (embeddinggemma adds two) and a few for a cut that re-tokenizes
#: slightly longer than the prefix it was taken from.
_CONTEXT_RESERVE = 8
#: Inputs tokenized at once while fitting a batch.
_TOKENIZE_CONCURRENCY = 16
#: Retries of a failure that passes on its own (a 503 while a model loads, a
#: read timeout), before the server is treated as down.
_TRANSIENT_RETRIES = 2
_RETRY_BACKOFF_S = 0.5


class LlamaCppEmbedder:
    """llama-server embedding adapter implementing the repowise Embedder protocol.

    Args:
        model: The model name the server routes on (a router preset's section,
            or the single model's alias). Falls back to
            ``REPOWISE_EMBEDDING_MODEL``, then ``embeddinggemma``.
        base_url: Server root, without ``/v1``. Falls back to
            ``LLAMACPP_BASE_URL``, then ``http://127.0.0.1:8080``.
        dimensions: Declared vector width, which the server's output must
            match. Falls back to ``REPOWISE_EMBEDDING_DIMS``, then to the width
            the model name implies. Never sent to the server, which returns its
            model's native width.
        timeout: Per-request timeout in seconds, or ``LLAMACPP_EMBEDDING_TIMEOUT``
            / ``REPOWISE_EMBEDDING_TIMEOUT``; the first request may wait for the
            server to load the model.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self._model = model or os.environ.get("REPOWISE_EMBEDDING_MODEL") or _DEFAULT_MODEL
        self._base_url = (
            base_url or os.environ.get("LLAMACPP_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        if dimensions is None:
            env = os.environ.get("REPOWISE_EMBEDDING_DIMS", "").strip()
            dimensions = int(env) if env.isdigit() and int(env) > 0 else None
        elif dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self._dimensions = dimensions or _infer_dimensions(self._model)
        self._timeout = resolve_embedding_timeout(
            timeout, _DEFAULT_TIMEOUT, provider_env="LLAMACPP_EMBEDDING_TIMEOUT"
        )
        self._context: int | None = None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def _token_budget(self, client: httpx.AsyncClient) -> int:
        """Tokens an input may hold: the model's context less the reserve."""
        if self._context is None:
            response = await client.get(f"{self._base_url}/props", params={"model": self._model})
            response.raise_for_status()
            self._context = int(response.json()["default_generation_settings"]["n_ctx"])
        return self._context - _CONTEXT_RESERVE

    async def _fit(self, client: httpx.AsyncClient, text: str, budget: int) -> tuple[str, int]:
        """*text* cut to *budget* tokens, and its token count before the cut.

        A token is at least one byte, so a text of no more bytes than the budget
        fits without asking the server; the count reported for it is 0.
        """
        if len(text.encode("utf-8")) <= budget:
            return text, 0
        response = await client.post(
            f"{self._base_url}/tokenize", json={"model": self._model, "content": text}
        )
        response.raise_for_status()
        tokens = response.json()["tokens"]
        if len(tokens) <= budget:
            return text, len(tokens)
        response = await client.post(
            f"{self._base_url}/detokenize",
            json={"model": self._model, "tokens": tokens[:budget]},
        )
        response.raise_for_status()
        return response.json()["content"], len(tokens)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch, cutting each input to the model's context first.

        Raises :class:`EmbedderUnavailableError` when the server cannot serve
        it after a retry and any recovery the host can make.
        """
        if not texts:
            return []
        try:
            return await self._embed_retrying(texts)
        except httpx.HTTPError as exc:
            if not _server_failure(exc):
                raise
            failure = exc
        verdict = await ensure_embedder()
        if verdict is not None and verdict.state == "serving":
            try:
                return await self._embed_retrying(texts)
            except httpx.HTTPError as exc:
                if not _server_failure(exc):
                    raise
                failure = exc
        error = EmbedderUnavailableError(_describe(failure), verdict)
        log.warning(
            "llamacpp_embed_unavailable",
            cause=error.cause,
            state=verdict.state if verdict else None,
            model=self._model,
        )
        note_keyword_only(error.warning)
        raise error from failure

    async def _embed_retrying(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(_TRANSIENT_RETRIES + 1):
            try:
                return await self._embed_once(texts)
            except httpx.HTTPError as exc:
                if attempt == _TRANSIENT_RETRIES or not _transient(exc):
                    raise
            await asyncio.sleep(_RETRY_BACKOFF_S * (attempt + 1))
        raise AssertionError("unreachable")  # pragma: no cover

    async def _embed_once(self, texts: list[str]) -> list[list[float]]:
        timeout = httpx.Timeout(self._timeout, connect=min(_CONNECT_TIMEOUT_S, self._timeout))
        async with httpx.AsyncClient(timeout=timeout) as client:
            budget = await self._token_budget(client)
            gate = asyncio.Semaphore(_TOKENIZE_CONCURRENCY)

            async def fit(text: str) -> tuple[str, int]:
                async with gate:
                    return await self._fit(client, text, budget)

            fitted = await asyncio.gather(*(fit(text) for text in texts))
            cut = [count for text, (kept, count) in zip(texts, fitted, strict=True) if kept != text]
            if cut:
                log.debug(
                    "llamacpp_embed_input_truncated",
                    truncated=len(cut),
                    of=len(texts),
                    budget=budget,
                    largest=max(cut),
                    model=self._model,
                )
            response = await client.post(
                f"{self._base_url}/v1/embeddings",
                json={"model": self._model, "input": [kept for kept, _ in fitted]},
            )
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda row: row["index"])

        if len(rows) != len(texts):
            raise ValueError(
                f"llama-server returned {len(rows)} embeddings for {len(texts)} inputs."
            )
        vectors = [[float(value) for value in row["embedding"]] for row in rows]
        widths = {len(vector) for vector in vectors}
        if widths != {self._dimensions}:
            actual = min(widths - {self._dimensions})
            raise ValueError(
                f"LlamaCppEmbedder declared {self._dimensions}-dimensional vectors but "
                f"model {self._model!r} returned {actual}. Set REPOWISE_EMBEDDING_DIMS={actual}."
            )
        return [_l2_normalize(vector) for vector in vectors]


def _transient(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 503
    # A connect that timed out found nothing serving; retrying it only delays recovery.
    return isinstance(exc, httpx.TimeoutException) and not isinstance(exc, httpx.ConnectTimeout)


def _server_failure(exc: httpx.HTTPError) -> bool:
    """A failure of the server rather than of the request: down, hung or erroring."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


def _describe(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} from {exc.request.url.path}"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector to unit length."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
