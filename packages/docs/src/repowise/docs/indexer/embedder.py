"""Doc-search embeddings from llama.cpp's ``llama-server`` serving EmbeddingGemma.

The server speaks the OpenAI embeddings API at ``/v1/embeddings`` and routes each
request by its ``model`` field. This client adds the two things the server leaves
to its caller. EmbeddingGemma was trained with a prompt naming each text's role,
``task: search result | query: ...`` for a query and ``title: ... | text: ...``
for a document, and retrieves worse without it. And the server refuses an input
longer than the model's context instead of truncating it, which fails the whole
batch, so an over-long document is cut first with the server's own tokenizer
(``/tokenize`` and ``/detokenize``), keeping its opening tokens.

On the host the worker reaches the server over TCP (``embedder_url``). In a
container it uses the Unix socket the server also listens on
(``embedder_socket``), so the server never listens beyond loopback.
"""

import asyncio
import logging
import math
import threading
from collections import OrderedDict
from typing import Any

import httpx

from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)

# EmbeddingGemma's retrieval prompts (its sentence-transformers configuration).
QUERY_PROMPT = "task: search result | query: "


def document_prompt(title: str | None) -> str:
    """The document-role prompt, naming the section the text sits under."""
    return f"title: {title or 'none'} | text: "


# Query embedding LRU cache: repeated queries skip the server round-trip.
_QUERY_CACHE_MAX_SIZE = 256
_query_cache: OrderedDict[str, list[float]] = OrderedDict()
_query_cache_lock = threading.Lock()
_query_cache_hits = 0
_query_cache_misses = 0
_http_client: httpx.AsyncClient | None = None
_http_client_lock = threading.Lock()
_token_budget: int | None = None

_HTTP_CONNECT_TIMEOUT_SECONDS = 5.0
# The first request after the server's idle sleep waits for the model to load.
_HTTP_READ_TIMEOUT_SECONDS = 60.0
_HTTP_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=100)
_EMBED_MAX_RETRIES = 2
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# Tokens held back from the context: the special tokens the server adds around
# every input, and a few for a cut that re-tokenizes slightly longer.
_CONTEXT_RESERVE = 8
_TOKENIZE_CONCURRENCY = 16


def _cache_get(text: str) -> list[float] | None:
    """Get embedding from cache, moving to end (most-recently-used)."""
    global _query_cache_hits
    with _query_cache_lock:
        if text in _query_cache:
            _query_cache.move_to_end(text)
            _query_cache_hits += 1
            return _query_cache[text]
        return None


def _cache_put(text: str, embedding: list[float]) -> None:
    """Store embedding in cache, evicting oldest if at capacity."""
    global _query_cache_misses
    with _query_cache_lock:
        _query_cache_misses += 1
        _query_cache[text] = embedding
        _query_cache.move_to_end(text)
        while len(_query_cache) > _QUERY_CACHE_MAX_SIZE:
            _query_cache.popitem(last=False)


def get_cache_stats() -> dict[str, int]:
    """Return query embedding cache statistics."""
    with _query_cache_lock:
        return {
            "size": len(_query_cache),
            "max_size": _QUERY_CACHE_MAX_SIZE,
            "hits": _query_cache_hits,
            "misses": _query_cache_misses,
        }


def _get_http_client() -> httpx.AsyncClient:
    """Return the process-local client for the embedding server."""
    global _http_client
    with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            settings = get_settings()
            transport = (
                httpx.AsyncHTTPTransport(uds=settings.embedder_socket, limits=_HTTP_LIMITS)
                if settings.embedder_socket
                else None
            )
            _http_client = httpx.AsyncClient(
                base_url=settings.embedder_url,
                transport=transport,
                timeout=httpx.Timeout(
                    connect=_HTTP_CONNECT_TIMEOUT_SECONDS,
                    read=_HTTP_READ_TIMEOUT_SECONDS,
                    write=_HTTP_READ_TIMEOUT_SECONDS,
                    pool=_HTTP_CONNECT_TIMEOUT_SECONDS,
                ),
                limits=_HTTP_LIMITS,
            )
        return _http_client


async def close_http_client() -> None:
    """Close the shared embedding client (controlled shutdown and tests)."""
    global _http_client
    with _http_client_lock:
        client = _http_client
        _http_client = None

    if client is not None and not client.is_closed:
        await client.aclose()


async def check_health() -> dict:
    """Report whether the embedding server answers (without loading the model)."""
    try:
        response = await _get_http_client().get("/health", timeout=5.0)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    if response.status_code == 200:
        return {"status": "ok"}
    return {"status": "error", "error": f"HTTP {response.status_code}"}


async def _request(method: str, path: str, **kwargs: Any) -> Any:
    """One server call, retried with light backoff on transient failures."""
    client = _get_http_client()
    last_exc: Exception | None = None

    for attempt in range(_EMBED_MAX_RETRIES + 1):
        try:
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if (
                exc.response.status_code not in _RETRYABLE_STATUS_CODES
                or attempt >= _EMBED_MAX_RETRIES
            ):
                raise
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            # HTTPX retires failed connections within its pool. Closing the
            # shared client here would also interrupt unrelated search/index
            # requests; the runtime owns closing it at shutdown.

        if attempt < _EMBED_MAX_RETRIES:
            await asyncio.sleep(min(0.2 * (2**attempt), 1.0))

    assert last_exc is not None
    raise last_exc


async def _get_token_budget(model: str) -> int:
    """Tokens one input may hold: the model's context less the reserve."""
    global _token_budget
    if _token_budget is None:
        props = await _request("GET", "/props", params={"model": model})
        _token_budget = int(props["default_generation_settings"]["n_ctx"]) - _CONTEXT_RESERVE
    return _token_budget


async def _fit(text: str, *, model: str, budget: int) -> str:
    """*text* cut to *budget* tokens. A token is at least one byte, so a text of
    no more bytes than the budget fits without asking the server."""
    if len(text.encode("utf-8")) <= budget:
        return text
    tokens = (await _request("POST", "/tokenize", json={"model": model, "content": text}))["tokens"]
    if len(tokens) <= budget:
        return text
    logger.debug("Cutting a %d-token input to %d tokens", len(tokens), budget)
    detokenized = await _request(
        "POST", "/detokenize", json={"model": model, "tokens": tokens[:budget]}
    )
    return detokenized["content"]


def _validate_embedding_response(
    payload: object, *, input_count: int, dimensions: int
) -> list[list[float]]:
    """Reject incomplete or malformed successful responses before publication."""
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != input_count:
        raise ValueError(f"llama-server must return one embedding for each of {input_count} inputs")
    try:
        vectors = [row["embedding"] for row in sorted(rows, key=lambda row: row["index"])]
    except (KeyError, TypeError) as exc:
        raise ValueError("llama-server returned a malformed embeddings response") from exc
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise ValueError(f"llama-server must return {dimensions}-dimensional embeddings")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("llama-server embeddings must contain finite numbers")
    return vectors


async def _embed(inputs: list[str], *, batch_size: int | None = None) -> list[list[float]]:
    """Embed prompted inputs in batches, each cut to the model's context."""
    if not inputs:
        return []

    settings = get_settings()
    batch_size = settings.embedding_batch_size if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive")
    model = settings.embedding_model
    budget = await _get_token_budget(model)
    gate = asyncio.Semaphore(_TOKENIZE_CONCURRENCY)

    async def fit(text: str) -> str:
        async with gate:
            return await _fit(text, model=model, budget=budget)

    embeddings: list[list[float]] = []
    for i in range(0, len(inputs), batch_size):
        batch = await asyncio.gather(*(fit(text) for text in inputs[i : i + batch_size]))
        payload = await _request("POST", "/v1/embeddings", json={"model": model, "input": batch})
        embeddings.extend(
            _validate_embedding_response(
                payload, input_count=len(batch), dimensions=settings.embedding_dimensions
            )
        )
    return embeddings


async def embed_documents(
    texts: list[str],
    titles: list[str | None] | None = None,
    *,
    batch_size: int | None = None,
) -> list[list[float]]:
    """Embed document chunks, each prompted with its section title when known."""
    if titles is None:
        titles = [None] * len(texts)
    if len(titles) != len(texts):
        raise ValueError("titles must match texts one for one")
    inputs = [document_prompt(title) + text for title, text in zip(titles, texts, strict=True)]
    embeddings = await _embed(inputs, batch_size=batch_size)
    logger.debug("Embedded %d chunks", len(texts))
    return embeddings


async def embed_query(query: str) -> list[float]:
    """Embed a search query, through an LRU cache of recent queries."""
    cached = _cache_get(query)
    if cached is not None:
        logger.debug("Embedding cache hit for: %s...", query[:50])
        return cached

    (embedding,) = await _embed([QUERY_PROMPT + query])
    _cache_put(query, embedding)
    return embedding
