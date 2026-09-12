"""TEI batch embedding client for doc-search."""

import asyncio
import logging
import threading
from collections import OrderedDict

import httpx

from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)

# Query embedding LRU cache — avoids redundant TEI round-trips for repeated queries.
# Inspired by NornicDB's embedding cache pattern. Thread-safe via lock.
_QUERY_CACHE_MAX_SIZE = 256
_query_cache: OrderedDict[str, list[float]] = OrderedDict()
_query_cache_lock = threading.Lock()
_query_cache_hits = 0
_query_cache_misses = 0
_http_client: httpx.AsyncClient | None = None
_http_client_lock = threading.Lock()

_HTTP_CONNECT_TIMEOUT_SECONDS = 5.0
_HTTP_READ_TIMEOUT_SECONDS = 60.0
_HTTP_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=100)
_EMBED_MAX_RETRIES = 2
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _max_embed_chars(settings) -> int:
    """Return the per-input character budget used for TEI requests."""
    return max(512, settings.chunk_max_tokens * 4)


def _prepare_embed_text(text: str, *, max_chars: int) -> str:
    """Trim pathological inputs so one bad chunk cannot 413 the whole batch."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


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
    """Return a process-local AsyncClient for TEI connection reuse."""
    global _http_client
    with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            _http_client = httpx.AsyncClient(
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
    """Close the shared TEI client.

    Primarily useful for tests and controlled shutdown hooks.
    """
    global _http_client
    with _http_client_lock:
        client = _http_client
        _http_client = None

    if client is not None and not client.is_closed:
        await client.aclose()


async def _post_embed_batch(
    *,
    settings,
    batch: list[str],
) -> list[list[float]]:
    """Embed one batch with light retry/backoff for transient TEI failures."""
    client = _get_http_client()
    last_exc: Exception | None = None

    for attempt in range(_EMBED_MAX_RETRIES + 1):
        try:
            response = await client.post(
                f"{settings.tei_host}/embed",
                json={"inputs": batch},
            )
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
            await close_http_client()
            client = _get_http_client()

        if attempt < _EMBED_MAX_RETRIES:
            await asyncio.sleep(min(0.2 * (2**attempt), 1.0))

    assert last_exc is not None
    raise last_exc


async def embed_texts(
    texts: list[str],
    batch_size: int | None = None,
) -> list[list[float]]:
    """
    Batch embed texts via TEI (Text Embeddings Inference).

    Args:
        texts: List of text strings to embed
        batch_size: Optional batch size override (default from settings)

    Returns:
        List of embedding vectors (768 dimensions each)

    Raises:
        httpx.HTTPStatusError: If TEI returns an error
    """
    if not texts:
        return []

    settings = get_settings()
    batch_size = batch_size or settings.embedding_batch_size
    max_chars = _max_embed_chars(settings)

    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [
            _prepare_embed_text(text, max_chars=max_chars) for text in texts[i : i + batch_size]
        ]

        logger.debug(
            "Embedding batch %d (%d texts, chars: %d)",
            i // batch_size + 1,
            len(batch),
            sum(len(text) for text in batch),
        )

        batch_embeddings = await _post_embed_batch(
            settings=settings,
            batch=batch,
        )
        embeddings.extend(batch_embeddings)

    logger.info(
        "Embedded %d texts in %d batches",
        len(texts),
        (len(texts) - 1) // batch_size + 1,
    )

    return embeddings


async def embed_single(text: str) -> list[float]:
    """
    Embed a single text, with LRU cache for query embeddings.

    Caches up to 256 recent query embeddings to avoid redundant TEI calls.
    Cache hits return in <1ms vs ~5-10ms for a TEI round-trip.

    Args:
        text: Text string to embed

    Returns:
        Embedding vector (768 dimensions)
    """
    cached = _cache_get(text)
    if cached is not None:
        logger.debug("Embedding cache hit for: %s...", text[:50])
        return cached

    embeddings = await embed_texts([text])
    result = embeddings[0]
    _cache_put(text, result)
    return result
