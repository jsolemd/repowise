"""OpenAI embedding support for repowise semantic search.

Uses the openai SDK with text-embedding-3-small by default (1536 dims).
Runs the synchronous SDK call in a thread pool to avoid blocking asyncio.

Installation:
    pip install openai

Usage:
    import asyncio
    from repowise.core.providers.embedding.openai import OpenAIEmbedder
    from repowise.core.persistence.vector_store import InMemoryVectorStore

    embedder = OpenAIEmbedder(api_key="sk-...")
    store = InMemoryVectorStore(embedder)
    await store.embed_and_upsert("page-1", "Some wiki content...", {})
    results = await store.search("auth service", limit=5)

Dimensions:
    text-embedding-3-small  → 1536 dims
    text-embedding-3-large  → 3072 dims
    text-embedding-ada-002  → 1536 dims

    REPOWISE_EMBEDDING_DIMS (or the ``dimensions`` arg) overrides the width and
    is passed to the API so the returned vectors match. See ``OpenAIEmbedder``.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any, ClassVar

import structlog

from repowise.core.providers.embedding.base import resolve_embedding_timeout
from repowise.core.source_search.chunks import smart_cap

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _input_policy() -> tuple[int, int]:
    """``(char cap, token cap)`` — the per-input ceiling, from its owner.

    Imported here rather than at module scope because reaching
    ``repowise.core.persistence`` pulls the ORM and the CRUD layer onto the
    path, and an embedder has to stay importable without them. Read from that
    module rather than restated, so there is one number to change.
    """
    from repowise.core.persistence.vector_store._base import (
        EMBED_TEXT_MAX_CHARS,
        EMBED_TEXT_MAX_TOKENS,
    )

    return EMBED_TEXT_MAX_CHARS, EMBED_TEXT_MAX_TOKENS


#: Longest unbroken word-character run a text may hold and still be handed to
#: the tokenizer. BPE merges within one such run, and the merge is quadratic in
#: its length: measured on cl100k, a 4k run costs 3.5 ms, an 8k run 14 ms and a
#: 30k run 200 ms — per input, per request. Above the threshold the cap falls
#: back to characters, which is what "where a tokenizer is cheaply available"
#: has to mean if the phrase is to bind on anything.
#:
#: This does not cost the case token counting exists for. Real dense text —
#: CJK prose, minified JSON, base64 — carries punctuation or digits that break
#: the run (measured: 30k of punctuated CJK tokenizes in 2.4 ms and reports
#: 28,500 tokens, which is the 3.5x undercount the character cap makes). What
#: it excludes is a single 4k-character word, which is generated data.
_TOKENIZER_MAX_RUN = 4096
_LONG_RUN_RE = re.compile(rf"\w{{{_TOKENIZER_MAX_RUN + 1},}}")


def _tokenizer_is_cheap(text: str) -> bool:
    """Whether *text* can be tokenized without paying the quadratic case.

    One regex search, ~0.6 ms on a 30,000-character input — two orders of
    magnitude below what it is there to avoid.
    """
    return _LONG_RUN_RE.search(text) is None


@lru_cache(maxsize=8)
def _token_counter_for(model: str) -> Callable[[str], int] | None:
    """A tiktoken counter for *model*, or ``None`` to fall back to characters.

    Resolved once per model and cached, because the first resolution may fetch
    an encoding file over the network; after that it is local and fast. Every
    failure path — tiktoken absent, offline, a model name it has never heard
    of — returns ``None`` rather than raising, since a cap measured in
    characters is what the rest of the codebase already uses and is strictly
    better than no cap at all.

    ``disallowed_special=()`` matters: by default tiktoken *raises* when the
    text contains ``<|endoftext|>``, and the text here is source code, which
    may legitimately contain that string.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    encoding: Any
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        # A local OpenAI-compatible endpoint's model name is not in tiktoken's
        # table. cl100k_base is what every current OpenAI embedding model uses,
        # so it is the right guess rather than a refusal.
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def _count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    return _count


class OpenAIEmbedder:
    """OpenAI embedding model adapter implementing the repowise Embedder protocol.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model:   Embedding model name. Default: "text-embedding-3-small".
        base_url: Optional custom base URL for OpenAI-compatible endpoints.
        dimensions: Output width for a model not in ``_DIMS`` (e.g. a local
            OpenAI-compatible embedder). Falls back to REPOWISE_EMBEDDING_DIMS,
            then to the known-model table, then 1536. An overridden width is
            also sent to the API so the returned vectors match the declaration;
            an endpoint that does not implement the parameter ignores it.
    """

    _DIMS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    # Also bounds the query path, where the SDK retries twice — a bigger default
    # would triple into the caller's budget. Local endpoints raise it by env.
    _DEFAULT_TIMEOUT: float = 10.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        timeout: float | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key= or set OPENAI_API_KEY env var."
            )
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._model = model
        self._timeout = resolve_embedding_timeout(
            timeout, self._DEFAULT_TIMEOUT, provider_env="OPENAI_EMBEDDING_TIMEOUT"
        )
        # When the user overrides the width, request that width from the API too,
        # so the returned vectors match the declaration instead of the model's
        # default — otherwise the store is sized to a width the vectors never
        # have. Servers that don't implement the parameter ignore it (returning
        # their native width, which the override then correctly declares); one
        # that can't honour it rejects the request loudly rather than silently
        # returning a mismatched width. No model is special-cased here: the
        # endpoint, not a hardcoded name table, decides what it accepts.
        self._dimensions, self._request_dimensions = self._resolve_dimensions(dimensions, model)
        self._client: object | None = None  # cached; created once on first embed()

    @classmethod
    def _resolve_dimensions(cls, dimensions: int | None, model: str) -> tuple[int, int | None]:
        """Resolve ``(declared_width, override)``.

        ``override`` is the user-chosen width — explicit arg or
        ``REPOWISE_EMBEDDING_DIMS`` — once validated, or ``None`` when the width
        falls back to the known-model table. It doubles as the value to request
        from the API: ``None`` means send no ``dimensions`` (keep the stock
        request byte-identical). Precedence for the declared width:
        explicit arg > REPOWISE_EMBEDDING_DIMS > known-model table > 1536.

        A local OpenAI-compatible embedder (e.g. a self-hosted model) is not in
        ``_DIMS``; without an override its width would silently default to 1536
        and mismatch the store. Mirrors the Ollama/Gemini embedders, which
        already honour REPOWISE_EMBEDDING_DIMS.
        """
        if dimensions is None:
            env = os.environ.get("REPOWISE_EMBEDDING_DIMS")
            if env:
                try:
                    dimensions = int(env)
                except ValueError:
                    # Match the message every other bad value raises below,
                    # instead of a raw "invalid literal for int()".
                    raise ValueError("dimensions must be a positive integer") from None
        if dimensions is None:
            return cls._DIMS.get(model, 1536), None
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        return dimensions, dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _cap_inputs(self, texts: list[str]) -> list[str]:
        """Hold every input to the per-input ceiling before it is sent.

        The batch writers cap on the way in (``iter_embed_chunks``), but three
        paths reach an embedder without passing through them — the single-item
        ``embed_and_upsert``, the source-search query leg, and the source
        indexer — and OpenAI does not truncate an oversized input, it refuses
        the whole request. A refusal there is not one lost page: it is the
        batch, and on the query leg it is the search. Capping here is the
        backstop for every caller at once, and it reuses the chunk recipe's
        head-and-tail cut rather than inventing a second truncation policy.
        """
        char_cap, token_cap = _input_policy()
        counter = _token_counter_for(self._model)
        capped: list[str] = []
        truncated: dict[str, int] = {}
        largest: dict[str, int] = {}
        for text in texts:
            # Every token is at least one character, so a text shorter than the
            # token budget is under both caps and needs neither the tokenizer
            # nor the scan that decides whether to use it. This is the whole
            # index path: a source chunk is capped at 6,000 characters upstream.
            if len(text) <= token_cap:
                capped.append(text)
                continue
            text_counter = counter if _tokenizer_is_cheap(text) else None
            cap = token_cap if text_counter is not None else char_cap
            result = smart_cap(text, cap, token_counter=text_counter)
            if result.truncated:
                truncated[result.cap_basis] = truncated.get(result.cap_basis, 0) + 1
                largest[result.cap_basis] = max(largest.get(result.cap_basis, 0), result.measured)
            capped.append(result.text)
        for basis, count in truncated.items():
            # Debug, not error: the batch path already reports its own cut at
            # error level, and this one fires for the same page a second time.
            # What is new here is the basis — a text the character cap waved
            # through can still be over the token limit, and that is the case
            # this line exists to name.
            log.debug(
                "openai_embed_input_truncated",
                truncated=count,
                of=len(texts),
                cap=token_cap if basis == "tokens" else char_cap,
                cap_basis=basis,
                largest=largest[basis],
                model=self._model,
            )
        return capped

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI.

        Runs the synchronous SDK call in a thread pool to avoid blocking the
        asyncio event loop.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of L2-normalized float vectors.
        """
        if not texts:
            return []

        texts = self._cap_inputs(texts)
        model = self._model
        timeout = self._timeout
        request_dimensions = self._request_dimensions
        expected_dimensions = self._dimensions

        def _embed_sync() -> list[list[float]]:
            import openai  # type: ignore[import-untyped]

            # Cache client — create once with timeout, reuse across calls.
            if self._client is None:
                self._client = openai.OpenAI(
                    api_key=self._api_key,
                    timeout=timeout,
                    base_url=self._base_url,
                )
            create_kwargs: dict[str, Any] = {"model": model, "input": texts}
            if request_dimensions is not None:
                create_kwargs["dimensions"] = request_dimensions
            response = self._client.embeddings.create(**create_kwargs)  # type: ignore[union-attr]
            raw_vectors = [list(item.embedding) for item in response.data]
            widths = {len(v) for v in raw_vectors}
            if widths and widths != {expected_dimensions}:
                actual = min(widths - {expected_dimensions})
                if request_dimensions is not None:
                    hint = (
                        f"Set REPOWISE_EMBEDDING_DIMS={actual} to match the server's"
                        f" native output, or remove the override to use the model's default."
                    )
                else:
                    hint = (
                        f"The width {expected_dimensions} came from the built-in _DIMS table for"
                        f" {model!r}. Add or update OpenAIEmbedder._DIMS[{model!r}] = {actual},"
                        f" or set REPOWISE_EMBEDDING_DIMS={actual}."
                    )
                raise ValueError(
                    f"OpenAIEmbedder declared {expected_dimensions}-dimensional vectors but the"
                    f" API returned {actual} (model={model!r}). The endpoint likely ignored"
                    f" the 'dimensions' parameter. {hint}"
                )
            return [_l2_normalize(v) for v in raw_vectors]

        return await asyncio.to_thread(_embed_sync)


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector to unit length (cosine similarity = dot product)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        norm = 1.0
    return [x / norm for x in vec]
