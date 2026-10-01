"""A docs search whose query cannot be embedded answers from BM25, and says so."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from repowise.docs.indexer import QueryEmbeddingUnavailableError, search_hybrid
from repowise.docs.library.models import LibraryState, LibraryStatus
from repowise.docs.tools.handlers import handle_query_docs, handle_query_docs_multi

_CAUSE = "ConnectError: [Errno 2] No such file or directory"


@pytest.fixture
def library() -> LibraryState:
    return LibraryState(
        library_id="/langfuse/langfuse-docs",
        repo="langfuse/langfuse-docs",
        name="Langfuse",
        description="LLM observability platform",
        docs_path="pages/",
        branch="main",
        status=LibraryStatus.READY,
        chunk_count=5000,
        file_count=100,
    )


def _point() -> SimpleNamespace:
    return SimpleNamespace(
        id="chunk-1",
        score=7.5,
        payload={
            "content": "Use `observe` to wrap generations and traces.",
            "breadcrumb": ["Tracing", "observe"],
            "breadcrumb_text": "Tracing > observe",
            "file_path": "pages/docs/tracing/observe.mdx",
            "section_anchor": "observe",
            "chunk_type": "doc",
            "doc_category": "reference",
        },
    )


async def test_hybrid_search_names_an_embedding_failure(monkeypatch) -> None:
    async def refused(query: str) -> list[float]:
        raise httpx.ConnectError("[Errno 2] No such file or directory")

    monkeypatch.setattr("repowise.docs.indexer.search.embed_query", refused)
    with pytest.raises(QueryEmbeddingUnavailableError, match="ConnectError"):
        await search_hybrid("/a/b", "observe", client=AsyncMock())


async def test_search_falls_back_to_keywords_and_leads_with_the_warning(library) -> None:
    bm25 = AsyncMock(return_value=[_point()])
    result = await handle_query_docs(
        {"library_id": library.library_id, "query": "observe decorator", "format": "json"},
        get_library_fn=AsyncMock(return_value=library),
        search_hybrid_fn=AsyncMock(side_effect=QueryEmbeddingUnavailableError(_CAUSE)),
        search_bm25_fn=bm25,
        get_code_chunks_for_section_fn=AsyncMock(return_value=[]),
        get_sibling_chunks_fn=AsyncMock(return_value=[]),
    )

    assert next(iter(result)) == "warning"
    assert result["warning"].startswith("Keyword-only results")
    assert _CAUSE in result["warning"]
    assert result["keyword_only"] == _CAUSE
    assert result["search_mode"] == "keyword_only"
    assert [hit["chunk_id"] for hit in result["results"]] == ["chunk-1"]
    bm25.assert_awaited_once()


async def test_multi_library_search_falls_back_per_library(library) -> None:
    result = await handle_query_docs_multi(
        {"library_ids": [library.library_id], "query": "observe", "format": "markdown"},
        get_library_fn=AsyncMock(return_value=library),
        search_hybrid_fn=AsyncMock(side_effect=QueryEmbeddingUnavailableError(_CAUSE)),
        search_bm25_fn=AsyncMock(return_value=[_point()]),
    )

    assert next(iter(result)) == "warning"
    assert result["keyword_only"] == _CAUSE
    assert "Keyword-only results" in result["content"]
    assert "observe.mdx" in result["content"]
