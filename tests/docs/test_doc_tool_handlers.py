"""Focused handler and helper tests for the doc-search subsystem."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from repowise.docs.library.models import (
    IndexJob,
    JobEnqueueDisposition,
    JobEnqueueResult,
    JobType,
    LibraryState,
    LibraryStatus,
)
from repowise.docs.tools.handlers import (
    ToolError,
    _compute_match_score,
    _derive_title,
    _extract_section,
    handle_expand_chunk,
    handle_list_libraries,
    handle_query_docs,
    handle_query_docs_multi,
    handle_read_doc,
    handle_resolve_library_id,
    handle_update_library,
)


@pytest.fixture
def sample_library() -> LibraryState:
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


def test_compute_match_score_prefers_exact_matches(sample_library: LibraryState) -> None:
    assert _compute_match_score("/langfuse/langfuse-docs", sample_library) == 1.0
    assert _compute_match_score("langfuse", sample_library) >= 0.85
    assert _compute_match_score("postgresql", sample_library) == 0.0


def test_compute_match_score_normalizes_hyphens_and_case() -> None:
    """npm-style names like 'framer-motion' match 'Framer Motion'."""
    framer = LibraryState(
        library_id="/framer/motion",
        repo="framer/motion",
        name="Framer Motion",
        description="Production-ready motion library for React",
        docs_path="",
        branch="main",
        status=LibraryStatus.READY,
        chunk_count=3000,
        file_count=900,
    )
    # npm package name → library name (hyphen vs space)
    assert _compute_match_score("framer-motion", framer) >= 0.95
    # exact library_id
    assert _compute_match_score("/framer/motion", framer) == 1.0
    # partial name
    assert _compute_match_score("motion", framer) > 0
    # unrelated
    assert _compute_match_score("tailwind", framer) == 0.0


def test_derive_title_prefers_breadcrumb_then_file_path() -> None:
    assert (
        _derive_title(["SDK", "Decorators", "observe"], "ignored.md")
        == "SDK > Decorators > observe"
    )
    assert _derive_title([], "pages/docs/python-sdk.mdx") == "python-sdk"


def test_extract_section_supports_github_and_mdx_anchors() -> None:
    github = "# Getting Started\n\nHello\n\n## Next\n\nWorld"
    mdx = "# Observability [#observability]\n\nTrace all the things\n\n## Child\n\nNested"

    assert _extract_section(github, "getting-started") == "# Getting Started\n\nHello"
    assert (
        _extract_section(mdx, "observability")
        == "# Observability [#observability]\n\nTrace all the things\n\n## Child\n\nNested"
    )
    assert _extract_section(github, "missing") is None


@pytest.mark.asyncio
async def test_resolve_library_id_returns_best_match(sample_library: LibraryState) -> None:
    result = await handle_resolve_library_id(
        {"library_name": "langfuse"},
        list_libraries_fn=AsyncMock(return_value=[sample_library]),
    )

    assert result["library_id"] == "/langfuse/langfuse-docs"
    assert result["confidence"] >= 0.85


@pytest.mark.asyncio
async def test_query_docs_pending_library_returns_warning_markdown() -> None:
    library = LibraryState(
        library_id="/test/pending",
        repo="test/pending",
        name="Pending Docs",
        description="Pending index",
        docs_path="docs/",
        branch="main",
        status=LibraryStatus.PENDING,
    )

    result = await handle_query_docs(
        {
            "library_id": library.library_id,
            "query": "getting started",
            "format": "markdown",
        },
        get_library_fn=AsyncMock(return_value=library),
    )

    assert result["format"] == "markdown"
    assert "pending indexing" in result["content"].lower()


@pytest.mark.asyncio
async def test_query_docs_returns_preview_first_results(sample_library: LibraryState) -> None:
    point = SimpleNamespace(
        id="chunk-1",
        score=0.91,
        payload={
            "content": "Trace LLM calls with the observe decorator. Then inspect spans in the UI for details.",
            "breadcrumb": ["Tracing", "observe"],
            "breadcrumb_text": "Tracing > observe",
            "file_path": "pages/docs/tracing/observe.mdx",
            "section_anchor": "observe",
            "source_url": "https://github.com/langfuse/langfuse-docs/blob/main/pages/docs/tracing/observe.mdx#observe",
            "chunk_type": "doc",
            "doc_category": "reference",
        },
    )

    result = await handle_query_docs(
        {
            "library_id": sample_library.library_id,
            "query": "observe decorator",
            "format": "json",
        },
        get_library_fn=AsyncMock(return_value=sample_library),
        search_hybrid_fn=AsyncMock(return_value=[point]),
        get_code_chunks_for_section_fn=AsyncMock(return_value=[]),
        get_sibling_chunks_fn=AsyncMock(return_value=[]),
    )

    hit = result["results"][0]
    assert hit["chunk_id"] == "chunk-1"
    assert hit["title"] == "Tracing > observe"
    assert "preview" in hit
    assert hit["preview"].startswith("Trace LLM calls")
    assert "content" not in hit
    assert hit["anchor"] == "observe"
    assert result["recommended_tool"] == "read_doc"


@pytest.mark.asyncio
async def test_query_docs_uses_persisted_freshness_state_for_warnings(
    sample_library: LibraryState,
) -> None:
    point = SimpleNamespace(
        id="chunk-1",
        score=0.91,
        payload={
            "content": "Trace LLM calls with the observe decorator.",
            "breadcrumb": ["Tracing", "observe"],
            "breadcrumb_text": "Tracing > observe",
            "file_path": "pages/docs/tracing/observe.mdx",
            "section_anchor": "observe",
            "source_url": "https://example.com/observe",
            "chunk_type": "doc",
            "doc_category": "reference",
        },
    )
    stale_library = sample_library.model_copy(
        update={
            "last_freshness_state": "stale",
            "last_freshness_error": "",
        }
    )

    result = await handle_query_docs(
        {
            "library_id": stale_library.library_id,
            "query": "observe decorator",
            "format": "json",
        },
        get_library_fn=AsyncMock(return_value=stale_library),
        search_hybrid_fn=AsyncMock(return_value=[point]),
        get_code_chunks_for_section_fn=AsyncMock(return_value=[]),
        get_sibling_chunks_fn=AsyncMock(return_value=[]),
    )

    assert result["stale"] is True
    assert any("outdated" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_query_docs_multi_supports_markdown_output(sample_library: LibraryState) -> None:
    point = SimpleNamespace(
        id="chunk-1",
        score=0.88,
        payload={
            "content": "Use `observe` to wrap generations and traces.",
            "breadcrumb": ["Tracing", "observe"],
            "breadcrumb_text": "Tracing > observe",
            "file_path": "pages/docs/tracing/observe.mdx",
            "section_anchor": "observe",
            "source_url": "https://github.com/langfuse/langfuse-docs/blob/main/pages/docs/tracing/observe.mdx#observe",
            "chunk_type": "doc",
            "doc_category": "reference",
        },
    )

    result = await handle_query_docs_multi(
        {
            "library_ids": [sample_library.library_id],
            "query": "observe decorator",
            "format": "markdown",
        },
        get_library_fn=AsyncMock(return_value=sample_library),
        search_hybrid_fn=AsyncMock(return_value=[point]),
    )

    assert result["format"] == "markdown"
    assert "Langfuse" in result["content"]
    assert "observe.mdx" in result["content"]


@pytest.mark.asyncio
async def test_query_docs_multi_json_includes_detected_intent(sample_library: LibraryState) -> None:
    point = SimpleNamespace(
        id="chunk-1",
        score=0.88,
        payload={
            "content": "Use `observe` to wrap generations and traces.",
            "breadcrumb": ["Tracing", "observe"],
            "breadcrumb_text": "Tracing > observe",
            "file_path": "pages/docs/tracing/observe.mdx",
            "section_anchor": "observe",
            "source_url": "https://github.com/langfuse/langfuse-docs/blob/main/pages/docs/tracing/observe.mdx#observe",
            "chunk_type": "doc",
            "doc_category": "reference",
        },
    )

    result = await handle_query_docs_multi(
        {
            "library_ids": [sample_library.library_id],
            "query": "observe decorator",
            "format": "json",
        },
        get_library_fn=AsyncMock(return_value=sample_library),
        search_hybrid_fn=AsyncMock(return_value=[point]),
    )

    assert result["search_mode"] == "hybrid_multi"
    assert result["detected_intent"]


@pytest.mark.asyncio
async def test_list_libraries_filters_ready_status(sample_library: LibraryState) -> None:
    list_libraries = AsyncMock(return_value=[sample_library])
    count_libraries = AsyncMock(return_value=1)

    result = await handle_list_libraries(
        {"include_stats": True, "status": "ready"},
        list_libraries_fn=list_libraries,
        count_libraries_fn=count_libraries,
    )

    list_libraries.assert_awaited_once_with(status=LibraryStatus.READY, offset=0, limit=None)
    count_libraries.assert_awaited_once_with(status=LibraryStatus.READY)
    assert result["total"] == 1
    assert result["returned"] == 1
    assert result["offset"] == 0
    assert result["libraries"][0]["library_id"] == sample_library.library_id
    assert "next_offset" not in result  # whole list returned


@pytest.mark.asyncio
async def test_list_libraries_surfaces_ready_warning_message(sample_library: LibraryState) -> None:
    warned_library = sample_library.model_copy(
        update={"error_message": "Skipped 2 file(s) during indexing: docs/vendor.js"}
    )
    result = await handle_list_libraries(
        {"include_stats": True},
        list_libraries_fn=AsyncMock(return_value=[warned_library]),
        count_libraries_fn=AsyncMock(return_value=1),
    )

    assert result["libraries"][0]["warning_message"] == warned_library.error_message


@pytest.mark.asyncio
async def test_list_libraries_paginates_with_next_offset(sample_library: LibraryState) -> None:
    # Fake a page of 2 libraries with total=5, offset=0, limit=2.
    libs = [sample_library, sample_library.model_copy(update={"library_id": "/acme/widget"})]
    list_libraries = AsyncMock(return_value=libs)
    count_libraries = AsyncMock(return_value=5)

    result = await handle_list_libraries(
        {"include_stats": False, "offset": 0, "limit": 2},
        list_libraries_fn=list_libraries,
        count_libraries_fn=count_libraries,
    )

    list_libraries.assert_awaited_once_with(status=None, offset=0, limit=2)
    assert result["total"] == 5
    assert result["returned"] == 2
    assert result["limit"] == 2
    assert result["offset"] == 0
    assert result["next_offset"] == 2  # agent can fetch page 2 from here

    # Preserve the later-page contract that previously lived in the retired
    # code-lane dogfood suite: offsets advance from the requested page, not
    # from zero.
    later_page = [
        sample_library.model_copy(update={"library_id": f"/acme/widget-{index}"})
        for index in range(10, 20)
    ]
    list_later_page = AsyncMock(return_value=later_page)
    count_later_page = AsyncMock(return_value=44)

    later_result = await handle_list_libraries(
        {"include_stats": False, "offset": 10, "limit": 10},
        list_libraries_fn=list_later_page,
        count_libraries_fn=count_later_page,
    )

    list_later_page.assert_awaited_once_with(status=None, offset=10, limit=10)
    assert later_result["returned"] == 10
    assert later_result["next_offset"] == 20


@pytest.mark.asyncio
async def test_read_doc_extracts_section(tmp_path: Path, sample_library: LibraryState) -> None:
    doc_path = tmp_path / "docs"
    doc_path.mkdir()
    file_path = doc_path / "guide.md"
    file_path.write_text(
        "# Intro\n\nskip\n\n## Install\n\nhello\n\n## Next\n\nworld", encoding="utf-8"
    )

    result = await handle_read_doc(
        {
            "library_id": sample_library.library_id,
            "path": "docs/guide.md",
            "section": "install",
        },
        get_library_fn=AsyncMock(return_value=sample_library),
        get_library_file_content_fn=AsyncMock(return_value=file_path.read_text(encoding="utf-8")),
    )

    assert result["section"] == "install"
    assert result["content"].startswith("## Install")
    assert "hello" in result["content"]
    assert "world" not in result["content"]


@pytest.mark.asyncio
async def test_update_library_returns_coalesced_when_job_exists(
    sample_library: LibraryState,
) -> None:
    result = await handle_update_library(
        {"library_id": sample_library.library_id, "force": True},
        get_library_fn=AsyncMock(return_value=sample_library),
        enqueue_job_fn=AsyncMock(
            return_value=JobEnqueueResult(
                disposition=JobEnqueueDisposition.COALESCED,
                job=IndexJob(
                    id="job-456",
                    library_id=sample_library.library_id,
                    job_type=JobType.FORCE,
                    priority=5,
                ),
            )
        ),
        clear_freshness_cache_fn=lambda _library_id: None,
    )

    assert result["disposition"] == "coalesced"
    assert result["job"]["id"] == "job-456"
    assert result["library_name"] == sample_library.name


@pytest.mark.asyncio
async def test_update_library_returns_job_metadata_when_queued(
    sample_library: LibraryState,
) -> None:
    job = IndexJob(
        id="job-123",
        library_id=sample_library.library_id,
        job_type=JobType.FORCE,
        priority=5,
    )
    result = await handle_update_library(
        {"library_id": sample_library.library_id, "force": True},
        get_library_fn=AsyncMock(return_value=sample_library),
        enqueue_job_fn=AsyncMock(
            return_value=JobEnqueueResult(
                disposition=JobEnqueueDisposition.QUEUED,
                job=job,
            )
        ),
        clear_freshness_cache_fn=lambda _library_id: None,
    )

    assert result == {
        "disposition": "queued",
        "library_id": sample_library.library_id,
        "library_name": sample_library.name,
        "job": {
            "id": "job-123",
            "type": "force",
            "priority": 5,
        },
    }


@pytest.mark.asyncio
async def test_query_docs_ready_library_includes_partial_index_warning(
    sample_library: LibraryState,
) -> None:
    warned_library = sample_library.model_copy(
        update={"error_message": "Skipped 1 file(s) during indexing: docs/vendor.js"}
    )
    point = SimpleNamespace(
        id="chunk-1",
        score=0.91,
        payload={
            "content": "Trace LLM calls with the observe decorator.",
            "breadcrumb": ["Tracing", "observe"],
            "breadcrumb_text": "Tracing > observe",
            "file_path": "pages/docs/tracing/observe.mdx",
            "section_anchor": "observe",
            "source_url": "https://example.test/observe",
            "chunk_type": "doc",
            "doc_category": "reference",
        },
    )

    result = await handle_query_docs(
        {
            "library_id": warned_library.library_id,
            "query": "observe decorator",
            "format": "json",
        },
        get_library_fn=AsyncMock(return_value=warned_library),
        search_hybrid_fn=AsyncMock(return_value=[point]),
        get_code_chunks_for_section_fn=AsyncMock(return_value=[]),
        get_sibling_chunks_fn=AsyncMock(return_value=[]),
    )

    assert "warnings" in result
    assert "indexed with warnings" in result["warnings"][0].lower()


@pytest.mark.asyncio
async def test_read_doc_rejects_missing_library() -> None:
    with pytest.raises(ToolError, match="Library not found"):
        await handle_read_doc(
            {"library_id": "/missing/lib", "path": "docs/guide.md"},
            get_library_fn=AsyncMock(return_value=None),
        )


@pytest.mark.asyncio
async def test_expand_chunk_uses_stored_line_spans(
    tmp_path: Path, sample_library: LibraryState
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    file_path = docs_dir / "guide.md"
    file_path.write_text(
        "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\n",
        encoding="utf-8",
    )

    chunk_record = SimpleNamespace(
        payload={
            "library_id": sample_library.library_id,
            "file_path": "docs/guide.md",
            "content": "line 3\nline 4",
            "line_start": 3,
            "line_end": 4,
            "section_anchor": "guide",
            "canonical_anchor": "guide",
        }
    )

    result = await handle_expand_chunk(
        {
            "chunk_id": "chunk-1",
            "lines_before": 1,
            "lines_after": 1,
        },
        get_chunk_by_id_fn=AsyncMock(return_value=chunk_record),
        get_library_fn=AsyncMock(return_value=sample_library),
        get_library_file_content_fn=AsyncMock(return_value=file_path.read_text(encoding="utf-8")),
    )

    assert result["line_start"] == 2
    assert result["line_end"] == 5
    assert result["chunk_anchor"] == "guide"
    assert result["anchor"] == "guide"
    assert "   2 | line 2" in result["content"]
