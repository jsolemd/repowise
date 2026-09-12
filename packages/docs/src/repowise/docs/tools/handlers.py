"""Canonical tool-entrypoint module for doc-search MCP handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from repowise.docs.db import delete_library as db_delete_library
from repowise.docs.db import get_library, upsert_library
from repowise.docs.db import list_libraries as db_list_libraries
from repowise.docs.formats import DEFAULT_INCLUDE_PATTERNS
from repowise.docs.indexer import (
    delete_by_library as qdrant_delete_by_library,
)
from repowise.docs.indexer import (
    get_chunk_by_id,
    get_code_chunks_for_section,
    get_sibling_chunks,
    search_exact_match,
    search_hybrid,
)
from repowise.docs.jobs import clear_freshness_cache, enqueue_job
from repowise.docs.library.git_manager import (
    delete_repo_cache,
    get_remote_head_sha,
    resolve_default_branch,
)
from repowise.docs.library.source_runtime import get_library_file_content

from . import chunks as chunk_tools
from . import libraries as library_tools
from . import library_resolution
from . import search as search_tools
from .common import ToolError
from .formatting import (
    _derive_title,
    _extract_summary,
    _format_as_markdown,
    _is_api_like_query,
    _maybe_suggest_read_doc,
    _summary_repeats_content,
)

ListLibrariesFn = Callable[..., Awaitable[Any]]
GetLibraryFn = Callable[..., Awaitable[Any]]
SearchFn = Callable[..., Awaitable[Any]]
CodeChunksFn = Callable[..., Awaitable[Any]]
SiblingChunksFn = Callable[..., Awaitable[Any]]
EnqueueJobFn = Callable[..., Awaitable[Any]]
DeleteLibraryFn = Callable[..., Awaitable[Any]]
ResolveBranchFn = Callable[..., Awaitable[Any]]
RemoteHeadFn = Callable[..., Awaitable[Any]]
UpsertLibraryFn = Callable[..., Awaitable[Any]]

_compute_match_score = library_resolution._compute_match_score
_extract_section = library_tools._extract_section
_freshness_decay_factor = search_tools._freshness_decay_factor


async def handle_resolve_library_id(
    arguments: dict[str, Any],
    *,
    list_libraries_fn: ListLibrariesFn | None = None,
) -> dict[str, Any]:
    list_libraries_fn = list_libraries_fn or db_list_libraries
    return await library_resolution.handle_resolve_library_id(
        arguments,
        list_libraries_fn=list_libraries_fn,
    )


async def handle_query_docs(
    arguments: dict[str, Any],
    *,
    get_library_fn: GetLibraryFn | None = None,
    search_exact_match_fn: SearchFn | None = None,
    search_hybrid_fn: SearchFn | None = None,
    get_code_chunks_for_section_fn: CodeChunksFn | None = None,
    get_sibling_chunks_fn: SiblingChunksFn | None = None,
) -> dict[str, Any]:
    get_library_fn = get_library_fn or get_library
    search_exact_match_fn = search_exact_match_fn or search_exact_match
    search_hybrid_fn = search_hybrid_fn or search_hybrid
    get_code_chunks_for_section_fn = get_code_chunks_for_section_fn or get_code_chunks_for_section
    get_sibling_chunks_fn = get_sibling_chunks_fn or get_sibling_chunks
    return await search_tools.handle_query_docs(
        arguments,
        get_library_fn=get_library_fn,
        search_exact_match_fn=search_exact_match_fn,
        search_hybrid_fn=search_hybrid_fn,
        get_code_chunks_for_section_fn=get_code_chunks_for_section_fn,
        get_sibling_chunks_fn=get_sibling_chunks_fn,
    )


async def handle_query_docs_multi(
    arguments: dict[str, Any],
    *,
    get_library_fn: GetLibraryFn | None = None,
    search_hybrid_fn: SearchFn | None = None,
) -> dict[str, Any]:
    get_library_fn = get_library_fn or get_library
    search_hybrid_fn = search_hybrid_fn or search_hybrid
    return await search_tools.handle_query_docs_multi(
        arguments,
        get_library_fn=get_library_fn,
        search_hybrid_fn=search_hybrid_fn,
    )


async def handle_list_libraries(
    arguments: dict[str, Any],
    *,
    list_libraries_fn: ListLibrariesFn | None = None,
    count_libraries_fn=None,
) -> dict[str, Any]:
    list_libraries_fn = list_libraries_fn or db_list_libraries
    if count_libraries_fn is None:
        from repowise.docs.db_libraries import count_libraries as _db_count_libraries

        count_libraries_fn = _db_count_libraries
    return await library_tools.handle_list_libraries(
        arguments,
        list_libraries_fn=list_libraries_fn,
        count_libraries_fn=count_libraries_fn,
    )


async def handle_read_doc(
    arguments: dict[str, Any],
    *,
    get_library_fn: GetLibraryFn | None = None,
    get_library_file_content_fn=None,
) -> dict[str, Any]:
    get_library_fn = get_library_fn or get_library
    get_library_file_content_fn = get_library_file_content_fn or get_library_file_content
    return await library_tools.handle_read_doc(
        arguments,
        get_library_fn=get_library_fn,
        get_library_file_content_fn=get_library_file_content_fn,
    )


async def handle_update_library(
    arguments: dict[str, Any],
    *,
    get_library_fn: GetLibraryFn | None = None,
    enqueue_job_fn: EnqueueJobFn | None = None,
    clear_freshness_cache_fn=None,
) -> dict[str, Any]:
    get_library_fn = get_library_fn or get_library
    enqueue_job_fn = enqueue_job_fn or enqueue_job
    clear_freshness_cache_fn = clear_freshness_cache_fn or clear_freshness_cache
    return await library_tools.handle_update_library(
        arguments,
        get_library_fn=get_library_fn,
        enqueue_job_fn=enqueue_job_fn,
        clear_freshness_cache_fn=clear_freshness_cache_fn,
    )


async def handle_expand_chunk(
    arguments: dict[str, Any],
    *,
    get_chunk_by_id_fn=None,
    get_library_fn: GetLibraryFn | None = None,
    get_library_file_content_fn=None,
) -> dict[str, Any]:
    get_chunk_by_id_fn = get_chunk_by_id_fn or get_chunk_by_id
    get_library_fn = get_library_fn or get_library
    get_library_file_content_fn = get_library_file_content_fn or get_library_file_content
    return await chunk_tools.handle_expand_chunk(
        arguments,
        get_chunk_by_id_fn=get_chunk_by_id_fn,
        get_library_fn=get_library_fn,
        get_library_file_content_fn=get_library_file_content_fn,
    )


async def handle_add_library(
    arguments: dict[str, Any],
    *,
    get_library_fn: GetLibraryFn | None = None,
    get_remote_head_sha_fn: RemoteHeadFn | None = None,
    resolve_default_branch_fn: ResolveBranchFn | None = None,
    upsert_library_fn: UpsertLibraryFn | None = None,
    enqueue_job_fn: EnqueueJobFn | None = None,
    default_include_patterns=DEFAULT_INCLUDE_PATTERNS,
) -> dict[str, Any]:
    get_library_fn = get_library_fn or get_library
    get_remote_head_sha_fn = get_remote_head_sha_fn or get_remote_head_sha
    resolve_default_branch_fn = resolve_default_branch_fn or resolve_default_branch
    upsert_library_fn = upsert_library_fn or upsert_library
    enqueue_job_fn = enqueue_job_fn or enqueue_job
    return await library_tools.handle_add_library(
        arguments,
        get_library_fn=get_library_fn,
        get_remote_head_sha_fn=get_remote_head_sha_fn,
        resolve_default_branch_fn=resolve_default_branch_fn,
        upsert_library_fn=upsert_library_fn,
        enqueue_job_fn=enqueue_job_fn,
        default_include_patterns=default_include_patterns,
    )


async def handle_export_bundle(arguments: dict[str, Any]) -> dict[str, Any]:
    return await library_tools.handle_export_bundle(arguments)


async def handle_import_bundle(arguments: dict[str, Any]) -> dict[str, Any]:
    return await library_tools.handle_import_bundle(arguments)


async def handle_delete_library(
    arguments: dict[str, Any],
    *,
    get_library_fn: GetLibraryFn | None = None,
    qdrant_delete_by_library_fn: DeleteLibraryFn | None = None,
    delete_repo_cache_fn=None,
    db_delete_library_fn: DeleteLibraryFn | None = None,
) -> dict[str, Any]:
    get_library_fn = get_library_fn or get_library
    qdrant_delete_by_library_fn = qdrant_delete_by_library_fn or qdrant_delete_by_library
    delete_repo_cache_fn = delete_repo_cache_fn or delete_repo_cache
    db_delete_library_fn = db_delete_library_fn or db_delete_library
    return await library_tools.handle_delete_library(
        arguments,
        get_library_fn=get_library_fn,
        qdrant_delete_by_library_fn=qdrant_delete_by_library_fn,
        delete_repo_cache_fn=delete_repo_cache_fn,
        db_delete_library_fn=db_delete_library_fn,
    )


__all__ = [
    "ToolError",
    "_compute_match_score",
    "_derive_title",
    "_extract_section",
    "_extract_summary",
    "_format_as_markdown",
    "_freshness_decay_factor",
    "_is_api_like_query",
    "_maybe_suggest_read_doc",
    "_summary_repeats_content",
    "handle_add_library",
    "handle_delete_library",
    "handle_expand_chunk",
    "handle_export_bundle",
    "handle_import_bundle",
    "handle_list_libraries",
    "handle_query_docs",
    "handle_query_docs_multi",
    "handle_read_doc",
    "handle_resolve_library_id",
    "handle_update_library",
]
