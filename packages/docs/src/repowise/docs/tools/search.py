"""Search-oriented doc-search tool handlers."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from repowise.docs.chunking.models import canonicalize_section_anchor
from repowise.docs.db import get_library
from repowise.docs.db import list_libraries as db_list_libraries
from repowise.docs.indexer import (
    get_code_chunks_for_section,
    get_sibling_chunks,
    search_exact_match,
    search_hybrid,
)
from repowise.docs.jobs import FreshnessState
from repowise.docs.library.models import LibraryState, LibraryStatus
from repowise.docs.search.intent import detect_intent

from .common import ToolError, get_arg
from .formatting import (
    _build_preview,
    _compact_code_block,
    _derive_title,
    _extract_summary,
    _format_as_markdown,
)
from .library_resolution import _resolve_library

logger = logging.getLogger(__name__)

# Freshness decay parameters (inspired by NornicDB's memory decay model)
_FRESHNESS_FRESH_DAYS = 7
_FRESHNESS_STALE_DAYS = 90
_FRESHNESS_BOOST = 1.05
_FRESHNESS_PENALTY = 0.90
_DEFAULT_SEARCH_DOCS_LIMIT = 6
_DEFAULT_SEARCH_DOCS_MULTI_LIMIT = 3
_RELATED_SECTION_FILE_LIMIT = 2
_RELATED_SECTION_PER_FILE_LIMIT = 4
_MAX_CODE_BLOCKS_PER_HIT = 1


def _resolve_output_format(arguments: dict[str, Any], *, default: str = "json") -> str:
    """Resolve the unified output selector while tolerating legacy `format` callers."""
    value = str(arguments.get("output") or arguments.get("format") or default).lower()
    if value not in {"json", "markdown"}:
        raise ToolError(f"Invalid format: {value}. Use 'json' or 'markdown'.")
    return value


def _build_search_hit(
    point,
    *,
    library_id: str | None = None,
    library_name: str | None = None,
) -> dict[str, Any]:
    """Build a compact search hit for LLM-oriented navigation."""
    payload = point.payload or {}
    breadcrumb = payload.get("breadcrumb", [])
    file_path = payload.get("file_path", "")
    content = payload.get("content", "")
    chunk_anchor = payload.get("section_anchor", "")
    canonical_anchor = payload.get("canonical_anchor") or canonicalize_section_anchor(chunk_anchor)
    result = {
        "chunk_id": point.id,
        "score": round(point.score, 4) if point.score else None,
        "title": _derive_title(breadcrumb, file_path),
        "preview": _build_preview(content),
        "breadcrumb_text": payload.get("breadcrumb_text", ""),
        "file_path": file_path,
        "anchor": canonical_anchor,
        "source_url": payload.get("source_url", ""),
        "chunk_type": payload.get("chunk_type", "doc"),
        "doc_category": payload.get("doc_category", "other"),
    }
    if chunk_anchor and chunk_anchor != canonical_anchor:
        result["chunk_anchor"] = chunk_anchor
    line_start = payload.get("line_start")
    line_end = payload.get("line_end")
    if line_start is not None:
        result["line_start"] = line_start
    if line_end is not None:
        result["line_end"] = line_end
    summary = _extract_summary(content)
    if summary and summary != result["preview"]:
        result["summary"] = summary
    if library_id:
        result["library_id"] = library_id
    if library_name:
        result["library_name"] = library_name
    return result


def _freshness_decay_factor(indexed_at: datetime | None) -> float:
    """Compute a gentle multiplicative freshness factor for search scoring."""
    if indexed_at is None:
        return 1.0

    now = datetime.now(UTC)
    if indexed_at.tzinfo is None:
        indexed_at = indexed_at.replace(tzinfo=UTC)

    age_days = (now - indexed_at).total_seconds() / 86400
    if age_days <= _FRESHNESS_FRESH_DAYS:
        return _FRESHNESS_BOOST
    if age_days <= _FRESHNESS_STALE_DAYS:
        t = (age_days - _FRESHNESS_FRESH_DAYS) / (_FRESHNESS_STALE_DAYS - _FRESHNESS_FRESH_DAYS)
        return _FRESHNESS_BOOST + t * (1.0 - _FRESHNESS_BOOST)
    if age_days <= _FRESHNESS_STALE_DAYS * 2:
        t = (age_days - _FRESHNESS_STALE_DAYS) / _FRESHNESS_STALE_DAYS
        return 1.0 + t * (_FRESHNESS_PENALTY - 1.0)
    return _FRESHNESS_PENALTY


async def handle_query_docs(
    arguments: dict[str, Any],
    *,
    get_library_fn=get_library,
    list_libraries_fn=db_list_libraries,
    search_exact_match_fn=search_exact_match,
    search_hybrid_fn=search_hybrid,
    get_code_chunks_for_section_fn=get_code_chunks_for_section,
    get_sibling_chunks_fn=get_sibling_chunks,
) -> dict[str, Any]:
    """Search documentation using hybrid search (BM25 + semantic)."""
    library_id = str(get_arg(arguments, "library_id", aliases=("libraryId",), default="")).strip()
    query = arguments.get("query", "").strip()
    limit = min(max(int(arguments.get("limit", _DEFAULT_SEARCH_DOCS_LIMIT)), 1), 50)
    chunk_types = get_arg(arguments, "chunk_types", aliases=("chunkTypes",))
    output_format = _resolve_output_format(arguments)
    exact_match = bool(get_arg(arguments, "exact_match", aliases=("exactMatch",), default=False))

    include_code_blocks = get_arg(arguments, "include_code_blocks", aliases=("includeCodeBlocks",))
    if include_code_blocks is None:
        include_code_blocks = output_format == "markdown"

    if not library_id:
        raise ToolError("library_id is required")
    if not query:
        raise ToolError("query is required")
    library, canonical_library_id = await _resolve_library(
        library_id,
        get_library_fn=get_library_fn,
        list_libraries_fn=list_libraries_fn,
    )
    if not library:
        raise ToolError(f"Library not found: {library_id}")
    # Switch to the canonical casing for every downstream query so we
    # don't accidentally pass a non-matching shape to Qdrant / SQL.
    library_id = canonical_library_id

    if library.status == LibraryStatus.PENDING:
        pending_response = {
            "results": [],
            "total": 0,
            "library_id": library_id,
            "query": query,
            "warnings": ["Library is pending indexing. No documents available yet."],
        }
        if output_format == "markdown":
            return {
                "content": _format_as_markdown(
                    [], query, library_id, pending_response["warnings"][0]
                ),
                "format": "markdown",
            }
        return pending_response

    intent = detect_intent(query)
    logger.debug("Detected intent for '%s': %s", query[:30], intent.value)

    if exact_match:
        points = await search_exact_match_fn(
            library_id=library_id,
            query=query,
            limit=limit,
            chunk_types=chunk_types,
            phrase_boost=2.0,
        )
    else:
        points = await search_hybrid_fn(
            library_id=library_id,
            query=query,
            limit=limit,
            chunk_types=chunk_types,
            intent=intent,
        )

    freshness_factor = _freshness_decay_factor(library.indexed_at)
    if freshness_factor != 1.0:
        for point in points:
            if point.score is not None:
                point.score *= freshness_factor
        points.sort(key=lambda p: p.score if p.score else 0, reverse=True)
        logger.debug(
            "Applied freshness decay factor %.3f (indexed_at=%s)",
            freshness_factor,
            library.indexed_at,
        )

    results = []
    code_block_requests: list[tuple[int, str, str]] = []
    for point in points:
        result = _build_search_hit(point)
        result.setdefault("library_id", library_id)
        results.append(result)
        anchor = result.get("chunk_anchor") or result.get("anchor")
        if include_code_blocks and result["chunk_type"] == "doc" and anchor:
            code_block_requests.append((len(results) - 1, result["file_path"], str(anchor)))

    if code_block_requests:
        code_block_results = await asyncio.gather(
            *(
                get_code_chunks_for_section_fn(
                    library_id=library_id,
                    file_path=file_path,
                    section_anchor=section_anchor,
                )
                for _, file_path, section_anchor in code_block_requests
            ),
            return_exceptions=True,
        )
        for (result_idx, file_path, section_anchor), code_blocks in zip(
            code_block_requests,
            code_block_results,
            strict=False,
        ):
            if isinstance(code_blocks, Exception):
                logger.debug(
                    "Could not fetch code blocks for %s#%s: %s",
                    file_path,
                    section_anchor,
                    code_blocks,
                )
                continue
            if code_blocks:
                results[result_idx]["code_blocks"] = [
                    _compact_code_block(block) for block in code_blocks[:_MAX_CODE_BLOCKS_PER_HIT]
                ]

    response = {
        "results": results,
        "total": len(results),
        "library_id": library_id,
        "normalized_library_id": canonical_library_id,
        "query": query,
        "search_mode": "exact" if exact_match else "hybrid",
        "detected_intent": intent.value,
    }

    related_sections: list[dict[str, Any]] = []
    try:
        file_paths_seen: set[str] = set()
        result_ids = [r["chunk_id"] for r in results]
        file_paths: list[str] = []
        for result in results:
            file_path = result.get("file_path", "")
            if not file_path or file_path in file_paths_seen:
                continue
            file_paths_seen.add(file_path)
            file_paths.append(file_path)

        sibling_results = await asyncio.gather(
            *(
                get_sibling_chunks_fn(
                    library_id=library_id,
                    file_path=file_path,
                    exclude_ids=result_ids,
                )
                for file_path in file_paths
            ),
            return_exceptions=True,
        )

        for file_path, siblings in zip(file_paths, sibling_results, strict=False):
            if isinstance(siblings, Exception):
                logger.debug("Could not fetch related sections for %s: %s", file_path, siblings)
                continue
            if siblings:
                related_sections.append(
                    {
                        "file_path": file_path,
                        "sections": siblings[:_RELATED_SECTION_PER_FILE_LIMIT],
                    }
                )

        if related_sections:
            if len(related_sections) > _RELATED_SECTION_FILE_LIMIT:
                related_sections = related_sections[:_RELATED_SECTION_FILE_LIMIT]
            response["related_sections"] = related_sections
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not fetch related sections: %s", e)

    if results:
        response["recommended_tool"] = "read_doc"

    warnings: list[str] = []
    if library.status == LibraryStatus.INDEXING:
        warnings.append("Library is currently being indexed. Results may be incomplete.")
    elif library.status == LibraryStatus.ERROR:
        warnings.append(f"Library has indexing errors: {library.error_message}")
    elif library.error_message:
        warnings.append(f"Library indexed with warnings: {library.error_message}")
    elif library.last_freshness_state == FreshnessState.STALE.value:
        warnings.append(
            "Documentation may be outdated - newer version available upstream. "
            "Use update_doc_library to refresh."
        )
        response["stale"] = True
    elif library.last_freshness_state == FreshnessState.UNKNOWN.value:
        detail = library.last_freshness_error or "freshness verification is currently unavailable"
        warnings.append(
            f"Documentation freshness could not be verified upstream. Last check status: {detail}."
        )
    warning = " | ".join(warnings) if warnings else None
    if warnings:
        response["warnings"] = warnings

    if output_format == "markdown":
        return {
            "content": _format_as_markdown(
                results,
                query,
                library_id,
                warning=warning,
                related_sections=related_sections if related_sections else None,
            ),
            "format": "markdown",
        }
    return response


async def handle_query_docs_multi(
    arguments: dict[str, Any],
    *,
    get_library_fn=get_library,
    list_libraries_fn=db_list_libraries,
    search_hybrid_fn=search_hybrid,
) -> dict[str, Any]:
    """Search across multiple documentation libraries simultaneously."""
    library_ids = get_arg(arguments, "library_ids", aliases=("libraryIds",), default=[])
    query = arguments.get("query", "").strip()
    limit_per_library = min(
        max(
            int(
                get_arg(
                    arguments,
                    "limit_per_library",
                    aliases=("limitPerLibrary",),
                    default=_DEFAULT_SEARCH_DOCS_MULTI_LIMIT,
                )
            ),
            1,
        ),
        20,
    )
    chunk_types = get_arg(arguments, "chunk_types", aliases=("chunkTypes",))
    output_format = _resolve_output_format(arguments)

    if not library_ids:
        raise ToolError("library_ids is required and must contain at least one library")
    if not query:
        raise ToolError("query is required")
    if len(library_ids) > 10:
        raise ToolError("Maximum 10 libraries can be searched at once")
    libraries: list[LibraryState] = []
    unmatched_library_ids: list[str] = []
    skipped_libraries: list[dict[str, str]] = []
    normalizations: dict[str, str] = {}
    intent = detect_intent(query)
    for lib_id in library_ids:
        lib, canonical = await _resolve_library(
            lib_id,
            get_library_fn=get_library_fn,
            list_libraries_fn=list_libraries_fn,
        )
        if canonical != lib_id:
            normalizations[lib_id] = canonical
        if not lib:
            unmatched_library_ids.append(lib_id)
        elif lib.status == LibraryStatus.PENDING:
            skipped_libraries.append(
                {
                    "library_id": lib.library_id,
                    "reason": "pending_indexing",
                }
            )
        else:
            libraries.append(lib)

    if not libraries:
        failure_bits: list[str] = []
        if unmatched_library_ids:
            failure_bits.append("Library not found: " + ", ".join(unmatched_library_ids))
        if skipped_libraries:
            failure_bits.append(
                "Library pending indexing: " + ", ".join(s["library_id"] for s in skipped_libraries)
            )
        raise ToolError(f"No searchable libraries found. Errors: {'; '.join(failure_bits)}")

    async def search_one_library(lib: LibraryState) -> tuple[str, list[dict], str | None]:
        warnings: list[str] = []
        if lib.status == LibraryStatus.INDEXING:
            warnings.append("Currently being indexed - results may be incomplete")
        elif lib.status == LibraryStatus.ERROR:
            warnings.append(f"Has indexing errors: {lib.error_message}")
        elif lib.error_message:
            warnings.append(f"Indexed with warnings: {lib.error_message}")

        try:
            points = await search_hybrid_fn(
                library_id=lib.library_id,
                query=query,
                limit=limit_per_library,
                chunk_types=chunk_types,
                intent=intent,
            )
            freshness_factor = _freshness_decay_factor(lib.indexed_at)
            if freshness_factor != 1.0:
                for point in points:
                    if point.score is not None:
                        point.score *= freshness_factor

            results = []
            for point in points:
                results.append(
                    _build_search_hit(
                        point,
                        library_id=lib.library_id,
                        library_name=lib.name,
                    )
                )
            warning = " | ".join(warnings) if warnings else None
            return lib.library_id, results, warning
        except Exception as e:
            logger.error("Error searching %s: %s", lib.library_id, e)
            return lib.library_id, [], f"Search error: {e!s}"

    search_results = await asyncio.gather(*(search_one_library(lib) for lib in libraries))

    all_results: list[dict] = []
    library_summaries: list[dict] = []
    warnings: list[str] = []
    for lib_id, results, warning in search_results:
        lib = next(library for library in libraries if library.library_id == lib_id)
        all_results.extend(results)
        library_summaries.append(
            {"library_id": lib_id, "name": lib.name, "result_count": len(results)}
        )
        if warning:
            warnings.append(f"{lib.name}: {warning}")

    all_results.sort(key=lambda x: x.get("score") or 0, reverse=True)
    response = {
        "results": all_results,
        "total": len(all_results),
        "libraries": library_summaries,
        "query": query,
        "search_mode": "hybrid_multi",
        "detected_intent": intent.value,
    }
    if unmatched_library_ids:
        response["unmatched_library_ids"] = unmatched_library_ids
    if skipped_libraries:
        response["skipped_libraries"] = skipped_libraries
    if warnings:
        response["warnings"] = warnings
    if normalizations:
        response["normalized_library_ids"] = normalizations
    if output_format == "markdown":
        warning_lines: list[str] = []
        for lib_id in unmatched_library_ids:
            warning_lines.append(f"Library not found: {lib_id}")
        for skipped in skipped_libraries:
            warning_lines.append(f"Library {skipped['library_id']} skipped ({skipped['reason']})")
        if warnings:
            warning_lines.extend(warnings)
        warning_text = " | ".join(warning_lines) if warning_lines else None
        return {
            "content": _format_as_markdown(
                all_results,
                query,
                "multiple",
                warning=warning_text,
            ),
            "format": "markdown",
        }
    return response
