"""Retrieval and search helpers for doc-search Qdrant indexing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient, models

from repowise.docs.config import get_settings
from repowise.docs.indexer.collection import get_qdrant_client
from repowise.docs.indexer.embedder import embed_single
from repowise.docs.indexer.scoring import (
    ANCHOR_SPLIT_SUFFIX_PATTERN,
    apply_exact_match_boosts,
    apply_search_boosts,
    is_exact_lookup_weak_path,
    surface_query_tokens,
)
from repowise.docs.indexer.shared import build_file_filter, build_library_filter

if TYPE_CHECKING:
    from repowise.docs.search.intent import QueryIntent

logger = logging.getLogger(__name__)

_SURFACE_RESCUE_LIMIT_MIN = 40
_SURFACE_RESCUE_LIMIT_SCALE = 4


def _surface_rescue_query(query: str) -> str | None:
    """Return the leading exact surface token for a descriptive query, if useful."""
    normalized = query.strip()
    if not normalized or " " not in normalized:
        return None
    tokens = surface_query_tokens(normalized)
    if not tokens:
        return None
    token = tokens[0]
    if len(token) < 3:
        return None
    return token


def _merge_scored_points(
    primary: list[models.ScoredPoint],
    rescue: list[models.ScoredPoint],
) -> list[models.ScoredPoint]:
    """Merge result sets by point id, keeping the strongest score per point."""
    merged: dict[object, models.ScoredPoint] = {}
    for point in primary:
        merged[point.id] = point
    for point in rescue:
        existing = merged.get(point.id)
        if existing is None or (point.score or 0.0) > (existing.score or 0.0):
            merged[point.id] = point
    return list(merged.values())


def _sort_results(
    results: list[models.ScoredPoint],
    *,
    prefer_strong_lookup_paths: bool = False,
) -> list[models.ScoredPoint]:
    """Sort results by score, optionally demoting weak lookup paths."""
    if not prefer_strong_lookup_paths:
        results.sort(key=lambda p: p.score if p.score else 0, reverse=True)
        return results

    results.sort(
        key=lambda point: (
            is_exact_lookup_weak_path(str((point.payload or {}).get("file_path", "") or "")),
            -(point.score or 0.0),
        )
    )
    return results


async def search_hybrid(
    library_id: str,
    query: str,
    limit: int = 10,
    chunk_types: list[str] | None = None,
    intent: QueryIntent | None = None,
    client: QdrantClient | None = None,
) -> list[models.ScoredPoint]:
    """Hybrid search with BM25 + dense vectors using RRF fusion."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    logger.debug("Embedding query: %s...", query[:50])
    query_embedding = await embed_single(query)

    query_filter = build_library_filter(library_id, chunk_types)

    prefetch_limit = max(200, limit * 10)
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="bm25",
                limit=prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=query_embedding,
                using="dense",
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=prefetch_limit,
        with_payload=True,
    )

    results = list(response.points)
    results = apply_search_boosts(results, settings, intent=intent, query=query)

    rescue_query = _surface_rescue_query(query)
    if rescue_query:
        rescue_limit = max(_SURFACE_RESCUE_LIMIT_MIN, limit * _SURFACE_RESCUE_LIMIT_SCALE)
        rescue_response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=models.Document(text=rescue_query, model="Qdrant/bm25"),
            using="bm25",
            limit=rescue_limit,
            query_filter=query_filter,
            with_payload=True,
        )
        rescue_results = list(rescue_response.points)
        rescue_results = apply_search_boosts(rescue_results, settings, intent=intent, query=query)
        results = _merge_scored_points(results, rescue_results)

    if intent is not None and intent.value == "api_lookup":
        results = apply_exact_match_boosts(results, query, phrase_boost=1.25)

    _sort_results(results, prefer_strong_lookup_paths=bool(intent and intent.value == "api_lookup"))
    results = results[:limit]

    intent_str = f", intent={intent.value}" if intent else ""
    logger.info(
        "Hybrid search for '%s...' in %s: %d results%s",
        query[:30],
        library_id,
        len(results),
        intent_str,
    )

    return results


async def search_dense(
    library_id: str,
    query: str,
    limit: int = 10,
    chunk_types: list[str] | None = None,
    client: QdrantClient | None = None,
) -> list[models.ScoredPoint]:
    """Dense (semantic) search only."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    query_embedding = await embed_single(query)
    query_filter = build_library_filter(library_id, chunk_types)

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_embedding,
        using="dense",
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )

    return response.points


async def get_chunk_by_id(
    chunk_id: str,
    client: QdrantClient | None = None,
) -> models.Record | None:
    """Retrieve a single chunk by its ID."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()

    try:
        results = client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[chunk_id],
            with_payload=True,
        )
        return results[0] if results else None
    except Exception as e:
        logger.warning("Failed to retrieve chunk %s: %s", chunk_id, e)
        return None


async def get_sibling_chunks(
    library_id: str,
    file_path: str,
    exclude_ids: list[str] | None = None,
    client: QdrantClient | None = None,
) -> list[dict]:
    """Get other chunks from the same file for related-sections rendering."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    exclude_ids = exclude_ids or []

    response = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=build_file_filter(library_id, file_path),
        limit=50,
        with_payload=["section_anchor", "breadcrumb", "breadcrumb_text", "chunk_type"],
    )

    siblings = []
    seen_anchors: set[str] = set()

    for point in response[0]:
        if point.id in exclude_ids:
            continue

        payload = point.payload or {}
        anchor = payload.get("section_anchor", "")
        if not anchor or anchor in seen_anchors:
            continue
        seen_anchors.add(anchor)

        siblings.append(
            {
                "anchor": anchor,
                "breadcrumb": payload.get("breadcrumb", []),
                "breadcrumb_text": payload.get("breadcrumb_text", ""),
                "chunk_type": payload.get("chunk_type", "doc"),
            }
        )

    return siblings


async def get_code_chunks_for_section(
    library_id: str,
    file_path: str,
    section_anchor: str,
    client: QdrantClient | None = None,
) -> list[dict]:
    """Fetch code chunks from the same section as a doc chunk."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    base_anchor = ANCHOR_SPLIT_SUFFIX_PATTERN.sub("", section_anchor)

    response, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=build_file_filter(library_id, file_path, chunk_type="code"),
        with_payload=True,
        limit=50,
    )

    code_chunks = []
    for point in response:
        payload = point.payload or {}
        anchor = payload.get("section_anchor", "")
        if anchor == base_anchor or anchor.startswith(f"{base_anchor}-code-part"):
            code_chunks.append(
                {
                    "language": payload.get("metadata", {}).get("language", ""),
                    "content": payload.get("content", ""),
                }
            )

    if code_chunks:
        logger.debug(
            "Found %d code blocks for %s#%s (normalized to #%s)",
            len(code_chunks),
            file_path,
            section_anchor,
            base_anchor,
        )
    elif section_anchor != base_anchor:
        logger.debug(
            "No code blocks found for %s#%s (tried base anchor #%s)",
            file_path,
            section_anchor,
            base_anchor,
        )

    return code_chunks


async def search_bm25(
    library_id: str,
    query: str,
    limit: int = 10,
    chunk_types: list[str] | None = None,
    client: QdrantClient | None = None,
) -> list[models.ScoredPoint]:
    """BM25 (keyword) search only."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    query_filter = build_library_filter(library_id, chunk_types)

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=models.Document(text=query, model="Qdrant/bm25"),
        using="bm25",
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )

    return response.points


async def search_exact_match(
    library_id: str,
    query: str,
    limit: int = 10,
    chunk_types: list[str] | None = None,
    phrase_boost: float = 2.0,
    client: QdrantClient | None = None,
) -> list[models.ScoredPoint]:
    """BM25 search with phrase boost for exact API lookups."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    query_filter = build_library_filter(library_id, chunk_types)
    prefetch_limit = max(50, limit * 3)

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=models.Document(text=query, model="Qdrant/bm25"),
        using="bm25",
        limit=prefetch_limit,
        query_filter=query_filter,
        with_payload=True,
    )

    results = list(response.points)
    results = apply_exact_match_boosts(results, query, phrase_boost=phrase_boost)
    results = apply_search_boosts(results, settings, query=query)
    _sort_results(results, prefer_strong_lookup_paths=True)
    results = results[:limit]

    logger.info(
        "Exact match search for '%s...' in %s: %d results (phrase_boost=%s)",
        query[:30],
        library_id,
        len(results),
        phrase_boost,
    )

    return results
