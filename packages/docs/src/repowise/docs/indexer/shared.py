"""Shared helpers for doc-search indexing and retrieval."""

from __future__ import annotations

from qdrant_client import models


def build_library_filter(
    library_id: str,
    chunk_types: list[str] | None = None,
) -> models.Filter:
    """Build a Qdrant filter for a library, optionally constrained by chunk type."""
    conditions = [
        models.FieldCondition(
            key="library_id",
            match=models.MatchValue(value=library_id),
        ),
    ]

    if chunk_types:
        conditions.append(
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchAny(any=chunk_types),
            ),
        )

    return models.Filter(must=conditions)


def build_file_filter(
    library_id: str,
    file_path: str,
    chunk_type: str | None = None,
) -> models.Filter:
    """Build a Qdrant filter for a specific file within a library."""
    conditions = [
        models.FieldCondition(
            key="library_id",
            match=models.MatchValue(value=library_id),
        ),
        models.FieldCondition(
            key="file_path",
            match=models.MatchValue(value=file_path),
        ),
    ]

    if chunk_type is not None:
        conditions.append(
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchValue(value=chunk_type),
            ),
        )

    return models.Filter(must=conditions)


def build_stale_file_filter(
    library_id: str,
    file_path: str,
    keep_chunk_ids: list[str] | None = None,
) -> models.Filter:
    """Build a Qdrant filter for stale chunks in one file.

    This targets all points for the file except the currently valid chunk IDs.
    It allows the indexer to upsert replacement chunks first and then delete only
    stale leftovers, avoiding churn and preserving data on partial failures.
    """
    filter_payload = build_file_filter(library_id, file_path)
    if keep_chunk_ids:
        filter_payload.must_not = [
            models.HasIdCondition(has_id=keep_chunk_ids),
        ]
    return filter_payload
