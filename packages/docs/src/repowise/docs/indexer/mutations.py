"""Write and maintenance helpers for doc-search Qdrant indexing."""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from repowise.docs.chunking.models import DocChunk
from repowise.docs.config import get_settings
from repowise.docs.indexer.collection import get_qdrant_client
from repowise.docs.indexer.embedder import embed_texts
from repowise.docs.indexer.shared import (
    build_file_filter,
    build_library_filter,
    build_stale_file_filter,
)

logger = logging.getLogger(__name__)


async def upsert_chunks(
    chunks: list[DocChunk],
    client: QdrantClient | None = None,
) -> int:
    """Upsert document chunks with dense embeddings to Qdrant."""
    if not chunks:
        return 0

    if client is None:
        client = get_qdrant_client()

    settings = get_settings()

    logger.info("Embedding %d chunks...", len(chunks))
    texts = [c.content for c in chunks]
    embeddings = await embed_texts(texts)

    points: list[models.PointStruct] = []
    for chunk, embedding in zip(chunks, embeddings, strict=False):
        points.append(
            models.PointStruct(
                id=chunk.id,
                vector={
                    "dense": embedding,
                    "bm25": models.Document(
                        text=chunk.content,
                        model="Qdrant/bm25",
                    ),
                },
                payload=chunk.to_dict(),
            )
        )

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=batch,
            wait=True,
        )
        logger.debug("Upserted batch %d (%d points)", i // batch_size + 1, len(batch))

    logger.info("Upserted %d points to '%s'", len(points), settings.qdrant_collection)
    return len(points)


async def delete_by_file(
    library_id: str,
    file_path: str,
    client: QdrantClient | None = None,
) -> int:
    """Delete all chunks for a specific file."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    count_result = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=build_file_filter(library_id, file_path),
    )
    count = count_result.count

    if count == 0:
        return 0

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=build_file_filter(library_id, file_path),
        ),
        wait=True,
    )

    logger.info("Deleted %d chunks for %s:%s", count, library_id, file_path)
    return count


async def delete_stale_file_chunks(
    library_id: str,
    file_path: str,
    keep_chunk_ids: list[str],
    client: QdrantClient | None = None,
) -> int:
    """Delete stale chunks for one file while preserving the current chunk IDs."""
    if not keep_chunk_ids:
        return await delete_by_file(library_id, file_path, client)

    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    stale_filter = build_stale_file_filter(library_id, file_path, keep_chunk_ids)
    count_result = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=stale_filter,
    )
    count = count_result.count

    if count == 0:
        return 0

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(filter=stale_filter),
        wait=True,
    )
    logger.info(
        "Deleted %d stale chunks for %s:%s while preserving %d current chunks",
        count,
        library_id,
        file_path,
        len(keep_chunk_ids),
    )
    return count


async def list_indexed_file_paths(
    library_id: str,
    client: QdrantClient | None = None,
) -> set[str]:
    """Return the set of file paths currently indexed for a library."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    file_paths: set[str] = set()
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=build_library_filter(library_id),
            with_payload=["file_path"],
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            file_path = payload.get("file_path")
            if file_path:
                file_paths.add(str(file_path))
        if offset is None:
            break

    return file_paths


async def delete_files_outside_scope(
    library_id: str,
    valid_file_paths: set[str],
    client: QdrantClient | None = None,
) -> dict[str, int | list[str]]:
    """Delete indexed files that no longer belong to the library discovery scope."""
    if client is None:
        client = get_qdrant_client()

    indexed_file_paths = await list_indexed_file_paths(library_id, client)
    stale_paths = sorted(indexed_file_paths - valid_file_paths)
    if not stale_paths:
        return {"files_removed": 0, "chunks_removed": 0, "paths": []}

    chunks_removed = 0
    for file_path in stale_paths:
        chunks_removed += await delete_by_file(library_id, file_path, client)

    logger.info(
        "Deleted %d out-of-scope files for library %s (%d chunks)",
        len(stale_paths),
        library_id,
        chunks_removed,
    )
    return {
        "files_removed": len(stale_paths),
        "chunks_removed": chunks_removed,
        "paths": stale_paths,
    }


async def delete_by_library(
    library_id: str,
    client: QdrantClient | None = None,
) -> int:
    """Delete all chunks for a library."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    count_result = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=build_library_filter(library_id),
    )
    count = count_result.count

    if count == 0:
        return 0

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=build_library_filter(library_id),
        ),
        wait=True,
    )

    logger.info("Deleted %d chunks for library %s", count, library_id)
    return count


async def relabel_library_chunks(
    old_library_id: str,
    new_library_id: str,
    client: QdrantClient | None = None,
) -> int:
    """Move all Qdrant chunks from one logical library ID to another."""
    if old_library_id == new_library_id:
        return 0

    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    count_result = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=build_library_filter(old_library_id),
    )
    count = count_result.count
    if count == 0:
        return 0

    client.set_payload(
        collection_name=settings.qdrant_collection,
        payload={"library_id": new_library_id},
        points=models.FilterSelector(filter=build_library_filter(old_library_id)),
        wait=True,
    )
    logger.info(
        "Relabeled %d chunks from library %s to %s",
        count,
        old_library_id,
        new_library_id,
    )
    return count


async def get_chunk_count(
    library_id: str | None = None,
    client: QdrantClient | None = None,
) -> int:
    """Get the count of indexed chunks."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()

    if library_id:
        result = client.count(
            collection_name=settings.qdrant_collection,
            count_filter=build_library_filter(library_id),
        )
    else:
        result = client.count(collection_name=settings.qdrant_collection)

    return result.count
