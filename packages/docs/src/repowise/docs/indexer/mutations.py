"""Write and maintenance helpers for doc-search Qdrant indexing."""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient, models

from repowise.docs.chunking.models import DocChunk
from repowise.docs.config import get_settings
from repowise.docs.indexer.collection import get_qdrant_client
from repowise.docs.indexer.embedder import embed_documents
from repowise.docs.indexer.shared import (
    build_file_filter,
    build_library_filter,
    build_stale_file_filter,
)

logger = logging.getLogger(__name__)


async def upsert_chunks(
    chunks: list[DocChunk],
    client: AsyncQdrantClient | None = None,
) -> int:
    """Upsert document chunks with dense embeddings to Qdrant."""
    if not chunks:
        return 0

    if client is None:
        client = get_qdrant_client()

    settings = get_settings()

    logger.info("Embedding %d chunks...", len(chunks))
    embeddings = await embed_documents(
        [c.content for c in chunks], [c.breadcrumb_text for c in chunks]
    )

    points: list[models.PointStruct] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
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
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=batch,
            wait=True,
        )
        logger.debug("Upserted batch %d (%d points)", i // batch_size + 1, len(batch))

    logger.info("Upserted %d points to '%s'", len(points), settings.qdrant_collection)
    return len(points)


async def reembed_all(client: AsyncQdrantClient | None = None, page_size: int = 256) -> int:
    """Re-embed every stored chunk with the current embedder, in place.

    Run once after the embedding model changes, with the worker stopped, so no
    query meets a mix of two models' vectors. Chunk text, payloads and BM25
    vectors stay as they are; only the dense vectors are replaced.
    """
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    total = 0
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=settings.qdrant_collection,
            limit=page_size,
            offset=offset,
            with_payload=["content", "breadcrumb_text"],
            with_vectors=False,
        )
        if points:
            payloads = [point.payload or {} for point in points]
            embeddings = await embed_documents(
                [str(payload.get("content") or "") for payload in payloads],
                [payload.get("breadcrumb_text") or None for payload in payloads],
            )
            await client.update_vectors(
                collection_name=settings.qdrant_collection,
                points=[
                    models.PointVectors(id=point.id, vector={"dense": embedding})
                    for point, embedding in zip(points, embeddings, strict=True)
                ],
                wait=True,
            )
            total += len(points)
            if total % 10_000 < len(points):
                logger.info("Re-embedded %d chunks", total)
        if offset is None:
            break

    logger.info("Re-embedded %d chunks in '%s'", total, settings.qdrant_collection)
    return total


async def delete_by_file(
    library_id: str,
    file_path: str,
    client: AsyncQdrantClient | None = None,
) -> int:
    """Delete all chunks for a specific file."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    count_result = await client.count(
        collection_name=settings.qdrant_collection,
        count_filter=build_file_filter(library_id, file_path),
    )
    count = count_result.count

    if count == 0:
        return 0

    await client.delete(
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
    client: AsyncQdrantClient | None = None,
) -> int:
    """Delete stale chunks for one file while preserving the current chunk IDs."""
    if not keep_chunk_ids:
        return await delete_by_file(library_id, file_path, client)

    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    stale_filter = build_stale_file_filter(library_id, file_path, keep_chunk_ids)
    count_result = await client.count(
        collection_name=settings.qdrant_collection,
        count_filter=stale_filter,
    )
    count = count_result.count

    if count == 0:
        return 0

    await client.delete(
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
    client: AsyncQdrantClient | None = None,
) -> set[str]:
    """Return the set of file paths currently indexed for a library."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    file_paths: set[str] = set()
    offset = None

    while True:
        points, offset = await client.scroll(
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
    client: AsyncQdrantClient | None = None,
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
    client: AsyncQdrantClient | None = None,
) -> int:
    """Delete all chunks for a library."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    count_result = await client.count(
        collection_name=settings.qdrant_collection,
        count_filter=build_library_filter(library_id),
    )
    count = count_result.count

    if count == 0:
        return 0

    await client.delete(
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
    client: AsyncQdrantClient | None = None,
) -> int:
    """Move all Qdrant chunks from one logical library ID to another."""
    if old_library_id == new_library_id:
        return 0

    if client is None:
        client = get_qdrant_client()

    settings = get_settings()
    count_result = await client.count(
        collection_name=settings.qdrant_collection,
        count_filter=build_library_filter(old_library_id),
    )
    count = count_result.count
    if count == 0:
        return 0

    await client.set_payload(
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
    client: AsyncQdrantClient | None = None,
) -> int:
    """Get the count of indexed chunks."""
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()

    if library_id:
        result = await client.count(
            collection_name=settings.qdrant_collection,
            count_filter=build_library_filter(library_id),
        )
    else:
        result = await client.count(collection_name=settings.qdrant_collection)

    return result.count
