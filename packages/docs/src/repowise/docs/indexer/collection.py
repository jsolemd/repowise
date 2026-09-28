"""Qdrant collection management for doc-search."""

import logging
import threading

from qdrant_client import AsyncQdrantClient, models

from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "doc-search"
DOC_SEARCH_INDEXING_THRESHOLD = 10000
_CLIENT: AsyncQdrantClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_qdrant_client() -> AsyncQdrantClient:
    """Get a process-local Qdrant client instance."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            settings = get_settings()
            # The SDK's constructor version probe is synchronous even for its
            # async client. Versions are pinned by the worker/deployment; health
            # checks own connectivity, and all data requests stay cancellable.
            _CLIENT = AsyncQdrantClient(
                url=settings.qdrant_host, timeout=5, check_compatibility=False
            )
        return _CLIENT


async def close_qdrant_client() -> None:
    """Close the cached Qdrant client, primarily for tests and shutdown."""
    global _CLIENT
    with _CLIENT_LOCK:
        client = _CLIENT
        _CLIENT = None

    if client is not None:
        await client.close()


def _collection_updates(info: models.CollectionInfo, dimensions: int) -> dict:
    """Return only drift from the storage policy this module owns."""
    vectors = info.config.params.vectors
    dense = vectors.get("dense") if isinstance(vectors, dict) else None
    if dense is None or dense.size != dimensions or dense.distance != models.Distance.COSINE:
        raise ValueError(f"Docs collection requires a {dimensions}D cosine vector named 'dense'")
    updates = {}
    if dense.on_disk is not True:
        updates["vectors_config"] = {"dense": models.VectorParamsDiff(on_disk=True)}
    sparse = (info.config.params.sparse_vectors or {}).get("bm25")
    if (
        sparse is None
        or sparse.index is None
        or sparse.index.on_disk is not True
        or sparse.modifier != models.Modifier.IDF
    ):
        updates["sparse_vectors_config"] = {
            "bm25": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True),
                modifier=models.Modifier.IDF,
            )
        }
    if info.config.hnsw_config.on_disk is not True:
        updates["hnsw_config"] = models.HnswConfigDiff(on_disk=True)
    optimizer = info.config.optimizer_config
    if (
        optimizer.indexing_threshold != DOC_SEARCH_INDEXING_THRESHOLD
        or optimizer.memmap_threshold != DOC_SEARCH_INDEXING_THRESHOLD
    ):
        updates["optimizers_config"] = models.OptimizersConfigDiff(
            indexing_threshold=DOC_SEARCH_INDEXING_THRESHOLD,
            memmap_threshold=DOC_SEARCH_INDEXING_THRESHOLD,
        )
    return updates


async def ensure_collection(client: AsyncQdrantClient | None = None) -> None:
    """
    Create the doc-search collection if it doesn't exist.

    Sets up:
    - Dense vectors (768 dims, cosine similarity) for semantic search
    - BM25 sparse vectors for keyword search
    - Payload indexes for efficient filtering

    Args:
        client: Optional Qdrant client (creates one if not provided)
    """
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()

    if not await client.collection_exists(settings.qdrant_collection):
        logger.info(f"Creating collection '{settings.qdrant_collection}'")

        # Create collection with dense + sparse (BM25) vectors
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True),
                    modifier=models.Modifier.IDF,  # Use IDF for better BM25 scoring
                )
            },
            hnsw_config=models.HnswConfigDiff(on_disk=True),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=DOC_SEARCH_INDEXING_THRESHOLD,
                memmap_threshold=DOC_SEARCH_INDEXING_THRESHOLD,
            ),
            on_disk_payload=True,
        )

    # Read server state on every pass. This makes an unchanged refresh read-only
    # and also repairs a run interrupted between creation and payload indexes.
    info = await client.get_collection(settings.qdrant_collection)
    updates = _collection_updates(info, settings.embedding_dimensions)
    if updates:
        await client.update_collection(collection_name=settings.qdrant_collection, **updates)
    for name in ("library_id", "file_path", "chunk_type"):
        index = info.payload_schema.get(name)
        if index is not None and index.data_type != models.PayloadSchemaType.KEYWORD:
            raise ValueError(f"Docs payload index {name!r} must be keyword typed")
        if index is None:
            await client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    logger.info(
        "Collection '%s' ensured with dense (%sD) + BM25 vectors (disk-optimized)",
        settings.qdrant_collection,
        settings.embedding_dimensions,
    )


async def delete_collection(client: AsyncQdrantClient | None = None) -> bool:
    """
    Delete the doc-search collection if it exists.

    Args:
        client: Optional Qdrant client

    Returns:
        True if collection was deleted, False if it didn't exist
    """
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()

    if not await client.collection_exists(settings.qdrant_collection):
        logger.info(f"Collection '{settings.qdrant_collection}' does not exist")
        return False

    await client.delete_collection(collection_name=settings.qdrant_collection)
    logger.info(f"Deleted collection '{settings.qdrant_collection}'")
    return True


async def get_collection_info(client: AsyncQdrantClient | None = None) -> dict | None:
    """
    Get information about the doc-search collection.

    Args:
        client: Optional Qdrant client

    Returns:
        Collection info dict or None if collection doesn't exist
    """
    if client is None:
        client = get_qdrant_client()

    settings = get_settings()

    try:
        info = await client.get_collection(collection_name=settings.qdrant_collection)
        return {
            "name": settings.qdrant_collection,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
            "status": info.status.value if hasattr(info.status, "value") else str(info.status),
        }
    except Exception as e:
        logger.debug(
            f"Could not get collection info for '{settings.qdrant_collection}': "
            f"{type(e).__name__}: {e}"
        )
        return None
