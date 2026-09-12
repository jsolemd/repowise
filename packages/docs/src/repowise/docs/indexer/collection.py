"""Qdrant collection management for doc-search."""

import logging
import threading

from qdrant_client import QdrantClient, models

from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "doc-search"
DOC_SEARCH_INDEXING_THRESHOLD = 10000
_CLIENT: QdrantClient | None = None
_CLIENT_LOCK = threading.Lock()


def get_qdrant_client() -> QdrantClient:
    """Get a process-local Qdrant client instance."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            settings = get_settings()
            _CLIENT = QdrantClient(url=settings.qdrant_host)
        return _CLIENT


def close_qdrant_client() -> None:
    """Close the cached Qdrant client, primarily for tests and shutdown."""
    global _CLIENT
    with _CLIENT_LOCK:
        client = _CLIENT
        _CLIENT = None

    if client is not None:
        client.close()


def ensure_collection(client: QdrantClient | None = None) -> None:
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

    # Check if collection already exists
    collections = client.get_collections().collections
    exists = any(c.name == settings.qdrant_collection for c in collections)
    if exists:
        logger.info(f"Collection '{settings.qdrant_collection}' already exists")
        try:
            client.update_collection(
                collection_name=settings.qdrant_collection,
                vectors_config={
                    "dense": models.VectorParamsDiff(on_disk=True),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True),
                        modifier=models.Modifier.IDF,
                    )
                },
                hnsw_config=models.HnswConfigDiff(on_disk=True),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=DOC_SEARCH_INDEXING_THRESHOLD,
                    memmap_threshold=DOC_SEARCH_INDEXING_THRESHOLD,
                ),
            )
        except Exception as e:
            logger.warning(
                "Failed to apply disk optimization to existing collection '%s': %s",
                settings.qdrant_collection,
                e,
            )
    else:
        logger.info(f"Creating collection '{settings.qdrant_collection}'")

        # Create collection with dense + sparse (BM25) vectors
        client.create_collection(
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

    # Create payload indexes for efficient filtering
    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="library_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="file_path",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=settings.qdrant_collection,
        field_name="chunk_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    logger.info(
        "Collection '%s' ensured with dense (%sD) + BM25 vectors (disk-optimized)",
        settings.qdrant_collection,
        settings.embedding_dimensions,
    )


def delete_collection(client: QdrantClient | None = None) -> bool:
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

    collections = client.get_collections().collections
    if not any(c.name == settings.qdrant_collection for c in collections):
        logger.info(f"Collection '{settings.qdrant_collection}' does not exist")
        return False

    client.delete_collection(collection_name=settings.qdrant_collection)
    logger.info(f"Deleted collection '{settings.qdrant_collection}'")
    return True


def get_collection_info(client: QdrantClient | None = None) -> dict | None:
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
        info = client.get_collection(collection_name=settings.qdrant_collection)
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
