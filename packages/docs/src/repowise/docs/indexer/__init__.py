"""Indexer module for doc-search."""

from repowise.docs.indexer.collection import (
    COLLECTION_NAME,
    close_qdrant_client,
    delete_collection,
    ensure_collection,
    get_collection_info,
    get_qdrant_client,
)
from repowise.docs.indexer.embedder import embed_single, embed_texts
from repowise.docs.indexer.incremental import (
    ChangeSet,
    compute_changes,
    compute_file_hash,
)
from repowise.docs.indexer.mutations import (
    delete_by_file,
    delete_by_library,
    delete_files_outside_scope,
    delete_stale_file_chunks,
    get_chunk_count,
    list_indexed_file_paths,
    relabel_library_chunks,
    upsert_chunks,
)
from repowise.docs.indexer.search import (
    get_chunk_by_id,
    get_code_chunks_for_section,
    get_sibling_chunks,
    search_bm25,
    search_dense,
    search_exact_match,
    search_hybrid,
)

__all__ = [
    # Collection management
    "COLLECTION_NAME",
    "ChangeSet",
    "close_qdrant_client",
    "compute_changes",
    # Incremental indexing
    "compute_file_hash",
    "delete_by_file",
    "delete_by_library",
    "delete_collection",
    "delete_files_outside_scope",
    "delete_stale_file_chunks",
    "embed_single",
    # Embedding
    "embed_texts",
    "ensure_collection",
    "get_chunk_by_id",
    "get_chunk_count",
    "get_code_chunks_for_section",
    "get_collection_info",
    "get_qdrant_client",
    "get_sibling_chunks",
    "list_indexed_file_paths",
    "relabel_library_chunks",
    "search_bm25",
    "search_dense",
    "search_exact_match",
    # Search operations
    "search_hybrid",
    # Indexing operations
    "upsert_chunks",
]
