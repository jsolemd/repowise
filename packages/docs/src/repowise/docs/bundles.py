"""Documentation bundle export/import for disaster recovery and sharing.

Inspired by CodeGraphContext's .cgc bundle format. Exports a library's
indexed chunks as a compressed JSON archive that can be restored without
re-indexing from source.

Bundle format (.docbundle.json.gz):
    {
        "version": 1,
        "library": { ... library config ... },
        "chunks": [ { ... chunk payload ... }, ... ],
        "exported_at": "2026-02-28T12:00:00Z",
        "chunk_count": 1234
    }
"""

import gzip
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import models

from repowise.docs.chunking.models import ChunkType, DocCategory, DocChunk, FileType
from repowise.docs.config import get_settings
from repowise.docs.db import SnapshotFileInput, get_library, list_snapshot_files, replace_snapshot
from repowise.docs.indexer import upsert_chunks
from repowise.docs.indexer.collection import get_qdrant_client

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 1


async def export_bundle(
    library_id: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Export a library's indexed chunks as a compressed bundle.

    Args:
        library_id: Library to export
        output_path: Optional output path (defaults to cache_root/bundles/)

    Returns:
        Dict with export metadata (path, chunk_count, size_bytes)
    """
    library = await get_library(library_id)
    if not library:
        raise ValueError(f"Library not found: {library_id}")

    settings = get_settings()
    client = get_qdrant_client()

    # Scroll through all chunks for this library
    chunks: list[dict] = []
    offset = None

    while True:
        response, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="library_id",
                        match=models.MatchValue(value=library_id),
                    ),
                ]
            ),
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # Skip vectors — we re-embed on import
        )

        for point in response:
            chunk_data = point.payload or {}
            chunk_data["_point_id"] = str(point.id)
            chunks.append(chunk_data)

        if next_offset is None:
            break
        offset = next_offset

    # Build bundle
    bundle = {
        "version": BUNDLE_VERSION,
        "library": {
            "library_id": library.library_id,
            "source_type": library.source_type.value,
            "name": library.name,
            "repo": library.repo,
            "source_subpath": library.source_subpath,
            "branch": library.branch,
            "docs_path": library.docs_path,
            "description": library.description,
        },
        "chunks": chunks,
        "snapshot_files": [],
        "exported_at": datetime.now(UTC).isoformat(),
        "chunk_count": len(chunks),
    }
    if library.source_type.value == "snapshot":
        bundle["snapshot_files"] = [
            {
                "file_path": file.file_path,
                "content": file.content,
                "source_url": file.source_url,
                "content_hash": file.content_hash,
            }
            for file in await list_snapshot_files(library_id)
        ]

    # Determine output path
    if output_path is None:
        bundle_dir = settings.cache_root / "bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        safe_name = library_id.strip("/").replace("/", "_")
        output_path = bundle_dir / f"{safe_name}.docbundle.json.gz"

    # Write compressed
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(bundle, f)

    size_bytes = output_path.stat().st_size

    logger.info(
        f"Exported bundle for {library_id}: {len(chunks)} chunks, "
        f"{size_bytes / 1024:.1f} KB -> {output_path}"
    )

    return {
        "library_id": library_id,
        "path": str(output_path),
        "chunk_count": len(chunks),
        "size_bytes": size_bytes,
    }


async def import_bundle(
    bundle_path: Path,
) -> dict[str, Any]:
    """
    Import a previously exported bundle, re-embedding and upserting chunks.

    The library must already exist in the database. This function:
    1. Reads the compressed bundle
    2. Re-embeds all chunk content via TEI
    3. Upserts to Qdrant with fresh embeddings

    Args:
        bundle_path: Path to .docbundle.json.gz file

    Returns:
        Dict with import metadata (library_id, chunk_count, status)
    """
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    # Read bundle
    with gzip.open(bundle_path, "rt", encoding="utf-8") as f:
        bundle = json.load(f)

    if bundle.get("version") != BUNDLE_VERSION:
        raise ValueError(
            f"Unsupported bundle version: {bundle.get('version')} (expected {BUNDLE_VERSION})"
        )

    library_info = bundle.get("library", {})
    library_id = library_info.get("library_id")
    chunks_data = bundle.get("chunks", [])
    snapshot_files = bundle.get("snapshot_files", [])

    if not library_id:
        raise ValueError("Bundle missing library_id")

    if not chunks_data:
        if snapshot_files:
            await replace_snapshot(
                library_id,
                [
                    SnapshotFileInput(
                        file_path=file["file_path"],
                        content=file["content"],
                        source_url=file.get("source_url"),
                    )
                    for file in snapshot_files
                ],
            )
        return {
            "library_id": library_id,
            "chunk_count": 0,
            "status": "empty_bundle",
        }

    # Convert payload dicts back to DocChunk objects for upsert
    doc_chunks: list[DocChunk] = []
    for chunk_data in chunks_data:
        # Remove internal fields
        chunk_data.pop("_point_id", None)

        try:
            # Parse enum values from stored strings
            chunk_type_str = chunk_data.get("chunk_type", "doc")
            try:
                chunk_type = ChunkType(chunk_type_str)
            except ValueError:
                chunk_type = ChunkType.DOC

            doc_category_str = chunk_data.get("doc_category", "other")
            try:
                doc_category = DocCategory(doc_category_str)
            except ValueError:
                doc_category = DocCategory.OTHER

            file_type_str = chunk_data.get("file_type", "other")
            try:
                file_type = FileType(file_type_str)
            except ValueError:
                file_type = FileType.OTHER

            doc_chunk = DocChunk(
                library_id=chunk_data.get("library_id", library_id),
                file_path=chunk_data.get("file_path", ""),
                commit_sha=chunk_data.get("commit_sha", "bundle-import"),
                chunk_type=chunk_type,
                content=chunk_data.get("content", ""),
                breadcrumb=chunk_data.get("breadcrumb", []),
                section_anchor=chunk_data.get("section_anchor", ""),
                source_url=chunk_data.get("source_url", ""),
                doc_category=doc_category,
                file_type=file_type,
                parent_chunk_id=chunk_data.get("parent_chunk_id"),
                metadata=chunk_data.get("metadata", {}),
                heading_level=chunk_data.get("heading_level"),
                token_count=chunk_data.get("token_count"),
                canonical_anchor=chunk_data.get("canonical_anchor"),
                line_start=chunk_data.get("line_start"),
                line_end=chunk_data.get("line_end"),
            )
            doc_chunks.append(doc_chunk)
        except Exception as e:
            logger.warning(f"Skipping malformed chunk: {e}")
            continue

    # Upsert in batches (this handles embedding + Qdrant upsert)
    batch_size = 50
    total_upserted = 0
    for i in range(0, len(doc_chunks), batch_size):
        batch = doc_chunks[i : i + batch_size]
        count = await upsert_chunks(batch)
        total_upserted += count
        logger.info(
            f"Import progress: {total_upserted}/{len(doc_chunks)} chunks "
            f"({total_upserted * 100 // len(doc_chunks)}%)"
        )

    if snapshot_files:
        await replace_snapshot(
            library_id,
            [
                SnapshotFileInput(
                    file_path=file["file_path"],
                    content=file["content"],
                    source_url=file.get("source_url"),
                )
                for file in snapshot_files
            ],
        )

    logger.info(f"Imported bundle for {library_id}: {total_upserted} chunks from {bundle_path}")

    return {
        "library_id": library_id,
        "chunk_count": total_upserted,
        "status": "imported",
        "source": str(bundle_path),
    }


async def list_bundles() -> list[dict[str, Any]]:
    """List available bundles in the cache directory."""
    settings = get_settings()
    bundle_dir = settings.cache_root / "bundles"

    if not bundle_dir.exists():
        return []

    bundles = []
    for path in sorted(bundle_dir.glob("*.docbundle.json.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                # Read only the first few fields without loading all chunks
                raw = f.read(4096)
                # Parse just enough to get metadata
                json.loads(raw + "]}")
        except Exception:
            # Fallback: just report file info
            pass

        bundles.append(
            {
                "path": str(path),
                "name": path.stem.replace(".docbundle.json", ""),
                "size_bytes": path.stat().st_size,
                "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
            }
        )

    return bundles
