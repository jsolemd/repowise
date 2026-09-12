"""Storage-oriented helpers for docs graph metadata sync."""

from __future__ import annotations

from collections.abc import Iterable

from repowise.docs.library.models import LibraryFile


def file_rows(files: Iterable[LibraryFile]) -> list[dict[str, object]]:
    return [
        {
            "file_path": file.file_path,
            "content_hash": file.content_hash,
            "chunk_count": file.chunk_count,
            "indexed_at": file.indexed_at.isoformat() if file.indexed_at else None,
            "updated_at": file.updated_at.isoformat() if file.updated_at else None,
        }
        for file in files
    ]


async def fetch_docs_graph_file_counts(client) -> dict[str, int]:
    rows = await client.execute_read(
        """
        MATCH (lib:DocLibrary)
        OPTIONAL MATCH (lib)-[:HAS_FILE]->(doc:DocFile)
        RETURN lib.library_id AS library_id, count(doc) AS file_count
        """
    )
    return {
        str(row["library_id"]): int(row.get("file_count") or 0)
        for row in rows
        if row.get("library_id")
    }
