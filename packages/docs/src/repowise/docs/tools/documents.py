"""Browse the published file inventory without loading document contents."""

from __future__ import annotations

from typing import Any

from repowise.docs.db_libraries import get_library
from repowise.docs.db_runtime import get_connection
from repowise.docs.tools.common import ToolError


async def list_document_page(
    library_id: str, query: str, offset: int, limit: int
) -> tuple[list[dict], int]:
    """Read a stable page and exact total from one database snapshot."""
    where = "library_id = $1 AND strpos(lower(file_path), lower($2)) > 0"
    async with get_connection() as conn:  # noqa: SIM117
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            total = await conn.fetchval(
                f"SELECT count(*) FROM doc_search.library_files WHERE {where}",
                library_id,
                query,
            )
            rows = await conn.fetch(
                f"SELECT file_path, chunk_count, indexed_at FROM doc_search.library_files WHERE {where} "
                "ORDER BY file_path LIMIT $3 OFFSET $4",
                library_id,
                query,
                limit,
                offset,
            )
    return [dict(row) for row in rows], int(total)


async def handle_list_documents(
    arguments: dict[str, Any], *, get_library_fn=None, list_page_fn=None
) -> dict:
    """List indexed files for either a Git or snapshot documentation source."""
    library_id = arguments["library_id"]
    library = await (get_library_fn or get_library)(library_id)
    if library is None:
        raise ToolError(f"Library not found: {library_id}")
    offset, limit = int(arguments.get("offset", 0)), int(arguments.get("limit", 50))
    if offset < 0 or not 1 <= limit <= 200:
        raise ToolError("offset must be nonnegative and limit must be between 1 and 200")
    rows, total = await (list_page_fn or list_document_page)(
        library_id, arguments.get("query", ""), offset, limit
    )
    return {
        "library_id": library_id,
        "library": {
            "name": library.name,
            "repo": library.repo,
            "branch": library.branch,
            "source_type": library.source_type,
            "status": library.status,
            "indexed_ref": library.current_sha,
            "indexed_at": library.indexed_at,
        },
        "files": rows,
        "pagination": {
            "total": total,
            "offset": offset,
            "limit": limit,
            "returned": len(rows),
            "has_more": offset + len(rows) < total,
            "next_offset": offset + len(rows) if offset + len(rows) < total else None,
        },
    }
