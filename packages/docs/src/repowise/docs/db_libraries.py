"""Library and library-file persistence for doc-search."""

from __future__ import annotations

from datetime import datetime

import asyncpg

from repowise.docs.db_runtime import get_connection
from repowise.docs.library.models import (
    LibraryConfig,
    LibraryFile,
    LibrarySourceType,
    LibraryState,
    LibraryStatus,
)

ALLOWED_UPDATE_FIELDS = {
    "status",
    "current_sha",
    "indexed_at",
    "freshness_checked_at",
    "next_freshness_check_at",
    "last_freshness_state",
    "last_remote_sha",
    "last_freshness_error",
    "chunk_count",
    "file_count",
    "error_message",
    "graph_synced_at",
    "graph_sync_error",
    "branch",
}


def _row_to_library_state(row: asyncpg.Record) -> LibraryState:
    """Convert a database row to LibraryState."""
    return LibraryState(
        library_id=row["library_id"],
        source_type=LibrarySourceType(row["source_type"])
        if row["source_type"]
        else LibrarySourceType.GIT,
        repo=row["repo"],
        name=row["name"],
        description=row["description"],
        source_subpath=row["source_subpath"] or "",
        docs_path=row["docs_path"],
        branch=row["branch"],
        include_patterns=list(row["include_patterns"]) if row["include_patterns"] else [],
        exclude_patterns=list(row["exclude_patterns"]) if row["exclude_patterns"] else [],
        priority=row["priority"] if row["priority"] is not None else 5,
        status=LibraryStatus(row["status"]),
        current_sha=row["current_sha"],
        indexed_at=row["indexed_at"],
        freshness_checked_at=row["freshness_checked_at"],
        next_freshness_check_at=row["next_freshness_check_at"],
        last_freshness_state=row["last_freshness_state"],
        last_remote_sha=row["last_remote_sha"],
        last_freshness_error=row["last_freshness_error"],
        error_message=row["error_message"],
        graph_synced_at=row["graph_synced_at"],
        graph_sync_error=row["graph_sync_error"],
        chunk_count=row["chunk_count"],
        file_count=row["file_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def get_library(library_id: str) -> LibraryState | None:
    """Get a library by ID."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM doc_search.libraries WHERE library_id = $1",
            library_id,
        )
        if row is None:
            return None
        return _row_to_library_state(row)


async def list_libraries(
    status: LibraryStatus | None = None,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[LibraryState]:
    """List all libraries, optionally filtered by status.

    ``offset`` and ``limit`` are pushed into the SQL so paginated reads
    don't materialize the full list client-side. When ``limit`` is
    ``None`` the entire (possibly filtered) list is returned.
    """
    params: list = []
    where_sql = ""
    if status is not None:
        params.append(status.value)
        where_sql = f" WHERE status = ${len(params)}"
    sql = f"SELECT * FROM doc_search.libraries{where_sql} ORDER BY name"
    if limit is not None:
        params.append(int(limit))
        sql += f" LIMIT ${len(params)}"
    if offset and offset > 0:
        params.append(int(offset))
        sql += f" OFFSET ${len(params)}"
    async with get_connection() as conn:
        rows = await conn.fetch(sql, *params)
    return [_row_to_library_state(row) for row in rows]


async def count_libraries(status: LibraryStatus | None = None) -> int:
    """Return the total number of libraries (optionally filtered)."""
    async with get_connection() as conn:
        if status is None:
            row = await conn.fetchrow("SELECT COUNT(*) AS n FROM doc_search.libraries")
        else:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM doc_search.libraries WHERE status = $1",
                status.value,
            )
        return int(row["n"]) if row else 0


async def upsert_library(config: LibraryConfig) -> LibraryState:
    """Insert or update a library from config."""
    library_id = config.library_id
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO doc_search.libraries (
                library_id, source_type, repo, name, description, source_subpath, docs_path, branch,
                include_patterns, exclude_patterns, priority
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (library_id) DO UPDATE SET
                source_type = EXCLUDED.source_type,
                repo = EXCLUDED.repo,
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                source_subpath = EXCLUDED.source_subpath,
                docs_path = EXCLUDED.docs_path,
                branch = EXCLUDED.branch,
                include_patterns = EXCLUDED.include_patterns,
                exclude_patterns = EXCLUDED.exclude_patterns,
                priority = EXCLUDED.priority
            WHERE
                doc_search.libraries.source_type IS DISTINCT FROM EXCLUDED.source_type
                OR doc_search.libraries.repo IS DISTINCT FROM EXCLUDED.repo
                OR doc_search.libraries.name IS DISTINCT FROM EXCLUDED.name
                OR doc_search.libraries.description IS DISTINCT FROM EXCLUDED.description
                OR doc_search.libraries.source_subpath IS DISTINCT FROM EXCLUDED.source_subpath
                OR doc_search.libraries.docs_path IS DISTINCT FROM EXCLUDED.docs_path
                OR doc_search.libraries.branch IS DISTINCT FROM EXCLUDED.branch
                OR doc_search.libraries.include_patterns IS DISTINCT FROM EXCLUDED.include_patterns
                OR doc_search.libraries.exclude_patterns IS DISTINCT FROM EXCLUDED.exclude_patterns
                OR doc_search.libraries.priority IS DISTINCT FROM EXCLUDED.priority
            RETURNING *
            """,
            library_id,
            config.source_type.value,
            config.repo,
            config.name,
            config.description,
            config.source_subpath,
            config.docs_path,
            config.branch,
            config.include_patterns,
            config.exclude_patterns,
            config.priority,
        )
        if row is not None:
            return _row_to_library_state(row)
    # ON CONFLICT WHERE clause filtered the update — row exists but unchanged
    result = await get_library(library_id)
    if result is None:
        raise RuntimeError(f"Failed to upsert library {library_id}")
    return result


async def migrate_library_alias(
    legacy_library_id: str,
    config: LibraryConfig,
) -> LibraryState | None:
    """Move a legacy library ID onto a new canonical library ID.

    This preserves existing file metadata and useful state while allowing the
    canonical library identity to change independently from the backing repo.
    """
    library_id = config.library_id
    if not library_id or legacy_library_id == library_id:
        return await get_library(library_id or legacy_library_id)

    async with get_connection() as conn:  # noqa: SIM117
        async with conn.transaction():
            legacy_row = await conn.fetchrow(
                "SELECT * FROM doc_search.libraries WHERE library_id = $1",
                legacy_library_id,
            )
            if legacy_row is None:
                return await get_library(library_id)

            await conn.execute(
                """
                INSERT INTO doc_search.libraries (
                    library_id, source_type, repo, name, description, source_subpath, docs_path, branch,
                    include_patterns, exclude_patterns, priority, status, current_sha,
                    indexed_at, freshness_checked_at, next_freshness_check_at,
                    last_freshness_state, last_remote_sha, last_freshness_error,
                    error_message, graph_synced_at, graph_sync_error, chunk_count,
                    file_count, created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8,
                    $9, $10, $11, $12, $13,
                    $14, $15, $16,
                    $17, $18, $19,
                    $20, $21, $22, $23,
                    $24, $25, $26
                )
                ON CONFLICT (library_id) DO UPDATE SET
                    source_type = EXCLUDED.source_type,
                    repo = EXCLUDED.repo,
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    source_subpath = EXCLUDED.source_subpath,
                    docs_path = EXCLUDED.docs_path,
                    branch = EXCLUDED.branch,
                    include_patterns = EXCLUDED.include_patterns,
                    exclude_patterns = EXCLUDED.exclude_patterns,
                    priority = EXCLUDED.priority,
                    status = CASE
                        WHEN doc_search.libraries.status = 'pending' AND EXCLUDED.status <> 'pending'
                            THEN EXCLUDED.status
                        ELSE doc_search.libraries.status
                    END,
                    current_sha = COALESCE(doc_search.libraries.current_sha, EXCLUDED.current_sha),
                    indexed_at = COALESCE(doc_search.libraries.indexed_at, EXCLUDED.indexed_at),
                    freshness_checked_at = COALESCE(
                        doc_search.libraries.freshness_checked_at,
                        EXCLUDED.freshness_checked_at
                    ),
                    next_freshness_check_at = COALESCE(
                        doc_search.libraries.next_freshness_check_at,
                        EXCLUDED.next_freshness_check_at
                    ),
                    last_freshness_state = COALESCE(
                        doc_search.libraries.last_freshness_state,
                        EXCLUDED.last_freshness_state
                    ),
                    last_remote_sha = COALESCE(doc_search.libraries.last_remote_sha, EXCLUDED.last_remote_sha),
                    last_freshness_error = CASE
                        WHEN COALESCE(doc_search.libraries.last_freshness_error, '') = ''
                            THEN EXCLUDED.last_freshness_error
                        ELSE doc_search.libraries.last_freshness_error
                    END,
                    error_message = CASE
                        WHEN COALESCE(doc_search.libraries.error_message, '') = ''
                            THEN EXCLUDED.error_message
                        ELSE doc_search.libraries.error_message
                    END,
                    graph_synced_at = COALESCE(doc_search.libraries.graph_synced_at, EXCLUDED.graph_synced_at),
                    graph_sync_error = CASE
                        WHEN COALESCE(doc_search.libraries.graph_sync_error, '') = ''
                            THEN EXCLUDED.graph_sync_error
                        ELSE doc_search.libraries.graph_sync_error
                    END,
                    chunk_count = GREATEST(doc_search.libraries.chunk_count, EXCLUDED.chunk_count),
                    file_count = GREATEST(doc_search.libraries.file_count, EXCLUDED.file_count),
                    created_at = LEAST(doc_search.libraries.created_at, EXCLUDED.created_at),
                    updated_at = GREATEST(doc_search.libraries.updated_at, EXCLUDED.updated_at)
                """,
                library_id,
                config.source_type.value,
                config.repo,
                config.name,
                config.description,
                config.source_subpath,
                config.docs_path,
                config.branch,
                config.include_patterns,
                config.exclude_patterns,
                config.priority,
                legacy_row["status"],
                legacy_row["current_sha"],
                legacy_row["indexed_at"],
                legacy_row["freshness_checked_at"],
                legacy_row["next_freshness_check_at"],
                legacy_row["last_freshness_state"],
                legacy_row["last_remote_sha"],
                legacy_row["last_freshness_error"],
                legacy_row["error_message"],
                legacy_row["graph_synced_at"],
                legacy_row["graph_sync_error"],
                legacy_row["chunk_count"],
                legacy_row["file_count"],
                legacy_row["created_at"],
                legacy_row["updated_at"],
            )

            await conn.execute(
                """
                INSERT INTO doc_search.library_files (
                    library_id, file_path, content_hash, chunk_count, indexed_at, created_at, updated_at
                )
                SELECT
                    $2, file_path, content_hash, chunk_count, indexed_at, created_at, updated_at
                FROM doc_search.library_files
                WHERE library_id = $1
                ON CONFLICT (library_id, file_path) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    chunk_count = GREATEST(
                        doc_search.library_files.chunk_count,
                        EXCLUDED.chunk_count
                    ),
                    indexed_at = COALESCE(
                        GREATEST(doc_search.library_files.indexed_at, EXCLUDED.indexed_at),
                        doc_search.library_files.indexed_at,
                        EXCLUDED.indexed_at
                    ),
                    updated_at = NOW()
                """,
                legacy_library_id,
                library_id,
            )

            await conn.execute(
                """
                INSERT INTO doc_search.graph_sync_jobs (
                    library_id, action, status, attempts, next_attempt_at, last_error, created_at, updated_at
                )
                SELECT
                    $2,
                    action,
                    'pending',
                    0,
                    NOW(),
                    NULL,
                    created_at,
                    NOW()
                FROM doc_search.graph_sync_jobs
                WHERE library_id = $1
                ON CONFLICT (library_id) DO UPDATE SET
                    action = CASE
                        WHEN doc_search.graph_sync_jobs.action = 'delete' OR EXCLUDED.action = 'delete'
                            THEN 'delete'
                        ELSE 'upsert'
                    END,
                    status = 'pending',
                    attempts = 0,
                    next_attempt_at = NOW(),
                    last_error = NULL,
                    updated_at = NOW()
                """,
                legacy_library_id,
                library_id,
            )

            await conn.execute(
                "DELETE FROM doc_search.graph_sync_jobs WHERE library_id = $1",
                legacy_library_id,
            )
            await conn.execute(
                "DELETE FROM doc_search.libraries WHERE library_id = $1",
                legacy_library_id,
            )

    return await get_library(library_id)


async def update_library_status(
    library_id: str,
    *,
    status: LibraryStatus | None = None,
    current_sha: str | None = None,
    indexed_at: datetime | None = None,
    freshness_checked_at: datetime | None = None,
    next_freshness_check_at: datetime | None = None,
    last_freshness_state: str | None = None,
    last_remote_sha: str | None = None,
    last_freshness_error: str | None = None,
    error_message: str | None = None,
    chunk_count: int | None = None,
    file_count: int | None = None,
    graph_synced_at: datetime | None = None,
    graph_sync_error: str | None = None,
    branch: str | None = None,
) -> LibraryState | None:
    """Update library indexing state."""
    update_fields: dict[str, object] = {}
    if status is not None:
        update_fields["status"] = status.value
    if current_sha is not None:
        update_fields["current_sha"] = current_sha
    if indexed_at is not None:
        update_fields["indexed_at"] = indexed_at
    if freshness_checked_at is not None:
        update_fields["freshness_checked_at"] = freshness_checked_at
    if next_freshness_check_at is not None:
        update_fields["next_freshness_check_at"] = next_freshness_check_at
    if last_freshness_state is not None:
        update_fields["last_freshness_state"] = last_freshness_state
    if last_remote_sha is not None:
        update_fields["last_remote_sha"] = last_remote_sha
    if last_freshness_error is not None:
        update_fields["last_freshness_error"] = last_freshness_error
    if error_message is not None:
        update_fields["error_message"] = error_message
    if chunk_count is not None:
        update_fields["chunk_count"] = chunk_count
    if file_count is not None:
        update_fields["file_count"] = file_count
    if graph_synced_at is not None:
        update_fields["graph_synced_at"] = graph_synced_at
    if graph_sync_error is not None:
        update_fields["graph_sync_error"] = graph_sync_error
    if branch is not None:
        update_fields["branch"] = branch

    for key in update_fields:
        if key not in ALLOWED_UPDATE_FIELDS:
            raise ValueError(f"Invalid field: {key}")

    updates = []
    values = []
    idx = 1
    for field, value in update_fields.items():
        updates.append(f"{field} = ${idx}")
        values.append(value)
        idx += 1

    if not updates:
        return await get_library(library_id)

    values.append(library_id)
    query = f"""
        UPDATE doc_search.libraries
        SET {", ".join(updates)}
        WHERE library_id = ${idx}
        RETURNING *
    """

    async with get_connection() as conn:
        row = await conn.fetchrow(query, *values)
        if row is not None:
            return _row_to_library_state(row)
    return None


async def mark_graph_sync_pending(
    library_id: str,
    *,
    error: str,
) -> LibraryState | None:
    """Clear the successful graph-sync marker and persist a pending reason."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            UPDATE doc_search.libraries
            SET graph_synced_at = NULL,
                graph_sync_error = $1,
                updated_at = NOW()
            WHERE library_id = $2
            RETURNING *
            """,
            error,
            library_id,
        )
        if row is not None:
            return _row_to_library_state(row)
    return None


async def delete_library(library_id: str) -> bool:
    """Delete a library and its files (cascade)."""
    async with get_connection() as conn:
        result = await conn.execute(
            "DELETE FROM doc_search.libraries WHERE library_id = $1",
            library_id,
        )
        return result == "DELETE 1"


def _row_to_library_file(row: asyncpg.Record) -> LibraryFile:
    """Convert a database row to LibraryFile."""
    return LibraryFile(
        library_id=row["library_id"],
        file_path=row["file_path"],
        content_hash=row["content_hash"],
        chunk_count=row["chunk_count"],
        indexed_at=row["indexed_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def get_library_files(library_id: str) -> list[LibraryFile]:
    """Get all files for a library."""
    async with get_connection() as conn:
        rows = await conn.fetch(
            "SELECT * FROM doc_search.library_files WHERE library_id = $1 ORDER BY file_path",
            library_id,
        )
        return [_row_to_library_file(row) for row in rows]


async def get_library_file(library_id: str, file_path: str) -> LibraryFile | None:
    """Get a specific file record."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM doc_search.library_files WHERE library_id = $1 AND file_path = $2",
            library_id,
            file_path,
        )
        if row is None:
            return None
        return _row_to_library_file(row)


async def upsert_library_file(
    library_id: str,
    file_path: str,
    content_hash: str,
    chunk_count: int = 0,
) -> LibraryFile:
    """Insert or update a library file record."""
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO doc_search.library_files (
                library_id, file_path, content_hash, chunk_count, indexed_at
            ) VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (library_id, file_path) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                chunk_count = EXCLUDED.chunk_count,
                indexed_at = NOW(),
                updated_at = NOW()
            RETURNING *
            """,
            library_id,
            file_path,
            content_hash,
            chunk_count,
        )
    if row is None:
        raise RuntimeError(f"Failed to upsert file {library_id}:{file_path}")
    return _row_to_library_file(row)


async def delete_library_file(library_id: str, file_path: str) -> bool:
    """Delete a library file record."""
    async with get_connection() as conn:
        result = await conn.execute(
            "DELETE FROM doc_search.library_files WHERE library_id = $1 AND file_path = $2",
            library_id,
            file_path,
        )
        return result == "DELETE 1"


async def delete_library_files(library_id: str) -> int:
    """Delete all files for a library. Returns count deleted."""
    async with get_connection() as conn:
        result = await conn.execute(
            "DELETE FROM doc_search.library_files WHERE library_id = $1",
            library_id,
        )
        return int(result.split()[-1]) if result.startswith("DELETE") else 0
