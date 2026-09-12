"""Current-state storage for snapshot-backed documentation libraries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import asyncpg

from repowise.docs.db_runtime import get_connection
from repowise.docs.library.models import SnapshotFile, SnapshotState


@dataclass(frozen=True)
class SnapshotFileInput:
    """Current file content to publish for a snapshot-backed library."""

    file_path: str
    content: str
    source_url: str | None = None


def normalize_snapshot_path(file_path: str) -> str:
    normalized = file_path.strip().replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("snapshot file_path must not be empty")
    return normalized


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_manifest_hash(items: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for file_path, content_hash in sorted(items):
        digest.update(file_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def compose_snapshot_source_ref(source_ref: str | None, manifest_hash: str) -> str:
    """Build a content-sensitive source ref for snapshot freshness checks.

    Scraped docs may have an upstream build ID or release label that does not change for
    every content-level edit. We fold the manifest hash into the stored source ref so the
    scheduler and incremental indexer always see content changes, even when the upstream
    label is coarse.
    """
    normalized = (source_ref or "").strip()
    if not normalized:
        return manifest_hash
    if normalized == manifest_hash or normalized.endswith(f":{manifest_hash}"):
        return normalized
    return f"{normalized}:{manifest_hash}"


def _row_to_snapshot_state(row: asyncpg.Record) -> SnapshotState:
    return SnapshotState(
        library_id=row["library_id"],
        source_ref=row["source_ref"],
        manifest_hash=row["manifest_hash"],
        file_count=row["file_count"],
        published_at=row["published_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_snapshot_file(row: asyncpg.Record) -> SnapshotFile:
    return SnapshotFile(
        library_id=row["library_id"],
        file_path=row["file_path"],
        content=row["content"],
        content_hash=row["content_hash"],
        source_url=row["source_url"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def get_snapshot_state(library_id: str) -> SnapshotState | None:
    async with get_connection() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM doc_search.snapshot_states WHERE library_id = $1",
            library_id,
        )
        return _row_to_snapshot_state(row) if row else None


async def list_snapshot_files(library_id: str) -> list[SnapshotFile]:
    async with get_connection() as conn:
        rows = await conn.fetch(
            "SELECT * FROM doc_search.snapshot_files WHERE library_id = $1 ORDER BY file_path",
            library_id,
        )
        return [_row_to_snapshot_file(row) for row in rows]


async def get_snapshot_file(library_id: str, file_path: str) -> SnapshotFile | None:
    normalized_path = normalize_snapshot_path(file_path)
    async with get_connection() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM doc_search.snapshot_files
            WHERE library_id = $1 AND file_path = $2
            """,
            library_id,
            normalized_path,
        )
        return _row_to_snapshot_file(row) if row else None


async def replace_snapshot(
    library_id: str,
    files: list[SnapshotFileInput],
    *,
    source_ref: str | None = None,
) -> tuple[SnapshotState, bool]:
    """Replace the current file set for a snapshot-backed library.

    Returns the resulting snapshot state and a boolean indicating whether the
    manifest changed.
    """
    normalized: dict[str, SnapshotFileInput] = {}
    for file in files:
        path = normalize_snapshot_path(file.file_path)
        normalized[path] = SnapshotFileInput(
            file_path=path,
            content=file.content,
            source_url=file.source_url,
        )

    hashed_items = [(path, compute_content_hash(file.content)) for path, file in normalized.items()]
    manifest_hash = compute_manifest_hash(hashed_items)
    effective_source_ref = compose_snapshot_source_ref(source_ref, manifest_hash)

    existing_state = await get_snapshot_state(library_id)
    if (
        existing_state
        and existing_state.manifest_hash == manifest_hash
        and existing_state.source_ref == effective_source_ref
        and existing_state.file_count == len(normalized)
    ):
        return existing_state, False

    async with get_connection() as conn, conn.transaction():
        file_rows = [
            (
                library_id,
                path,
                normalized[path].content,
                content_hash,
                normalized[path].source_url,
            )
            for path, content_hash in hashed_items
        ]
        if file_rows:
            await conn.executemany(
                """
                    INSERT INTO doc_search.snapshot_files (
                        library_id, file_path, content, content_hash, source_url
                    ) VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (library_id, file_path) DO UPDATE SET
                        content = EXCLUDED.content,
                        content_hash = EXCLUDED.content_hash,
                        source_url = EXCLUDED.source_url,
                        updated_at = NOW()
                    """,
                file_rows,
            )
            await conn.execute(
                """
                    DELETE FROM doc_search.snapshot_files
                    WHERE library_id = $1
                      AND NOT (file_path = ANY($2::text[]))
                    """,
                library_id,
                list(normalized.keys()),
            )
        else:
            await conn.execute(
                "DELETE FROM doc_search.snapshot_files WHERE library_id = $1",
                library_id,
            )

        await conn.execute(
            """
                INSERT INTO doc_search.snapshot_states (
                    library_id, source_ref, manifest_hash, file_count, published_at
                ) VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (library_id) DO UPDATE SET
                    source_ref = EXCLUDED.source_ref,
                    manifest_hash = EXCLUDED.manifest_hash,
                    file_count = EXCLUDED.file_count,
                    published_at = NOW(),
                    updated_at = NOW()
                """,
            library_id,
            effective_source_ref,
            manifest_hash,
            len(normalized),
        )

    state = await get_snapshot_state(library_id)
    if state is None:
        raise RuntimeError(f"Failed to persist snapshot state for {library_id}")
    return state, True
