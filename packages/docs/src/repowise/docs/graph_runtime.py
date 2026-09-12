"""Retired docs-to-Neo4j runtime kept as a compatibility surface."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class DocsGraphSyncState:
    last_action: str | None = None
    last_library_id: str | None = None
    last_file_count: int | None = None
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None


CLIENT: Any | None = None
CLIENT_LOCK: asyncio.Lock | None = None
SCHEMA_READY = False
SYNC_STATE = DocsGraphSyncState()

SCHEMA_STATEMENTS = [
    (
        "doc_library_id_unique",
        "CREATE CONSTRAINT doc_library_id_unique IF NOT EXISTS "
        "FOR (lib:DocLibrary) REQUIRE lib.library_id IS UNIQUE",
    ),
    (
        "doc_file_unique",
        "CREATE CONSTRAINT doc_file_unique IF NOT EXISTS "
        "FOR (file:DocFile) REQUIRE (file.library_id, file.file_path) IS UNIQUE",
    ),
    (
        "doc_library_repo_idx",
        "CREATE INDEX doc_library_repo_idx IF NOT EXISTS FOR (lib:DocLibrary) ON (lib.repo)",
    ),
    (
        "doc_library_source_type_idx",
        "CREATE INDEX doc_library_source_type_idx IF NOT EXISTS "
        "FOR (lib:DocLibrary) ON (lib.source_type)",
    ),
    (
        "doc_library_source_subpath_idx",
        "CREATE INDEX doc_library_source_subpath_idx IF NOT EXISTS "
        "FOR (lib:DocLibrary) ON (lib.source_subpath)",
    ),
]


def get_client_lock() -> asyncio.Lock:
    global CLIENT_LOCK
    if CLIENT_LOCK is None:
        CLIENT_LOCK = asyncio.Lock()
    return CLIENT_LOCK


def docs_graph_enabled() -> bool:
    """Docs metadata graph sync is intentionally retired."""
    return False


def record_sync_success(*, action: str, library_id: str, file_count: int | None = None) -> None:
    SYNC_STATE.last_action = action
    SYNC_STATE.last_library_id = library_id
    SYNC_STATE.last_file_count = file_count
    SYNC_STATE.last_attempt_at = datetime.now(UTC)
    SYNC_STATE.last_success_at = SYNC_STATE.last_attempt_at
    SYNC_STATE.last_error = None


def record_sync_failure(*, action: str, library_id: str, error: str) -> None:
    SYNC_STATE.last_action = action
    SYNC_STATE.last_library_id = library_id
    SYNC_STATE.last_attempt_at = datetime.now(UTC)
    SYNC_STATE.last_error = error


async def get_docs_graph_client() -> None:
    """Return no client: the docs transport has no Neo4j dependency."""
    return None


async def close_docs_graph_client() -> None:
    global CLIENT, SCHEMA_READY
    async with get_client_lock():
        client = CLIENT
        CLIENT = None
        SCHEMA_READY = False
    if client is not None:
        await client.close()


async def ensure_docs_graph_schema(client: Any) -> None:
    """Compatibility no-op for callers that have not yet removed graph hooks."""
    _ = client
