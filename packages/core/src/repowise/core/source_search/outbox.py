"""Transactional source-index change capture and recovery state.

The queue lives in ``wiki.db`` because changed symbol bounds and the fact that
they need derived indexing must commit together.  FTS and Lance are consumers;
they are never allowed to infer a change set from whichever files happen to be
on disk after the SQL transaction has already advanced.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from repowise.core.ingestion.models import compute_content_hash
from repowise.core.ingestion.parse_cache import parser_fingerprint
from repowise.core.persistence.models import Repository, SourceIndexUpdate

from .chunks import parser_eligible, window_eligible
from .manifest import default_manifest_path, read_manifest

__all__ = [
    "BLOCKED",
    "BUILDING",
    "PENDING",
    "PUBLISHED",
    "READY",
    "SourceChange",
    "SourceUpdateRecord",
    "enqueue_full_update",
    "enqueue_incremental_update",
    "load_unpublished_updates",
    "mark_update_state",
    "mark_updates_ready",
    "suppress_incremental_paths",
]

PENDING = "pending"
BUILDING = "building"
READY = "ready"
PUBLISHED = "published"
BLOCKED = "blocked"

_SUPPRESSED_INCREMENTAL_PATHS: ContextVar[frozenset[str]] = ContextVar(
    "source_search_suppressed_incremental_paths",
    default=frozenset(),
)


@contextmanager
def suppress_incremental_paths(paths: set[str] | frozenset[str]) -> Iterator[None]:
    """Skip duplicate outbox changes already captured by the watch fast lane.

    The suppression is context-local and path-scoped.  A heavy watcher update
    can therefore omit only the saved paths the fast lane committed while
    still capturing unrelated commit-range changes discovered by its normal
    change detector.
    """

    normalised = frozenset(str(path).replace("\\", "/") for path in paths if path)
    current = _SUPPRESSED_INCREMENTAL_PATHS.get()
    token = _SUPPRESSED_INCREMENTAL_PATHS.set(current | normalised)
    try:
        yield
    finally:
        _SUPPRESSED_INCREMENTAL_PATHS.reset(token)


@dataclass(frozen=True, slots=True)
class SourceChange:
    """The durable, minimal account of one changed path."""

    path: str
    status: str
    old_path: str | None
    content_hash: str | None
    parse_state: str


@dataclass(frozen=True, slots=True)
class SourceUpdateRecord:
    sequence: int
    generation_id: str
    parent_generation_id: str | None
    mode: str
    state: str
    parser_fingerprint: str
    changes: tuple[SourceChange, ...]
    upstream_ready: bool
    attempts: int
    last_error: str | None
    artifact: dict[str, Any] | None

    @property
    def ref(self):
        from .generation import GenerationRef

        return GenerationRef(self.generation_id, self.sequence)


def _active_generation(repo_path: Path) -> str | None:
    manifest = read_manifest(default_manifest_path(repo_path))
    return manifest.generation_id if manifest is not None else None


def _raw_hash(path: Path) -> str | None:
    try:
        return compute_content_hash(path.read_bytes())
    except OSError:
        return None


def _change_records(
    repo_path: Path,
    file_diffs: list[Any] | None,
    parsed_files: list[Any] | None,
) -> list[SourceChange]:
    parsed = {pf.file_info.path: pf for pf in parsed_files or []}
    changes: list[SourceChange] = []
    for diff in file_diffs or []:
        path = str(getattr(diff, "path", "") or "").replace("\\", "/")
        if not path:
            continue
        status = str(getattr(diff, "status", "modified") or "modified")
        old_path = getattr(diff, "old_path", None)
        old_path = str(old_path).replace("\\", "/") if old_path else None
        if status == "deleted":
            changes.append(SourceChange(path, status, old_path, None, "deleted"))
            continue

        parsed_file = parsed.get(path) or getattr(diff, "new_parsed", None)
        explicit_hash = getattr(diff, "content_hash", None)
        content_hash = (
            str(explicit_hash)
            if explicit_hash
            else str(getattr(parsed_file, "content_hash", "") or "")
            or _raw_hash(repo_path / path)
        )
        # The unconditional window formats are authoritative from bytes even
        # when the AST parser has no grammar for them.  For code, absence from
        # parsed_files means the old chunks must survive and be marked stale.
        explicit_state = str(getattr(diff, "parse_state", "") or "")
        if explicit_state:
            parse_state = explicit_state
        elif parsed_file is not None:
            parse_state = "parsed"
        elif parser_eligible(path):
            parse_state = "failed"
        elif window_eligible(path, indexed_symbols=0):
            parse_state = "window"
        else:
            parse_state = "unindexed"
        changes.append(SourceChange(path, status, old_path, content_hash, parse_state))
    return sorted(changes, key=lambda change: (change.path, change.old_path or ""))


def _dedupe_key(
    *,
    parent_generation_id: str | None,
    mode: str,
    parser: str,
    changes: list[SourceChange],
) -> str:
    payload = {
        "parent_generation_id": parent_generation_id,
        "mode": mode,
        "parser_fingerprint": parser,
        "changes": [asdict(change) for change in changes],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


async def _enqueue(
    session: Any,
    repository_id: str,
    repo_path: Path,
    *,
    mode: str,
    changes: list[SourceChange],
    upstream_ready: bool,
    upstream_error: str | None,
) -> SourceIndexUpdate:
    parser = parser_fingerprint()
    parent = _active_generation(repo_path)
    dedupe = _dedupe_key(
        parent_generation_id=parent,
        mode=mode,
        parser=parser,
        changes=changes,
    )
    existing = (
        await session.execute(
            select(SourceIndexUpdate).where(
                SourceIndexUpdate.repository_id == repository_id,
                SourceIndexUpdate.dedupe_key == dedupe,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    row = SourceIndexUpdate(
        generation_id=uuid4().hex,
        repository_id=repository_id,
        parent_generation_id=parent,
        mode=mode,
        state=PENDING if upstream_ready else BLOCKED,
        dedupe_key=dedupe,
        parser_fingerprint=parser,
        change_set_json=json.dumps([asdict(change) for change in changes], sort_keys=True),
        upstream_ready=upstream_ready,
        last_error=upstream_error,
    )
    session.add(row)
    await session.flush()
    return row


async def enqueue_incremental_update(
    session: Any,
    repository_id: str,
    repo_path: Path | str,
    *,
    file_diffs: list[Any] | None,
    parsed_files: list[Any] | None,
    upstream_ready: bool = True,
    upstream_error: str | None = None,
) -> SourceIndexUpdate | None:
    """Capture one incremental change set in the caller's SQL transaction."""

    root = Path(repo_path).resolve()
    suppressed = _SUPPRESSED_INCREMENTAL_PATHS.get()
    changes = [
        change
        for change in _change_records(root, file_diffs, parsed_files)
        if change.path not in suppressed and (change.old_path or "") not in suppressed
    ]
    if not changes:
        return None
    return await _enqueue(
        session,
        repository_id,
        root,
        mode="incremental",
        changes=changes,
        upstream_ready=upstream_ready,
        upstream_error=upstream_error,
    )


async def enqueue_full_update(
    session: Any,
    repository_id: str,
    repo_path: Path | str | None = None,
    *,
    parsed_files: list[Any] | None = None,
    upstream_ready: bool = True,
    upstream_error: str | None = None,
) -> SourceIndexUpdate:
    """Capture a full-reconcile request in the caller's SQL transaction."""

    if repo_path is None:
        repo_path = (
            await session.execute(select(Repository.local_path).where(Repository.id == repository_id))
        ).scalar_one()
    root = Path(repo_path).resolve()
    # The consumer derives the complete corpus after the authoritative SQL
    # commit.  One digest salts deduplication without copying thousands of paths
    # into every full-reconcile row.
    snapshot = hashlib.sha256()
    for pf in sorted(parsed_files or [], key=lambda item: item.file_info.path):
        snapshot.update(pf.file_info.path.encode("utf-8"))
        snapshot.update(b"\0")
        snapshot.update(str(getattr(pf, "content_hash", "") or "").encode("ascii"))
        snapshot.update(b"\n")
    changes = [
        SourceChange(
            path="__full__",
            status="snapshot",
            old_path=None,
            content_hash=snapshot.hexdigest(),
            parse_state="parsed",
        )
    ]
    return await _enqueue(
        session,
        repository_id,
        root,
        mode="full",
        changes=changes,
        upstream_ready=upstream_ready,
        upstream_error=upstream_error,
    )


def _record(row: SourceIndexUpdate) -> SourceUpdateRecord:
    raw = json.loads(row.change_set_json or "[]")
    return SourceUpdateRecord(
        sequence=row.sequence,
        generation_id=row.generation_id,
        parent_generation_id=row.parent_generation_id,
        mode=row.mode,
        state=row.state,
        parser_fingerprint=row.parser_fingerprint,
        changes=tuple(SourceChange(**item) for item in raw),
        upstream_ready=row.upstream_ready,
        attempts=row.attempts,
        last_error=row.last_error,
        artifact=json.loads(row.artifact_json) if row.artifact_json else None,
    )


async def load_unpublished_updates(session: Any, repository_id: str) -> list[SourceUpdateRecord]:
    rows = (
        (
            await session.execute(
                select(SourceIndexUpdate)
                .where(
                    SourceIndexUpdate.repository_id == repository_id,
                    SourceIndexUpdate.state.in_([PENDING, BUILDING, READY, BLOCKED]),
                )
                .order_by(SourceIndexUpdate.sequence)
            )
        )
        .scalars()
        .all()
    )
    return [_record(row) for row in rows]


async def mark_update_state(
    session: Any,
    generation_ids: list[str],
    state: str,
    *,
    error: str | None = None,
) -> None:
    """Advance outbox rows; callers own the surrounding transaction."""

    if not generation_ids:
        return
    now = datetime.now(UTC)
    rows = (
        (
            await session.execute(
                select(SourceIndexUpdate).where(SourceIndexUpdate.generation_id.in_(generation_ids))
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        row.state = state
        row.last_error = error
        row.attempts += 1 if state == BUILDING else 0
        if state == READY:
            row.ready_at = now
        if state == PUBLISHED:
            row.published_at = now


async def mark_updates_ready(
    session: Any,
    generation_ids: list[str],
    artifact: dict[str, Any],
) -> None:
    """Persist the verified publication candidate before the manifest flip."""

    if not generation_ids:
        return
    rows = (
        (
            await session.execute(
                select(SourceIndexUpdate).where(SourceIndexUpdate.generation_id.in_(generation_ids))
            )
        )
        .scalars()
        .all()
    )
    payload = json.dumps(artifact, sort_keys=True)
    now = datetime.now(UTC)
    for row in rows:
        row.state = READY
        row.last_error = None
        row.artifact_json = payload
        row.ready_at = now
