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
from .manifest import SourceIndexManifest, default_manifest_path, read_manifest

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

_HISTORY_PAGE_SIZE = 128
_SUCCESSFUL_PARSE_STATES = frozenset({"parsed", "window"})


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


def _decode_changes(row: SourceIndexUpdate) -> tuple[SourceChange, ...] | None:
    """Decode one ledger change-set, failing closed for suppression purposes."""

    try:
        raw = json.loads(row.change_set_json or "[]")
    except (TypeError, ValueError):
        return None
    if not isinstance(raw, list):
        return None

    expected = {"path", "status", "old_path", "content_hash", "parse_state"}
    decoded: list[SourceChange] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != expected:
            return None
        try:
            change = SourceChange(**item)
        except (TypeError, ValueError):
            return None
        if (
            not isinstance(change.path, str)
            or not change.path
            or not isinstance(change.status, str)
            or not isinstance(change.parse_state, str)
            or (change.old_path is not None and not isinstance(change.old_path, str))
            or (change.content_hash is not None and not isinstance(change.content_hash, str))
        ):
            return None
        decoded.append(change)
    return tuple(decoded)


def _artifact_matches_manifest(
    row: SourceIndexUpdate,
    manifest: SourceIndexManifest,
) -> bool:
    """Whether *row* carries the complete artifact made active by *manifest*."""

    if not row.artifact_json:
        return False
    try:
        artifact = json.loads(row.artifact_json)
    except (TypeError, ValueError):
        return False
    return isinstance(artifact, dict) and artifact == manifest.to_dict()


def _row_matches_manifest(
    row: SourceIndexUpdate,
    manifest: SourceIndexManifest,
    parser: str,
) -> bool:
    """Whether *row* is the exact ledger witness for the active manifest."""

    return bool(
        row.sequence == manifest.generation_sequence
        and row.generation_id == manifest.generation_id
        and row.state in {READY, PUBLISHED}
        and row.upstream_ready
        and row.parser_fingerprint == parser
        and _artifact_matches_manifest(row, manifest)
    )


def _conservative_modified(change: SourceChange) -> bool:
    """Whether a change is safe to compare as an already-active file effect."""

    normalised_old = None if change.old_path in {None, "", change.path} else change.old_path
    return bool(
        change.status == "modified"
        and normalised_old is None
        and change.parse_state in _SUCCESSFUL_PARSE_STATES
        and change.content_hash
    )


def _same_active_effect(active: SourceChange, candidate: SourceChange) -> bool:
    """Compare the conservative semantic effect represented by two changes."""

    return bool(
        _conservative_modified(active)
        and _conservative_modified(candidate)
        and active.path == candidate.path
        and active.content_hash == candidate.content_hash
    )


async def _later_touched_paths(
    session: Any,
    repository_id: str,
    active_sequence: int,
    candidate_paths: set[str],
) -> set[str]:
    """Return candidates with any ledger work newer than the active manifest.

    A full or malformed later row can affect any path, so uncertainty blocks
    every candidate rather than manufacturing a no-op.
    """

    touched: set[str] = set()
    cursor = active_sequence
    while candidate_paths - touched:
        rows = list(
            (
                await session.execute(
                    select(SourceIndexUpdate)
                    .where(
                        SourceIndexUpdate.repository_id == repository_id,
                        SourceIndexUpdate.sequence > cursor,
                    )
                    .order_by(SourceIndexUpdate.sequence)
                    .limit(_HISTORY_PAGE_SIZE)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            break
        for row in rows:
            cursor = row.sequence
            changes = _decode_changes(row)
            if row.mode != "incremental" or changes is None:
                return set(candidate_paths)
            if any(change.path == "__full__" for change in changes):
                return set(candidate_paths)
            for change in changes:
                for path in (change.path, change.old_path):
                    if path in candidate_paths:
                        touched.add(path)
        if len(rows) < _HISTORY_PAGE_SIZE:
            break
    return touched


async def _filter_already_active_changes(
    session: Any,
    repository_id: str,
    repo_path: Path,
    changes: list[SourceChange],
    parser: str,
) -> tuple[list[SourceChange], SourceIndexManifest | None]:
    """Remove conservative no-op changes using the active manifest's ledger.

    The filesystem manifest remains the publication authority.  History is
    evidence only when its exact active row agrees with that manifest; every
    malformed, stale, concurrent, or otherwise uncertain case keeps the change.
    The returned manifest is a compare-and-bind token that the caller must
    recheck immediately before allocating a sequence.
    """

    manifest = read_manifest(default_manifest_path(repo_path))
    if manifest is None or manifest.generation_sequence <= 0:
        return changes, None

    active_row = (
        await session.execute(
            select(SourceIndexUpdate).where(
                SourceIndexUpdate.repository_id == repository_id,
                SourceIndexUpdate.sequence == manifest.generation_sequence,
                SourceIndexUpdate.generation_id == manifest.generation_id,
            )
        )
    ).scalar_one_or_none()
    if active_row is None or not _row_matches_manifest(active_row, manifest, parser):
        return changes, None

    by_path: dict[str, list[SourceChange]] = {}
    for change in changes:
        by_path.setdefault(change.path, []).append(change)
    candidates = {
        path: path_changes[0]
        for path, path_changes in by_path.items()
        if len(path_changes) == 1
        and path not in manifest.stale_files
        and _conservative_modified(path_changes[0])
    }
    if not candidates:
        return changes, None

    later_touched = await _later_touched_paths(
        session,
        repository_id,
        manifest.generation_sequence,
        set(candidates),
    )
    unresolved = set(candidates) - later_touched
    if not unresolved:
        return changes, None

    already_active: set[str] = set()
    cursor = manifest.generation_sequence + 1
    while unresolved:
        rows = list(
            (
                await session.execute(
                    select(SourceIndexUpdate)
                    .where(
                        SourceIndexUpdate.repository_id == repository_id,
                        SourceIndexUpdate.sequence < cursor,
                        SourceIndexUpdate.sequence <= manifest.generation_sequence,
                    )
                    .order_by(SourceIndexUpdate.sequence.desc())
                    .limit(_HISTORY_PAGE_SIZE)
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            break
        stop = False
        for row in rows:
            cursor = row.sequence
            visible_ready = bool(
                row.state == READY
                and row.upstream_ready
                and _artifact_matches_manifest(row, manifest)
            )
            if row.state != PUBLISHED and not visible_ready:
                stop = True
                break
            row_changes = _decode_changes(row)
            if row.mode != "incremental" or row_changes is None:
                stop = True
                break
            if any(change.path == "__full__" for change in row_changes):
                stop = True
                break

            for path in tuple(unresolved):
                effects = [
                    change
                    for change in row_changes
                    if change.path == path or change.old_path == path
                ]
                if not effects:
                    continue
                unresolved.remove(path)
                if (
                    len(effects) == 1
                    and row.parser_fingerprint == parser
                    and _same_active_effect(effects[0], candidates[path])
                ):
                    already_active.add(path)
            if not unresolved:
                break
        if stop or len(rows) < _HISTORY_PAGE_SIZE:
            break

    if not already_active:
        return changes, None
    return [change for change in changes if change.path not in already_active], manifest


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
            else str(getattr(parsed_file, "content_hash", "") or "") or _raw_hash(repo_path / path)
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
    *,
    mode: str,
    changes: list[SourceChange],
    upstream_ready: bool,
    upstream_error: str | None,
    parser: str,
    parent_generation_id: str | None,
) -> SourceIndexUpdate:
    dedupe = _dedupe_key(
        parent_generation_id=parent_generation_id,
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
        parent_generation_id=parent_generation_id,
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
    parser = parser_fingerprint()
    suppression_manifest: SourceIndexManifest | None = None
    original_changes = changes
    if upstream_ready and upstream_error is None:
        changes, suppression_manifest = await _filter_already_active_changes(
            session,
            repository_id,
            root,
            changes,
            parser,
        )

    # Bind suppression and parent selection to one final publication snapshot.
    # If the manifest moved during the SQL history scan, abandon durable
    # suppression and enqueue the original effects against the new parent.
    parent_manifest = read_manifest(default_manifest_path(root))
    if suppression_manifest is not None and parent_manifest != suppression_manifest:
        changes = original_changes
    if not changes:
        return None
    return await _enqueue(
        session,
        repository_id,
        mode="incremental",
        changes=changes,
        upstream_ready=upstream_ready,
        upstream_error=upstream_error,
        parser=parser,
        parent_generation_id=(
            parent_manifest.generation_id if parent_manifest is not None else None
        ),
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
            await session.execute(
                select(Repository.local_path).where(Repository.id == repository_id)
            )
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
        mode="full",
        changes=changes,
        upstream_ready=upstream_ready,
        upstream_error=upstream_error,
        parser=parser_fingerprint(),
        parent_generation_id=_active_generation(root),
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
