"""Git-tracked JSONL authority for curated architectural decisions.

The journal is deliberately small and independent of SQLAlchemy.  It owns the
durable file contract (validation, locking, crash-safe replacement, and
content-hash based external-edit detection); :mod:`journal_projection` mirrors
that authority into RepoWise's derived decision tables.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = [
    "DECISIONS_JOURNAL_ENV",
    "DecisionAnchor",
    "DecisionJournal",
    "DecisionJournalConfigurationError",
    "DecisionJournalError",
    "DecisionJournalMutationDisabledError",
    "DecisionJournalSnapshot",
    "DecisionJournalValidationError",
    "JournalDecision",
    "decisions_journal_path",
    "resolve_decisions_journal_path",
]

DECISIONS_JOURNAL_ENV = "REPOWISE_DECISIONS_JOURNAL"

_DECISION_ID_RE = re.compile(r"^dec-[0-9a-f]{8}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_EXTERNAL_EDIT_RETRIES = 8

CrashHook = Callable[[str], None]


class DecisionJournalError(RuntimeError):
    """Base error for journal configuration, parsing, and mutation failures."""


class DecisionJournalConfigurationError(DecisionJournalError):
    """The path-valued journal setting is unsafe or malformed."""


class DecisionJournalValidationError(DecisionJournalError, ValueError):
    """A journal record or requested mutation violates the data contract."""


class DecisionJournalMutationDisabledError(DecisionJournalError):
    """A RepoWise mutation has no lossless representation in journal mode."""


def decisions_journal_path() -> Path | None:
    """Return the configured repo-relative journal path, evaluated now.

    Presence enables journal mode.  Absolute paths and parent traversal are
    rejected so one repository can never redirect another repository's
    decision authority.
    """

    raw = os.environ.get(DECISIONS_JOURNAL_ENV, "").strip()
    if not raw:
        return None
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise DecisionJournalConfigurationError(
            f"{DECISIONS_JOURNAL_ENV} must be a repository-relative path without '..'"
        )
    if not path.parts or str(path) in {"", "."}:
        raise DecisionJournalConfigurationError(f"{DECISIONS_JOURNAL_ENV} must name a JSONL file")
    return Path(*path.parts)


def resolve_decisions_journal_path(repo_root: str | Path) -> Path | None:
    """Resolve the configured journal underneath *repo_root*."""

    relative = decisions_journal_path()
    if relative is None:
        return None
    root = Path(repo_root).resolve()
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:  # also catches a symlinked parent escaping root
        raise DecisionJournalConfigurationError(
            f"{DECISIONS_JOURNAL_ENV} resolves outside the repository"
        ) from exc
    return resolved


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, *, field: str, required: bool) -> str | None:
    if value is None:
        if required:
            raise DecisionJournalValidationError(f"{field} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise DecisionJournalValidationError(f"{field} must be an ISO-8601 string or null")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DecisionJournalValidationError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise DecisionJournalValidationError(f"{field} must include a timezone")
    return text


def _required_text(value: object, field: str, *, preserve: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionJournalValidationError(f"{field} is required")
    return value if preserve else value.strip()


def _optional_decision_id(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _DECISION_ID_RE.fullmatch(value):
        raise DecisionJournalValidationError(f"{field} must be null or match dec-xxxxxxxx")
    return value


@dataclass(frozen=True, slots=True)
class DecisionAnchor:
    file: str
    symbol: str | None
    file_sha: str | None

    @classmethod
    def from_mapping(cls, value: object) -> DecisionAnchor:
        if not isinstance(value, Mapping):
            raise DecisionJournalValidationError("each anchor must be an object")
        file = _required_text(value.get("file"), "anchor.file")
        _validate_relative_anchor(file)
        symbol_value = value.get("symbol")
        if symbol_value is not None and (
            not isinstance(symbol_value, str) or not symbol_value.strip()
        ):
            raise DecisionJournalValidationError("anchor.symbol must be a non-empty string or null")
        symbol = symbol_value.strip() if isinstance(symbol_value, str) else None
        sha_value = value.get("file_sha")
        if sha_value is not None and (
            not isinstance(sha_value, str) or not _SHA256_RE.fullmatch(sha_value)
        ):
            raise DecisionJournalValidationError(
                "anchor.file_sha must be a lowercase SHA-256 hex digest or null"
            )
        return cls(file=file, symbol=symbol, file_sha=sha_value)

    def to_dict(self) -> dict[str, str | None]:
        return {"file": self.file, "symbol": self.symbol, "file_sha": self.file_sha}


@dataclass(frozen=True, slots=True)
class JournalDecision:
    id: str
    title: str
    decision: str
    why: str
    anchors: tuple[DecisionAnchor, ...]
    status: str
    supersedes: str | None
    superseded_by: str | None
    recorded_at: str
    confirmed_at: str | None

    @classmethod
    def from_mapping(cls, value: object) -> JournalDecision:
        if not isinstance(value, Mapping):
            raise DecisionJournalValidationError("each JSONL row must be an object")
        decision_id = _required_text(value.get("id"), "id")
        if not _DECISION_ID_RE.fullmatch(decision_id):
            raise DecisionJournalValidationError("id must match dec-xxxxxxxx")
        anchors_value = value.get("anchors")
        if not isinstance(anchors_value, list) or not anchors_value:
            raise DecisionJournalValidationError("anchors must be a non-empty array")
        anchors = tuple(DecisionAnchor.from_mapping(item) for item in anchors_value)
        if len({(a.file, a.symbol) for a in anchors}) != len(anchors):
            raise DecisionJournalValidationError("duplicate file/symbol anchors are not allowed")
        status = _required_text(value.get("status"), "status")
        if status not in {"active", "superseded"}:
            raise DecisionJournalValidationError("status must be active or superseded")
        return cls(
            id=decision_id,
            title=_required_text(value.get("title"), "title", preserve=True),
            decision=_required_text(value.get("decision"), "decision", preserve=True),
            why=_required_text(value.get("why"), "why", preserve=True),
            anchors=anchors,
            status=status,
            supersedes=_optional_decision_id(value.get("supersedes"), "supersedes"),
            superseded_by=_optional_decision_id(value.get("superseded_by"), "superseded_by"),
            recorded_at=_parse_timestamp(
                value.get("recorded_at"), field="recorded_at", required=True
            )
            or "",
            confirmed_at=_parse_timestamp(
                value.get("confirmed_at"), field="confirmed_at", required=False
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "confirmed_at": self.confirmed_at,
            "decision": self.decision,
            "id": self.id,
            "recorded_at": self.recorded_at,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "title": self.title,
            "why": self.why,
        }


@dataclass(frozen=True, slots=True)
class DecisionJournalSnapshot:
    path: Path
    content: bytes
    content_hash: str
    mtime_ns: int | None
    decisions: tuple[JournalDecision, ...]


@dataclass(slots=True)
class _JournalDocument:
    lines: list[bytes]
    records: list[tuple[int, JournalDecision]]

    @property
    def by_id(self) -> dict[str, JournalDecision]:
        return {record.id: record for _, record in self.records}

    def replace_record(self, decision: JournalDecision) -> None:
        for offset, (line_index, current) in enumerate(self.records):
            if current.id != decision.id:
                continue
            ending = b"\r\n" if self.lines[line_index].endswith(b"\r\n") else b"\n"
            self.lines[line_index] = _encode_decision(decision, ending=ending)
            self.records[offset] = (line_index, decision)
            return
        raise DecisionJournalValidationError(f"decision {decision.id!r} not found")

    def append_record(self, decision: JournalDecision) -> None:
        if self.lines and not self.lines[-1].endswith((b"\n", b"\r")):
            self.lines[-1] += b"\n"
        line_index = len(self.lines)
        self.lines.append(_encode_decision(decision))
        self.records.append((line_index, decision))

    def render(self) -> bytes:
        return b"".join(self.lines)


def _encode_decision(decision: JournalDecision, *, ending: bytes = b"\n") -> bytes:
    text = json.dumps(decision.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return text.encode("utf-8") + ending


def _parse_document(content: bytes, *, path: Path) -> _JournalDocument:
    try:
        lines = content.splitlines(keepends=True)
        if content and not lines:
            lines = [content]
        records: list[tuple[int, JournalDecision]] = []
        seen: set[str] = set()
        for line_number, raw in enumerate(lines, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DecisionJournalValidationError(
                    f"{path}:{line_number}: invalid UTF-8 JSON object"
                ) from exc
            try:
                decision = JournalDecision.from_mapping(payload)
            except DecisionJournalValidationError as exc:
                raise DecisionJournalValidationError(f"{path}:{line_number}: {exc}") from exc
            if decision.id in seen:
                raise DecisionJournalValidationError(
                    f"{path}:{line_number}: duplicate decision id {decision.id}"
                )
            seen.add(decision.id)
            records.append((line_number - 1, decision))
    except DecisionJournalValidationError:
        raise

    by_id = {record.id: record for _, record in records}
    for _, record in records:
        if record.supersedes is not None:
            old = by_id.get(record.supersedes)
            if old is None:
                raise DecisionJournalValidationError(
                    f"{path}: {record.id} supersedes missing decision {record.supersedes}"
                )
            if old.status != "superseded" or old.superseded_by != record.id:
                raise DecisionJournalValidationError(
                    f"{path}: supersession {record.id} -> {old.id} is not bidirectional"
                )
        if record.superseded_by is not None:
            successor = by_id.get(record.superseded_by)
            if successor is None or successor.supersedes != record.id:
                raise DecisionJournalValidationError(
                    f"{path}: superseded_by on {record.id} is not mirrored by its successor"
                )
        if record.status == "superseded" and record.superseded_by is None:
            raise DecisionJournalValidationError(
                f"{path}: superseded decision {record.id} must name superseded_by"
            )
        if record.status == "active" and record.superseded_by is not None:
            raise DecisionJournalValidationError(
                f"{path}: active decision {record.id} cannot name superseded_by"
            )
    return _JournalDocument(lines=lines, records=records)


def _validate_relative_anchor(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized.rstrip("/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise DecisionJournalValidationError(
            "anchor.file must be a non-empty repository-relative path without '..'"
        )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DecisionJournal:
    """Read and mutate one repository's configured decision journal."""

    def __init__(self, repo_root: str | Path, path: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve()
        if path is None:
            resolved = resolve_decisions_journal_path(self.repo_root)
            if resolved is None:
                raise DecisionJournalConfigurationError(
                    f"{DECISIONS_JOURNAL_ENV} is not configured"
                )
            self.path = resolved
        else:
            candidate = Path(path)
            self.path = (
                candidate.resolve(strict=False)
                if candidate.is_absolute()
                else (self.repo_root / candidate).resolve(strict=False)
            )
            try:
                self.path.relative_to(self.repo_root)
            except ValueError as exc:
                raise DecisionJournalConfigurationError(
                    "decision journal path resolves outside the repository"
                ) from exc
        self.lock_path = (self.repo_root / ".repowise" / "decisions-journal.lock").resolve(
            strict=False
        )
        try:
            self.lock_path.relative_to(self.repo_root)
        except ValueError as exc:
            raise DecisionJournalConfigurationError(
                "decision journal lock path resolves outside the repository"
            ) from exc

    def snapshot(self) -> DecisionJournalSnapshot:
        """Read and validate current bytes under the cooperating-writer lock."""

        with self._lock(exclusive=False):
            content = self._read_content()
            stat = self.path.stat() if self.path.exists() else None
        document = _parse_document(content, path=self.path)
        return DecisionJournalSnapshot(
            path=self.path,
            content=content,
            content_hash=hashlib.sha256(content).hexdigest(),
            mtime_ns=stat.st_mtime_ns if stat is not None else None,
            decisions=tuple(record for _, record in document.records),
        )

    def list(self) -> list[JournalDecision]:
        return list(self.snapshot().decisions)

    def get(self, decision_id: str) -> JournalDecision:
        for decision in self.snapshot().decisions:
            if decision.id == decision_id:
                return decision
        raise DecisionJournalValidationError(f"decision {decision_id!r} not found")

    def record(
        self,
        *,
        title: str,
        decision: str,
        why: str,
        anchors: Sequence[Mapping[str, Any]],
        supersedes: str | None = None,
        decision_id: str | None = None,
        crash_hook: CrashHook | None = None,
    ) -> JournalDecision:
        """Append a confirmed decision, optionally superseding an older one."""

        title = _required_text(title, "title")
        decision_text = _required_text(decision, "decision")
        why_text = _required_text(why, "why")
        stamped = self._stamp_anchors(anchors)
        requested_id = _optional_decision_id(decision_id, "id") if decision_id else None
        supersedes_id = _optional_decision_id(supersedes, "supersedes")

        def change(document: _JournalDocument) -> JournalDecision:
            existing = document.by_id
            new_id = requested_id or self._new_id(set(existing))
            if new_id in existing:
                raise DecisionJournalValidationError(f"decision {new_id!r} already exists")
            if supersedes_id is not None and supersedes_id not in existing:
                raise DecisionJournalValidationError(
                    f"supersedes target {supersedes_id!r} not found"
                )
            timestamp = _utc_now_iso()
            new = JournalDecision(
                id=new_id,
                title=title,
                decision=decision_text,
                why=why_text,
                anchors=stamped,
                status="active",
                supersedes=supersedes_id,
                superseded_by=None,
                recorded_at=timestamp,
                confirmed_at=timestamp,
            )
            if supersedes_id is not None:
                old = existing[supersedes_id]
                if old.superseded_by is not None:
                    raise DecisionJournalValidationError(
                        f"decision {old.id} is already superseded by {old.superseded_by}"
                    )
                document.replace_record(replace(old, status="superseded", superseded_by=new.id))
            document.append_record(new)
            return new

        return self._mutate(change, crash_hook=crash_hook)

    def confirm(
        self,
        decision_id: str,
        *,
        crash_hook: CrashHook | None = None,
    ) -> JournalDecision:
        """Re-stamp every anchor after the caller has verified the decision."""

        _optional_decision_id(decision_id, "id")

        def change(document: _JournalDocument) -> JournalDecision:
            current = document.by_id.get(decision_id)
            if current is None:
                raise DecisionJournalValidationError(f"decision {decision_id!r} not found")
            if current.status != "active":
                raise DecisionJournalValidationError(
                    f"cannot confirm superseded decision {decision_id}"
                )
            anchors = self._stamp_anchors([anchor.to_dict() for anchor in current.anchors])
            confirmed = replace(current, anchors=anchors, confirmed_at=_utc_now_iso())
            document.replace_record(confirmed)
            return confirmed

        return self._mutate(change, crash_hook=crash_hook)

    def supersede(
        self,
        decision_id: str,
        *,
        superseded_by: str,
        crash_hook: CrashHook | None = None,
    ) -> JournalDecision:
        """Atomically connect an existing successor to the decision it replaces."""

        _optional_decision_id(decision_id, "id")
        _optional_decision_id(superseded_by, "superseded_by")

        def change(document: _JournalDocument) -> JournalDecision:
            existing = document.by_id
            old = existing.get(decision_id)
            successor = existing.get(superseded_by)
            if old is None or successor is None:
                missing = decision_id if old is None else superseded_by
                raise DecisionJournalValidationError(f"decision {missing!r} not found")
            if old.id == successor.id:
                raise DecisionJournalValidationError("a decision cannot supersede itself")
            if old.superseded_by not in {None, successor.id}:
                raise DecisionJournalValidationError(
                    f"decision {old.id} is already superseded by {old.superseded_by}"
                )
            if successor.supersedes not in {None, old.id}:
                raise DecisionJournalValidationError(
                    f"decision {successor.id} already supersedes {successor.supersedes}"
                )
            updated_old = replace(old, status="superseded", superseded_by=successor.id)
            updated_successor = replace(successor, supersedes=old.id)
            document.replace_record(updated_old)
            document.replace_record(updated_successor)
            return updated_old

        return self._mutate(change, crash_hook=crash_hook)

    def replace_anchor_files(
        self,
        decision_id: str,
        files: Sequence[str],
        *,
        crash_hook: CrashHook | None = None,
    ) -> JournalDecision:
        """Replace governed files, preserving symbols on anchors that remain."""

        _optional_decision_id(decision_id, "id")
        if not files:
            raise DecisionJournalValidationError(
                "a journal decision must retain at least one affected file anchor"
            )

        def change(document: _JournalDocument) -> JournalDecision:
            current = document.by_id.get(decision_id)
            if current is None:
                raise DecisionJournalValidationError(f"decision {decision_id!r} not found")
            symbols = {anchor.file.rstrip("/"): anchor.symbol for anchor in current.anchors}
            requested: list[dict[str, Any]] = []
            for file in files:
                normalized = str(_validate_relative_anchor(file))
                requested.append({"file": file, "symbol": symbols.get(normalized)})
            anchors = self._stamp_anchors(requested)
            updated = replace(current, anchors=anchors, confirmed_at=_utc_now_iso())
            document.replace_record(updated)
            return updated

        return self._mutate(change, crash_hook=crash_hook)

    def lock_acquirable(self) -> bool:
        """Return whether an exclusive journal lock can be obtained now."""

        try:
            with self._lock(exclusive=True, blocking=False):
                return True
        except (BlockingIOError, OSError):
            return False

    def _new_id(self, existing: set[str]) -> str:
        for _ in range(64):
            candidate = f"dec-{secrets.token_hex(4)}"
            if candidate not in existing:
                return candidate
        raise DecisionJournalError("could not allocate a unique decision id")

    def _stamp_anchors(self, anchors: Sequence[Mapping[str, Any]]) -> tuple[DecisionAnchor, ...]:
        if not anchors:
            raise DecisionJournalValidationError("at least one anchor is required")
        stamped: list[DecisionAnchor] = []
        seen: set[tuple[str, str | None]] = set()
        for raw in anchors:
            if not isinstance(raw, Mapping):
                raise DecisionJournalValidationError("each anchor must be an object")
            raw_file = _required_text(raw.get("file"), "anchor.file")
            relative = _validate_relative_anchor(raw_file)
            full = (self.repo_root / Path(*relative.parts)).resolve(strict=False)
            try:
                full.relative_to(self.repo_root)
            except ValueError as exc:
                raise DecisionJournalValidationError(
                    f"anchor {raw_file!r} resolves outside the repository"
                ) from exc
            if not full.exists():
                raise DecisionJournalValidationError(f"anchor {raw_file!r} does not exist")
            symbol_value = raw.get("symbol")
            if symbol_value is not None and (
                not isinstance(symbol_value, str) or not symbol_value.strip()
            ):
                raise DecisionJournalValidationError(
                    "anchor.symbol must be a non-empty string or null"
                )
            symbol = symbol_value.strip() if isinstance(symbol_value, str) else None
            file_value = relative.as_posix() + ("/" if full.is_dir() else "")
            key = (file_value, symbol)
            if key in seen:
                continue
            seen.add(key)
            stamped.append(
                DecisionAnchor(
                    file=file_value,
                    symbol=symbol,
                    file_sha=_sha256_file(full) if full.is_file() else None,
                )
            )
        if not stamped:
            raise DecisionJournalValidationError("at least one unique anchor is required")
        return tuple(stamped)

    def _mutate(
        self,
        change: Callable[[_JournalDocument], JournalDecision],
        *,
        crash_hook: CrashHook | None,
    ) -> JournalDecision:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(exclusive=True):
            for _ in range(_MAX_EXTERNAL_EDIT_RETRIES):
                before = self._read_content()
                document = _parse_document(before, path=self.path)
                result = change(document)
                payload = document.render()
                if self._replace_if_unchanged(before, payload, crash_hook=crash_hook):
                    return result
        raise DecisionJournalError(
            "decision journal kept changing during mutation; retry after the external editor settles"
        )

    def _replace_if_unchanged(
        self,
        before: bytes,
        payload: bytes,
        *,
        crash_hook: CrashHook | None,
    ) -> bool:
        current_mode = self.path.stat().st_mode & 0o777 if self.path.exists() else 0o644
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        tmp_path = Path(tmp_name)
        try:
            os.fchmod(fd, current_mode)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if crash_hook is not None:
                crash_hook("before_replace")
            if self._read_content() != before:
                return False
            os.replace(tmp_path, self.path)
            self._fsync_parent()
            if crash_hook is not None:
                crash_hook("after_rename")
            return True
        finally:
            with suppress(FileNotFoundError):
                tmp_path.unlink()

    def _read_content(self) -> bytes:
        try:
            return self.path.read_bytes()
        except FileNotFoundError:
            return b""

    def _fsync_parent(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(self.path.parent, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @contextmanager
    def _lock(self, *, exclusive: bool, blocking: bool = True) -> Iterator[None]:
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - Unix is the supported runtime today
            raise DecisionJournalError("decision journal locking requires POSIX flock") from exc

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, operation)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
