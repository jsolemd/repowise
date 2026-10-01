"""Projection and write-through services for the decision JSONL journal."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, or_, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.analysis.decisions.journal import (
    CrashHook,
    DecisionAnchor,
    DecisionJournal,
    DecisionJournalConfigurationError,
    DecisionJournalError,
    JournalDecision,
    resolve_decisions_journal_path,
)
from repowise.core.analysis.decisions.semantic_match import (
    DECISION_VECTOR_PREFIX,
    decision_vector_item,
    upsert_decision_vectors,
)
from repowise.core.persistence.decision_graph import (
    sync_decision_node_links,
    upsert_decision_edge,
)
from repowise.core.persistence.models import (
    DecisionEdge,
    DecisionEvidence,
    DecisionNodeLink,
    DecisionRecord,
    Repository,
)

__all__ = [
    "DECISION_JOURNAL_EDGE_EVIDENCE",
    "DECISION_JOURNAL_SOURCE",
    "DecisionJournalHealth",
    "DecisionProjectionPendingError",
    "confirm_journal_decision",
    "journal_record",
    "record_journal_decision",
    "refresh_decision_journal",
    "replace_journal_anchor_files",
    "supersede_journal_decision",
]

DECISION_JOURNAL_SOURCE = "journal"
DECISION_JOURNAL_EDGE_EVIDENCE = "decision-journal"


@dataclass(frozen=True, slots=True)
class DecisionJournalHealth:
    path: str
    content_hash: str
    projected_count: int
    last_refresh: datetime
    lock_acquirable: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "projected_count": self.projected_count,
            "last_refresh": self.last_refresh.isoformat(),
            "lock_acquirable": self.lock_acquirable,
        }


_state_lock = threading.Lock()
_vector_hash_by_store: dict[tuple[str, int], str] = {}


def _as_datetime(value: str | None, *, fallback: datetime | None = None) -> datetime:
    if value is None:
        if fallback is None:
            raise DecisionJournalConfigurationError("journal timestamp is missing")
        return fallback
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _project_decision(record: DecisionRecord, decision: JournalDecision, staleness: float) -> None:
    """Write one journal entry's fields onto its projected row."""
    recorded_at = _as_datetime(decision.recorded_at)
    confirmed_at = (
        _as_datetime(decision.confirmed_at, fallback=recorded_at)
        if decision.confirmed_at is not None
        else None
    )
    updated_at = confirmed_at or recorded_at
    files = _affected_files(decision)
    record.title = decision.title
    # An unconfirmed row is a proposal, not a rule. Projecting it as
    # ``active`` would enrol it in every reader that counts governance —
    # the health summary, the decision block ``get_answer`` injects, the
    # alignment score — so the missing confirmation has to survive the
    # projection rather than being flattened away here. Only ``active`` is
    # remapped: a superseded record is history whether or not anyone ever
    # confirmed it.
    record.status = (
        "proposed"
        if decision.status == "active" and decision.confirmed_at is None
        else decision.status
    )
    record.context = ""
    record.decision = decision.decision
    record.rationale = decision.why
    record.alternatives_json = "[]"
    record.consequences_json = "[]"
    record.affected_files_json = json.dumps(files)
    record.affected_modules_json = "[]"
    record.tags_json = "[]"
    record.evidence_commits_json = "[]"
    record.source = DECISION_JOURNAL_SOURCE
    # Journal identity is the exact ``dec-xxxxxxxx`` id. Keeping a first
    # anchor in ``evidence_file`` would also engage the legacy
    # (repo,title,source,evidence_file) dedup constraint and make two
    # canonical ids with the same title/anchor impossible to project.
    # Anchors and affected_files retain the complete code linkage.
    record.evidence_file = None
    record.evidence_line = None
    record.confidence = 1.0
    record.verification = "exact"
    record.last_code_change = None
    record.staleness_score = staleness
    record.supersedes = decision.supersedes
    record.superseded_by = decision.superseded_by
    record.anchors_json = _anchors_json(decision)
    _set_instant(record, "confirmed_at", confirmed_at)
    _set_instant(record, "created_at", recorded_at)
    _set_instant(record, "updated_at", updated_at)


def _set_instant(record: DecisionRecord, name: str, value: datetime | None) -> None:
    """Assign a timestamp column only when it names a different instant.

    SQLite hands ``DateTime(timezone=True)`` columns back naive, while the
    journal parses to aware UTC. Python never finds a naive and an aware
    datetime equal, so a plain assignment dirtied every row and turned each
    refresh, including the one before every read, into a rewrite of the whole
    projection under the store's write lock.
    """
    stored = getattr(record, name)
    if stored is not None and value is not None:
        if stored.tzinfo is None:
            stored = stored.replace(tzinfo=UTC)
        if stored == value:
            return
    elif stored is None and value is None:
        return
    setattr(record, name, value)


def _anchors_json(decision: JournalDecision) -> str:
    return json.dumps(
        [anchor.to_dict() for anchor in decision.anchors],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _affected_files(decision: JournalDecision) -> list[str]:
    return list(dict.fromkeys(anchor.file for anchor in decision.anchors))


def _vector_item(decision: JournalDecision) -> tuple[str, str, dict] | None:
    return decision_vector_item(
        decision.id,
        title=decision.title,
        decision=decision.decision,
        evidence_file=decision.anchors[0].file if decision.anchors else None,
    )


def _projected_vector_item(record: DecisionRecord) -> tuple[str, str, dict] | None:
    """The item a projected row was embedded from, before this refresh rewrites it."""

    anchors = json.loads(record.anchors_json or "[]")
    return decision_vector_item(
        record.id,
        title=record.title,
        decision=record.decision or "",
        evidence_file=anchors[0].get("file") if anchors else None,
    )


def _anchor_stale(repo_root: Path, anchor: DecisionAnchor) -> bool:
    if anchor.file_sha is None:
        return False
    path = (repo_root / Path(*anchor.file.rstrip("/").split("/"))).resolve(strict=False)
    try:
        path.relative_to(repo_root)
    except ValueError:
        # A tracked symlink may have changed since confirmation. Never follow
        # an anchor outside the indexed repository merely to score staleness.
        return True
    if not path.is_file():
        return True
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return True
    return digest.hexdigest() != anchor.file_sha


def _staleness_score(repo_root: Path, decision: JournalDecision) -> float:
    hash_anchors = [anchor for anchor in decision.anchors if anchor.file_sha is not None]
    if not hash_anchors:
        return 0.0
    stale = sum(_anchor_stale(repo_root, anchor) for anchor in hash_anchors)
    return stale / len(hash_anchors)


def _staleness_scores(repo_root: Path, decisions: Sequence[JournalDecision]) -> dict[str, float]:
    """Hash journal anchors in one worker-thread task, outside the event loop."""

    return {decision.id: _staleness_score(repo_root, decision) for decision in decisions}


async def _repository(
    session: AsyncSession,
    repository_id: str,
    repo_root: str | Path | None,
) -> tuple[Repository, Path]:
    repository = await session.get(Repository, repository_id)
    if repository is None:
        raise DecisionJournalConfigurationError(f"repository {repository_id!r} not found")
    root = Path(repo_root or repository.local_path).resolve()
    return repository, root


async def refresh_decision_journal(
    session: AsyncSession,
    repository_id: str,
    *,
    repo_root: str | Path | None = None,
    vector_store: Any | None = None,
) -> DecisionJournalHealth | None:
    """Rebuild the derived SQL/vector projection from current journal bytes.

    The function intentionally checks the file on every call.  Reprojection is
    idempotent, so a process crash after the durable rename but before this
    function runs heals on the next read without a recovery command.
    """

    repository, root = await _repository(session, repository_id, repo_root)
    if resolve_decisions_journal_path(root) is None:
        return None

    journal = DecisionJournal(root)
    snapshot = await asyncio.to_thread(journal.snapshot)
    decisions = list(snapshot.decisions)
    decision_ids = {decision.id for decision in decisions}
    staleness_by_id = await asyncio.to_thread(_staleness_scores, root, decisions)

    projected = list(
        (
            await session.execute(
                select(DecisionRecord).where(
                    DecisionRecord.repository_id == repository.id,
                    DecisionRecord.source == DECISION_JOURNAL_SOURCE,
                )
            )
        )
        .scalars()
        .all()
    )
    projected_by_id = {record.id: record for record in projected}

    if decision_ids:
        collisions = list(
            (
                await session.execute(
                    select(DecisionRecord).where(
                        DecisionRecord.id.in_(decision_ids),
                        or_(
                            DecisionRecord.repository_id != repository.id,
                            DecisionRecord.source != DECISION_JOURNAL_SOURCE,
                        ),
                    )
                )
            )
            .scalars()
            .all()
        )
        if collisions:
            ids = ", ".join(sorted(record.id for record in collisions))
            raise DecisionJournalConfigurationError(
                f"journal ids collide with DecisionRecord rows outside this projection: {ids}"
            )

    removed_ids = sorted(set(projected_by_id) - decision_ids)
    if removed_ids:
        await session.execute(
            delete(DecisionEvidence).where(DecisionEvidence.decision_id.in_(removed_ids))
        )
        await session.execute(
            delete(DecisionNodeLink).where(DecisionNodeLink.decision_id.in_(removed_ids))
        )
        await session.execute(
            delete(DecisionEdge).where(
                or_(
                    DecisionEdge.src_decision_id.in_(removed_ids),
                    DecisionEdge.dst_decision_id.in_(removed_ids),
                )
            )
        )
        await session.execute(delete(DecisionRecord).where(DecisionRecord.id.in_(removed_ids)))

    # Read before the projection below overwrites the rows they come from.
    embedded = {
        item[0]: item
        for record in projected
        if (item := _projected_vector_item(record)) is not None
    }
    records: list[DecisionRecord] = []
    for decision in decisions:
        record = projected_by_id.get(decision.id)
        if record is None:
            record = DecisionRecord(id=decision.id, repository_id=repository.id)
            session.add(record)
        _project_decision(record, decision, staleness_by_id[decision.id])
        records.append(record)

    await session.flush()

    # Only a decision whose links differ is re-synced. sync_decision_node_links
    # replaces a decision's links wholesale, and this runs before every
    # decision read, so calling it for every record rewrote every link under
    # the store's write lock on reads that changed nothing.
    stored_links: dict[str, set[tuple[str, str]]] = {}
    link_rows = await session.execute(
        select(
            DecisionNodeLink.decision_id,
            DecisionNodeLink.node_id,
            DecisionNodeLink.link_type,
        ).where(DecisionNodeLink.repository_id == repository.id)
    )
    for decision_id, node_id, link_type in link_rows:
        stored_links.setdefault(decision_id, set()).add((node_id, link_type))
    for decision, record in zip(decisions, records, strict=True):
        files = _affected_files(decision)
        if stored_links.get(record.id, set()) == {(path, "file") for path in files if path}:
            continue
        await sync_decision_node_links(
            session,
            repository.id,
            record.id,
            files=files,
            modules=[],
        )

    # Diff rather than delete-and-reinsert: this runs before every decision
    # read, and rewriting unchanged edges took the store's write lock on reads
    # that changed nothing. upsert_decision_edge is idempotent on (src, dst,
    # kind), so re-stating a surviving edge writes nothing.
    wanted_edges = {
        (decision.id, decision.supersedes)
        for decision in decisions
        if decision.supersedes is not None
    }
    stored_edges = (
        await session.execute(
            select(DecisionEdge).where(
                DecisionEdge.repository_id == repository.id,
                DecisionEdge.kind == "supersedes",
                DecisionEdge.evidence == DECISION_JOURNAL_EDGE_EVIDENCE,
            )
        )
    ).scalars()
    for edge in stored_edges:
        if (edge.src_decision_id, edge.dst_decision_id) not in wanted_edges:
            await session.delete(edge)
    for decision in decisions:
        if decision.supersedes is None:
            continue
        await upsert_decision_edge(
            session,
            repository_id=repository.id,
            src_decision_id=decision.id,
            dst_decision_id=decision.supersedes,
            kind="supersedes",
            confidence=1.0,
            evidence=DECISION_JOURNAL_EDGE_EVIDENCE,
        )

    await session.flush()

    if vector_store is not None:
        vector_key = (str(snapshot.path), id(vector_store))
        with _state_lock:
            prior_vector_hash = _vector_hash_by_store.get(vector_key)
        items = [item for decision in decisions if (item := _vector_item(decision)) is not None]
        expected_vector_ids = {item[0] for item in items}
        stored_decision_ids: set[str] | None = None
        list_page_ids = getattr(vector_store, "list_page_ids", None)
        if callable(list_page_ids):
            try:
                stored_ids = await list_page_ids()
                stored_decision_ids = {
                    page_id for page_id in stored_ids if page_id.startswith(DECISION_VECTOR_PREFIX)
                }
            except Exception:
                # Vector projection is best-effort, matching the existing
                # decision embedding contract. The content-hash cache remains
                # a safe fallback for stores that cannot enumerate ids.
                stored_decision_ids = None

        if stored_decision_ids is None:
            pending = items if prior_vector_hash != snapshot.content_hash else []
        else:
            # Only a decision whose vector is missing or whose title, text or
            # first anchor changed is written. Re-embedding the whole journal on
            # every change rewrote ~150 unchanged rows per recorded decision:
            # 1,487 of SoleMD.Make's 3,357 wiki-table row writes on 2026-10-01.
            pending = [
                item
                for item in items
                if item[0] not in stored_decision_ids or embedded.get(item[0]) != item
            ]
        if pending:
            await upsert_decision_vectors(vector_store, pending)
        with _state_lock:
            _vector_hash_by_store[vector_key] = snapshot.content_hash

        stale_vector_ids = {f"{DECISION_VECTOR_PREFIX}{decision_id}" for decision_id in removed_ids}
        if stored_decision_ids is not None:
            stale_vector_ids.update(stored_decision_ids - expected_vector_ids)
        if stale_vector_ids:
            with contextlib.suppress(Exception):
                await vector_store.delete_many(sorted(stale_vector_ids))

    refreshed_at = datetime.now(UTC)
    lock_acquirable = await asyncio.to_thread(journal.lock_acquirable)
    return DecisionJournalHealth(
        path=str(snapshot.path),
        content_hash=snapshot.content_hash,
        projected_count=len(decisions),
        last_refresh=refreshed_at,
        lock_acquirable=lock_acquirable,
    )


class DecisionProjectionPendingError(DecisionJournalError):
    """The journal write landed; the store was too busy to project it yet.

    The JSONL is the authority and is already durable, so the write succeeded.
    Only the derived rows are behind, and every decision read re-projects the
    journal before answering, so they catch up on the next read without a
    retry. ``record`` is built from the journal entry and is not attached to
    the session. A caller must not repeat the write: the journal already has
    it, and a second ``record`` would mint a duplicate under a new id.
    """

    def __init__(self, record: DecisionRecord) -> None:
        super().__init__(
            f"decision {record.id} is in the journal; the index store was busy, "
            "so its projection catches up on the next read"
        )
        self.record = record


def _is_store_busy(exc: OperationalError) -> bool:
    return "database is locked" in str(exc.orig or exc)


async def journal_record(root: Path, repository_id: str, decision_id: str) -> DecisionRecord | None:
    """One decision as the projection would write it, read from the journal.

    The row is transient: built from the JSONL, never added to a session. It
    answers for a write whose projection is pending.
    """
    snapshot = await asyncio.to_thread(DecisionJournal(root).snapshot)
    decision = next((d for d in snapshot.decisions if d.id == decision_id), None)
    if decision is None:
        return None
    staleness = (await asyncio.to_thread(_staleness_scores, root, [decision]))[decision.id]
    record = DecisionRecord(id=decision.id, repository_id=repository_id)
    _project_decision(record, decision, staleness)
    return record


async def _project_written(
    session: AsyncSession,
    repository_id: str,
    root: Path,
    vector_store: Any | None,
    decision_id: str,
) -> DecisionRecord:
    """Project a journal write, or report it pending when the store is busy.

    An index update holds the store's single SQLite write lock for longer than
    the busy timeout. Failing the call then reported an error for a write the
    journal had already kept, and a caller that retried minted a duplicate.
    The savepoint keeps a failed projection out of the caller's transaction,
    so its commit has nothing left to write.
    """
    try:
        async with session.begin_nested():
            await refresh_decision_journal(
                session, repository_id, repo_root=root, vector_store=vector_store
            )
    except OperationalError as exc:
        if not _is_store_busy(exc):
            raise
        record = await journal_record(root, repository_id, decision_id)
        if record is None:
            raise
        raise DecisionProjectionPendingError(record) from exc
    record = await session.get(DecisionRecord, decision_id)
    if record is None:  # defensive: refresh either projects or raises
        raise DecisionJournalConfigurationError(f"journal decision {decision_id} was not projected")
    return record


async def _journal_for_write(
    session: AsyncSession,
    repository_id: str,
) -> tuple[DecisionJournal, Path]:
    _repository_row, root = await _repository(session, repository_id, None)
    if resolve_decisions_journal_path(root) is None:
        raise DecisionJournalConfigurationError("decision journal mode is not enabled")
    return DecisionJournal(root), root


async def record_journal_decision(
    session: AsyncSession,
    repository_id: str,
    *,
    title: str,
    decision: str,
    why: str,
    anchors: Sequence[Mapping[str, Any]],
    supersedes: str | None = None,
    confirmed: bool = True,
    vector_store: Any | None = None,
    crash_hook: CrashHook | None = None,
) -> DecisionRecord:
    journal, root = await _journal_for_write(session, repository_id)
    written = await asyncio.to_thread(
        journal.record,
        title=title,
        decision=decision,
        why=why,
        anchors=anchors,
        supersedes=supersedes,
        confirmed=confirmed,
        crash_hook=crash_hook,
    )
    return await _project_written(session, repository_id, root, vector_store, written.id)


async def confirm_journal_decision(
    session: AsyncSession,
    repository_id: str,
    decision_id: str,
    *,
    vector_store: Any | None = None,
    crash_hook: CrashHook | None = None,
) -> DecisionRecord:
    journal, root = await _journal_for_write(session, repository_id)
    await asyncio.to_thread(journal.confirm, decision_id, crash_hook=crash_hook)
    return await _project_written(session, repository_id, root, vector_store, decision_id)


async def supersede_journal_decision(
    session: AsyncSession,
    repository_id: str,
    decision_id: str,
    *,
    superseded_by: str,
    vector_store: Any | None = None,
    crash_hook: CrashHook | None = None,
) -> DecisionRecord:
    journal, root = await _journal_for_write(session, repository_id)
    await asyncio.to_thread(
        journal.supersede,
        decision_id,
        superseded_by=superseded_by,
        crash_hook=crash_hook,
    )
    return await _project_written(session, repository_id, root, vector_store, decision_id)


async def replace_journal_anchor_files(
    session: AsyncSession,
    repository_id: str,
    decision_id: str,
    files: Sequence[str],
    *,
    vector_store: Any | None = None,
    crash_hook: CrashHook | None = None,
) -> DecisionRecord:
    journal, root = await _journal_for_write(session, repository_id)
    await asyncio.to_thread(
        journal.replace_anchor_files,
        decision_id,
        files,
        crash_hook=crash_hook,
    )
    return await _project_written(session, repository_id, root, vector_store, decision_id)
