"""Decision operations over the canonical JSONL journal.

Five verbs — ``record``, ``list``, ``get``, ``confirm``, ``supersede`` — sitting
directly on the store :mod:`repowise.core.analysis.decisions.journal` owns. This
module holds the store-facing half so the MCP tool above it stays argument
shaping and error rendering; nothing here imports FastMCP, so the verbs are
testable without a server.

Three rules the journal already enforces are re-stated here because this is the
first surface an *agent* can call, and each one is the reason a verb is shaped
the way it is:

**Recording is not confirming.** ``record`` appends with ``confirmed_at`` null,
which the projection reads as ``proposed``. An agent inferring a decision from a
diff has not reviewed it, and the store has to be able to tell that apart from a
rule the team stands behind. ``confirm`` is the separate verb that stamps it.

**History is append-only.** ``supersede`` links a successor to what it replaces
and flips the old record's status; the old record stays in the file, readable,
with the link pointing both ways. Nothing here deletes.

**A journal that cannot be written is not a journal.** Every mutating verb
refuses visibly when ``REPOWISE_DECISIONS_JOURNAL`` is unset, points outside the
repository, or names a path this process cannot write — it never falls back to
the derived SQLite table. A decision written only to a local derived store never
reaches git, so no teammate ever sees it; "recorded" would be a false report.
Reads keep working in that state and say the journal is unavailable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import case, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.analysis.decisions.journal import (
    DECISIONS_JOURNAL_ENV,
    DecisionJournalConfigurationError,
    DecisionJournalError,
    DecisionJournalMutationDisabledError,
    DecisionJournalValidationError,
    decisions_journal_path,
    resolve_decisions_journal_path,
)
from repowise.core.analysis.decisions.journal_projection import (
    DECISION_JOURNAL_SOURCE,
    confirm_journal_decision,
    record_journal_decision,
    refresh_decision_journal,
    supersede_journal_decision,
)
from repowise.core.persistence.models import DecisionRecord
from repowise.core.persistence.sql import LIKE_ESCAPE, escape_like

__all__ = [
    "DECISION_STATUSES",
    "DecisionNotFoundError",
    "DecisionOpsError",
    "DecisionOpsValidationError",
    "JournalUnavailableError",
    "confirm_decision",
    "get_decision",
    "journal_location",
    "journal_relative_path",
    "journal_status",
    "list_decisions",
    "parse_anchor",
    "record_decision",
    "supersede_decision",
]

log = structlog.get_logger(__name__)

#: Statuses a caller may filter ``list`` by. ``proposed`` is the projection's
#: reading of an unconfirmed journal row, not a status the JSONL spells.
DECISION_STATUSES = ("proposed", "active", "superseded")

#: Lower sorts first. A confirmed rule leads, a proposal awaiting review comes
#: next, and history sits last.
_STATUS_RANK = {"active": 0, "proposed": 1, "superseded": 2}

_ANCHOR_SYMBOL_SEPARATOR = "::"

_DISABLED_GUIDANCE = (
    f"Set {DECISIONS_JOURNAL_ENV} to a repository-relative JSONL path (for example "
    "'.repowise/decisions.jsonl') and make sure the working tree is writable. Reads "
    "keep working without it; writes do not, because a decision that never reaches "
    "the git-tracked journal is invisible to everyone else."
)


class DecisionOpsError(RuntimeError):
    """Base for failures a caller is meant to read and act on."""

    guidance: str | None = None


class JournalUnavailableError(DecisionOpsError):
    """The canonical journal is not configured, not resolvable, or not writable."""

    guidance = _DISABLED_GUIDANCE


class DecisionOpsValidationError(DecisionOpsError):
    """The requested mutation violates the journal's data contract."""


class DecisionNotFoundError(DecisionOpsError):
    """No journal decision carries the requested id."""


# ---------------------------------------------------------------------------
# Journal availability
# ---------------------------------------------------------------------------


def journal_relative_path() -> Path:
    """Return the configured journal path, or raise the visible-disabled error.

    Evaluated per call rather than cached: the env var is what enables journal
    mode, and a long-lived MCP server outlives any one reading of it.
    """
    try:
        configured = decisions_journal_path()
    except DecisionJournalConfigurationError as exc:
        raise JournalUnavailableError(
            f"The configured decision journal is unusable: {exc}"
        ) from exc
    if configured is None:
        raise JournalUnavailableError(
            f"Decision journal mode is off: {DECISIONS_JOURNAL_ENV} is not set, so there "
            "is no canonical store to write to."
        )
    return configured


def journal_location() -> str | None:
    """Return the configured journal path for a read to report, or ``None``.

    The non-raising counterpart of :func:`journal_relative_path`. Reads stay
    available when the journal is off — they serve the last projected state —
    but they have to say so, because "no decisions" and "no journal" are
    different answers and only one of them means the repository is ungoverned.
    """
    try:
        configured = decisions_journal_path()
    except DecisionJournalConfigurationError:
        return None
    return str(configured) if configured is not None else None


def journal_status(repo_root: str | Path) -> dict[str, str | bool | None]:
    """Report the configured journal path and whether this repository has it.

    Like :func:`journal_location`, this is a read-side disclosure and never
    raises for an unusable configuration. ``exists`` is deliberately separate
    from journal mode: one environment setting applies across a workspace, but
    each repository owns a different file beneath its own root.
    """
    try:
        relative = decisions_journal_path()
        resolved = resolve_decisions_journal_path(repo_root)
    except DecisionJournalConfigurationError:
        return {"relative": None, "path": None, "exists": False}
    return {
        "relative": str(relative) if relative is not None else None,
        "path": str(resolved) if resolved is not None else None,
        "exists": resolved.is_file() if resolved is not None else False,
    }


def _translate(exc: Exception) -> DecisionOpsError:
    """Map a journal-layer failure onto the error the caller should see.

    ``DecisionJournalValidationError`` subclasses the base error, so it is
    matched first; a contract violation is the caller's input being wrong, which
    is a different report from the store being unavailable.
    """
    if isinstance(exc, DecisionJournalValidationError):
        return DecisionOpsValidationError(str(exc))
    if isinstance(exc, DecisionJournalMutationDisabledError | DecisionJournalConfigurationError):
        return JournalUnavailableError(str(exc))
    if isinstance(exc, OSError):
        # The lock file, the temp file, and the durable rename all live under
        # the repository. A read-only checkout fails here rather than at the
        # configuration check, and it is the same answer to the caller: this
        # journal cannot be written right now.
        return JournalUnavailableError(f"The decision journal is not writable: {exc}")
    return JournalUnavailableError(f"The decision journal could not be updated: {exc}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_anchor(raw: str) -> dict[str, str | None]:
    """Parse ``path`` or ``path::Symbol`` into the journal's anchor mapping.

    ``::`` is the symbol separator the rest of RepoWise already uses for symbol
    ids, so an agent that just read ``packages/core/src/thing.py::Thing`` out of
    another tool can hand it straight back.
    """
    text = raw.strip()
    if not text:
        raise DecisionOpsValidationError("an anchor cannot be empty")
    file, separator, symbol = text.partition(_ANCHOR_SYMBOL_SEPARATOR)
    file = file.strip()
    symbol = symbol.strip()
    if not file:
        raise DecisionOpsValidationError(f"anchor {raw!r} names a symbol but no file")
    if separator and not symbol:
        raise DecisionOpsValidationError(f"anchor {raw!r} ends in '::' with no symbol after it")
    return {"file": file, "symbol": symbol or None}


def parse_timestamp(value: str, *, argument: str) -> datetime:
    """Parse an ISO-8601 instant or bare date into an aware UTC datetime.

    Raises rather than dropping the filter. A dropped time bound returns *more*
    rows than were asked for, and a caller reading a widened result as an answer
    is the failure this whole module is shaped around.
    """
    text = value.strip()
    if not text:
        raise DecisionOpsValidationError(f"{argument} cannot be empty")
    try:
        parsed: datetime | date = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = date.fromisoformat(text)
        except ValueError as exc:
            raise DecisionOpsValidationError(
                f"{argument}={value!r} is not an ISO-8601 timestamp or date "
                "(for example '2026-08-01' or '2026-08-01T12:00:00Z')"
            ) from exc
    if not isinstance(parsed, datetime):
        parsed = datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
    if parsed.tzinfo is None:
        # A bare local timestamp is read as UTC, matching how the journal
        # stamps its own; guessing the server's zone would make the same
        # filter select differently on two machines.
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _require(value: str | None, *, argument: str, action: str) -> str:
    text = (value or "").strip()
    if not text:
        raise DecisionOpsValidationError(f"{argument} is required to {action} a decision")
    return text


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _naive_utc(value: datetime) -> datetime:
    """Match the UTC-naive representation SQLite returns for this column."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    # SQLite drops timezone metadata from ``DateTime(timezone=True)`` values.
    # The projection writes UTC, so a naive read is UTC too — never host-local.
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def serialize(record: DecisionRecord, *, anchors: bool = True) -> dict[str, Any]:
    """Render one projected journal row as the payload shape every verb returns.

    ``status`` is the projection's, not the JSONL's: an unconfirmed row reads
    ``proposed`` here and ``active`` in the file. That is the point of the
    projection, and ``confirmed_at`` is carried alongside so a reader can see
    which of the two it is looking at.
    """
    payload: dict[str, Any] = {
        "id": record.id,
        "title": record.title,
        "decision": record.decision,
        "why": record.rationale,
        "status": record.status,
        "confirmed": record.confirmed_at is not None,
        "recorded_at": _iso(record.created_at),
        "confirmed_at": _iso(record.confirmed_at),
        "supersedes": record.supersedes,
        "superseded_by": record.superseded_by,
        "affected_files": json.loads(record.affected_files_json or "[]"),
        "staleness_score": record.staleness_score,
    }
    if anchors:
        payload["anchors"] = json.loads(record.anchors_json or "[]")
    return payload


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def _reconcile(
    session: AsyncSession,
    repository_id: str,
    *,
    repo_root: Path,
    vector_store: Any | None = None,
) -> None:
    """Rebuild the projection from current journal bytes before reading it.

    Unconditional, and cheap by design: this is the path that picks up an edit
    made to the JSONL outside RepoWise — a hand edit, a merge, a teammate's
    commit — so skipping it when nothing looked changed is how a read goes
    stale. :func:`refresh_decision_journal` is idempotent.
    """
    try:
        await refresh_decision_journal(
            session,
            repository_id,
            repo_root=repo_root,
            vector_store=vector_store,
        )
    except (DecisionJournalError, OSError) as exc:
        raise _translate(exc) from exc


def _order() -> tuple[Any, ...]:
    """A total order over projected journal rows.

    Status rank first, then newest, then id. The id tiebreak is load-bearing
    rather than defensive: the journal stamps whole seconds, so two decisions
    recorded in the same second carry identical timestamps, and without a final
    key two calls could return them in either order.
    """
    return (
        case(_STATUS_RANK, value=DecisionRecord.status, else_=len(_STATUS_RANK)),
        DecisionRecord.created_at.desc(),
        DecisionRecord.id.asc(),
    )


async def list_decisions(
    session: AsyncSession,
    repository_id: str,
    *,
    repo_root: Path,
    status: str | None = None,
    query: str | None = None,
    recorded_after: datetime | None = None,
    recorded_before: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
    vector_store: Any | None = None,
) -> tuple[list[DecisionRecord], int]:
    """Return one page of the canonical view plus the total the filters matched.

    Scoped to journal-sourced rows. In journal mode the extraction pipeline
    writes no candidates at all (``bulk_upsert_decisions`` short-circuits), so
    this scope is what makes the answer equal the file's contents rather than
    the file plus whatever a pre-journal index happened to leave behind.
    """
    await _reconcile(session, repository_id, repo_root=repo_root, vector_store=vector_store)

    conditions = [
        DecisionRecord.repository_id == repository_id,
        DecisionRecord.source == DECISION_JOURNAL_SOURCE,
    ]
    if status is not None:
        conditions.append(DecisionRecord.status == status)
    if recorded_after is not None:
        conditions.append(DecisionRecord.created_at >= _naive_utc(recorded_after))
    if recorded_before is not None:
        conditions.append(DecisionRecord.created_at <= _naive_utc(recorded_before))
    if query:
        needle = f"%{escape_like(query.strip())}%"
        conditions.append(
            or_(
                DecisionRecord.title.ilike(needle, escape=LIKE_ESCAPE),
                DecisionRecord.decision.ilike(needle, escape=LIKE_ESCAPE),
                DecisionRecord.rationale.ilike(needle, escape=LIKE_ESCAPE),
            )
        )

    matched = list(
        (await session.execute(select(DecisionRecord).where(*conditions).order_by(*_order())))
        .scalars()
        .all()
    )
    # Paged in Python over an already-scoped set. The journal is a hand-curated
    # file — hundreds of rows at the outside — so a second COUNT round-trip buys
    # nothing, and slicing the ordered list keeps the total and the page
    # guaranteed consistent with each other.
    return matched[offset : offset + limit], len(matched)


async def get_decision(
    session: AsyncSession,
    repository_id: str,
    decision_id: str,
    *,
    repo_root: Path,
    vector_store: Any | None = None,
) -> tuple[DecisionRecord, list[DecisionRecord]]:
    """Return one decision plus its full supersession chain, oldest first.

    The chain is walked in both directions from the requested record, so asking
    about any link returns the same history — which is what makes a superseded
    record still worth reading rather than a dead end.
    """
    await _reconcile(session, repository_id, repo_root=repo_root, vector_store=vector_store)

    record = await session.get(DecisionRecord, decision_id)
    if (
        record is None
        or record.repository_id != repository_id
        or record.source != DECISION_JOURNAL_SOURCE
    ):
        raise DecisionNotFoundError(f"No journal decision {decision_id!r} in this repository.")

    chain = [record]
    seen = {record.id}
    cursor = record
    while cursor.supersedes is not None and cursor.supersedes not in seen:
        older = await session.get(DecisionRecord, cursor.supersedes)
        if older is None:
            break
        chain.insert(0, older)
        seen.add(older.id)
        cursor = older
    cursor = record
    while cursor.superseded_by is not None and cursor.superseded_by not in seen:
        newer = await session.get(DecisionRecord, cursor.superseded_by)
        if newer is None:
            break
        chain.append(newer)
        seen.add(newer.id)
        cursor = newer
    return record, chain


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


async def record_decision(
    session: AsyncSession,
    repository_id: str,
    *,
    repo_root: Path,
    actor: str,
    title: str,
    decision: str,
    why: str,
    anchors: Sequence[str],
    supersedes: str | None = None,
    vector_store: Any | None = None,
) -> DecisionRecord:
    """Append an **unconfirmed** decision to the journal.

    Lands with ``confirmed_at`` null, which the projection reads as
    ``proposed``. Confirming is a separate act by a separate verb; this one
    cannot mint a rule the team stands behind, however it is called.
    """
    journal_relative_path()
    title = _require(title, argument="title", action="record")
    decision = _require(decision, argument="decision", action="record")
    why = _require(why, argument="why", action="record")
    actor = _require(actor, argument="actor", action="record")
    if not anchors:
        raise DecisionOpsValidationError(
            "at least one anchor is required — a decision governs specific code, and "
            "an unanchored record cannot be scored for staleness or found from a file"
        )
    parsed = [parse_anchor(anchor) for anchor in anchors]

    try:
        record = await record_journal_decision(
            session,
            repository_id,
            title=title,
            decision=decision,
            why=why,
            anchors=parsed,
            supersedes=supersedes,
            confirmed=False,
            vector_store=vector_store,
        )
    except (DecisionJournalError, OSError) as exc:
        raise _translate(exc) from exc

    log.info(
        "decisions.recorded",
        repository_id=repository_id,
        decision_id=record.id,
        actor=actor,
        supersedes=supersedes,
    )
    return record


async def confirm_decision(
    session: AsyncSession,
    repository_id: str,
    decision_id: str,
    *,
    repo_root: Path,
    actor: str,
    vector_store: Any | None = None,
) -> DecisionRecord:
    """Stamp a proposal as confirmed and re-hash the code it governs.

    Re-anchoring is the substance of the verb, not bookkeeping: confirming says
    the decision matches the code as it stands now, so the anchor hashes are
    re-taken and staleness restarts from this moment.
    """
    journal_relative_path()
    actor = _require(actor, argument="actor", action="confirm")
    await _reconcile(session, repository_id, repo_root=repo_root, vector_store=vector_store)
    existing = await session.get(DecisionRecord, decision_id)
    if (
        existing is None
        or existing.repository_id != repository_id
        or existing.source != DECISION_JOURNAL_SOURCE
    ):
        raise DecisionNotFoundError(f"No journal decision {decision_id!r} in this repository.")

    try:
        record = await confirm_journal_decision(
            session,
            repository_id,
            decision_id,
            vector_store=vector_store,
        )
    except (DecisionJournalError, OSError) as exc:
        raise _translate(exc) from exc

    log.info(
        "decisions.confirmed",
        repository_id=repository_id,
        decision_id=decision_id,
        actor=actor,
    )
    return record


async def supersede_decision(
    session: AsyncSession,
    repository_id: str,
    decision_id: str,
    *,
    superseded_by: str,
    repo_root: Path,
    actor: str,
    vector_store: Any | None = None,
) -> tuple[DecisionRecord, DecisionRecord]:
    """Link an existing successor to the decision it replaces.

    Both records must already be in the journal, and both stay in it: the old
    one flips to ``superseded`` and gains ``superseded_by``, the successor gains
    ``supersedes``, and the pair is written in a single atomic replacement.
    Returns ``(retired, successor)``.
    """
    journal_relative_path()
    actor = _require(actor, argument="actor", action="supersede")
    superseded_by = _require(superseded_by, argument="superseded_by", action="supersede")
    await _reconcile(session, repository_id, repo_root=repo_root, vector_store=vector_store)
    for candidate in (decision_id, superseded_by):
        existing = await session.get(DecisionRecord, candidate)
        if (
            existing is None
            or existing.repository_id != repository_id
            or existing.source != DECISION_JOURNAL_SOURCE
        ):
            raise DecisionNotFoundError(f"No journal decision {candidate!r} in this repository.")

    try:
        retired = await supersede_journal_decision(
            session,
            repository_id,
            decision_id,
            superseded_by=superseded_by,
            vector_store=vector_store,
        )
    except (DecisionJournalError, OSError) as exc:
        raise _translate(exc) from exc

    successor = await session.get(DecisionRecord, superseded_by)
    if successor is None:  # defensive: the projection either wrote it or raised
        raise DecisionNotFoundError(f"successor {superseded_by!r} was not projected")

    log.info(
        "decisions.superseded",
        repository_id=repository_id,
        decision_id=decision_id,
        superseded_by=superseded_by,
        actor=actor,
    )
    return retired, successor
