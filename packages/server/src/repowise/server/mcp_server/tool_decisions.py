"""MCP Tool: manage_decision — record, review, and retire architectural decisions.

The agent-facing half of :mod:`repowise.server.services.decisions_ops`. Five
verbs behind one tool name rather than five tools, because the default MCP
surface is deliberately small and these five are one workflow: propose a
decision, read what is already proposed, confirm it, retire it when something
replaces it.

The verb that writes cannot confirm. ``record`` lands a proposal an agent
inferred; ``confirm`` is the separate act of a person saying it is a rule. That
split is the whole reason this tool is safe to expose, so it is enforced in the
service layer rather than in this tool's arguments — there is no parameter here
that promotes a record on the way in.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from repowise.core.persistence.database import get_session
from repowise.core.registry import mcp_tool_registry as mcp
from repowise.server.mcp_server._helpers import (
    _get_repo,
    _is_workspace_mode,
    _resolve_repo_context,
    _unsupported_repo_all,
    attach_ignored_arguments,
    resolve_enum_argument,
)
from repowise.server.mcp_server._meta import build_meta as _build_meta
from repowise.server.services import decisions_ops as ops

#: The verbs, and what each one needs. Kept as data so the tool can name the
#: whole vocabulary when a caller misses, rather than failing one guess at a
#: time.
_ACTIONS: dict[str, str] = {
    "record": "append a new proposal (needs title, decision, why, anchors, actor)",
    "list": "page the canonical view (optional status, query, time range)",
    "get": "one decision plus its full supersession chain (needs decision_id)",
    "confirm": "mark a proposal reviewed and re-anchor it (needs decision_id, actor)",
    "supersede": "retire a decision in favour of another (needs decision_id, superseded_by, actor)",
}

_MUTATING = frozenset({"record", "confirm", "supersede"})

#: Page size ceiling. The journal is hand-curated and small; this exists to keep
#: one call from filling an agent's context with the entire governance history.
_MAX_LIMIT = 200


def _error(message: str, *, guidance: str | None = None, **extra: Any) -> dict[str, Any]:
    """Shape a failure as a readable response rather than a protocol error.

    An MCP-level isError teaches an agent to abandon the server for the rest of
    the session, so every failure this tool can anticipate comes back as a
    normal payload that says what went wrong and what would work instead.
    """
    payload: dict[str, Any] = {"error": message}
    if guidance:
        payload["guidance"] = guidance
    payload.update(extra)
    payload["_meta"] = _build_meta()
    return payload


@mcp.tool(default=False)
async def manage_decision(
    action: str,
    decision_id: str | None = None,
    title: str | None = None,
    decision: str | None = None,
    why: str | None = None,
    anchors: list[str] | None = None,
    supersedes: str | None = None,
    superseded_by: str | None = None,
    actor: str | None = None,
    status: str | None = None,
    query: str | None = None,
    recorded_after: str | None = None,
    recorded_before: str | None = None,
    limit: int = 50,
    offset: int = 0,
    repo: str | None = None,
) -> dict[str, Any]:
    """Record, review, confirm, and retire this repository's architectural decisions.

    Backed by a git-tracked JSONL journal, so every write lands as a diff a
    person reviews and commits.

    ``record`` always lands ``proposed`` — call it for a non-obvious choice the
    next reader would otherwise re-derive, not to restate what the code says.
    ``confirm`` promotes it to a rule and is a person's call; only do it when
    asked. ``supersede`` retires a decision for another already recorded, and
    both stay readable. Writes refuse visibly when the journal is off.

    Args:
        action: record, list, get, confirm, or supersede.
        decision_id: the ``dec-xxxxxxxx`` id (get / confirm / supersede).
        title: short name (record).
        decision: what was chosen (record).
        why: what forced the choice, the part not re-derivable (record).
        anchors: repo-relative files governed, ``path`` or ``path::Symbol``;
            at least one, each must exist (record).
        supersedes: id this record replaces, retired in the same write (record).
        superseded_by: id of the successor (supersede).
        actor: who is asking. Required on every write.
        status: filter — proposed, active, superseded (list).
        query: case-insensitive match over title, decision, why (list).
        recorded_after / recorded_before: ISO-8601 bounds (list).
        limit: page size, capped at 200 (list).
        offset: rows to skip (list).
        repo: workspace repo alias; ``all`` is not supported.
    """
    started = time.perf_counter()

    verb = (action or "").strip().lower()
    if verb not in _ACTIONS:
        # Never resolved leniently. A dropped filter still answers the question
        # asked; a dropped *verb* would have to guess whether the caller meant
        # to read or to write.
        return _error(
            f"Unknown action {action!r}.",
            guidance="Pass one of: " + "; ".join(f"{k} — {v}" for k, v in _ACTIONS.items()),
        )

    if repo == "all":
        return _error(_unsupported_repo_all("manage_decision")["error"])

    ignored: list[dict[str, Any]] = []
    resolved_status = resolve_enum_argument(
        status,
        ops.DECISION_STATUSES,
        argument="status",
        ignored=ignored,
    )

    try:
        after = (
            ops.parse_timestamp(recorded_after, argument="recorded_after")
            if recorded_after
            else None
        )
        before = (
            ops.parse_timestamp(recorded_before, argument="recorded_before")
            if recorded_before
            else None
        )
        if after is not None and before is not None and after > before:
            raise ops.DecisionOpsValidationError(
                f"recorded_after ({recorded_after}) is later than recorded_before "
                f"({recorded_before}), so no decision can match."
            )
    except ops.DecisionOpsValidationError as exc:
        return _error(str(exc))

    # The disabled check runs before any repository or session work: a caller
    # whose journal is off should learn that, not a database error from three
    # layers down.
    if verb in _MUTATING:
        try:
            ops.journal_relative_path()
        except ops.JournalUnavailableError as exc:
            return _error(str(exc), guidance=exc.guidance, journal_available=False)

    try:
        ctx = await _resolve_repo_context(repo)
    except LookupError as exc:
        return _error(str(exc))

    repository: Any = None
    try:
        async with get_session(ctx.session_factory) as session:
            repository = await _get_repo(session, None if _is_workspace_mode() else repo)
            payload = await _dispatch(
                verb,
                session=session,
                repository_id=repository.id,
                ctx=ctx,
                decision_id=decision_id,
                title=title,
                decision=decision,
                why=why,
                anchors=anchors,
                supersedes=supersedes,
                superseded_by=superseded_by,
                actor=actor,
                status=resolved_status,
                query=query,
                after=after,
                before=before,
                limit=max(1, min(limit, _MAX_LIMIT)),
                offset=max(0, offset),
            )
    except ops.JournalUnavailableError as exc:
        return _error(str(exc), guidance=exc.guidance, journal_available=False)
    except ops.DecisionOpsError as exc:
        # The base, not each subclass: a failure this layer models is one the
        # caller is meant to read, and a subclass added later should reach them
        # as guidance rather than as an isError the shield had to catch.
        return _error(str(exc), guidance=exc.guidance)
    except LookupError as exc:
        return _error(str(exc))

    payload["action"] = verb
    if verb == "list" and limit > _MAX_LIMIT:
        # Said rather than applied quietly. A caller that asked for 500 and got
        # 200 back has no way to tell a clamp from the end of the data, and
        # ``total`` alone does not say which.
        payload["limit_note"] = (
            f"Requested limit={limit} was clamped to {_MAX_LIMIT}. "
            f"Page with offset to read the remaining {max(0, payload.get('total', 0) - _MAX_LIMIT)}."
        )
    attach_ignored_arguments(payload, ignored)
    payload["_meta"] = _build_meta(
        timing_ms=(time.perf_counter() - started) * 1000,
        repository=repository,
    )
    return payload


async def _dispatch(
    verb: str,
    *,
    session: Any,
    repository_id: str,
    ctx: Any,
    decision_id: str | None,
    title: str | None,
    decision: str | None,
    why: str | None,
    anchors: list[str] | None,
    supersedes: str | None,
    superseded_by: str | None,
    actor: str | None,
    status: str | None,
    query: str | None,
    after: datetime | None,
    before: datetime | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Run one verb. Raises ``DecisionOpsError``; the caller shapes the failure."""
    root = ctx.path
    store = getattr(ctx, "decision_store", None)

    if verb == "list":
        rows, total = await ops.list_decisions(
            session,
            repository_id,
            repo_root=root,
            status=status,
            query=query,
            recorded_after=after,
            recorded_before=before,
            limit=limit,
            offset=offset,
            vector_store=store,
        )
        location = ops.journal_location()
        payload: dict[str, Any] = {
            "decisions": [ops.serialize(row, anchors=False) for row in rows],
            "total": total,
            "offset": offset,
            "returned": len(rows),
            "journal": location,
            "journal_available": location is not None,
        }
        if location is None:
            # Reads survive a disabled journal by serving the last projection,
            # and have to say which of the two "no rows" they are reporting.
            payload["note"] = (
                "The decision journal is not configured, so these rows are the last "
                "projected state and no new decision can be recorded."
            )
        return payload

    if verb == "get":
        if not decision_id:
            raise ops.DecisionOpsValidationError("decision_id is required to get a decision")
        record, chain = await ops.get_decision(
            session,
            repository_id,
            decision_id.strip(),
            repo_root=root,
            vector_store=store,
        )
        location = ops.journal_location()
        return {
            "decision": ops.serialize(record),
            # Oldest first, so the chain reads as the history it is. A record
            # with no supersession is a one-entry chain rather than an absent
            # field, so a caller need not special-case it.
            "chain": [ops.serialize(link, anchors=False) for link in chain],
            "journal": location,
            "journal_available": location is not None,
        }

    if verb == "record":
        record = await ops.record_decision(
            session,
            repository_id,
            repo_root=root,
            actor=actor or "",
            title=title or "",
            decision=decision or "",
            why=why or "",
            anchors=anchors or [],
            supersedes=(supersedes or "").strip() or None,
            vector_store=store,
        )
        return {
            "decision": ops.serialize(record),
            "recorded": True,
            "next": (
                "Landed as 'proposed'. A person confirms it — either "
                f"`manage_decision(action='confirm', decision_id='{record.id}')` when "
                "asked to, or by committing the journal diff."
            ),
        }

    if verb == "confirm":
        if not decision_id:
            raise ops.DecisionOpsValidationError("decision_id is required to confirm a decision")
        record = await ops.confirm_decision(
            session,
            repository_id,
            decision_id.strip(),
            repo_root=root,
            actor=actor or "",
            vector_store=store,
        )
        return {"decision": ops.serialize(record), "confirmed": True}

    if not decision_id:
        raise ops.DecisionOpsValidationError("decision_id is required to supersede a decision")
    retired, successor = await ops.supersede_decision(
        session,
        repository_id,
        decision_id.strip(),
        superseded_by=(superseded_by or "").strip(),
        repo_root=root,
        actor=actor or "",
        vector_store=store,
    )
    return {
        "decision": ops.serialize(retired),
        "successor": ops.serialize(successor),
        "superseded": True,
        # Said explicitly because "superseded" reads like a delete to an agent
        # that has only seen CRUD surfaces.
        "note": "The retired record stays in the journal and stays readable.",
    }


__all__ = ["manage_decision"]
