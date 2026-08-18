"""/api/repos/{repo_id}/decisions — Architectural decision record endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.analysis.decisions.journal import (
    DecisionJournalError,
    decisions_journal_path,
)
from repowise.core.analysis.decisions.journal_projection import (
    DecisionJournalHealth,
    confirm_journal_decision,
    record_journal_decision,
    refresh_decision_journal,
    replace_journal_anchor_files,
    supersede_journal_decision,
)
from repowise.core.persistence import crud, decision_graph
from repowise.core.persistence.models import DecisionEvidence
from repowise.server.deps import get_db_session, verify_api_key
from repowise.server.schemas import (
    DecisionCodeEdge,
    DecisionCountsResponse,
    DecisionCreate,
    DecisionEvidenceResponse,
    DecisionGraphEdge,
    DecisionGraphNode,
    DecisionGraphResponse,
    DecisionLineageEntry,
    DecisionRecordResponse,
    DecisionStatusUpdate,
)
from repowise.server.schemas.decisions import EvidencePreview
from repowise.server.search_helpers import resolve_repo_vector_store

router = APIRouter(
    tags=["decisions"],
    dependencies=[Depends(verify_api_key)],
)


def _journal_enabled() -> bool:
    try:
        return decisions_journal_path() is not None
    except DecisionJournalError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


async def _journal_vector_store(request: Request, repo_id: str, *, create: bool) -> object | None:
    """Resolve the repo-specific vector store without misrouting workspace repos."""

    primary_id = getattr(request.app.state, "primary_vector_repo_id", None)
    workspace_sessions = getattr(request.app.state, "workspace_sessions", None)
    if primary_id == repo_id or (primary_id is None and not workspace_sessions):
        return getattr(request.app.state, "vector_store", None)
    return await resolve_repo_vector_store(request.app.state, repo_id, create=create)


async def _refresh_journal(
    request: Request,
    session: AsyncSession,
    repo_id: str,
    *,
    create_vector_store: bool = False,
) -> DecisionJournalHealth | None:
    if not _journal_enabled():
        return None
    try:
        vector_store = await _journal_vector_store(request, repo_id, create=create_vector_store)
        return await refresh_decision_journal(session, repo_id, vector_store=vector_store)
    except DecisionJournalError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/api/repos/{repo_id}/decisions",
    response_model=list[DecisionRecordResponse],
)
async def list_decisions(
    repo_id: str,
    request: Request,
    status: str | None = Query(None, description="Filter by status"),
    source: str | None = Query(None, description="Filter by source"),
    tag: str | None = Query(None, description="Filter by tag"),
    module: str | None = Query(None, description="Filter by module path"),
    include_proposed: bool = Query(True),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort: str = Query(
        "priority",
        pattern="^(priority|recent)$",
        description="priority: confirmed rules first, then likeliest proposals. recent: newest first.",
    ),
    session: AsyncSession = Depends(get_db_session),
) -> list[DecisionRecordResponse]:
    """List architectural decision records for a repository.

    Each row carries an ``evidence_preview`` (the top-ranked evidence row's
    verbatim quote) plus the total ``evidence_count``, so the table can show
    provenance without N+1 calls to the per-decision /evidence endpoint.

    Defaults to ``sort=priority``. Newest-first buried every confirmed
    decision under the unreviewed proposals the indexer had just mined, so
    page one was entirely machine guesses.
    """
    await _refresh_journal(request, session, repo_id)
    decisions = await crud.list_decisions(
        session,
        repo_id,
        status=status,
        source=source,
        tag=tag,
        module=module,
        include_proposed=include_proposed,
        limit=limit,
        offset=offset,
        sort=sort,
    )
    items = [DecisionRecordResponse.from_orm(d) for d in decisions]

    ids = [d.id for d in decisions]
    if ids:
        rows = (
            await session.execute(
                select(DecisionEvidence)
                .where(DecisionEvidence.decision_id.in_(ids))
                .order_by(
                    DecisionEvidence.source_rank.desc(),
                    DecisionEvidence.confidence.desc(),
                )
            )
        ).scalars()
        counts: dict[str, int] = {}
        best: dict[str, DecisionEvidence] = {}
        for ev in rows:
            counts[ev.decision_id] = counts.get(ev.decision_id, 0) + 1
            # Rows arrive best-first, so the first row per decision wins.
            best.setdefault(ev.decision_id, ev)
        for item in items:
            item.evidence_count = counts.get(item.id, 0)
            top = best.get(item.id)
            if top is not None and top.source_quote:
                item.evidence_preview = EvidencePreview(
                    source=top.source,
                    source_quote=top.source_quote,
                    verification=top.verification,
                    evidence_file=top.evidence_file,
                    evidence_line=top.evidence_line,
                )
    return items


@router.get(
    "/api/repos/{repo_id}/decisions/health",
)
async def decision_health(
    repo_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Get decision health summary: stale, proposed, ungoverned hotspots."""
    journal_health = await _refresh_journal(request, session, repo_id)
    summary = await crud.get_decision_health_summary(session, repo_id)
    return {
        "summary": summary["summary"],
        "stale_decisions": [DecisionRecordResponse.from_orm(d) for d in summary["stale_decisions"]],
        "proposed_awaiting_review": [
            DecisionRecordResponse.from_orm(d) for d in summary["proposed_awaiting_review"]
        ],
        "ungoverned_hotspots": summary["ungoverned_hotspots"],
        "journal": journal_health.to_dict() if journal_health is not None else None,
    }


@router.get(
    "/api/repos/{repo_id}/decisions/counts",
    response_model=DecisionCountsResponse,
)
async def decision_counts(
    repo_id: str,
    request: Request,
    source: str | None = Query(None, description="Filter by source"),
    tag: str | None = Query(None, description="Filter by tag"),
    module: str | None = Query(None, description="Filter by module path"),
    include_proposed: bool = Query(True),
    session: AsyncSession = Depends(get_db_session),
) -> DecisionCountsResponse:
    """Counts by status, as a grouped COUNT rather than a page of rows.

    Declared above ``/{decision_id}`` on purpose: FastAPI matches in
    declaration order, so a literal sub-path below it would be swallowed by
    the parameterised route and "counts" would be looked up as a decision id.
    """
    await _refresh_journal(request, session, repo_id)
    counts = await crud.count_decisions_by_status(
        session,
        repo_id,
        source=source,
        tag=tag,
        module=module,
        include_proposed=include_proposed,
    )
    return DecisionCountsResponse(
        total=counts["total"],
        active=counts["active"],
        proposed=counts["proposed"],
        superseded=counts["superseded"],
        deprecated=counts["deprecated"],
    )


@router.get(
    "/api/repos/{repo_id}/decisions/graph",
    response_model=DecisionGraphResponse,
)
async def get_decision_graph(
    repo_id: str,
    request: Request,
    limit: int = Query(200, ge=1, le=500),
    session: AsyncSession = Depends(get_db_session),
) -> DecisionGraphResponse:
    """Return the full decision graph for a repository.

    Nodes are capped at *limit* (default 200), preferring active + superseded +
    proposed statuses. Decision→decision typed edges and decision→code links are
    returned without an additional cap (they scale with the node set).
    """
    await _refresh_journal(request, session, repo_id)
    # Fetch decisions ordered by staleness (most relevant first): active, then
    # superseded/proposed, then deprecated. Use list_decisions without status
    # filter so we get all statuses, capped.
    all_decisions = await crud.list_decisions(
        session,
        repo_id,
        include_proposed=True,
        limit=limit,
        offset=0,
    )

    nodes = [DecisionGraphNode.from_orm(d) for d in all_decisions]

    raw_edges = await decision_graph.list_all_decision_edges(session, repo_id)
    decision_edges = [
        DecisionGraphEdge(
            src=e.src_decision_id,
            dst=e.dst_decision_id,
            kind=e.kind,
            confidence=e.confidence,
            evidence=e.evidence,
        )
        for e in raw_edges
    ]

    raw_links = await decision_graph.list_decision_node_links(session, repo_id)
    code_edges = [
        DecisionCodeEdge(
            decision_id=lnk.decision_id,
            node_id=lnk.node_id,
            link_type=lnk.link_type,
        )
        for lnk in raw_links
    ]

    return DecisionGraphResponse(nodes=nodes, decision_edges=decision_edges, code_edges=code_edges)


@router.get(
    "/api/repos/{repo_id}/decisions/{decision_id}",
    response_model=DecisionRecordResponse,
)
async def get_decision(
    repo_id: str,
    decision_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> DecisionRecordResponse:
    """Get a single decision record by ID."""
    await _refresh_journal(request, session, repo_id)
    rec = await crud.get_decision(session, decision_id)
    if rec is None or rec.repository_id != repo_id:
        raise HTTPException(status_code=404, detail="Decision not found")
    return DecisionRecordResponse.from_orm(rec)


@router.get(
    "/api/repos/{repo_id}/decisions/{decision_id}/evidence",
)
async def list_decision_evidence(
    repo_id: str,
    decision_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return provenance evidence rows for a single decision record.

    Returns ``{"evidence": [...]}`` where each item carries the verbatim source
    quote, evidence file/line/commit, per-source confidence, and verification
    badge (``exact`` | ``fuzzy`` | ``unverified``). 404 if the decision does not
    exist or belongs to a different repository.
    """
    await _refresh_journal(request, session, repo_id)
    rec = await crud.get_decision(session, decision_id)
    if rec is None or rec.repository_id != repo_id:
        raise HTTPException(status_code=404, detail="Decision not found")
    rows = await crud.list_decision_evidence(session, decision_id)
    return {"evidence": [DecisionEvidenceResponse.from_orm(r) for r in rows]}


@router.get(
    "/api/repos/{repo_id}/decisions/{decision_id}/lineage",
)
async def get_decision_lineage(
    repo_id: str,
    decision_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> dict:
    """Return the lineage chain for a decision (root → … → current).

    Walks ``supersedes``/``refines`` edges back to the earliest ancestor so the
    UI can render a timeline. An isolated decision returns a single-entry chain.
    404 if the decision does not exist or belongs to a different repository.
    """
    await _refresh_journal(request, session, repo_id)
    rec = await crud.get_decision(session, decision_id)
    if rec is None or rec.repository_id != repo_id:
        raise HTTPException(status_code=404, detail="Decision not found")
    chain = await decision_graph.build_lineage_chain(session, decision_id)
    return {"lineage": [DecisionLineageEntry(**entry) for entry in chain]}


@router.post(
    "/api/repos/{repo_id}/decisions",
    response_model=DecisionRecordResponse,
    status_code=201,
)
async def create_decision(
    repo_id: str,
    body: DecisionCreate,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> DecisionRecordResponse:
    """Create a new decision record (e.g. from CLI capture via API)."""
    if _journal_enabled():
        await _refresh_journal(request, session, repo_id, create_vector_store=True)
        unsupported = []
        if body.context:
            unsupported.append("context")
        if body.alternatives:
            unsupported.append("alternatives")
        if body.consequences:
            unsupported.append("consequences")
        if body.affected_modules:
            unsupported.append("affected_modules")
        if body.tags:
            unsupported.append("tags")
        if unsupported:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Decision journal mode cannot losslessly store fields: "
                    + ", ".join(unsupported)
                ),
            )
        if not body.rationale.strip() or not body.affected_files:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Decision journal mode requires rationale (the why) and at least "
                    "one affected file anchor"
                ),
            )
        try:
            vector_store = await _journal_vector_store(request, repo_id, create=True)
            rec = await record_journal_decision(
                session,
                repo_id,
                title=body.title,
                decision=body.decision,
                why=body.rationale,
                anchors=[{"file": file, "symbol": None} for file in body.affected_files],
                vector_store=vector_store,
            )
        except DecisionJournalError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return DecisionRecordResponse.from_orm(rec)

    rec = await crud.upsert_decision(
        session,
        repository_id=repo_id,
        title=body.title,
        status="active",
        context=body.context,
        decision=body.decision,
        rationale=body.rationale,
        alternatives=body.alternatives,
        consequences=body.consequences,
        affected_files=body.affected_files,
        affected_modules=body.affected_modules,
        tags=body.tags,
        source="cli",
        confidence=1.0,
    )
    return DecisionRecordResponse.from_orm(rec)


@router.patch(
    "/api/repos/{repo_id}/decisions/{decision_id}",
    response_model=DecisionRecordResponse,
)
async def patch_decision(
    repo_id: str,
    decision_id: str,
    body: DecisionStatusUpdate,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> DecisionRecordResponse:
    """Update a decision record.

    Accepts status transitions (confirm / deprecate / supersede) and / or
    governance edits (``affected_modules``, ``affected_files``). Any field
    left as ``None`` in the body is preserved.
    """
    await _refresh_journal(request, session, repo_id, create_vector_store=_journal_enabled())
    rec = await crud.get_decision(session, decision_id)
    if rec is None or rec.repository_id != repo_id:
        raise HTTPException(status_code=404, detail="Decision not found")

    if _journal_enabled():
        if rec.source != "journal":
            raise HTTPException(
                status_code=409,
                detail="Only journal-projected decisions are mutable in journal mode",
            )
        if body.affected_modules:
            raise HTTPException(
                status_code=409,
                detail="Decision journal mode does not support affected_modules",
            )
        if body.status is not None and body.affected_files is not None:
            raise HTTPException(
                status_code=409,
                detail="Change status and affected_files in separate journal mutations",
            )
        try:
            vector_store = await _journal_vector_store(request, repo_id, create=True)
            if body.status == "active":
                if body.superseded_by is not None:
                    raise HTTPException(
                        status_code=400,
                        detail="superseded_by requires status='superseded'",
                    )
                rec = await confirm_journal_decision(
                    session,
                    repo_id,
                    decision_id,
                    vector_store=vector_store,
                )
            elif body.status == "superseded":
                if body.superseded_by is None:
                    raise HTTPException(
                        status_code=409,
                        detail="Journal supersession requires superseded_by",
                    )
                rec = await supersede_journal_decision(
                    session,
                    repo_id,
                    decision_id,
                    superseded_by=body.superseded_by,
                    vector_store=vector_store,
                )
            elif body.status is not None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Status {body.status!r} has no canonical journal representation; "
                        "only confirm (active) and supersede are enabled"
                    ),
                )
            elif body.superseded_by is not None:
                raise HTTPException(
                    status_code=400,
                    detail="superseded_by requires status='superseded'",
                )
            if body.affected_files is not None:
                rec = await replace_journal_anchor_files(
                    session,
                    repo_id,
                    decision_id,
                    body.affected_files,
                    vector_store=vector_store,
                )
        except DecisionJournalError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return DecisionRecordResponse.from_orm(rec)

    if body.status is not None:
        try:
            rec = await crud.update_decision_status(
                session,
                decision_id,
                body.status,
                superseded_by=body.superseded_by,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if rec is None:
            raise HTTPException(status_code=404, detail="Decision not found")
    elif body.superseded_by is not None:
        raise HTTPException(
            status_code=400,
            detail="superseded_by requires status='superseded'",
        )

    if body.affected_modules is not None or body.affected_files is not None:
        rec = await crud.update_decision_metadata(
            session,
            decision_id,
            affected_modules=body.affected_modules,
            affected_files=body.affected_files,
        )
        if rec is None:
            raise HTTPException(status_code=404, detail="Decision not found")

    assert rec is not None
    return DecisionRecordResponse.from_orm(rec)
