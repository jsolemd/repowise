"""Observable health of the source-search publication pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .fts import SourceFTSIndex
from .generation import GenerationRef
from .manifest import EmbedderIdentity, default_manifest_path, inspect_manifest
from .worktree import (
    UNCHECKED,
    WorkingTreeDivergence,
    divergence_from_candidates,
    refine_with_ingest_record,
    working_tree_candidates,
)

__all__ = [
    "CODE_COUNT_MISMATCH",
    "CODE_MISSING",
    "CODE_UNREADABLE",
    "COMPONENT_DENSE",
    "COMPONENT_LEXICAL",
    "COMPONENT_MANIFEST",
    "COMPONENT_PUBLICATION",
    "COMPONENT_QUEUE",
    "EVIDENCE_PRESERVING_CODES",
    "INTEGRITY_CODES",
    "INTEGRITY_COMPONENTS",
    "IntegrityFinding",
    "SourceIndexStatus",
    "inspect_source_index",
]

# Which store or stage the fault belongs to. Consumers switch on these, never
# on the prose in ``IntegrityFinding.detail``.
COMPONENT_MANIFEST = "manifest"
COMPONENT_QUEUE = "queue"
COMPONENT_LEXICAL = "lexical"
COMPONENT_DENSE = "dense"
COMPONENT_PUBLICATION = "publication"

# What kind of fault it is.
CODE_UNREADABLE = "unreadable"
CODE_MISSING = "missing"
CODE_COUNT_MISMATCH = "count_mismatch"

INTEGRITY_COMPONENTS = frozenset(
    {
        COMPONENT_MANIFEST,
        COMPONENT_QUEUE,
        COMPONENT_LEXICAL,
        COMPONENT_DENSE,
        COMPONENT_PUBLICATION,
    }
)
INTEGRITY_CODES = frozenset({CODE_UNREADABLE, CODE_MISSING, CODE_COUNT_MISMATCH})

#: Codes that leave enough evidence standing to judge the publication anyway:
#: a count mismatch is a *known* divergence, so an observer may report the
#: stronger claim "stale".  Every other code — including one a consumer has
#: never seen — destroyed the evidence, and the honest verdict is "unknown".
#: Consumers must treat this set as the allowlist, not its complement, so that
#: a code added here later is the only way a fault becomes judgeable.
EVIDENCE_PRESERVING_CODES = frozenset({CODE_COUNT_MISMATCH})


@dataclass(frozen=True, slots=True)
class IntegrityFinding:
    """One structured integrity fault, named by the code that produced it.

    ``component`` and ``code`` come from the constants above and are the only
    fields a consumer may classify on.  ``detail`` is free text for humans and
    may be reworded at any time; a consumer that substring-matches it silently
    changes verdict when someone edits a message string in this module.
    """

    component: str
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"component": self.component, "code": self.code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class SourceIndexStatus:
    state: str
    generation_id: str | None
    generation_sequence: int | None
    indexed_commit: str | None
    recipe_fingerprint: str | None
    pending_updates: int
    blocked_updates: int
    building_updates: int
    ready_updates: int
    manifest_state: str = "missing"
    manifest_error: str | None = None
    built_at: str | None = None
    published_at: str | None = None
    embedder: EmbedderIdentity | None = None
    parser_fingerprint: str | None = None
    symbol_chunks: int = 0
    file_window_chunks: int = 0
    files_covered: int = 0
    stale_files: dict[str, str] = field(default_factory=dict)
    #: Indexed paths the working tree has changed *since* the build, read live
    #: rather than from anything the build recorded. Distinct from
    #: :attr:`stale_files`, which is the build's own account of files it tried
    #: to chunk and could not.
    working_tree: WorkingTreeDivergence = UNCHECKED
    #: The active generation's own record of the uncommitted bytes it ingested,
    #: path -> content hash, straight off the manifest. Answers a different
    #: question from :attr:`working_tree`: not "what has moved since the build"
    #: but "which paths did this build read dirty at all". Every build writes
    #: it, which is what makes it the surface a consumer must ask — a full
    #: rebuild writes nothing to ``state.json``, so a consumer reading that
    #: file instead reports the emptiest answer on the dirtiest build.
    working_tree_ingest: dict[str, str] = field(default_factory=dict)
    expected_chunks: int = 0
    fts_chunks: int | None = None
    vector_chunks: int | None = None
    lance_table: str | None = None
    fts_path: str | None = None
    last_error: str | None = None
    integrity_findings: tuple[IntegrityFinding, ...] = ()

    @property
    def degraded(self) -> bool:
        return self.state in {"degraded", "inconsistent"}

    @property
    def integrity_errors(self) -> tuple[str, ...]:
        """Human-readable details of :attr:`integrity_findings`, in order.

        Kept for display surfaces (``repowise doctor``, ``/health``).  It is
        deliberately derived rather than stored so no caller can classify on
        these strings and drift away from the structured findings beside them.
        """

        return tuple(finding.detail for finding in self.integrity_findings)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["degraded"] = self.degraded
        result["integrity_errors"] = list(self.integrity_errors)
        result["working_tree"] = self.working_tree.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class _SourceUpdateSnapshot:
    active: Any | None
    counts: dict[str, int]
    outstanding_total: int
    last_error: str | None


async def _source_update_snapshot(
    repo: Path,
    db_url: str | None,
    *,
    active_sequence: int,
    active_generation_id: str | None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> _SourceUpdateSnapshot:
    """Own a short session; dispose only engines this inspection created."""
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        resolve_db_url,
    )

    engine = None
    try:
        if session_factory is None:
            engine = create_engine(db_url or resolve_db_url(repo))
            session_factory = create_session_factory(engine)
        async with session_factory() as session:
            return await _read_source_updates(session, repo, active_sequence, active_generation_id)
    finally:
        if engine is not None:
            await engine.dispose()


async def _read_source_updates(
    session: AsyncSession,
    repo: Path,
    active_sequence: int,
    active_generation_id: str | None,
) -> _SourceUpdateSnapshot:
    """Read this repository's publication and outstanding queue in one session."""
    from repowise.core.persistence.models import Repository, SourceIndexUpdate

    repositories = list((await session.execute(select(Repository))).scalars().all())
    repository_id = None
    for row in repositories:
        try:
            if row.local_path and Path(row.local_path).resolve() == repo:
                repository_id = row.id
                break
        except OSError:
            continue
    if repository_id is None and len(repositories) == 1:
        repository_id = repositories[0].id
    if repository_id is None:
        return _SourceUpdateSnapshot(None, {}, 0, None)

    active_candidate = None
    if active_sequence > 0:
        active_candidate = (
            (
                await session.execute(
                    select(SourceIndexUpdate).where(
                        SourceIndexUpdate.repository_id == repository_id,
                        SourceIndexUpdate.sequence == active_sequence,
                    )
                )
            )
            .scalars()
            .one_or_none()
        )
    outstanding_filter = (
        SourceIndexUpdate.repository_id == repository_id,
        SourceIndexUpdate.sequence > active_sequence,
    )
    count_rows = (
        await session.execute(
            select(SourceIndexUpdate.state, func.count())
            .where(*outstanding_filter)
            .group_by(SourceIndexUpdate.state)
        )
    ).all()
    counts = {str(state): int(count) for state, count in count_rows}
    last_error_value = None
    # The grouped read already proves an empty queue has no error.
    if counts:
        last_error_value = (
            await session.execute(
                select(SourceIndexUpdate.last_error)
                .where(
                    *outstanding_filter,
                    SourceIndexUpdate.last_error.is_not(None),
                    SourceIndexUpdate.last_error != "",
                )
                .order_by(SourceIndexUpdate.sequence.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    active = (
        active_candidate
        if active_candidate is not None
        and active_candidate.generation_id == active_generation_id
        and active_candidate.state in {"ready", "published"}
        else None
    )
    return _SourceUpdateSnapshot(
        active=active,
        counts=counts,
        outstanding_total=sum(counts.values()),
        last_error=str(last_error_value) if last_error_value else None,
    )


def _read_fts_facts(
    fts_path: Path,
    generation: GenerationRef,
    membership_probe: Sequence[str],
) -> tuple[int | None, set[str] | None, str | None]:
    """Row count and path membership from one open of the lexical store.

    Synchronous and blocking (a SQLite open plus two queries), so callers on an
    event loop hand it to a worker thread — and the connection is created and
    used inside that one thread, which is what ``sqlite3``'s default
    same-thread check requires. ``read_only=True`` is what keeps a status read
    from creating or migrating the store it is reporting on.

    *membership_probe* is the live working-tree change set; the second query
    answers which of those paths the generation actually serves. It is one open
    rather than two because the count and the membership are facts about the
    same generation and must not be read from different ones.
    """

    try:
        with SourceFTSIndex(fts_path, generation=generation, read_only=True) as fts:
            return fts.count(), fts.indexed_among(membership_probe), None
    except Exception as exc:
        return None, None, f"{exc}"


async def inspect_source_index(
    repo_path: Path | str,
    *,
    embedder: Any | None = None,
    db_url: str | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    verify_stores: bool = True,
    fts: SourceFTSIndex | None = None,
    working_tree_max_age: float = 0.0,
) -> SourceIndexStatus:
    """Inspect publication, queue, live working tree, and cross-store parity.

    *fts* lets a caller that already holds the active lexical store — the
    search path holds one open for every query — supply it instead of paying
    a second open. Passing it is also the only way to learn working-tree
    divergence with ``verify_stores=False``: which paths the generation serves
    is a question only that store can answer, and a caller that skipped it
    gets the unchecked verdict rather than a clean-looking empty one.

    *working_tree_max_age* is how stale a cached live-git read may be. Zero,
    the default, always re-reads: a caller asking this function what the index
    is worth should not be answered from a cache.

    *session_factory* reuses the host's database engine, when supplied. Each
    inspection still opens and closes its own session and reads the queue
    live; the engine's lifecycle remains with the host. Standalone callers
    resolve *db_url* and own a temporary engine as before.
    """

    repo = Path(repo_path).resolve()
    manifest_result = inspect_manifest(default_manifest_path(repo))
    manifest = manifest_result.manifest
    active_sequence = manifest.generation_sequence if manifest is not None else 0
    findings: list[IntegrityFinding] = []
    if manifest_result.state == "unreadable":
        findings.append(
            IntegrityFinding(
                COMPONENT_MANIFEST,
                CODE_UNREADABLE,
                f"source manifest unreadable: {manifest_result.error}",
            )
        )
    try:
        updates = await _source_update_snapshot(
            repo,
            db_url,
            active_sequence=active_sequence,
            active_generation_id=manifest.generation_id if manifest is not None else None,
            session_factory=session_factory,
        )
    except Exception as exc:
        updates = _SourceUpdateSnapshot(None, {}, 0, None)
        findings.append(
            IntegrityFinding(COMPONENT_QUEUE, CODE_UNREADABLE, f"source outbox unreadable: {exc}")
        )

    active_update = updates.active
    pending = updates.counts.get("pending", 0)
    blocked = updates.counts.get("blocked", 0)
    building = updates.counts.get("building", 0)
    ready = updates.counts.get("ready", 0)
    last_error = updates.last_error

    expected = manifest.symbol_chunks + manifest.file_window_chunks if manifest is not None else 0
    fts_count: int | None = None
    vector_count: int | None = None
    working_tree = UNCHECKED
    # The live half of freshness. Only meaningful once there is a published
    # generation to be behind: with no manifest the question is not "what has
    # changed since the build" but "was there a build".
    candidates: dict[str, str] = {}
    candidate_error: str | None = None
    if manifest is not None:
        candidates, candidate_error = await asyncio.to_thread(
            working_tree_candidates, repo, max_age=working_tree_max_age
        )
    # The ingest record widens the membership probe: a recorded path with no
    # live git divergence (a post-build revert) still needs its served-or-not
    # answer before the refinement below may flag it.
    ingest_record = dict(manifest.working_tree_ingest) if manifest is not None else {}
    probe_paths = sorted(set(candidates) | set(ingest_record))
    if manifest is not None and fts is not None:
        # The caller's store, so its connection stays on the caller's thread —
        # ``sqlite3`` refuses cross-thread use, and the query is bounded by the
        # change set rather than the corpus.
        try:
            probe: set[str] | None = fts.indexed_among(probe_paths)
        except Exception as exc:
            probe = None
            candidate_error = candidate_error or f"lexical_store_unreadable: {type(exc).__name__}"
        working_tree = divergence_from_candidates(candidates, probe, error=candidate_error)
        working_tree = refine_with_ingest_record(working_tree, ingest_record, probe, repo)
    if manifest is not None and verify_stores:
        generation = GenerationRef(manifest.generation_id, manifest.generation_sequence)
        fts_path = repo / manifest.fts_path
        if not fts_path.is_file():
            findings.append(
                IntegrityFinding(
                    COMPONENT_LEXICAL, CODE_MISSING, f"FTS store missing: {manifest.fts_path}"
                )
            )
        else:
            fts_count, indexed_probe, fts_error = await asyncio.to_thread(
                _read_fts_facts, fts_path, generation, probe_paths
            )
            if fts is None:
                working_tree = divergence_from_candidates(
                    candidates,
                    indexed_probe,
                    error=candidate_error,
                )
                working_tree = refine_with_ingest_record(
                    working_tree, ingest_record, indexed_probe, repo
                )
            if fts_error is not None:
                findings.append(
                    IntegrityFinding(
                        COMPONENT_LEXICAL, CODE_UNREADABLE, f"FTS store unreadable: {fts_error}"
                    )
                )
            elif fts_count != expected:
                findings.append(
                    IntegrityFinding(
                        COMPONENT_LEXICAL,
                        CODE_COUNT_MISMATCH,
                        f"FTS count mismatch: expected {expected}, found {fts_count}",
                    )
                )

        if embedder is not None:
            lance_path = repo / ".repowise" / "lancedb"
            if not lance_path.is_dir():
                findings.append(
                    IntegrityFinding(
                        COMPONENT_DENSE, CODE_MISSING, "Lance store missing: .repowise/lancedb"
                    )
                )
            else:
                try:
                    from .vector_store import SourceChunkVectorStore

                    store = SourceChunkVectorStore(
                        str(lance_path),
                        embedder=embedder,
                        table_name=manifest.lance_table,
                        generation=generation,
                    )
                    try:
                        vector_count = await store.count()
                    finally:
                        await store.close()
                    if vector_count != expected:
                        findings.append(
                            IntegrityFinding(
                                COMPONENT_DENSE,
                                CODE_COUNT_MISMATCH,
                                f"Lance count mismatch: expected {expected}, found {vector_count}",
                            )
                        )
                except Exception as exc:
                    findings.append(
                        IntegrityFinding(
                            COMPONENT_DENSE, CODE_UNREADABLE, f"Lance store unreadable: {exc}"
                        )
                    )

    # ``state`` stays the *publication pipeline's* verdict and deliberately
    # does not move for working-tree divergence. A checkout with unsaved edits
    # is the normal condition of a repository someone is working in; folding it
    # in here would make ``repowise doctor`` (which reads ``state == "current"``
    # as health) permanently red while saying nothing about the pipeline. The
    # divergence rides on its own field instead, where every consumer that
    # speaks staleness reads it.
    if findings:
        state = "inconsistent"
    elif manifest is None:
        state = "degraded" if updates.outstanding_total else "missing"
    elif blocked or last_error or manifest.stale_files:
        state = "degraded"
    elif pending or building or ready:
        state = "pending"
    else:
        state = "current"

    return SourceIndexStatus(
        state=state,
        generation_id=manifest.generation_id if manifest else None,
        generation_sequence=manifest.generation_sequence if manifest else None,
        indexed_commit=manifest.indexed_commit if manifest else None,
        recipe_fingerprint=manifest.recipe_fingerprint if manifest else None,
        pending_updates=pending,
        blocked_updates=blocked,
        building_updates=building,
        ready_updates=ready,
        manifest_state=manifest_result.state,
        manifest_error=manifest_result.error,
        built_at=manifest.built_at if manifest else None,
        published_at=(
            active_update.published_at.isoformat()
            if active_update is not None and active_update.published_at is not None
            else None
        ),
        embedder=manifest.embedder if manifest else None,
        parser_fingerprint=(
            str(active_update.parser_fingerprint) if active_update is not None else None
        ),
        symbol_chunks=manifest.symbol_chunks if manifest else 0,
        file_window_chunks=manifest.file_window_chunks if manifest else 0,
        files_covered=manifest.files_covered if manifest else 0,
        stale_files=dict(manifest.stale_files) if manifest else {},
        working_tree=working_tree,
        working_tree_ingest=ingest_record,
        expected_chunks=expected,
        fts_chunks=fts_count,
        vector_chunks=vector_count,
        lance_table=manifest.lance_table if manifest else None,
        fts_path=manifest.fts_path if manifest else None,
        last_error=last_error,
        integrity_findings=tuple(findings),
    )
