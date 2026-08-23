"""Persistence for reference sites.

Follows the house contract for stores: the store binds one ``AsyncSession``,
flushes, and never commits — the caller owns the transaction.

Writes are replace-then-insert rather than upsert. A site's identity is its
position, and an edit moves positions, so "the sites for this file are now
exactly these" is the only statement that stays true after a line is added
near the top of a file. Upserting by position would leave every shifted site
behind as a ghost.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from sqlalchemy import delete, func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.persistence.models import _new_uuid, _now_utc
from repowise.core.refsites.coverage import CoverageStatus
from repowise.core.refsites.records import (
    CoverageRecord,
    ExtractionResult,
    ReferenceSiteRecord,
)
from repowise.core.refsites.schema import ReferenceSite, ReferenceSiteCoverage
from repowise.core.refsites.taxonomy import (
    EXTRACTOR_VERSION,
    ReferenceKind,
    ReferenceOrigin,
)

#: Rows per INSERT. Mirrors the persistence layer's batch size, which exists to
#: stay under SQLite's bound-parameter ceiling.
_BATCH_SIZE = 500

#: Kinds whose ``name`` is a module path rather than a symbol name. They share
#: a column with symbol names but not a namespace, so a by-name lookup for a
#: symbol must not return them: a function called ``helpers`` would otherwise
#: pull in every ``import helpers`` statement at import-binding confidence,
#: which is a high-confidence false positive — the worst kind for a rename.
_MODULE_NAMESPACE_KINDS = frozenset({str(ReferenceKind.IMPORT)})


def _to_record(row: ReferenceSite) -> ReferenceSiteRecord:
    """Project an ORM row back to the immutable record."""
    return ReferenceSiteRecord(
        file_path=row.file_path,
        language=row.language,
        name=row.name,
        kind=ReferenceKind(row.kind),
        start_line=row.start_line,
        end_line=row.end_line,
        start_col=row.start_col,
        end_col=row.end_col,
        range_exact=row.range_exact,
        origin=ReferenceOrigin(row.resolution_origin),
        tier=row.tier,
        target_symbol_id=row.target_symbol_id,
        enclosing_symbol_id=row.enclosing_symbol_id,
        occurrence_index=row.occurrence_index,
        extractor_version=row.extractor_version,
    )


def _to_row(repository_id: str, site: ReferenceSiteRecord) -> dict:
    return {
        "id": _new_uuid(),
        "repository_id": repository_id,
        "file_path": site.file_path,
        "language": site.language,
        "name": site.name,
        "kind": str(site.kind),
        "start_line": site.start_line,
        "end_line": site.end_line,
        "start_col": site.start_col,
        "end_col": site.end_col,
        "range_exact": site.range_exact,
        "target_symbol_id": site.target_symbol_id,
        "enclosing_symbol_id": site.enclosing_symbol_id,
        "resolution_origin": str(site.origin),
        "confidence": site.confidence,
        "tier": site.tier,
        "occurrence_index": site.occurrence_index,
        "extractor_version": site.extractor_version,
        "created_at": _now_utc(),
    }


def _to_coverage_record(row: ReferenceSiteCoverage) -> CoverageRecord:
    return CoverageRecord(
        language=row.language,
        status=CoverageStatus(row.status),
        files_seen=row.files_seen,
        files_extracted=row.files_extracted,
        sites=row.sites,
        reason=row.reason,
        limitations=tuple(json.loads(row.limitations_json)),
    )


class SqlReferenceSiteStore:
    """The reference-site store, backed by ``reference_sites``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @property
    def session(self) -> AsyncSession:
        return self._session

    # -- writes ------------------------------------------------------------

    async def _insert(self, repository_id: str, sites: Sequence[ReferenceSiteRecord]) -> None:
        rows = [_to_row(repository_id, site) for site in sites]
        for start in range(0, len(rows), _BATCH_SIZE):
            await self._session.execute(insert(ReferenceSite), rows[start : start + _BATCH_SIZE])
        await self._session.flush()

    async def replace_repository(self, repository_id: str, result: ExtractionResult) -> None:
        """Make the stored sites for *repository_id* exactly ``result``."""
        await self._session.execute(
            delete(ReferenceSite).where(ReferenceSite.repository_id == repository_id)
        )
        await self._insert(repository_id, result.sites)
        await self.replace_coverage(repository_id, result.coverage)

    async def replace_files(
        self,
        repository_id: str,
        file_paths: Iterable[str],
        sites: Sequence[ReferenceSiteRecord],
    ) -> None:
        """Replace the sites of the named files only, for incremental re-extraction."""
        paths = list(file_paths)
        for start in range(0, len(paths), _BATCH_SIZE):
            await self._session.execute(
                delete(ReferenceSite).where(
                    ReferenceSite.repository_id == repository_id,
                    ReferenceSite.file_path.in_(paths[start : start + _BATCH_SIZE]),
                )
            )
        await self._insert(repository_id, sites)

    async def replace_coverage(
        self, repository_id: str, coverage: Sequence[CoverageRecord]
    ) -> None:
        await self._session.execute(
            delete(ReferenceSiteCoverage).where(
                ReferenceSiteCoverage.repository_id == repository_id
            )
        )
        if coverage:
            await self._session.execute(
                insert(ReferenceSiteCoverage),
                [
                    {
                        "id": _new_uuid(),
                        "repository_id": repository_id,
                        "language": record.language,
                        "status": str(record.status),
                        "files_seen": record.files_seen,
                        "files_extracted": record.files_extracted,
                        "sites": record.sites,
                        "reason": record.reason,
                        "limitations_json": json.dumps(list(record.limitations)),
                        "extractor_version": EXTRACTOR_VERSION,
                        "updated_at": _now_utc(),
                    }
                    for record in coverage
                ],
            )
        await self._session.flush()

    # -- reads -------------------------------------------------------------

    def _base(self, repository_id: str):
        return select(ReferenceSite).where(ReferenceSite.repository_id == repository_id)

    @staticmethod
    def _ordered(query):
        # LIMIT without ORDER BY is nondeterministic; every read here is
        # ordered by the confidence it reports, then by position so a caller
        # can diff two runs.
        return query.order_by(
            ReferenceSite.confidence.desc(),
            ReferenceSite.file_path,
            ReferenceSite.start_line,
            ReferenceSite.start_col,
            ReferenceSite.kind,
        )

    async def sites_for_target(
        self,
        repository_id: str,
        target_symbol_id: str,
        *,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> list[ReferenceSiteRecord]:
        """Every site bound to *target_symbol_id*."""
        query = self._base(repository_id).where(
            ReferenceSite.target_symbol_id == target_symbol_id,
            ReferenceSite.confidence >= min_confidence,
        )
        query = self._ordered(query)
        if limit is not None:
            query = query.limit(limit)
        result = await self._session.execute(query)
        return [_to_record(row) for row in result.scalars().all()]

    async def unbound_sites_named(
        self,
        repository_id: str,
        name: str,
        *,
        min_confidence: float = 0.0,
        limit: int | None = None,
    ) -> list[ReferenceSiteRecord]:
        """Sites that spell *name* but bind to nothing.

        These are the candidates a rename must show and must not apply: an
        ambiguous global, an unresolved call, a string literal. Excluding them
        would make the preview look complete when it is merely confident.
        """
        query = self._base(repository_id).where(
            ReferenceSite.name == name,
            ReferenceSite.target_symbol_id.is_(None),
            ReferenceSite.confidence >= min_confidence,
            ReferenceSite.kind.not_in(_MODULE_NAMESPACE_KINDS),
        )
        query = self._ordered(query)
        if limit is not None:
            query = query.limit(limit)
        result = await self._session.execute(query)
        return [_to_record(row) for row in result.scalars().all()]

    async def sites_named(
        self,
        repository_id: str,
        name: str,
        *,
        include_module_paths: bool = False,
        limit: int | None = None,
    ) -> list[ReferenceSiteRecord]:
        """Every site spelling *name* in the symbol namespace, bound or not.

        Set ``include_module_paths`` to also match import-statement sites,
        whose ``name`` is a module path — that is the entry point a *module*
        rename would use, and it is off by default so a symbol rename cannot
        pick them up by accident.
        """
        query = self._base(repository_id).where(ReferenceSite.name == name)
        if not include_module_paths:
            query = query.where(ReferenceSite.kind.not_in(_MODULE_NAMESPACE_KINDS))
        query = self._ordered(query)
        if limit is not None:
            query = query.limit(limit)
        result = await self._session.execute(query)
        return [_to_record(row) for row in result.scalars().all()]

    async def sites_in_file(self, repository_id: str, file_path: str) -> list[ReferenceSiteRecord]:
        query = self._ordered(self._base(repository_id).where(ReferenceSite.file_path == file_path))
        result = await self._session.execute(query)
        return [_to_record(row) for row in result.scalars().all()]

    async def definitions_in_range(
        self,
        repository_id: str,
        file_path: str,
        start_line: int,
        end_line: int,
        *,
        limit: int | None = None,
    ) -> list[ReferenceSiteRecord]:
        """Definition sites whose declaration falls inside *start_line*..*end_line*.

        Answers "what is defined in this span" for a caller holding a region of
        a file and no parse of it — a search result naming its lines, say. Runs
        off ``ix_reference_sites_repo_file``, which is why the file is filtered
        in SQL and the range is a narrowing on top of it rather than a scan.

        The bound is on where the definition *starts*, not on the whole body
        being enclosed. A span that opens inside the range and runs past its
        end is still declared here, and it is the declaration this answers
        about; requiring containment of the body would drop exactly the symbol
        a window happens to cut through, which is the common case rather than
        an edge one.

        Both ends are inclusive, matching how every span in this codebase is
        written (1-indexed, both ends in).
        """
        query = self._base(repository_id).where(
            ReferenceSite.file_path == file_path,
            ReferenceSite.kind == str(ReferenceKind.DEFINITION),
            ReferenceSite.start_line >= start_line,
            ReferenceSite.start_line <= end_line,
        )
        query = self._ordered(query)
        if limit is not None:
            query = query.limit(limit)
        result = await self._session.execute(query)
        return [_to_record(row) for row in result.scalars().all()]

    async def coverage(self, repository_id: str) -> list[CoverageRecord]:
        query = (
            select(ReferenceSiteCoverage)
            .where(ReferenceSiteCoverage.repository_id == repository_id)
            .order_by(ReferenceSiteCoverage.language)
        )
        result = await self._session.execute(query)
        return [_to_coverage_record(row) for row in result.scalars().all()]

    async def count(self, repository_id: str) -> int:
        """Row count for the repository.

        Counted in SQL rather than by loading ids: both MCP tools call this on
        every request to tell "no extraction has run" apart from "no sites
        match", and pulling every row id across the wire to take its length
        would make that cheap check the most expensive thing they do.
        """
        result = await self._session.execute(
            select(func.count())
            .select_from(ReferenceSite)
            .where(ReferenceSite.repository_id == repository_id)
        )
        return int(result.scalar_one())


__all__ = ["SqlReferenceSiteStore"]
