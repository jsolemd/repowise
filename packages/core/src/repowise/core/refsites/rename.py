"""Rename preview — every site a rename would touch, and nothing else.

This module performs no writes and has no apply path, by construction. It
answers one question: *if you renamed this symbol, what would have to change,
and how sure are we about each one?*

Two things make the answer trustworthy rather than merely confident:

**It reads the store, not the graph.** A graph edge knows that A calls B; it
does not know where, and it collapses repeated calls onto one row. A preview
built on edges would report files, not positions, and would undercount. Every
site below comes from a stored occurrence row — delete one row and exactly one
site disappears from the preview.

**It reports what it could not prove.** Sites that spell the name but bind to
nothing — an ambiguous global, an unresolved call, a name inside a string — are
included at their rubric floor and counted in the caveats. Dropping them would
make the preview look complete when it is only confident, which is the failure
mode that turns a rename into a silent breakage.
"""

from __future__ import annotations

from repowise.core.refsites.coverage import CoverageStatus
from repowise.core.refsites.records import (
    CoverageRecord,
    ReferenceSiteRecord,
    RenamePreview,
    RenameSite,
)
from repowise.core.refsites.store import SqlReferenceSiteStore
from repowise.core.refsites.taxonomy import (
    CONFIDENCE_CERTAIN,
    ReferenceKind,
)


def symbol_name(symbol_id: str) -> str:
    """Best-effort name from a symbol id of the form ``path::Owner::name``.

    Only a fallback: the definition site's recorded ``name`` is authoritative
    because it is the text actually in the file, discriminator suffixes and
    all.
    """
    tail = symbol_id.rsplit("::", 1)[-1]
    return tail.split("~", 1)[0]


def _to_site(record: ReferenceSiteRecord) -> RenameSite:
    return RenameSite(
        file_path=record.file_path,
        language=record.language,
        name=record.name,
        kind=record.kind,
        start_line=record.start_line,
        end_line=record.end_line,
        start_col=record.start_col,
        end_col=record.end_col,
        range_exact=record.range_exact,
        origin=record.origin,
        confidence=record.confidence,
        tier=record.tier,
        occurrence_index=record.occurrence_index,
        enclosing_symbol_id=record.enclosing_symbol_id,
        mechanically_safe=record.confidence >= CONFIDENCE_CERTAIN and record.range_exact,
    )


def _caveats(
    sites: tuple[RenameSite, ...],
    coverage: list[CoverageRecord],
    collisions: list[ReferenceSiteRecord],
    new_name: str | None,
) -> tuple[str, ...]:
    """Everything that could make this preview wrong, worst first."""
    out: list[str] = []

    blind = [
        record
        for record in coverage
        if record.status is CoverageStatus.NONE and record.files_seen > 0
    ]
    for record in blind:
        out.append(
            f"{record.files_seen} {record.language} file(s) were read but produce no reference "
            f"sites of any kind ({record.reason}). A reference in those files is invisible here."
        )

    for record in coverage:
        if record.status in (CoverageStatus.PARTIAL, CoverageStatus.UNGATED) and record.files_seen:
            out.append(
                f"{record.language} coverage is {record.status}: {record.reason}. "
                f"{record.files_seen} file(s) of it were read."
            )

    unproven = [site for site in sites if not site.mechanically_safe]
    if unproven:
        low = sum(1 for site in unproven if site.confidence < CONFIDENCE_CERTAIN)
        inexact = sum(1 for site in unproven if not site.range_exact)
        out.append(
            f"{len(unproven)} of {len(sites)} site(s) are not mechanically safe: "
            f"{low} below the {CONFIDENCE_CERTAIN:.2f} confidence bar, "
            f"{inexact} without a verified column range."
        )

    if not any(site.kind is ReferenceKind.DEFINITION for site in sites):
        out.append(
            "No definition site is recorded for this symbol. The parse layer deletes symbols "
            "declared inside anonymous callables, so the definition may exist in the source "
            "without existing in the index."
        )

    if new_name and collisions:
        files = sorted({record.file_path for record in collisions})
        out.append(
            f"'{new_name}' is already spelled at {len(collisions)} site(s) in "
            f"{len(files)} file(s); the rename may shadow or collide."
        )

    return tuple(out)


async def preview_rename(
    store: SqlReferenceSiteStore,
    repository_id: str,
    symbol_id: str,
    *,
    new_name: str | None = None,
    min_confidence: float = 0.0,
) -> RenamePreview:
    """Return every site a rename of *symbol_id* would touch.

    Sites bound to the symbol come first at their resolution confidence.
    Sites that merely spell the same name and bind to nothing follow at their
    rubric floor — they are leads for a human, not edits.
    """
    bound = await store.sites_for_target(repository_id, symbol_id, min_confidence=min_confidence)

    definition = next((record for record in bound if record.kind is ReferenceKind.DEFINITION), None)
    old_name = definition.name if definition is not None else symbol_name(symbol_id)

    unbound = await store.unbound_sites_named(
        repository_id, old_name, min_confidence=min_confidence
    )
    collisions = await store.sites_named(repository_id, new_name) if new_name else []

    records = sorted(
        (*bound, *unbound),
        key=lambda record: (
            -record.confidence,
            record.file_path,
            record.start_line,
            record.start_col,
            str(record.kind),
        ),
    )
    sites = tuple(_to_site(record) for record in records)
    coverage = await store.coverage(repository_id)

    return RenamePreview(
        symbol_id=symbol_id,
        old_name=old_name,
        new_name=new_name,
        sites=sites,
        coverage=tuple(coverage),
        caveats=_caveats(sites, coverage, collisions, new_name),
    )


__all__ = ["preview_rename", "symbol_name"]
