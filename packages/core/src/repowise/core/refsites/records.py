"""Immutable records exchanged between extraction, the store, and rename preview.

These are plain frozen dataclasses, matching ``core.ingestion.models``: the
extractor mints one per occurrence across every file in a repository, so
per-object validation overhead is paid tens of thousands of times.

``ReferenceSiteRecord`` is the store's unit. It is deliberately *not* shaped
like a graph edge: it has a position, it has a provenance, and two records may
share both endpoints and still be distinct rows. That is the whole point — the
graph collapses N call sites onto one edge, and this store is what remembers
the N.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from repowise.core.refsites.coverage import CoverageStatus
from repowise.core.refsites.taxonomy import (
    EXTRACTOR_VERSION,
    ReferenceKind,
    ReferenceOrigin,
    confidence_for,
)


@dataclass(frozen=True, slots=True)
class ReferenceSiteRecord:
    """One occurrence of one name at one position.

    ``target_symbol_id`` is ``None`` when the reference could not be bound to
    a definition. That is a first-class state, not a failure: an unbound site
    still tells a rename that something in this file spells this name.
    """

    file_path: str
    language: str
    name: str
    kind: ReferenceKind
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    #: True when ``start_col``/``end_col`` were located in the source text.
    #: False means the columns are a zero fallback and only the line is trusted.
    range_exact: bool
    origin: ReferenceOrigin
    tier: str
    target_symbol_id: str | None = None
    enclosing_symbol_id: str | None = None
    #: 0-based index of this occurrence among same-named occurrences on the
    #: same line. Non-zero is the fingerprint of a site the parse layer's
    #: same-line dedup would have dropped.
    occurrence_index: int = 0
    extractor_version: str = EXTRACTOR_VERSION

    @property
    def confidence(self) -> float:
        """Rubric confidence, derived so it can never drift from ``origin``."""
        return confidence_for(self.origin)

    @property
    def position_key(self) -> tuple[str, int, int, str, str]:
        """The identity a store row is unique on."""
        return (self.file_path, self.start_line, self.start_col, self.name, str(self.kind))


@dataclass(frozen=True, slots=True)
class CoverageRecord:
    """What extraction actually saw for one language in one repository.

    Measured, not declared: ``files_seen`` counts files of this language the
    traverser handed us, ``sites`` counts what came back. A language with
    ``files_seen > 0`` and ``sites == 0`` and ``status == "none"`` is the
    honest form of "we read nothing here, and here is why".
    """

    language: str
    status: CoverageStatus
    files_seen: int
    files_extracted: int
    sites: int
    reason: str
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Everything one extraction pass produced."""

    sites: tuple[ReferenceSiteRecord, ...]
    coverage: tuple[CoverageRecord, ...]

    def by_language(self) -> dict[str, CoverageRecord]:
        return {record.language: record for record in self.coverage}


@dataclass(frozen=True, slots=True)
class RenameSite:
    """One site a rename would touch, as reported by the preview."""

    file_path: str
    language: str
    name: str
    kind: ReferenceKind
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    range_exact: bool
    origin: ReferenceOrigin
    confidence: float
    tier: str
    occurrence_index: int
    enclosing_symbol_id: str | None
    #: True when confidence clears the certain bar and the columns are exact,
    #: i.e. a rewriter could patch this site unattended.
    mechanically_safe: bool


@dataclass(frozen=True, slots=True)
class RenamePreview:
    """The answer to "what would renaming this symbol touch" — and nothing else.

    This type has no apply path and the module that builds it performs no
    writes. A preview is evidence for a human or a downstream rewriter; the
    decision to edit is not ours.
    """

    symbol_id: str
    old_name: str
    new_name: str | None
    sites: tuple[RenameSite, ...]
    coverage: tuple[CoverageRecord, ...]
    #: Reasons the preview may be incomplete, in the order a reader should
    #: weigh them. Empty means every language in the repo is fully gated and
    #: every site cleared the certain bar.
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def files_touched(self) -> tuple[str, ...]:
        return tuple(sorted({site.file_path for site in self.sites}))


__all__ = [
    "CoverageRecord",
    "ExtractionResult",
    "ReferenceSiteRecord",
    "RenamePreview",
    "RenameSite",
]
