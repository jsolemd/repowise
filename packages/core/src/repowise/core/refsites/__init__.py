"""Reference sites — a separate store of *where every name is spelled*.

The dependency graph answers "does A use B". It is unique per
``(source, target, edge_type)``, which is what makes it cheap to traverse and
what makes it lossy: N call sites collapse onto one edge, an import statement
keeps no line, and a symbol declared inside an anonymous callable is deleted
before it can become a node at all. Those losses are correct for a graph and
fatal for a rename.

This package is the other half. It records one row per *occurrence* — file,
range, kind, resolution confidence, and the symbol ids involved — in its own
tables, with its own migration, joined to nothing. It is not a coordinate on
``GraphEdge`` and must not become one: the moment an occurrence column is
added to an edge row, the edge table loses the uniqueness that makes it fast,
and the occurrence loses the position that makes it useful.

Entry points:

``extract_repository``
    Parse a repository and produce every site plus measured coverage.
``SqlReferenceSiteStore``
    Persist and query them.
``preview_rename``
    Every site a rename would touch, with per-site confidence. Reads the
    store. Applies nothing.
"""

from __future__ import annotations

from repowise.core.refsites.coverage import (
    CoverageStatus,
    LanguageCoverage,
    coverage_for,
    declared_coverage,
    gated_languages,
)
from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.records import (
    CoverageRecord,
    ExtractionResult,
    ReferenceSiteRecord,
    RenamePreview,
    RenameSite,
)
from repowise.core.refsites.rename import preview_rename
from repowise.core.refsites.schema import ReferenceSite, ReferenceSiteCoverage, ensure_schema
from repowise.core.refsites.store import SqlReferenceSiteStore
from repowise.core.refsites.taxonomy import (
    CONFIDENCE_ADVISORY,
    CONFIDENCE_CERTAIN,
    EXTRACTOR_VERSION,
    ReferenceKind,
    ReferenceOrigin,
)

__all__ = [
    "CONFIDENCE_ADVISORY",
    "CONFIDENCE_CERTAIN",
    "EXTRACTOR_VERSION",
    "CoverageRecord",
    "CoverageStatus",
    "ExtractionResult",
    "LanguageCoverage",
    "ReferenceKind",
    "ReferenceOrigin",
    "ReferenceSite",
    "ReferenceSiteCoverage",
    "ReferenceSiteRecord",
    "RenamePreview",
    "RenameSite",
    "SqlReferenceSiteStore",
    "coverage_for",
    "declared_coverage",
    "ensure_schema",
    "extract_repository",
    "gated_languages",
    "preview_rename",
]
