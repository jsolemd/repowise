"""ORM tables for the reference-site store.

Two tables, both new, neither of them a coordinate on anything that already
exists. In particular ``reference_sites`` is **not** an extension of
``graph_edges``: a graph edge answers "does A use B", is unique per
(source, target, type), and collapses every call site onto one row. A
reference site answers "where, exactly, and how sure are we", is unique per
*position*, and is expected to have many rows per edge. Folding one into the
other would have forced the edge table to lose its uniqueness constraint,
which is what makes edge lookups cheap.

The models bind the shared :class:`Base` so they live in the same database,
migration chain and session as everything else — sharing a declarative base is
not the same as sharing a table. Because the runtime schema path
(``init_db`` -> ``Base.metadata.create_all`` + ``_reconcile_schema``) only
sees tables whose module has been imported, :func:`ensure_schema` exists so a
consumer can guarantee the tables without the shared bootstrap having to know
about this package.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import Mapped, mapped_column

from repowise.core.persistence.models import Base, _new_uuid, _now_utc


class ReferenceSite(Base):
    """One occurrence of one name at one position in one file."""

    __tablename__ = "reference_sites"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_uuid)
    repository_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    #: The identifier text as it appears at this position. Stored rather than
    #: derived from the target, because an aliased import site spells a name
    #: the target symbol does not have.
    name: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    # Columns are NOT NULL with a 0 fallback rather than nullable, so the
    # uniqueness constraint below actually constrains: SQLite treats NULLs as
    # distinct, so a nullable column would let unlocated sites duplicate
    # without limit. ``range_exact`` carries the truth instead.
    start_col: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    end_col: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    range_exact: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: NULL means the reference could not be bound to a definition. It is a
    #: recorded outcome, not a missing value — see ``ReferenceOrigin.UNRESOLVED``.
    target_symbol_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    enclosing_symbol_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_origin: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Denormalised from ``resolution_origin`` via the rubric so the database
    #: can order and threshold without loading rows into Python. The rubric
    #: remains the only writer.
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    tier: Mapped[str] = mapped_column(String(8), nullable=False)
    occurrence_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    extractor_version: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc
    )

    __table_args__ = (
        # Position identity. ``name`` and ``kind`` are in the key because a
        # method call contributes a CALL site for the method and a RECEIVER
        # site for the object, and an import statement contributes an IMPORT
        # site and one IMPORT_BINDING per name — all legitimately at or near
        # the same position.
        UniqueConstraint(
            "repository_id",
            "file_path",
            "start_line",
            "start_col",
            "name",
            "kind",
            name="uq_reference_site_position",
        ),
        # The rename-preview query: every site for one target, highest
        # confidence first. Without this it is a full scan of the largest
        # table in the store, which on a repo of any size is the whole cost
        # of the tool.
        Index(
            "ix_reference_sites_repo_target",
            "repository_id",
            "target_symbol_id",
            "confidence",
        ),
        # The "I only know the name" entry point, used when a caller passes a
        # bare identifier instead of a symbol id, and by the unresolved-site
        # sweep that a preview needs in order to report what it could not bind.
        Index("ix_reference_sites_repo_name", "repository_id", "name"),
        # Per-file replacement during incremental re-extraction.
        Index("ix_reference_sites_repo_file", "repository_id", "file_path"),
    )


class ReferenceSiteCoverage(Base):
    """Measured, per-repository, per-language extraction coverage.

    Persisted rather than recomputed so a query can disclose coverage without
    re-walking the repository. A row with ``files_seen > 0`` and ``sites = 0``
    is the record that keeps an uncovered language from reading as an absence
    of references.
    """

    __tablename__ = "reference_site_coverage"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_uuid)
    repository_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    language: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    files_seen: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_extracted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sites: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    limitations_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    extractor_version: Mapped[str] = mapped_column(String(16), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now_utc, onupdate=_now_utc
    )

    __table_args__ = (
        UniqueConstraint("repository_id", "language", name="uq_reference_site_coverage"),
    )


#: The tables this package owns. Named so :func:`ensure_schema` and the
#: migration cannot disagree about the set.
REFSITE_TABLES = (ReferenceSite.__table__, ReferenceSiteCoverage.__table__)


def _create(connection: Connection) -> None:
    Base.metadata.create_all(connection, tables=list(REFSITE_TABLES), checkfirst=True)


async def ensure_schema(engine: AsyncEngine) -> None:
    """Create this package's tables if they are absent. Idempotent, additive.

    ``init_db`` creates every table reachable from ``Base.metadata``, but a
    table is only reachable once its module has been imported, and nothing in
    the shared bootstrap imports this package. Rather than reach into that
    bootstrap, a consumer calls this once after opening an engine. Existing
    databases gain the tables; databases that already have them are untouched.
    The Alembic revision remains the authority for a managed upgrade — this is
    the runtime path, exactly as ``_reconcile_schema`` is for added columns.
    """
    async with engine.begin() as connection:
        await connection.run_sync(_create)


__all__ = [
    "REFSITE_TABLES",
    "ReferenceSite",
    "ReferenceSiteCoverage",
    "ensure_schema",
]
