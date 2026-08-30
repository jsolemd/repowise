"""Add the reference-site store and its measured coverage table.

A graph edge is unique per ``(source, target, edge_type)``: that uniqueness is
what makes traversal cheap, and it is also why every call site to a function
collapses onto one row. Renames, and any other question of the form "where
exactly", need the rows the edge discards.

Rather than widen ``graph_edges`` with occurrence columns — which would have
meant dropping ``uq_graph_edge_typed`` and paying for it on every traversal —
this revision adds a separate store keyed on *position*. Nothing joins the two
at the schema level; a reference site names symbol ids as text, exactly as
``graph_nodes.node_id`` does.

``reference_site_coverage`` records what extraction actually saw per language.
It exists so a query that returns no sites can distinguish "this symbol is
unreferenced" from "nobody taught the extractor to read this language".

Revision ID: 0063
Revises: 0062
Create Date: 2026-08-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0063"
down_revision: str | None = "0062"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SITE_TARGET_INDEX = "ix_reference_sites_repo_target"
_SITE_NAME_INDEX = "ix_reference_sites_repo_name"
_SITE_FILE_INDEX = "ix_reference_sites_repo_file"


def upgrade() -> None:
    op.create_table(
        "reference_sites",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("repository_id", sa.String(length=32), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        # NOT NULL with a 0 fallback, never nullable: SQLite treats NULLs as
        # distinct, so a nullable column would let the uniqueness constraint
        # below admit unlimited duplicates of an unlocated site.
        sa.Column("start_col", sa.Integer(), nullable=False),
        sa.Column("end_col", sa.Integer(), nullable=False),
        sa.Column("range_exact", sa.Boolean(), nullable=False),
        sa.Column("target_symbol_id", sa.Text(), nullable=True),
        sa.Column("enclosing_symbol_id", sa.Text(), nullable=True),
        sa.Column("resolution_origin", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("tier", sa.String(length=8), nullable=False),
        sa.Column("occurrence_index", sa.Integer(), nullable=False),
        sa.Column("extractor_version", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "repository_id",
            "file_path",
            "start_line",
            "start_col",
            "name",
            "kind",
            name="uq_reference_site_position",
        ),
    )
    op.create_index(
        _SITE_TARGET_INDEX,
        "reference_sites",
        ["repository_id", "target_symbol_id", "confidence"],
        unique=False,
    )
    op.create_index(_SITE_NAME_INDEX, "reference_sites", ["repository_id", "name"], unique=False)
    op.create_index(
        _SITE_FILE_INDEX, "reference_sites", ["repository_id", "file_path"], unique=False
    )

    op.create_table(
        "reference_site_coverage",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("repository_id", sa.String(length=32), nullable=False),
        sa.Column("language", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("files_seen", sa.Integer(), nullable=False),
        sa.Column("files_extracted", sa.Integer(), nullable=False),
        sa.Column("sites", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("limitations_json", sa.Text(), nullable=False),
        sa.Column("extractor_version", sa.String(length=16), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("repository_id", "language", name="uq_reference_site_coverage"),
    )


def downgrade() -> None:
    op.drop_table("reference_site_coverage")
    op.drop_index(_SITE_FILE_INDEX, table_name="reference_sites")
    op.drop_index(_SITE_NAME_INDEX, table_name="reference_sites")
    op.drop_index(_SITE_TARGET_INDEX, table_name="reference_sites")
    op.drop_table("reference_sites")
