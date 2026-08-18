"""Preserve canonical decision-journal fields in the SQL projection.

The external JSONL authority carries ordered file/symbol/SHA anchors, both
directions of a supersession chain, and a separate confirmation timestamp.
These columns keep that information lossless while the existing affected-file
and decision-edge tables remain query-friendly derivatives.

``init_db``'s additive reconciler handles local SQLite stores that do not run
Alembic; this migration covers managed databases.

Revision ID: 0055
Revises: 0054
Create Date: 2026-08-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0055"
down_revision: str | None = "0054"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "decision_records",
        sa.Column("anchors_json", sa.Text(), nullable=False, server_default="[]"),
    )
    op.add_column(
        "decision_records",
        sa.Column("supersedes", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "decision_records",
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("decision_records", "confirmed_at")
    op.drop_column("decision_records", "supersedes")
    op.drop_column("decision_records", "anchors_json")
