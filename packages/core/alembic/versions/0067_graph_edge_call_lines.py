"""Preserve call-site lines on resolved graph edges.

Several calls between the same symbol pair collapse to one edge. The line list
lets performance analysis bind a loop/lock call to that resolved edge instead
of repeating a bare-name lookup. Existing rows default to an empty list and use
the documented compatibility fallback until the next index refresh.

Revision ID: 0067
Revises: 0066
Create Date: 2026-08-20

Re-headed from 0055/0054 at integration: upstream v0.45.0 and the fork both
minted revision 0055 under different filenames, which git merges without
conflict while alembic gains a second head. The fork's 0055 keeps its id;
this upstream migration moves to the tail. Safe to reorder — it only adds
graph_edges.call_lines_json; 0055-0058 touch unrelated tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0067"
down_revision: str | None = "0066"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "graph_edges",
        sa.Column("call_lines_json", sa.Text(), nullable=False, server_default="[]"),
    )


def downgrade() -> None:
    op.drop_column("graph_edges", "call_lines_json")
