"""Add the transactional source-index update outbox.

Revision ID: 0064
Revises: 0063
Create Date: 2026-08-17
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0064"
down_revision: str | None = "0063"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "source_index_updates",
        sa.Column("sequence", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("generation_id", sa.String(length=32), nullable=False),
        sa.Column("repository_id", sa.String(length=32), nullable=False),
        sa.Column("parent_generation_id", sa.String(length=32), nullable=True),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("dedupe_key", sa.String(length=64), nullable=False),
        sa.Column("parser_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("change_set_json", sa.Text(), nullable=False),
        sa.Column("artifact_json", sa.Text(), nullable=True),
        sa.Column("upstream_ready", sa.Boolean(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ready_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["repository_id"], ["repositories.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("sequence"),
        sa.UniqueConstraint("generation_id"),
        sa.UniqueConstraint(
            "repository_id", "dedupe_key", name="uq_source_index_update_dedupe"
        ),
    )
    op.create_index(
        "ix_source_index_updates_repo_state",
        "source_index_updates",
        ["repository_id", "state", "sequence"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_source_index_updates_repo_state", table_name="source_index_updates")
    op.drop_table("source_index_updates")
