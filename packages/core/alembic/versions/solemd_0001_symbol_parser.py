"""Record the parser that completely refreshed persisted symbols.

Revision ID: solemd_0001
Revises: solemd_0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "solemd_0001"
down_revision: str | None = "solemd_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "repositories", sa.Column("symbols_parser_fingerprint", sa.String(64), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("repositories", "symbols_parser_fingerprint")
