"""Give symbol parent display text and exact identity separate columns.

``graph_nodes.parent_symbol_id`` historically stored ``parent_name``.  A3's
lexical-local model needs a fetchable, collision-safe parent ID, so this
migration preserves the legacy text in ``parent_name`` and backfills the ID
only when one same-repository, same-file type is uniquely provable.  The same
exact-ID column is added to ``wiki_symbols``.

Revision ID: 0057
Revises: 0056
Create Date: 2026-08-21
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0057"
down_revision: str | None = "0056"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PARENT_INDEX = "ix_wiki_symbols_repo_parent_symbol_id"
_TYPE_KINDS = "'class', 'struct', 'interface', 'enum', 'trait', 'impl'"


def _unique_parent_id_sql(table: str, id_column: str) -> str:
    return f"""
        UPDATE {table} AS child
        SET parent_symbol_id = CASE
            WHEN (
                SELECT COUNT(*)
                FROM {table} AS parent
                WHERE parent.repository_id = child.repository_id
                  AND parent.file_path = child.file_path
                  AND parent.name = child.parent_name
                  AND parent.kind IN ({_TYPE_KINDS})
                  AND parent.{id_column} <> child.{id_column}
            ) = 1
            THEN (
                SELECT parent.{id_column}
                FROM {table} AS parent
                WHERE parent.repository_id = child.repository_id
                  AND parent.file_path = child.file_path
                  AND parent.name = child.parent_name
                  AND parent.kind IN ({_TYPE_KINDS})
                  AND parent.{id_column} <> child.{id_column}
            )
            ELSE NULL
        END
        WHERE child.parent_name IS NOT NULL
    """


def upgrade() -> None:
    with op.batch_alter_table("graph_nodes") as batch_op:
        batch_op.add_column(sa.Column("parent_name", sa.String(length=255), nullable=True))
    with op.batch_alter_table("wiki_symbols") as batch_op:
        batch_op.add_column(sa.Column("parent_symbol_id", sa.Text(), nullable=True))

    bind = op.get_bind()
    # Every value written by pre-0057 code is display text, not an ID.
    bind.execute(
        sa.text(
            "UPDATE graph_nodes SET parent_name = parent_symbol_id "
            "WHERE node_type = 'symbol' AND parent_symbol_id IS NOT NULL"
        )
    )
    bind.execute(sa.text(_unique_parent_id_sql("graph_nodes", "node_id")))
    bind.execute(sa.text(_unique_parent_id_sql("wiki_symbols", "symbol_id")))

    op.create_index(
        _PARENT_INDEX,
        "wiki_symbols",
        ["repository_id", "parent_symbol_id"],
        unique=False,
    )


def downgrade() -> None:
    # Restore the pre-0057 graph contract before removing the display column.
    op.get_bind().execute(
        sa.text("UPDATE graph_nodes SET parent_symbol_id = parent_name WHERE node_type = 'symbol'")
    )
    op.drop_index(_PARENT_INDEX, table_name="wiki_symbols")
    with op.batch_alter_table("wiki_symbols") as batch_op:
        batch_op.drop_column("parent_symbol_id")
    with op.batch_alter_table("graph_nodes") as batch_op:
        batch_op.drop_column("parent_name")
