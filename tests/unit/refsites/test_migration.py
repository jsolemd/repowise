"""Migration 0063 adds the reference-site store, and takes it away again.

Locks three things the runtime path cannot prove on its own: the revision
chains onto 0062, the tables and indexes it declares match the ORM models
(so a managed upgrade and ``ensure_schema`` cannot diverge), and the downgrade
leaves the database exactly as it found it.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config

import repowise.core
from repowise.core.refsites.schema import REFSITE_TABLES

_TABLES = ("reference_sites", "reference_site_coverage")
_INDEXES = (
    "ix_reference_sites_repo_target",
    "ix_reference_sites_repo_name",
    "ix_reference_sites_repo_file",
)

# packages/core/src/repowise/core/__init__.py -> packages/core.
#
# Derived from the imported package rather than from ``Path("packages/core")``,
# which is what the older migration tests do: that form resolves against the
# working directory, and several CLI tests leave the process chdir'd inside a
# pytest tmp_path. Alembic still has to run with ``packages/core`` as the cwd
# (``script_location`` is relative), so the chdir stays — only the target is
# now independent of whoever ran before us.
_CORE_ROOT = Path(repowise.core.__file__).resolve().parents[3]


def _run_migration(db_path: Path, revision: str, *, downgrade: bool = False) -> None:
    url = f"sqlite+aiosqlite:///{db_path}"
    previous_url = os.environ.get("DATABASE_URL")
    try:
        previous_cwd = Path.cwd()
    except OSError:
        # An earlier test left us in a directory that has since been removed.
        previous_cwd = _CORE_ROOT
    os.environ["DATABASE_URL"] = url
    try:
        os.chdir(_CORE_ROOT)
        config = Config("alembic.ini")
        config.set_main_option("sqlalchemy.url", url)
        with patch("logging.config.fileConfig"):
            operation = command.downgrade if downgrade else command.upgrade
            operation(config, revision)
    finally:
        os.chdir(previous_cwd)
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url


def _names(db_path: Path, kind: str) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))
        return {row[0] for row in rows}


@pytest.fixture
def migrated(tmp_path: Path) -> Path:
    db_path = tmp_path / "wiki.db"
    _run_migration(db_path, "0063")
    return db_path


def test_upgrade_creates_the_tables_and_indexes(migrated: Path):
    tables = _names(migrated, "table")
    assert set(_TABLES) <= tables

    indexes = _names(migrated, "index")
    assert set(_INDEXES) <= indexes


def _table_shape(conn: sqlite3.Connection, name: str) -> dict[str, object]:
    """Everything PRAGMA can say about a table: columns with type/null/pk,
    foreign keys with their delete rule, and every index with its columns
    and uniqueness. Name-set equality alone would bless a Float column that
    one path created as Text, or a NOT NULL the other path dropped."""
    columns = {
        row[1]: {"type": row[2].upper(), "notnull": bool(row[3]), "pk": bool(row[5])}
        for row in conn.execute(f"PRAGMA table_info({name})")
    }
    foreign_keys = {
        (row[2], row[3], row[4], row[6])  # (target table, from, to, on_delete)
        for row in conn.execute(f"PRAGMA foreign_key_list({name})")
    }
    indexes = {}
    for index_row in conn.execute(f"PRAGMA index_list({name})"):
        index_name, unique = index_row[1], bool(index_row[2])
        if index_name.startswith("sqlite_autoindex"):
            continue
        cols = tuple(r[2] for r in conn.execute(f"PRAGMA index_info({index_name})"))
        indexes[index_name] = {"unique": unique, "columns": cols}
    return {"columns": columns, "foreign_keys": foreign_keys, "indexes": indexes}


def test_migration_schema_matches_the_orm_models(migrated: Path, tmp_path: Path):
    """A managed upgrade and ``Base.metadata.create_all`` must agree.

    Both are live paths — Alembic for a managed database, ``ensure_schema``
    for a runtime one — so a divergence produces a store that works on one
    machine and not the next. Compared shape-for-shape, not name-for-name:
    types, nullability, primary keys, foreign keys (including the CASCADE
    the delete path relies on), and every index with its uniqueness.
    """
    from sqlalchemy import create_engine as _sync_engine

    orm_db = tmp_path / "orm.db"
    engine = _sync_engine(f"sqlite:///{orm_db}")
    for table in REFSITE_TABLES:
        table.create(engine)
    engine.dispose()

    with sqlite3.connect(migrated) as via_migration, sqlite3.connect(orm_db) as via_orm:
        for table in REFSITE_TABLES:
            assert _table_shape(via_migration, table.name) == _table_shape(via_orm, table.name), (
                table.name
            )


def test_position_uniqueness_is_enforced_by_the_database(migrated: Path):
    """Two rows may share a target; they may not share a position."""
    with sqlite3.connect(migrated) as conn:
        conn.execute("INSERT INTO repositories (id, name, local_path) VALUES ('r', 'r', '/tmp/r')")
        row = (
            "r",
            "a.ts",
            "typescript",
            "foo",
            "call",
            7,
            7,
            24,
            27,
            1,
            "a.ts::foo",
            None,
            "same_file",
            0.95,
            "ast",
            0,
            "refsites/1",
            "2026-08-22 00:00:00",
        )
        columns = (
            "repository_id, file_path, language, name, kind, start_line, end_line, "
            "start_col, end_col, range_exact, target_symbol_id, enclosing_symbol_id, "
            "resolution_origin, confidence, tier, occurrence_index, extractor_version, "
            "created_at"
        )
        placeholders = ", ".join("?" * 18)
        conn.execute(
            f"INSERT INTO reference_sites (id, {columns}) VALUES ('1', {placeholders})", row
        )
        # Same target, different column — allowed, and the whole point.
        conn.execute(
            f"INSERT INTO reference_sites (id, {columns}) VALUES ('2', {placeholders})",
            (*row[:7], 47, 50, *row[9:]),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"INSERT INTO reference_sites (id, {columns}) VALUES ('3', {placeholders})",
                row,
            )


def test_downgrade_removes_everything_it_added(tmp_path: Path):
    db_path = tmp_path / "wiki.db"
    _run_migration(db_path, "0063")
    before_tables = _names(db_path, "table")

    _run_migration(db_path, "0062", downgrade=True)

    remaining = _names(db_path, "table")
    assert not set(_TABLES) & remaining
    assert not set(_INDEXES) & _names(db_path, "index")
    # Nothing else was disturbed on the way out.
    assert remaining == before_tables - set(_TABLES)
