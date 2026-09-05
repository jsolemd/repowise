"""Migration 0062 separates parent display text from exact symbol identity."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config


def _run_migration(db_path: Path, revision: str, *, downgrade: bool = False) -> None:
    core_root = Path("packages/core").resolve()
    url = f"sqlite+aiosqlite:///{db_path}"
    previous_url = os.environ.get("DATABASE_URL")
    previous_cwd = Path.cwd()
    os.environ["DATABASE_URL"] = url
    try:
        os.chdir(core_root)
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


def _seed_legacy_rows(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO repositories (id, name, local_path) VALUES (?, ?, ?)",
            ("repo", "repo", "/tmp/repo"),
        )
        graph_rows = [
            ("gp", "src/a.py::Owner", "class", "Owner", None),
            ("gc", "src/a.py::Owner::run", "method", "run", "Owner"),
            ("ga1", "src/b.py::Duplicate", "class", "Duplicate", None),
            ("ga2", "src/b.py::scope::Duplicate", "class", "Duplicate", None),
            ("gac", "src/b.py::Duplicate::run", "method", "run", "Duplicate"),
        ]
        for row_id, node_id, kind, name, legacy_parent in graph_rows:
            conn.execute(
                "INSERT INTO graph_nodes "
                "(id, repository_id, node_id, node_type, file_path, kind, name, "
                "parent_symbol_id) VALUES (?, 'repo', ?, 'symbol', ?, ?, ?, ?)",
                (row_id, node_id, node_id.split("::", 1)[0], kind, name, legacy_parent),
            )

        symbol_rows = [
            ("wp", "src/a.py::Owner", "src/a.py", "Owner", "class", None),
            ("wc", "src/a.py::Owner::run", "src/a.py", "run", "method", "Owner"),
            ("wa1", "src/b.py::Duplicate", "src/b.py", "Duplicate", "class", None),
            (
                "wa2",
                "src/b.py::scope::Duplicate",
                "src/b.py",
                "Duplicate",
                "class",
                None,
            ),
            (
                "wac",
                "src/b.py::Duplicate::run",
                "src/b.py",
                "run",
                "method",
                "Duplicate",
            ),
        ]
        for row_id, symbol_id, file_path, name, kind, parent_name in symbol_rows:
            conn.execute(
                "INSERT INTO wiki_symbols "
                "(id, repository_id, file_path, symbol_id, name, qualified_name, kind, "
                "parent_name) VALUES (?, 'repo', ?, ?, ?, ?, ?, ?)",
                (row_id, file_path, symbol_id, name, name, kind, parent_name),
            )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def test_migration_0057_upgrade_and_downgrade_preserve_parent_semantics(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "migration.db"
    _run_migration(db_path, "0064")
    _seed_legacy_rows(db_path)

    _run_migration(db_path, "0065")
    with sqlite3.connect(db_path) as conn:
        assert "parent_name" in _columns(conn, "graph_nodes")
        assert "parent_symbol_id" in _columns(conn, "wiki_symbols")
        graph = conn.execute(
            "SELECT parent_name, parent_symbol_id FROM graph_nodes WHERE id = 'gc'"
        ).fetchone()
        symbol = conn.execute(
            "SELECT parent_name, parent_symbol_id FROM wiki_symbols WHERE id = 'wc'"
        ).fetchone()
        assert graph == ("Owner", "src/a.py::Owner")
        assert symbol == ("Owner", "src/a.py::Owner")
        # Ambiguous legacy names stay useful as display text but do not become
        # fabricated identifiers.
        assert conn.execute(
            "SELECT parent_name, parent_symbol_id FROM graph_nodes WHERE id = 'gac'"
        ).fetchone() == ("Duplicate", None)
        assert conn.execute(
            "SELECT parent_name, parent_symbol_id FROM wiki_symbols WHERE id = 'wac'"
        ).fetchone() == ("Duplicate", None)
        indexes = {str(row[1]) for row in conn.execute("PRAGMA index_list(wiki_symbols)")}
        assert "ix_wiki_symbols_repo_parent_symbol_id" in indexes

    _run_migration(db_path, "0064", downgrade=True)
    with sqlite3.connect(db_path) as conn:
        assert "parent_name" not in _columns(conn, "graph_nodes")
        assert "parent_symbol_id" not in _columns(conn, "wiki_symbols")
        assert conn.execute(
            "SELECT parent_symbol_id FROM graph_nodes WHERE id = 'gc'"
        ).fetchone() == ("Owner",)
        assert conn.execute(
            "SELECT parent_symbol_id FROM graph_nodes WHERE id = 'gac'"
        ).fetchone() == ("Duplicate",)
