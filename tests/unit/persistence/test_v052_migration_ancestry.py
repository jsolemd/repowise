"""The absorption must advance shipped fork heads without skipping new DDL."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory


def _config(path: Path) -> Config:
    config = Config()
    root = Path(__file__).resolve().parents[3] / "packages/core/alembic"
    config.set_main_option("script_location", str(root))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{path}")
    return config


def test_shipped_fork_head_is_ancestor_of_upstream_tail():
    scripts = ScriptDirectory.from_config(_config(Path("unused.db")))
    assert scripts.get_heads() == ["0076"]
    revisions = list(scripts.iterate_revisions("head", "solemd_0001"))
    assert [r.revision for r in reversed(revisions)] == [f"{r:04}" for r in range(65, 77)]
    assert scripts.get_revision("solemd_0001").down_revision == "solemd_0005"


def test_existing_fork_head_gets_upstream_ddl(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db = tmp_path / "fork.db"
    config = _config(db)
    with patch("logging.config.fileConfig"):
        command.upgrade(config, "solemd_0001")
        with sqlite3.connect(db) as conn:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert "source_index_updates" in tables
            assert "reference_sites" in tables
            assert "doc_drift_findings" not in tables
            conn.execute(
                "INSERT INTO repositories (id, name, local_path) VALUES ('repo', 'repo', '/tmp/repo')"
            )
        command.upgrade(config, "head")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("0076",)
        assert conn.execute("SELECT name FROM repositories WHERE id='repo'").fetchone() == ("repo",)
        columns = {r[1] for r in conn.execute("PRAGMA table_info(repositories)")}
        assert "symbols_parser_fingerprint" in columns
        assert "function_mod_p80" in columns
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"doc_drift_findings", "doc_drift_references", "reference_sites"} <= tables


def test_ambiguous_intermediate_fork_stamp_is_refused_without_rewriting(tmp_path, monkeypatch):
    import pytest

    monkeypatch.delenv("DATABASE_URL", raising=False)
    db = tmp_path / "intermediate.db"
    config = _config(db)
    with patch("logging.config.fileConfig"):
        command.upgrade(config, "solemd_0002")
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE alembic_version SET version_num='0065'")
        with pytest.raises(RuntimeError, match=r"Ambiguous pre-v0\.52"):
            command.upgrade(config, "head")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("0065",)
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE name='doc_drift_findings'"
            ).fetchone()
            is None
        )
