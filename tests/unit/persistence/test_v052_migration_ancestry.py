"""The absorption must advance shipped fork heads without skipping new DDL."""

from __future__ import annotations

import importlib.util
import io
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
    assert scripts.get_heads() == ["0095"]
    revisions = list(scripts.iterate_revisions("head", "solemd_0001"))
    assert [r.revision for r in reversed(revisions)] == [f"{r:04}" for r in range(65, 96)]
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
        assert conn.execute("SELECT version_num FROM alembic_version").fetchone() == ("0095",)
        assert conn.execute("SELECT name FROM repositories WHERE id='repo'").fetchone() == ("repo",)
        columns = {r[1] for r in conn.execute("PRAGMA table_info(repositories)")}
        assert "symbols_parser_fingerprint" in columns
        assert "function_mod_p80" in columns
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"doc_drift_findings", "doc_drift_references", "reference_sites"} <= tables
        assert "code_origin" in {
            r[1] for r in conn.execute("PRAGMA table_info(health_file_metrics)")
        }
        assert "commit_shas_json" in {
            r[1] for r in conn.execute("PRAGMA table_info(git_function_blame)")
        }
        assert "digest" in {r[1] for r in conn.execute("PRAGMA table_info(wiki_pages)")}
        assert "digest" in {r[1] for r in conn.execute("PRAGMA table_info(page_fts)")}


def test_v055_postgresql_ddl_preserves_additive_columns_and_nullable_scores():
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    root = Path(__file__).resolve().parents[3] / "packages/core/alembic/versions"
    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        for revision in range(92, 96):
            path = next(root.glob(f"{revision:04}_*.py"))
            spec = importlib.util.spec_from_file_location(f"migration_{revision}", path)
            migration = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migration)
            migration.upgrade()
    ddl = output.getvalue()
    assert "ALTER TABLE health_file_metrics ADD COLUMN code_origin VARCHAR(16)" in ddl
    assert "ALTER TABLE git_function_blame ADD COLUMN commit_shas_json TEXT" in ddl
    for column in ("score", "max_ccn", "max_nesting"):
        assert f"ALTER TABLE health_file_metrics ALTER COLUMN {column} DROP NOT NULL" in ddl
    assert "ALTER TABLE wiki_pages ADD COLUMN digest TEXT DEFAULT '' NOT NULL" in ddl
    assert "CREATE INDEX idx_wiki_pages_fts ON wiki_pages USING GIN" in ddl
    assert "COALESCE(digest,'')" in ddl


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
