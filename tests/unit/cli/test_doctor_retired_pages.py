"""Native repair reconciles live authority without reviving retired pages."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from repowise.cli.commands.doctor_cmd import repo_checks
from repowise.core.persistence import FullTextSearch, create_engine
from repowise.core.persistence.vector_store import LanceDBVectorStore
from repowise.core.providers.embedding.base import MockEmbedder

LIVE = "file_page:live.py"
RETIRED = "file_page:gone.py"
MISSING = "repo_overview:repo"
THIN = "file_page:thin.py"
WARNING = "Tombstone full-text removal"


@pytest.fixture(autouse=True)
def _offline_embedder(monkeypatch):
    # Exercise real stores and the shared embedding recipe without contacting
    # the host's configured provider or changing production environment pins.
    monkeypatch.setattr("repowise.cli.providers.resolve_embedder_for_repo", lambda _p: "mock")
    monkeypatch.setattr("repowise.cli.providers.build_embedder", lambda *_a: MockEmbedder())


async def _seed(tmp_path: Path) -> tuple[Path, str]:
    import git

    from repowise.cli.helpers import save_config_partial
    from repowise.core.persistence import create_session_factory, get_session
    from repowise.core.persistence.crud import upsert_page, upsert_repository
    from repowise.core.persistence.database import init_db
    from repowise.core.persistence.models import DecisionRecord

    root = (tmp_path / "repo").resolve()
    root.mkdir()
    git.Repo.init(root)
    (root / ".repowise").mkdir()
    save_config_partial(root, embedder="mock")
    engine = create_engine(f"sqlite+aiosqlite:///{root}/.repowise/wiki.db")
    await init_db(engine)
    async with get_session(create_session_factory(engine)) as session:
        repo = await upsert_repository(session, name="repo", local_path=str(root))
        for pid, status, body in [
            (LIVE, "stale", "A retained live page with useful descriptive content."),
            (RETIRED, "tombstone", "Historical page for removed implementation."),
            (MISSING, "fresh", "A live orientation page requiring its missing vector."),
            (THIN, "expired", ""),
        ]:
            page = await upsert_page(
                session,
                page_id=pid,
                repository_id=repo.id,
                page_type="file_page",
                title=pid,
                content=body,
                summary="",
                target_path=pid.split(":", 1)[1],
                source_hash="hash",
                model_name="template",
                provider_name="template",
            )
            page.freshness_status = status
        decision = DecisionRecord(
            repository_id=repo.id,
            title="Keep decisions",
            decision="Retain accepted decisions independently of pages",
        )
        session.add(decision)
        await session.commit()
        did = f"decision:{decision.id}"
    fts = FullTextSearch(engine)
    await fts.ensure_index()
    for pid in (LIVE, RETIRED, THIN):
        await fts.index(pid, pid, "Stored searchable content", target_path=pid)
    await engine.dispose()
    vs = LanceDBVectorStore(str(root / ".repowise/lancedb"), embedder=MockEmbedder())
    for pid in (LIVE, RETIRED, THIN, did):
        await vs.embed_and_upsert(pid, "Stored vector input", {"title": pid})
    await vs.close()
    state = {
        "last_sync_commit": "source-witness",
        "custom_witness": {"value": 17},
        "index_scope": {
            "version": 1,
            "analysis": {
                "unavailable": [WARNING, "Unrelated analysis"],
                "skipped": ["Independent skip"],
            },
        },
    }
    (root / ".repowise/state.json").write_text(json.dumps(state))
    return root, did


def _fts_ids(root: Path) -> set[str]:
    with sqlite3.connect(root / ".repowise/wiki.db") as conn:
        return {r[0] for r in conn.execute("SELECT page_id FROM page_fts")}


async def _vector_ids(root: Path) -> set[str]:
    vs = LanceDBVectorStore(str(root / ".repowise/lancedb"), embedder=MockEmbedder())
    try:
        return await vs.list_page_ids()
    finally:
        await vs.close()


def test_report_uses_live_authority(tmp_path: Path):
    root, _ = asyncio.run(_seed(tmp_path))
    _, checks = repo_checks._run_repo_checks(root, repair=False)
    rows = {c.name: c.detail for c in checks}
    assert rows["SQL ↔ Vector Store"] == "1 missing, 1 orphaned"
    assert rows["SQL ↔ FTS Index"] == "1 missing, 1 orphaned"
    assert "SQL=4, Vector=4" in rows["Coordinator drift"]


@pytest.mark.parametrize("legacy_warning", [False, True])
def test_repair_retires_search_rows_and_preserves_live_decisions(tmp_path: Path, legacy_warning):
    root, did = asyncio.run(_seed(tmp_path))
    if legacy_warning:
        state_path = root / ".repowise/state.json"
        state = json.loads(state_path.read_text())
        state["degraded"] = [WARNING, "Unrelated analysis"]
        state_path.write_text(json.dumps(state))
    repo_checks._run_repo_checks(root, repair=True)
    assert _fts_ids(root) == {LIVE, MISSING, THIN}
    assert asyncio.run(_vector_ids(root)) == {LIVE, MISSING, THIN, did}
    with sqlite3.connect(root / ".repowise/wiki.db") as conn:
        assert conn.execute(
            "SELECT freshness_status FROM wiki_pages WHERE id=?", (RETIRED,)
        ).fetchone() == ("tombstone",)
    state = json.loads((root / ".repowise/state.json").read_text())
    assert state["index_scope"]["analysis"] == {
        "unavailable": ["Unrelated analysis"],
        "skipped": ["Independent skip"],
    }
    assert state["last_sync_commit"] == "source-witness"
    assert state["custom_witness"] == {"value": 17}
    if legacy_warning:
        assert state["degraded"] == ["Unrelated analysis"]


def test_upgrade_does_not_refill_tombstones(tmp_path: Path):
    root, _ = asyncio.run(_seed(tmp_path))
    with sqlite3.connect(root / ".repowise/wiki.db") as conn:
        conn.execute("DROP TABLE page_fts")
        conn.execute("CREATE VIRTUAL TABLE page_fts USING fts5(page_id UNINDEXED, title, content)")
        conn.execute("INSERT INTO page_fts VALUES (?, 'live', 'body')", (LIVE,))

    async def upgrade():
        engine = create_engine(f"sqlite+aiosqlite:///{root}/.repowise/wiki.db")
        try:
            await FullTextSearch(engine).ensure_index()
        finally:
            await engine.dispose()

    asyncio.run(upgrade())
    assert _fts_ids(root) == {LIVE, MISSING, THIN}


def test_failed_fts_cleanup_retains_debt_and_warning_but_repairs_vectors(
    tmp_path: Path, monkeypatch
):
    root, did = asyncio.run(_seed(tmp_path))
    original = FullTextSearch.delete_many

    async def fail(self, ids):
        raise OSError("injected unavailable FTS writer")

    monkeypatch.setattr(FullTextSearch, "delete_many", fail)
    repo_checks._run_repo_checks(root, repair=True)
    from repowise.core.pipeline.cleanup_debt import load_cleanup_debt

    assert RETIRED in load_cleanup_debt(root)["fts"]
    assert (
        WARNING
        in json.loads((root / ".repowise/state.json").read_text())["index_scope"]["analysis"][
            "unavailable"
        ]
    )
    assert asyncio.run(_vector_ids(root)) == {LIVE, MISSING, THIN, did}
    monkeypatch.setattr(FullTextSearch, "delete_many", original)
    repo_checks._run_repo_checks(root, repair=True)
    assert RETIRED not in _fts_ids(root)
    assert not any(load_cleanup_debt(root).values())


def test_repair_refuses_active_native_writer(tmp_path: Path):
    from repowise.core.update_lock import strict_update_lock

    root, _ = asyncio.run(_seed(tmp_path))
    before = _fts_ids(root), asyncio.run(_vector_ids(root))
    with strict_update_lock(root):
        repo_checks._run_repo_checks(root, repair=True)
    assert (_fts_ids(root), asyncio.run(_vector_ids(root))) == before


def test_revived_page_is_rechecked_before_deletion(tmp_path: Path, monkeypatch):
    root, _ = asyncio.run(_seed(tmp_path))
    from repowise.core.pipeline import cleanup_debt

    original = cleanup_debt.exclude_live_cleanup_ids

    async def revive(path, engine, ids):
        from sqlalchemy import text

        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE wiki_pages SET freshness_status='fresh' WHERE id=:id"), {"id": RETIRED}
            )
        return await original(path, engine, ids)

    monkeypatch.setattr(cleanup_debt, "exclude_live_cleanup_ids", revive)
    repo_checks._run_repo_checks(root, repair=True)
    assert RETIRED in _fts_ids(root)
    assert RETIRED in asyncio.run(_vector_ids(root))


def test_authority_failure_preserves_search_rows_and_retry_intent(tmp_path: Path, monkeypatch):
    root, _ = asyncio.run(_seed(tmp_path))
    from repowise.core.pipeline import cleanup_debt

    async def unavailable(*_a):
        raise OSError("authority unavailable")

    monkeypatch.setattr(cleanup_debt, "exclude_live_cleanup_ids", unavailable)
    before = _fts_ids(root), asyncio.run(_vector_ids(root))
    repo_checks._run_repo_checks(root, repair=True)
    assert (_fts_ids(root), asyncio.run(_vector_ids(root))) == before
    assert cleanup_debt.load_cleanup_debt(root) == {"fts": {RETIRED}, "vectors": {RETIRED}}
    assert (
        WARNING
        in json.loads((root / ".repowise/state.json").read_text())["index_scope"]["analysis"][
            "unavailable"
        ]
    )


def test_only_retired_pages_are_reconciled_without_reembedding(tmp_path: Path):
    root, did = asyncio.run(_seed(tmp_path))
    with sqlite3.connect(root / ".repowise/wiki.db") as conn:
        conn.execute("UPDATE wiki_pages SET freshness_status='tombstone'")
    repo_checks._run_repo_checks(root, repair=True)
    assert _fts_ids(root) == set()
    assert asyncio.run(_vector_ids(root)) == {did}
    with sqlite3.connect(root / ".repowise/wiki.db") as conn:
        assert conn.execute("SELECT count(*) FROM wiki_pages").fetchone()[0] == 4


def test_report_only_does_not_remove_retired_rows(tmp_path: Path):
    root, _ = asyncio.run(_seed(tmp_path))
    before = (
        _fts_ids(root),
        asyncio.run(_vector_ids(root)),
        (root / ".repowise/state.json").read_bytes(),
    )
    repo_checks._run_repo_checks(root, repair=False)
    assert (
        _fts_ids(root),
        asyncio.run(_vector_ids(root)),
        (root / ".repowise/state.json").read_bytes(),
    ) == before


def test_failed_schema_upgrade_does_not_block_retirement(tmp_path: Path, monkeypatch):
    root, did = asyncio.run(_seed(tmp_path))

    async def fail(self):
        raise OSError("schema upgrade unavailable")

    monkeypatch.setattr(FullTextSearch, "ensure_index", fail)
    repo_checks._run_repo_checks(root, repair=True)
    assert _fts_ids(root) == {LIVE, MISSING, THIN}
    assert asyncio.run(_vector_ids(root)) == {LIVE, MISSING, THIN, did}


def test_empty_fts_inventory_is_missing_live_content(tmp_path: Path):
    root, _ = asyncio.run(_seed(tmp_path))
    with sqlite3.connect(root / ".repowise/wiki.db") as conn:
        conn.execute("DELETE FROM page_fts")
    _, checks = repo_checks._run_repo_checks(root, repair=False)
    assert {c.name: c.detail for c in checks}["SQL ↔ FTS Index"] == "3 missing, 0 orphaned"
    repo_checks._run_repo_checks(root, repair=True)
    assert _fts_ids(root) == {LIVE, MISSING, THIN}


def test_uncommitted_decision_vector_stays_with_projection_owner(tmp_path: Path):
    root, did = asyncio.run(_seed(tmp_path))
    pending = "decision:still-being-projected"

    async def project():
        vs = LanceDBVectorStore(str(root / ".repowise/lancedb"), embedder=MockEmbedder())
        try:
            await vs.embed_and_upsert(pending, "Pending decision", {"title": "Pending"})
        finally:
            await vs.close()

    asyncio.run(project())
    repo_checks._run_repo_checks(root, repair=True)
    assert asyncio.run(_vector_ids(root)) == {LIVE, MISSING, THIN, did, pending}
    from repowise.core.pipeline.cleanup_debt import load_cleanup_debt

    assert not any(load_cleanup_debt(root).values())


def test_pending_debt_retries_even_after_search_row_is_already_gone(tmp_path: Path):
    root, _ = asyncio.run(_seed(tmp_path))
    repo_checks._run_repo_checks(root, repair=True)
    from repowise.core.pipeline.cleanup_debt import load_cleanup_debt, record_cleanup_debt

    record_cleanup_debt(root, "fts", {RETIRED})
    repo_checks._run_repo_checks(root, repair=True)
    assert not any(load_cleanup_debt(root).values())
