"""A completed template render must win over the same run's decay plan."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy import select

from repowise.cli.commands.update_cmd import deterministic
from repowise.cli.main import cli
from repowise.core.generation.models import STUB_FALLBACK_ERROR, GeneratedPage
from repowise.core.persistence import create_engine, create_session_factory, get_session
from repowise.core.persistence.models import Page
from tests.unit.cli.test_update_reconciles_stale_pages import (
    _db_url,
    _freshness,
    _repo_at_head,
    _seed,
    _state_at,
)


def _page(path: str, *, page_type: str = "file_page", **overrides) -> GeneratedPage:
    now = datetime.now(UTC).isoformat()
    return GeneratedPage(
        page_id=f"{page_type}:{path}",
        page_type=page_type,
        title=path,
        content=f"# Current {path}\n",
        source_hash="current",
        model_name="template",
        provider_name="template",
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        generation_level=0,
        target_path=path,
        created_at=now,
        updated_at=now,
        **overrides,
    )


@pytest.fixture(autouse=True)
def _isolated_page_update(monkeypatch):
    # These tests exercise page persistence, not the independent source outbox.
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "0")
    monkeypatch.setenv("REPOWISE_TOOLS_NO_GENERATIVE", "1")
    monkeypatch.setenv("REPOWISE_SKIP_EDITOR_SETUP", "1")
    monkeypatch.setattr(deterministic, "deterministic_embedder_name", lambda _cfg: "mock")
    for key in ("REPOWISE_DB_URL", "REPOWISE_DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)


def test_successful_page_stays_fresh_while_unrendered_pages_decay(tmp_path: Path) -> None:
    repo, _ = _repo_at_head(tmp_path)
    before = {
        "foo.py": "stale",
        "omitted.py": "stale",
        "budget.py": "fresh",
        "spotlight_only.py": "fresh",
        "failed.py": "fresh",
        "expired.py": "expired",
        "tombstone.py": "tombstone",
        "untouched.py": "fresh",
    }
    asyncio.run(_seed(repo, before))
    pages = [
        _page("foo.py"),
        _page("spotlight_only.py::fn", page_type="symbol_spotlight"),
        _page("failed.py", metadata={STUB_FALLBACK_ERROR: "render failed"}),
        _page("expired.py", freshness_status="expired"),
    ]
    degraded: list[str] = []

    deterministic.persist_deterministic_pages(
        repo_path=repo,
        generated_pages=pages,
        decay_paths=[path for path in before if path != "untouched.py"],
        degraded=degraded,
    )

    assert not degraded
    statuses = asyncio.run(_freshness(repo))
    assert statuses == {
        **before,
        "foo.py": "fresh",
        "budget.py": "stale",
        "spotlight_only.py": "stale",
        "spotlight_only.py::fn": "fresh",
        "failed.py": "stale",
    }


def test_failed_page_transaction_does_not_attest_a_successful_render(
    tmp_path: Path, monkeypatch
) -> None:
    from repowise.core import persistence

    repo, _ = _repo_at_head(tmp_path)
    asyncio.run(_seed(repo, {"foo.py": "stale", "budget.py": "fresh"}))
    real_upsert = persistence.upsert_pages_from_generated

    async def fail_after_upsert(*args, **kwargs):
        await real_upsert(*args, **kwargs)
        raise RuntimeError("interrupted page transaction")

    monkeypatch.setattr(persistence, "upsert_pages_from_generated", fail_after_upsert)
    with pytest.raises(RuntimeError, match="interrupted page transaction"):
        deterministic.persist_deterministic_pages(
            repo_path=repo,
            generated_pages=[_page("foo.py")],
            decay_paths=["foo.py", "budget.py"],
            degraded=[],
        )

    assert asyncio.run(_freshness(repo)) == {"foo.py": "stale", "budget.py": "fresh"}


@pytest.mark.parametrize("render_result", ["success", "empty", "failure"])
def test_dirty_update_persists_decay_after_stale_page_reconciliation(
    tmp_path: Path, monkeypatch, render_result: str
) -> None:
    repo, head = _repo_at_head(tmp_path)
    before = {"foo.py": "fresh" if render_result == "empty" else "stale"}
    if render_result == "failure":
        before["budget.py"] = "fresh"
        (repo / "budget.py").write_text("def budget():\n    return 1\n", encoding="utf-8")
    asyncio.run(_seed(repo, before))
    _state_at(repo, head)
    (repo / "foo.py").write_text("def foo():\n    return 2\n", encoding="utf-8")
    if render_result == "failure":

        def failed_render(**kwargs):
            kwargs["degraded"].append("Template generation: injected failure")
            return []

        monkeypatch.setattr(deterministic, "regenerate_deterministic_pages", failed_render)
    elif render_result == "empty":
        monkeypatch.setattr(deterministic, "regenerate_deterministic_pages", lambda **_: [])

    result = CliRunner().invoke(
        cli,
        [
            "update",
            str(repo),
            "--no-workspace",
            "--index-only",
            "--include-working-tree",
            "--cascade-budget",
            "0",
            "--no-agents",
        ],
    )

    assert result.exit_code == 0, result.output
    expected = "fresh" if render_result == "success" else "stale"
    assert asyncio.run(_freshness(repo))["foo.py"] == expected
    if render_result == "failure":
        assert asyncio.run(_freshness(repo))["budget.py"] == "stale"
    if render_result == "success":
        assert "Reconciling stale structural pages: 1" in result.output
        assert "Re-rendered" in result.output

        async def content() -> str:
            engine = create_engine(_db_url(repo))
            try:
                async with get_session(create_session_factory(engine)) as session:
                    return (
                        await session.execute(
                            select(Page.content).where(Page.id == "file_page:foo.py")
                        )
                    ).scalar_one()
            finally:
                await engine.dispose()

        assert asyncio.run(content()) != "# foo.py\n"
