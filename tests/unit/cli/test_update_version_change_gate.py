"""An analyzer or parser change must get past the "already up to date" return.

With no new commit, ``head == base_ref`` is the normal state of a checkout, so
the version stamps are the only thing that can tell ``update`` that every stored
score or graph edge was written by an older build.
"""

from __future__ import annotations

import pytest

from repowise.cli.commands.update_cmd import command as upd_cmd
from repowise.cli.helpers import load_state, save_state
from repowise.core.analysis.health import HEALTH_ANALYZER_VERSION

from .test_update_up_to_date_lock import _indexed_repo, _invoke_update

UP_TO_DATE = "Already up to date"


@pytest.fixture(autouse=True)
def _independent_source_lane(monkeypatch):
    # These upstream cases exercise graph/analyzer drift with no source work.
    # The pending-source composition case below enables that lane explicitly.
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "0")


def test_current_versions_stay_up_to_date(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _ = _indexed_repo(tmp_path)
    state = load_state(repo)
    save_state(repo, {**state, "health_analyzer_version": HEALTH_ANALYZER_VERSION})
    rescored: list[int] = []
    monkeypatch.setattr(upd_cmd, "parser_changed", lambda _p: False)
    monkeypatch.setattr(
        "repowise.cli.commands.update_cmd.persistence.run_decay_health_rescore",
        lambda *a, **k: rescored.append(1) or True,
    )

    out = _invoke_update(repo)

    assert UP_TO_DATE in out
    assert not rescored


def test_stale_analyzer_version_rescores_an_unchanged_checkout(tmp_path) -> None:
    repo, c2 = _indexed_repo(tmp_path)
    save_state(repo, {**load_state(repo), "health_analyzer_version": HEALTH_ANALYZER_VERSION - 1})

    out = _invoke_update(repo)

    assert UP_TO_DATE not in out
    assert "Health analyzer changed" in out
    state = load_state(repo)
    assert state["health_analyzer_version"] == HEALTH_ANALYZER_VERSION
    assert state["last_sync_commit"] == c2
    # Stamped, so the next run is quiet again.
    assert UP_TO_DATE in _invoke_update(repo)


def test_parser_change_reparses_an_unchanged_checkout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _ = _indexed_repo(tmp_path)
    save_state(repo, {**load_state(repo), "health_analyzer_version": HEALTH_ANALYZER_VERSION})
    monkeypatch.setattr(upd_cmd, "parser_changed", lambda _p: True)
    rescored: list[int] = []
    monkeypatch.setattr(
        "repowise.cli.commands.update_cmd.persistence.run_decay_health_rescore",
        lambda *a, **k: rescored.append(1) or True,
    )

    out = _invoke_update(repo)

    assert UP_TO_DATE not in out
    assert "Parser changed" in out
    assert rescored


@pytest.mark.parametrize("source_work", ["pending", "recipe"])
@pytest.mark.parametrize("graph_stale", [False, True])
def test_source_work_does_not_hide_graph_parser_drift(
    tmp_path, monkeypatch, source_work, graph_stale
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from sqlalchemy import update

    from repowise.cli import source_search_runtime
    from repowise.core.ingestion.parse_cache import parser_fingerprint
    from repowise.core.persistence import (
        create_engine,
        create_session_factory,
        get_session,
    )
    from repowise.core.persistence.models import Repository
    from repowise.core.persistence.parser_state import symbol_parser_refresh_required
    from repowise.core.pipeline.full_index import index_repo_full

    from .test_update_up_to_date_lock import _git, _repo_with_three_commits

    repo, *_ = _repo_with_three_commits(tmp_path)
    (repo / "a.py").write_text(
        "def caller():\n    return callee()\n\ndef callee():\n    return 1\n"
    )
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-m", "source graph")
    head = _git(repo, "rev-parse", "HEAD")
    asyncio.run(index_repo_full(repo))
    save_state(
        repo,
        {
            "last_sync_commit": head,
            "last_docs_commit": head,
            "docs_enabled": False,
            "health_analyzer_version": HEALTH_ANALYZER_VERSION,
        },
    )

    async def stale_graph_only():
        engine = create_engine(f"sqlite+aiosqlite:///{repo / '.repowise' / 'wiki.db'}")
        try:
            async with get_session(create_session_factory(engine)) as session:
                await session.execute(
                    update(Repository).values(
                        graph_edges_parser_fingerprint=(
                            "old-graph-parser" if graph_stale else parser_fingerprint()
                        ),
                        symbols_parser_fingerprint=parser_fingerprint(),
                    )
                )
        finally:
            await engine.dispose()

    asyncio.run(stale_graph_only())
    assert upd_cmd.parser_changed(repo) is graph_stale
    assert not asyncio.run(symbol_parser_refresh_required(repo))
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")
    monkeypatch.setattr(
        source_search_runtime,
        "configured_source_pending_updates",
        AsyncMock(return_value=source_work == "pending"),
    )
    monkeypatch.setattr(
        source_search_runtime,
        "configured_source_recipe_changes",
        AsyncMock(return_value=("recipe drift",) if source_work == "recipe" else ()),
    )
    monkeypatch.setattr(
        source_search_runtime,
        "reconcile_configured_source_index",
        AsyncMock(return_value=SimpleNamespace(status="degraded", error="injected source failure")),
    )
    rescored = []
    monkeypatch.setattr(
        "repowise.cli.commands.update_cmd.persistence.run_decay_health_rescore",
        lambda *a, **k: rescored.append(1) or True,
    )

    out = _invoke_update(repo)

    assert ("Parser changed" in out) is graph_stale
    assert rescored == ([1] if graph_stale else [])
    assert not upd_cmd.parser_changed(repo)
