"""Saved edits invalidate documentation once; stale recovery must converge."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from repowise.cli.commands.update_cmd import deterministic, working_tree
from repowise.cli.helpers import load_state, save_state
from repowise.cli.main import cli
from repowise.core.ingestion.change_detector import ChangeDetector
from repowise.core.update_lock import release_update_lock
from repowise.core.working_tree_state import capture_working_tree
from tests.unit.cli.test_update_reconciles_stale_pages import (
    _freshness,
    _git,
    _repo_at_head,
    _seed,
    _state_at,
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "0")
    monkeypatch.setenv("REPOWISE_SKIP_EDITOR_SETUP", "1")
    monkeypatch.setenv("REPOWISE_TOOLS_NO_GENERATIVE", "1")
    monkeypatch.setattr(deterministic, "deterministic_embedder_name", lambda _cfg: "mock")
    for key in ("REPOWISE_DB_URL", "REPOWISE_DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)


def update(repo: Path, *, success: bool = True) -> str:
    try:
        result = CliRunner().invoke(
            cli,
            [
                "update",
                str(repo),
                "--no-workspace",
                "--index-only",
                "--include-working-tree",
                "--cascade-budget",
                "2",
                "--no-agents",
            ],
        )
        if success:
            assert result.exit_code == 0, result.output
        else:
            assert result.exit_code != 0, result.output
        return result.output
    finally:
        # CliRunner does not exit a process, so model the CLI's atexit cleanup.
        release_update_lock(repo)


def checkpoint(repo: Path) -> dict:
    state = {}
    assert working_tree.checkpoint_documentation(repo, state, capture_working_tree(repo))
    return state


def test_unchanged_dirty_updates_drain_stale_pages_and_then_idle(tmp_path):
    repo, _ = _repo_at_head(tmp_path)
    for n in range(6):
        (repo / f"use_{n}.py").write_text(
            f"from foo import foo\ndef use_{n}():\n    return foo()\n"
        )
    _git(repo, "add", "*.py")
    _git(repo, "commit", "-m", "importers")
    head = _git(repo, "rev-parse", "HEAD")
    paths = ["foo.py", *[f"use_{n}.py" for n in range(6)]]
    asyncio.run(_seed(repo, dict.fromkeys(paths, "stale")))
    _state_at(repo, head)
    update(repo)  # Establish current renderer keys before testing a real edit.
    assert all(asyncio.run(_freshness(repo))[p] == "fresh" for p in paths)

    (repo / "foo.py").write_text("def foo():\n    return 2\n")
    update(repo)
    first = asyncio.run(_freshness(repo))
    assert any(first[p] == "stale" for p in paths)
    assert working_tree.DOCS_CHECKPOINT in load_state(repo)

    output = update(repo)
    assert "Reconciling stale structural pages:" in output
    assert all(asyncio.run(_freshness(repo))[p] == "fresh" for p in paths)
    assert "Already up to date" in update(repo)

    (repo / "foo.py").write_text("def foo():\n    return 3\n")
    update(repo)
    assert any(asyncio.run(_freshness(repo))[p] == "stale" for p in paths)
    update(repo)
    assert all(asyncio.run(_freshness(repo))[p] == "fresh" for p in paths)


@pytest.mark.parametrize("stage", [False, True])
def test_filters_processed_content_but_keeps_another_edit(tmp_path, stage):
    repo, _ = _repo_at_head(tmp_path)
    (repo / "foo.py").write_text("def foo():\n    return 2\n")
    if stage:
        _git(repo, "add", "foo.py")
    state = checkpoint(repo)
    detector = ChangeDetector(repo)
    assert not working_tree.filter_documentation_diffs(
        detector.get_working_tree_changes(), state, capture_working_tree(repo)
    )
    (repo / "foo.py").write_text("def foo():\n    return 3\n")
    changes = working_tree.filter_documentation_diffs(
        detector.get_working_tree_changes(), state, capture_working_tree(repo)
    )
    assert [d.path for d in changes] == ["foo.py"]


@pytest.mark.parametrize("untracked", [False, True])
def test_revert_or_untracked_deletion_survives_watcher_cleanup(tmp_path, untracked):
    repo, head = _repo_at_head(tmp_path)
    path = "new.py" if untracked else "foo.py"
    (repo / path).write_text("def revised():\n    return 2\n")
    state = checkpoint(repo)
    state.update(last_sync_commit=head, last_docs_commit=head, docs_mode="deterministic")
    asyncio.run(_seed(repo, {path: "stale"}))
    save_state(repo, state)
    update(repo)
    if untracked:
        (repo / path).unlink()
    else:
        _git(repo, "restore", path)
    # The watcher has already consumed the revert; its path ledger is empty.
    state = load_state(repo)
    state["working_tree_paths"] = []
    state["working_tree_checkpoint"] = capture_working_tree(repo)
    save_state(repo, state)
    assert working_tree.documentation_update_reason(repo, state)
    update(repo)
    assert asyncio.run(_freshness(repo))[path] == ("tombstone" if untracked else "fresh")
    assert working_tree.documentation_update_reason(repo, load_state(repo)) is None


def test_watcher_checkpoint_cannot_acknowledge_documentation(tmp_path):
    repo, _ = _repo_at_head(tmp_path)
    (repo / "foo.py").write_text("def foo():\n    return 2\n")
    state = {"working_tree_checkpoint": capture_working_tree(repo)}
    assert working_tree.documentation_update_reason(repo, state)
    assert working_tree.filter_documentation_diffs(
        ChangeDetector(repo).get_working_tree_changes(), state, capture_working_tree(repo)
    )


def test_concurrent_save_retains_previous_documentation_checkpoint(tmp_path):
    repo, _ = _repo_at_head(tmp_path)
    state = checkpoint(repo)
    previous = json.dumps(state, sort_keys=True)
    (repo / "foo.py").write_text("def foo():\n    return 2\n")
    before = capture_working_tree(repo)
    (repo / "foo.py").write_text("def foo():\n    return 3\n")
    assert not working_tree.checkpoint_documentation(repo, state, before)
    assert json.dumps(state, sort_keys=True) == previous
    assert working_tree.documentation_update_reason(repo, state)


def test_failed_render_does_not_acknowledge_dirty_content(tmp_path, monkeypatch):
    repo, head = _repo_at_head(tmp_path)
    asyncio.run(_seed(repo, {"foo.py": "stale"}))
    _state_at(repo, head)
    update(repo)
    before = load_state(repo)[working_tree.DOCS_CHECKPOINT]
    (repo / "foo.py").write_text("def foo():\n    return 2\n")
    real_render = deterministic.regenerate_deterministic_pages

    def failed_render(**kwargs):
        kwargs["degraded"].append("Template page refresh: injected failure")
        return []

    monkeypatch.setattr(deterministic, "regenerate_deterministic_pages", failed_render)
    update(repo)
    assert load_state(repo)[working_tree.DOCS_CHECKPOINT] == before
    monkeypatch.setattr(deterministic, "regenerate_deterministic_pages", real_render)
    update(repo)
    assert load_state(repo)[working_tree.DOCS_CHECKPOINT] != before


@pytest.mark.parametrize("failed_first", [False, True])
def test_committed_change_is_not_hidden_by_source_watcher(tmp_path, monkeypatch, failed_first):
    repo, head = _repo_at_head(tmp_path)
    asyncio.run(_seed(repo, {"foo.py": "stale"}))
    _state_at(repo, head)
    update(repo)
    (repo / "bar.py").write_text("def bar():\n    return 2\n")
    _git(repo, "add", "bar.py")
    _git(repo, "commit", "-m", "new source")
    state = load_state(repo)
    state["last_sync_commit"] = _git(repo, "rev-parse", "HEAD")
    state["working_tree_checkpoint"] = capture_working_tree(repo)
    save_state(repo, state)
    assert working_tree.documentation_update_reason(repo, state)
    if failed_first:
        real_render = deterministic.regenerate_deterministic_pages

        def failed_render(**kwargs):
            kwargs["degraded"].append("Template page refresh: injected failure")
            return []

        monkeypatch.setattr(deterministic, "regenerate_deterministic_pages", failed_render)
        update(repo)
        assert load_state(repo)[working_tree.DOCS_CHECKPOINT]["head"] == head
        monkeypatch.setattr(deterministic, "regenerate_deterministic_pages", real_render)
    update(repo)
    assert asyncio.run(_freshness(repo))["bar.py"] == "fresh"
    assert working_tree.documentation_update_reason(repo, load_state(repo)) is None
