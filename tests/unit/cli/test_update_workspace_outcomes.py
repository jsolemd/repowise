"""Workspace failures reach Click/JSON after independent members finish."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from repowise.cli import source_search_runtime
from repowise.cli.commands import workspace_cmd
from repowise.cli.commands.update_cmd import command, workspace
from repowise.cli.main import cli
from repowise.core import workspace as core_workspace
from repowise.core.source_search.lifecycle import SourceLifecycleResult
from repowise.core.workspace import update as core_update
from repowise.core.workspace.config import RepoEntry, WorkspaceConfig


@pytest.fixture
def invoke_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep native CLI routing/reporting, replacing each repository's work."""
    real_run_update = command.run_update

    def invoke(
        members: dict[str, tuple[str, str]],
        *,
        progress: str = "json",
        current: bool = False,
        dry_run: bool = False,
        source_outcome: object = None,
        error_message: str = "primary persist failed",
        include_working_tree: bool = False,
        checkpoint: dict | None = None,
    ):
        attempted: list[str] = []
        succeeded: list[str] = []
        hooks: list[list[str]] = []
        editor_refreshes: list[bool] = []
        for alias, (mode, _outcome) in members.items():
            state_dir = tmp_path / alias / ".repowise"
            state_dir.mkdir(parents=True)
            state = {
                "last_sync_commit": "old",
                "docs_mode": "deterministic" if mode == "docs" else "none",
            }
            if current:
                from repowise.core.repo_config import config_fingerprint

                state["config_fingerprint"] = config_fingerprint(state_dir.parent)
            if checkpoint is not None:
                state["working_tree_checkpoint"] = checkpoint
            (state_dir / "state.json").write_text(json.dumps(state))
        WorkspaceConfig(
            repos=[RepoEntry(alias=alias, path=alias) for alias in members],
        ).save(tmp_path)
        monkeypatch.setattr(
            core_workspace, "check_repo_staleness", lambda *_: (not current, None, 1)
        )
        monkeypatch.setattr(
            source_search_runtime,
            "configured_source_pending_updates",
            AsyncMock(return_value=False),
        )
        monkeypatch.setattr(
            source_search_runtime, "configured_source_recipe_changes", AsyncMock(return_value=())
        )

        async def update_core(_root, _config, *, only_aliases=None, on_repo_done, **_kwargs):
            results = []
            for alias, (mode, outcome) in members.items():
                if mode != "core" or (only_aliases is not None and alias not in only_aliases):
                    continue
                attempted.append(alias)
                result = core_workspace.RepoUpdateResult(
                    alias=alias,
                    updated=outcome == "updated",
                    error=error_message if outcome == "failed" else None,
                    skipped_reason={"deferred": "in_flight", "noop": "up_to_date"}.get(outcome),
                )
                if result.updated:
                    succeeded.append(alias)
                results.append(result)
                on_repo_done(result)
            return results

        def update_docs(**kwargs):
            if not kwargs["no_workspace"]:
                return real_run_update(**kwargs)
            alias = Path(kwargs["path"]).name
            attempted.append(alias)
            outcome = members[alias][1]
            if outcome == "failed":
                raise RuntimeError(error_message)
            if outcome == "updated":
                succeeded.append(alias)
            return {
                "updated": command.UpdateOutcome.REGENERATED,
                "deferred": command.UpdateOutcome.DEFERRED,
                "noop": command.UpdateOutcome.NOOP,
            }[outcome]

        async def reconcile(paths):
            return dict.fromkeys(paths, source_outcome)

        async def cross_repo(_config, _root, aliases):
            hooks.append(aliases)

        monkeypatch.setattr(core_workspace, "update_workspace", update_core)
        monkeypatch.setattr(command, "run_update", update_docs)
        monkeypatch.setattr(source_search_runtime, "reconcile_configured_source_indexes", reconcile)
        monkeypatch.setattr(workspace, "_remove_tombstoned_page_vectors", lambda *_: None)
        monkeypatch.setattr(
            workspace,
            "_refresh_workspace_editor_project_files",
            lambda **_: editor_refreshes.append(True),
        )
        monkeypatch.setattr(workspace_cmd, "inherit_workspace_distill_verdict", lambda *_: None)
        monkeypatch.setattr(core_update, "sync_workspace_state_from_disk", lambda *_: None)
        monkeypatch.setattr(core_update, "run_cross_repo_hooks", cross_repo)
        args = ["update", str(tmp_path), "--workspace", "--no-docs", "--progress", progress]
        if dry_run:
            args.append("--dry-run")
        if include_working_tree:
            args.append("--include-working-tree")
        try:
            runner = CliRunner(mix_stderr=False)
        except TypeError:  # Click >= 8.2 always separates the streams.
            runner = CliRunner()
        result = runner.invoke(cli, args)
        events = (
            [json.loads(line) for line in result.stdout.splitlines()] if progress == "json" else []
        )
        return result, events, attempted, succeeded, hooks, editor_refreshes

    return invoke


def test_completed_dirty_snapshot_skips_workspace_docs(invoke_workspace, monkeypatch, tmp_path):
    from repowise.core import working_tree_state

    checkpoint = {"head": "old", "files": {"app.py": ["saved-hash", 1, 2, 3]}}
    monkeypatch.setattr(working_tree_state, "capture_working_tree", lambda _: checkpoint)
    result, _, attempted, _, hooks, _ = invoke_workspace(
        {"app": ("docs", "updated")},
        current=True,
        include_working_tree=True,
        checkpoint=checkpoint,
    )
    assert result.exit_code == 0, result.output
    assert attempted == []
    assert hooks == []


@pytest.mark.parametrize("outcome", ["updated", "deferred", "failed"])
def test_only_completed_docs_work_checkpoints_dirty_files(
    invoke_workspace, monkeypatch, tmp_path, outcome
):
    from repowise.core import working_tree_state

    checkpoint = {"head": "old", "files": {"app.py": ["saved-hash", 1, 2, 3]}}
    monkeypatch.setattr(working_tree_state, "capture_working_tree", lambda _: checkpoint)
    result, _, attempted, _, _, _ = invoke_workspace(
        {"app": ("docs", outcome)}, include_working_tree=True
    )
    assert result.exit_code == (1 if outcome == "failed" else 0), result.output
    assert attempted == ["app"]
    state = json.loads((tmp_path / "app/.repowise/state.json").read_text())
    assert state.get("working_tree_checkpoint") == (checkpoint if outcome == "updated" else None)


@pytest.mark.parametrize("progress", ["rich", "json"])
@pytest.mark.parametrize(
    "members",
    [
        {"bad": ("core", "failed")},
        {"bad": ("docs", "failed")},
        {"bad": ("core", "failed"), "good": ("core", "updated")},
        {"bad": ("docs", "failed"), "good": ("docs", "updated")},
        {"bad": ("core", "failed"), "good": ("docs", "updated")},
        {"bad": ("docs", "failed"), "good": ("core", "updated"), "later": ("docs", "updated")},
        {"bad": ("core", "failed"), "also_bad": ("docs", "failed"), "busy": ("docs", "deferred")},
    ],
)
def test_primary_failures_exit_nonzero_after_siblings_finish(invoke_workspace, members, progress):
    result, events, attempted, succeeded, hooks, refreshed = invoke_workspace(
        members, progress=progress
    )

    assert result.exit_code == 1, result.output
    assert set(attempted) == set(members)
    assert set(succeeded) == {
        alias for alias, (_, outcome) in members.items() if outcome == "updated"
    }
    assert refreshed == [True]
    output = (result.stdout + result.stderr).lower()
    assert "workspace update failed" in output
    assert "workspace update complete" not in output
    if any(mode == "docs" for mode, _ in members.values()) and succeeded:
        assert len(hooks) == 1
        assert set(hooks[0]) == set(succeeded)
    if "also_bad" in members:
        assert "2 failed" in output
    if progress == "json":
        assert events[-1]["event"] == "error"
        assert "primary persist failed" in events[-1]["message"]
        assert not any(event["event"] == "done" for event in events)
        for alias, (_, outcome) in members.items():
            if outcome == "failed":
                assert alias in events[-1]["message"]


@pytest.mark.parametrize("mode", ["core", "docs"])
@pytest.mark.parametrize(
    ("repo_outcome", "expected"),
    [("updated", "regenerated"), ("deferred", "deferred"), ("noop", "noop")],
)
def test_nonfailed_workspace_reports_actual_outcome(invoke_workspace, mode, repo_outcome, expected):
    result, events, *_ = invoke_workspace({"repo": (mode, repo_outcome)})
    assert result.exit_code == 0, result.output
    assert events[-1]["event"] == "done"
    assert events[-1]["ok"] is True
    assert events[-1]["outcome"] == expected
    if repo_outcome == "deferred":
        assert "workspace update complete" not in result.stderr.lower()


@pytest.mark.parametrize(
    ("current", "dry_run", "expected"), [(True, False, "noop"), (False, True, "dry_run")]
)
def test_workspace_without_updates_reports_no_regeneration(
    invoke_workspace, current, dry_run, expected
):
    result, events, attempted, *_ = invoke_workspace(
        {"repo": ("core", "updated")},
        current=current,
        dry_run=dry_run,
    )
    assert result.exit_code == 0, result.output
    assert not attempted
    assert events[-1]["ok"] is True
    assert events[-1]["outcome"] == expected


@pytest.mark.parametrize("docs", [False, True])
@pytest.mark.parametrize(
    "source_outcome",
    [RuntimeError("durable source retry"), "busy", "degraded"],
)
def test_durable_source_degradation_is_not_primary_failure(invoke_workspace, docs, source_outcome):
    if isinstance(source_outcome, str):
        source_outcome = SourceLifecycleResult(
            status=source_outcome,
            generation_id=None,
            generation_sequence=None,
            updates_consumed=0,
            embedded=0,
            reused=0,
            chunks=0,
            stale_files=0,
            total_seconds=0.0,
        )
    members = {"core": ("core", "updated")}
    if docs:
        members["docs"] = ("docs", "noop")
    result, events, attempted, succeeded, *_ = invoke_workspace(
        members, source_outcome=source_outcome
    )
    assert result.exit_code == 0, result.output
    assert set(attempted) == set(members)
    assert succeeded == ["core"]
    assert events[-1]["ok"] is True
    assert events[-1]["outcome"] == "regenerated"
    if isinstance(source_outcome, Exception):
        assert "source search reconcile deferred" in result.stderr


@pytest.mark.parametrize("docs", [False, True])
def test_empty_exception_message_still_fails_workspace(invoke_workspace, docs):
    members = {"bad": ("core", "failed")}
    if docs:
        members["good"] = ("docs", "updated")
    result, events, attempted, *_ = invoke_workspace(members, error_message="")
    assert result.exit_code == 1, result.output
    assert set(attempted) == set(members)
    assert "✗ bad:" in result.stderr
    assert events[-1]["event"] == "error"
    assert "bad:" in events[-1]["message"]
    assert not any(event["event"] == "done" for event in events)
