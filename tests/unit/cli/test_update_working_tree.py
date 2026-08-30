"""CLI coverage for opt-in working-tree updates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from repowise.cli.commands.update_cmd import command as update_cmd
from repowise.cli.commands.update_cmd import workspace as workspace_cmd
from repowise.cli.helpers import CommandTarget
from repowise.cli.main import cli
from repowise.core.workspace.config import RepoEntry, WorkspaceConfig


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _workspace_target(
    tmp_path: Path,
    *,
    working_tree_paths: list[str] | None = None,
) -> tuple[CommandTarget, Path]:
    ws_root = tmp_path / "workspace"
    repo = ws_root / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@test.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    head = _git(repo, "rev-parse", "HEAD")

    state_dir = repo / ".repowise"
    state_dir.mkdir()
    state = {
        "last_sync_commit": head,
        "docs_mode": "none",
        "working_tree_paths": working_tree_paths or [],
    }
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    config = WorkspaceConfig(
        repos=[
            RepoEntry(
                path="repo",
                alias="repo",
                last_commit_at_index=head,
            )
        ],
        default_repo="repo",
    )
    target = CommandTarget(
        mode="workspace",
        ws_root=ws_root,
        ws_config=config,
    )
    return target, repo


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], False), (["--include-working-tree"], True)],
)
def test_update_cli_forwards_include_working_tree(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected: bool,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(update_cmd, "run_update", lambda **kwargs: captured.update(kwargs))

    result = CliRunner().invoke(cli, ["update", *extra_args])

    assert result.exit_code == 0, result.output
    assert captured["include_working_tree"] is expected


def test_update_cli_forwards_one_run_mass_deletion_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(update_cmd, "run_update", lambda **kwargs: captured.update(kwargs))

    result = CliRunner().invoke(cli, ["update", "--accept-mass-deletion"])

    assert result.exit_code == 0, result.output
    assert captured["accept_mass_deletion"] is True


@pytest.mark.parametrize(
    (
        "include_working_tree",
        "dirty",
        "working_tree_paths",
        "expected_reason",
        "would_update",
    ),
    [
        (True, True, None, "uncommitted changes", True),
        (True, False, ["app.py"], "working-tree cleanup", True),
        (True, False, None, None, False),
        (False, True, None, None, False),
    ],
)
def test_workspace_working_tree_staleness(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    *,
    include_working_tree: bool,
    dirty: bool,
    working_tree_paths: list[str] | None,
    expected_reason: str | None,
    would_update: bool,
) -> None:
    target, repo = _workspace_target(
        tmp_path,
        working_tree_paths=working_tree_paths,
    )
    if dirty:
        (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

    workspace_cmd._workspace_update(
        target,
        dry_run=True,
        index_only=True,
        include_working_tree=include_working_tree,
    )

    output = capsys.readouterr().out
    if expected_reason is not None:
        assert expected_reason in output
    else:
        assert "uncommitted changes" not in output
        assert "working-tree cleanup" not in output
    if would_update:
        assert "1 repo(s) would be updated" in output
    else:
        assert "All repos are up to date" in output


def test_workspace_dispatch_forwards_include_working_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import repowise.core.workspace as core_workspace
    from repowise.cli import source_search_runtime
    from repowise.core.workspace import RepoUpdateResult

    target, repo = _workspace_target(tmp_path)
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    captured: dict[str, object] = {}

    async def fake_update_workspace(*_args: object, **kwargs: object):
        captured.update(kwargs)
        return [RepoUpdateResult(alias="repo", updated=False, skipped_reason="up_to_date")]

    async def fake_reconcile(_paths: list[Path]) -> dict[Path, object]:
        return {}

    monkeypatch.setattr(core_workspace, "update_workspace", fake_update_workspace)
    monkeypatch.setattr(
        source_search_runtime,
        "reconcile_configured_source_indexes",
        fake_reconcile,
    )
    monkeypatch.setattr(
        workspace_cmd,
        "_refresh_workspace_editor_project_files",
        lambda **_kwargs: None,
    )

    workspace_cmd._workspace_update(
        target,
        index_only=True,
        include_working_tree=True,
    )

    assert captured["include_working_tree"] is True


def test_workspace_acceptance_reopens_and_forwards_a_persisted_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import repowise.core.workspace as core_workspace
    from repowise.cli import source_search_runtime
    from repowise.core.workspace import RepoUpdateResult

    target, repo = _workspace_target(tmp_path)
    state_path = repo / ".repowise" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["prune_refusals"] = {
        "from_commit": state["last_sync_commit"],
        "findings": [
            {
                "table": "graph_nodes",
                "candidate_paths": 30,
                "persisted_paths": 40,
            }
        ],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    captured: dict[str, object] = {}

    async def fake_update_workspace(*_args: object, **kwargs: object):
        captured.update(kwargs)
        return [RepoUpdateResult(alias="repo", updated=False, skipped_reason="up_to_date")]

    async def fake_reconcile(_paths: list[Path]) -> dict[Path, object]:
        return {}

    monkeypatch.setattr(core_workspace, "update_workspace", fake_update_workspace)
    monkeypatch.setattr(
        source_search_runtime,
        "reconcile_configured_source_indexes",
        fake_reconcile,
    )
    monkeypatch.setattr(
        workspace_cmd,
        "_refresh_workspace_editor_project_files",
        lambda **_kwargs: None,
    )

    workspace_cmd._workspace_update(
        target,
        index_only=True,
        accept_mass_deletion=True,
    )

    assert "accepted mass-deletion repair" in capsys.readouterr().out
    assert captured["accept_mass_deletion"] is True
