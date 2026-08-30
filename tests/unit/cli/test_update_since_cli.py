"""The CLI is loud when a person supplies an unusable ``--since`` ref."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from repowise.cli.main import cli

EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-qm", "seed")
    return repo


@pytest.mark.parametrize("ref", ["not-a-real-ref", EMPTY_TREE])
def test_unresolvable_since_exits_nonzero_with_the_working_recipe(tmp_path, ref):
    repo = _repo(tmp_path)

    result = CliRunner().invoke(
        cli,
        ["update", str(repo), "--no-workspace", "--index-only", "--since", ref],
    )

    assert result.exit_code != 0
    assert "does not resolve to a commit" in result.output
    assert f"git commit-tree {EMPTY_TREE} -m anchor" in result.output
    assert "--since <that sha>" in result.output


def test_commit_since_resolves_before_the_update_pipeline(tmp_path):
    repo = _repo(tmp_path)

    result = CliRunner().invoke(
        cli,
        ["update", str(repo), "--no-workspace", "--index-only", "--since", "HEAD"],
    )

    assert result.exit_code == 0, result.output
    assert "Already up to date" in result.output
