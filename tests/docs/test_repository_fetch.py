"""fetch_latest must follow a branch change on a single-branch clone."""

from __future__ import annotations

import pytest

from repowise.docs.library import repository


@pytest.mark.asyncio
async def test_fetch_latest_uses_explicit_refspec_for_configured_branch(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    calls: list[tuple[str, ...]] = []

    async def fake_run_git(*args, cwd=None):
        calls.append(tuple(args))
        return ""

    monkeypatch.setattr(repository, "run_git", fake_run_git)

    await repository.fetch_latest(tmp_path, "master")

    assert calls[0] == (
        "fetch",
        "--depth",
        "1",
        "origin",
        "+refs/heads/master:refs/remotes/origin/master",
    )
    assert calls[1] == ("reset", "--hard", "origin/master")
