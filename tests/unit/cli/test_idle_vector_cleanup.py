"""An idle update must finish failed index deletion without reindexing."""

from __future__ import annotations

import asyncio
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner

from repowise.cli.commands.update_cmd import incremental, workspace
from repowise.cli.helpers import CommandTarget
from repowise.cli.main import cli
from repowise.core.pipeline.cleanup_debt import load_cleanup_debt, record_cleanup_debt
from repowise.core.repo_config import config_fingerprint
from repowise.core.update_lock import (
    read_update_lock,
    release_update_lock,
    try_acquire_update_lock,
)
from repowise.core.workspace.config import RepoEntry, WorkspaceConfig


@pytest.fixture(params=["single", "workspace"])
def idle_update(request, tmp_path):
    from repowise.core.pipeline.full_index import index_repo_full

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("config", "user.email", "test@test.com")
    git("config", "user.name", "Test")
    (repo / "app.py").write_text("VALUE = 1\n")
    git("add", "app.py")
    git("commit", "-m", "initial")
    head = git("rev-parse", "HEAD")
    (repo / ".repowise").mkdir()
    asyncio.run(index_repo_full(repo))
    (repo / ".repowise/state.json").write_text(
        json.dumps(
            {
                "last_sync_commit": head,
                "docs_mode": "none",
                "config_fingerprint": config_fingerprint(repo),
            }
        )
    )
    config = WorkspaceConfig(
        repos=[RepoEntry(path="repo", alias="repo", last_commit_at_index=head)]
    )

    def run(*, dry_run=False):
        if request.param == "single":
            result = CliRunner().invoke(
                cli,
                ["update", str(repo), "--no-workspace", "--index-only"]
                + (["--dry-run"] if dry_run else []),
            )
            assert result.exit_code == 0, result.output
            assert "Already up to date" in result.output
            return result.output
        else:
            workspace._workspace_update(
                CommandTarget(mode="workspace", ws_root=tmp_path, ws_config=config),
                index_only=True,
                dry_run=dry_run,
            )

    return repo, head, run


def test_idle_update_retries_cleanup_and_releases_its_store(idle_update, monkeypatch):
    repo, _head, run = idle_update
    page_id = "file_page:removed.py"
    record_cleanup_debt(repo, "vectors", {page_id})

    async def delete(page_ids):
        assert read_update_lock(repo) is not None
        assert page_ids == [page_id]

    store = SimpleNamespace(delete_many=AsyncMock(side_effect=delete), close=AsyncMock())
    build = Mock(return_value=store)
    monkeypatch.setattr(incremental, "_build_update_vector_store", build)

    run()

    store.delete_many.assert_awaited_once_with([page_id])
    store.close.assert_awaited_once()
    assert load_cleanup_debt(repo)["vectors"] == set()
    assert read_update_lock(repo) is None
    run()
    build.assert_called_once()  # Nothing to open on the next idle check.


@pytest.mark.parametrize("mode", ["dry_run", "busy"])
@pytest.mark.parametrize("kind", ["fts", "vectors"])
def test_idle_cleanup_preserves_debt_when_not_allowed(idle_update, monkeypatch, mode, kind):
    repo, head, run = idle_update
    record_cleanup_debt(repo, kind, {"retired"})
    build = Mock(side_effect=AssertionError("must not open the vector store"))
    monkeypatch.setattr(incremental, "_build_update_vector_store", build)
    if mode == "busy":
        assert try_acquire_update_lock(repo, head) is None
    try:
        run(dry_run=mode == "dry_run")
        assert load_cleanup_debt(repo)[kind] == {"retired"}
        build.assert_not_called()
        if mode == "busy":
            assert read_update_lock(repo) is not None
    finally:
        if mode == "busy":
            release_update_lock(repo)


def test_idle_cleanup_failure_stays_retryable(idle_update, monkeypatch):
    repo, _head, run = idle_update
    record_cleanup_debt(repo, "vectors", {"retired"})
    store = SimpleNamespace(
        delete_many=AsyncMock(side_effect=OSError("offline")), close=AsyncMock()
    )
    monkeypatch.setattr(incremental, "_build_update_vector_store", Mock(return_value=store))
    run()
    assert load_cleanup_debt(repo)["vectors"] == {"retired"}
    assert read_update_lock(repo) is None
    store.close.assert_awaited_once()
    store.delete_many.side_effect = None
    run()
    assert load_cleanup_debt(repo)["vectors"] == set()
    assert store.close.await_count == 2


@pytest.mark.parametrize("kind", ["fts", "vectors"])
def test_idle_cleanup_defers_when_native_lock_creation_fails(
    idle_update, monkeypatch, capsys, kind
):
    from repowise.core import update_lock

    repo, _head, run = idle_update
    record_cleanup_debt(repo, kind, {"retired"})
    build = Mock(side_effect=AssertionError("must not open the vector store"))
    monkeypatch.setattr(incremental, "_build_update_vector_store", build)
    # The native lock API intentionally returns None on unexpected OSError.
    # Cleanup needs proof of ownership in addition to that advisory result.
    monkeypatch.setattr(update_lock.os, "link", Mock(side_effect=OSError("read-only")))

    output = run() or capsys.readouterr().out

    assert "update lock ownership could not be verified" in output
    assert load_cleanup_debt(repo)[kind] == {"retired"}
    assert read_update_lock(repo) is None
    build.assert_not_called()


def test_idle_update_retries_fts_only_debt(idle_update, monkeypatch):
    from repowise.cli._repo_session import open_repo_db
    from repowise.core.persistence import FullTextSearch, get_session, upsert_page
    from tests.unit.persistence.helpers import make_page_kwargs

    repo, _head, run = idle_update
    page_id = "file_page:removed.py"

    async def indexed_ids(*, seed=False):
        engine, factory, repo_id = await open_repo_db(repo, repo_name="test")
        try:
            fts = FullTextSearch(engine)
            if seed:
                async with get_session(factory) as session:
                    await upsert_page(
                        session,
                        **make_page_kwargs(
                            repo_id,
                            page_id=page_id,
                            freshness_status="tombstone",
                        ),
                    )
                await fts.ensure_index()
                await fts.index(
                    page_id, "Removed", "xylophone", summary="Removed", target_path="removed.py"
                )
            return await fts.list_indexed_ids()
        finally:
            await engine.dispose()

    assert page_id in asyncio.run(indexed_ids(seed=True))
    record_cleanup_debt(repo, "fts", {page_id})
    build = Mock(side_effect=AssertionError("FTS cleanup must not construct vectors"))
    monkeypatch.setattr(incremental, "_build_update_vector_store", build)

    run()

    assert load_cleanup_debt(repo)["fts"] == set()
    assert page_id not in asyncio.run(indexed_ids())
    assert read_update_lock(repo) is None
    build.assert_not_called()


def test_idle_fts_failure_does_not_block_vector_cleanup(idle_update, monkeypatch, capsys):
    from repowise.core.persistence.search import FullTextSearch

    repo, _head, run = idle_update
    for kind in ("fts", "vectors"):
        record_cleanup_debt(repo, kind, {"retired"})
    store = SimpleNamespace(delete_many=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(incremental, "_build_update_vector_store", Mock(return_value=store))
    monkeypatch.setattr(
        FullTextSearch, "delete_many", AsyncMock(side_effect=OSError("FTS offline"))
    )

    output = run() or capsys.readouterr().out

    assert "FTS offline" in output
    assert load_cleanup_debt(repo) == {"fts": {"retired"}, "vectors": set()}
    store.delete_many.assert_awaited_once_with(["retired"])
    store.close.assert_awaited_once()
    assert read_update_lock(repo) is None


def test_empty_retry_opens_no_resources_or_lock(tmp_path, monkeypatch):
    from repowise.core import update_lock
    from repowise.core.persistence import database

    acquire = Mock(side_effect=AssertionError("must not acquire an idle lock"))
    engine = Mock(side_effect=AssertionError("must not open SQL"))
    adapter = Mock(side_effect=AssertionError("must not open vectors"))
    monkeypatch.setattr(update_lock, "try_acquire_update_lock", acquire)
    monkeypatch.setattr(database, "create_engine", engine)
    monkeypatch.setattr(incremental, "_build_update_vector_store", adapter)

    incremental.retry_retired_page_cleanup(tmp_path)

    acquire.assert_not_called()
    engine.assert_not_called()
    adapter.assert_not_called()


def test_unavailable_authority_discloses_both_pending_indexes(tmp_path, monkeypatch, capsys):
    for kind in ("fts", "vectors"):
        record_cleanup_debt(tmp_path, kind, {"retired"})
    build = Mock(side_effect=AssertionError("must not open vectors without SQL authority"))
    monkeypatch.setattr(incremental, "_build_update_vector_store", build)

    incremental.retry_retired_page_cleanup(tmp_path)

    output = capsys.readouterr().out
    assert "FTS removal deferred" in output
    assert "vector removal deferred" in output
    assert output.count("page authority is unavailable") == 2
    assert load_cleanup_debt(tmp_path) == {"fts": {"retired"}, "vectors": {"retired"}}
    assert read_update_lock(tmp_path) is None
    build.assert_not_called()
