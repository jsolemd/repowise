"""The up-to-date freshness stamp never waits on, or fails beside, a live update.

The watcher's startup pass found Make up to date and stamped its store while
the reconciliation timer held Make's update lock and re-rendered 319 wiki
pages in one transaction. The stamp outlasted SQLite's busy timeout, raised
"database is locked", and aborted the watcher's whole step for the repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

from repowise.core.persistence import create_engine, create_session_factory, get_session, init_db
from repowise.core.persistence.crud import get_repository_by_path
from repowise.core.persistence.database import resolve_db_url
from repowise.core.update_lock import (
    read_update_lock,
    release_update_lock,
    try_acquire_update_lock,
)
from repowise.core.workspace import update as ws_update

HEAD = "b" * 40


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    for name in ("REPOWISE_DB_URL", "REPOWISE_DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "repo"
    (path / ".repowise").mkdir(parents=True)
    (path / ".repowise" / "wiki.db").write_bytes(b"")
    return path


async def _stamped_head(repo: Path) -> str | None:
    engine = create_engine(resolve_db_url(repo))
    try:
        await init_db(engine)
        async with get_session(create_session_factory(engine)) as session:
            row = await get_repository_by_path(session, str(repo))
            return row.head_commit if row is not None else None
    finally:
        await engine.dispose()


async def test_an_idle_repo_is_stamped(repo):
    assert await ws_update.reconcile_idle_repo_head_commit(repo, HEAD) is True
    assert await _stamped_head(repo) == HEAD


async def test_a_live_update_keeps_the_stamp_for_itself(repo):
    assert try_acquire_update_lock(repo, HEAD) is None
    try:
        assert await ws_update.reconcile_idle_repo_head_commit(repo, HEAD) is False
        # The lock was only read: it is still the owner's to release.
        assert read_update_lock(repo) is not None
    finally:
        release_update_lock(repo)
    assert await _stamped_head(repo) is None


async def test_a_busy_store_defers_the_stamp(repo, monkeypatch):
    async def locked(*_args):
        raise OperationalError("UPDATE repositories", {}, Exception("database is locked"))

    monkeypatch.setattr(ws_update, "reconcile_repo_head_commit", locked)
    assert await ws_update.reconcile_idle_repo_head_commit(repo, HEAD) is False


async def test_any_other_store_error_still_raises(repo, monkeypatch):
    async def broken(*_args):
        raise OperationalError("UPDATE repositories", {}, Exception("no such table: repositories"))

    monkeypatch.setattr(ws_update, "reconcile_repo_head_commit", broken)
    with pytest.raises(OperationalError):
        await ws_update.reconcile_idle_repo_head_commit(repo, HEAD)
