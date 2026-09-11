"""Cancelling index reads must return their real SQLite connections."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anyio
import pytest
from sqlalchemy import event, text

from repowise.core.workspace import repo_index as module
from tests.unit.workspace._repo_index import make_repo_index


@pytest.mark.parametrize("stage", ["lookup", "load"])
async def test_cancelled_open_returns_connection(tmp_path, monkeypatch, stage):
    from repowise.core.persistence.crud import repository
    from repowise.core.workspace import registry

    seed = await make_repo_index(tmp_path, {})
    await seed.close()
    original_open = registry.open_repo_db
    checked_out = set()
    sessions = []
    engines = []

    async def open_db(path):
        engine, factory = await original_open(path)
        engines.append(engine)
        event.listen(engine.sync_engine, "checkout", lambda c, r, p: checked_out.add(id(c)))
        event.listen(engine.sync_engine, "checkin", lambda c, r: checked_out.discard(id(c)))

        def session_factory():
            session = factory()
            sessions.append(session)
            return session

        return engine, session_factory

    monkeypatch.setattr(registry, "open_repo_db", open_db)
    try:
        with anyio.CancelScope() as scope:

            async def cancel(session, *args):
                await session.execute(text("SELECT 1"))
                scope.cancel()
                await anyio.lowlevel.checkpoint()

            if stage == "lookup":
                monkeypatch.setattr(repository, "get_repository_by_path", cancel)
            else:

                async def load(index, *args):
                    await cancel(index.session)

                monkeypatch.setattr(module.RepoIndex, "_load", load)
            await module.open_repo_index("alpha", tmp_path)
        assert not checked_out
    finally:
        # Keep a failing regression from leaking into the following tests.
        for session in sessions:
            await session.close()
        for engine in engines:
            await engine.dispose()


async def test_cancelled_close_finishes_before_return(tmp_path):
    index = await make_repo_index(tmp_path, {})
    returned = []
    event.listen(index._engine.sync_engine, "checkin", lambda c, r: returned.append(True))
    try:
        with anyio.CancelScope() as scope:
            scope.cancel()
            await index.close()
        assert returned
    finally:
        await index.close()


async def test_cancelled_workspace_open_closes_already_opened_siblings(tmp_path, monkeypatch):
    index = await make_repo_index(tmp_path / "alpha", {})
    returned = []
    event.listen(index._engine.sync_engine, "checkin", lambda c, r: returned.append(True))
    first_ready = asyncio.Event()
    second_started = asyncio.Event()

    async def open_index(alias, path):
        if alias == "alpha":
            first_ready.set()
            return index
        second_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(module, "open_repo_index", open_index)
    config = SimpleNamespace(repos=[SimpleNamespace(alias=a, path=a) for a in ("alpha", "beta")])
    task = asyncio.create_task(module.open_workspace_index(config, tmp_path))
    try:
        await first_ready.wait()
        await second_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert returned
    finally:
        await index.close()
