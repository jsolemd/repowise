"""Cancelling index reads must return their real SQLite connections."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import event

from repowise.core.workspace import repo_index as module
from tests.unit.workspace._repo_index import make_repo_index


@pytest.mark.parametrize("operation", ["open", "close"])
async def test_cancelled_cleanup_returns_connection(tmp_path, monkeypatch, operation):
    index = await make_repo_index(tmp_path, {})
    closing, resume = asyncio.Event(), asyncio.Event()
    opened, returned = [], []

    def block_close(index):
        opened.append(index)
        event.listen(index._engine.sync_engine, "checkin", lambda c, r: returned.append(True))
        original_close = index.session.close

        async def close():
            closing.set()
            await resume.wait()
            await original_close()

        monkeypatch.setattr(index.session, "close", close)

    if operation == "open":
        await index.close()

        async def fail(index, *args):
            block_close(index)
            raise LookupError("load failed")

        monkeypatch.setattr(module.RepoIndex, "_load", fail)
        task = asyncio.create_task(module.open_repo_index("alpha", tmp_path))
    else:
        block_close(index)
        task = asyncio.create_task(index.close())
    try:
        await closing.wait()
        task.cancel()
        resume.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert returned
    finally:
        resume.set()
        for index in opened:
            await index.close()


async def test_cancelled_close_preserves_cancellation_when_session_close_fails():
    closing, resume = asyncio.Event(), asyncio.Event()
    disposed = []

    async def close():
        closing.set()
        await resume.wait()
        raise RuntimeError("close failed")

    async def dispose():
        disposed.append(True)

    task = asyncio.create_task(
        module._close_index_db(SimpleNamespace(close=close), SimpleNamespace(dispose=dispose))
    )
    await closing.wait()
    task.cancel()
    resume.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert disposed


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
