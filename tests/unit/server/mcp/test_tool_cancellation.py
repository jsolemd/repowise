"""The AnyIO HTTP boundary must allow asyncio tool cleanup to finish."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time

import anyio
import pytest
from sqlalchemy import event, text

from repowise.core.persistence.database import create_engine, create_session_factory, get_session
from repowise.server.mcp_server import _state
from repowise.server.mcp_server._failure_shield import finish_on_cancel


@pytest.fixture(autouse=True)
def warmed_runtime(monkeypatch):
    monkeypatch.setattr(_state, "_lancedb_ready", None)


async def test_http_cancellation_finishes_tool_cleanup_before_return():
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def tool():
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            await anyio.lowlevel.checkpoint()
            closed.set()

    async with anyio.create_task_group() as group:
        group.start_soon(finish_on_cancel(tool))
        await entered.wait()
        group.cancel_scope.cancel()
    assert closed.is_set()


async def test_tool_retains_request_context():
    request = contextvars.ContextVar("request")
    token = request.set("the-current-request")
    try:

        async def tool():
            return request.get()

        assert await finish_on_cancel(tool)() == "the-current-request"
    finally:
        request.reset(token)


@pytest.mark.parametrize("stage", ["handler", "budget_lookup"])
async def test_cancel_during_real_sqlite_io_returns_connection(tmp_path, caplog, monkeypatch, stage):
    from repowise.server.mcp_server import _budget, tool_middleware

    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'cancel.db'}")
    factory = create_session_factory(engine)
    entered = asyncio.Event()
    checked_out = set()
    caplog.set_level(logging.ERROR, logger="sqlalchemy.pool")

    @event.listens_for(engine.sync_engine, "connect")
    def configure(connection, record):
        connection.create_function("wait_for_cancel", 0, lambda: time.sleep(0.1))

    event.listen(engine.sync_engine, "checkout", lambda c, r, p: checked_out.add(id(r)))
    event.listen(engine.sync_engine, "checkin", lambda c, r: checked_out.discard(id(r)))

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def executing(connection, cursor, statement, parameters, context, executemany):
        if "wait_for_cancel" in statement:
            entered.set()

    async def read_db():
        async with get_session(factory) as session:
            await session.execute(text("SELECT wait_for_cancel()"))

    async def resolve(*args, **kwargs):
        if stage == "budget_lookup":
            await read_db()
        return tmp_path

    async def tool():
        await read_db()
        return {}

    monkeypatch.setattr(_budget, "resolve_response_budget_repo_root", resolve)
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(tool_middleware(tool))
            await entered.wait()
            group.cancel_scope.cancel()
        assert not checked_out
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    finally:
        await engine.dispose()
