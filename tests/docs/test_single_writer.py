"""Unit contracts for the docs background-runtime advisory lock.

No live database: the dedicated connection is injected as a stub so the lock
protocol (try, hold, unlock, close) is checked as a protocol.
"""

from __future__ import annotations

from typing import Any

import pytest

import repowise.docs.jobs.single_writer as single_writer
from repowise.docs.jobs.single_writer import (
    DOCS_BACKGROUND_LOCK_KEY,
    acquire_background_lock,
    holds_background_lock,
    release_background_lock,
)


class _StubConnection:
    """Records the advisory-lock statements a real asyncpg session would see."""

    def __init__(self, *, acquired: bool = True, unlock_raises: bool = False) -> None:
        self.acquired = acquired
        self.unlock_raises = unlock_raises
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.calls.append((query, args))
        if "pg_try_advisory_lock" in query:
            return self.acquired
        if "pg_advisory_unlock" in query:
            if self.unlock_raises:
                raise RuntimeError("connection already gone")
            return True
        raise AssertionError(f"unexpected query: {query}")

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def reset_lock_state(monkeypatch: pytest.MonkeyPatch) -> None:
    single_writer._lock_connection = None
    monkeypatch.setattr(
        single_writer,
        "get_settings",
        lambda: type("S", (), {"postgres_dsn": "postgresql://stub/doc_search"})(),
    )
    yield
    single_writer._lock_connection = None


def _connector(conn: _StubConnection):
    async def connect(dsn: str) -> _StubConnection:
        conn.dsn = dsn
        return conn

    return connect


@pytest.mark.asyncio
async def test_acquiring_takes_the_fixed_key_on_a_dedicated_connection() -> None:
    conn = _StubConnection(acquired=True)

    assert await acquire_background_lock(connect=_connector(conn)) is True
    assert holds_background_lock() is True
    assert conn.calls == [("SELECT pg_try_advisory_lock($1)", (DOCS_BACKGROUND_LOCK_KEY,))]
    # The connection stays open: a session-level advisory lock dies with its
    # session, so closing here would silently drop the lock.
    assert conn.closed is False


@pytest.mark.asyncio
async def test_a_contended_lock_is_refused_and_its_connection_closed() -> None:
    conn = _StubConnection(acquired=False)

    assert await acquire_background_lock(connect=_connector(conn)) is False
    assert holds_background_lock() is False
    assert conn.closed is True


@pytest.mark.asyncio
async def test_acquiring_twice_reuses_the_held_lock() -> None:
    first = _StubConnection(acquired=True)
    second = _StubConnection(acquired=True)

    assert await acquire_background_lock(connect=_connector(first)) is True
    assert await acquire_background_lock(connect=_connector(second)) is True
    assert second.calls == []
    assert second.closed is False


@pytest.mark.asyncio
async def test_releasing_unlocks_then_closes_the_connection() -> None:
    conn = _StubConnection(acquired=True)
    await acquire_background_lock(connect=_connector(conn))

    await release_background_lock()

    assert conn.calls[-1] == ("SELECT pg_advisory_unlock($1)", (DOCS_BACKGROUND_LOCK_KEY,))
    assert conn.closed is True
    assert holds_background_lock() is False


@pytest.mark.asyncio
async def test_releasing_without_the_lock_is_a_no_op() -> None:
    await release_background_lock()
    assert holds_background_lock() is False


@pytest.mark.asyncio
async def test_a_failed_unlock_still_closes_the_session() -> None:
    """Closing the session drops the lock anyway, so unlock is best effort."""
    conn = _StubConnection(acquired=True, unlock_raises=True)
    await acquire_background_lock(connect=_connector(conn))

    await release_background_lock()

    assert conn.closed is True
    assert holds_background_lock() is False


@pytest.mark.asyncio
async def test_a_failed_lock_probe_closes_the_connection_and_propagates() -> None:
    class _Exploding(_StubConnection):
        async def fetchval(self, query: str, *args: Any) -> Any:
            raise RuntimeError("server closed the connection")

    conn = _Exploding()
    with pytest.raises(RuntimeError, match="server closed the connection"):
        await acquire_background_lock(connect=_connector(conn))

    assert conn.closed is True
    assert holds_background_lock() is False


@pytest.mark.asyncio
async def test_acquiring_without_a_dsn_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        single_writer,
        "get_settings",
        lambda: type("S", (), {"postgres_dsn": ""})(),
    )
    with pytest.raises(ValueError, match="postgres_dsn"):
        await acquire_background_lock(connect=_connector(_StubConnection()))


def test_the_lock_key_is_a_frozen_signed_64_bit_literal() -> None:
    """Drifting the key would silently allow two background owners."""
    assert DOCS_BACKGROUND_LOCK_KEY == -6896715891042309208
    assert -(2**63) <= DOCS_BACKGROUND_LOCK_KEY < 2**63
