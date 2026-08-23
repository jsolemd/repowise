"""The store creates its own tables, and degrades honestly when they are absent.

``init_db`` only creates tables reachable from ``Base.metadata``, and a table
is only reachable once its module has been imported — which nothing in the
shared bootstrap does for this package. ``ensure_schema`` closes that gap
without reaching into the bootstrap; these tests hold both halves of that
arrangement in place.
"""

from __future__ import annotations

import sqlalchemy as sa

from repowise.core.persistence.database import create_engine, init_db
from repowise.core.refsites.schema import REFSITE_TABLES, ensure_schema
from repowise.server.mcp_server.tool_refsites import get_reference_sites

_TABLE_NAMES = {table.name for table in REFSITE_TABLES}


async def _tables(engine) -> set[str]:
    async with engine.connect() as connection:
        rows = await connection.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type = 'table'")
        )
        return {row[0] for row in rows}


async def test_ensure_schema_creates_the_tables():
    engine = create_engine("sqlite+aiosqlite:///:memory:", use_static_pool=True)
    try:
        async with engine.begin() as connection:
            # Only the repositories table, so the FK target exists.
            from repowise.core.persistence.models import Base, Repository

            await connection.run_sync(Base.metadata.create_all, tables=[Repository.__table__])
        assert not _TABLE_NAMES & await _tables(engine)

        await ensure_schema(engine)
        assert await _tables(engine) >= _TABLE_NAMES
    finally:
        await engine.dispose()


async def test_ensure_schema_is_idempotent():
    engine = create_engine("sqlite+aiosqlite:///:memory:", use_static_pool=True)
    try:
        await init_db(engine)
        await ensure_schema(engine)
        before = await _tables(engine)
        await ensure_schema(engine)
        assert await _tables(engine) == before
    finally:
        await engine.dispose()


async def test_tool_reports_not_indexed_when_the_tables_are_missing(mcp_state, async_engine):
    """A server whose database predates this package must not raise at an agent.

    An unhandled ``OperationalError`` reaches FastMCP as a protocol-level
    error, which teaches the caller to abandon the server for the session. A
    shaped ``not_indexed`` response says the same thing recoverably.
    """
    async with async_engine.begin() as connection:
        for table in REFSITE_TABLES:
            await connection.execute(sa.text(f"DROP TABLE {table.name}"))

    result = await get_reference_sites("calc.ts::computeTotal")

    assert result["status"] == "not_indexed"
    assert result["sites"] == []
    assert "does not mean the symbol is unreferenced" in result["explanation"]
