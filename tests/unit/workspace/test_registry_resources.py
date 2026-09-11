"""A failed context load must return connections before registry ownership."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import anyio
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from repowise.core.workspace import registry as module
from tests.unit.workspace.test_registry import _make_workspace


@pytest.mark.parametrize("stage", ["database", "fts"])
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_failed_load_disposes_engine(tmp_path, monkeypatch, stage, failure):
    from repowise.core.persistence import database
    from repowise.core.persistence.search import FullTextSearch

    config = _make_workspace(tmp_path, ["repo"])
    original = database.create_engine
    engines = []

    def create(*args, **kwargs):
        engine = original(*args, **kwargs)
        engines.append(engine)
        return engine

    disposed = []
    engine_type = AsyncEngine
    real_dispose = engine_type.dispose

    async def dispose(engine, *args, **kwargs):
        disposed.append(engine)
        await real_dispose(engine, *args, **kwargs)

    monkeypatch.setattr(database, "create_engine", create)
    monkeypatch.setattr(engine_type, "dispose", dispose)
    if stage == "database":
        monkeypatch.setattr(database, "init_db", AsyncMock(side_effect=failure("startup")))
    else:
        monkeypatch.setattr(FullTextSearch, "ensure_index", AsyncMock(side_effect=failure("fts")))
    registry = module.RepoRegistry(tmp_path, config)

    with pytest.raises(failure):
        await registry.get_default()
    assert disposed == engines
    assert len(disposed) == 1
    assert registry._contexts == {}
    await registry.close()
    assert len(disposed) == 1


@pytest.mark.parametrize("stage", ["database", "fts"])
async def test_cancelled_anyio_scope_finishes_disposal(tmp_path, monkeypatch, stage):
    from repowise.core.persistence import database
    from repowise.core.persistence.search import FullTextSearch

    config = _make_workspace(tmp_path, ["repo"])
    real_dispose = AsyncEngine.dispose
    completed = []

    async def dispose(engine, *args, **kwargs):
        await anyio.lowlevel.checkpoint()
        await real_dispose(engine, *args, **kwargs)
        completed.append(engine)

    monkeypatch.setattr(AsyncEngine, "dispose", dispose)
    registry = module.RepoRegistry(tmp_path, config)
    with anyio.CancelScope() as scope:
        async def cancel(*args, **kwargs):
            scope.cancel()
            await anyio.lowlevel.checkpoint()

        if stage == "database":
            monkeypatch.setattr(database, "init_db", cancel)
        else:
            monkeypatch.setattr(FullTextSearch, "ensure_index", cancel)
        await registry.get_default()
    assert len(completed) == 1
    assert registry._contexts == {}
    await registry.close()
    assert len(completed) == 1
