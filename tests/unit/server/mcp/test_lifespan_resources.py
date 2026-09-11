"""HTTP clients must share the process-owned runtime and release it exactly once."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest

from repowise.server.mcp_server import _server, _state


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    import repowise.core.workspace.registry as registry_module
    from repowise.server.mcp_server import _test_impact

    instances = []
    ready = asyncio.Event()
    ready.set()
    context = SimpleNamespace(
        session_factory=object(), fts=object(), vector_store=object(),
        decision_store=object(), vector_store_ready=ready,
    )
    load = AsyncMock(return_value=context)

    class Registry:
        def __init__(self, **kwargs):
            self.close = AsyncMock()
            instances.append(self)

        async def get_default(self):
            return await load()

        def get_default_alias(self):
            return "infra"

    async def warm():
        _state._lancedb_ready.set()

    config = SimpleNamespace(repos=[], get_repo=lambda alias: None)
    monkeypatch.setattr(registry_module, "RepoRegistry", Registry)
    monkeypatch.setattr(_server, "_detect_workspace", lambda path: (tmp_path, config, None))
    monkeypatch.setattr(_server, "_warm_lancedb", warm)
    cleanup = AsyncMock()
    monkeypatch.setattr(_test_impact, "close_test_impact_indexes", cleanup)
    for field in (
        "_registry", "_workspace_root", "_session_factory", "_fts",
        "_vector_store", "_decision_store", "_vector_store_ready",
        "_lancedb_ready", "_cross_repo_enricher",
    ):
        monkeypatch.setattr(_state, field, None)
    monkeypatch.setattr(_state, "_force_single_repo", False)
    monkeypatch.setattr(_state, "_runtime_users", 0)
    monkeypatch.setattr(_state, "_runtime_context", None)
    monkeypatch.setattr(_state, "_runtime_lock", None)
    return SimpleNamespace(instances=instances, load=load, cleanup=cleanup)


async def test_overlapping_clients_share_one_runtime(runtime):
    async with _server._lifespan(_server.mcp):
        first = _state._registry
        async with _server._lifespan(_server.mcp):
            assert _state._registry is first
            assert len(runtime.instances) == 1
        assert _state._registry is first
        first.close.assert_not_awaited()

    first.close.assert_awaited_once()
    runtime.cleanup.assert_awaited_once()
    assert _state._registry is None


async def test_one_client_exit_keeps_other_client_resources(runtime):
    first = _server._lifespan(_server.mcp)
    second = _server._lifespan(_server.mcp)
    await first.__aenter__()
    registry = _state._registry
    try:
        await second.__aenter__()
        await first.__aexit__(None, None, None)
        assert _state._registry is registry
        registry.close.assert_not_awaited()
    finally:
        await second.__aexit__(None, None, None)

    registry.close.assert_awaited_once()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_last_client_failure_releases_runtime(runtime, failure):
    with pytest.raises(failure):
        async with _server._lifespan(_server.mcp):
            raise failure("client stopped")

    runtime.instances[0].close.assert_awaited_once()


async def test_cancelled_anyio_scope_still_finishes_async_cleanup(runtime):
    closed = asyncio.Event()

    async def close():
        await anyio.sleep(0)
        closed.set()

    with anyio.CancelScope() as scope:
        async with _server._lifespan(_server.mcp):
            runtime.instances[0].close.side_effect = close
            scope.cancel()
            await anyio.sleep(0)

    assert closed.is_set()
    assert _state._runtime_users == 0
    assert _state._registry is None
    runtime.cleanup.assert_awaited_once()
    assert _state._registry is None


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_startup_failure_releases_acquired_resources(runtime, failure):
    runtime.load.side_effect = failure("load failed")

    with pytest.raises(failure):
        async with _server._lifespan(_server.mcp):
            pytest.fail("Failed startup must not yield")

    runtime.instances[0].close.assert_awaited_once()
    assert _state._lancedb_ready is None


async def test_concurrent_cold_clients_share_initialization(runtime):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def client():
        async with _server._lifespan(_server.mcp):
            entered.set()
            await release.wait()

    tasks = [asyncio.create_task(client()) for _ in range(4)]
    try:
        await entered.wait()
        await asyncio.sleep(0)
        assert len(runtime.instances) == 1
        runtime.load.assert_awaited_once()
    finally:
        release.set()
        await asyncio.gather(*tasks)

    runtime.instances[0].close.assert_awaited_once()
