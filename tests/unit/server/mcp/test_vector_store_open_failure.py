"""A semantic index on disk that cannot be opened must not pass for healthy.

A partial LanceDB install imports fine and lacks ``connect_async``. The server
must preserve that native store failure through its empty in-memory fallback.
Request retrieval discloses it for the affected repository and lane; the
embedder's configuration state remains independent. Only an index that exists
and fails counts: no index at all is a keyless repo, not a failure.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from repowise.core.persistence.vector_store import InMemoryVectorStore, LanceDBVectorStore
from repowise.core.persistence.vector_store.lancedb_store import LanceDBUnavailableError
from repowise.core.providers.embedding.base import MockEmbedder
from repowise.server.mcp_server import _server, _state
from repowise.server.mcp_server._meta import build_meta


@pytest.fixture(autouse=True)
def fresh_store_errors(monkeypatch):
    monkeypatch.setattr(_state, "_vector_store_errors", {}, raising=False)


@pytest.fixture
def healthy_openai(monkeypatch):
    monkeypatch.setattr(
        _state,
        "_embedder_status",
        {"active": "openai", "requested": "openai", "degraded": False},
        raising=False,
    )
    monkeypatch.setattr(_state, "_vector_store", None, raising=False)
    monkeypatch.setattr(_state, "_decision_store", None, raising=False)
    monkeypatch.setattr(_state, "_vector_store_ready", None, raising=False)
    monkeypatch.setattr(_server, "_query_embedder", MockEmbedder)


@pytest.fixture
def broken_lancedb(monkeypatch):
    async def _boom(self):
        raise LanceDBUnavailableError("LanceDB is missing or broken (AttributeError: connect_async)")

    monkeypatch.setattr(LanceDBVectorStore, "_ensure_connected", _boom)


def test_an_unopenable_index_records_a_native_store_failure(
    tmp_path, healthy_openai, broken_lancedb
):
    (tmp_path / ".repowise" / "lancedb").mkdir(parents=True)

    asyncio.run(_server._load_vector_stores(str(tmp_path)))

    assert isinstance(_state._vector_store, InMemoryVectorStore)
    meta = build_meta(timing_ms=1.0)
    assert meta["embedder_degraded"] is False
    assert "connect_async" in _state._vector_store_errors[""]
    assert "reinstall" in _state._vector_store_errors[""]


def test_the_mark_survives_resolving_the_embedder_again(
    tmp_path, healthy_openai, broken_lancedb, monkeypatch
):
    """The workspace registry resolves the embedder on every repo (re)load,
    which rewrites ``_embedder_status``. The store failure must outlive that."""
    (tmp_path / ".repowise" / "lancedb").mkdir(parents=True)
    asyncio.run(_server._load_vector_stores(str(tmp_path)))

    monkeypatch.setattr(_server, "_configured_embedder_name", lambda: "mock")
    _server._resolve_embedder()

    meta = build_meta(timing_ms=1.0)
    assert meta["embedder_degraded"] is False
    assert "connect_async" in _state._vector_store_errors[""]


def test_a_locked_table_is_not_blamed_on_the_install(tmp_path, healthy_openai, monkeypatch):
    async def _locked(self):
        raise OSError("database is locked")

    monkeypatch.setattr(LanceDBVectorStore, "_ensure_connected", _locked)
    (tmp_path / ".repowise" / "lancedb").mkdir(parents=True)

    asyncio.run(_server._load_vector_stores(str(tmp_path)))

    warning = _state._vector_store_errors[""]
    assert "database is locked" in warning
    assert "reinstall" not in warning
    assert "retry" in warning


def test_no_index_on_disk_is_not_a_failure(tmp_path, healthy_openai, broken_lancedb):
    asyncio.run(_server._load_vector_stores(str(tmp_path)))

    assert build_meta(timing_ms=1.0)["embedder_degraded"] is False


def test_the_workspace_registry_reports_an_unopenable_index(tmp_path, broken_lancedb):
    from repowise.core.workspace.registry import RepoRegistry

    (tmp_path / ".repowise" / "lancedb").mkdir(parents=True)
    reported: list[tuple[str, BaseException]] = []

    async def _go():
        ctx = SimpleNamespace(alias="api", vector_store_ready=asyncio.Event())
        registry = SimpleNamespace(
            _on_vector_store_error=lambda alias, exc: reported.append((alias, exc)),
            _contexts={"api": ctx},
            _vs_tasks={},
        )
        await RepoRegistry._load_vector_stores(registry, ctx, tmp_path, MockEmbedder())
        return ctx

    ctx = asyncio.run(_go())

    assert [alias for alias, _ in reported] == ["api"]
    assert "connect_async" in str(reported[0][1])
    assert ctx.vector_store_ready.is_set()


@pytest.fixture
def bridge_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(_state, "_embedder_status", None, raising=False)
    monkeypatch.setattr(
        "repowise.cli.providers.embedders.resolve_embedder_for_repo", lambda p: "openai"
    )
    monkeypatch.setattr(
        "repowise.cli.providers.embedders.build_embedder", lambda name, _p=None: MockEmbedder()
    )
    (tmp_path / ".repowise" / "lancedb").mkdir(parents=True)
    return tmp_path


def test_the_cli_bridge_reports_an_unopenable_index(bridge_repo, broken_lancedb):
    from repowise.cli import tool_bridge

    async def _go():
        return await tool_bridge._connect_or_degrade(
            await tool_bridge._open_vector_store(bridge_repo)
        )

    store = asyncio.run(_go())

    assert isinstance(store, InMemoryVectorStore)
    assert build_meta(timing_ms=1.0)["embedder_degraded"] is False
    assert "connect_async" in _state._vector_store_errors[""]


def test_the_cli_bridge_does_not_open_the_store_for_other_tools(bridge_repo, monkeypatch):
    """Importing lancedb costs about a second; tools that never read vectors
    must not pay it."""
    from repowise.cli import tool_bridge

    connected: list[bool] = []

    async def _spy(self):
        connected.append(True)

    monkeypatch.setattr(LanceDBVectorStore, "_ensure_connected", _spy)

    assert "get_context" not in tool_bridge._VECTOR_TOOLS
    store = asyncio.run(tool_bridge._open_vector_store(bridge_repo))

    assert isinstance(store, LanceDBVectorStore)
    assert connected == []
    assert {"search_codebase", "get_answer"} <= tool_bridge._VECTOR_TOOLS


def test_runtime_restart_clears_old_vector_store_failures(monkeypatch):
    state = SimpleNamespace(_vector_store_errors={"api": "prior failure"})
    monkeypatch.setattr(_server, "_state", state)
    _server._clear_runtime_state()
    assert state._vector_store_errors == {}


@pytest.mark.parametrize("index_exists", [True, False])
async def test_workspace_recovery_clears_its_alias_error(
    tmp_path, healthy_openai, monkeypatch, index_exists
):
    from repowise.core.workspace.registry import RepoRegistry

    if index_exists:
        (tmp_path / ".repowise" / "lancedb").mkdir(parents=True)
    _server._mark_vector_store_unreadable(OSError("prior failure"), "api")
    assert "api" in _state._vector_store_errors

    async def opened(self):
        return None

    monkeypatch.setattr(LanceDBVectorStore, "_ensure_connected", opened)
    ctx = SimpleNamespace(alias="api", vector_store_ready=asyncio.Event())
    registry = SimpleNamespace(
        _on_vector_store_error=lambda alias, exc: _server._mark_vector_store_unreadable(exc, alias),
        _contexts={"api": ctx},
        _vs_tasks={},
    )
    await RepoRegistry._load_vector_stores(registry, ctx, tmp_path, MockEmbedder())
    assert _state._vector_store_errors == {}
    assert build_meta()["embedder_degraded"] is False
    assert ctx.vector_store_ready.is_set()


@pytest.mark.parametrize("failure", [None, OSError("stale open failure")])
async def test_stale_workspace_loader_cannot_change_current_error(
    tmp_path, healthy_openai, monkeypatch, failure
):
    from repowise.core.workspace.registry import RepoRegistry

    (tmp_path / ".repowise" / "lancedb").mkdir(parents=True)
    _state._vector_store_errors["api"] = "current context's failure"

    async def opened(self):
        if failure is not None:
            raise failure

    monkeypatch.setattr(LanceDBVectorStore, "_ensure_connected", opened)
    old = SimpleNamespace(alias="api", vector_store_ready=asyncio.Event())
    current = SimpleNamespace(alias="api", vector_store_ready=asyncio.Event())
    registry = SimpleNamespace(
        _on_vector_store_error=lambda alias, exc: _server._mark_vector_store_unreadable(exc, alias),
        _contexts={"api": current},
        _vs_tasks={},
    )
    await RepoRegistry._load_vector_stores(registry, old, tmp_path, MockEmbedder())
    assert _state._vector_store_errors == {"api": "current context's failure"}
    assert not current.vector_store_ready.is_set()
