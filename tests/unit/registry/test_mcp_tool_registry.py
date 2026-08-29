"""MCPToolRegistry behavior."""

from __future__ import annotations

from typing import Any

import pytest

from repowise.core.registry import MCPToolRegistry


class _FakeMCP:
    """Minimal stand-in for FastMCP — captures ``mcp.tool()(fn)`` calls."""

    def __init__(self) -> None:
        self.registered: list[Any] = []

    def tool(self) -> Any:
        def _decorator(fn: Any) -> Any:
            self.registered.append(fn)
            return fn

        return _decorator


@pytest.fixture
def registry() -> MCPToolRegistry:
    return MCPToolRegistry()


def test_register_decorator_paren_form(registry):
    @registry.register()
    async def my_tool() -> dict:
        return {}

    assert my_tool in registry.tools()


def test_register_decorator_bare_form(registry):
    @registry.register
    async def my_tool() -> dict:
        return {}

    assert my_tool in registry.tools()


def test_tool_alias_matches_register(registry):
    @registry.tool()
    async def my_tool() -> dict:
        return {}

    assert my_tool in registry.tools()


def test_legacy_default_workspace_tool_infers_utility_tier(registry):
    @registry.tool(requires_workspace=True)
    async def workspace_utility() -> dict:
        return {}

    entry = registry.entries()[0]
    assert entry.default is True
    assert entry.requires_workspace is True
    assert entry.tier == "utility"


def test_apply_registers_with_server(registry):
    @registry.register
    async def t1() -> dict:
        return {}

    @registry.register
    async def t2() -> dict:
        return {}

    mcp = _FakeMCP()
    registry.apply(mcp)
    assert t1 in mcp.registered
    assert t2 in mcp.registered


def test_apply_is_idempotent_per_server(registry):
    @registry.register
    async def t1() -> dict:
        return {}

    mcp = _FakeMCP()
    registry.apply(mcp)
    registry.apply(mcp)
    assert mcp.registered.count(t1) == 1


def test_apply_supports_multiple_servers(registry):
    @registry.register
    async def t1() -> dict:
        return {}

    a = _FakeMCP()
    b = _FakeMCP()
    registry.apply(a)
    registry.apply(b)
    assert t1 in a.registered
    assert t1 in b.registered


class _MetadataMCP:
    """Stand-in that records the ``title``/``annotations`` each tool arrives with."""

    def __init__(self) -> None:
        self.registered: dict[str, dict[str, Any]] = {}

    def tool(self, **options: Any) -> Any:
        def _decorator(fn: Any) -> Any:
            self.registered[fn.__name__] = options
            return fn

        return _decorator


def test_apply_fills_metadata_from_the_resolver(registry):
    @registry.register
    async def t1() -> dict:
        return {}

    mcp = _MetadataMCP()
    registry.apply(mcp, metadata=lambda entry: (f"Title {entry.name}", {"readOnlyHint": True}))
    assert mcp.registered["t1"] == {
        "title": "Title t1",
        "annotations": {"readOnlyHint": True},
    }


def test_entry_metadata_wins_over_the_resolver(registry):
    @registry.register(title="Declared", annotations={"readOnlyHint": False})
    async def t1() -> dict:
        return {}

    mcp = _MetadataMCP()
    registry.apply(mcp, metadata=lambda entry: ("Resolved", {"readOnlyHint": True}))
    assert mcp.registered["t1"] == {
        "title": "Declared",
        "annotations": {"readOnlyHint": False},
    }


def test_apply_without_metadata_passes_no_options(registry):
    """A tool with nothing to say still reaches the server as ``mcp.tool()``."""

    @registry.register
    async def t1() -> dict:
        return {}

    mcp = _MetadataMCP()
    registry.apply(mcp)
    assert mcp.registered["t1"] == {}


def test_reset_clears_tools(registry):
    @registry.register
    async def t1() -> dict:
        return {}

    registry.reset()
    assert registry.tools() == []
