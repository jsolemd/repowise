"""Tests for the unified doc-tool bridge."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import repowise.docs.server.mcp_tools as doc_tools_module


@pytest.mark.asyncio
async def test_call_doc_tool_translates_arguments_and_formats_json(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _handler(arguments: dict[str, object]) -> dict[str, object]:
        seen.update(arguments)
        return {
            "library_id": arguments["library_id"],
            "query": arguments["query"],
            "output": arguments["format"],
        }

    monkeypatch.setattr(doc_tools_module, "ensure_docs_runtime", AsyncMock())
    monkeypatch.setitem(doc_tools_module._TOOL_HANDLERS, "search_docs", _handler)

    ctx = SimpleNamespace(
        settings=SimpleNamespace(project="solemd.graph", workspace="/workspaces/SoleMD.Graph")
    )
    response = await doc_tools_module.call_doc_tool(
        ctx,
        "search_docs",
        {
            "library_id": "/mantinedev/mantine",
            "query": "combobox selected option styles",
            "exact_match": True,
        },
        "json",
    )

    assert seen == {
        "library_id": "/mantinedev/mantine",
        "query": "combobox selected option styles",
        "exact_match": True,
        "format": "json",
    }
    payload = json.loads(response[0].text)
    assert payload["status"] == "success"
    assert payload["tool"] == "search_docs"
    assert payload["payload"]["library_id"] == "/mantinedev/mantine"
    assert payload["payload"]["query"] == "combobox selected option styles"
    assert payload["payload"]["output"] == "json"
    assert "scope" not in payload["payload"]


@pytest.mark.asyncio
async def test_call_doc_tool_wraps_markdown_success_in_the_standard_envelope(monkeypatch) -> None:
    async def _handler(arguments: dict[str, object]) -> dict[str, object]:
        assert arguments == {
            "library_id": "/framer/motion",
            "path": "docs/motion-component.md",
            "format": "markdown",
        }
        return {"content": "# Motion Component\n", "format": "markdown"}

    monkeypatch.setattr(doc_tools_module, "ensure_docs_runtime", AsyncMock())
    monkeypatch.setitem(doc_tools_module._TOOL_HANDLERS, "read_doc", _handler)

    ctx = SimpleNamespace(
        settings=SimpleNamespace(project="solemd.graph", workspace="/workspaces/SoleMD.Graph")
    )
    response = await doc_tools_module.call_doc_tool(
        ctx,
        "read_doc",
        {
            "library_id": "/framer/motion",
            "path": "docs/motion-component.md",
        },
        "markdown",
    )

    text = response[0].text
    assert "- status: `success`" in text
    assert "- tool: `read_doc`" in text
    assert "# Motion Component" in text


@pytest.mark.asyncio
async def test_search_docs_multi_library_ids_route_to_multi_handler(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _multi(arguments: dict[str, object]) -> dict[str, object]:
        seen.update(arguments)
        return {"results": [{"score": 1.0}], "total": 1}

    async def _single(arguments: dict[str, object]) -> dict[str, object]:
        seen["single_called"] = True
        return {"results": []}

    monkeypatch.setattr(doc_tools_module, "ensure_docs_runtime", AsyncMock())
    monkeypatch.setitem(doc_tools_module._TOOL_HANDLERS, "search_docs_multi", _multi)
    monkeypatch.setitem(doc_tools_module._TOOL_HANDLERS, "search_docs", _single)

    ctx = SimpleNamespace(
        settings=SimpleNamespace(project="solemd.graph", workspace="/workspaces/SoleMD.Graph")
    )
    response = await doc_tools_module.call_doc_tool(
        ctx,
        "search_docs",
        {"library_ids": ["/a/x", "/b/y"], "query": "combobox", "limit": 8},
        "json",
    )

    payload = json.loads(response[0].text)
    assert payload["status"] == "success"
    assert payload["tool"] == "search_docs"
    assert seen["library_ids"] == ["/a/x", "/b/y"]
    # search_docs `limit` maps onto the multi handler's per-library cap.
    assert seen["limit_per_library"] == 8
    assert "single_called" not in seen


@pytest.mark.asyncio
async def test_search_docs_single_library_id_uses_single_path(monkeypatch) -> None:
    seen: dict[str, object] = {}

    async def _single(arguments: dict[str, object]) -> dict[str, object]:
        seen.update(arguments)
        return {"results": [], "library_id": arguments.get("library_id")}

    async def _multi(arguments: dict[str, object]) -> dict[str, object]:
        seen["multi_called"] = True
        return {"results": []}

    monkeypatch.setattr(doc_tools_module, "ensure_docs_runtime", AsyncMock())
    monkeypatch.setitem(doc_tools_module._TOOL_HANDLERS, "search_docs", _single)
    monkeypatch.setitem(doc_tools_module._TOOL_HANDLERS, "search_docs_multi", _multi)

    ctx = SimpleNamespace(
        settings=SimpleNamespace(project="solemd.graph", workspace="/workspaces/SoleMD.Graph")
    )
    response = await doc_tools_module.call_doc_tool(
        ctx,
        "search_docs",
        {"library_ids": ["/only/one"], "query": "combobox"},
        "json",
    )

    payload = json.loads(response[0].text)
    assert payload["status"] == "success"
    assert seen["library_id"] == "/only/one"
    assert "library_ids" not in seen
    assert "multi_called" not in seen


@pytest.mark.asyncio
async def test_call_doc_tool_reports_unknown_doc_tool_as_error_json() -> None:
    ctx = SimpleNamespace(
        settings=SimpleNamespace(project="solemd.graph", workspace="/workspaces/SoleMD.Graph")
    )
    response = await doc_tools_module.call_doc_tool(
        ctx,
        "unknown_doc_tool",
        {},
        "json",
    )

    payload = json.loads(response[0].text)
    assert payload["status"] == "error"
    assert payload["payload"]["error_code"] == "unknown_tool"


@pytest.mark.asyncio
async def test_call_doc_tool_blocks_mutations_when_docs_background_runtime_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(doc_tools_module, "ensure_docs_runtime", AsyncMock())
    monkeypatch.setattr(doc_tools_module, "docs_runtime_accepts_mutations", lambda: False)

    ctx = SimpleNamespace(
        settings=SimpleNamespace(project="solemd.graph", workspace="/workspaces/SoleMD.Graph")
    )
    response = await doc_tools_module.call_doc_tool(
        ctx,
        "update_doc_library",
        {"library_id": "/mantinedev/mantine"},
        "json",
    )

    payload = json.loads(response[0].text)
    assert payload["status"] == "error"
    assert payload["payload"]["error_code"] == "docs_runtime_not_ready"


@pytest.mark.asyncio
async def test_call_doc_tool_rejects_background_mutations_without_runtime(monkeypatch) -> None:
    monkeypatch.setattr(doc_tools_module, "ensure_docs_runtime", AsyncMock())
    monkeypatch.setattr(doc_tools_module, "docs_runtime_accepts_mutations", lambda: False)

    ctx = SimpleNamespace(
        settings=SimpleNamespace(project="solemd.graph", workspace="/workspaces/SoleMD.Graph")
    )
    response = await doc_tools_module.call_doc_tool(
        ctx,
        "update_doc_library",
        {
            "library_id": "/mantinedev/mantine",
            "force": True,
        },
        "json",
    )

    payload = json.loads(response[0].text)
    assert payload["status"] == "error"
    assert payload["payload"]["error_code"] == "docs_runtime_not_ready"
