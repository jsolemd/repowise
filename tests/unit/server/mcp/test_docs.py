"""Documentation tools share RepoWise without adopting code-index dependencies."""

import inspect
from unittest.mock import AsyncMock

from repowise.core.registry import mcp_tool_registry
from repowise.docs.catalog import get_doc_tool_definitions
from repowise.docs.client import DocsUnavailable
from repowise.server.mcp_server import ensure_full_surface, tool_docs, tool_middleware
from repowise.server.mcp_server._budget import enforce_response_budget

PUBLIC = {tool.name for tool in get_doc_tool_definitions()} - {"search_docs_multi"}


def test_docs_roster_and_argument_contracts_are_registered_as_opt_ins():
    ensure_full_surface()
    entries = {entry.name: entry for entry in mcp_tool_registry.entries()}
    assert entries.keys() >= PUBLIC
    for definition in get_doc_tool_definitions():
        if definition.name not in PUBLIC:
            continue
        entry = entries[definition.name]
        assert entry.default is False
        signature = inspect.signature(entry.fn)
        schema = definition.inputSchema
        assert set(signature.parameters) == set(schema["properties"])
        required = {
            name
            for name, param in signature.parameters.items()
            if param.default is inspect.Parameter.empty
        }
        assert required == set(schema.get("required", []))


async def test_documentation_does_not_resolve_a_code_repository(monkeypatch):
    from repowise.server.mcp_server import _budget

    resolver = AsyncMock(side_effect=AssertionError("Docs must not query a code index"))
    monkeypatch.setattr(_budget, "resolve_response_budget_repo_root", resolver)
    call = AsyncMock(
        return_value={"status": "success", "payload": {"library_id": "/owner/library"}}
    )
    monkeypatch.setattr(tool_docs.DocsClient, "call_tool", call)
    result = await tool_middleware(tool_docs.add_doc_library)(repo="owner/library", name="Library")
    assert result["payload"]["library_id"] == "/owner/library"
    assert result["trust"]["evidence_kind"] == "documentation"
    resolver.assert_not_awaited()


async def test_unavailable_worker_is_a_docs_result(monkeypatch):
    monkeypatch.setattr(
        tool_docs.DocsClient, "call_tool", AsyncMock(side_effect=DocsUnavailable("worker offline"))
    )
    result = await tool_middleware(tool_docs.list_doc_files)(library_id="/a/b")
    assert result["status"] == "unavailable"
    assert result["trust"]["evidence_kind"] == "documentation"
    assert "solemd infra up" in result["next_action"]


def test_shedding_preserves_document_identity_and_updates_next_page():
    payload = {
        "status": "success",
        "tool": "list_doc_files",
        "payload": {
            "library_id": "/codeatlas/example",
            "files": [
                {"file_path": f"docs/{i:04}-{'a' * 120}.md", "chunk_count": 4} for i in range(200)
            ],
            "pagination": {
                "offset": 50,
                "total": 500,
                "returned": 200,
                "limit": 200,
                "next_offset": 250,
                "has_more": True,
            },
        },
    }
    result = enforce_response_budget(
        "list_doc_files",
        payload,
        signature=inspect.signature(tool_docs.list_doc_files),
        args=(),
        kwargs={},
    )
    body = result["payload"]
    assert body["library_id"] == "/codeatlas/example"
    assert len(body["files"]) < 200
    assert body["pagination"]["total"] == 500
    assert body["pagination"]["returned"] == len(body["files"])
    assert body["pagination"]["next_offset"] == 50 + len(body["files"])


def test_long_document_has_explicit_truncation_and_retains_path():
    result = enforce_response_budget(
        "read_doc",
        {
            "status": "success",
            "payload": {
                "content": "x" * 200000,
                "path": "docs/api.md",
                "library_id": "/a/b",
                "truncated": False,
            },
        },
        signature=inspect.signature(tool_docs.read_doc),
        args=(),
        kwargs={},
    )
    assert result["payload"]["path"] == "docs/api.md"
    assert result["payload"]["truncated"] is True
    assert len(result["payload"]["content"]) < 200000
