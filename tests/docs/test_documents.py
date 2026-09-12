"""File browsing reads a bounded, consistent metadata page for every source type."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from repowise.docs.tools import documents
from repowise.docs.tools.common import ToolError


@pytest.mark.parametrize("source_type", ["git", "snapshot"])
async def test_pagination_preserves_identity_and_exact_total(source_type):
    library = SimpleNamespace(
        name="Example",
        repo="a/b",
        branch="main",
        source_type=source_type,
        status="ready",
        current_sha="abc",
        indexed_at=None,
    )
    lookup = AsyncMock(return_value=library)
    page = AsyncMock(
        return_value=([{"file_path": "docs/API.md", "chunk_count": 2, "indexed_at": None}], 101)
    )
    result = await documents.handle_list_documents(
        {"library_id": "/codeatlas/example", "query": "API", "offset": 50, "limit": 50},
        get_library_fn=lookup,
        list_page_fn=page,
    )
    assert result["library_id"] == "/codeatlas/example"
    assert result["library"]["source_type"] == source_type
    assert result["pagination"] == {
        "total": 101,
        "offset": 50,
        "limit": 50,
        "returned": 1,
        "has_more": True,
        "next_offset": 51,
    }
    page.assert_awaited_once_with("/codeatlas/example", "API", 50, 50)


async def test_missing_library_does_not_query_files():
    page = AsyncMock()
    with pytest.raises(ToolError, match="Library not found"):
        await documents.handle_list_documents(
            {"library_id": "/missing/lib"},
            get_library_fn=AsyncMock(return_value=None),
            list_page_fn=page,
        )
    page.assert_not_awaited()


async def test_page_uses_one_readonly_snapshot_and_literal_parameter_filter(monkeypatch):
    calls = []

    @asynccontextmanager
    async def transaction(**kwargs):
        calls.append(kwargs)
        yield

    conn = SimpleNamespace(
        transaction=transaction,
        fetchval=AsyncMock(return_value=2),
        fetch=AsyncMock(return_value=[]),
    )

    @asynccontextmanager
    async def connection():
        yield conn

    monkeypatch.setattr(documents, "get_connection", connection)
    assert await documents.list_document_page("/a/b", "%_' OR 1=1", 50, 10) == ([], 2)
    assert calls == [{"isolation": "repeatable_read", "readonly": True}]
    sql, *params = conn.fetch.call_args.args
    assert "ORDER BY file_path LIMIT $3 OFFSET $4" in sql
    assert "%_' OR 1=1" not in sql
    assert params == ["/a/b", "%_' OR 1=1", 10, 50]
    assert conn.fetchval.call_args.args[1:] == ("/a/b", "%_' OR 1=1")
