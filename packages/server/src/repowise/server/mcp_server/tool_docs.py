"""Library documentation tools on the canonical RepoWise MCP surface."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from repowise.core.registry import mcp_tool_registry as mcp
from repowise.docs.client import DocsClient, DocsUnavailable


async def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        result = await DocsClient().call_tool(
            name, {key: value for key, value in arguments.items() if value is not None}
        )
    except DocsUnavailable as exc:
        return {
            "status": "unavailable",
            "error": str(exc),
            "tool": name,
            "next_action": "Start the docs worker with solemd infra up, then retry.",
            "trust": {"evidence_kind": "documentation", "unavailable": True},
        }
    result.setdefault("trust", {})["evidence_kind"] = "documentation"
    return result


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="read_only",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_doc_files(
    library_id: str,
    query: str = "",
    offset: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1, le=200)] = 50,
    output: Literal["json"] = "json",
) -> dict[str, Any]:
    """Browse indexed file paths with a filter and exact pagination totals."""
    return await _call("list_doc_files", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="read_only",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def resolve_library_id(
    library_name: str,
    query: str | None = None,
    output: Literal["json"] = "json",
) -> dict[str, Any]:
    """Resolve a library name to its full indexed documentation library ID."""
    return await _call("resolve_library_id", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="read_only",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def search_docs(
    query: str,
    library_id: str | None = None,
    library_ids: Annotated[list[str], Field(min_length=1, max_length=10)] | None = None,
    limit: Annotated[int, Field(ge=1, le=50)] = 6,
    chunk_types: list[str] | None = None,
    include_code_blocks: bool | None = None,
    exact_match: bool = False,
    output: Literal["markdown", "json"] = "json",
) -> dict[str, Any]:
    """Search indexed external documentation in one or several libraries.

    Use exact_match=True for API or function lookups."""
    return await _call("search_docs", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="read_only",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def expand_doc_chunk(
    chunk_id: str,
    lines_before: Annotated[int, Field(ge=0, le=200)] = 30,
    lines_after: Annotated[int, Field(ge=0, le=200)] = 30,
    output: Literal["markdown", "json"] = "json",
) -> dict[str, Any]:
    """Expand one documentation chunk with surrounding file context."""
    return await _call("expand_doc_chunk", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="read_only",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_doc_libraries(
    include_stats: bool = True,
    status: Literal["pending", "indexing", "ready", "error"] | None = None,
    filter: str | None = None,
    offset: Annotated[int, Field(ge=0, le=10000)] = 0,
    limit: Annotated[int, Field(ge=1, le=200)] | None = None,
    output: Literal["json"] = "json",
) -> dict[str, Any]:
    """List indexed libraries, their status, and coverage.

    Use offset and limit for pagination, or filter to match a name, repo, or description.
    """
    return await _call("list_doc_libraries", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="read_only",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def read_doc(
    library_id: str,
    path: str,
    file_path: str | None = None,
    section: str | None = None,
    max_tokens: Annotated[int, Field(ge=100, le=50000)] = 10000,
    output: Literal["markdown", "json"] = "json",
) -> dict[str, Any]:
    """Read a full indexed documentation file or one anchored section."""
    return await _call("read_doc", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="mutating",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def update_doc_library(
    library_id: str,
    force: bool = False,
    output: Literal["json"] = "json",
) -> dict[str, Any]:
    """Queue a docs reindex for one library."""
    return await _call("update_doc_library", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="mutating",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def add_doc_library(
    repo: str,
    name: str,
    description: str | None = None,
    docs_path: str | None = None,
    branch: str = "main",
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    output: Literal["json"] = "json",
) -> dict[str, Any]:
    """Register a new documentation library and queue its initial index."""
    return await _call("add_doc_library", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="mutating",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def delete_doc_library(
    library_id: str,
    confirm: bool,
    output: Literal["json"] = "json",
) -> dict[str, Any]:
    """Delete an indexed documentation library and all of its stored data."""
    return await _call("delete_doc_library", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="mutating",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def export_doc_bundle(
    library_id: str,
    output: Literal["json"] = "json",
) -> dict[str, Any]:
    """Export one library's indexed docs as a portable bundle."""
    return await _call("export_doc_bundle", locals())


@mcp.tool(
    default=False,
    trust_kind="documentation",
    safety="mutating",
    annotations={
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def import_doc_bundle(
    bundle_path: str,
    output: Literal["json"] = "json",
) -> dict[str, Any]:
    """Import a previously exported documentation bundle."""
    return await _call("import_doc_bundle", locals())
