"""Canonical doc-search MCP tool definitions, independent of the code host."""

from __future__ import annotations

from mcp.types import Tool

DOC_CHUNK_TYPE_ENUM = ["doc", "code", "config", "changelog", "llms_txt"]


def _output_property(*, default: str = "json") -> dict:
    return {
        "type": "string",
        "description": "Output format",
        "enum": ["markdown", "json"],
        "default": default,
    }


def _json_output_property() -> dict:
    return {
        "type": "string",
        "description": "Output format",
        "enum": ["json"],
        "default": "json",
    }


def _closed_object_schema(*, properties: dict, required: list[str] | None = None) -> dict:
    schema: dict[str, object] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def get_doc_tool_definitions() -> list[Tool]:
    return [
        Tool(
            name="list_doc_files",
            description="Browse indexed documentation files within a library, with a path filter and exact pagination totals.",
            inputSchema=_closed_object_schema(
                properties={
                    "library_id": {"type": "string", "description": "Full indexed library ID."},
                    "query": {
                        "type": "string",
                        "description": "Case-insensitive literal path substring.",
                        "default": "",
                    },
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                    "output": _json_output_property(),
                },
                required=["library_id"],
            ),
        ),
        Tool(
            name="resolve_library_id",
            description="Resolve a library name to its full indexed documentation library ID.",
            inputSchema=_closed_object_schema(
                properties={
                    "library_name": {
                        "type": "string",
                        "description": "Library name or partial ID to resolve.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Deprecated alias for library_name.",
                        "deprecated": True,
                    },
                    "output": _json_output_property(),
                },
                required=["library_name"],
            ),
        ),
        Tool(
            name="search_docs",
            description=(
                "Search indexed external documentation for one library using hybrid retrieval. "
                "Use `exact_match=true` for API/function lookups."
            ),
            inputSchema=_closed_object_schema(
                properties={
                    "library_id": {"type": "string", "description": "Full library ID."},
                    "library_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Search several libraries at once. >1 entry searches all; a single entry equals library_id.",
                        "minItems": 1,
                        "maxItems": 10,
                    },
                    "query": {"type": "string", "description": "Documentation search query."},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return.",
                        "default": 6,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "chunk_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": DOC_CHUNK_TYPE_ENUM},
                        "description": "Optional chunk-type filter.",
                    },
                    "include_code_blocks": {
                        "type": "boolean",
                        "description": "Include adjacent code blocks from the same section.",
                    },
                    "exact_match": {
                        "type": "boolean",
                        "description": "Prefer exact API/name matches over semantic similarity.",
                        "default": False,
                    },
                    "output": _output_property(default="json"),
                },
                required=["query"],
            ),
        ),
        Tool(
            name="search_docs_multi",
            description="Search across multiple indexed documentation libraries at once.",
            inputSchema=_closed_object_schema(
                properties={
                    "library_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of library IDs to search across.",
                        "minItems": 1,
                        "maxItems": 10,
                    },
                    "query": {"type": "string", "description": "Documentation search query."},
                    "limit_per_library": {
                        "type": "integer",
                        "description": "Maximum results per library.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "chunk_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": DOC_CHUNK_TYPE_ENUM},
                        "description": "Optional chunk-type filter.",
                    },
                    "output": _output_property(default="json"),
                },
                required=["library_ids", "query"],
            ),
        ),
        Tool(
            name="expand_doc_chunk",
            description="Expand one documentation chunk with surrounding file context.",
            inputSchema=_closed_object_schema(
                properties={
                    "chunk_id": {"type": "string", "description": "Chunk ID to expand."},
                    "lines_before": {
                        "type": "integer",
                        "description": "Lines of context before the chunk.",
                        "default": 30,
                        "minimum": 0,
                        "maximum": 200,
                    },
                    "lines_after": {
                        "type": "integer",
                        "description": "Lines of context after the chunk.",
                        "default": 30,
                        "minimum": 0,
                        "maximum": 200,
                    },
                    "output": _output_property(default="json"),
                },
                required=["chunk_id"],
            ),
        ),
        Tool(
            name="list_doc_libraries",
            description="List indexed documentation libraries and their status. Paginated: pass `offset` + `limit`; `next_offset` in the response points to the next page. Pass `filter` to substring-match against library_id, name, description, or repo (case-insensitive) — useful when the registry has dozens of libraries (e.g. `filter='gsap'`).",
            inputSchema=_closed_object_schema(
                properties={
                    "include_stats": {
                        "type": "boolean",
                        "description": "Include chunk/file/index statistics.",
                        "default": True,
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "indexing", "ready", "error"],
                        "description": "Optional library-status filter.",
                    },
                    "filter": {
                        "type": "string",
                        "description": "Case-insensitive substring matched against library_id, name, description, or repo.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Zero-based offset into the ordered result list.",
                        "default": 0,
                        "minimum": 0,
                        "maximum": 10000,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max libraries to return on this page (omit for all).",
                        "minimum": 1,
                        "maximum": 200,
                    },
                    "output": _json_output_property(),
                },
            ),
        ),
        Tool(
            name="read_doc",
            description="Read a full indexed documentation file or one anchored section.",
            inputSchema=_closed_object_schema(
                properties={
                    "library_id": {"type": "string", "description": "Full library ID."},
                    "path": {"type": "string", "description": "Path within the cached repo."},
                    "file_path": {
                        "type": "string",
                        "description": "Deprecated alias for path.",
                        "deprecated": True,
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional section anchor to extract.",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to return.",
                        "default": 10000,
                        "minimum": 100,
                        "maximum": 50000,
                    },
                    "output": _output_property(default="json"),
                },
                required=["library_id", "path"],
            ),
        ),
        Tool(
            name="update_doc_library",
            description="Queue a docs reindex for one library.",
            inputSchema=_closed_object_schema(
                properties={
                    "library_id": {"type": "string", "description": "Full library ID."},
                    "force": {
                        "type": "boolean",
                        "description": "Force a full reindex instead of incremental update.",
                        "default": False,
                    },
                    "output": _json_output_property(),
                },
                required=["library_id"],
            ),
        ),
        Tool(
            name="add_doc_library",
            description="Register a new documentation library and queue its initial index.",
            inputSchema=_closed_object_schema(
                properties={
                    "repo": {
                        "type": "string",
                        "description": "GitHub repository in owner/repo format.",
                    },
                    "name": {"type": "string", "description": "Human-readable library name."},
                    "description": {
                        "type": "string",
                        "description": "Optional description of the library.",
                    },
                    "docs_path": {
                        "type": "string",
                        "description": "Path to docs within the repository.",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Git branch to index.",
                        "default": "main",
                    },
                    "include_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional include globs.",
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional exclude globs.",
                    },
                    "output": _json_output_property(),
                },
                required=["repo", "name"],
            ),
        ),
        Tool(
            name="delete_doc_library",
            description="Delete an indexed documentation library and all of its stored data.",
            inputSchema=_closed_object_schema(
                properties={
                    "library_id": {"type": "string", "description": "Full library ID."},
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to confirm deletion.",
                    },
                    "output": _json_output_property(),
                },
                required=["library_id", "confirm"],
            ),
        ),
        Tool(
            name="export_doc_bundle",
            description="Export one library's indexed docs as a portable bundle.",
            inputSchema=_closed_object_schema(
                properties={
                    "library_id": {"type": "string", "description": "Full library ID."},
                    "output": _json_output_property(),
                },
                required=["library_id"],
            ),
        ),
        Tool(
            name="import_doc_bundle",
            description="Import a previously exported documentation bundle.",
            inputSchema=_closed_object_schema(
                properties={
                    "bundle_path": {
                        "type": "string",
                        "description": "Path to the .docbundle.json.gz file.",
                    },
                    "output": _json_output_property(),
                },
                required=["bundle_path"],
            ),
        ),
    ]
