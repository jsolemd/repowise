"""Canonical doc-search MCP dispatch for the docs host."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.types import TextContent

from repowise.docs.runtime import docs_runtime_accepts_mutations, ensure_docs_runtime
from repowise.docs.server.response import respond, respond_error
from repowise.docs.tools.common import ToolError
from repowise.docs.tools.documents import handle_list_documents
from repowise.docs.tools.handlers import (
    handle_add_library,
    handle_delete_library,
    handle_expand_chunk,
    handle_export_bundle,
    handle_import_bundle,
    handle_list_libraries,
    handle_query_docs,
    handle_query_docs_multi,
    handle_read_doc,
    handle_resolve_library_id,
    handle_update_library,
)

DocHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


_TOOL_HANDLERS: dict[str, DocHandler] = {
    "list_doc_files": handle_list_documents,
    "resolve_library_id": handle_resolve_library_id,
    "search_docs": handle_query_docs,
    "search_docs_multi": handle_query_docs_multi,
    "expand_doc_chunk": handle_expand_chunk,
    "list_doc_libraries": handle_list_libraries,
    "read_doc": handle_read_doc,
    "update_doc_library": handle_update_library,
    "add_doc_library": handle_add_library,
    "delete_doc_library": handle_delete_library,
    "export_doc_bundle": handle_export_bundle,
    "import_doc_bundle": handle_import_bundle,
}

_BACKGROUND_REQUIRED_TOOLS = frozenset({"add_doc_library", "update_doc_library"})

#: The four library mutations. Served natively by this process; the readyz
#: payload advertises this set so the served surface and the reported surface
#: cannot drift.
MUTATION_TOOL_NAMES = frozenset(
    {
        "add_doc_library",
        "delete_doc_library",
        "import_doc_bundle",
        "update_doc_library",
    }
)


def _translate_arguments(name: str, arguments: dict[str, Any], output: str) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "output":
            continue
        translated[key] = value

    if name in {"search_docs", "search_docs_multi", "read_doc"}:
        translated.setdefault("format", output)
    return translated


def _format_result(result: dict[str, Any], output: str) -> str:
    if output == "markdown":
        content = result.get("content")
        if isinstance(content, str):
            return content
    return json.dumps(result, separators=(",", ":"), ensure_ascii=False, default=str)


def _next_action(name: str, result: dict[str, Any]) -> str:
    if name == "resolve_library_id":
        return (
            "Use `search_docs` with the returned `library_id`."
            if result.get("library_id")
            else "Retry with a more specific name or use `list_doc_libraries`."
        )
    if name == "search_docs":
        payload_results = result.get("results") or []
        if result.get("recommended_tool") == "read_doc" and result.get("recommended_start"):
            return "Use `read_doc` on the suggested path for full context."
        if payload_results:
            return "Use `expand_doc_chunk` or `read_doc` on the top hit for more context."
        return "Refine the query, change the library, or call `resolve_library_id` again."
    if name == "search_docs_multi":
        return "Use `search_docs` or `read_doc` on the strongest library-specific hit."
    if name == "expand_doc_chunk":
        return "Use `read_doc` for the full file or section once the chunk is confirmed."
    if name == "list_doc_libraries":
        return "Use `resolve_library_id` or `search_docs` with a ready library."
    if name == "read_doc":
        return "Use `search_docs` or `expand_doc_chunk` to navigate adjacent sections."
    if name in {"update_doc_library", "add_doc_library"}:
        return "Check `list_doc_libraries` or docs health to confirm the queued job completes."
    if name == "delete_doc_library":
        return "Re-run `list_doc_libraries` to confirm the library was removed."
    if name in {"export_doc_bundle", "import_doc_bundle"}:
        return "Use `list_doc_libraries` or `search_docs` to verify the bundle state."
    return "Use the next docs navigation tool on the returned payload."


def _success_confidence(name: str, result: dict[str, Any]) -> str:
    if name == "resolve_library_id":
        try:
            return "high" if float(result.get("confidence", 0.0) or 0.0) >= 0.9 else "medium"
        except (TypeError, ValueError):
            return "medium"
    if name in {"search_docs", "search_docs_multi"}:
        results = result.get("results")
        return "high" if isinstance(results, list) and results else "medium"
    return "high"


async def call_doc_tool(
    ctx: Any,
    name: str,
    arguments: dict[str, Any],
    output: str,
) -> list[TextContent]:
    """Dispatch one docs call against the local, natively served runtime."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return respond_error(
            ctx,
            tool_name=name,
            confidence="high",
            next_action="Use one of the canonical search_docs or library-management tools.",
            output=output,
            code="unknown_tool",
            message=f"Unknown doc tool: {name}",
            include_scope=False,
        )

    # Fold the deprecated search_docs_multi tool into search_docs: a
    # ``library_ids`` array with >1 entry routes to the multi-library handler;
    # a single entry behaves like the normal single-library search_docs path.
    if name == "search_docs" and isinstance(arguments.get("library_ids"), list):
        ids = [str(v).strip() for v in arguments["library_ids"] if str(v).strip()]
        arguments = {k: v for k, v in arguments.items() if k != "library_ids"}
        if len(ids) > 1:
            handler = _TOOL_HANDLERS["search_docs_multi"]
            arguments["library_ids"] = ids
            if "limit" in arguments and "limit_per_library" not in arguments:
                arguments["limit_per_library"] = arguments.pop("limit")
        elif ids:
            arguments.setdefault("library_id", ids[0])

    try:
        await ensure_docs_runtime(start_background=False)
        if name in _BACKGROUND_REQUIRED_TOOLS and not docs_runtime_accepts_mutations():
            return respond_error(
                ctx,
                tool_name=name,
                confidence="high",
                next_action="Wait for the docs worker and scheduler to come up, then retry this mutation.",
                output=output,
                code="docs_runtime_not_ready",
                message=f"{name} requires the docs background worker and scheduler to be running.",
                include_scope=False,
            )
        translated = _translate_arguments(name, arguments, output)
        result = await handler(translated)
        return respond(
            ctx,
            tool_name=name,
            status="success",
            confidence=_success_confidence(name, result),
            next_action=_next_action(name, result),
            body=_format_result(result, output),
            output=output,
            payload=result,
            include_scope=False,
        )
    except ToolError as exc:
        return respond_error(
            ctx,
            tool_name=name,
            confidence="high",
            next_action="Fix the tool arguments or library identifier and retry.",
            output=output,
            code="invalid_arguments",
            message=str(exc),
            include_scope=False,
        )
