"""Bounded documentation tool executor shared by RepoWise transports."""

from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from mcp.types import TextContent, Tool

from repowise.docs.catalog import get_doc_tool_definitions
from repowise.docs.config import Settings, get_settings
from repowise.docs.server.mcp_tools import call_doc_tool
from repowise.docs.server.response import output_mode, respond_error
from repowise.docs.server.validation import validate_tool_arguments

logger = logging.getLogger("doc-search-mcp")

HIDDEN_DOC_TOOL_NAMES = frozenset({"search_docs_multi"})


def get_public_doc_tool_definitions() -> list[Tool]:
    """Return the ten advertised tools; deprecated aliases remain callable."""
    return [
        tool for tool in get_doc_tool_definitions() if str(tool.name) not in HIDDEN_DOC_TOOL_NAMES
    ]


def get_doc_tool_schema_by_name() -> dict[str, dict]:
    """Return schemas for advertised tools and hidden compatibility aliases."""
    return {str(tool.name): dict(tool.inputSchema or {}) for tool in get_doc_tool_definitions()}


class DocToolExecutor:
    """Validate, bound, and dispatch docs tools with one byte-stable envelope."""

    def __init__(
        self,
        *,
        ctx: Any | None = None,
        settings: Settings | Any | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()
        self.ctx = ctx or SimpleNamespace(
            settings=SimpleNamespace(project="solemd.infra"),
            tool_semaphore=asyncio.Semaphore(
                int(getattr(resolved_settings, "max_concurrent_tool_calls", 4))
            ),
        )
        self.settings = resolved_settings
        self.schemas = get_doc_tool_schema_by_name()

    def _timeout_seconds(self, tool_name: str) -> float:
        if tool_name in {"search_docs", "search_docs_multi", "read_doc"}:
            return float(getattr(self.settings, "tool_timeout_search_seconds", 20.0))
        return float(getattr(self.settings, "tool_timeout_seconds", 30.0))

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any] | object | None,
    ) -> list[TextContent]:
        """Execute one call through the canonical docs validation boundary."""
        request_id = uuid4().hex[:8]
        started = time.perf_counter()
        arguments_obj: object = arguments if arguments is not None else {}
        output = output_mode(
            self.schemas,
            name,
            arguments_obj if isinstance(arguments_obj, dict) else None,
        )

        if name not in self.schemas:
            return respond_error(
                self.ctx,
                tool_name=name,
                confidence="high",
                next_action="Use one of the documentation tools listed in allowed_tools.",
                output=output,
                code="unknown_tool",
                message=f"Unknown doc tool: {name}",
                details={
                    "tool": name,
                    "allowed_tools": sorted(self.schemas),
                },
                include_scope=False,
            )

        validation_details = validate_tool_arguments(
            self.schemas,
            name,
            arguments_obj,
        )
        if validation_details is not None:
            return respond_error(
                self.ctx,
                tool_name=name,
                confidence="high",
                next_action="Fix tool arguments to match schema and retry.",
                output=output,
                code="invalid_arguments",
                message=f"Invalid arguments for tool '{name}'.",
                details=validation_details,
                include_scope=False,
            )

        assert isinstance(arguments_obj, dict)
        timeout = self._timeout_seconds(name)
        try:
            async with asyncio.timeout(timeout):
                async with self.ctx.tool_semaphore:
                    return await call_doc_tool(
                        self.ctx,
                        name,
                        arguments_obj,
                        output,
                    )
        except TimeoutError:
            logger.warning(
                "[%s] Doc tool %s timed out after %.0fs",
                request_id,
                name,
                timeout,
            )
            return respond_error(
                self.ctx,
                tool_name=name,
                confidence="low",
                next_action=(
                    "Retry with a narrower library/query scope or check docs backend health."
                ),
                output=output,
                code="timeout",
                message=f"tool '{name}' timed out after {timeout:.0f}s.",
                context={"timeout_seconds": timeout, "backend": "docs"},
                include_scope=False,
            )
        except Exception as exc:
            logger.exception("[%s] Error in doc tool %s", request_id, name)
            return respond_error(
                self.ctx,
                tool_name=name,
                confidence="low",
                next_action=(
                    "Retry with narrower inputs; if persistent, inspect docs runtime health."
                ),
                output=output,
                code="internal_error",
                message=str(exc),
                context={"backend": "docs"},
                include_scope=False,
            )
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("[%s] Doc tool %s finished in %.0fms", request_id, name, elapsed_ms)
