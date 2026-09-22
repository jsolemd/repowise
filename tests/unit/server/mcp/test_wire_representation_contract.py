"""Pin the fork's flat MCP transport across every registered tool.

Upstream v0.52 pins an accidental SDK ``{result: payload}`` envelope caused by
unevaluated signatures. This fork already evaluates signatures and registers
``served_tool``: text and structured content carry the same flat answer,
protocol metadata carries diagnostics, and visible trust survives hosts that
discard protocol metadata. Test the actual registered callable and SDK seam.
"""

from __future__ import annotations

import inspect
import json
from typing import Annotated, get_args, get_origin

from mcp.server.fastmcp.utilities.func_metadata import func_metadata
from mcp.types import CallToolResult

from repowise.core.registry import mcp_tool_registry

# Full registry, before the separate production selection gate chooses 36.
EXPECTED_TOOL_COUNT = 41
SAMPLE_SCOPE = {
    "projection": "compact",
    "status": "degraded",
    "fingerprint": "partial-index",
    "full": "get_overview(include=['index_scope'])",
    "degraded_analyses": ["git_history"],
}
SAMPLE_PAYLOAD = {
    "answer": "resolved in one call",
    "citations": ["packages/core/src/repowise/core/ingestion/parser.py"],
    "rows": [{"path": "a/b.py", "lines": [10, 20]}],
    "_meta": {
        "timing_ms": 1.5,
        "index_behind": True,
        "retrieval_degraded": True,
        "index_scope": SAMPLE_SCOPE,
    },
}
EXPECTED_FLAT = {
    "answer": SAMPLE_PAYLOAD["answer"],
    "citations": SAMPLE_PAYLOAD["citations"],
    "rows": SAMPLE_PAYLOAD["rows"],
    "trust": {
        "index_behind": True,
        "retrieval_degraded": True,
        "index_scope": SAMPLE_SCOPE,
    },
}


def _registry_entries():
    from repowise.server.mcp_server import ensure_full_surface

    ensure_full_surface()
    return sorted(mcp_tool_registry.tools(), key=lambda fn: fn.__name__)


def _registered_callables():
    from repowise.server.mcp_server import served_tool

    return [served_tool(fn) for fn in _registry_entries()]


def _convert(fn):
    from repowise.server.mcp_server._wire import as_call_tool_result

    # The outer served wrapper constructs this protocol object; FastMCP then
    # passes it through using the actual served callable's output schema.
    metadata = func_metadata(fn)
    result = metadata.convert_result(as_call_tool_result(dict(SAMPLE_PAYLOAD)))
    assert isinstance(result, CallToolResult), fn.__name__
    assert metadata.output_schema is not None, fn.__name__
    return result


def test_the_whole_registered_surface_is_under_test():
    names = [fn.__name__ for fn in _registry_entries()]
    assert len(names) == EXPECTED_TOOL_COUNT, names
    assert len(set(names)) == len(names)


def test_every_tool_serves_a_readable_text_representation():
    for fn in _registered_callables():
        result = _convert(fn)
        assert len(result.content) == 1, fn.__name__
        assert json.loads(result.content[0].text) == EXPECTED_FLAT, fn.__name__
        assert result.meta == SAMPLE_PAYLOAD["_meta"], fn.__name__


def test_every_tool_serves_flat_structured_content_with_visible_trust():
    for fn in _registered_callables():
        result = _convert(fn)
        assert result.structuredContent == EXPECTED_FLAT, fn.__name__
        assert result.structuredContent == json.loads(result.content[0].text), fn.__name__
        assert "result" not in result.structuredContent, fn.__name__
        assert "_meta" not in result.structuredContent, fn.__name__
        # Dropping protocol metadata cannot hide stale/degraded coverage.
        assert result.structuredContent["trust"]["index_scope"] == SAMPLE_SCOPE


def test_every_served_signature_advertises_an_evaluated_flat_payload():
    for fn in _registered_callables():
        annotation = inspect.signature(fn, eval_str=True).return_annotation
        assert get_origin(annotation) is Annotated, fn.__name__
        protocol, payload = get_args(annotation)
        assert protocol is CallToolResult, fn.__name__
        assert get_origin(payload) is dict, fn.__name__
        assert get_args(payload)[0] is str, fn.__name__
        schema = func_metadata(fn).output_schema
        assert schema["type"] == "object", fn.__name__
        assert "result" not in schema.get("properties", {}), fn.__name__
