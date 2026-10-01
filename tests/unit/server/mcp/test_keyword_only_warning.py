"""A response whose semantic lane did not run leads with a top-level warning."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

from repowise.core.providers.embedding.outage import EmbedderVerdict, note_keyword_only
from repowise.server.mcp_server import tool_docs
from repowise.server.mcp_server._wire import as_call_tool_result, wire

_WARNING = "Keyword-only results: semantic search did not run because the embedding server failed."


async def test_a_failure_noted_deep_in_the_request_leads_the_response() -> None:
    # The embedder notes the failure from inside tasks the tool spawns, and
    # the tool itself may swallow the error and return keyword hits.
    async def search(query: str) -> dict[str, Any]:
        async def embed() -> None:
            note_keyword_only(_WARNING)

        await asyncio.gather(embed(), embed())
        return {"results": [{"id": 1}], "_meta": {"timing_ms": 3}}

    result = await wire(search)(query="parse config")

    assert next(iter(result.structuredContent)) == "warning"
    assert result.structuredContent["warning"] == _WARNING
    assert json.loads(result.content[0].text)["warning"] == _WARNING
    assert result.content[0].text.startswith('{"warning":')


async def test_a_whole_response_carries_no_warning() -> None:
    async def search(query: str) -> dict[str, Any]:
        return {"results": [{"id": 1}], "_meta": {"timing_ms": 3}}

    result = await wire(search)(query="parse config")
    assert "warning" not in result.structuredContent


def test_a_vector_leg_that_failed_for_another_reason_is_read_from_the_envelope() -> None:
    reason = "Semantic (vector) retrieval did not run for this query (timeout)."
    result = as_call_tool_result(
        {
            "results": [],
            "_meta": {"retrieval_degraded": ["vector"], "retrieval_degraded_reason": reason},
        }
    )
    assert next(iter(result.structuredContent)) == "warning"
    assert result.structuredContent["warning"] == reason


def test_a_tool_warning_follows_the_keyword_only_one() -> None:
    result = as_call_tool_result({"warning": "No counted file changes."}, keyword_only=[_WARNING])
    assert result.structuredContent["warning"] == f"{_WARNING} No counted file changes."


_KEYWORD_ONLY = {
    "warning": "Keyword-only results: the worker's own account.",
    "keyword_only": "ConnectError: [Errno 2] No such file or directory",
    "results": [{"chunk_id": "bm25-hit"}],
    "search_mode": "keyword_only",
}
_HYBRID = {"results": [{"chunk_id": "hybrid-hit"}], "search_mode": "hybrid"}


async def test_docs_search_retries_once_the_host_has_the_embedder_back(monkeypatch) -> None:
    call = AsyncMock(side_effect=[dict(_KEYWORD_ONLY), dict(_HYBRID)])
    monkeypatch.setattr(tool_docs.DocsClient, "call_tool", call)
    monkeypatch.setattr(
        tool_docs, "ensure_embedder", AsyncMock(return_value=EmbedderVerdict("serving"))
    )

    result = await wire(tool_docs.search_docs)(query="observe", library_id="/a/b")

    assert result.structuredContent["results"] == [{"chunk_id": "hybrid-hit"}]
    assert "warning" not in result.structuredContent
    assert call.await_count == 2


async def test_docs_search_in_gaming_mode_leads_with_the_hosts_verdict(monkeypatch) -> None:
    monkeypatch.setattr(
        tool_docs.DocsClient, "call_tool", AsyncMock(return_value=dict(_KEYWORD_ONLY))
    )
    verdict = EmbedderVerdict(
        "off_by_design", detail="gaming", remedy="solemd gaming resume", gaming=True
    )
    monkeypatch.setattr(tool_docs, "ensure_embedder", AsyncMock(return_value=verdict))

    result = await wire(tool_docs.search_docs)(query="observe", library_id="/a/b")

    warning = result.structuredContent["warning"]
    assert next(iter(result.structuredContent)) == "warning"
    assert "gaming mode" in warning and "solemd gaming resume" in warning
    assert "the worker's own account" not in warning
    assert result.structuredContent["results"] == [{"chunk_id": "bm25-hit"}]
