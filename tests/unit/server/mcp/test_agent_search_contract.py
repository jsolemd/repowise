"""The source lane must preserve explicit agent search instructions."""

from unittest.mock import AsyncMock

import pytest

from repowise.server.mcp_server import tool_search
from repowise.server.mcp_server._wire import as_call_tool_result


@pytest.mark.parametrize(
    "arguments",
    [
        {"mode": "symbol"},
        {"mode": "hybrid", "kind": "test"},
        {"mode": "hybrid", "symbol_kind": "class"},
        {"mode": "hybrid", "page_type": "module_page"},
    ],
)
async def test_explicit_lookup_contract_uses_native_resolver(monkeypatch, arguments):
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")
    monkeypatch.setattr(tool_search, "_is_workspace_mode", lambda: False)
    native = AsyncMock(return_value={"results": [], "mode": arguments["mode"]})
    source = AsyncMock()
    monkeypatch.setattr(tool_search, "_structured_search", native)
    monkeypatch.setattr(tool_search, "mcp_coordinator", source)
    await tool_search.search_codebase("SourceSearchCoordinator", **arguments)
    native.assert_awaited_once()
    source.assert_not_awaited()
    received = native.call_args.args
    assert received[2] == arguments.get("page_type")
    assert received[3] == arguments.get("kind")
    assert received[4] == arguments.get("symbol_kind")


@pytest.mark.parametrize("query", ["as_list_raw", "SourceSearchCoordinator"])
async def test_an_auto_mode_identifier_still_reaches_the_source_lane(monkeypatch, query):
    """The owner claim an identifier lookup exists to produce.

    ``auto`` resolves a bare identifier to ``symbol``, and a guard that tested
    the *resolved* mode sent it to the native rows — which carry
    ``exact_match`` but no ``selected_owner``, no ``confidence`` and no
    per-row ``evidence``. The caller never asked for the native resolver's
    argument semantics; they asked who owns a name, and the source lane's
    exact-identifier router is what answers that.
    """
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")
    monkeypatch.setattr(tool_search, "_is_workspace_mode", lambda: False)
    native = AsyncMock(return_value={"results": [], "mode": "symbol"})
    owned = {
        "results": [
            {
                "target_path": "conceptatlas/atlas/content.py",
                "symbol_path": "conceptatlas/atlas/content.py::as_list_raw",
                "evidence": {"exact_name": True, "lane": "source"},
            }
        ],
        "mode": "symbol",
        "confidence": "confident",
        "selected_owner": {
            "file": "conceptatlas/atlas/content.py",
            "reason": "exact_name",
        },
    }
    coordinator = AsyncMock()
    coordinator.search = AsyncMock(return_value=owned)
    monkeypatch.setattr(tool_search, "_structured_search", native)
    monkeypatch.setattr(tool_search, "mcp_coordinator", AsyncMock(return_value=coordinator))

    result = await tool_search.search_codebase(query)

    native.assert_not_awaited()
    coordinator.search.assert_awaited_once()
    # The resolved mode still travels: the source lane routes on it too.
    assert coordinator.search.call_args.kwargs["mode"] == "symbol"
    assert result["selected_owner"]["file"] == "conceptatlas/atlas/content.py"
    assert result["confidence"] == "confident"
    assert result["results"][0]["evidence"]


async def test_an_auto_resolved_path_stays_on_the_native_resolver(monkeypatch):
    """A path is a filename lookup, not a retrieval — the rule that predates the split."""
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")
    monkeypatch.setattr(tool_search, "_is_workspace_mode", lambda: False)
    native = AsyncMock(return_value={"results": [], "mode": "path"})
    source = AsyncMock()
    monkeypatch.setattr(tool_search, "_structured_search", native)
    monkeypatch.setattr(tool_search, "mcp_coordinator", source)

    await tool_search.search_codebase("src/repowise/core/foo.py")

    native.assert_awaited_once()
    source.assert_not_awaited()


def test_trust_survives_a_client_that_only_forwards_tool_content():
    import json

    envelope = {
        "contract_version": 1,
        "timing_ms": 27,
        "index_behind": True,
        "source_search": {
            "status": "stale",
            "generation_id": "read-1",
            "published_generation_id": "published-2",
        },
    }
    original = {"results": [], "_meta": envelope}
    wire = as_call_tool_result(original)
    visible = json.loads(wire.content[0].text)
    assert visible == wire.structuredContent
    assert visible["trust"]["index_behind"] is True
    assert visible["trust"]["source_search"]["generation_id"] == "read-1"
    assert "timing_ms" not in visible["trust"]
    assert wire.meta == envelope
    assert original == {"results": [], "_meta": envelope}


def test_existing_index_status_trust_is_preserved():
    wire = as_call_tool_result(
        {
            "trust": {"search_results": "unknown", "reasons": ["missing_index"]},
            "_meta": {"index_behind": True},
        }
    )
    assert wire.structuredContent["trust"] == {
        "search_results": "unknown",
        "reasons": ["missing_index"],
        "index_behind": True,
    }


def test_text_is_compact_json_without_changing_source_bytes():
    import json

    payload = {"source": "def f():\n    return 1\n", "results": [{"name": "f"}]}
    wire = as_call_tool_result(payload)
    text = wire.content[0].text
    assert "\n" not in text
    assert json.loads(text) == payload == wire.structuredContent


def test_federated_trust_drops_diagnostics_but_preserves_unavailable_members():
    from repowise.server.mcp_server._meta import agent_trust

    trust = agent_trust(
        {
            "source_search": {
                "repos": {
                    "a": {
                        "generation_id": "a1",
                        "status": "current",
                        "timing_ms": 300,
                        "symbol_chunks": 80000,
                        "failed_legs": [{"leg": "dense", "code": "timeout"}],
                    },
                    "b": {"unavailable": "source index unavailable"},
                }
            }
        }
    )
    assert trust["source_search"]["repos"] == {
        "a": {
            "generation_id": "a1",
            "status": "current",
            "failed_legs": [{"leg": "dense", "code": "timeout"}],
        },
        "b": {"unavailable": "source index unavailable"},
    }


def test_visible_trust_is_budgeted_and_survives_the_emergency_guard(tmp_path):
    import inspect

    from repowise.server.mcp_server._budget import enforce_response_budget
    from repowise.server.mcp_server._budget.budgeter import response_chars
    from repowise.server.mcp_server._meta import finalize_trust_envelope

    payload = finalize_trust_envelope(
        {
            "kpis": {"file_count": 2},
            "unexpected_large_block": "x" * 100000,
            "_meta": {"index_behind": True, "source_search": {"status": "stale"}},
        }
    )
    assert payload["trust"]["index_behind"] is True
    bounded = enforce_response_budget(
        "get_health",
        payload,
        signature=inspect.Signature(),
        args=(),
        kwargs={},
        repo_root=tmp_path,
    )
    budget = bounded["_meta"]["response_budget"]
    assert bounded["trust"] == payload["trust"]
    assert response_chars(bounded) == budget["serialized_chars"] <= budget["limit_chars"]
    wire = as_call_tool_result(bounded)
    assert wire.structuredContent["trust"] == bounded["trust"]
    assert response_chars(wire.structuredContent) <= budget["serialized_chars"]
