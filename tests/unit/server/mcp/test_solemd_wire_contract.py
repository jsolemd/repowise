"""Pin the MCP wire contract consumed by SoleMD's RepoWise integration.

These assertions intentionally describe today's transport, including behavior
that later formalization units will change.  A wire change and its new pins
must land together so an upstream absorption cannot silently move consumers.
"""

from __future__ import annotations

from importlib.metadata import version

import pytest

from repowise.core.registry import mcp_tool_registry
from repowise.server.mcp_server._tool_selection import resolve_enabled_tools

# Deliberately duplicated from
# SoleMD.Infra/infra/repowise/systemd/repowise-mcp.service.  Keeping the fork
# test hermetic makes the two repositories independent at test time; the exact
# duplicate is the pin that forces a coordinated edit when the served surface
# changes.
SOLEMD_TOOL_OPT_INS = (
    "+get_dependents,+get_dependency_path,+get_execution_flows,+build_task_slice,"
    "+get_task_slice,+extend_task_slice,+find_clones,+find_patterns,+manage_decision,"
    "+get_reference_sites,+preview_symbol_rename,+get_query_quality,+get_architecture,"
    "+get_blast_radius"
)

EXPECTED_SERVED_TOOLS = frozenset(
    {
        "build_task_slice",
        "extend_task_slice",
        "find_clones",
        "find_patterns",
        "get_architecture",
        "get_blast_radius",
        "get_change_risk",
        "get_context",
        "get_dead_code",
        "get_dependency_path",
        "get_dependents",
        "get_execution_flows",
        "get_health",
        "get_index_status",
        "get_overview",
        "get_query_quality",
        "get_reference_sites",
        "get_risk",
        "get_symbol",
        "get_task_slice",
        "get_why",
        "list_repos",
        "manage_decision",
        "preview_symbol_rename",
        "search_codebase",
    }
)


def _served_tool_names(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    monkeypatch.setenv("REPOWISE_TOOLS_NO_GENERATIVE", "1")

    # Tool modules register lazily, so resolve only after the full in-process
    # registry has been populated.
    from repowise.server.mcp_server import ensure_full_surface

    ensure_full_surface()
    return resolve_enabled_tools(
        mcp_tool_registry.entries(),
        is_workspace=True,
        override=SOLEMD_TOOL_OPT_INS,
    )


def test_solemd_served_tool_names_are_exact(monkeypatch: pytest.MonkeyPatch):
    actual = _served_tool_names(monkeypatch)
    assert actual == EXPECTED_SERVED_TOOLS, (
        f"served tool drift: missing={sorted(EXPECTED_SERVED_TOOLS - actual)}, "
        f"extra={sorted(actual - EXPECTED_SERVED_TOOLS)}"
    )


@pytest.mark.asyncio
async def test_served_tools_have_no_title_or_annotations_and_wrap_result(
    monkeypatch: pytest.MonkeyPatch,
):
    from repowise.server.mcp_server import ensure_full_surface

    served_names = _served_tool_names(monkeypatch)
    advertised = {tool.name: tool for tool in await ensure_full_surface().list_tools()}
    assert served_names <= advertised.keys()

    metadata_drift = {
        name: {"title": advertised[name].title, "annotations": advertised[name].annotations}
        for name in sorted(served_names)
        if advertised[name].title is not None or advertised[name].annotations is not None
    }
    assert not metadata_drift, (
        "served tools gained title/annotations; unit 4.1 must change these pins "
        f"with the wire contract: {metadata_drift}"
    )

    wrapper_drift = {
        name: schema
        for name in sorted(served_names)
        if (schema := advertised[name].outputSchema) is None
        or set(schema.get("properties", {})) != {"result"}
        or schema.get("required") != ["result"]
    }
    assert not wrapper_drift, (
        "served tools no longer use the structuredContent {'result': ...} wrapper; "
        f"unit 4.2 must change these pins with the transport: {wrapper_drift}"
    )


def test_server_info_version_is_the_mcp_sdk_version():
    from repowise.server.mcp_server import ensure_full_surface

    options = ensure_full_surface()._mcp_server.create_initialization_options()
    assert options.server_version == version("mcp")
    # Unit 4.1 deliberately changes serverInfo.version to the fork version.
    assert options.server_version != version("repowise")


def test_search_symbol_rows_keep_the_solemd_identity_keys():
    from repowise.core.source_search.coordinator import _Item

    row = _Item(
        key="src/example.py::Outer::inner",
        lane="source",
        file="src/example.py",
        name="inner",
        kind="function",
        snippet="def inner(): ...",
        source="symbol",
        target_id="src/example.py::Outer::inner",
        contains=("Outer::inner::local",),
    ).to_result()

    assert row["symbol_path"] == "Outer::inner"
    assert row["contains_symbols"] == ["Outer::inner::local"]


def test_degraded_search_meta_keeps_the_solemd_disclosure_keys():
    from repowise.server.mcp_server.tool_search import (
        _begin_retrieval_record,
        _record_vector_leg,
        _retrieval_disclosure,
    )

    _begin_retrieval_record()
    _record_vector_leg("error", "wire-contract probe")
    disclosure = _retrieval_disclosure()

    assert disclosure["retrieval_degraded"] == ["vector"]
    assert "wire-contract probe" in disclosure["retrieval_degraded_reason"]


def test_federated_meta_keeps_repo_freshness():
    from repowise.server.mcp_server._meta import federated_freshness

    freshness = federated_freshness([("infra", None, None)])
    assert freshness["repo_freshness"] == {"infra": {}}


def test_source_manifest_round_trips_working_tree_ingest():
    from repowise.core.source_search.manifest import EmbedderIdentity, SourceIndexManifest

    manifest = SourceIndexManifest(
        recipe_fingerprint="recipe",
        corpus_hash="corpus",
        symbol_chunks=1,
        file_window_chunks=1,
        files_covered=1,
        indexed_commit="a" * 40,
        built_at="2026-08-29T00:00:00Z",
        embedder=EmbedderIdentity(provider="ollama", model="embeddinggemma", dims=768),
        working_tree_ingest={"src/example.py": "content-hash"},
    )

    encoded = manifest.to_dict()
    assert encoded["working_tree_ingest"] == {"src/example.py": "content-hash"}
    assert SourceIndexManifest.from_dict(encoded).working_tree_ingest == {
        "src/example.py": "content-hash"
    }
