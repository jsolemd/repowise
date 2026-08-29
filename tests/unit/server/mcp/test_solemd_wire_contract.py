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


#: The one served tool that changes state — it appends to (and retires entries
#: in) the git-tracked decisions journal. Everything else served answers from
#: the index and the working tree without touching either.
EXPECTED_WRITER_TOOLS = frozenset({"manage_decision"})

#: Hints every served reader must advertise.
READER_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

#: Hints the served writer must advertise: not read-only, but additive (a
#: decision is superseded, never deleted) and not idempotent (a second record
#: appends a second entry).
WRITER_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}


@pytest.mark.asyncio
async def test_served_tools_carry_title_and_annotations_and_wrap_result(
    monkeypatch: pytest.MonkeyPatch,
):
    from repowise.server.mcp_server import ensure_full_surface

    served_names = _served_tool_names(monkeypatch)
    advertised = {tool.name: tool for tool in await ensure_full_surface().list_tools()}
    assert served_names <= advertised.keys()

    missing_titles = sorted(name for name in served_names if not advertised[name].title)
    assert not missing_titles, f"served tools with no tools/list title: {missing_titles}"

    annotation_drift = {}
    for name in sorted(served_names):
        expected = WRITER_ANNOTATIONS if name in EXPECTED_WRITER_TOOLS else READER_ANNOTATIONS
        annotations = advertised[name].annotations
        actual = annotations.model_dump(exclude_none=True) if annotations else None
        if actual != expected:
            annotation_drift[name] = {"expected": expected, "actual": actual}
    assert not annotation_drift, (
        f"served tool annotations drifted from the 24-reader + 1-writer split: {annotation_drift}"
    )
    assert len(served_names - EXPECTED_WRITER_TOOLS) == 24

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


def test_server_info_version_is_the_fork_version():
    """``serverInfo.version`` names repowise, not the SDK that transports it.

    ``FastMCP.__init__`` takes no version, so the low-level ``Server`` keeps
    ``version=None`` and ``create_initialization_options`` falls back to
    ``importlib.metadata.version("mcp")``. Every client used to be told the SDK
    version where the protocol asks for the server's.
    """
    from repowise.server.mcp_server import ensure_full_surface

    options = ensure_full_surface()._mcp_server.create_initialization_options()
    assert options.server_version == version("repowise")
    assert options.server_version != version("mcp")


@pytest.mark.asyncio
async def test_healthz_answers_on_the_streamable_http_app():
    """The HTTP transport carries a plain GET liveness probe.

    MCP itself has no GET endpoint, so without this the only way to ask a
    running server whether it is up is a full ``initialize`` handshake.
    """
    import httpx

    from repowise.server.mcp_server import ensure_full_surface

    mcp = ensure_full_surface()
    app = mcp.streamable_http_app()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:7350") as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == version("repowise")
    assert body["tools"] == len(await mcp.list_tools())
    assert isinstance(body["workspace"], bool)


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
