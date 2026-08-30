"""Pin the MCP wire contract consumed by SoleMD's RepoWise integration.

These assertions intentionally describe today's transport, including behavior
that later formalization units will change.  A wire change and its new pins
must land together so an upstream absorption cannot silently move consumers.
"""

from __future__ import annotations

import json
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


@pytest.fixture(autouse=True)
def _restore_full_surface():
    """Undo the bind-time generative purge this module performs.

    ``_served_tool_names`` sets ``REPOWISE_TOOLS_NO_GENERATIVE`` and calls
    ``ensure_full_surface``, and the ``snapshot_full_surface`` step inside it
    deletes the generative tools from the *process-wide* FastMCP singleton —
    and, if this is the first caller in the process, from the ``_full_surface``
    snapshot that every later ``apply_tool_selection`` rebuilds from.
    ``monkeypatch`` puts the env var back at teardown; nothing puts the tools
    back, and ``snapshot_full_surface`` keeps its first non-empty snapshot for
    the life of the process, so the removal outlives this module.

    A later test that compares what the server serves against the surface
    guide then sees ``get_answer`` on one side only —
    ``test_tool_selection.py::test_apply_trims_and_restores_live_server``
    passes alone and fails once this module has run.
    ``test_no_generative_surface.py`` sidesteps the same trap with the same
    save/restore in its ``rebind`` fixture.

    Binding here, before the env var is set, also fixes ``_full_surface`` from
    an unpurged surface, so the "first snapshot wins" rule captures every tool
    rather than whichever subset the first caller's environment allowed.
    """
    from repowise.server.mcp_server import _tool_selection, ensure_full_surface

    mcp = ensure_full_surface()
    saved_snapshot = _tool_selection._full_surface
    saved_tools = dict(mcp._tool_manager._tools)
    try:
        yield
    finally:
        _tool_selection._full_surface = saved_snapshot
        mcp._tool_manager._tools = dict(saved_tools)


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
async def test_served_tools_carry_title_and_annotations_and_a_flat_output_schema(
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

    schema_drift = {
        name: schema
        for name in sorted(served_names)
        if (schema := advertised[name].outputSchema) is None
        or schema.get("type") != "object"
        or "result" in schema.get("properties", {})
        or schema.get("required")
    }
    assert not schema_drift, (
        "a served tool's outputSchema is not the flat payload's. Unit 4.2 removed the "
        "structuredContent {'result': ...} wrapper, nothing may reintroduce it, and no "
        f"served tool may go without an output schema: {schema_drift}"
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
    assert body["workspace"] is False


@pytest.mark.asyncio
async def test_healthz_reports_workspace_mode_before_any_session(tmp_path, monkeypatch):
    """``workspace`` is read from the repo path, not from lifespan state.

    On the HTTP transports FastMCP enters the server lifespan when an MCP
    session initializes, not when the ASGI app starts, so ``_state``'s
    ``_workspace_root`` is still ``None`` for every probe that lands before the
    first client connects — which is exactly when a startup probe runs.
    """
    import httpx

    from repowise.server.mcp_server import _state, ensure_full_surface

    (tmp_path / ".repowise-workspace.yaml").write_text("repos: []\n", encoding="utf-8")
    monkeypatch.setattr(_state, "_repo_path", str(tmp_path))
    monkeypatch.setattr(_state, "_workspace_root", None)

    app = ensure_full_surface().streamable_http_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:7350") as client:
        response = await client.get("/healthz")

    assert response.json()["workspace"] is True


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


# ---------------------------------------------------------------------------
# Unit 4.2: the wire shape itself, read off raw JSON-RPC frames.
# ---------------------------------------------------------------------------


async def _wire_session(tool: str, arguments: dict) -> tuple[dict, dict]:
    """One transport session: the ``tools/list`` and ``tools/call`` frames.

    No SDK client — a client library re-shapes exactly the fields under test,
    so the JSON-RPC frames are read as they go out. Two constraints shape this:
    ``ASGITransport`` does not run the app's lifespan and the streamable-HTTP
    session manager starts there, so the lifespan is entered by hand; and that
    manager refuses a second ``run()``, so both calls share one session.
    """
    import httpx

    from repowise.server.mcp_server import ensure_full_surface

    app = ensure_full_surface().streamable_http_app()
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }

    def _frame(response: httpx.Response) -> dict:
        assert response.status_code == 200, response.text
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:") :].strip())
        raise AssertionError(f"no JSON-RPC frame in transport reply: {response.text[:400]}")

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        # The Host header is validated, so the base URL has to look like the bind.
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:7350"
        ) as client:
            opened = await client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "solemd-wire-contract", "version": "0"},
                    },
                },
            )
            assert opened.status_code == 200, opened.text
            if session := opened.headers.get("mcp-session-id"):
                headers = {**headers, "mcp-session-id": session}
            await client.post(
                "/mcp",
                headers=headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            listed = _frame(
                await client.post(
                    "/mcp",
                    headers=headers,
                    json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                )
            )
            called = _frame(
                await client.post(
                    "/mcp",
                    headers=headers,
                    json={
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": tool, "arguments": arguments},
                    },
                )
            )
    return listed["result"], called["result"]


@pytest.mark.asyncio
async def test_the_served_wire_shape_over_a_real_transport(monkeypatch: pytest.MonkeyPatch):
    """What a client actually receives, read off raw JSON-RPC frames.

    ``get_index_status`` with nothing wired is the cheapest served tool that
    still answers: the failure shield shapes "no index yet" into a payload
    carrying the same trust envelope as any other response. Shape is what is
    pinned here, never content.

    ``test_served_tools_carry_title_and_annotations_and_a_flat_output_schema``
    reads the schemas off the server object; this reads them off the wire,
    because the schema a client validates against is the serialised one.
    """
    from repowise.server.mcp_server._meta import MCP_CONTRACT_VERSION

    served = _served_tool_names(monkeypatch)
    listed, result = await _wire_session("get_index_status", {})

    schemas = {tool["name"]: tool.get("outputSchema") for tool in listed["tools"]}
    wrapped = {
        name: schema
        for name, schema in schemas.items()
        if name in served and (schema is None or "result" in schema.get("properties", {}))
    }
    assert not wrapped, f"served tools still advertise the result wrapper on the wire: {wrapped}"

    structured = result["structuredContent"]
    assert "result" not in structured, (
        f"structuredContent is still wrapped in {{'result': ...}}: {sorted(structured)}"
    )
    assert "_meta" not in structured, (
        "the trust envelope is still a payload key; it belongs on the protocol result"
    )
    assert result["_meta"]["contract_version"] == MCP_CONTRACT_VERSION, (
        f"no trust envelope on the JSON-RPC result: {sorted(result)}"
    )
    # One copy of the envelope on the wire, so no client can read a second.
    assert json.loads(result["content"][0]["text"]) == structured


@pytest.mark.asyncio
async def test_the_tool_function_itself_still_carries_meta_in_its_payload(setup_mcp):
    """The promotion is wire-only, and this is the half that must not move.

    ``repowise search`` awaits ``search_codebase`` in process
    (``cli/tool_bridge.py``) and reads ``payload["_meta"]`` to say when an
    answer came back lexical-only. That path calls the registered *function*,
    which the MCP middleware never touches, so the envelope has to stay where
    it is on the dict a tool returns.
    """
    from repowise.server.mcp_server._meta import MCP_CONTRACT_VERSION
    from repowise.server.mcp_server.tool_index_status import get_index_status

    payload = await get_index_status()

    assert isinstance(payload, dict)
    assert payload["_meta"]["contract_version"] == MCP_CONTRACT_VERSION
