"""Contracts for the native documentation transport.

``repowise-docs`` is the single docs writer: it serves reads, serves the four
library mutations locally, and owns the background runtime. These tests pin
that shape, the fail-closed mutation auth on ``/docs/action/index``, and the
byte-level response contract that MCP clients already depend on.

The response and schema contracts are checked against a frozen golden master
(``fixtures/legacy_response_golden_master.json``) rather than against the
retired code package. Importing that package here would make these tests
disappear with it.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

import repowise.docs.jobs.single_writer as single_writer
import repowise.docs.runtime as docs_runtime
import repowise.docs.server.mcp_tools as mcp_tools
import repowise.docs.transport.http_server as docs_http
from repowise.docs.graph_status import get_docs_graph_sync_status
from repowise.docs.server.executor import (
    DocToolExecutor,
    get_doc_tool_schema_by_name,
    get_public_doc_tool_definitions,
)
from repowise.docs.server.response import render_response as render_docs_response

PUBLIC_TOOL_NAMES = [
    "list_doc_files",
    "resolve_library_id",
    "search_docs",
    "expand_doc_chunk",
    "list_doc_libraries",
    "read_doc",
    "update_doc_library",
    "add_doc_library",
    "delete_doc_library",
    "export_doc_bundle",
    "import_doc_bundle",
]
MUTATION_TOOL_NAMES = [
    "add_doc_library",
    "delete_doc_library",
    "import_doc_bundle",
    "update_doc_library",
]

GOLDEN_MASTER = json.loads(
    (Path(__file__).parent / "fixtures" / "legacy_response_golden_master.json").read_text(
        encoding="utf-8"
    )
)

# The payloads the golden master was rendered from. Kept beside the fixture so
# a diff shows both the input and the frozen output.
GOLDEN_PAYLOADS: dict[str, dict[str, object]] = {
    "resolve_library_id": {"library_id": "/owner/repo", "confidence": 0.98},
    "search_docs": {
        "library_id": "/owner/repo",
        "query": "API Foo",
        "results": [
            {
                "id": "chunk-1",
                "file_path": "docs/api.md",
                "title": "Foo",
                "anchor": "foo",
                "chunk_anchor": "foo",
                "chunk_type": "doc",
                "score": 0.95,
                "retrieval_rank_features": ["semantic", "lexical"],
                "rank_explain": {
                    "matched_on": ["Foo"],
                    "quality_score": 1.0,
                    "file_signals": ["api", "heading"],
                },
                "file_facets": {
                    "route_paths": ["/foo"],
                    "search_hints": ["noise"],
                },
            }
        ],
    },
    "expand_doc_chunk": {"chunk_id": "chunk-1", "file_path": "docs/api.md"},
    "list_doc_libraries": {"libraries": [], "offset": 0, "returned": 0},
    "read_doc": {"library_id": "/owner/repo", "path": "docs/api.md"},
    "update_doc_library": {"library_id": "/owner/repo", "status": "queued"},
    "add_doc_library": {"library_id": "/owner/repo", "status": "queued"},
    "delete_doc_library": {"library_id": "/owner/repo", "status": "deleted"},
    "export_doc_bundle": {"library_id": "/owner/repo", "status": "exported"},
    "import_doc_bundle": {"library_id": "/owner/repo", "status": "imported"},
}


def _settings(
    *,
    webhook_token: str = "",
    allow_unauthenticated: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        host="127.0.0.1",
        port=8200,
        log_level="INFO",
        max_concurrent_tool_calls=4,
        tool_timeout_search_seconds=20.0,
        tool_timeout_seconds=30.0,
        webhook_token=webhook_token,
        allow_unauthenticated=allow_unauthenticated,
    )


@pytest.fixture(autouse=True)
def reset_docs_runtime_state() -> None:
    def _reset() -> None:
        docs_runtime._STATE.initialized = False
        docs_runtime._STATE.background_started = False
        docs_runtime._STATE.worker_id = None
        docs_runtime._STATE.bootstrap_summary = None
        docs_runtime._STATE.lock = None
        docs_runtime._STATE.background_declined_reason = None
        single_writer._lock_connection = None

    _reset()
    yield
    _reset()


@pytest.fixture
def started_app(monkeypatch: pytest.MonkeyPatch):
    """Build the host with the runtime stubbed out, ready for TestClient."""

    def _build(settings: SimpleNamespace | None = None):
        monkeypatch.setattr(docs_http, "ensure_docs_runtime", AsyncMock())
        monkeypatch.setattr(docs_http, "close_docs_runtime", AsyncMock())
        return docs_http.create_app(settings=settings or _settings())

    return _build


# --------------------------------------------------------------------------
# Tool surface and response bytes
# --------------------------------------------------------------------------


def test_docs_worker_preserves_legacy_schemas_and_adds_file_browsing() -> None:
    tools = get_public_doc_tool_definitions()
    assert [str(tool.name) for tool in tools] == PUBLIC_TOOL_NAMES

    schemas = get_doc_tool_schema_by_name()
    assert "search_docs_multi" in schemas
    assert {
        name: schema for name, schema in schemas.items() if name != "list_doc_files"
    } == GOLDEN_MASTER["doc_tool_schemas"]


@pytest.mark.parametrize("tool_name", sorted(GOLDEN_PAYLOADS))
def test_docs_renderer_preserves_the_frozen_response_bytes(tool_name: str) -> None:
    rendered = render_docs_response(
        tool_name=tool_name,
        status="success",
        confidence="high",
        next_action="Continue with the next docs tool.",
        body="body",
        output="json",
        payload=copy.deepcopy(GOLDEN_PAYLOADS[tool_name]),
    )
    assert rendered == GOLDEN_MASTER["success"][tool_name]


def test_docs_renderer_preserves_the_frozen_error_bytes() -> None:
    rendered = render_docs_response(
        tool_name="search_docs",
        status="error",
        confidence="high",
        next_action="Fix tool arguments to match schema and retry.",
        body="Error (`invalid_arguments`): query is required",
        output="json",
        payload={"error": "query is required", "error_code": "invalid_arguments"},
    )
    assert rendered == GOLDEN_MASTER["error"]


def test_graph_status_is_explicitly_disabled() -> None:
    status = get_docs_graph_sync_status()
    assert status["enabled"] is False
    assert status["state"] == "disabled"
    assert status["reason"] == "docs_neo4j_metadata_sync_retired"


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("health_status_code", "expected_status", "expected_body_status"),
    [(200, 200, "ok"), (503, 503, "degraded")],
)
def test_readyz_reports_native_mode_and_the_local_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
    started_app,
    health_status_code: int,
    expected_status: int,
    expected_body_status: str,
) -> None:
    monkeypatch.setattr(
        docs_http,
        "get_health_status",
        AsyncMock(
            return_value=(
                {
                    "status": "ok" if health_status_code == 200 else "degraded",
                    "runtime_mode": "native",
                    "runtime": {
                        "worker": "ok",
                        "scheduler": "ok",
                        "recovery": "ok",
                    },
                    "details": {"docs_graph": get_docs_graph_sync_status()},
                },
                health_status_code,
            )
        ),
    )

    with TestClient(started_app()) as client:
        response = client.get("/readyz")

    assert response.status_code == expected_status
    payload = response.json()
    assert payload["status"] == expected_body_status
    assert payload["runtime_mode"] == "native"
    assert payload["writer"] == {"status": "native"}
    assert payload["mutation_tools"] == {"mode": "native", "tools": MUTATION_TOOL_NAMES}
    # The worker/scheduler/recovery keys survive the relocation and now report
    # this process's own loops instead of a remote writer's.
    assert payload["runtime"] == {"worker": "ok", "scheduler": "ok", "recovery": "ok"}
    assert payload["details"]["docs_graph"]["state"] == "disabled"


def test_readyz_advertises_exactly_the_tools_served_as_mutations(
    monkeypatch: pytest.MonkeyPatch,
    started_app,
) -> None:
    monkeypatch.setattr(
        docs_http,
        "get_health_status",
        AsyncMock(return_value=({"status": "ok"}, 200)),
    )
    with TestClient(started_app()) as client:
        payload = client.get("/readyz").json()

    assert payload["mutation_tools"]["tools"] == sorted(mcp_tools.MUTATION_TOOL_NAMES)


def test_worker_does_not_publish_a_second_mcp_endpoint(started_app) -> None:
    with TestClient(started_app()) as client:
        assert client.post("/mcp/", json={"method": "tools/list"}).status_code == 404


@pytest.mark.parametrize(
    "name",
    [
        "add_doc_library",
        "update_doc_library",
        "delete_doc_library",
        "import_doc_bundle",
        "export_doc_bundle",
    ],
)
def test_internal_tool_mutations_require_operator_auth(started_app, name):
    with TestClient(started_app(_settings(webhook_token="secret"))) as client:
        assert client.post(f"/docs/tools/{name}", json={}).status_code == 401


def test_internal_tool_interface_rejects_unknown_and_invalid_requests(started_app):
    with TestClient(started_app()) as client:
        assert client.post("/docs/tools/not_a_tool", json={}).status_code == 404
        assert client.post("/docs/tools/list_doc_files", content="{}").status_code == 415
        assert (
            client.post(
                "/docs/tools/list_doc_files",
                content="{",
                headers={"content-type": "application/json"},
            ).status_code
            == 400
        )
        assert (
            client.post("/docs/tools/list_doc_files", json={"query": "x" * 65536}).status_code
            == 413
        )
        invalid = client.post(
            "/docs/tools/list_doc_files", json={"library_id": "/test/lib", "limit": 201}
        )
        assert invalid.json()["status"] == "error"


def test_internal_tool_interface_dispatches_validated_arguments(started_app, monkeypatch):
    monkeypatch.setattr(mcp_tools, "ensure_docs_runtime", AsyncMock())
    handler = AsyncMock(return_value={"files": [], "pagination": {"total": 0}})
    monkeypatch.setitem(mcp_tools._TOOL_HANDLERS, "list_doc_files", handler)
    with TestClient(started_app()) as client:
        response = client.post(
            "/docs/tools/list_doc_files", json={"library_id": "/test/lib", "output": "json"}
        )
        assert response.status_code == 200
        assert response.json()["payload"]["files"] == []
    handler.assert_awaited_once()


# --------------------------------------------------------------------------
# /docs/action/index mutation auth
# --------------------------------------------------------------------------


@pytest.fixture
def index_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record reaching the real enqueue handler without touching a database."""
    calls: list[str] = []

    async def fake_index_handler(request):
        calls.append(request.url.query)
        return JSONResponse({"job_id": "job-1", "status": "queued"}, status_code=201)

    monkeypatch.setattr(docs_http.admin_routes, "index_handler", fake_index_handler)
    return calls


def test_index_refuses_token_less_mutations_by_default(
    started_app,
    index_calls: list[str],
) -> None:
    """No token and no opt-in: fail CLOSED, and never reach the enqueue path."""
    with TestClient(started_app()) as client:
        response = client.post("/docs/action/index?library=/owner/repo")

    assert response.status_code == 401
    assert response.json()["error"] == "unauthenticated_mutation_refused"
    assert index_calls == []


def test_index_allows_token_less_mutations_when_operator_opts_in(
    started_app,
    index_calls: list[str],
) -> None:
    app = started_app(_settings(allow_unauthenticated=True))
    with TestClient(app) as client:
        response = client.post("/docs/action/index?library=/owner/repo")

    assert response.status_code == 201
    assert index_calls == ["library=/owner/repo"]


def test_index_requires_a_matching_token_when_one_is_configured(
    started_app,
    index_calls: list[str],
) -> None:
    app = started_app(_settings(webhook_token="secret"))
    with TestClient(app) as client:
        missing = client.post("/docs/action/index?library=/owner/repo")
        wrong = client.post(
            "/docs/action/index?library=/owner/repo",
            headers={"X-Doc-Search-Token": "not-the-secret"},
        )
        correct = client.post(
            "/docs/action/index?library=/owner/repo",
            headers={"X-Doc-Search-Token": "secret"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert wrong.json()["error"] == "Invalid or missing auth token"
    assert correct.status_code == 201
    assert index_calls == ["library=/owner/repo"]


def test_index_still_accepts_the_legacy_token_header(
    started_app,
    index_calls: list[str],
) -> None:
    """One-release fallback for callers written against the proxy era."""
    app = started_app(_settings(webhook_token="secret"))
    with TestClient(app) as client:
        response = client.post(
            "/docs/action/index?library=/owner/repo",
            headers={"X-Code-Search-Token": "secret"},
        )

    assert response.status_code == 201
    assert index_calls == ["library=/owner/repo"]


def test_index_auth_opt_in_is_not_inherited_by_the_read_only_job_routes(
    monkeypatch: pytest.MonkeyPatch,
    started_app,
) -> None:
    """GET /docs/action/jobs reads state, so it is not behind the token gate."""
    listed: list[str] = []

    async def fake_list_jobs_handler(request):
        listed.append(request.url.path)
        return JSONResponse({"jobs": [], "count": 0})

    monkeypatch.setattr(docs_http.admin_routes, "list_jobs_handler", fake_list_jobs_handler)
    with TestClient(started_app()) as client:
        response = client.get("/docs/action/jobs")

    assert response.status_code == 200
    assert listed == ["/docs/action/jobs"]


def test_docs_libraries_route_is_mounted(
    monkeypatch: pytest.MonkeyPatch,
    started_app,
) -> None:
    """The admin route table is not the served one: this host mounts its own.

    ``/docs/libraries`` has to be registered twice -- once in
    ``build_admin_routes`` and once here -- or it exists only under test.
    Read-only, so it stays outside the mutation token gate.
    """
    served: list[str] = []

    async def fake_libraries_handler(request):
        served.append(request.url.path)
        return JSONResponse({"libraries": [], "summary": {"total_libraries": 0}})

    monkeypatch.setattr(docs_http.admin_routes, "libraries_handler", fake_libraries_handler)
    with TestClient(started_app()) as client:
        response = client.get("/docs/libraries")

    assert response.status_code == 200
    assert response.json()["libraries"] == []
    assert served == ["/docs/libraries"]


# --------------------------------------------------------------------------
# Native mutation dispatch
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mutations_run_against_local_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inverse of the retired proxy contract: mutations execute here."""
    monkeypatch.setattr(mcp_tools, "ensure_docs_runtime", AsyncMock())
    monkeypatch.setattr(mcp_tools, "docs_runtime_accepts_mutations", lambda: True)
    handled: list[str] = []

    for name in MUTATION_TOOL_NAMES:

        async def handler(
            arguments: dict[str, object],
            *,
            _tool_name: str = name,
        ) -> dict[str, object]:
            handled.append(_tool_name)
            return {"library_id": "/owner/repo", "status": "ok"}

        monkeypatch.setitem(mcp_tools._TOOL_HANDLERS, name, handler)

    executor = DocToolExecutor(settings=_settings())
    calls = {
        "add_doc_library": {"repo": "owner/repo", "name": "Repo"},
        "delete_doc_library": {"library_id": "/owner/repo", "confirm": True},
        "import_doc_bundle": {"bundle_path": "/tmp/repo.docbundle.json.gz"},
        "update_doc_library": {"library_id": "/owner/repo"},
    }
    for name, arguments in calls.items():
        rendered = await executor.execute(name, dict(arguments))
        assert json.loads(rendered[0].text)["status"] == "success", name

    assert sorted(handled) == MUTATION_TOOL_NAMES


@pytest.mark.asyncio
async def test_mutations_needing_the_worker_refuse_while_it_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reads-only process must not accept work nothing will drain."""
    monkeypatch.setattr(mcp_tools, "ensure_docs_runtime", AsyncMock())
    monkeypatch.setattr(mcp_tools, "docs_runtime_accepts_mutations", lambda: False)

    async def handler(_arguments: dict[str, object]) -> dict[str, object]:
        raise AssertionError("mutation ran without a background runtime")

    monkeypatch.setitem(mcp_tools._TOOL_HANDLERS, "add_doc_library", handler)

    executor = DocToolExecutor(settings=_settings())
    rendered = await executor.execute("add_doc_library", {"repo": "owner/repo", "name": "Repo"})
    payload = json.loads(rendered[0].text)
    assert payload["status"] == "error"
    assert payload["payload"]["error_code"] == "docs_runtime_not_ready"


# --------------------------------------------------------------------------
# Background runtime ownership
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_starts_the_background_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(docs_http, "ensure_docs_runtime", ensure)
    monkeypatch.setattr(docs_http, "close_docs_runtime", close)

    app = docs_http.create_app(settings=_settings())
    async with app.router.lifespan_context(app):
        assert app.state.runtime_mode == "native"

    ensure.assert_awaited_once_with(start_background=True)
    close.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_host_starts_exactly_one_background_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async_mocks = {
        "init_pool": AsyncMock(),
        "initialize_health_checker": AsyncMock(),
        "sync_library_registry": AsyncMock(return_value={}),
        "list_libraries": AsyncMock(return_value=[]),
        "bootstrap_missing_snapshot_libraries": AsyncMock(return_value={}),
        "start_worker": AsyncMock(return_value="writer-1"),
        "start_scheduler": AsyncMock(),
        "start_recovery_task": AsyncMock(),
        "stop_worker": AsyncMock(),
        "stop_scheduler": AsyncMock(),
        "stop_recovery_task": AsyncMock(),
        "close_health_checker": AsyncMock(),
        "close_docs_embedding_http_client": AsyncMock(),
        "close_docs_remote_http_client": AsyncMock(),
        "close_pool": AsyncMock(),
        "acquire_background_lock": AsyncMock(return_value=True),
        "release_background_lock": AsyncMock(),
    }
    for name, mock in async_mocks.items():
        monkeypatch.setattr(docs_runtime, name, mock)
    monkeypatch.setattr(docs_runtime, "close_qdrant_client", MagicMock())

    await docs_runtime.ensure_docs_runtime(start_background=True)
    app = docs_http.create_app(settings=_settings())
    async with app.router.lifespan_context(app):
        assert docs_runtime._STATE.background_started is True
        assert async_mocks["start_worker"].await_count == 1
        assert async_mocks["start_scheduler"].await_count == 1
        assert async_mocks["start_recovery_task"].await_count == 1
        # The registry sync and snapshot bootstrap are writer-side work and
        # must not be repeated by a second ensure_docs_runtime call either.
        assert async_mocks["sync_library_registry"].await_count == 1
        assert async_mocks["acquire_background_lock"].await_count == 1


@pytest.mark.asyncio
async def test_a_second_process_serves_reads_and_starts_no_background_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The advisory-lock refusal path: degrade to reads, never double-write."""
    async_mocks = {
        "init_pool": AsyncMock(),
        "initialize_health_checker": AsyncMock(),
        "sync_library_registry": AsyncMock(return_value={}),
        "list_libraries": AsyncMock(return_value=[]),
        "bootstrap_missing_snapshot_libraries": AsyncMock(return_value={}),
        "start_worker": AsyncMock(return_value="writer-1"),
        "start_scheduler": AsyncMock(),
        "start_recovery_task": AsyncMock(),
        "acquire_background_lock": AsyncMock(return_value=False),
    }
    for name, mock in async_mocks.items():
        monkeypatch.setattr(docs_runtime, name, mock)

    await docs_runtime.ensure_docs_runtime(start_background=True)

    assert docs_runtime._STATE.initialized is True
    assert docs_runtime._STATE.background_started is False
    assert docs_runtime._STATE.background_declined_reason is not None
    assert docs_runtime.docs_runtime_accepts_reads() is True
    assert docs_runtime.docs_runtime_accepts_mutations() is False
    for name in (
        "sync_library_registry",
        "bootstrap_missing_snapshot_libraries",
        "start_worker",
        "start_scheduler",
        "start_recovery_task",
    ):
        assert async_mocks[name].await_count == 0, name


@pytest.mark.asyncio
async def test_a_failed_background_start_gives_the_lock_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = AsyncMock()
    async_mocks = {
        "init_pool": AsyncMock(),
        "initialize_health_checker": AsyncMock(),
        "sync_library_registry": AsyncMock(return_value={}),
        "list_libraries": AsyncMock(return_value=[]),
        "bootstrap_missing_snapshot_libraries": AsyncMock(return_value={}),
        "start_worker": AsyncMock(return_value="writer-1"),
        "start_scheduler": AsyncMock(side_effect=RuntimeError("scheduler boom")),
        "stop_worker": AsyncMock(),
        "acquire_background_lock": AsyncMock(return_value=True),
        "release_background_lock": release,
    }
    for name, mock in async_mocks.items():
        monkeypatch.setattr(docs_runtime, name, mock)

    with pytest.raises(RuntimeError, match="scheduler boom"):
        await docs_runtime.ensure_docs_runtime(start_background=True)

    async_mocks["stop_worker"].assert_awaited_once()
    release.assert_awaited_once()
    assert docs_runtime._STATE.background_started is False


def test_started_host_imports_no_retired_code_modules() -> None:
    """The docs host must stand alone once the retired package is deleted."""
    codeatlas_root = Path(__file__).parents[2]
    probe = r"""
import asyncio
import sys
from types import SimpleNamespace

import repowise.docs.transport.http_server as host

async def noop(*args, **kwargs):
    return None

async def main():
    host.ensure_docs_runtime = noop
    host.close_docs_runtime = noop
    settings = SimpleNamespace(
        max_concurrent_tool_calls=4,
        tool_timeout_search_seconds=20.0,
        tool_timeout_seconds=30.0,
        webhook_token="",
        allow_unauthenticated=False,
    )
    app = host.create_app(settings=settings)
    async with app.router.lifespan_context(app):
        retired_package = "_".join(("code", "search"))
        loaded = sorted(
            name for name in sys.modules
            if name == retired_package or name.startswith(f"{retired_package}.")
        )
        if loaded:
            raise RuntimeError(f"code-search imports leaked into docs host: {loaded}")
        print("started-clean")

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=codeatlas_root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "started-clean"
