"""RepoWise docs API authentication, bounded worker transport, and errors."""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from repowise.docs.client import DocsClient, DocsUnavailable
from repowise.server.routers.docs_libraries import router


@pytest.fixture(autouse=True)
def docs_url(monkeypatch):
    monkeypatch.setenv("REPOWISE_DOCS_URL", "http://docs.test:9999/")
    monkeypatch.delenv("REPOWISE_API_KEY", raising=False)


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_inventory_preserves_fields_and_uses_configured_url(client, respx_mock):
    payload = {
        "libraries": [{"library_id": "/codeatlas/cosmograph"}],
        "inventory": {"total_files": 120},
    }
    route = respx_mock.get("http://docs.test:9999/docs/libraries").respond(json=payload)
    response = await client.get("/api/docs-libraries")
    assert response.status_code == 200
    assert response.json() == payload
    assert route.call_count == 1


@pytest.mark.parametrize(
    "body", ["<html>Unavailable</html>", "null", "[]", "x" * (4 * 1024 * 1024 + 1)]
)
async def test_invalid_or_oversized_worker_response_is_bounded(client, respx_mock, body):
    respx_mock.get("http://docs.test:9999/docs/libraries").respond(text=body)
    assert (await client.get("/api/docs-libraries")).status_code == 502


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ReadTimeout("timeout")])
async def test_worker_down_is_actionable_and_does_not_retry(client, respx_mock, error):
    route = respx_mock.get("http://docs.test:9999/docs/libraries").mock(side_effect=error)
    response = await client.get("/api/docs-libraries")
    assert response.status_code == 502
    assert "solemd infra up" in response.json()["detail"]
    assert route.call_count == 1


async def test_tools_share_worker_auth_and_preserve_envelope(client, respx_mock, monkeypatch):
    monkeypatch.setenv("REPOWISE_DOCS_TOKEN", "worker-token")
    payload = {"status": "success", "payload": {"files": [], "pagination": {"total": 0}}}
    route = respx_mock.post("http://docs.test:9999/docs/tools/list_doc_files").respond(json=payload)
    response = await client.post(
        "/api/docs-libraries/tools/list_doc_files", json={"library_id": "/a/b"}
    )
    assert response.json() == payload
    request = route.calls.last.request
    assert request.headers["X-Doc-Search-Token"] == "worker-token"
    assert request.extensions["timeout"]["connect"] == 3
    assert request.extensions["timeout"]["read"] == 35


async def test_validation_failures_keep_worker_message(client, respx_mock):
    respx_mock.post("http://docs.test:9999/docs/tools/read_doc").respond(
        json={"status": "error", "payload": {"error": "Library not found"}}
    )
    response = await client.post("/api/docs-libraries/tools/read_doc", json={})
    assert response.status_code == 422
    assert response.json()["detail"] == "Library not found"


async def test_arbitrary_tools_and_browser_form_posts_are_rejected(client):
    assert (await client.post("/api/docs-libraries/tools/other", json={})).status_code == 404
    assert (
        await client.post("/api/docs-libraries/tools/add_doc_library", data={"repo": "x"})
    ).status_code == 415
    assert (
        await client.post(
            "/api/docs-libraries/tools/add_doc_library",
            json={},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
    ).status_code == 403
    assert (await client.post("/api/docs-libraries/tools/read_doc", json=[])).status_code == 422
    assert (
        await client.post(
            "/api/docs-libraries/tools/read_doc",
            content="{",
            headers={"Content-Type": "application/json"},
        )
    ).status_code == 400
    assert (
        await client.post("/api/docs-libraries/tools/read_doc", json={"path": "x" * 65536})
    ).status_code == 413


async def test_docs_api_uses_existing_api_key_policy(client, monkeypatch):
    from repowise.server import deps

    monkeypatch.setattr(deps, "_API_KEY", "api-secret")
    assert (await client.get("/api/docs-libraries")).status_code == 401
    assert (
        await client.post("/api/docs-libraries/tools/delete_doc_library", json={})
    ).status_code == 401


async def test_worker_auth_failure_is_not_exposed_as_dashboard_login(client, respx_mock):
    respx_mock.get("http://docs.test:9999/docs/libraries").respond(403)
    response = await client.get("/api/docs-libraries")
    assert response.status_code == 502
    assert "worker token" in response.json()["detail"]


async def test_cancellation_reaches_worker_connection(respx_mock):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def wait(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    respx_mock.get("http://docs.test:9999/docs/libraries").mock(side_effect=wait)
    task = asyncio.create_task(DocsClient().inventory())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


async def test_deadline_covers_streaming_and_closes_the_response(respx_mock, monkeypatch):
    from repowise.docs import client as docs_client

    native_timeout = asyncio.timeout
    deadlines = []
    started = asyncio.Event()

    def capture_timeout(seconds):
        assert seconds == 40.0
        deadline = native_timeout(seconds)
        deadlines.append(deadline)
        return deadline

    class StalledBody(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"libraries":'
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    stream = StalledBody()
    route = respx_mock.get("http://docs.test:9999/docs/libraries").respond(stream=stream)
    monkeypatch.setattr(docs_client.asyncio, "timeout", capture_timeout)
    task = asyncio.create_task(DocsClient().inventory())
    await started.wait()
    # Expire the real timeout after the response starts; no wall-clock budget
    # or sleep is needed to prove that a dribbling response cannot hold it open.
    deadlines[0].reschedule(asyncio.get_running_loop().time())
    with pytest.raises(DocsUnavailable, match="deadline"):
        await task
    assert stream.closed
    assert route.call_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status": [], "payload": {}},
        {"status": "unknown", "payload": {}},
        {"status": "success", "payload": None},
        {"status": "success", "payload": []},
        {"status": "success", "payload": {}, "trust": None},
        {"status": "error", "payload": {"error": {"message": "invalid"}}},
        {"output": "markdown", "content": "unexpected output"},
    ],
)
async def test_malformed_tool_envelopes_are_worker_failures(client, respx_mock, payload):
    route = respx_mock.post("http://docs.test:9999/docs/tools/read_doc").respond(json=payload)
    response = await client.post("/api/docs-libraries/tools/read_doc", json={"output": "json"})
    assert response.status_code == 502
    assert "invalid tool response" in response.json()["detail"]
    assert route.call_count == 1


async def test_requested_markdown_remains_supported(client, respx_mock):
    payload = {"output": "markdown", "content": "# Tool Response\n\nDocument text"}
    respx_mock.post("http://docs.test:9999/docs/tools/read_doc").respond(json=payload)
    response = await client.post("/api/docs-libraries/tools/read_doc", json={"output": "markdown"})
    assert response.status_code == 200
    assert response.json() == payload
