"""The hard no-generative policy reaches the product's own generative surfaces.

``REPOWISE_TOOLS_NO_GENERATIVE`` already stops indexing, page generation, and the
agent tool surface. Two REST routes still reached a provider after that work:
``POST /api/repos/{id}/chat/messages``, which sends repo content to a chat model,
and the provider smoke test, which does a live ``generate`` round-trip. Both are
closed here, and ``GET /api/meta/policy`` exists so the dashboard can decline to
draw an affordance rather than discover the refusal by receiving it.

Every test asserts the provider seam was never reached, not merely that the
response looked right — a refusal that still resolved a provider would be a
refusal that already leaked configuration and could have leaked content.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from repowise.core.generative_policy import NO_GENERATIVE_ENV
from tests.unit.server.conftest import create_test_repo


@pytest.fixture
async def client(app) -> AsyncClient:
    """The shared test app plus the chat router, which it does not mount.

    Overrides the package fixture rather than editing it: every other server
    test gets its app unchanged, and the one suite that exercises the chat route
    is the only one that pays for mounting it.
    """
    from repowise.server.routers import chat as chat_router_module

    if not any(
        getattr(route, "path", None) == "/api/repos/{repo_id}/chat/messages" for route in app.routes
    ):
        app.include_router(chat_router_module.router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def provider_calls(monkeypatch) -> list[str]:
    """Poison provider resolution: any call records itself and then raises."""
    calls: list[str] = []

    def _poisoned(*args, **kwargs):
        calls.append("get_chat_provider_instance")
        raise AssertionError("provider resolution must not happen under the hard policy")

    import repowise.server.provider_config as pc

    monkeypatch.setattr(pc, "get_chat_provider_instance", _poisoned)
    # ``chat.py`` binds the symbol at import time; ``providers.py`` imports it
    # inside the handler. Patch both so neither route can slip past the spy.
    import repowise.server.routers.chat as chat_router

    monkeypatch.setattr(chat_router, "get_chat_provider_instance", _poisoned)
    return calls


async def _chat(client: AsyncClient, repo_id: str):
    return await client.post(
        f"/api/repos/{repo_id}/chat/messages",
        json={"message": "who owns the search coordinator?"},
    )


@pytest.mark.asyncio
async def test_chat_is_refused_under_the_hard_policy(
    client: AsyncClient, monkeypatch, provider_calls
) -> None:
    repo = await create_test_repo(client)
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")

    resp = await _chat(client, repo["id"])

    assert resp.status_code == 409
    assert NO_GENERATIVE_ENV in resp.json()["detail"]
    assert provider_calls == []


@pytest.mark.asyncio
async def test_chat_repo_dotenv_policy_is_honoured_without_the_env_var(
    client: AsyncClient, monkeypatch, provider_calls
) -> None:
    """A repo-local policy must refuse even when the process environment is clean.

    This is the fail-closed case the dashboard depends on: if the server said
    "allowed" from process environment alone, the UI would offer a chat that the
    server then refuses.
    """
    repo = await create_test_repo(client)
    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)
    repowise_dir = Path(repo["local_path"]) / ".repowise"
    repowise_dir.mkdir(parents=True, exist_ok=True)
    (repowise_dir / ".env").write_text(f"{NO_GENERATIVE_ENV}=true\n", encoding="utf-8")

    resp = await _chat(client, repo["id"])

    assert resp.status_code == 409
    assert provider_calls == []


@pytest.mark.asyncio
async def test_chat_still_resolves_a_provider_with_the_policy_off(
    client: AsyncClient, monkeypatch, provider_calls
) -> None:
    """Flag-off behaviour is unchanged: the route reaches provider resolution.

    The poisoned resolver makes resolution fail, which the route reports as its
    existing "no chat provider available" 422. The assertion that matters is
    that the seam was reached at all, and that the failure is not the policy's.
    """
    repo = await create_test_repo(client)
    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)

    resp = await _chat(client, repo["id"])

    assert resp.status_code == 422
    assert NO_GENERATIVE_ENV not in resp.text
    assert provider_calls == ["get_chat_provider_instance"]


@pytest.mark.asyncio
async def test_provider_validate_is_refused_in_its_own_result_shape(
    client: AsyncClient, monkeypatch, provider_calls
) -> None:
    """The smoke test reports the refusal through ``{ok, error}``, not a status.

    This endpoint's contract is that the settings UI always receives something
    renderable; a policy refusal is a result, not an exception.
    """
    repo = await create_test_repo(client)
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")

    resp = await client.post(f"/api/providers/anthropic/validate?repo_id={repo['id']}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert NO_GENERATIVE_ENV in body["error"]
    assert body["model"] is None
    assert provider_calls == []


@pytest.mark.asyncio
async def test_policy_endpoint_reports_the_hard_policy(client: AsyncClient, monkeypatch) -> None:
    repo = await create_test_repo(client)
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")

    resp = await client.get(f"/api/meta/policy?repo_id={repo['id']}")

    assert resp.status_code == 200
    assert resp.json() == {
        "generative_disabled": True,
        "generative_policy_source": NO_GENERATIVE_ENV,
    }


@pytest.mark.asyncio
async def test_policy_endpoint_reports_an_unrestricted_deployment(
    client: AsyncClient, monkeypatch
) -> None:
    repo = await create_test_repo(client)
    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)

    resp = await client.get(f"/api/meta/policy?repo_id={repo['id']}")

    assert resp.json()["generative_disabled"] is False


@pytest.mark.asyncio
async def test_policy_endpoint_sees_a_repo_local_policy(client: AsyncClient, monkeypatch) -> None:
    """The disclosure and the refusal must agree, including on dotenv-only policy."""
    repo = await create_test_repo(client)
    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)
    repowise_dir = Path(repo["local_path"]) / ".repowise"
    repowise_dir.mkdir(parents=True, exist_ok=True)
    (repowise_dir / ".env").write_text(f"{NO_GENERATIVE_ENV}=on\n", encoding="utf-8")

    resp = await client.get(f"/api/meta/policy?repo_id={repo['id']}")

    assert resp.json()["generative_disabled"] is True


@pytest.mark.asyncio
async def test_policy_endpoint_resolves_an_unknown_repo_without_failing(
    client: AsyncClient, monkeypatch
) -> None:
    """A bad ``repo_id`` degrades to the server-wide answer, never a 500."""
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")

    resp = await client.get("/api/meta/policy?repo_id=does-not-exist")

    assert resp.status_code == 200
    assert resp.json()["generative_disabled"] is True
