"""The dashboard's selected repository must scope hybrid source retrieval."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.unit.server.conftest import create_test_repo


@pytest.mark.parametrize("scope", ["selected", "all", "unknown", "unindexed"])
async def test_dashboard_hybrid_uses_workspace_scope(client, app, tmp_path, monkeypatch, scope):
    from repowise.server.mcp_server import _source_federation
    from repowise.server.routers import search

    repo = await create_test_repo(client, tmp_path)
    registry = SimpleNamespace(
        resolve_repo_param=lambda path: "secondary" if path == repo["local_path"] else "unexpected"
    )
    app.state.repo_registry = registry
    monkeypatch.setattr(search, "source_search_enabled", lambda: True)
    primary = SimpleNamespace(search=AsyncMock(return_value={"results": [{"repo": "primary"}]}))
    monkeypatch.setattr(search, "rest_coordinator", AsyncMock(return_value=primary))
    workspace = AsyncMock(return_value={"results": [{"repo": "secondary"}]})
    monkeypatch.setattr(_source_federation, "workspace_source_search", workspace)
    params = {"query": "owner", "limit": 3}
    if scope != "all":
        params["repo_id"] = (
            repo["id"]
            if scope == "selected"
            else "ws:empty"
            if scope == "unindexed"
            else "missing-id"
        )
    response = await client.get("/api/search", params=params)
    assert response.status_code == 200
    if scope in {"unknown", "unindexed"}:
        assert response.json() == []
        workspace.assert_not_awaited()
    else:
        assert response.json()["results"] == [{"repo": "secondary"}]
        workspace.assert_awaited_once()
        assert workspace.await_args.kwargs["repo"] == (
            "secondary" if scope == "selected" else "all"
        )
        assert workspace.await_args.kwargs["registry"] is registry
        assert workspace.await_args.kwargs["limit"] == 3
    primary.search.assert_not_awaited()


async def test_dashboard_fulltext_keeps_its_native_route(client, app, monkeypatch):
    from repowise.server.mcp_server import _source_federation
    from repowise.server.routers import search

    app.state.repo_registry = SimpleNamespace()
    monkeypatch.setattr(search, "source_search_enabled", lambda: True)
    workspace = AsyncMock()
    monkeypatch.setattr(_source_federation, "workspace_source_search", workspace)
    await app.state.fts.index("file_page:owner.py", "owner", "canary routing")
    response = await client.get(
        "/api/search", params={"query": "canary", "search_type": "fulltext"}
    )
    assert response.status_code == 200
    assert response.json()[0]["page_id"] == "file_page:owner.py"
    workspace.assert_not_awaited()
