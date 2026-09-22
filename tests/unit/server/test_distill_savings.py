"""Unit tests for GET /api/repos/{repo_id}/savings."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from httpx import AsyncClient

from repowise.core.distill.store import OmissionStore

from .conftest import create_test_repo


@pytest.mark.parametrize("legacy_store", [False, True])
async def test_savings_endpoint_surfaces_bounded_mcp_usage_counts(
    client: AsyncClient, tmp_path: Path, legacy_store: bool
) -> None:
    repo = await create_test_repo(client, tmp_path)
    repo_dir = Path(repo["local_path"])
    store = OmissionStore(repo_dir / ".repowise" / "omissions" / "omissions.db")
    store.record_mcp_usage(
        tool="search_codebase",
        duration_ms=40,
        error=False,
        no_match=False,
        degraded=False,
        replaced_tokens=1000,
        delivered_tokens=200,
    )
    store.record_mcp_usage(
        tool="search_codebase",
        duration_ms=60,
        error=False,
        no_match=True,
        degraded=True,
        replaced_tokens=0,
        delivered_tokens=100,
    )
    store.record_mcp_usage(
        tool="get_context",
        duration_ms=20,
        error=True,
        no_match=False,
        degraded=False,
        replaced_tokens=0,
        delivered_tokens=50,
    )
    store.close()
    db_path = repo_dir / ".repowise" / "omissions" / "omissions.db"
    if legacy_store:
        # Existing fork stores have daily MCP aggregates but no upstream ledger.
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP TABLE savings_events")
    before = db_path.read_bytes()

    resp = await client.get(f"/api/repos/{repo['id']}/savings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is not legacy_store
    assert db_path.read_bytes() == before  # Reporting must not upgrade/write the sidecar.
    assert data["mcp_usage_calls"] == 3
    assert data["mcp_usage_error_calls"] == 1
    assert data["mcp_usage_no_match_calls"] == 1
    assert data["mcp_usage_degraded_calls"] == 1
    assert data["mcp_usage_avg_duration_ms"] == 40.0
    assert data["mcp_usage_window_days"] == 30
    by_tool = {row["tool"]: row for row in data["mcp_usage_per_tool"]}
    assert by_tool["search_codebase"]["calls"] == 2
    assert by_tool["search_codebase"]["saved_tokens"] == 800
    assert by_tool["get_context"]["error_calls"] == 1
