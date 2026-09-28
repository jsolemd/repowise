"""Regression test for issue #970 — ``repowise serve`` over a workspace.

The HTTP lifespan used to leave ``_state._registry`` unset, so chat tool
calls resolved aliases against the primary repo's database only and blew up
with ``LookupError: Repository not found: <alias>`` for every repo.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import repowise.server.mcp_server as mcp_mod
from repowise.core.persistence.database import init_db
from repowise.core.persistence.models import GenerationJob, Repository
from repowise.server.app import create_app, lifespan

_NOW = datetime(2026, 7, 21, 10, 0, 0, tzinfo=UTC)


async def _make_repo(root: Path, alias: str) -> None:
    """Create ``<root>/<alias>/.repowise/wiki.db`` with one repository row."""
    repo_path = root / alias
    (repo_path / ".repowise").mkdir(parents=True)
    db = repo_path / ".repowise" / "wiki.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db.as_posix()}")
    await init_db(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        session.add(
            Repository(
                id=f"{alias}-id",
                name=alias,
                url=f"https://example.com/{alias}",
                local_path=str(repo_path),
                default_branch="main",
                settings_json="{}",
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
        await session.commit()
    await engine.dispose()


@pytest.fixture(autouse=True)
def restore_tool_globals():
    """The real lifespan writes process-global MCP tool state — put it back."""
    saved = (
        mcp_mod._registry,
        mcp_mod._workspace_root,
        mcp_mod._cross_repo_enricher,
        mcp_mod._session_factory,
        mcp_mod._fts,
        mcp_mod._vector_store,
    )
    yield
    (
        mcp_mod._registry,
        mcp_mod._workspace_root,
        mcp_mod._cross_repo_enricher,
        mcp_mod._session_factory,
        mcp_mod._fts,
        mcp_mod._vector_store,
    ) = saved


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("REPOWISE_DB_URL", raising=False)
    monkeypatch.delenv("REPOWISE_DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".repowise-workspace.yaml").write_text(
        "version: 1\n"
        "default_repo: boot\n"
        "repos:\n"
        "- path: boot\n"
        "  alias: boot\n"
        "  is_primary: true\n"
        "- path: gateway\n"
        "  alias: gateway\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduler_enabled", [True, False])
async def test_lifespan_publishes_the_repo_registry(workspace, monkeypatch, scheduler_enabled):
    if scheduler_enabled:
        monkeypatch.delenv("REPOWISE_SCHEDULER_ENABLED", raising=False)
    else:
        monkeypatch.setenv("REPOWISE_SCHEDULER_ENABLED", "0")
    await _make_repo(workspace, "boot")
    await _make_repo(workspace, "gateway")

    from repowise.server.mcp_server._helpers import _resolve_repo_context

    app = create_app()
    async with lifespan(app):
        if scheduler_enabled:
            assert {job.id for job in app.state.scheduler.get_jobs()} == {
                "staleness_check", "polling_fallback",
            }
            assert app.state.scheduler.running
        else:
            assert app.state.scheduler is None
            async with app.state.session_factory() as session:
                assert (await session.scalars(select(GenerationJob))).all() == []

        registry = mcp_mod._registry
        assert registry is not None, "workspace mode never reached the chat tools"
        assert sorted(registry.get_all_aliases()) == ["boot", "gateway"]
        assert mcp_mod._workspace_root == str(workspace)

        # Both repos resolve — the non-primary one is the case that used to
        # raise LookupError no matter what.
        for alias in ("boot", "gateway"):
            ctx = await _resolve_repo_context(alias)
            assert ctx.alias == alias

        if not scheduler_enabled:
            # Disabling automatic refresh must preserve an explicit user sync.
            with patch("repowise.server.routers.repos.execute_job", new_callable=AsyncMock) as execute:
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://localhost"
                ) as client:
                    response = await client.post("/api/repos/boot-id/sync")
                assert response.status_code == 202
                await asyncio.gather(*app.state.background_tasks)
                execute.assert_awaited_once()
                assert execute.await_args.args[0] == response.json()["job_id"]
            async with app.state.session_factory() as session:
                job = (await session.scalars(select(GenerationJob))).one()
                assert job.id == response.json()["job_id"]
                assert job.status == "pending"

    # Shutdown puts the globals back so a later single-repo server is clean.
    assert mcp_mod._registry is None
    assert mcp_mod._workspace_root is None
