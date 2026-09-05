"""The two delegation points: what the flag switches, and what it must not.

The coordinator is opt-in, and "opt-in" has to mean the stock path is not
merely equivalent with the flag off but *untouched* — the guard has to
short-circuit before it can construct anything, reach any store, or change any
ordering. Each test here asserts the negative as well as the positive: which
call was made, and that the other one was not.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from repowise.core.source_search import SOURCE_SEARCH_ENV

STOCK = {"results": [{"stock": True}]}
COORDINATED = {"results": [], "confidence": "no_match", "selected_owner": None}


class _Coordinator:
    """Records that it was asked, and what for."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, **kwargs: Any) -> dict:
        self.queries.append((query, kwargs.get("limit", 0)))
        return COORDINATED


@pytest.fixture
def mcp_guard(monkeypatch):
    """The MCP handler with both sides of the guard stubbed out.

    Yields ``(spy_calls, coordinator_slot)``: appending a coordinator to the
    slot is how a test says "the flag is on and a coordinator was built".
    """
    from repowise.server.mcp_server import tool_search

    asked: list[int] = []
    slot: list[Any] = []

    async def _spy() -> Any:
        asked.append(1)
        return slot[0] if slot else None

    async def _stock(*args: Any, **kwargs: Any) -> dict:
        return STOCK

    monkeypatch.setattr(tool_search, "mcp_coordinator", _spy)
    monkeypatch.setattr(tool_search, "_structured_search", _stock)
    monkeypatch.delenv(SOURCE_SEARCH_ENV, raising=False)
    return asked, slot


# ---------------------------------------------------------------------------
# MCP: search_codebase
# ---------------------------------------------------------------------------


async def test_mcp_flag_off_never_asks_for_a_coordinator(mcp_guard):
    from repowise.server.mcp_server.tool_search import search_codebase

    asked, _ = mcp_guard
    assert await search_codebase("MyClass.method") == STOCK
    assert asked == []


async def test_mcp_flag_on_delegates(mcp_guard, monkeypatch):
    from repowise.server.mcp_server.tool_search import search_codebase

    asked, slot = mcp_guard
    coordinator = _Coordinator()
    slot.append(coordinator)
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")

    assert await search_codebase("how retrieval works", limit=7, mode="hybrid") == COORDINATED
    assert asked == [1]
    assert coordinator.queries == [("how retrieval works", 7)]


async def test_mcp_falls_back_when_the_repo_has_no_source_index(mcp_guard, monkeypatch):
    """No index is not an error — it is the stock product, unchanged."""
    from repowise.server.mcp_server.tool_search import search_codebase

    asked, _ = mcp_guard
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")

    assert await search_codebase("how retrieval works", mode="hybrid") == STOCK
    assert asked == [1]


@pytest.mark.parametrize("query", ["src/pkg/module.py", "MyClass.method"])
async def test_mcp_exact_queries_stay_on_the_stock_resolver(mcp_guard, monkeypatch, query):
    """Exact file and symbol intent must not become a semantic guess."""
    from repowise.server.mcp_server.tool_search import search_codebase

    asked, slot = mcp_guard
    slot.append(_Coordinator())
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")

    assert await search_codebase(query) == STOCK
    assert asked == []


# ---------------------------------------------------------------------------
# REST: GET /api/search
# ---------------------------------------------------------------------------


class _Store:
    """Stands in for either stock leg — both are asked the same way."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, limit: int = 10) -> list:
        self.queries.append(query)
        return []


def _request() -> Any:
    state = SimpleNamespace(
        db_url="",
        vector_store=None,
        fts=None,
        workspace_fts={},
        workspace_config=None,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.fixture
def rest_guard(monkeypatch):
    from repowise.server.routers import search as search_router

    asked: list[int] = []
    slot: list[Any] = []

    async def _spy(app_state: Any) -> Any:
        asked.append(1)
        return slot[0] if slot else None

    monkeypatch.setattr(search_router, "rest_coordinator", _spy)
    monkeypatch.delenv(SOURCE_SEARCH_ENV, raising=False)
    return asked, slot


async def test_rest_flag_off_never_asks_for_a_coordinator(rest_guard):
    from repowise.server.routers.search import search

    asked, _ = rest_guard
    fts, vectors = _Store(), _Store()
    result = await search(
        request=_request(),
        query="how retrieval works",
        search_type="semantic",
        limit=10,
        repo_id=None,
        fts=fts,
        vector_store=vectors,
    )
    assert result == []
    assert asked == []
    assert vectors.queries == ["how retrieval works"]


async def test_rest_flag_on_delegates_and_serves_the_envelope(rest_guard, monkeypatch):
    from fastapi.responses import JSONResponse

    from repowise.server.routers.search import search

    asked, slot = rest_guard
    coordinator = _Coordinator()
    slot.append(coordinator)
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")

    fts, vectors = _Store(), _Store()
    result = await search(
        request=_request(),
        query="how retrieval works",
        search_type="semantic",
        limit=4,
        repo_id=None,
        fts=fts,
        vector_store=vectors,
    )
    assert isinstance(result, JSONResponse)
    assert asked == [1]
    assert coordinator.queries == [("how retrieval works", 4)]
    # The stock legs never ran.
    assert fts.queries == [] and vectors.queries == []


async def test_rest_fulltext_is_left_alone(rest_guard, monkeypatch):
    """``fulltext`` names one index; a fusion would answer a different question."""
    from repowise.server.routers.search import search

    asked, slot = rest_guard
    slot.append(_Coordinator())
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")

    fts, vectors = _Store(), _Store()
    result = await search(
        request=_request(),
        query="how retrieval works",
        search_type="fulltext",
        limit=10,
        repo_id=None,
        fts=fts,
        vector_store=vectors,
    )
    assert result == []
    assert asked == []
    assert fts.queries == ["how retrieval works"]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_a_repository_with_no_source_index_builds_no_coordinator(tmp_path):
    from repowise.server.source_search_wiring import _build

    assert _build(tmp_path, SimpleNamespace(_embedder=object()), None) is None


def test_a_mock_embedder_builds_no_coordinator(tmp_path):
    """A keyless store cannot query a corpus a real embedder wrote."""
    from repowise.core.source_search.fts import SourceFTSIndex, default_fts_path
    from repowise.core.source_search.manifest import default_manifest_path
    from repowise.server.source_search_wiring import _build

    SourceFTSIndex(default_fts_path(tmp_path)).close()
    default_manifest_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    default_manifest_path(tmp_path).write_text("{}", encoding="utf-8")

    from repowise.core.providers.embedding.base import KeylessEmbedder

    assert _build(tmp_path, SimpleNamespace(_embedder=KeylessEmbedder()), None) is None


async def test_wiki_tombstone_lookup_reads_the_current_page_state(tmp_path):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    from repowise.core.persistence import get_session, init_db, upsert_page, upsert_repository
    from repowise.server.source_search_wiring import _wiki_tombstone_ids

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        await init_db(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with get_session(factory) as session:
            repo = await upsert_repository(session, name="repo", local_path=str(tmp_path))
            for page_id, status in (
                ("file_page:gone.py", "tombstone"),
                ("file_page:live.py", "fresh"),
            ):
                await upsert_page(
                    session,
                    page_id=page_id,
                    repository_id=repo.id,
                    page_type="file_page",
                    title=page_id,
                    content=page_id,
                    target_path=page_id.split(":", 1)[1],
                    source_hash="h",
                    model_name="mock",
                    provider_name="mock",
                    freshness_status=status,
                )

        assert await _wiki_tombstone_ids(tmp_path, factory) == frozenset({"file_page:gone.py"})
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("db_url", "expected"),
    [
        ("sqlite+aiosqlite:////repo/.repowise/wiki.db", "/repo"),
        ("sqlite+aiosqlite:////repo/elsewhere/wiki.db", None),
        ("postgresql+asyncpg://host/db", None),
        ("", None),
    ],
)
def test_the_rest_repo_root_is_read_back_off_the_database_path(db_url, expected):
    from repowise.server.source_search_wiring import _repo_root_from_db_url

    result = _repo_root_from_db_url(db_url)
    assert (str(result) if result is not None else None) == expected
