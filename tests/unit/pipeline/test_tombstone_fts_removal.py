"""A tombstoned page leaves the full-text index instead of holding a slot.

A tombstone documents a file that no longer exists, and every serving layer
already drops one: hydration in the answer pipeline discards it, and the
search tools filter it out. But retrieval fetches a fixed number of rows
*before* any of that runs, so a tombstone still occupies one of those slots
and pushes a real candidate out of the fetch entirely. The page cannot be an
answer either way; the cost is the page it displaces.

Deleting the row is the only fix that works before the fetch. Filtering
afterwards is what already happens, and it is too late.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from repowise.core.persistence.crud import upsert_page, upsert_repository
from repowise.core.persistence.database import init_db
from repowise.core.persistence.models import Page
from repowise.core.persistence.search import FullTextSearch
from repowise.core.pipeline.persist import mark_tombstone_pages, tombstone_candidates


@pytest.fixture
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest.fixture
async def session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as sess:
        yield sess


async def _seed(session, fts: FullTextSearch, *paths: str) -> str:
    repo = await upsert_repository(session, name="r", local_path="/tmp/r")
    await session.commit()
    for path in paths:
        await upsert_page(
            session,
            page_id=f"file_page:{path}",
            repository_id=repo.id,
            page_type="file_page",
            title=f"File: {path}",
            content=f"# Overview\n\nThe module at {path} handles xylophone tuning.",
            summary=f"What {path} does.",
            target_path=path,
            source_hash="h",
            model_name="mock",
            provider_name="mock",
        )
        await fts.index(
            f"file_page:{path}",
            f"File: {path}",
            f"# Overview\n\nThe module at {path} handles xylophone tuning.",
            summary=f"What {path} does.",
            target_path=path,
        )
    await session.commit()
    return repo.id


async def test_marking_returns_the_page_ids_it_marked(session, engine):
    """The caller cannot delete rows it has not been told about.

    The count this used to return is derivable from the list; the list is not
    derivable from the count.
    """
    fts = FullTextSearch(engine)
    await fts.ensure_index()
    repo_id = await _seed(session, fts, "src/gone.py", "src/kept.py")

    marked = await mark_tombstone_pages(session, repo_id, [("src/gone.py", [])])

    assert marked == ["file_page:src/gone.py"]


async def test_a_tombstoned_page_is_deleted_from_the_full_text_index(session, engine):
    fts = FullTextSearch(engine)
    await fts.ensure_index()
    repo_id = await _seed(session, fts, "src/gone.py", "src/kept.py")
    assert len(await fts.search("xylophone", limit=10)) == 2

    marked = await mark_tombstone_pages(session, repo_id, [("src/gone.py", [])])
    await session.commit()
    await fts.delete_many(marked)

    assert [r.page_id for r in await fts.search("xylophone", limit=10)] == ["file_page:src/kept.py"]


async def test_a_deleted_file_tombstones_every_file_derived_page_and_fts_row(session, engine):
    """File, spotlight, API, and infrastructure pages leave serving together."""
    fts = FullTextSearch(engine)
    await fts.ensure_index()
    repo_id = await _seed(session, fts, "src/gone.py", "src/kept.py")
    derived = (
        (
            "symbol_spotlight:src/gone.py::Widget.render",
            "symbol_spotlight",
            "src/gone.py::Widget.render",
        ),
        ("api_contract:src/gone.py", "api_contract", "src/gone.py"),
        ("infra_page:src/gone.py", "infra_page", "src/gone.py"),
        # A structural target is not a file even when its spelling collides
        # with one. It must not inherit the file's tombstone.
        ("module_page:src/gone.py", "module_page", "src/gone.py"),
    )
    for page_id, page_type, target_path in derived:
        await upsert_page(
            session,
            page_id=page_id,
            repository_id=repo_id,
            page_type=page_type,
            title=page_id,
            content=f"# {page_id}\n\nTunes xylophones.",
            summary="The derived page.",
            target_path=target_path,
            source_hash="h",
            model_name="mock",
            provider_name="mock",
        )
        await fts.index(
            page_id,
            page_id,
            f"# {page_id}\n\nTunes xylophones.",
            summary="The derived page.",
            target_path=target_path,
        )
    await session.commit()

    marked = await mark_tombstone_pages(session, repo_id, [("src/gone.py", [])])
    await session.commit()
    await fts.delete_many(marked)

    expected = {
        "file_page:src/gone.py",
        "symbol_spotlight:src/gone.py::Widget.render",
        "api_contract:src/gone.py",
        "infra_page:src/gone.py",
    }
    assert set(marked) == expected
    rows = await session.execute(select(Page.id, Page.freshness_status).where(Page.id.in_(marked)))
    assert dict(rows.all()) == {page_id: "tombstone" for page_id in expected}
    structural = await session.execute(
        select(Page.freshness_status).where(Page.id == "module_page:src/gone.py")
    )
    assert structural.scalar_one() == "fresh"
    assert {hit.page_id for hit in await fts.search("xylophone", limit=10)} == {
        "file_page:src/kept.py",
        "module_page:src/gone.py",
    }


async def test_marking_nothing_returns_an_empty_list(session, engine):
    """An empty list, never ``None`` — the caller passes it straight to delete."""
    fts = FullTextSearch(engine)
    await fts.ensure_index()
    repo_id = await _seed(session, fts, "src/kept.py")

    assert await mark_tombstone_pages(session, repo_id, []) == []
    assert await mark_tombstone_pages(session, repo_id, [("src/never-existed.py", [])]) == []


async def test_a_renamed_file_is_dropped_under_its_old_path(session, engine):
    """A rename tombstones the old page, which must leave the index too.

    The new path gets its own page on the next run. Leaving the old one
    searchable means both are candidates for the same file, and only one of
    them describes code that exists.
    """
    fts = FullTextSearch(engine)
    await fts.ensure_index()
    repo_id = await _seed(session, fts, "src/old.py")
    diffs = [SimpleNamespace(status="renamed", path="src/new.py", old_path="src/old.py")]

    marked = await mark_tombstone_pages(session, repo_id, tombstone_candidates(diffs))
    await session.commit()
    await fts.delete_many(marked)

    assert await fts.search("xylophone", limit=10) == []
