"""Retiring a module page whose code is gone, without emptying the wiki.

``update --index-only`` on a deterministic wiki re-renders file pages and
leaves every repo-wide page frozen. A module page whose directory was deleted
therefore stays ``fresh`` forever and keeps answering searches, because no
sweep on that path ever asks whether its files still exist.

``reconcile_module_pages`` closes that, and the interesting half is what it
refuses to do: a derivation that under-produces looks exactly like a repo that
lost most of its code, so the same mass-deletion floor the file prune uses
stands between a wrong authoritative set and a deleted wiki.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from repowise.core.persistence.crud import upsert_page, upsert_repository
from repowise.core.persistence.database import init_db
from repowise.core.persistence.models import Page
from repowise.core.pipeline.persist import reconcile_module_pages


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


async def _seed(session, *targets: str, page_type: str = "module_page") -> str:
    """One page per target. Returns the repository id."""
    repo = await upsert_repository(session, name="r", local_path="/tmp/r")
    await session.commit()
    for target in targets:
        await upsert_page(
            session,
            page_id=f"{page_type}:{target}",
            repository_id=repo.id,
            page_type=page_type,
            title=target,
            content=f"# {target}",
            summary=f"What {target} does.",
            target_path=target,
            source_hash="h",
            model_name="mock",
            provider_name="mock",
            structural_key=f"concept-{target}",
        )
    await session.commit()
    return repo.id


async def _stored_ids(session, page_type: str = "module_page") -> set[str]:
    rows = await session.execute(select(Page.id).where(Page.page_type == page_type))
    return set(rows.scalars().all())


async def test_module_pages_outside_authoritative_set_are_deleted(session):
    repo_id = await _seed(session, "src/core", "src/gone", "src/also_gone")

    swept, refusals = await reconcile_module_pages(session, repo_id, {"module_page:src/core"})

    assert refusals == []
    assert set(swept) == {"module_page:src/gone", "module_page:src/also_gone"}
    assert await _stored_ids(session) == {"module_page:src/core"}


async def test_authoritative_pages_and_other_page_types_survive(session):
    """The reconcile owns one page type. The rest of the wiki is not its business."""
    repo_id = await _seed(session, "src/core", "src/gone")
    await _seed(session, "src/app.py", page_type="file_page")
    await _seed(session, "cycle-1", page_type="scc_page")
    await _seed(session, "layer:Data", page_type="layer_page")

    swept, refusals = await reconcile_module_pages(session, repo_id, {"module_page:src/core"})

    assert refusals == []
    assert swept == ["module_page:src/gone"]
    assert await _stored_ids(session) == {"module_page:src/core"}
    assert await _stored_ids(session, "file_page") == {"file_page:src/app.py"}
    assert await _stored_ids(session, "scc_page") == {"scc_page:cycle-1"}
    assert await _stored_ids(session, "layer_page") == {"layer_page:layer:Data"}


async def test_floor_refuses_and_records_prune_refusal(session):
    """25 of 30 gone is a broken derivation far more often than a gutted repo."""
    targets = [f"src/m{i}" for i in range(30)]
    repo_id = await _seed(session, *targets)
    keep = {f"module_page:src/m{i}" for i in range(5)}

    swept, refusals = await reconcile_module_pages(session, repo_id, keep)

    assert swept == []
    assert len(refusals) == 1
    refusal = refusals[0]
    assert refusal.table == "wiki_module_pages"
    assert (refusal.candidate_paths, refusal.persisted_paths) == (25, 30)
    # Nothing was deleted: a refusal that still took half the table would be
    # the failure this guard exists to prevent.
    assert len(await _stored_ids(session)) == 30


async def test_floor_does_not_fire_below_twenty_rows(session):
    """The measured live case: 14 dead of 30, under both limits, so it prunes.

    14 clears neither gate on its own — it is below the 20-row floor *and*
    below half of 30 — and the guard needs both. This is the case the whole
    ticket exists for, so it must not be the one the safety net catches.
    """
    targets = [f"src/m{i}" for i in range(30)]
    repo_id = await _seed(session, *targets)
    keep = {f"module_page:src/m{i}" for i in range(14, 30)}

    swept, refusals = await reconcile_module_pages(session, repo_id, keep)

    assert refusals == []
    assert len(swept) == 14
    assert await _stored_ids(session) == keep


async def test_accept_mass_deletion_overrides_the_floor(session):
    targets = [f"src/m{i}" for i in range(30)]
    repo_id = await _seed(session, *targets)
    keep = {"module_page:src/m0"}

    swept, refusals = await reconcile_module_pages(
        session, repo_id, keep, accept_mass_deletion=True
    )

    assert refusals == []
    assert len(swept) == 29
    assert await _stored_ids(session) == keep


async def test_no_stale_pages_is_a_no_op(session):
    repo_id = await _seed(session, "src/core")

    swept, refusals = await reconcile_module_pages(
        session, repo_id, {"module_page:src/core", "module_page:src/new"}
    )

    assert (swept, refusals) == ([], [])
    assert await _stored_ids(session) == {"module_page:src/core"}
