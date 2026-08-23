"""Reference sites survive the persistence round trip.

Locks the store's write contract (replace, never upsert-by-position), the
namespace split between module paths and symbol names, and the coverage rows
that keep an unreadable language from looking like an unreferenced symbol.
"""

from __future__ import annotations

import pytest
from sqlalchemy import delete, func, select

from repowise.core.refsites.coverage import CoverageStatus
from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.records import ExtractionResult
from repowise.core.refsites.schema import ReferenceSite, ReferenceSiteCoverage
from repowise.core.refsites.store import SqlReferenceSiteStore
from repowise.core.refsites.taxonomy import ReferenceKind


@pytest.fixture
async def stored(async_session, repo_id, toy_repo):
    """Extract the toy repo into the store and hand back everything needed."""
    result = extract_repository(toy_repo)
    store = SqlReferenceSiteStore(async_session)
    await store.replace_repository(repo_id, result)
    await async_session.commit()
    return store, repo_id, result


async def test_round_trip_preserves_every_site(stored, async_session, repo_id):
    store, _, result = stored
    assert await store.count(repo_id) == len(result.sites)

    written = await async_session.execute(
        select(func.count())
        .select_from(ReferenceSite)
        .where(ReferenceSite.repository_id == repo_id)
    )
    assert written.scalar_one() == len(result.sites)


async def test_confidence_is_persisted_from_the_rubric(stored, async_session, repo_id):
    """No stored float differs from the rubric value its origin implies."""
    from repowise.core.refsites.taxonomy import ORIGIN_CONFIDENCE, ReferenceOrigin

    rows = (
        await async_session.execute(
            select(ReferenceSite.resolution_origin, ReferenceSite.confidence).where(
                ReferenceSite.repository_id == repo_id
            )
        )
    ).all()

    assert rows
    for origin, confidence in rows:
        assert confidence == pytest.approx(ORIGIN_CONFIDENCE[ReferenceOrigin(origin)])


async def test_sites_for_target_are_ordered_by_confidence(stored, repo_id):
    store, _, _ = stored
    sites = await store.sites_for_target(repo_id, "calc.ts::computeTotal")

    assert sites
    assert [site.confidence for site in sites] == sorted(
        (site.confidence for site in sites), reverse=True
    )
    assert sites[0].kind is ReferenceKind.DEFINITION


async def test_min_confidence_filters_in_sql(stored, repo_id):
    store, _, _ = stored
    everything = await store.sites_for_target(repo_id, "calc.ts::computeTotal")
    certain = await store.sites_for_target(repo_id, "calc.ts::computeTotal", min_confidence=0.90)

    assert len(certain) < len(everything)
    assert all(site.confidence >= 0.90 for site in certain)


async def test_module_paths_do_not_answer_symbol_name_lookups(stored, repo_id):
    """An ``import helpers`` statement is not a site of a symbol named helpers."""
    store, _, _ = stored
    symbol_namespace = await store.sites_named(repo_id, "helpers")
    module_namespace = await store.sites_named(repo_id, "helpers", include_module_paths=True)

    assert all(site.kind is not ReferenceKind.IMPORT for site in symbol_namespace)
    assert any(site.kind is ReferenceKind.IMPORT for site in module_namespace)


async def test_coverage_rows_record_the_unreadable_language(stored, repo_id):
    store, _, _ = stored
    coverage = {record.language: record for record in await store.coverage(repo_id)}

    assert coverage["yaml"].status is CoverageStatus.NONE
    assert coverage["yaml"].files_seen == 1
    assert coverage["yaml"].sites == 0
    assert coverage["shell"].limitations


async def test_replace_repository_is_idempotent(stored, async_session, repo_id, toy_repo):
    """Re-extracting the same tree leaves the same row count, not double."""
    store, _, result = stored
    await store.replace_repository(repo_id, extract_repository(toy_repo))
    await async_session.commit()

    assert await store.count(repo_id) == len(result.sites)
    coverage_rows = (
        await async_session.execute(
            select(func.count())
            .select_from(ReferenceSiteCoverage)
            .where(ReferenceSiteCoverage.repository_id == repo_id)
        )
    ).scalar_one()
    assert coverage_rows == len(result.coverage)


async def test_replace_files_touches_only_the_named_files(stored, async_session, repo_id):
    """Per-file replacement is how incremental re-extraction stays correct."""
    store, _, _ = stored
    before_widget = await store.sites_in_file(repo_id, "widget.ts")
    before_calc = await store.sites_in_file(repo_id, "calc.ts")
    assert before_widget and before_calc

    await store.replace_files(repo_id, ["widget.ts"], [])
    await async_session.commit()

    assert await store.sites_in_file(repo_id, "widget.ts") == []
    assert len(await store.sites_in_file(repo_id, "calc.ts")) == len(before_calc)


async def test_positions_are_unique_per_kind(stored, async_session, repo_id):
    """No two rows claim the same (file, line, col, name, kind)."""
    rows = (
        await async_session.execute(
            select(
                ReferenceSite.file_path,
                ReferenceSite.start_line,
                ReferenceSite.start_col,
                ReferenceSite.name,
                ReferenceSite.kind,
            ).where(ReferenceSite.repository_id == repo_id)
        )
    ).all()
    assert len(rows) == len(set(rows))


async def test_empty_extraction_clears_the_store(stored, async_session, repo_id):
    store, _, _ = stored
    await store.replace_repository(repo_id, ExtractionResult(sites=(), coverage=()))
    await async_session.commit()

    assert await store.count(repo_id) == 0
    assert await store.coverage(repo_id) == []


async def test_deleting_the_repository_cascades(stored, async_session, repo_id):
    """Sites are owned by their repository, not orphaned when it goes."""
    from repowise.core.persistence.models import Repository

    store, _, _ = stored
    assert await store.count(repo_id) > 0

    await async_session.execute(delete(Repository).where(Repository.id == repo_id))
    await async_session.commit()

    assert await store.count(repo_id) == 0
