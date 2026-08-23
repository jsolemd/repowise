"""Rename preview reports every site, from the store, and applies nothing.

The three mutation legs are the load-bearing tests here:

(a) delete one occurrence row and the preview must lose exactly that site —
    which is only true if the preview reads the store rather than the graph;
(b) an unresolvable reference must appear at low confidence rather than
    disappear;
(c) a symbol whose references live in an uncovered language must produce an
    explicit coverage disclosure, never a clean empty result.
"""

from __future__ import annotations

import pytest
from sqlalchemy import delete, select

from repowise.core.refsites.coverage import CoverageStatus
from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.rename import preview_rename
from repowise.core.refsites.schema import ReferenceSite
from repowise.core.refsites.store import SqlReferenceSiteStore
from repowise.core.refsites.taxonomy import ReferenceKind, ReferenceOrigin
from tests.unit.refsites.conftest import COMPUTE_TOTAL_GROUND_TRUTH, TARGET_SYMBOL_ID


@pytest.fixture
async def store(async_session, repo_id, toy_repo):
    sql_store = SqlReferenceSiteStore(async_session)
    await sql_store.replace_repository(repo_id, extract_repository(toy_repo))
    await async_session.commit()
    return sql_store


def _triples(preview):
    return tuple(
        sorted(
            (site.file_path, site.start_line, str(site.kind))
            for site in preview.sites
            if site.name == "computeTotal"
        )
    )


async def test_preview_matches_hand_counted_ground_truth(store, repo_id):
    """Every site of ``computeTotal``, and no others."""
    preview = await preview_rename(store, repo_id, TARGET_SYMBOL_ID)

    assert preview.old_name == "computeTotal"
    assert _triples(preview) == COMPUTE_TOTAL_GROUND_TRUTH
    assert preview.files_touched == ("calc.ts", "legacy.js", "panel.tsx", "widget.ts")


async def test_preview_orders_by_confidence_and_flags_what_is_safe(store, repo_id):
    preview = await preview_rename(store, repo_id, TARGET_SYMBOL_ID)

    confidences = [site.confidence for site in preview.sites]
    assert confidences == sorted(confidences, reverse=True)

    definition = preview.sites[0]
    assert definition.kind is ReferenceKind.DEFINITION
    assert definition.confidence == 1.0
    assert definition.mechanically_safe

    # The recovered same-line sibling is real but not safe to rewrite blind.
    sibling = next(site for site in preview.sites if site.occurrence_index > 0)
    assert sibling.confidence == 0.40
    assert not sibling.mechanically_safe
    assert sibling.range_exact


# --------------------------------------------------------------------------
# Mutation leg (a) — the preview reads the store, not the graph
# --------------------------------------------------------------------------


async def test_deleting_one_row_removes_exactly_that_site(store, async_session, repo_id):
    """Delete a single occurrence row; the preview loses that site and no other.

    The row is deleted directly through the ORM rather than through the store's
    API, so nothing in the read path can compensate. If the preview were built
    from graph edges — where all four ``widget.ts`` calls collapse onto one
    edge — deleting one occurrence would change nothing, or everything.
    """
    before = await preview_rename(store, repo_id, TARGET_SYMBOL_ID)
    victim = next(
        site
        for site in before.sites
        if site.file_path == "widget.ts"
        and site.start_line == 4
        and site.kind is ReferenceKind.CALL
    )

    deleted = await async_session.execute(
        delete(ReferenceSite).where(
            ReferenceSite.repository_id == repo_id,
            ReferenceSite.file_path == victim.file_path,
            ReferenceSite.start_line == victim.start_line,
            ReferenceSite.start_col == victim.start_col,
            ReferenceSite.name == victim.name,
            ReferenceSite.kind == str(victim.kind),
        )
    )
    assert deleted.rowcount == 1
    await async_session.commit()

    after = await preview_rename(store, repo_id, TARGET_SYMBOL_ID)

    assert len(after.sites) == len(before.sites) - 1
    lost = set(_triples(before)) - set(_triples(after))
    assert lost == {("widget.ts", 4, "call")}
    # Every other site is untouched, including the other widget.ts calls that
    # a graph edge would have shared a row with.
    assert ("widget.ts", 7, "call") in _triples(after)
    assert ("widget.ts", 13, "call") in _triples(after)


async def test_deleting_the_definition_row_is_reported_as_a_caveat(store, async_session, repo_id):
    """Losing the definition site changes the preview's own account of itself."""
    await async_session.execute(
        delete(ReferenceSite).where(
            ReferenceSite.repository_id == repo_id,
            ReferenceSite.target_symbol_id == TARGET_SYMBOL_ID,
            ReferenceSite.kind == str(ReferenceKind.DEFINITION),
        )
    )
    await async_session.commit()

    preview = await preview_rename(store, repo_id, TARGET_SYMBOL_ID)

    assert not any(site.kind is ReferenceKind.DEFINITION for site in preview.sites)
    assert any("No definition site is recorded" in caveat for caveat in preview.caveats)
    # The name still resolves, from the symbol id, so the preview stays usable.
    assert preview.old_name == "computeTotal"


# --------------------------------------------------------------------------
# Mutation leg (b) — an unresolvable reference surfaces, it does not vanish
# --------------------------------------------------------------------------


async def test_unresolvable_reference_surfaces_at_low_confidence(store, repo_id):
    """``makeUser`` binds to nothing; both its sites are still reported."""
    preview = await preview_rename(store, repo_id, "widget.ts::makeUser")

    assert preview.old_name == "makeUser"
    assert len(preview.sites) == 2
    assert {(site.start_line, str(site.kind)) for site in preview.sites} == {
        (10, str(ReferenceKind.IDENTIFIER)),
        (12, str(ReferenceKind.CALL)),
    }
    assert all(site.confidence <= 0.25 for site in preview.sites)
    assert not any(site.mechanically_safe for site in preview.sites)
    assert any("not mechanically safe" in caveat for caveat in preview.caveats)
    assert any("anonymous callables" in caveat for caveat in preview.caveats)


async def test_unresolved_call_is_not_filtered_out_by_a_confidence_floor(store, repo_id):
    """A floor removes sites; it must never be the reason one was never there."""
    unfiltered = await preview_rename(store, repo_id, "app.py::orphan")
    assert unfiltered.sites

    orphan_call = await preview_rename(store, repo_id, "app.py::not_a_real_function")
    assert orphan_call.old_name == "not_a_real_function"
    assert len(orphan_call.sites) == 1
    assert orphan_call.sites[0].origin is ReferenceOrigin.UNRESOLVED
    assert orphan_call.sites[0].confidence == 0.20

    # Raising the floor removes it — and that is a filter the caller asked for,
    # visibly, not a silent absence.
    filtered = await preview_rename(
        store, repo_id, "app.py::not_a_real_function", min_confidence=0.5
    )
    assert filtered.sites == ()


async def test_string_reference_is_reported_as_a_lead(store, repo_id):
    """The name inside a string appears, at the floor, marked unsafe."""
    preview = await preview_rename(store, repo_id, "helpers.py::compute_total")
    string_sites = [site for site in preview.sites if site.kind is ReferenceKind.STRING_REF]

    assert len(string_sites) == 1
    assert string_sites[0].file_path == "app.py"
    assert string_sites[0].confidence == 0.10
    assert not string_sites[0].mechanically_safe


# --------------------------------------------------------------------------
# Mutation leg (c) — uncovered languages disclose, never return empty-success
# --------------------------------------------------------------------------


async def test_uncovered_language_is_disclosed_in_every_preview(store, repo_id):
    """The YAML file is named as a blind spot even when the rename looks clean."""
    preview = await preview_rename(store, repo_id, TARGET_SYMBOL_ID)

    yaml = next(record for record in preview.coverage if record.language == "yaml")
    assert yaml.status is CoverageStatus.NONE
    assert yaml.files_seen == 1
    assert any("yaml" in caveat and "invisible" in caveat for caveat in preview.caveats)


async def test_symbol_referenced_only_from_yaml_is_not_an_empty_success(store, repo_id):
    """The one real reference lives in a file we cannot read. Say so.

    ``yaml_only_helper`` is called from ``pipeline.yaml`` and nowhere else. A
    preview that returned "one site, the definition, all clear" would be
    actively misleading, so the coverage block and the caveats have to carry
    the blind spot.
    """
    preview = await preview_rename(store, repo_id, "app.py::yaml_only_helper")

    assert [str(site.kind) for site in preview.sites] == [str(ReferenceKind.DEFINITION)]
    assert preview.caveats, "a preview with an unreadable language present must caveat"
    blind = [caveat for caveat in preview.caveats if "yaml" in caveat]
    assert blind and "invisible" in blind[0]
    assert "no tree-sitter query file" in blind[0]


async def test_partial_language_coverage_is_disclosed(store, repo_id):
    """Shell and SQL are declared partial and say what they cannot do."""
    preview = await preview_rename(store, repo_id, "deploy.sh::compute_total")
    joined = " ".join(preview.caveats)

    assert "shell coverage is partial" in joined
    assert "sql coverage is partial" in joined


# --------------------------------------------------------------------------
# The preview writes nothing
# --------------------------------------------------------------------------


async def test_preview_mutates_nothing(store, async_session, repo_id):
    """A preview is evidence. Running one leaves the store byte-identical."""

    def snapshot():
        return async_session.execute(
            select(
                ReferenceSite.file_path,
                ReferenceSite.start_line,
                ReferenceSite.start_col,
                ReferenceSite.name,
                ReferenceSite.kind,
                ReferenceSite.confidence,
                ReferenceSite.target_symbol_id,
            )
            .where(ReferenceSite.repository_id == repo_id)
            .order_by(ReferenceSite.file_path, ReferenceSite.start_line, ReferenceSite.start_col)
        )

    before = (await snapshot()).all()
    await preview_rename(store, repo_id, TARGET_SYMBOL_ID, new_name="totalOf")
    await preview_rename(store, repo_id, "helpers.py::compute_total")
    after = (await snapshot()).all()

    assert before == after


async def test_new_name_collision_is_reported(store, repo_id):
    """Renaming onto an existing name is a caveat, not a silent overwrite."""
    preview = await preview_rename(store, repo_id, TARGET_SYMBOL_ID, new_name="widgetTotal")

    assert any("already spelled" in caveat for caveat in preview.caveats)
    assert preview.new_name == "widgetTotal"


async def test_no_collision_caveat_for_a_fresh_name(store, repo_id):
    preview = await preview_rename(store, repo_id, TARGET_SYMBOL_ID, new_name="nothingSpellsThis")
    assert not any("already spelled" in caveat for caveat in preview.caveats)
