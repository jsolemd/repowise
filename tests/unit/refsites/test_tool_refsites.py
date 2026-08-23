"""The MCP surface returns occurrences and never returns a bare empty success.

Locks the two response contracts that matter to an agent reading this tool:
an empty ``sites`` list always comes with a ``status`` and a coverage block
explaining which of the three possible reasons it is, and the rename preview
declares in its own payload that it changes nothing.
"""

from __future__ import annotations

import pytest

from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.store import SqlReferenceSiteStore
from repowise.server.mcp_server.tool_refsites import (
    get_reference_sites,
    preview_symbol_rename,
)
from tests.unit.refsites.conftest import TARGET_SYMBOL_ID


@pytest.fixture
async def indexed(mcp_state, async_session, repo_id, toy_repo):
    store = SqlReferenceSiteStore(async_session)
    await store.replace_repository(repo_id, extract_repository(toy_repo))
    await async_session.commit()
    return repo_id


async def test_sites_are_returned_per_occurrence(indexed):
    result = await get_reference_sites(TARGET_SYMBOL_ID)

    assert result["status"] == "coverage_limited"
    assert result["total"] == 11
    assert result["counts_by_kind"]["call"] == 6
    assert result["counts_by_kind"]["definition"] == 1

    # Four occurrences in one file, each with its own position.
    widget = [site for site in result["sites"] if site["file"] == "widget.ts"]
    assert len(widget) == 5
    assert len({(site["start_line"], site["start_col"]) for site in widget}) == 5


async def test_bare_name_resolves_when_unique(indexed):
    result = await get_reference_sites("computeTotal")

    assert result["target"]["matched_by"] == "name"
    assert result["target"]["symbol_id"] == TARGET_SYMBOL_ID
    assert result["total"] == 11


async def test_ambiguous_name_returns_candidates(indexed):
    """``compute_total`` is defined in both helpers.py and deploy.sh."""
    result = await get_reference_sites("compute_total")

    assert result["status"] == "ambiguous"
    assert set(result["candidates"]) == {
        "deploy.sh::compute_total",
        "helpers.py::compute_total",
    }
    assert result["sites"] == []
    assert "retry" in result["explanation"]


async def test_coverage_block_names_the_unreadable_language(indexed):
    result = await get_reference_sites(TARGET_SYMBOL_ID)
    coverage = result["coverage"]

    assert coverage["uncovered_languages_present"] == ["yaml"]
    assert set(coverage["partial_languages_present"]) == {"shell", "sql"}
    assert coverage["complete"] is False

    yaml = next(row for row in coverage["observed"] if row["language"] == "yaml")
    assert yaml["files_seen"] == 1
    assert yaml["sites"] == 0
    assert "no tree-sitter query file" in yaml["reason"]

    # The static gate is reported alongside what was observed here.
    declared = {row["language"] for row in coverage["declared"]}
    assert declared == {"python", "typescript", "javascript", "shell", "sql"}


async def test_empty_result_is_never_a_bare_success(indexed):
    """A name with no sites still explains itself."""
    result = await get_reference_sites("no_such_symbol_anywhere")

    assert result["status"] == "not_found"
    assert result["sites"] == []
    assert result["explanation"]


async def test_unindexed_repository_says_so(mcp_state):
    """No extraction pass has run: that is not the same as no references."""
    result = await get_reference_sites(TARGET_SYMBOL_ID)

    assert result["status"] == "not_indexed"
    assert result["sites"] == []
    assert "does not mean the symbol is unreferenced" in result["explanation"]

    preview = await preview_symbol_rename(TARGET_SYMBOL_ID)
    assert preview["status"] == "not_indexed"
    assert preview["applies_changes"] is False


async def test_unindexed_repository_says_so_for_a_bare_name_too(mcp_state):
    """The bare-name path must not report an empty store as ``not_found``.

    A symbol id short-circuits target resolution, so it reached the
    "is anything stored" check no matter where that check sat. A bare name
    does not: it resolves by looking for a definition row, finds none because
    nothing was ever extracted, and used to return
    ``not_found: No definition named 'computeTotal' is recorded`` — an answer
    that is indistinguishable from a genuinely absent symbol, which is the one
    thing this tool exists to never do.
    """
    for query in ("computeTotal", "no_such_symbol_anywhere"):
        result = await get_reference_sites(query)
        assert result["status"] == "not_indexed", query
        assert "does not mean the symbol is unreferenced" in result["explanation"]

        preview = await preview_symbol_rename(query)
        assert preview["status"] == "not_indexed", query


async def test_kind_filter_and_confidence_floor(indexed):
    calls = await get_reference_sites(TARGET_SYMBOL_ID, kinds=["call"])
    assert {site["kind"] for site in calls["sites"]} == {"call"}

    certain = await get_reference_sites(TARGET_SYMBOL_ID, min_confidence=0.90)
    assert all(site["confidence"] >= 0.90 for site in certain["sites"])
    assert certain["total"] < calls["total"] + 5


async def test_pagination_reports_the_whole_total(indexed):
    page = await get_reference_sites(TARGET_SYMBOL_ID, limit=3)

    assert len(page["sites"]) == 3
    assert page["total"] == 11
    assert page["pagination"]["has_more"] is True
    assert page["pagination"]["next_offset"] == 3


async def test_rename_preview_declares_that_it_changes_nothing(indexed):
    preview = await preview_symbol_rename(TARGET_SYMBOL_ID, new_name="totalOf")

    assert preview["applies_changes"] is False
    assert preview["old_name"] == "computeTotal"
    assert preview["new_name"] == "totalOf"
    assert preview["summary"]["total"] == 11
    # This fixture's repository row points at a directory that is not a git
    # checkout, so working-tree freshness cannot be established — and an
    # unverifiable position is never asserted mechanically safe. The
    # verified-tree split (10 safe / 1 review) lives in
    # ``test_tool_freshness.py`` against a real committed checkout.
    assert preview["summary"]["mechanically_safe"] == 0
    assert preview["summary"]["needs_review"] == 11
    assert preview["_meta"]["working_tree"]["checked"] is False
    assert preview["files_touched"] == ["calc.ts", "legacy.js", "panel.tsx", "widget.ts"]


async def test_rename_preview_carries_its_caveats(indexed):
    preview = await preview_symbol_rename(TARGET_SYMBOL_ID)
    joined = " ".join(preview["caveats"])

    assert "yaml" in joined
    assert "invisible" in joined
    assert "not mechanically safe" in joined


async def test_rename_preview_marks_the_recovered_sibling_unsafe(indexed):
    preview = await preview_symbol_rename(TARGET_SYMBOL_ID)
    sibling = next(site for site in preview["sites"] if site["occurrence_index"] > 0)

    assert sibling["file"] == "widget.ts"
    assert sibling["start_line"] == 7
    assert sibling["tier"] == "textual"
    assert sibling["mechanically_safe"] is False


async def test_repo_all_is_refused_by_both_tools(indexed):
    assert "error" in await get_reference_sites(TARGET_SYMBOL_ID, repo="all")
    assert "error" in await preview_symbol_rename(TARGET_SYMBOL_ID, repo="all")
