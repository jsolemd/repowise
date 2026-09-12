"""Focused tests for doc-search search scoring helpers."""

import pytest

from repowise.docs.config import Settings
from repowise.docs.indexer.scoring import apply_exact_match_boosts, apply_search_boosts
from repowise.docs.search.intent import QueryIntent


@pytest.mark.unit
def test_apply_search_boosts_combines_relevance_signals(mock_qdrant_point):
    """Boosts from category, intent, path, heading, and token count should compose."""
    point = mock_qdrant_point(
        "chunk-1",
        score=1.0,
        doc_category="reference",
        file_type="source",
        file_path="docs/sdk/python.mdx",
        heading_level="2",
        token_count="500",
    )

    settings = Settings(
        doc_category_boosts={"reference": 1.25, "other": 1.0},
        file_type_boosts={"source": 0.90, "other": 1.0},
        intent_file_type_boosts={"implementation": {"source": 1.25}},
        path_hierarchy_boosts={r"(?:^|/)docs/sdk/": 1.25, r"(?:^|/)docs/": 1.10},
        heading_level_boosts={2: 1.10},
        token_count_thresholds={"min_substantive": 400, "max_fragment": 100},
        token_count_boosts={"substantive": 1.05, "fragment": 0.90, "normal": 1.0},
    )

    apply_search_boosts([point], settings, intent=QueryIntent.IMPLEMENTATION)

    assert point.score == pytest.approx(2.436328125)


@pytest.mark.unit
def test_apply_search_boosts_skips_invalid_path_regex(mock_qdrant_point, caplog):
    """A malformed path regex should be ignored without aborting scoring."""
    point = mock_qdrant_point("chunk-2", score=1.0, file_path="docs/api.mdx")

    settings = Settings(
        path_hierarchy_boosts={"(": 9.0, r"(?:^|/)docs/": 1.10},
    )

    with caplog.at_level("WARNING"):
        apply_search_boosts([point], settings)

    assert point.score == pytest.approx(1.10)
    assert any("Invalid path pattern" in message for message in caplog.messages)


@pytest.mark.unit
def test_apply_search_boosts_prefers_surface_owner_for_descriptive_component_query(
    mock_qdrant_point,
):
    """Hybrid concept queries should still favor the explicitly named component page."""
    stack = mock_qdrant_point(
        "stack-reference",
        score=1.0,
        doc_category="reference",
        chunk_type="doc",
        breadcrumb=["Components", "Stack"],
        breadcrumb_text="Components > Stack",
        file_path="core/stack.mdx",
        section_anchor="stack",
    )
    simple_grid = mock_qdrant_point(
        "simple-grid-spacing",
        score=1.2,
        doc_category="guide",
        chunk_type="doc",
        breadcrumb=["Layout", "SimpleGrid"],
        breadcrumb_text="Layout > SimpleGrid",
        file_path="layout/simple-grid.mdx",
        section_anchor="spacing-and-verticalspacing-props",
    )

    apply_search_boosts(
        [simple_grid, stack],
        Settings(),
        intent=QueryIntent.CONCEPT,
        query="Stack component spacing vertical layout",
    )

    assert stack.score > simple_grid.score


@pytest.mark.unit
def test_apply_search_boosts_prefers_exact_component_page_over_semantic_gap_match(
    mock_qdrant_point,
):
    """Component-led concept queries should still prefer the canonical component page."""
    stack = mock_qdrant_point(
        "stack-reference",
        score=0.42,
        doc_category="reference",
        chunk_type="doc",
        breadcrumb=["Components", "Stack"],
        breadcrumb_text="Components > Stack",
        file_path="core/stack.mdx",
        section_anchor="stack",
    )
    section_gaps = mock_qdrant_point(
        "ring-progress-section-gaps",
        score=0.64,
        doc_category="other",
        chunk_type="doc",
        breadcrumb=["RingProgress", "Section gaps"],
        breadcrumb_text="RingProgress > Section gaps",
        file_path="core/ring-progress.mdx",
        section_anchor="section-gaps",
    )

    apply_search_boosts(
        [section_gaps, stack],
        Settings(),
        intent=QueryIntent.CONCEPT,
        query="Stack gap motion layout patterns",
    )

    assert stack.score > section_gaps.score


@pytest.mark.unit
def test_apply_search_boosts_penalizes_error_sections_for_non_debug_queries(mock_qdrant_point):
    """Non-debug concept queries should not prefer troubleshooting headings."""
    selection = mock_qdrant_point(
        "selection-doc",
        score=1.0,
        doc_category="guide",
        chunk_type="doc",
        breadcrumb=["Features", "Selection"],
        breadcrumb_text="Features > Selection",
        file_path="docs/features/selection.md",
        section_anchor="selection",
    )
    error = mock_qdrant_point(
        "selection-error",
        score=1.25,
        doc_category="other",
        chunk_type="doc",
        breadcrumb=["Configuration", "Error: both points and links specified"],
        breadcrumb_text="Configuration > Error: both points and links specified",
        file_path="docs-widget/configuration.md",
        section_anchor="error-both-points-and-links-specified",
    )

    apply_search_boosts(
        [error, selection],
        Settings(),
        intent=QueryIntent.CONCEPT,
        query="selection selected points programmatic selection",
    )

    assert selection.score > error.score


@pytest.mark.unit
def test_apply_exact_match_boosts_prefers_exact_leaf_and_file_stem(mock_qdrant_point):
    """Bare exact lookups should outrank dotted breadcrumb variants."""
    exact = mock_qdrant_point(
        "stack-reference",
        score=1.0,
        breadcrumb=["Components", "Stack"],
        breadcrumb_text="Components > Stack",
        file_path="core/stack.mdx",
        section_anchor="stack",
    )
    dotted_variant = mock_qdrant_point(
        "modal-stack",
        score=1.0,
        breadcrumb=["Components", "Modal.Stack"],
        breadcrumb_text="Components > Modal.Stack",
        file_path="modals/modal-stack.mdx",
        section_anchor="modal-stack",
    )

    apply_exact_match_boosts([dotted_variant, exact], "Stack", phrase_boost=2.0)

    assert exact.score > dotted_variant.score
    assert exact.score == pytest.approx(6.8607)
    assert dotted_variant.score == pytest.approx(0.779625)


@pytest.mark.unit
def test_apply_exact_match_boosts_penalizes_example_paths_for_bare_lookup(mock_qdrant_point):
    """Exact lookups should prefer canonical pages over demo/example variants."""
    reference = mock_qdrant_point(
        "stack-reference",
        score=1.0,
        breadcrumb=["Components", "Stack"],
        breadcrumb_text="Components > Stack",
        file_path="components/stack.mdx",
        section_anchor="stack",
    )
    example = mock_qdrant_point(
        "stack-demo",
        score=1.0,
        breadcrumb=["Examples", "Stack demo"],
        breadcrumb_text="Examples > Stack demo",
        file_path="examples/stack-demo.mdx",
        section_anchor="stack-demo",
    )

    apply_exact_match_boosts([example, reference], "Stack", phrase_boost=2.0)

    assert reference.score > example.score
    assert example.score == pytest.approx(0.921375)


@pytest.mark.unit
def test_apply_exact_match_boosts_penalizes_github_issue_template_paths(mock_qdrant_point):
    """Exact lookups should prefer canonical docs over issue-template FAQ surfaces."""
    reference = mock_qdrant_point(
        "animate-presence-reference",
        score=1.0,
        breadcrumb=["Components", "AnimatePresence"],
        breadcrumb_text="Components > AnimatePresence",
        file_path="docs/react/animate-presence.mdx",
        section_anchor="animatepresence",
    )
    issue_template = mock_qdrant_point(
        "animate-presence-issue-template",
        score=1.0,
        breadcrumb=["FAQs", "`AnimatePresence` isn't working"],
        breadcrumb_text="FAQs > `AnimatePresence` isn't working",
        file_path=".github/ISSUE_TEMPLATE/bug_report.md",
        section_anchor="animatepresence-isnt-working",
    )

    apply_exact_match_boosts([issue_template, reference], "AnimatePresence", phrase_boost=2.0)

    assert reference.score > issue_template.score


@pytest.mark.unit
def test_apply_exact_match_boosts_normalizes_breadcrumb_symbol_surfaces(mock_qdrant_point):
    """Exact API lookups should recognize breadcrumb leaves with args/comments markup."""
    canonical = mock_qdrant_point(
        "start-transition",
        score=1.0,
        breadcrumb=["Reference", "`startTransition(action)` {/*starttransition*/}"],
        breadcrumb_text="Reference {/*reference*/} > `startTransition(action)` {/*starttransition*/}",
        file_path="src/content/reference/react/startTransition.md",
        section_anchor="starttransitionaction-starttransition",
    )
    subsection = mock_qdrant_point(
        "start-transition-caveats",
        score=1.0,
        breadcrumb=[
            "Reference",
            "`startTransition(action)` {/*starttransition*/}",
            "Caveats {/*starttransition-caveats*/}",
        ],
        breadcrumb_text=(
            "Reference {/*reference*/} > `startTransition(action)` {/*starttransition*/} > "
            "Caveats {/*starttransition-caveats*/}"
        ),
        file_path="src/content/reference/react/startTransition.md",
        section_anchor="caveats-starttransition-caveats",
    )

    apply_exact_match_boosts([subsection, canonical], "startTransition", phrase_boost=2.0)

    assert canonical.score > subsection.score


@pytest.mark.unit
def test_apply_exact_match_boosts_prefers_canonical_surface_from_descriptive_query(
    mock_qdrant_point,
):
    """Descriptive queries should still favor the explicitly named API/component page."""
    canonical = mock_qdrant_point(
        "stack-reference",
        score=1.0,
        breadcrumb=["Components", "Stack"],
        breadcrumb_text="Components > Stack",
        file_path="core/stack.mdx",
        section_anchor="usage",
        chunk_type="doc",
    )
    related = mock_qdrant_point(
        "group-reference",
        score=1.0,
        breadcrumb=["Components", "Group"],
        breadcrumb_text="Components > Group",
        file_path="core/group.mdx",
        section_anchor="usage",
        chunk_type="doc",
        content="Group is a horizontal flex container. Use Stack for vertical layout and spacing.",
    )
    fallback = mock_qdrant_point(
        "flex-reference",
        score=1.0,
        breadcrumb=["Components", "Difference from Group and Stack"],
        breadcrumb_text="Components > Difference from Group and Stack",
        file_path="core/flex.mdx",
        section_anchor="difference-from-group-and-stack",
        chunk_type="doc",
        content="Flex is an alternative to Group and Stack for layout control.",
    )

    apply_exact_match_boosts(
        [fallback, related, canonical],
        "Stack component spacing groups vertical layout",
        phrase_boost=2.0,
    )

    assert canonical.score > related.score
    assert canonical.score > fallback.score


@pytest.mark.unit
def test_apply_exact_match_boosts_demotes_compound_variants_for_descriptive_query(
    mock_qdrant_point,
):
    """Descriptive queries should still penalize compound variants like Drawer.Stack."""
    canonical = mock_qdrant_point(
        "stack-reference",
        score=1.0,
        breadcrumb=["Components", "Stack"],
        breadcrumb_text="Components > Stack",
        file_path="core/stack.mdx",
        section_anchor="usage",
        chunk_type="doc",
    )
    compound = mock_qdrant_point(
        "drawer-stack",
        score=1.0,
        breadcrumb=["Components", "Drawer.Stack"],
        breadcrumb_text="Components > Drawer.Stack",
        file_path="core/drawer.mdx",
        section_anchor="drawerstack",
        chunk_type="doc",
        content="Drawer.Stack manages multiple drawers.",
    )

    apply_exact_match_boosts(
        [compound, canonical],
        "Stack gap layout patterns",
        phrase_boost=2.0,
    )

    assert canonical.score > compound.score


@pytest.mark.unit
def test_apply_exact_match_boosts_demotes_results_without_primary_surface(mock_qdrant_point):
    """API-like descriptive lookups should demote results that never surface the primary token."""
    canonical = mock_qdrant_point(
        "stack-reference",
        score=1.0,
        breadcrumb=["Components", "Stack"],
        breadcrumb_text="Components > Stack",
        file_path="core/stack.mdx",
        section_anchor="usage",
        chunk_type="doc",
    )
    unrelated = mock_qdrant_point(
        "stacked-chart",
        score=1.0,
        breadcrumb=["Charts", "Stacked area chart"],
        breadcrumb_text="Charts > Stacked area chart",
        file_path="charts/area-chart.mdx",
        section_anchor="stacked-area-chart",
        chunk_type="doc",
        content="Stacked area chart usage and layout.",
    )

    apply_exact_match_boosts(
        [unrelated, canonical],
        "Stack component gap align justify props",
        phrase_boost=2.0,
    )

    assert canonical.score > unrelated.score
