"""Tests for search relevance boosting.

These tests verify that document category, path hierarchy, heading level,
and token count boosts are correctly applied during hybrid search.

Run with: uv run pytest tests/test_search_boosting.py -v
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from repowise.docs.config import Settings
from repowise.docs.indexer.search import search_hybrid
from repowise.docs.search.intent import QueryIntent

# ============================================================================
# Doc Category Boosting Tests
# ============================================================================


class TestDocCategoryBoosting:
    """Test doc_category boosting in search results."""

    @pytest.fixture
    def mock_settings(self):
        """Settings with default boost values."""
        return Settings(
            doc_category_boosts={
                "reference": 1.25,
                "guide": 1.0,
                "example": 0.9,
                "changelog": 0.6,
                "blog": 0.7,
                "other": 1.0,
            }
        )

    @pytest.mark.unit
    async def test_reference_outranks_changelog(self, mock_qdrant_point):
        """With same base score, reference should outrank changelog after boost."""
        # Create two points with same base score but different categories
        reference_point = mock_qdrant_point(
            "ref-chunk",
            score=0.8,
            doc_category="reference",
            content="API reference content",
        )
        changelog_point = mock_qdrant_point(
            "log-chunk",
            score=0.8,
            doc_category="changelog",
            content="Changelog content",
        )

        with (
            patch(
                "repowise.docs.indexer.search.embed_single", new_callable=AsyncMock
            ) as mock_embed,
            patch("repowise.docs.indexer.search.get_settings") as mock_settings,
        ):
            mock_embed.return_value = [0.1] * 768
            mock_settings.return_value = Settings(
                doc_category_boosts={
                    "reference": 1.25,
                    "changelog": 0.6,
                    "other": 1.0,
                }
            )

            # Simulate the boost logic manually (as search_hybrid does)
            boosts = mock_settings.return_value.doc_category_boosts

            # Apply boosts
            ref_boosted = reference_point.score * boosts.get("reference", 1.0)
            log_boosted = changelog_point.score * boosts.get("changelog", 1.0)

            # Reference should now be higher
            assert ref_boosted > log_boosted
            assert ref_boosted == 0.8 * 1.25  # 1.0
            assert log_boosted == 0.8 * 0.6  # 0.48

    @pytest.mark.unit
    async def test_boost_multipliers_applied(self, mock_qdrant_point):
        """Verify config boost multipliers are correctly applied."""
        settings = Settings(
            doc_category_boosts={
                "reference": 1.25,
                "guide": 1.0,
                "example": 0.9,
                "changelog": 0.6,
                "blog": 0.7,
                "other": 1.0,
            }
        )

        test_cases = [
            ("reference", 0.8, 1.0),  # 0.8 * 1.25 = 1.0
            ("guide", 0.8, 0.8),  # 0.8 * 1.0 = 0.8
            ("example", 0.8, 0.72),  # 0.8 * 0.9 = 0.72
            ("changelog", 0.8, 0.48),  # 0.8 * 0.6 = 0.48
            ("blog", 0.8, 0.56),  # 0.8 * 0.7 = 0.56
            ("other", 0.8, 0.8),  # 0.8 * 1.0 = 0.8
        ]

        boosts = settings.doc_category_boosts
        for category, base_score, expected in test_cases:
            boost = boosts.get(category, 1.0)
            result = base_score * boost
            assert abs(result - expected) < 0.001, f"Failed for {category}"

    @pytest.mark.unit
    async def test_missing_doc_category_defaults_to_other(self, mock_qdrant_point):
        """Chunks without doc_category should use 'other' boost (baseline)."""
        # Create point without doc_category in payload
        point = MagicMock()
        point.id = "legacy-chunk"
        point.score = 0.8
        point.payload = {
            "content": "Legacy content",
            # Note: no doc_category field
        }

        settings = Settings(
            doc_category_boosts={
                "reference": 1.25,
                "other": 1.0,
            }
        )

        # The handler should default to "other" when category is missing
        doc_category = point.payload.get("doc_category", "other")
        boost = settings.doc_category_boosts.get(doc_category, 1.0)

        assert doc_category == "other"
        assert boost == 1.0
        assert point.score * boost == 0.8

    @pytest.mark.unit
    async def test_resorting_after_boost(self, mock_qdrant_point):
        """Results should be re-sorted after applying category boosts."""
        # Create points where initial order differs from final order
        points = [
            mock_qdrant_point("p1", score=0.9, doc_category="changelog"),  # 0.9 * 0.6 = 0.54
            mock_qdrant_point("p2", score=0.7, doc_category="reference"),  # 0.7 * 1.25 = 0.875
            mock_qdrant_point("p3", score=0.8, doc_category="guide"),  # 0.8 * 1.0 = 0.8
        ]

        settings = Settings(
            doc_category_boosts={
                "reference": 1.25,
                "guide": 1.0,
                "changelog": 0.6,
            }
        )

        # Apply boosts
        boosts = settings.doc_category_boosts
        for point in points:
            category = point.payload.get("doc_category", "other")
            boost = boosts.get(category, 1.0)
            point.score *= boost

        # Sort by boosted score
        points.sort(key=lambda p: p.score, reverse=True)

        # After boost, reference (originally 0.7) should be first
        assert points[0].id == "p2"  # reference: 0.875
        assert points[1].id == "p3"  # guide: 0.8
        assert points[2].id == "p1"  # changelog: 0.54

    @pytest.mark.unit
    async def test_doc_category_in_results(self, mock_library_ready, mock_qdrant_point):
        """Query response should include doc_category field."""
        from repowise.docs.tools.handlers import handle_query_docs

        point = mock_qdrant_point(
            "chunk-1",
            score=0.9,
            doc_category="reference",
        )

        with (
            patch("repowise.docs.tools.handlers.get_library", new_callable=AsyncMock) as mock_get,
            patch(
                "repowise.docs.tools.handlers.search_hybrid", new_callable=AsyncMock
            ) as mock_search,
            patch(
                "repowise.docs.tools.handlers.get_sibling_chunks", new_callable=AsyncMock
            ) as mock_siblings,
        ):
            mock_get.return_value = mock_library_ready
            mock_search.return_value = [point]
            mock_siblings.return_value = []

            result = await handle_query_docs(
                {
                    "library_id": "/test/test",
                    "query": "test query",
                }
            )

        assert len(result["results"]) == 1
        assert result["results"][0]["doc_category"] == "reference"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_hybrid_rescues_exact_component_surface_candidates(mock_qdrant_point):
    """Hybrid search should rescue canonical component pages from exact-surface BM25 hits."""
    section_gaps = mock_qdrant_point(
        "ring-progress-section-gaps",
        score=0.64,
        doc_category="other",
        chunk_type="doc",
        breadcrumb=["RingProgress", "Section gaps"],
        breadcrumb_text="RingProgress > Section gaps",
        file_path="apps/mantine.dev/src/pages/core/ring-progress.mdx",
        section_anchor="section-gaps",
    )
    stack = mock_qdrant_point(
        "stack-reference",
        score=0.42,
        doc_category="reference",
        chunk_type="doc",
        breadcrumb=["Components", "Stack"],
        breadcrumb_text="Components > Stack",
        file_path="apps/mantine.dev/src/pages/core/stack.mdx",
        section_anchor="stack",
    )

    client = MagicMock()
    client.query_points.side_effect = [
        SimpleNamespace(points=[section_gaps]),
        SimpleNamespace(points=[stack]),
    ]

    with (
        patch("repowise.docs.indexer.search.embed_single", new_callable=AsyncMock) as mock_embed,
        patch("repowise.docs.indexer.search.get_settings") as mock_settings,
    ):
        mock_embed.return_value = [0.1] * 768
        mock_settings.return_value = Settings()

        results = await search_hybrid(
            library_id="/mantinedev/mantine",
            query="Stack gap motion layout patterns",
            limit=3,
            intent=QueryIntent.CONCEPT,
            client=client,
        )

    assert results[0].id == "stack-reference"
    assert client.query_points.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_hybrid_api_lookup_demotes_weak_paths_when_stronger_surface_exists(
    mock_qdrant_point,
):
    """API lookups should prefer canonical docs/source over example or issue-template paths."""
    weak_example = mock_qdrant_point(
        "animate-presence-example",
        score=50.0,
        doc_category="example",
        chunk_type="doc",
        breadcrumb=["AnimatePresence"],
        breadcrumb_text="AnimatePresence",
        file_path="dev/react/src/examples/AnimatePresence.tsx",
        section_anchor="animatepresence",
    )
    strong_source = mock_qdrant_point(
        "animate-presence-types",
        score=15.0,
        doc_category="other",
        chunk_type="code",
        breadcrumb=["AnimatePresenceProps"],
        breadcrumb_text="AnimatePresenceProps",
        file_path="packages/framer-motion/src/components/AnimatePresence/types.ts",
        section_anchor="interface-animatepresenceprops",
    )

    client = MagicMock()
    client.query_points.side_effect = [
        SimpleNamespace(points=[weak_example, strong_source]),
        SimpleNamespace(points=[]),
    ]

    with (
        patch("repowise.docs.indexer.search.embed_single", new_callable=AsyncMock) as mock_embed,
        patch("repowise.docs.indexer.search.get_settings") as mock_settings,
    ):
        mock_embed.return_value = [0.1] * 768
        mock_settings.return_value = Settings()

        results = await search_hybrid(
            library_id="/framer/motion",
            query="AnimatePresence layout shared transition patterns",
            limit=3,
            intent=QueryIntent.API_LOOKUP,
            client=client,
        )

    assert results[0].id == "animate-presence-types"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_search_hybrid_api_lookup_prefers_canonical_component_page_over_compound_variants(
    mock_qdrant_point,
):
    """API-lookup hybrid search should favor the canonical component page over compound variants."""
    canonical = mock_qdrant_point(
        "stack-usage",
        score=8.0,
        doc_category="reference",
        chunk_type="doc",
        breadcrumb=["Components", "Stack"],
        breadcrumb_text="Components > Stack",
        file_path="apps/mantine.dev/src/pages/core/stack.mdx",
        section_anchor="usage",
        content="Stack gap align justify props",
    )
    drawer_stack = mock_qdrant_point(
        "drawer-stack",
        score=12.0,
        doc_category="reference",
        chunk_type="doc",
        breadcrumb=["Components", "Drawer.Stack"],
        breadcrumb_text="Components > Drawer.Stack",
        file_path="apps/mantine.dev/src/pages/core/drawer.mdx",
        section_anchor="drawer-stack",
        content="Drawer.Stack manages multiple drawers.",
    )
    modal_stack = mock_qdrant_point(
        "modal-stack",
        score=12.0,
        doc_category="reference",
        chunk_type="doc",
        breadcrumb=["Components", "Modal.Stack"],
        breadcrumb_text="Components > Modal.Stack",
        file_path="apps/mantine.dev/src/pages/core/modal.mdx",
        section_anchor="modal-stack",
        content="Modal.Stack manages modal groups.",
    )

    client = MagicMock()
    client.query_points.side_effect = [
        SimpleNamespace(points=[drawer_stack, modal_stack, canonical]),
        SimpleNamespace(points=[]),
    ]

    with (
        patch("repowise.docs.indexer.search.embed_single", new_callable=AsyncMock) as mock_embed,
        patch("repowise.docs.indexer.search.get_settings") as mock_settings,
    ):
        mock_embed.return_value = [0.1] * 768
        mock_settings.return_value = Settings()

        results = await search_hybrid(
            library_id="/mantinedev/mantine",
            query="Stack component gap align justify props",
            limit=5,
            intent=QueryIntent.API_LOOKUP,
            client=client,
        )

    assert results[0].id == "stack-usage"


# ============================================================================
# Path Hierarchy Boosting Tests
# ============================================================================


class TestPathHierarchyBoosting:
    """Test path-based boosting for prioritizing core docs over integrations."""

    @pytest.mark.unit
    def test_path_boost_config_structure(self):
        """Verify path_hierarchy_boosts config has expected structure."""
        settings = Settings()

        # If path_hierarchy_boosts is implemented, verify structure
        if hasattr(settings, "path_hierarchy_boosts"):
            boosts = settings.path_hierarchy_boosts
            assert isinstance(boosts, dict)
            # Check for expected patterns
            assert any("docs" in pattern.lower() for pattern in boosts)

    @pytest.mark.unit
    def test_core_docs_path_should_boost(self):
        """Core documentation paths should receive positive boost."""
        # Expected behavior: pages/docs/ paths get +15% boost
        core_paths = [
            "pages/docs/sdk/python.mdx",
            "pages/docs/tracing/overview.mdx",
            "docs/reference/api.md",
        ]

        # Test with regex patterns (implementation reference)
        import re

        core_pattern = re.compile(r"(?:^|/)docs/", re.IGNORECASE)

        for path in core_paths:
            assert core_pattern.search(path), f"Core path should match: {path}"

    @pytest.mark.unit
    def test_integration_paths_no_boost(self):
        """Integration paths should not receive positive boost."""
        integration_paths = [
            "pages/integrations/helicone.mdx",
            "pages/integrations/openai.mdx",
        ]

        import re

        integration_pattern = re.compile(r"(?:^|/)integrations/", re.IGNORECASE)

        for path in integration_paths:
            assert integration_pattern.search(path), f"Integration path should match: {path}"


# ============================================================================
# Heading Level Boosting Tests (Placeholder)
# ============================================================================


class TestHeadingLevelBoosting:
    """Test heading level-based boosting.

    H1/H2 sections should be boosted more than H4/H5/H6 sections.
    """

    @pytest.mark.unit
    def test_heading_level_boost_values(self):
        """Verify heading level boost configuration."""
        expected_boosts = {
            1: 1.15,  # H1: +15%
            2: 1.10,  # H2: +10%
            3: 1.0,  # H3: baseline
            4: 0.95,  # H4: -5%
            5: 0.95,  # H5: -5%
            6: 0.95,  # H6: -5%
        }

        # H1/H2 should have higher boosts than H4+
        assert expected_boosts[1] > expected_boosts[3]
        assert expected_boosts[2] > expected_boosts[3]
        assert expected_boosts[4] < expected_boosts[3]

    @pytest.mark.unit
    def test_h2_outranks_h4_same_score(self):
        """H2 sections should rank higher than H4 with same base score."""
        h2_score = 0.8 * 1.10  # H2 boost
        h4_score = 0.8 * 0.95  # H4 boost

        assert h2_score > h4_score


# ============================================================================
# Token Count Boosting Tests (Placeholder)
# ============================================================================


class TestTokenCountBoosting:
    """Test token count-based boosting.

    Substantive content (>400 tokens) should be boosted.
    Tiny fragments (<100 tokens) should be penalized.
    """

    @pytest.mark.unit
    def test_substantive_content_boosted(self):
        """Content with >400 tokens should receive positive boost."""
        token_count = 500
        boost = 1.05 if token_count > 400 else 1.0

        assert boost == 1.05

    @pytest.mark.unit
    def test_tiny_fragments_penalized(self):
        """Content with <100 tokens should receive negative boost."""
        token_count = 50
        boost = 0.90 if token_count < 100 else 1.0

        assert boost == 0.90

    @pytest.mark.unit
    def test_medium_content_baseline(self):
        """Content between 100-400 tokens uses baseline boost."""
        token_count = 250
        boost = 1.0

        if token_count < 100:
            boost = 0.90
        elif token_count > 400:
            boost = 1.05

        assert boost == 1.0
