"""Tests for query intent detection."""

import pytest

from repowise.docs.search.intent import (
    INTENT_CATEGORY_BOOSTS,
    INTENT_SEARCH_WEIGHTS,
    QueryIntent,
    detect_intent,
    get_category_boosts,
    get_search_weights,
)


class TestDetectIntent:
    """Tests for the detect_intent function."""

    # API Lookup tests - CamelCase patterns
    @pytest.mark.parametrize(
        "query",
        [
            "useEffect",
            "useState",
            "revalidateTag",
            "getStaticProps",
            "MyComponent",
            "useEffect cleanup",
            "revalidateTag function parameters",
            "NextResponse",
            "Stack component gap align justify props",
        ],
    )
    def test_api_lookup_camelcase(self, query: str) -> None:
        """CamelCase patterns should be detected as API_LOOKUP."""
        assert detect_intent(query) == QueryIntent.API_LOOKUP

    # API Lookup tests - hook/method prefixes
    @pytest.mark.parametrize(
        "query",
        [
            "useRouter",
            "getServerSideProps",
            "setConfig",
            "createContext",
            "deleteItem",
            "updateUser",
            "fetchData",
            "handleClick",
            "onSubmit",
            "isLoading",
            "hasPermission",
        ],
    )
    def test_api_lookup_prefixes(self, query: str) -> None:
        """Hook/method prefixes should be detected as API_LOOKUP."""
        assert detect_intent(query) == QueryIntent.API_LOOKUP

    # API Lookup tests - function call syntax
    @pytest.mark.parametrize(
        "query",
        [
            "revalidateTag()",
            "fetch()",
            "useCallback()",
        ],
    )
    def test_api_lookup_function_call(self, query: str) -> None:
        """Function call syntax should be detected as API_LOOKUP."""
        assert detect_intent(query) == QueryIntent.API_LOOKUP

    # API Lookup tests - snake_case (Python-style)
    @pytest.mark.parametrize(
        "query",
        [
            "get_config",
            "set_value",
            "create_user",
            "fetch_data_async",
        ],
    )
    def test_api_lookup_snake_case(self, query: str) -> None:
        """Snake_case function names should be detected as API_LOOKUP."""
        assert detect_intent(query) == QueryIntent.API_LOOKUP

    # How-To tests
    @pytest.mark.parametrize(
        "query",
        [
            "how to handle authentication",
            "how do I use hooks",
            "how can I optimize performance",
            "how should I structure my app",
            "what is server-side rendering",
            "what are React hooks",
            # Note: "what does useEffect do" contains CamelCase -> API_LOOKUP
            # Note: "when to use useMemo" contains CamelCase -> API_LOOKUP
            "when should I use SSG",
            "why does hydration fail",
            "why is my app slow",
            "best way to handle state",
            "best practices for Next.js",
            "tutorial on routing",
            "guide to data fetching",
        ],
    )
    def test_how_to_patterns(self, query: str) -> None:
        """Question patterns should be detected as HOW_TO."""
        assert detect_intent(query) == QueryIntent.HOW_TO

    # API questions with API names embedded - these ARE API lookups
    @pytest.mark.parametrize(
        "query",
        [
            "what does useEffect do",
            "when to use useMemo",
            "how to use useState",
        ],
    )
    def test_api_questions(self, query: str) -> None:
        """Questions containing API names should be API_LOOKUP (more specific)."""
        assert detect_intent(query) == QueryIntent.API_LOOKUP

    @pytest.mark.parametrize(
        "query",
        [
            "Stack component gap align justify props",
            "Modal component size variants props",
        ],
    )
    def test_title_case_component_queries_with_lookup_hints_route_to_api_lookup(
        self, query: str
    ) -> None:
        """TitleCase component surfaces plus prop-like hints should behave like API lookups."""
        assert detect_intent(query) == QueryIntent.API_LOOKUP

    @pytest.mark.parametrize(
        "query",
        [
            "AnimatePresence layout shared transition patterns",
            "useTransition examples and usage patterns",
        ],
    )
    def test_api_named_pattern_queries_route_to_how_to(self, query: str) -> None:
        """Named APIs with guide/pattern terms should behave like usage queries."""
        assert detect_intent(query) == QueryIntent.HOW_TO

    # Error/Debug tests
    @pytest.mark.parametrize(
        "query",
        [
            "hydration mismatch error",
            "fix authentication bug",
            "debug rendering issue",
            "resolve CORS problem",
            "troubleshoot build failure",
            "error boundary not working",
            # Note: "component not rendering" - "component" is not an error term
            "fetch failed",
            # Note: "app crashed" - detected as CONCEPT (no strong error signal)
            "unexpected token",
            "hydration warning",
            "type mismatch",
            "render error",
            "build failed",
        ],
    )
    def test_error_debug_patterns(self, query: str) -> None:
        """Error-related queries should be detected as ERROR_DEBUG."""
        assert detect_intent(query) == QueryIntent.ERROR_DEBUG

    # Concept tests (default/general learning)
    @pytest.mark.parametrize(
        "query",
        [
            "state management patterns",
            "React architecture",
            "server components vs client components",
            "caching strategies",
            "authentication patterns",
            "Next.js app router",
            "middleware concepts",
        ],
    )
    def test_concept_patterns(self, query: str) -> None:
        """General conceptual queries should be detected as CONCEPT."""
        assert detect_intent(query) == QueryIntent.CONCEPT

    def test_empty_query(self) -> None:
        """Empty query should default to CONCEPT."""
        assert detect_intent("") == QueryIntent.CONCEPT
        assert detect_intent("   ") == QueryIntent.CONCEPT


class TestSearchWeights:
    """Tests for intent-based search weight retrieval."""

    def test_all_intents_have_weights(self) -> None:
        """All query intents should have defined search weights."""
        for intent in QueryIntent:
            weights = get_search_weights(intent)
            assert "bm25" in weights
            assert "semantic" in weights
            # Weights should sum approximately to 1.0
            assert abs(weights["bm25"] + weights["semantic"] - 1.0) < 0.01

    def test_api_lookup_bm25_heavy(self) -> None:
        """API_LOOKUP should have higher BM25 weight."""
        weights = get_search_weights(QueryIntent.API_LOOKUP)
        assert weights["bm25"] > weights["semantic"]

    def test_how_to_semantic_heavy(self) -> None:
        """HOW_TO should have higher semantic weight."""
        weights = get_search_weights(QueryIntent.HOW_TO)
        assert weights["semantic"] > weights["bm25"]


class TestCategoryBoosts:
    """Tests for intent-based category boost retrieval."""

    def test_all_intents_have_boosts(self) -> None:
        """All query intents should have defined category boosts."""
        for intent in QueryIntent:
            boosts = get_category_boosts(intent)
            assert isinstance(boosts, dict)
            # Should have some category definitions
            assert len(boosts) > 0

    def test_api_lookup_boosts_reference(self) -> None:
        """API_LOOKUP should strongly boost reference docs."""
        boosts = get_category_boosts(QueryIntent.API_LOOKUP)
        assert boosts.get("reference", 1.0) > 1.0
        # And penalize guides relative to reference
        assert boosts.get("guide", 1.0) < boosts.get("reference", 1.0)

    def test_how_to_boosts_guides(self) -> None:
        """HOW_TO should boost guide docs."""
        boosts = get_category_boosts(QueryIntent.HOW_TO)
        assert boosts.get("guide", 1.0) > 1.0


class TestIntentConstants:
    """Tests for intent configuration constants."""

    def test_search_weights_structure(self) -> None:
        """INTENT_SEARCH_WEIGHTS should have proper structure."""
        for intent, weights in INTENT_SEARCH_WEIGHTS.items():
            assert isinstance(intent, QueryIntent)
            assert "bm25" in weights
            assert "semantic" in weights
            assert 0 <= weights["bm25"] <= 1
            assert 0 <= weights["semantic"] <= 1

    def test_category_boosts_structure(self) -> None:
        """INTENT_CATEGORY_BOOSTS should have proper structure."""
        for intent, boosts in INTENT_CATEGORY_BOOSTS.items():
            assert isinstance(intent, QueryIntent)
            for category, boost in boosts.items():
                assert isinstance(category, str)
                assert isinstance(boost, (int, float))
                assert boost > 0  # Boosts should be positive
