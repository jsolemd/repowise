"""Query intent detection for doc-search.

Detects the user's intent from query patterns to optimize search strategy.
Different query types benefit from different BM25/semantic weight ratios
and document category boosts.
"""

import re
from enum import Enum


class QueryIntent(Enum):
    """Types of documentation query intents."""

    API_LOOKUP = "api_lookup"
    """Looking for specific API/function documentation. BM25-heavy search."""

    HOW_TO = "how_to"
    """How-to questions, tutorials. Semantic-heavy search."""

    ERROR_DEBUG = "error_debug"
    """Error/debugging queries. Semantic + troubleshooting boost."""

    CONCEPT = "concept"
    """Conceptual questions, general learning. Balanced hybrid search."""

    IMPLEMENTATION = "implementation"
    """Looking for source code implementation. Boosts source over docs."""


# Search weight configurations by intent
# Higher bm25 = more keyword matching, higher semantic = more meaning matching
INTENT_SEARCH_WEIGHTS: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.API_LOOKUP: {"bm25": 0.7, "semantic": 0.3},
    QueryIntent.HOW_TO: {"bm25": 0.3, "semantic": 0.7},
    QueryIntent.ERROR_DEBUG: {"bm25": 0.4, "semantic": 0.6},
    QueryIntent.CONCEPT: {"bm25": 0.4, "semantic": 0.6},
    QueryIntent.IMPLEMENTATION: {"bm25": 0.5, "semantic": 0.5},  # Balanced for finding code
}

# Document category boost multipliers by intent
# Applied on top of base category boosts from config
INTENT_CATEGORY_BOOSTS: dict[QueryIntent, dict[str, float]] = {
    QueryIntent.API_LOOKUP: {
        "reference": 1.5,  # Strong boost for reference docs
        "guide": 0.8,  # Slight penalty for guides
        "example": 0.9,
        "changelog": 0.5,
        "blog": 0.5,
    },
    QueryIntent.HOW_TO: {
        "guide": 1.3,  # Boost guides/tutorials
        "reference": 0.9,
        "example": 1.2,  # Boost examples too
        "changelog": 0.6,
        "blog": 0.8,
    },
    QueryIntent.ERROR_DEBUG: {
        "guide": 1.2,
        "reference": 1.0,
        "example": 1.1,
        "changelog": 0.7,
        "blog": 0.8,
    },
    QueryIntent.CONCEPT: {
        "guide": 1.1,
        "reference": 1.0,
        "example": 1.0,
        "changelog": 0.7,
        "blog": 0.9,
    },
    QueryIntent.IMPLEMENTATION: {
        "reference": 1.2,  # Reference docs may have implementation notes
        "guide": 0.9,
        "example": 1.1,  # Examples show real code
        "changelog": 0.6,
        "blog": 0.7,
    },
}

# Compiled regex patterns for intent detection
# CamelCase pattern (e.g., useEffect, revalidateTag, MyComponent)
_CAMEL_CASE_PATTERN = re.compile(r"[A-Z][a-z]+[A-Z]|[a-z]+[A-Z]")

# Function call pattern (e.g., function(), method())
_FUNCTION_CALL_PATTERN = re.compile(r"\w+\(\)")

# Hook/method prefix pattern (e.g., use*, get*, set*, create*, etc.)
_HOOK_PREFIX_PATTERN = re.compile(
    r"^(use|get|set|create|delete|update|fetch|post|put|handle|on|is|has|should|can|will)"
    r"[A-Z_]",
    re.IGNORECASE,
)

_TITLE_CASE_SURFACE_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9]*(?:\.[A-Z][A-Za-z0-9]*)?$")

_COMPONENT_LOOKUP_HINT_PATTERN = re.compile(
    r"\b("
    r"component|components|prop|props|parameter|parameters|param|params|option|options|"
    r"gap|align|justify|spacing|layout|size|sizes|style|styles|variant|variants|"
    r"classname|class|radius|offset|color|colors|hook|hooks"
    r")\b",
    re.IGNORECASE,
)

# How-to question patterns
_HOW_TO_PATTERNS = [
    re.compile(r"^how\s+(to|do|can|should)\b", re.IGNORECASE),
    re.compile(r"^what\s+(is|are|does)\b", re.IGNORECASE),
    re.compile(r"^when\s+(to|should)\b", re.IGNORECASE),
    re.compile(r"^why\s+(does|is|should)\b", re.IGNORECASE),
    re.compile(r"^where\s+(to|should|do)\b", re.IGNORECASE),
    re.compile(r"^best\s+(way|practice)", re.IGNORECASE),
    re.compile(r"\btutorial\b", re.IGNORECASE),
    re.compile(r"\bguide\b", re.IGNORECASE),
]

# Guide/pattern hints that should prevent an API-named query from being treated
# as a bare reference lookup. These usually indicate the caller wants examples,
# layout guidance, or usage patterns rather than a raw signature page.
_API_GUIDE_HINT_PATTERN = re.compile(
    r"\b("
    r"example|examples|guide|guides|layout|layouts|pattern|patterns|practice|practices|"
    r"recipe|recipes|spacing|transition|transitions|tutorial|usage|workflow|workflows"
    r")\b",
    re.IGNORECASE,
)

# Error/debugging patterns
_ERROR_DEBUG_PATTERNS = [
    re.compile(r"\b(error|err|exception|bug|issue|problem)\b", re.IGNORECASE),
    re.compile(r"\b(fix|solve|debug|resolve|troubleshoot)\b", re.IGNORECASE),
    re.compile(r"\b(fail|failed|failing|crash|broken)\b", re.IGNORECASE),
    re.compile(r"\b(mismatch|warning|unexpected)\b", re.IGNORECASE),
    re.compile(r"\b(not working|doesn't work|won't)\b", re.IGNORECASE),
    re.compile(r"\bhydration\b", re.IGNORECASE),  # Common React SSR error
]

# Implementation/source code patterns
_IMPLEMENTATION_PATTERNS = [
    re.compile(r"\bhow\s+(?:does|is)\s+\S+\s+implemented\b", re.IGNORECASE),
    re.compile(r"\bimplementation\s+(?:of|details?)\b", re.IGNORECASE),
    re.compile(r"\bsource\s+code\b", re.IGNORECASE),
    re.compile(r"\bunder\s+the\s+hood\b", re.IGNORECASE),
    re.compile(r"\binternal(?:ly|s)?\b", re.IGNORECASE),
    re.compile(r"\bactual\s+code\b", re.IGNORECASE),
    re.compile(r"\breal\s+implementation\b", re.IGNORECASE),
    re.compile(r"\bhow\s+\S+\s+works\s+internally\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+is\s+\S+\s+defined\b", re.IGNORECASE),
    re.compile(r"\bshow\s+(?:me\s+)?(?:the\s+)?source\b", re.IGNORECASE),
]


def detect_intent(query: str) -> QueryIntent:
    """
    Detect query intent from text patterns.

    Uses a priority-based approach:
    1. Check for implementation/source code patterns (highest priority)
    2. Check for API lookup patterns (CamelCase, function names)
    3. Check for how-to question patterns
    4. Check for error/debugging patterns
    5. Default to concept (general learning)

    Args:
        query: The search query string

    Returns:
        Detected QueryIntent enum value

    Examples:
        >>> detect_intent("useEffect")
        QueryIntent.API_LOOKUP
        >>> detect_intent("how to handle authentication")
        QueryIntent.HOW_TO
        >>> detect_intent("hydration mismatch error")
        QueryIntent.ERROR_DEBUG
        >>> detect_intent("how is observe decorator implemented")
        QueryIntent.IMPLEMENTATION
        >>> detect_intent("state management patterns")
        QueryIntent.CONCEPT
    """
    query = query.strip()
    if not query:
        return QueryIntent.CONCEPT

    # Get first word for prefix checks
    first_word = query.split()[0] if query.split() else query

    # 1. Implementation: Source code/internal implementation requests
    # Check this BEFORE how-to patterns since "how is X implemented" should match here
    for pattern in _IMPLEMENTATION_PATTERNS:
        if pattern.search(query):
            return QueryIntent.IMPLEMENTATION

    api_guide_hint = bool(_API_GUIDE_HINT_PATTERN.search(query))

    # 2. API Lookup: CamelCase, function calls, hook prefixes
    # Check for CamelCase patterns anywhere in query
    if _CAMEL_CASE_PATTERN.search(query):
        if api_guide_hint:
            return QueryIntent.HOW_TO
        return QueryIntent.API_LOOKUP

    # Check for function call syntax
    if _FUNCTION_CALL_PATTERN.search(query):
        if api_guide_hint:
            return QueryIntent.HOW_TO
        return QueryIntent.API_LOOKUP

    # Check for hook/method prefixes on first word
    if _HOOK_PREFIX_PATTERN.match(first_word):
        if api_guide_hint:
            return QueryIntent.HOW_TO
        return QueryIntent.API_LOOKUP

    # Check for snake_case (likely Python API)
    if "_" in first_word and not first_word.startswith("_"):  # noqa: SIM102
        # Verify it's likely a function name (multiple underscores or alphanumeric)
        if re.match(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$", first_word):
            if api_guide_hint:
                return QueryIntent.HOW_TO
            return QueryIntent.API_LOOKUP

    # Check for Pascal/TitleCase component names paired with prop or API-surface hints
    if _TITLE_CASE_SURFACE_PATTERN.match(first_word) and _COMPONENT_LOOKUP_HINT_PATTERN.search(
        query
    ):
        return QueryIntent.API_LOOKUP

    # 3. How-To: Question patterns, tutorial requests
    for pattern in _HOW_TO_PATTERNS:
        if pattern.search(query):
            return QueryIntent.HOW_TO

    # 4. Error/Debug: Error-related terms
    for pattern in _ERROR_DEBUG_PATTERNS:
        if pattern.search(query):
            return QueryIntent.ERROR_DEBUG

    # 5. Default: Concept (general learning)
    return QueryIntent.CONCEPT


def get_search_weights(intent: QueryIntent) -> dict[str, float]:
    """
    Get BM25/semantic weight ratios for an intent.

    Args:
        intent: The detected query intent

    Returns:
        Dict with 'bm25' and 'semantic' weights (sum to 1.0)
    """
    return INTENT_SEARCH_WEIGHTS.get(intent, {"bm25": 0.5, "semantic": 0.5})


def get_category_boosts(intent: QueryIntent) -> dict[str, float]:
    """
    Get document category boost multipliers for an intent.

    Args:
        intent: The detected query intent

    Returns:
        Dict mapping category names to boost multipliers
    """
    return INTENT_CATEGORY_BOOSTS.get(intent, {})
