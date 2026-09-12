"""Search result scoring helpers for doc-search."""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from qdrant_client import models

from repowise.docs.config import Settings

if TYPE_CHECKING:
    from repowise.docs.search.intent import QueryIntent

logger = logging.getLogger(__name__)

# Pattern to strip split suffixes from section anchors (e.g., -part-2, -code-part-3)
# This allows matching code blocks to their parent doc chunks when sections are split.
ANCHOR_SPLIT_SUFFIX_PATTERN = re.compile(r"(-part-\d+|-code-part-\d+)$")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_EXACT_LOOKUP_WEAK_PATH_PATTERN = re.compile(
    r"(?:^|/)(?:examples?|demos?|demo|showcase|cookbook|recipes?|playground|storybook|stories?)(?:/|$)"
    r"|(?:^|/)\.github(?:/|$)"
    r"|(?:^|/)(?:issue|pull_request|pull-request)_template(?:s)?(?:/|$)"
    r"|(?:^|/)[^/]+\.story\.",
    re.IGNORECASE,
)
_SURFACE_COMMENT_PATTERN = re.compile(r"\{/\*.*?\*/\}")
_SURFACE_ARGS_PATTERN = re.compile(r"\(.*\)$")
_ERROR_SURFACE_PATTERN = re.compile(
    r"\b(error|warning|exception|troubleshoot(?:ing)?|caveat|common issue(?:s)?)\b",
    re.IGNORECASE,
)
_QUERY_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9._:/-]*")
_QUERY_TOKEN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "api",
        "apis",
        "component",
        "components",
        "example",
        "examples",
        "for",
        "function",
        "functions",
        "guide",
        "guides",
        "hook",
        "hooks",
        "how",
        "in",
        "of",
        "on",
        "prop",
        "props",
        "reference",
        "references",
        "the",
        "to",
        "use",
        "using",
        "with",
    }
)


def _normalize_search_text(value: object) -> str:
    """Normalize text for exact-match comparisons."""
    if value is None:
        return ""
    normalized = _WHITESPACE_PATTERN.sub(" ", str(value).strip().lower())
    return normalized


def _get_breadcrumb_leaf(payload: dict) -> str:
    """Return the last breadcrumb segment for a payload."""
    breadcrumb = payload.get("breadcrumb", [])
    if isinstance(breadcrumb, list):
        for item in reversed(breadcrumb):
            leaf = _normalize_search_text(item)
            if leaf:
                return leaf

    breadcrumb_text = _normalize_search_text(payload.get("breadcrumb_text", ""))
    if ">" in breadcrumb_text:
        return _normalize_search_text(breadcrumb_text.split(">")[-1])
    return breadcrumb_text


def _get_file_stem(payload: dict) -> str:
    """Return the normalized file stem for a payload path."""
    file_path = str(payload.get("file_path", "") or "").strip()
    if not file_path:
        return ""
    return _normalize_search_text(PurePosixPath(file_path).stem)


def _get_anchor(payload: dict) -> str:
    """Return the canonicalized anchor surface for a payload."""
    anchor = payload.get("canonical_anchor") or payload.get("section_anchor") or ""
    return _normalize_search_text(ANCHOR_SPLIT_SUFFIX_PATTERN.sub("", str(anchor)))


def _is_bare_lookup(query: str) -> bool:
    """Return True when query is a compact single-surface lookup."""
    return bool(query) and " " not in query


def _canonical_surface(surface: str) -> str:
    """Normalize breadcrumb/anchor display text to its canonical symbol surface."""
    normalized = _normalize_search_text(surface)
    if not normalized:
        return ""
    normalized = _SURFACE_COMMENT_PATTERN.sub("", normalized).strip()
    normalized = normalized.replace("`", "").replace("<", "").replace(">", "")
    normalized = _SURFACE_ARGS_PATTERN.sub("", normalized).strip()
    return normalized


def _is_trailing_variant(surface: str, query: str) -> bool:
    """Return True when a surface ends with the query as a dotted/slashed variant."""
    if not surface or not query or surface == query:
        return False
    return bool(
        surface.endswith(f".{query}")
        or surface.endswith(f"/{query}")
        or surface.endswith(f":{query}")
        or surface.endswith(f"_{query}")
        or surface.endswith(f"-{query}")
    )


def is_exact_lookup_weak_path(file_path: str) -> bool:
    """Return True for paths that are usually secondary for exact API/component lookups."""
    return bool(file_path and _EXACT_LOOKUP_WEAK_PATH_PATTERN.search(file_path))


def surface_query_tokens(query: str) -> list[str]:
    """Extract meaningful identifier-like tokens from a descriptive surface query."""
    normalized = _normalize_search_text(query)
    if not normalized:
        return []

    tokens: list[str] = []
    for raw in _QUERY_TOKEN_PATTERN.findall(normalized):
        token = _canonical_surface(raw)
        if len(token) < 2 or token in _QUERY_TOKEN_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _surface_match_multiplier(payload: dict, query: str) -> float:
    """Boost results whose public surfaces match the main identifier in a descriptive query."""
    query_tokens = surface_query_tokens(query)
    if not query_tokens:
        return 1.0

    surfaces_by_name = {
        "breadcrumb_leaf": _canonical_surface(_get_breadcrumb_leaf(payload)),
        "anchor": _canonical_surface(_get_anchor(payload)),
        "file_stem": _canonical_surface(_get_file_stem(payload)),
    }
    primary_surfaces = {surface for surface in surfaces_by_name.values() if surface}

    if not primary_surfaces:
        return 1.0

    multiplier = 1.0
    first_token = query_tokens[0]
    exact_surface_hits = {
        name for name, surface in surfaces_by_name.items() if surface and surface == first_token
    }
    if exact_surface_hits:
        if "breadcrumb_leaf" in exact_surface_hits or "file_stem" in exact_surface_hits:
            multiplier *= 2.35
        else:
            multiplier *= 2.0
        if len(exact_surface_hits) > 1:
            multiplier *= 1.0 + (0.18 * (len(exact_surface_hits) - 1))

    matched_tokens = {token for token in query_tokens if token in primary_surfaces}
    if matched_tokens:
        multiplier *= 1.0 + (0.12 * max(0, len(matched_tokens) - 1))

    return multiplier


def _is_error_like_surface(payload: dict) -> bool:
    """Return True when the public section title reads like troubleshooting content."""
    surfaces = (
        _canonical_surface(_get_breadcrumb_leaf(payload)),
        _canonical_surface(_get_anchor(payload)),
        _canonical_surface(_get_file_stem(payload)),
    )
    return any(surface and _ERROR_SURFACE_PATTERN.search(surface) for surface in surfaces)


def apply_exact_match_boosts(
    results: list[models.ScoredPoint],
    query: str,
    *,
    phrase_boost: float,
) -> list[models.ScoredPoint]:
    """Apply exact-match boosts before general search boosts.

    This favors exact component/API/reference surfaces over substring-only variants
    such as `Modal.Stack` for a bare `Stack` lookup.
    """
    normalized_query = _normalize_search_text(query)
    if not normalized_query:
        return results

    bare_lookup = _is_bare_lookup(normalized_query)
    query_tokens = [] if bare_lookup else surface_query_tokens(query)

    for point in results:
        if point.score is None:
            continue

        payload = point.payload or {}
        content = str(payload.get("content", "") or "")
        content_lower = content.lower()

        multiplier = 1.0

        if normalized_query in content_lower:
            multiplier *= phrase_boost
            if query in content:
                multiplier *= 1.25

        breadcrumb_text = _normalize_search_text(
            payload.get("breadcrumb_text", "") or " ".join(payload.get("breadcrumb", []))
        )
        if normalized_query in breadcrumb_text:
            multiplier *= 1.35

        surfaces = {
            "breadcrumb_leaf": _get_breadcrumb_leaf(payload),
            "file_stem": _get_file_stem(payload),
            "anchor": _get_anchor(payload),
        }
        canonical_surfaces = {
            name: _canonical_surface(surface) for name, surface in surfaces.items()
        }
        exact_hits = {
            name
            for name, surface in surfaces.items()
            if surface == normalized_query or canonical_surfaces.get(name) == normalized_query
        }
        exact_surface_count = len(exact_hits)
        file_path = str(payload.get("file_path", "") or "")
        chunk_type = str(payload.get("chunk_type", "") or "")
        if exact_surface_count > 0:
            if "breadcrumb_leaf" in exact_hits:
                multiplier *= 4.4
            elif "anchor" in exact_hits:
                multiplier *= 4.2
            else:
                multiplier *= 4.0
            if exact_surface_count > 1:
                multiplier *= 1.1

            subsection_anchor = surfaces["anchor"]
            if (
                "file_stem" in exact_hits
                and "breadcrumb_leaf" not in exact_hits
                and "anchor" not in exact_hits
                and subsection_anchor
                and subsection_anchor != normalized_query
            ):
                multiplier *= 0.5
        elif bare_lookup:
            trailing_variant_hits = sum(
                1
                for surface in surfaces.values()
                if _is_trailing_variant(surface, normalized_query)
            )
            if trailing_variant_hits:
                multiplier *= 0.55
        elif query_tokens:
            matching_tokens: list[tuple[int, str]] = []
            for index, token in enumerate(query_tokens):
                if token in exact_hits:
                    matching_tokens.append((index, token))
                    continue
                if token in surfaces.values() or token in canonical_surfaces.values():
                    matching_tokens.append((index, token))

            if matching_tokens:
                first_match_index = min(index for index, _ in matching_tokens)
                if first_match_index == 0:
                    multiplier *= 2.0
                elif first_match_index <= 2:
                    multiplier *= 1.55
                else:
                    multiplier *= 1.2
                if any(token == surfaces["file_stem"] for _, token in matching_tokens):
                    multiplier *= 1.15

            primary_token = query_tokens[0]
            if primary_token and primary_token not in canonical_surfaces.values():
                trailing_variant_hits = sum(
                    1
                    for surface in surfaces.values()
                    if _is_trailing_variant(surface, primary_token)
                )
                if trailing_variant_hits:
                    multiplier *= 0.78
                else:
                    multiplier *= 0.72

        if is_exact_lookup_weak_path(file_path):
            multiplier *= 0.65

        if chunk_type == "doc":
            multiplier *= 1.05
        elif chunk_type == "code":
            multiplier *= 0.9

        point.score *= multiplier

    return results


def apply_search_boosts(
    results: list[models.ScoredPoint],
    settings: Settings,
    intent: QueryIntent | None = None,
    query: str = "",
) -> list[models.ScoredPoint]:
    """Apply multi-factor boosting to search results in place."""
    doc_category_boosts = settings.doc_category_boosts
    file_type_boosts = settings.file_type_boosts
    intent_file_type_boosts = settings.intent_file_type_boosts
    path_boosts = settings.path_hierarchy_boosts
    heading_boosts = settings.heading_level_boosts
    token_thresholds = settings.token_count_thresholds
    token_boosts = settings.token_count_boosts

    intent_category_boosts: dict[str, float] = {}
    if intent is not None:
        from repowise.docs.search.intent import get_category_boosts

        intent_category_boosts = get_category_boosts(intent)

    compiled_path_patterns: list[tuple[re.Pattern[str], float]] = []
    for pattern, boost in path_boosts.items():
        try:
            compiled_path_patterns.append((re.compile(pattern, re.IGNORECASE), boost))
        except re.error:
            logger.warning("Invalid path pattern: %s", pattern)

    for point in results:
        if point.score is None:
            continue

        payload = point.payload or {}
        total_boost = 1.0

        doc_category = payload.get("doc_category", "other")
        total_boost *= doc_category_boosts.get(doc_category, 1.0)
        if doc_category in intent_category_boosts:
            total_boost *= intent_category_boosts[doc_category]

        file_type = payload.get("file_type", "other")
        total_boost *= file_type_boosts.get(file_type, 1.0)
        if intent is not None:
            intent_key = intent.value
            if intent_key in intent_file_type_boosts:
                intent_ft_boosts = intent_file_type_boosts[intent_key]
                if file_type in intent_ft_boosts:
                    total_boost *= intent_ft_boosts[file_type]

        file_path = payload.get("file_path", "")
        if file_path:
            for pattern, boost in compiled_path_patterns:
                if pattern.search(file_path):
                    total_boost *= boost
                    break

        heading_level = payload.get("heading_level")
        if heading_level is not None:
            try:
                level = int(heading_level)
                total_boost *= heading_boosts.get(level, 1.0)
            except (ValueError, TypeError):
                pass

        token_count = payload.get("token_count")
        if token_count is not None:
            try:
                count = int(token_count)
                min_substantive = token_thresholds.get("min_substantive", 400)
                max_fragment = token_thresholds.get("max_fragment", 100)

                if count > min_substantive:
                    total_boost *= token_boosts.get("substantive", 1.05)
                elif count < max_fragment:
                    total_boost *= token_boosts.get("fragment", 0.90)
            except (ValueError, TypeError):
                pass

        if query:
            total_boost *= _surface_match_multiplier(payload, query)

        if intent is not None and intent.value != "error_debug" and _is_error_like_surface(payload):
            total_boost *= 0.6

        point.score *= total_boost

    return results
