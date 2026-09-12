"""Document category and file type classification from file paths.

Classifies documentation files into categories (reference, guide, example,
changelog, blog) based on path patterns. Used for search result boosting
to prioritize canonical reference documentation.

Also classifies file types (documentation, source, test, example, config)
to enable boosting documentation over raw source code in search results.
"""

import re
from enum import StrEnum
from pathlib import Path

from repowise.docs.chunking.models import ChunkType, DocCategory
from repowise.docs.formats import (
    CONFIG_EXTENSIONS,
    CONFIG_FILENAMES,
    DOC_EXTENSIONS,
    SOURCE_EXTENSIONS,
)


class FileType(StrEnum):
    """File type classification for search result boosting.

    Used to boost documentation files over source code in typical usage queries,
    while still allowing source code to rank higher for implementation queries.
    """

    DOCUMENTATION = "documentation"  # .md, .mdx, .rst, .adoc, .ipynb
    EXAMPLE = "example"  # Files in examples/, cookbook/, tutorial/ paths
    SOURCE = "source"  # .py, .ts, .js, .go, .rs (actual source code)
    CONFIG = "config"  # .yaml, .yml, .toml, pyproject.toml
    TEST = "test"  # Files in test/, tests/, *_test.*, *.spec.*
    OTHER = "other"  # Uncategorized


# Path patterns for classification (order matters - first match wins)
# Patterns are case-insensitive and match path components
# Use (?:^|/) to match at start of path or after slash

# Reference documentation patterns
REFERENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|/)reference/", re.IGNORECASE),
    re.compile(r"(?:^|/)api-reference/", re.IGNORECASE),
    re.compile(r"(?:^|/)docs/api/", re.IGNORECASE),
    re.compile(r"(?:^|/)docs/sdk/", re.IGNORECASE),
    re.compile(r"(?:^|/)api/", re.IGNORECASE),
    re.compile(r"03-api-reference", re.IGNORECASE),  # Next.js style numbering
    re.compile(r"(?:^|/)sdk/", re.IGNORECASE),
    re.compile(r"(?:^|/)api-docs/", re.IGNORECASE),
]

# Guide/tutorial patterns
GUIDE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|/)learn/", re.IGNORECASE),
    re.compile(r"(?:^|/)tutorials?/", re.IGNORECASE),
    re.compile(r"(?:^|/)getting-started/", re.IGNORECASE),
    re.compile(r"(?:^|/)get-started/", re.IGNORECASE),
    re.compile(r"(?:^|/)guides?/", re.IGNORECASE),
    re.compile(r"(?:^|/)quickstart/", re.IGNORECASE),
    re.compile(r"01-getting-started", re.IGNORECASE),  # Next.js style
    re.compile(r"02-guides?", re.IGNORECASE),
]

# Example/cookbook patterns
EXAMPLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|/)examples?/", re.IGNORECASE),
    re.compile(r"(?:^|/)cookbooks?/", re.IGNORECASE),
    re.compile(r"(?:^|/)recipes?/", re.IGNORECASE),
    re.compile(r"(?:^|/)samples?/", re.IGNORECASE),
    re.compile(r"(?:^|/)demos?/", re.IGNORECASE),
]

# Changelog/release notes patterns
CHANGELOG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"changelog", re.IGNORECASE),
    re.compile(r"(?:^|/)releases?/", re.IGNORECASE),
    re.compile(r"history\.md", re.IGNORECASE),
    re.compile(r"changes\.md", re.IGNORECASE),
    re.compile(r"release-notes", re.IGNORECASE),
    re.compile(r"releasenotes", re.IGNORECASE),
    re.compile(r"what-?s-?new", re.IGNORECASE),
]

# Blog/announcement patterns
BLOG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|/)blog/", re.IGNORECASE),
    re.compile(r"(?:^|/)posts?/", re.IGNORECASE),
    re.compile(r"(?:^|/)articles?/", re.IGNORECASE),
    re.compile(r"(?:^|/)news/", re.IGNORECASE),
    re.compile(r"\d{4}/\d{2}/\d{2}/", re.IGNORECASE),  # Date-based paths (e.g., 2024/01/15/)
    re.compile(r"\d{4}-\d{2}-\d{2}", re.IGNORECASE),  # Date in filename (e.g., 2024-01-15-post.md)
]


def classify_doc_type(file_path: str, chunk_type: ChunkType) -> DocCategory:
    """
    Classify document category from file path and chunk type.

    The classification is used for search result boosting to prioritize
    canonical reference documentation over tutorials, changelogs, and blog posts.

    Classification priority (first match wins):
    1. ChunkType.CHANGELOG always maps to DocCategory.CHANGELOG
    2. Path-based pattern matching (reference > guide > example > changelog > blog)
    3. Falls back to DocCategory.OTHER

    Args:
        file_path: Relative path within the repo (e.g., "docs/api/tracing.mdx")
        chunk_type: Type of chunk (doc, code, config, changelog)

    Returns:
        DocCategory enum value

    Examples:
        >>> classify_doc_type("docs/reference/api.md", ChunkType.DOC)
        DocCategory.REFERENCE

        >>> classify_doc_type("guides/getting-started.md", ChunkType.DOC)
        DocCategory.GUIDE

        >>> classify_doc_type("CHANGELOG.md", ChunkType.CHANGELOG)
        DocCategory.CHANGELOG

        >>> classify_doc_type("blog/2024/01/15/release.md", ChunkType.DOC)
        DocCategory.BLOG
    """
    # ChunkType.CHANGELOG always maps to CHANGELOG category
    if chunk_type == ChunkType.CHANGELOG:
        return DocCategory.CHANGELOG

    # Normalize path for matching
    path = file_path.replace("\\", "/")  # Handle Windows paths

    # Check patterns in order of priority
    pattern_groups: list[tuple[list[re.Pattern[str]], DocCategory]] = [
        (REFERENCE_PATTERNS, DocCategory.REFERENCE),
        (GUIDE_PATTERNS, DocCategory.GUIDE),
        (EXAMPLE_PATTERNS, DocCategory.EXAMPLE),
        (CHANGELOG_PATTERNS, DocCategory.CHANGELOG),
        (BLOG_PATTERNS, DocCategory.BLOG),
    ]

    for patterns, category in pattern_groups:
        for pattern in patterns:
            if pattern.search(path):
                return category

    return DocCategory.OTHER


def classify_file_type(file_path: str) -> FileType:
    """
    Classify file type from path for search result boosting.

    The classification determines how source code vs documentation is boosted
    in search results. Documentation is typically boosted for "how to use X"
    queries, while source code can be boosted for "how is X implemented" queries.

    Classification priority (first match wins):
    1. Documentation files (.md, .mdx, .rst, .adoc, .ipynb)
    2. Test files (test/ directories, *_test.*, *.spec.*)
    3. Example files (examples/, cookbook/, tutorial/ paths)
    4. Config files (.yaml, .toml, pyproject.toml, etc.)
    5. Source code files (by extension)
    6. Falls back to OTHER

    Args:
        file_path: Relative path within the repo (e.g., "src/langfuse/decorators.py")

    Returns:
        FileType enum value

    Examples:
        >>> classify_file_type("docs/getting-started.md")
        FileType.DOCUMENTATION

        >>> classify_file_type("tests/test_decorators.py")
        FileType.TEST

        >>> classify_file_type("examples/openai-integration.py")
        FileType.EXAMPLE

        >>> classify_file_type("src/langfuse/decorators.py")
        FileType.SOURCE

        >>> classify_file_type("pyproject.toml")
        FileType.CONFIG
    """
    # Normalize path for matching
    path = file_path.replace("\\", "/")
    path_lower = path.lower()
    p = Path(path)
    ext = p.suffix.lower()
    filename = p.name.lower()

    # 1. Documentation files (always documentation regardless of path)
    if ext in DOC_EXTENSIONS:
        return FileType.DOCUMENTATION

    # 2. Test files (check before source - test files are often .py/.ts)
    if (
        "/test/" in path_lower
        or "/tests/" in path_lower
        or "/__tests__/" in path_lower
        or "_test." in path_lower
        or ".test." in path_lower
        or ".spec." in path_lower
        or filename.startswith("test_")
    ):
        return FileType.TEST

    # 3. Example files
    if (
        "/examples/" in path_lower
        or "/example/" in path_lower
        or "/cookbook/" in path_lower
        or "/cookbooks/" in path_lower
        or "/tutorial/" in path_lower
        or "/tutorials/" in path_lower
        or "/demo/" in path_lower
        or "/demos/" in path_lower
        or "/samples/" in path_lower
        or "/sample/" in path_lower
    ):
        return FileType.EXAMPLE

    # 4. Config files (by name or extension)
    if filename.startswith(".env"):
        return FileType.CONFIG
    if filename in CONFIG_FILENAMES:
        return FileType.CONFIG
    if path_lower.endswith(".gradle.kts"):
        return FileType.CONFIG
    if ext in CONFIG_EXTENSIONS:
        return FileType.CONFIG

    # 5. Source code files
    if ext in SOURCE_EXTENSIONS:
        return FileType.SOURCE

    return FileType.OTHER
