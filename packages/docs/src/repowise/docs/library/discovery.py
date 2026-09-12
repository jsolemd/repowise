"""Documentation discovery helpers for doc-search git management."""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path, PurePosixPath

from repowise.docs.formats import (
    AUTO_EXCLUDE_PATTERNS,
    DEFAULT_INCLUDE_PATTERNS,
    DOC_DISCOVERY_PATTERNS,
    SOURCE_EXTENSIONS,
)
from repowise.docs.library.git_exec import run_git

logger = logging.getLogger(__name__)

COMMON_DOCS_PATHS = [
    "docs",
    "doc",
    "documentation",
    "apps/docs",
    "apps/docs/content",
    "apps/docs/pages",
    "apps/documentation",
    "website/docs",
    "website/content",
    "website",
    "packages/docs",
    "packages/documentation",
    "pages",
    "pages/docs",
    "content",
    "content/docs",
    "guide",
    "guides",
    "manual",
    "wiki",
    "api-docs",
    "api",
]


def _matches_glob(path_str: str, pattern: str) -> bool:
    if fnmatch.fnmatch(path_str, pattern):
        return True
    return bool(pattern.startswith("**/") and fnmatch.fnmatch(path_str, pattern[3:]))


def _matches_doc_discovery_patterns(path_str: str) -> bool:
    return any(_matches_glob(path_str, pattern) for pattern in DOC_DISCOVERY_PATTERNS)


def _iter_discovery_matches(repo_dir: Path) -> list[str]:
    matches: set[str] = set()
    for pattern in DOC_DISCOVERY_PATTERNS:
        candidate_patterns = [pattern]
        if pattern.startswith("**/"):
            candidate_patterns.append(pattern[3:])
        for candidate_pattern in candidate_patterns:
            for match in repo_dir.glob(candidate_pattern):
                if match.is_file():
                    matches.add(match.relative_to(repo_dir).as_posix())
    return sorted(matches)


def _iter_matching_parent_dirs(path_str: str) -> list[str]:
    path = PurePosixPath(path_str)
    matches: list[str] = []
    for parent in path.parents:
        parent_str = str(parent)
        if parent_str in ("", "."):
            continue
        if any(
            parent_str == common_path or parent_str.endswith(f"/{common_path}")
            for common_path in COMMON_DOCS_PATHS
        ):
            matches.append(parent_str)
    return matches


def _discover_docs_path_from_paths(paths: list[str], preferred_path: str = "") -> str | None:
    """Infer the best docs path from tracked repository paths."""
    if preferred_path:
        preferred_prefix = preferred_path.rstrip("/\\")
        if any(
            path == preferred_prefix or path.startswith(f"{preferred_prefix}/") for path in paths
        ):
            return preferred_path

    candidate_counts: dict[str, int] = {}
    for path in paths:
        if not _matches_doc_discovery_patterns(path):
            continue
        for candidate in _iter_matching_parent_dirs(path):
            candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1

    if candidate_counts:
        return max(
            candidate_counts.items(),
            key=lambda item: (item[1], item[0].count("/"), len(item[0])),
        )[0]

    root_doc_count = sum(
        1 for path in paths if "/" not in path and _matches_doc_discovery_patterns(path)
    )
    if root_doc_count > 0:
        return ""
    return None


def discover_docs_path(repo_dir: Path, preferred_path: str = "") -> str | None:
    """Discover the documentation directory in a repository."""
    if preferred_path:
        preferred_full = repo_dir / preferred_path
        if preferred_full.exists() and preferred_full.is_dir():
            logger.debug("Using preferred docs path: %s", preferred_path)
            return preferred_path

        logger.info(
            "Preferred docs path '%s' not found, searching common locations...",
            preferred_path,
        )

    doc_paths = _iter_discovery_matches(repo_dir)
    discovered = _discover_docs_path_from_paths(doc_paths, preferred_path)
    if discovered == "":
        logger.info("No docs directory found, using repo root")
    elif discovered:
        logger.info("Discovered docs path: %s", discovered)
    else:
        logger.warning("No documentation files found in repository")
    return discovered


async def discover_docs_path_from_git(repo_dir: Path, preferred_path: str = "") -> str | None:
    """Discover docs path from the full git tree, even when the worktree is sparse."""
    output = await run_git("ls-tree", "-r", "--name-only", "HEAD", cwd=repo_dir)
    paths = [line.strip() for line in output.splitlines() if line.strip()]
    if not paths:
        return None
    discovered = _discover_docs_path_from_paths(paths, preferred_path)
    if discovered == "":
        logger.info("Discovered docs path from git tree: repo root")
    elif discovered:
        logger.info("Discovered docs path from git tree: %s", discovered)
    return discovered


def get_related_docs_repos(repo: str) -> list[str]:
    """Get potential related documentation repository names."""
    owner, name = repo.split("/", 1)
    patterns = [
        f"{owner}/{name}-docs",
        f"{owner}/{name}-documentation",
        f"{owner}/{name}-website",
        f"{owner}/{name}.github.io",
        f"{owner}/docs",
        f"{owner}/documentation",
    ]
    return [p for p in patterns if p != repo]


def list_doc_files(
    repo_dir: Path,
    docs_path: str = "",
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """List documentation files in a repository."""
    if include_patterns is None:
        include_patterns = list(DEFAULT_INCLUDE_PATTERNS)
    if exclude_patterns is None:
        exclude_patterns = []

    docs_path_has_version = bool(docs_path and re.search(r"[/\\]v\d|[/\\]\d+\.\d+", docs_path))

    default_version_excludes = [
        "**/v[0-9]*/**",
        "**/[0-9].[0-9x]*/**",
        "**/legacy/**",
        "**/deprecated/**",
        "**/archive/**",
        "**/archived/**",
        "**/old/**",
        "**/previous/**",
    ]

    default_excludes = default_version_excludes + AUTO_EXCLUDE_PATTERNS
    if docs_path_has_version:
        logger.info("Docs path '%s' contains version, skipping version excludes", docs_path)
        exclude_patterns = list(exclude_patterns) + default_excludes[2:]
    else:
        exclude_patterns = list(exclude_patterns) + default_excludes

    def _pattern_has_source_extension(pattern: str) -> bool:
        pattern_lower = pattern.lower()
        if pattern_lower.endswith(".gradle.kts"):
            return False
        return any(
            pattern_lower.endswith(ext) or pattern_lower.endswith(f"*{ext}")
            for ext in SOURCE_EXTENSIONS
        )

    has_source_patterns = any(
        _pattern_has_source_extension(pattern) for pattern in include_patterns
    )

    search_root = repo_dir
    if docs_path:
        search_root = repo_dir / docs_path
        if not search_root.exists():
            logger.info("Docs path '%s' not found, attempting auto-discovery...", docs_path)
            discovered = discover_docs_path(repo_dir, docs_path)
            if discovered is None:
                logger.warning("No documentation found in %s", repo_dir)
                return []
            if discovered == "":
                search_root = repo_dir
                logger.info("Using repo root for documentation")
            else:
                search_root = repo_dir / discovered
                logger.info("Auto-discovered docs at: %s", discovered)
    elif has_source_patterns:
        discovered = discover_docs_path(repo_dir)
        if discovered is None:
            logger.info("No docs path discovered; source patterns present, using repo root")
        elif discovered == "":
            logger.info("Using repo root for documentation and source patterns")
        else:
            search_root = repo_dir / discovered
            logger.info("Auto-discovered docs at: %s", discovered)
    else:
        discovered = discover_docs_path(repo_dir)
        if discovered is None:
            logger.warning("No documentation found in %s", repo_dir)
            return []
        if discovered != "":
            search_root = repo_dir / discovered
            logger.info("Auto-discovered docs at: %s", discovered)

    found_files: set[Path] = set()
    for pattern in include_patterns:
        for match in search_root.glob(pattern):
            if match.is_file():
                found_files.add(match.relative_to(repo_dir))

    result = []
    for file_path in sorted(found_files):
        file_str = str(file_path)
        if any(_matches_glob(file_str, exclude_pattern) for exclude_pattern in exclude_patterns):
            continue
        result.append(file_path)

    return result
