"""Tests for git_manager docs discovery functions."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from repowise.docs.library.git_manager import (
    COMMON_DOCS_PATHS,
    discover_docs_path,
    discover_docs_path_from_git,
    get_related_docs_repos,
    list_doc_files,
    resolve_repo_docs_path,
)


class TestDiscoverDocsPath:
    """Tests for discover_docs_path function."""

    def test_preferred_path_exists(self, tmp_path: Path):
        """Should return preferred path when it exists."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.md").write_text("# Hello")

        result = discover_docs_path(tmp_path, "docs")
        assert result == "docs"

    def test_preferred_path_not_found_discovers_alternative(self, tmp_path: Path):
        """Should discover alternative when preferred path doesn't exist."""
        # Create docs in a different location
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        (content_dir / "guide.md").write_text("# Guide")

        result = discover_docs_path(tmp_path, "docs")  # "docs" doesn't exist
        assert result == "content"

    def test_discovers_docs_directory(self, tmp_path: Path):
        """Should find standard 'docs' directory."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "getting-started.md").write_text("# Getting Started")

        result = discover_docs_path(tmp_path)
        assert result == "docs"

    def test_discovers_monorepo_apps_docs(self, tmp_path: Path):
        """Should find monorepo apps/docs structure."""
        apps_docs = tmp_path / "apps" / "docs"
        apps_docs.mkdir(parents=True)
        (apps_docs / "index.md").write_text("# Docs")

        result = discover_docs_path(tmp_path)
        assert result == "apps/docs"

    def test_discovers_nested_pages_docs_suffix(self, tmp_path: Path):
        """Should discover nested docs directories whose suffix matches a common docs path."""
        pages_dir = tmp_path / "apps" / "mantine.dev" / "src" / "pages" / "core"
        pages_dir.mkdir(parents=True)
        (pages_dir / "textarea.mdx").write_text("# Textarea")
        (pages_dir / "stack.mdx").write_text("# Stack")

        result = discover_docs_path(tmp_path, "docs")
        assert result == "apps/mantine.dev/src/pages"

    def test_prefers_path_with_more_markdown_files(self, tmp_path: Path):
        """Should prefer directory with more markdown files."""
        # Create docs/ with 1 file
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "one.md").write_text("# One")

        # Create website/ with 3 files
        website_dir = tmp_path / "website"
        website_dir.mkdir()
        (website_dir / "a.md").write_text("# A")
        (website_dir / "b.md").write_text("# B")
        (website_dir / "c.md").write_text("# C")

        result = discover_docs_path(tmp_path)
        assert result == "website"

    def test_returns_empty_string_for_root_markdown(self, tmp_path: Path):
        """Should return empty string when markdown is only at root."""
        (tmp_path / "README.md").write_text("# Project")
        (tmp_path / "CONTRIBUTING.md").write_text("# Contributing")

        result = discover_docs_path(tmp_path)
        assert result == ""

    def test_returns_none_when_no_docs(self, tmp_path: Path):
        """Should return None when no documentation found."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")

        result = discover_docs_path(tmp_path)
        assert result is None

    def test_counts_mdx_files(self, tmp_path: Path):
        """Should count .mdx files in addition to .md."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "page.mdx").write_text("# MDX Page")

        result = discover_docs_path(tmp_path)
        assert result == "docs"


class TestGetRelatedDocsRepos:
    """Tests for get_related_docs_repos function."""

    def test_returns_docs_variants(self):
        """Should return common documentation repo name patterns."""
        repos = get_related_docs_repos("langfuse/langfuse")

        assert "langfuse/langfuse-docs" in repos
        assert "langfuse/langfuse-documentation" in repos
        assert "langfuse/langfuse-website" in repos
        assert "langfuse/langfuse.github.io" in repos
        assert "langfuse/docs" in repos

    def test_excludes_original_repo(self):
        """Should not include the original repo in results."""
        repos = get_related_docs_repos("owner/docs")

        # "owner/docs" should not be in the list since it's the original
        assert "owner/docs" not in repos

    def test_handles_complex_repo_names(self):
        """Should handle repos with hyphens and underscores."""
        repos = get_related_docs_repos("vercel/next.js")

        assert "vercel/next.js-docs" in repos
        assert "vercel/next.js-website" in repos


class TestListDocFilesWithDiscovery:
    """Tests for list_doc_files with auto-discovery integration."""

    def test_specified_path_not_found_triggers_discovery(self, tmp_path: Path):
        """Should auto-discover when specified docs_path doesn't exist."""
        # Create docs in 'content/' instead of 'docs/'
        content_dir = tmp_path / "content"
        content_dir.mkdir()
        (content_dir / "guide.md").write_text("# Guide")

        # Request docs_path="docs" which doesn't exist
        files = list_doc_files(tmp_path, docs_path="docs")

        # Should discover and use 'content/' instead
        assert len(files) == 1
        assert Path("content/guide.md") in files

    def test_no_docs_path_triggers_discovery(self, tmp_path: Path):
        """Should auto-discover when no docs_path specified."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.md").write_text("# Index")

        files = list_doc_files(tmp_path, docs_path="")

        assert len(files) == 1
        assert Path("docs/index.md") in files

    def test_excludes_version_directories(self, tmp_path: Path):
        """Should exclude old version directories by default."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "current.md").write_text("# Current")

        # Old version directories that should be excluded
        v1_dir = docs_dir / "v1"
        v1_dir.mkdir()
        (v1_dir / "old.md").write_text("# Old")

        legacy_dir = docs_dir / "legacy"
        legacy_dir.mkdir()
        (legacy_dir / "deprecated.md").write_text("# Deprecated")

        files = list_doc_files(tmp_path, docs_path="docs")

        # Only current.md should be included
        assert len(files) == 1
        assert Path("docs/current.md") in files

    def test_returns_empty_when_no_docs_found(self, tmp_path: Path):
        """Should return empty list when no docs discovered."""
        # Only source files, no markdown
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('hello')")

        files = list_doc_files(tmp_path, docs_path="nonexistent")

        assert files == []

    def test_source_patterns_use_discovered_docs_path_when_available(self, tmp_path: Path):
        """Source include patterns should not force repo-root if docs path exists."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.md").write_text("# Docs")

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('hello')")

        files = list_doc_files(
            tmp_path,
            docs_path="",
            include_patterns=["**/*.md", "**/*.py"],
        )

        assert Path("docs/index.md") in files
        assert Path("src/main.py") not in files

    def test_source_patterns_fallback_to_repo_root_without_docs(self, tmp_path: Path):
        """If no docs path exists, source patterns should index from repo root."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('hello')")

        files = list_doc_files(
            tmp_path,
            docs_path="",
            include_patterns=["**/*.py"],
        )

        assert Path("src/main.py") in files

    def test_auto_excludes_filter_common_noisy_assets(self, tmp_path: Path):
        """Default discovery should skip generated, test, and vendor asset paths."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide")

        cypress_dir = tmp_path / "packages" / "demo" / "cypress" / "integration"
        cypress_dir.mkdir(parents=True)
        (cypress_dir / "drag.ts").write_text("describe('drag')", encoding="utf-8")

        mocks_dir = tmp_path / "packages" / "demo" / "src" / "mocks"
        mocks_dir.mkdir(parents=True)
        (mocks_dir / "sample.ts").write_text("export const sample = 1", encoding="utf-8")

        static_dir = docs_dir / "_static" / "prism"
        static_dir.mkdir(parents=True)
        (static_dir / "prism.js").write_text("console.log('vendor')", encoding="utf-8")

        scripts_dir = docs_dir / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "highlight.min.js").write_text("console.log('min')", encoding="utf-8")

        story_dir = tmp_path / "packages" / "ui" / "src"
        story_dir.mkdir(parents=True)
        (story_dir / "Button.story.ts").write_text("export const Basic = 1", encoding="utf-8")

        files = list_doc_files(
            tmp_path,
            docs_path="",
            include_patterns=["**/*.md", "**/*.ts", "**/*.js"],
        )

        assert files == [Path("docs/guide.md")]

    def test_root_level_exclude_patterns_match_without_leading_directory(self, tmp_path: Path):
        """Exclude patterns like **/dev/** should also match root-level dev paths."""
        dev_dir = tmp_path / "dev" / "react" / "src"
        dev_dir.mkdir(parents=True)
        (dev_dir / "example.tsx").write_text("export const Example = 1", encoding="utf-8")

        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        (test_dir / "view.spec.ts").write_text("describe('view')", encoding="utf-8")

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.md").write_text("# Docs", encoding="utf-8")

        files = list_doc_files(
            tmp_path,
            docs_path="",
            include_patterns=["**/*.md", "**/*.ts", "**/*.tsx"],
            exclude_patterns=["**/dev/**", "**/tests/**"],
        )

        assert files == [Path("docs/index.md")]


@pytest.mark.asyncio
async def test_discover_docs_path_from_git_finds_nested_pages_suffix(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    git_output = "\n".join(
        [
            "README.md",
            "apps/mantine.dev/src/pages/core/textarea.mdx",
            "apps/mantine.dev/src/pages/core/stack.mdx",
            "packages/core/src/Stack.tsx",
        ]
    )

    with patch("repowise.docs.library.discovery.run_git", new=AsyncMock(return_value=git_output)):
        result = await discover_docs_path_from_git(repo_dir, "docs")

    assert result == "apps/mantine.dev/src/pages"


@pytest.mark.asyncio
async def test_resolve_repo_docs_path_rewrites_sparse_checkout_when_preferred_path_missing(
    tmp_path: Path,
):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    with (
        patch(
            "repowise.docs.library.discovery.run_git",
            new=AsyncMock(return_value="apps/mantine.dev/src/pages/core/textarea.mdx\n"),
        ),
        patch(
            "repowise.docs.library.repository.run_git", new=AsyncMock(return_value="")
        ) as mock_run_git,
    ):
        result = await resolve_repo_docs_path(repo_dir, branch="master", docs_path="docs")

    assert result == "apps/mantine.dev/src/pages"
    mock_run_git.assert_any_await(
        "sparse-checkout",
        "set",
        "apps/mantine.dev/src/pages",
        cwd=repo_dir,
    )
    mock_run_git.assert_any_await("checkout", "master", cwd=repo_dir)


class TestCommonDocsPaths:
    """Tests for the COMMON_DOCS_PATHS constant."""

    def test_includes_standard_paths(self):
        """Should include standard documentation paths."""
        assert "docs" in COMMON_DOCS_PATHS
        assert "doc" in COMMON_DOCS_PATHS
        assert "documentation" in COMMON_DOCS_PATHS

    def test_includes_monorepo_paths(self):
        """Should include monorepo-style paths."""
        assert "apps/docs" in COMMON_DOCS_PATHS
        assert "packages/docs" in COMMON_DOCS_PATHS

    def test_includes_nextjs_paths(self):
        """Should include Next.js/Nextra documentation paths."""
        assert "pages" in COMMON_DOCS_PATHS
        assert "content" in COMMON_DOCS_PATHS
