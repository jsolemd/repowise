"""Which directory include_patterns match against, and what happens when nothing matches.

Regression for 2026-09-26: /observablehq/plot was added with root-relative patterns
(``docs/**/*.md``, ``src/**/*.d.ts``, ``README.md``) and no docs_path. Discovery narrowed
the search to ``docs/``, every pattern missed, and the library reported ``ready`` with
0 files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from repowise.docs.library import repository
from repowise.docs.library.git_manager import (
    describe_pattern_base,
    list_doc_files,
    resolve_doc_search_root,
)
from repowise.docs.library.models import (
    IndexJob,
    JobEnqueueDisposition,
    JobEnqueueResult,
    JobType,
    LibraryConfig,
    LibraryState,
    LibraryStatus,
)
from repowise.docs.pipeline import index_library
from repowise.docs.tools.handlers import handle_add_library


def _plot_like_repo(root: Path) -> Path:
    (root / "docs" / "marks").mkdir(parents=True)
    (root / "docs" / "index.md").write_text("# Plot\n")
    (root / "docs" / "marks" / "dot.md").write_text("# Dot\n")
    (root / "src").mkdir()
    (root / "src" / "plot.d.ts").write_text("export declare function plot(): void;\n")
    (root / "README.md").write_text("# Observable Plot\n")
    (root / "CHANGELOG.md").write_text("# Changelog\n")
    return root


@pytest.mark.unit
def test_empty_docs_path_with_root_relative_patterns_matches_from_the_root(tmp_path: Path):
    repo = _plot_like_repo(tmp_path)

    files = list_doc_files(
        repo,
        docs_path="",
        include_patterns=["docs/**/*.md", "src/**/*.d.ts", "README.md", "CHANGELOG.md"],
    )

    assert set(files) == {
        Path("docs/index.md"),
        Path("docs/marks/dot.md"),
        Path("src/plot.d.ts"),
        Path("README.md"),
        Path("CHANGELOG.md"),
    }


@pytest.mark.unit
def test_dot_docs_path_is_the_root_even_with_default_patterns(tmp_path: Path):
    repo = _plot_like_repo(tmp_path)

    files = list_doc_files(repo, docs_path=".")

    assert resolve_doc_search_root(repo, ".") == ""
    assert Path("README.md") in files
    assert Path("docs/index.md") in files


@pytest.mark.unit
def test_docs_path_is_the_base_for_relative_patterns(tmp_path: Path):
    repo = _plot_like_repo(tmp_path)

    files = list_doc_files(repo, docs_path="docs", include_patterns=["marks/*.md", "*.md"])

    assert set(files) == {Path("docs/index.md"), Path("docs/marks/dot.md")}


@pytest.mark.unit
def test_nothing_given_auto_discovers_the_docs_directory(tmp_path: Path):
    repo = _plot_like_repo(tmp_path)

    files = list_doc_files(repo, docs_path="")

    assert resolve_doc_search_root(repo, "") == "docs"
    assert Path("docs/index.md") in files
    assert Path("README.md") not in files


@pytest.mark.unit
def test_unanchored_patterns_keep_auto_discovery(tmp_path: Path):
    repo = _plot_like_repo(tmp_path)

    files = list_doc_files(repo, docs_path="", include_patterns=["**/*.md"])

    assert set(files) == {Path("docs/index.md"), Path("docs/marks/dot.md")}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "stored"),
    [("", ""), (".", "."), ("./", "."), ("./docs/", "docs"), ("docs\\src", "docs/src")],
)
def test_docs_path_normalization_keeps_explicit_root(raw, stored):
    config = LibraryConfig(repo="owner/repo", name="Repo", docs_path=raw)

    assert config.docs_path == stored


@pytest.mark.unit
def test_describe_pattern_base_names_each_case():
    assert describe_pattern_base(".", ["**/*.md"]).startswith("library root")
    assert describe_pattern_base("docs", ["**/*.md"]) == "'docs' (docs_path)"
    assert describe_pattern_base("", ["**/*.md", "README.md"]).startswith("library root")
    assert describe_pattern_base("", ["**/*.md"]).startswith("auto-discovered")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_repo_does_not_sparse_checkout_an_explicit_root(tmp_path, monkeypatch):
    calls: list[tuple[str, ...]] = []

    async def fake_run_git(*args, cwd=None):
        calls.append(tuple(args))
        return ""

    monkeypatch.setattr(repository, "run_git", fake_run_git)
    monkeypatch.setattr(repository, "get_repo_path", lambda repo: tmp_path / "clone")

    await repository.clone_repo("owner/repo", "main", docs_path=".")

    assert not any(call[0] == "sparse-checkout" for call in calls)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_library_keeps_dot_and_reports_the_pattern_base():
    stored = LibraryState(library_id="/owner/repo", repo="owner/repo", name="Repo")

    with (
        patch("repowise.docs.tools.handlers.get_remote_head_sha", new_callable=AsyncMock) as sha,
        patch("repowise.docs.tools.handlers.get_library", new_callable=AsyncMock) as get,
        patch("repowise.docs.tools.handlers.upsert_library", new_callable=AsyncMock) as upsert,
        patch("repowise.docs.tools.handlers.enqueue_job", new_callable=AsyncMock) as enqueue,
    ):
        sha.return_value = "abc123"
        get.return_value = None
        upsert.return_value = stored
        enqueue.return_value = JobEnqueueResult(
            disposition=JobEnqueueDisposition.QUEUED,
            job=IndexJob(id="job-1", library_id="/owner/repo", job_type=JobType.FULL, priority=3),
        )

        result = await handle_add_library(
            {
                "repo": "owner/repo",
                "name": "Repo",
                "docs_path": ".",
                "include_patterns": ["**/*.md"],
            }
        )

    assert upsert.call_args[0][0].docs_path == "."
    assert result["docs_path"] == "."
    assert result["pattern_base"].startswith("library root")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_index_matching_no_files_is_an_error_not_ready(tmp_path: Path):
    repo_dir = _plot_like_repo(tmp_path / "repo")
    library = LibraryState(
        library_id="/owner/repo",
        repo="owner/repo",
        name="Repo",
        docs_path="docs",
        include_patterns=["src/**/*.d.ts"],
        status=LibraryStatus.READY,
        current_sha="old-sha",
        file_count=3,
        chunk_count=9,
    )

    with (
        patch("repowise.docs.pipeline.get_qdrant_client", return_value=MagicMock()),
        patch("repowise.docs.pipeline.ensure_collection"),
        patch("repowise.docs.pipeline.get_library", new_callable=AsyncMock, return_value=library),
        patch(
            "repowise.docs.pipeline.update_library_status", new_callable=AsyncMock
        ) as update_status,
        patch("repowise.docs.pipeline.clone_repo", new_callable=AsyncMock, return_value=repo_dir),
        patch("repowise.docs.pipeline.get_head_sha", new_callable=AsyncMock, return_value="sha"),
        patch(
            "repowise.docs.pipeline.get_latest_source_ref",
            new_callable=AsyncMock,
            return_value="sha",
        ),
        patch("repowise.docs.pipeline.delete_files_outside_scope", new_callable=AsyncMock) as prune,
    ):
        update_status.return_value = True
        with pytest.raises(ValueError, match=r"Indexed 0 files.*src/\*\*/\*\.d\.ts.*under 'docs'"):
            await index_library("/owner/repo")

    final = update_status.await_args_list[-1].kwargs
    assert final["status"] == LibraryStatus.ERROR
    assert "matched nothing under 'docs'" in final["error_message"]
    assert not any(
        call.kwargs.get("status") == LibraryStatus.READY for call in update_status.await_args_list
    )
    # The earlier good index survives a misconfigured run.
    prune.assert_not_awaited()
