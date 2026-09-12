"""Focused tests for add-library defaults and format coverage."""

from unittest.mock import AsyncMock, patch

import pytest

from repowise.docs.formats import DEFAULT_INCLUDE_PATTERNS
from repowise.docs.library.models import (
    IndexJob,
    JobEnqueueDisposition,
    JobEnqueueResult,
    JobType,
    LibraryState,
    LibraryStatus,
)
from repowise.docs.tools.handlers import handle_add_library


@pytest.mark.unit
async def test_add_library_uses_broad_default_include_patterns() -> None:
    mock_lib = LibraryState(
        library_id="/owner/repo",
        repo="owner/repo",
        name="Repo",
        status=LibraryStatus.PENDING,
        docs_path="",
        branch="main",
    )

    with (
        patch(
            "repowise.docs.tools.handlers.get_remote_head_sha", new_callable=AsyncMock
        ) as mock_sha,
        patch("repowise.docs.tools.handlers.get_library", new_callable=AsyncMock) as mock_get,
        patch("repowise.docs.tools.handlers.upsert_library", new_callable=AsyncMock) as mock_upsert,
        patch("repowise.docs.tools.handlers.enqueue_job", new_callable=AsyncMock) as mock_enqueue,
    ):
        mock_sha.return_value = "abc123"
        mock_get.return_value = None
        mock_upsert.return_value = mock_lib
        mock_enqueue.return_value = JobEnqueueResult(
            disposition=JobEnqueueDisposition.QUEUED,
            job=IndexJob(
                id="job-123",
                library_id="/owner/repo",
                job_type=JobType.FULL,
                priority=3,
            ),
        )

        await handle_add_library({"repo": "owner/repo", "name": "Repo"})

    config = mock_upsert.call_args[0][0]
    required = {"**/*.md", "**/*.rst", "**/*.adoc", "**/*.java", "**/*.yaml"}
    assert required.issubset(set(config.include_patterns))
    assert set(DEFAULT_INCLUDE_PATTERNS).issubset(set(config.include_patterns))


@pytest.mark.unit
async def test_add_library_accepts_snake_case_include_and_exclude_patterns() -> None:
    mock_lib = LibraryState(
        library_id="/owner/repo",
        repo="owner/repo",
        name="Repo",
        status=LibraryStatus.PENDING,
        docs_path="",
        branch="main",
    )

    with (
        patch(
            "repowise.docs.tools.handlers.get_remote_head_sha", new_callable=AsyncMock
        ) as mock_sha,
        patch("repowise.docs.tools.handlers.get_library", new_callable=AsyncMock) as mock_get,
        patch("repowise.docs.tools.handlers.upsert_library", new_callable=AsyncMock) as mock_upsert,
        patch("repowise.docs.tools.handlers.enqueue_job", new_callable=AsyncMock) as mock_enqueue,
    ):
        mock_sha.return_value = "abc123"
        mock_get.return_value = None
        mock_upsert.return_value = mock_lib
        mock_enqueue.return_value = JobEnqueueResult(
            disposition=JobEnqueueDisposition.QUEUED,
            job=IndexJob(
                id="job-123",
                library_id="/owner/repo",
                job_type=JobType.FULL,
                priority=3,
            ),
        )

        await handle_add_library(
            {
                "repo": "owner/repo",
                "name": "Repo",
                "include_patterns": ["docs/**/*.mdx"],
                "exclude_patterns": ["docs/**/legacy/**"],
            }
        )

    config = mock_upsert.call_args[0][0]
    assert config.include_patterns == ["docs/**/*.mdx"]
    assert config.exclude_patterns == ["docs/**/legacy/**"]


@pytest.mark.unit
async def test_add_library_accepts_explicit_library_id_and_source_subpath() -> None:
    mock_lib = LibraryState(
        library_id="/codeatlas/cosmograph",
        repo="acme/docs-catalog",
        name="Cosmograph",
        description="GPU graph docs",
        source_subpath="cosmograph",
        status=LibraryStatus.PENDING,
        docs_path="",
        branch="main",
    )

    with (
        patch(
            "repowise.docs.tools.handlers.get_remote_head_sha", new_callable=AsyncMock
        ) as mock_sha,
        patch("repowise.docs.tools.handlers.get_library", new_callable=AsyncMock) as mock_get,
        patch("repowise.docs.tools.handlers.upsert_library", new_callable=AsyncMock) as mock_upsert,
        patch("repowise.docs.tools.handlers.enqueue_job", new_callable=AsyncMock) as mock_enqueue,
    ):
        mock_sha.return_value = "abc123"
        mock_get.return_value = None
        mock_upsert.return_value = mock_lib
        mock_enqueue.return_value = JobEnqueueResult(
            disposition=JobEnqueueDisposition.QUEUED,
            job=IndexJob(
                id="job-123",
                library_id="/codeatlas/cosmograph",
                job_type=JobType.FULL,
                priority=3,
            ),
        )

        result = await handle_add_library(
            {
                "repo": "acme/docs-catalog",
                "library_id": "/codeatlas/cosmograph",
                "source_subpath": "cosmograph",
                "name": "Cosmograph",
            }
        )

    config = mock_upsert.call_args[0][0]
    assert config.library_id == "/codeatlas/cosmograph"
    assert config.source_subpath == "cosmograph"
    assert result["library_id"] == "/codeatlas/cosmograph"
    assert result["source_subpath"] == "cosmograph"
