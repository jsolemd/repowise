"""Tests for doc-search indexing pipeline optimizations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from repowise.docs.library.models import LibraryFile, LibraryState, LibraryStatus
from repowise.docs.pipeline import index_library


def _build_library(
    *, current_sha: str | None, status: LibraryStatus = LibraryStatus.READY
) -> LibraryState:
    return LibraryState(
        library_id="/test/test",
        repo="test/test",
        name="Test Library",
        description="Test library",
        docs_path="docs",
        branch="main",
        status=status,
        current_sha=current_sha,
        chunk_count=9,
        file_count=1,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_index_library_skips_hash_scan_when_head_and_paths_unchanged(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    stored_file = LibraryFile(
        library_id="/test/test",
        file_path="docs/api.md",
        content_hash="abc123",
        chunk_count=3,
    )

    with (
        patch("repowise.docs.pipeline.get_qdrant_client", return_value=MagicMock()),
        patch("repowise.docs.pipeline.ensure_collection"),
        patch(
            "repowise.docs.pipeline.delete_files_outside_scope", new_callable=AsyncMock
        ) as mock_delete_outside_scope,
        patch("repowise.docs.pipeline.get_library", new_callable=AsyncMock) as mock_get_library,
        patch(
            "repowise.docs.pipeline.update_library_status", new_callable=AsyncMock
        ) as mock_update_status,
        patch("repowise.docs.pipeline.clone_repo", new_callable=AsyncMock) as mock_clone_repo,
        patch("repowise.docs.pipeline.get_head_sha", new_callable=AsyncMock) as mock_get_head_sha,
        patch(
            "repowise.docs.pipeline.get_latest_source_ref", new_callable=AsyncMock
        ) as mock_get_latest_source_ref,
        patch("repowise.docs.pipeline.list_doc_files", return_value=[Path("docs/api.md")]),
        patch(
            "repowise.docs.pipeline.get_library_files", new_callable=AsyncMock
        ) as mock_get_library_files,
        patch(
            "repowise.docs.pipeline.get_chunk_count", new_callable=AsyncMock
        ) as mock_get_chunk_count,
        patch(
            "repowise.docs.pipeline.compute_files_info",
            side_effect=AssertionError("unexpected hash scan"),
        ),
    ):
        mock_delete_outside_scope.return_value = {
            "files_removed": 0,
            "chunks_removed": 0,
            "paths": [],
        }
        mock_get_library.return_value = _build_library(current_sha="same-sha")
        mock_update_status.return_value = True
        mock_clone_repo.return_value = repo_dir
        mock_get_head_sha.return_value = "same-sha"
        mock_get_latest_source_ref.return_value = "same-sha"
        mock_get_library_files.return_value = [stored_file]
        mock_get_chunk_count.return_value = 9

        result = await index_library("/test/test")

    assert result["status"] == "unchanged"
    assert result["files_found"] == 1
    assert result["total_chunks"] == 9
    assert mock_get_library_files.await_count == 1
    assert mock_get_chunk_count.await_count == 1
    assert mock_update_status.await_count == 2
    ready_call = mock_update_status.await_args_list[-1]
    assert "indexed_at" not in ready_call.kwargs
    assert ready_call.kwargs["next_freshness_check_at"] is not None
    assert ready_call.kwargs["last_freshness_state"] == "fresh"
    assert ready_call.kwargs["last_remote_sha"] == "same-sha"
    assert ready_call.kwargs["last_freshness_error"] == ""
    mock_delete_outside_scope.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_index_library_hashes_when_head_matches_but_indexed_path_set_changed(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    with (
        patch("repowise.docs.pipeline.get_qdrant_client", return_value=MagicMock()),
        patch("repowise.docs.pipeline.ensure_collection"),
        patch(
            "repowise.docs.pipeline.delete_files_outside_scope", new_callable=AsyncMock
        ) as mock_delete_outside_scope,
        patch("repowise.docs.pipeline.get_library", new_callable=AsyncMock) as mock_get_library,
        patch(
            "repowise.docs.pipeline.update_library_status", new_callable=AsyncMock
        ) as mock_update_status,
        patch("repowise.docs.pipeline.clone_repo", new_callable=AsyncMock) as mock_clone_repo,
        patch("repowise.docs.pipeline.get_head_sha", new_callable=AsyncMock) as mock_get_head_sha,
        patch(
            "repowise.docs.pipeline.get_latest_source_ref", new_callable=AsyncMock
        ) as mock_get_latest_source_ref,
        patch("repowise.docs.pipeline.list_doc_files", return_value=[Path("docs/new.md")]),
        patch(
            "repowise.docs.pipeline.get_library_files", new_callable=AsyncMock
        ) as mock_get_library_files,
        patch(
            "repowise.docs.pipeline.get_chunk_count", new_callable=AsyncMock
        ) as mock_get_chunk_count,
        patch("repowise.docs.pipeline.compute_files_info") as mock_compute_files_info,
        patch("repowise.docs.pipeline.compute_changes") as mock_compute_changes,
    ):
        mock_delete_outside_scope.return_value = {
            "files_removed": 0,
            "chunks_removed": 0,
            "paths": [],
        }
        mock_get_library.return_value = _build_library(current_sha="same-sha")
        mock_update_status.return_value = True
        mock_clone_repo.return_value = repo_dir
        mock_get_head_sha.return_value = "same-sha"
        mock_get_latest_source_ref.return_value = "same-sha"
        mock_get_library_files.return_value = [
            LibraryFile(
                library_id="/test/test",
                file_path="docs/old.md",
                content_hash="abc123",
                chunk_count=3,
            )
        ]
        mock_compute_files_info.return_value = []
        mock_compute_changes.return_value.has_changes = False
        mock_compute_changes.return_value.added = set()
        mock_compute_changes.return_value.modified = set()
        mock_compute_changes.return_value.deleted = set()
        mock_get_chunk_count.return_value = 9

        result = await index_library("/test/test")

    assert result["status"] == "unchanged"
    assert mock_compute_files_info.call_count == 1
    assert mock_compute_changes.call_count == 1
    ready_call = mock_update_status.await_args_list[-1]
    assert "indexed_at" not in ready_call.kwargs
    assert ready_call.kwargs["next_freshness_check_at"] is not None
    assert ready_call.kwargs["last_freshness_state"] == "fresh"
    assert ready_call.kwargs["last_remote_sha"] == "same-sha"
    assert ready_call.kwargs["last_freshness_error"] == ""
    mock_delete_outside_scope.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_index_library_prunes_out_of_scope_qdrant_files_even_when_hash_scan_is_skipped(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    stored_file = LibraryFile(
        library_id="/test/test",
        file_path="docs/api.md",
        content_hash="abc123",
        chunk_count=3,
    )

    with (
        patch("repowise.docs.pipeline.get_qdrant_client", return_value=MagicMock()),
        patch("repowise.docs.pipeline.ensure_collection"),
        patch(
            "repowise.docs.pipeline.delete_files_outside_scope", new_callable=AsyncMock
        ) as mock_delete_outside_scope,
        patch("repowise.docs.pipeline.get_library", new_callable=AsyncMock) as mock_get_library,
        patch(
            "repowise.docs.pipeline.update_library_status", new_callable=AsyncMock
        ) as mock_update_status,
        patch("repowise.docs.pipeline.clone_repo", new_callable=AsyncMock) as mock_clone_repo,
        patch("repowise.docs.pipeline.get_head_sha", new_callable=AsyncMock) as mock_get_head_sha,
        patch(
            "repowise.docs.pipeline.get_latest_source_ref", new_callable=AsyncMock
        ) as mock_get_latest_source_ref,
        patch("repowise.docs.pipeline.list_doc_files", return_value=[Path("docs/api.md")]),
        patch(
            "repowise.docs.pipeline.get_library_files", new_callable=AsyncMock
        ) as mock_get_library_files,
        patch(
            "repowise.docs.pipeline.get_chunk_count", new_callable=AsyncMock
        ) as mock_get_chunk_count,
        patch(
            "repowise.docs.pipeline.compute_files_info",
            side_effect=AssertionError("unexpected hash scan"),
        ),
    ):
        mock_delete_outside_scope.return_value = {
            "files_removed": 1,
            "chunks_removed": 7,
            "paths": ["apps/mantine.dev/src/pages/core/textarea.mdx"],
        }
        mock_get_library.return_value = _build_library(current_sha="same-sha")
        mock_update_status.return_value = True
        mock_clone_repo.return_value = repo_dir
        mock_get_head_sha.return_value = "same-sha"
        mock_get_latest_source_ref.return_value = "same-sha"
        mock_get_library_files.return_value = [stored_file]
        mock_get_chunk_count.return_value = 3

        result = await index_library("/test/test")

    assert result["status"] == "unchanged"
    assert result["files_pruned"] == 1
    assert result["chunks_pruned"] == 7
    ready_call = mock_update_status.await_args_list[-1]
    assert "indexed_at" not in ready_call.kwargs
    assert ready_call.kwargs["last_freshness_state"] == "fresh"
    assert ready_call.kwargs["last_remote_sha"] == "same-sha"
    assert ready_call.kwargs["last_freshness_error"] == ""
    mock_delete_outside_scope.assert_awaited_once_with(
        "/test/test",
        {"docs/api.md"},
        ANY,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_index_library_deletes_only_stale_chunks_after_replacement_upsert(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "api.md").write_text("# API\nnew content\n", encoding="utf-8")

    fake_client = MagicMock()
    fake_chunk = MagicMock(id="chunk-new-1")
    change_set = MagicMock()
    change_set.added = set()
    change_set.modified = {"docs/api.md"}
    change_set.deleted = set()
    change_set.has_changes = True

    with (
        patch("repowise.docs.pipeline.get_qdrant_client", return_value=fake_client),
        patch("repowise.docs.pipeline.ensure_collection"),
        patch(
            "repowise.docs.pipeline.delete_files_outside_scope", new_callable=AsyncMock
        ) as mock_delete_outside_scope,
        patch("repowise.docs.pipeline.get_library", new_callable=AsyncMock) as mock_get_library,
        patch(
            "repowise.docs.pipeline.update_library_status", new_callable=AsyncMock
        ) as mock_update_status,
        patch("repowise.docs.pipeline.clone_repo", new_callable=AsyncMock) as mock_clone_repo,
        patch("repowise.docs.pipeline.get_head_sha", new_callable=AsyncMock) as mock_get_head_sha,
        patch(
            "repowise.docs.pipeline.get_latest_source_ref", new_callable=AsyncMock
        ) as mock_get_latest_source_ref,
        patch("repowise.docs.pipeline.list_doc_files", return_value=[Path("docs/api.md")]),
        patch("repowise.docs.pipeline.get_library_files", new_callable=AsyncMock, return_value=[]),
        patch(
            "repowise.docs.pipeline.compute_files_info",
            return_value=[MagicMock(path="docs/api.md", content_hash="hash-1")],
        ),
        patch("repowise.docs.pipeline.compute_changes", return_value=change_set),
        patch("repowise.docs.pipeline.chunk_file", return_value=[fake_chunk]),
        patch("repowise.docs.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert_chunks,
        patch(
            "repowise.docs.pipeline.delete_stale_file_chunks", new_callable=AsyncMock
        ) as mock_delete_stale,
        patch(
            "repowise.docs.pipeline.delete_by_file", new_callable=AsyncMock
        ) as mock_delete_by_file,
        patch("repowise.docs.pipeline.upsert_library_file", new_callable=AsyncMock),
        patch("repowise.docs.pipeline.get_chunk_count", new_callable=AsyncMock, return_value=1),
    ):
        mock_delete_outside_scope.return_value = {
            "files_removed": 0,
            "chunks_removed": 0,
            "paths": [],
        }
        mock_get_library.return_value = _build_library(current_sha="old-sha")
        mock_update_status.side_effect = [True, _build_library(current_sha="new-sha")]
        mock_clone_repo.return_value = repo_dir
        mock_get_head_sha.return_value = "new-sha"
        mock_get_latest_source_ref.return_value = "new-sha"

        result = await index_library("/test/test")

    assert result["status"] == "indexed"
    mock_upsert_chunks.assert_awaited_once()
    mock_delete_stale.assert_awaited_once_with(
        "/test/test", "docs/api.md", ["chunk-new-1"], fake_client
    )
    mock_delete_by_file.assert_not_awaited()
    ready_call = mock_update_status.await_args_list[-1]
    assert ready_call.kwargs["last_freshness_state"] == "fresh"
    assert ready_call.kwargs["last_remote_sha"] == "new-sha"
    assert ready_call.kwargs["last_freshness_error"] == ""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_index_library_clears_file_when_replacement_produces_no_chunks(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "api.md").write_text("", encoding="utf-8")

    change_set = MagicMock()
    change_set.added = set()
    change_set.modified = {"docs/api.md"}
    change_set.deleted = set()
    change_set.has_changes = True

    with (
        patch("repowise.docs.pipeline.get_qdrant_client", return_value=MagicMock()),
        patch("repowise.docs.pipeline.ensure_collection"),
        patch(
            "repowise.docs.pipeline.delete_files_outside_scope", new_callable=AsyncMock
        ) as mock_delete_outside_scope,
        patch("repowise.docs.pipeline.get_library", new_callable=AsyncMock) as mock_get_library,
        patch(
            "repowise.docs.pipeline.update_library_status", new_callable=AsyncMock
        ) as mock_update_status,
        patch("repowise.docs.pipeline.clone_repo", new_callable=AsyncMock) as mock_clone_repo,
        patch("repowise.docs.pipeline.get_head_sha", new_callable=AsyncMock) as mock_get_head_sha,
        patch(
            "repowise.docs.pipeline.get_latest_source_ref", new_callable=AsyncMock
        ) as mock_get_latest_source_ref,
        patch("repowise.docs.pipeline.list_doc_files", return_value=[Path("docs/api.md")]),
        patch("repowise.docs.pipeline.get_library_files", new_callable=AsyncMock, return_value=[]),
        patch(
            "repowise.docs.pipeline.compute_files_info",
            return_value=[MagicMock(path="docs/api.md", content_hash="hash-1")],
        ),
        patch("repowise.docs.pipeline.compute_changes", return_value=change_set),
        patch("repowise.docs.pipeline.chunk_file", return_value=[]),
        patch("repowise.docs.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert_chunks,
        patch(
            "repowise.docs.pipeline.delete_stale_file_chunks", new_callable=AsyncMock
        ) as mock_delete_stale,
        patch(
            "repowise.docs.pipeline.delete_by_file", new_callable=AsyncMock
        ) as mock_delete_by_file,
        patch("repowise.docs.pipeline.upsert_library_file", new_callable=AsyncMock),
        patch("repowise.docs.pipeline.get_chunk_count", new_callable=AsyncMock, return_value=0),
    ):
        mock_delete_outside_scope.return_value = {
            "files_removed": 0,
            "chunks_removed": 0,
            "paths": [],
        }
        mock_get_library.return_value = _build_library(current_sha="old-sha")
        mock_update_status.side_effect = [True, _build_library(current_sha="same-sha")]
        mock_clone_repo.return_value = repo_dir
        mock_get_head_sha.return_value = "same-sha"
        mock_get_latest_source_ref.return_value = "same-sha"

        result = await index_library("/test/test")

    assert result["status"] == "indexed"
    mock_upsert_chunks.assert_not_awaited()
    mock_delete_stale.assert_not_awaited()
    mock_delete_by_file.assert_awaited_once()
    ready_call = mock_update_status.await_args_list[-1]
    assert ready_call.kwargs["last_freshness_state"] == "fresh"
    assert ready_call.kwargs["last_remote_sha"] == "same-sha"
    assert ready_call.kwargs["last_freshness_error"] == ""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_index_library_raises_without_deleting_existing_chunks_when_file_processing_fails(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "guide.md").write_text("# Guide\n", encoding="utf-8")

    changes = MagicMock()
    changes.added = set()
    changes.modified = {Path("docs/guide.md")}
    changes.deleted = set()
    changes.has_changes = True

    with (
        patch("repowise.docs.pipeline.get_qdrant_client", return_value=MagicMock()),
        patch("repowise.docs.pipeline.ensure_collection"),
        patch(
            "repowise.docs.pipeline.delete_files_outside_scope", new_callable=AsyncMock
        ) as mock_delete_outside_scope,
        patch("repowise.docs.pipeline.get_library", new_callable=AsyncMock) as mock_get_library,
        patch(
            "repowise.docs.pipeline.update_library_status", new_callable=AsyncMock
        ) as mock_update_status,
        patch("repowise.docs.pipeline.clone_repo", new_callable=AsyncMock) as mock_clone_repo,
        patch("repowise.docs.pipeline.get_head_sha", new_callable=AsyncMock) as mock_get_head_sha,
        patch(
            "repowise.docs.pipeline.get_latest_source_ref", new_callable=AsyncMock
        ) as mock_get_latest_source_ref,
        patch("repowise.docs.pipeline.list_doc_files", return_value=[Path("docs/guide.md")]),
        patch("repowise.docs.pipeline.compute_files_info", return_value=[]),
        patch("repowise.docs.pipeline.compute_changes", return_value=changes),
        patch("repowise.docs.pipeline.get_library_files", new_callable=AsyncMock, return_value=[]),
        patch("repowise.docs.pipeline.chunk_file", side_effect=RuntimeError("boom")),
        patch("repowise.docs.pipeline.upsert_chunks", new_callable=AsyncMock),
        patch("repowise.docs.pipeline.upsert_library_file", new_callable=AsyncMock),
        patch("repowise.docs.pipeline.delete_library_file", new_callable=AsyncMock),
        patch(
            "repowise.docs.pipeline.delete_by_file", new_callable=AsyncMock
        ) as mock_delete_by_file,
    ):
        mock_delete_outside_scope.return_value = {
            "files_removed": 0,
            "chunks_removed": 0,
            "paths": [],
        }
        mock_get_library.return_value = _build_library(current_sha="old-sha")
        mock_update_status.return_value = True
        mock_clone_repo.return_value = repo_dir
        mock_get_head_sha.return_value = "new-sha"
        mock_get_latest_source_ref.return_value = "new-sha"

        with pytest.raises(RuntimeError, match=r"Indexing failed for 1 file\(s\): docs/guide\.md"):
            await index_library("/test/test")

    mock_delete_by_file.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_index_library_marks_library_ready_with_warning_when_some_files_fail(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "repo"
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (docs_dir / "broken.md").write_text("# Broken\n", encoding="utf-8")

    changes = MagicMock()
    changes.added = {"docs/guide.md", "docs/broken.md"}
    changes.modified = set()
    changes.deleted = set()
    changes.has_changes = True

    fake_chunk = MagicMock(id="chunk-ok-1")

    def _chunk_file(path: str, *_args, **_kwargs):
        if path.endswith("broken.md"):
            raise RuntimeError("boom")
        return [fake_chunk]

    with (
        patch("repowise.docs.pipeline.get_qdrant_client", return_value=MagicMock()),
        patch("repowise.docs.pipeline.ensure_collection"),
        patch(
            "repowise.docs.pipeline.delete_files_outside_scope", new_callable=AsyncMock
        ) as mock_delete_outside_scope,
        patch("repowise.docs.pipeline.get_library", new_callable=AsyncMock) as mock_get_library,
        patch(
            "repowise.docs.pipeline.update_library_status", new_callable=AsyncMock
        ) as mock_update_status,
        patch("repowise.docs.pipeline.clone_repo", new_callable=AsyncMock) as mock_clone_repo,
        patch("repowise.docs.pipeline.get_head_sha", new_callable=AsyncMock) as mock_get_head_sha,
        patch(
            "repowise.docs.pipeline.get_latest_source_ref", new_callable=AsyncMock
        ) as mock_get_latest_source_ref,
        patch(
            "repowise.docs.pipeline.list_doc_files",
            return_value=[Path("docs/guide.md"), Path("docs/broken.md")],
        ),
        patch(
            "repowise.docs.pipeline.compute_files_info",
            return_value=[
                MagicMock(path="docs/guide.md", content_hash="hash-guide"),
                MagicMock(path="docs/broken.md", content_hash="hash-broken"),
            ],
        ),
        patch("repowise.docs.pipeline.compute_changes", return_value=changes),
        patch("repowise.docs.pipeline.get_library_files", new_callable=AsyncMock, return_value=[]),
        patch("repowise.docs.pipeline.chunk_file", side_effect=_chunk_file),
        patch("repowise.docs.pipeline.upsert_chunks", new_callable=AsyncMock),
        patch("repowise.docs.pipeline.upsert_library_file", new_callable=AsyncMock),
        patch("repowise.docs.pipeline.delete_by_file", new_callable=AsyncMock),
        patch("repowise.docs.pipeline.delete_stale_file_chunks", new_callable=AsyncMock),
        patch("repowise.docs.pipeline.get_chunk_count", new_callable=AsyncMock, return_value=1),
    ):
        mock_delete_outside_scope.return_value = {
            "files_removed": 0,
            "chunks_removed": 0,
            "paths": [],
        }
        mock_get_library.return_value = _build_library(current_sha="old-sha")
        mock_update_status.return_value = True
        mock_clone_repo.return_value = repo_dir
        mock_get_head_sha.return_value = "new-sha"
        mock_get_latest_source_ref.return_value = "new-sha"

        result = await index_library("/test/test")

    assert result["status"] == "indexed"
    assert result["errors"] == ["docs/broken.md"]
    assert "warning" in result
    ready_call = mock_update_status.await_args_list[-1]
    assert ready_call.kwargs["status"] == LibraryStatus.READY
    assert ready_call.kwargs["last_freshness_state"] == "fresh"
    assert ready_call.kwargs["last_remote_sha"] == "new-sha"
    assert ready_call.kwargs["last_freshness_error"] == ""
    assert "Skipped 1 file(s)" in ready_call.kwargs["error_message"]
