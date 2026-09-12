"""Tests for snapshot-backed docs runtime automation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from repowise.docs.library.models import LibrarySourceType, LibraryState, LibraryStatus
from repowise.docs.scrapers import snapshot_runtime
from repowise.docs.scrapers.snapshot_runtime import SnapshotSourceHandler
from repowise.docs.snapshot_publish import SnapshotPublishResult


def _snapshot_library(library_id: str, **overrides) -> LibraryState:
    now = datetime.now(UTC)
    data = {
        "library_id": library_id,
        "repo": "",
        "name": library_id.rsplit("/", 1)[-1],
        "description": "snapshot docs",
        "docs_path": "",
        "branch": "main",
        "source_type": LibrarySourceType.SNAPSHOT,
        "status": LibraryStatus.READY,
        "current_sha": None,
        "indexed_at": now,
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return LibraryState(**data)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_refresh_snapshot_library_cleans_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    library_id = "/codeatlas/test-snapshot"
    observed: dict[str, object] = {}

    async def fake_fetch_into(*, output_dir: Path, persist_freshness: bool = False) -> str:
        assert persist_freshness is False
        observed["output_dir"] = output_dir
        (output_dir / "index.md").write_text("# Snapshot\n", encoding="utf-8")
        return "build-123"

    async def fake_publish_snapshot_directory(
        target_library_id: str,
        output_dir: Path,
        *,
        source_ref: str | None = None,
        queue_reindex: bool = True,
    ) -> SnapshotPublishResult:
        observed["publish_dir_exists"] = output_dir.exists()
        observed["published_paths"] = sorted(
            path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*.md")
        )
        assert queue_reindex is True
        return SnapshotPublishResult(
            library_id=target_library_id,
            changed=True,
            source_ref=f"{source_ref}:manifest-456",
            manifest_hash="manifest-456",
            file_count=1,
            queued_job_type="incremental",
            queue_disposition="queued",
        )

    monkeypatch.setitem(
        snapshot_runtime.SNAPSHOT_SOURCE_HANDLERS,
        library_id,
        SnapshotSourceHandler(
            library_id=library_id,
            probe_source_ref=lambda: "build-123",
            fetch_into=fake_fetch_into,
        ),
    )
    monkeypatch.setattr(
        snapshot_runtime,
        "publish_snapshot_directory",
        fake_publish_snapshot_directory,
    )

    result = await snapshot_runtime.refresh_snapshot_library(library_id)

    assert result.source_ref == "build-123:manifest-456"
    assert observed["publish_dir_exists"] is True
    assert observed["published_paths"] == ["index.md"]
    assert not Path(observed["output_dir"]).exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_bootstrap_missing_snapshot_libraries_refreshes_only_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_library = _snapshot_library("/codeatlas/missing")
    existing_library = _snapshot_library("/codeatlas/existing")
    failing_library = _snapshot_library("/codeatlas/failing")
    git_library = _snapshot_library(
        "/git/ignored",
        source_type=LibrarySourceType.GIT,
        repo="owner/repo",
    )

    async def fake_refresh_snapshot_library(library_id: str) -> SnapshotPublishResult:
        if library_id == failing_library.library_id:
            raise RuntimeError("fetch failed")
        return SnapshotPublishResult(
            library_id=library_id,
            changed=True,
            source_ref="build-123:manifest-456",
            manifest_hash="manifest-456",
            file_count=3,
            queued_job_type="full",
            queue_disposition="queued",
        )

    monkeypatch.setattr(
        snapshot_runtime,
        "supports_snapshot_automation",
        lambda library_id: library_id.startswith("/codeatlas/"),
    )
    monkeypatch.setattr(
        snapshot_runtime.db,
        "get_snapshot_state",
        AsyncMock(
            side_effect=lambda library_id: (
                object() if library_id == existing_library.library_id else None
            )
        ),
    )
    monkeypatch.setattr(
        snapshot_runtime,
        "refresh_snapshot_library",
        AsyncMock(side_effect=fake_refresh_snapshot_library),
    )
    update_library_status = AsyncMock()
    monkeypatch.setattr(snapshot_runtime.db, "update_library_status", update_library_status)

    summary = await snapshot_runtime.bootstrap_missing_snapshot_libraries(
        [missing_library, existing_library, failing_library, git_library]
    )

    assert summary == {
        "configured": 3,
        "attempted": 2,
        "published": 1,
        "failed": 1,
        "published_libraries": [missing_library.library_id],
        "failed_libraries": {failing_library.library_id: "fetch failed"},
    }
    update_library_status.assert_awaited_once_with(
        failing_library.library_id,
        status=LibraryStatus.ERROR,
        error_message="fetch failed",
        last_freshness_state="unknown",
        last_freshness_error="fetch failed",
    )
