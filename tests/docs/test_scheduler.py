"""Tests for the freshness scheduler.

Run with: uv run pytest tests/test_scheduler.py -v
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from repowise.docs.jobs.scheduler import (
    FreshnessResult,
    FreshnessState,
    _freshness_cache,
    check_library_freshness,
    clear_freshness_cache,
    get_scheduler_status,
    run_freshness_check,
)
from repowise.docs.library.models import LibrarySourceType, LibraryState, LibraryStatus
from repowise.docs.snapshot_publish import SnapshotPublishResult


def _ready_library(**overrides) -> LibraryState:
    now = datetime.now(UTC)
    base = {
        "library_id": "/test/test",
        "repo": "test/test",
        "name": "Test Library",
        "description": "test",
        "docs_path": "docs",
        "branch": "main",
        "status": LibraryStatus.READY,
        "current_sha": "abc123",
        "indexed_at": now,
        "freshness_checked_at": None,
        "chunk_count": 10,
        "file_count": 3,
        "created_at": now,
        "updated_at": now,
    }
    base.update(overrides)
    return LibraryState(**base)


class TestFreshnessCache:
    """Test freshness cache behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_freshness_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_freshness_cache()

    @pytest.mark.unit
    async def test_cache_miss_checks_remote_source_ref(self):
        """Cache miss should check the upstream source ref."""
        with patch(
            "repowise.docs.jobs.scheduler._get_latest_source_ref",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = "abc123"

            result = await check_library_freshness(_ready_library(current_sha="abc123"))

            assert result == FreshnessState.FRESH
            mock_get.assert_awaited_once()

    @pytest.mark.unit
    async def test_cache_hit_skips_remote_source_probe(self):
        """Cache hit should skip the upstream source probe."""
        # Pre-populate cache
        _freshness_cache["/test/test"] = FreshnessResult(
            state=FreshnessState.FRESH,
            checked_at=datetime.now(UTC),
        )

        with patch(
            "repowise.docs.jobs.scheduler._get_latest_source_ref",
            new_callable=AsyncMock,
        ) as mock_get:
            result = await check_library_freshness(_ready_library(current_sha="abc123"))

            assert result == FreshnessState.FRESH
            mock_get.assert_not_called()  # Should use cache

    @pytest.mark.unit
    async def test_bypass_cache_calls_remote_source_probe(self):
        """use_cache=False should bypass cache."""
        # Pre-populate cache
        _freshness_cache["/test/test"] = FreshnessResult(
            state=FreshnessState.FRESH,
            checked_at=datetime.now(UTC),
        )

        with patch(
            "repowise.docs.jobs.scheduler._get_latest_source_ref",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = "new123"

            result = await check_library_freshness(
                _ready_library(current_sha="old456"),
                use_cache=False,
            )

            assert result == FreshnessState.STALE
            mock_get.assert_called_once()

    @pytest.mark.unit
    async def test_clear_cache_single_library(self):
        """clear_freshness_cache can clear single library."""
        _freshness_cache["/a/a"] = FreshnessResult(FreshnessState.FRESH, datetime.now(UTC))
        _freshness_cache["/b/b"] = FreshnessResult(FreshnessState.STALE, datetime.now(UTC))

        clear_freshness_cache("/a/a")

        assert "/a/a" not in _freshness_cache
        assert "/b/b" in _freshness_cache

    @pytest.mark.unit
    async def test_clear_cache_all(self):
        """clear_freshness_cache(None) clears all."""
        _freshness_cache["/a/a"] = FreshnessResult(FreshnessState.FRESH, datetime.now(UTC))
        _freshness_cache["/b/b"] = FreshnessResult(FreshnessState.STALE, datetime.now(UTC))

        clear_freshness_cache()

        assert len(_freshness_cache) == 0


class TestFreshnessLogic:
    """Test staleness detection logic."""

    def setup_method(self):
        clear_freshness_cache()

    def teardown_method(self):
        clear_freshness_cache()

    @pytest.mark.unit
    async def test_no_current_sha_is_stale(self):
        """Library with no indexed SHA is stale."""
        result = await check_library_freshness(
            _ready_library(current_sha=None),
        )

        assert result == FreshnessState.STALE

    @pytest.mark.unit
    async def test_same_sha_is_fresh(self):
        """Same SHA = fresh."""
        with patch(
            "repowise.docs.jobs.scheduler._get_latest_source_ref",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = "abc123def456"

            result = await check_library_freshness(_ready_library(current_sha="abc123def456"))

            assert result == FreshnessState.FRESH

    @pytest.mark.unit
    async def test_snapshot_source_ref_prefix_is_fresh(self):
        """Snapshot current_sha values embed the manifest hash after the live source ref."""
        with patch(
            "repowise.docs.jobs.scheduler._get_latest_source_ref",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = "build-123"

            result = await check_library_freshness(
                _ready_library(
                    library_id="/codeatlas/cosmograph",
                    repo="",
                    source_type=LibrarySourceType.SNAPSHOT,
                    current_sha="build-123:manifest-456",
                )
            )

            assert result == FreshnessState.FRESH

    @pytest.mark.unit
    async def test_different_sha_is_stale(self):
        """Different SHA = stale."""
        with patch(
            "repowise.docs.jobs.scheduler._get_latest_source_ref",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = "new123456789"

            result = await check_library_freshness(_ready_library(current_sha="old987654321"))

            assert result == FreshnessState.STALE

    @pytest.mark.unit
    async def test_remote_probe_error_returns_unknown(self):
        """If the remote source probe fails, mark freshness as unknown instead of fresh."""
        with patch(
            "repowise.docs.jobs.scheduler._get_latest_source_ref",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = None  # API failed

            result = await check_library_freshness(_ready_library(current_sha="abc123"))

            assert result == FreshnessState.UNKNOWN


def test_scheduler_status_defaults_are_empty() -> None:
    status = get_scheduler_status()

    assert status["running"] is False
    assert status["last_started_at"] is None
    assert status["last_completed_at"] is None
    assert status["summary"] == {
        "checked": 0,
        "queued": 0,
        "ready_libraries": 0,
        "skipped": 0,
        "stale_error_retries": 0,
        "transient_retries": 0,
        "unknown": 0,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_freshness_check_skips_recently_indexed_library_on_restart(monkeypatch) -> None:
    lib = _ready_library(indexed_at=datetime.now(UTC))

    list_libraries = AsyncMock(side_effect=[[lib], []])
    freshness_check = AsyncMock()
    enqueue_job = AsyncMock()
    update_library_status = AsyncMock()

    monkeypatch.setattr("repowise.docs.jobs.scheduler.list_libraries", list_libraries)
    monkeypatch.setattr(
        "repowise.docs.jobs.scheduler.check_library_freshness_details", freshness_check
    )
    monkeypatch.setattr("repowise.docs.jobs.scheduler.enqueue_job", enqueue_job)
    monkeypatch.setattr("repowise.docs.jobs.scheduler.update_library_status", update_library_status)

    queued = await run_freshness_check()

    assert queued == 0
    freshness_check.assert_not_awaited()
    enqueue_job.assert_not_awaited()
    update_library_status.assert_awaited_once()
    kwargs = update_library_status.await_args.kwargs
    assert kwargs["next_freshness_check_at"].tzinfo is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_freshness_check_persists_checked_at_after_probe(monkeypatch) -> None:
    stale_checked_at = datetime.now(UTC).replace(year=2025)
    next_due_at = datetime.now(UTC).replace(year=2025, month=12)
    lib = _ready_library(
        indexed_at=stale_checked_at,
        freshness_checked_at=stale_checked_at,
        next_freshness_check_at=next_due_at,
    )

    list_libraries = AsyncMock(side_effect=[[lib], []])
    update_library_status = AsyncMock()

    monkeypatch.setattr("repowise.docs.jobs.scheduler.list_libraries", list_libraries)
    monkeypatch.setattr(
        "repowise.docs.jobs.scheduler.check_library_freshness_details",
        AsyncMock(
            return_value=FreshnessResult(
                state=FreshnessState.FRESH,
                checked_at=datetime.now(UTC),
                remote_sha="abc123",
            )
        ),
    )
    monkeypatch.setattr("repowise.docs.jobs.scheduler.enqueue_job", AsyncMock())
    monkeypatch.setattr("repowise.docs.jobs.scheduler.update_library_status", update_library_status)

    queued = await run_freshness_check()

    assert queued == 0
    update_library_status.assert_awaited_once()
    kwargs = update_library_status.await_args.kwargs
    assert kwargs["freshness_checked_at"].tzinfo is not None
    assert kwargs["next_freshness_check_at"].tzinfo is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_freshness_check_refreshes_stale_snapshot_library(monkeypatch) -> None:
    checked_at = datetime.now(UTC).replace(year=2025)
    lib = _ready_library(
        library_id="/codeatlas/cosmograph",
        repo="",
        source_type=LibrarySourceType.SNAPSHOT,
        current_sha="build-123:manifest-456",
        indexed_at=checked_at,
        freshness_checked_at=checked_at,
        next_freshness_check_at=checked_at,
    )

    update_library_status = AsyncMock()
    enqueue_job = AsyncMock()
    refresh_snapshot_library = AsyncMock(
        return_value=SnapshotPublishResult(
            library_id=lib.library_id,
            changed=True,
            source_ref="build-999:manifest-abc",
            manifest_hash="manifest-abc",
            file_count=72,
            queued_job_type="incremental",
            queue_disposition="queued",
        )
    )

    monkeypatch.setattr(
        "repowise.docs.jobs.scheduler.list_libraries",
        AsyncMock(side_effect=[[lib], []]),
    )
    monkeypatch.setattr(
        "repowise.docs.jobs.scheduler.check_library_freshness_details",
        AsyncMock(
            return_value=FreshnessResult(
                state=FreshnessState.STALE,
                checked_at=datetime.now(UTC),
                remote_sha="build-999",
            )
        ),
    )
    monkeypatch.setattr(
        "repowise.docs.jobs.scheduler._supports_snapshot_automation",
        lambda library_id: library_id == lib.library_id,
    )
    monkeypatch.setattr(
        "repowise.docs.jobs.scheduler._refresh_snapshot_library",
        refresh_snapshot_library,
    )
    monkeypatch.setattr("repowise.docs.jobs.scheduler.enqueue_job", enqueue_job)
    monkeypatch.setattr("repowise.docs.jobs.scheduler.update_library_status", update_library_status)

    queued = await run_freshness_check()

    assert queued == 1
    refresh_snapshot_library.assert_awaited_once_with(lib.library_id)
    enqueue_job.assert_not_awaited()
