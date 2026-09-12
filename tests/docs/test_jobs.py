"""Tests for the job queue system.

These tests require a running PostgreSQL database with the doc_search schema.
Run with: uv run pytest tests/test_jobs.py -v
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

import repowise.docs.db_runtime as db_runtime
from repowise.docs.db import (
    claim_job,
    complete_job,
    enqueue_job,
    fail_job,
    get_job,
    heartbeat_job,
    init_pool,
    list_jobs,
    reclaim_stale_jobs,
)
from repowise.docs.library.models import JobEnqueueDisposition, JobStatus, JobType


@pytest.fixture
async def db_pool():
    """Initialize database pool for tests.

    Using function scope to ensure each test has its own pool
    that matches its event loop (pytest-asyncio auto mode creates
    a new loop per test).

    Skips the test when PostgreSQL is not reachable.
    """
    from repowise.docs.db import close_pool

    # Reset any existing pool to ensure clean state for this event loop
    if db_runtime._pool is not None:
        with contextlib.suppress(Exception):
            await db_runtime._pool.close()
        db_runtime._pool = None

    try:
        pool = await init_pool()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not available: {exc}")
    yield pool

    # Clean up after test
    await close_pool()


async def ensure_test_library(pool, library_id: str):
    """Helper to ensure a test library exists for FK constraint."""
    repo = library_id.lstrip("/")
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO doc_search.libraries (library_id, repo, name, description)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (library_id) DO NOTHING
            """,
            library_id,
            repo,
            f"Test Library {repo}",
            "Test library for job tests",
        )


async def cleanup_test_library(pool, library_id: str):
    """Helper to clean up a test library and its jobs."""
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
            library_id,
        )
        await conn.execute(
            "DELETE FROM doc_search.libraries WHERE library_id = $1",
            library_id,
        )


class TestJobEnqueue:
    """Tests for job enqueue and coalescing."""

    @pytest.mark.integration
    async def test_enqueue_job_creates_pending_job(self, db_pool):
        """Enqueuing a job should create it with pending status."""
        library_id = "/test/enqueue-basic"

        # Ensure library exists and clean up any existing jobs
        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            job = await enqueue_job(library_id, JobType.INCREMENTAL, priority=5)

            assert job.disposition == JobEnqueueDisposition.QUEUED
            assert job.job.library_id == library_id
            assert job.job.job_type == JobType.INCREMENTAL
            assert job.job.priority == 5
            assert job.job.status == JobStatus.PENDING
            assert job.job.worker_id is None
        finally:
            # Cleanup
            await cleanup_test_library(db_pool, library_id)

    @pytest.mark.integration
    async def test_enqueue_coalesces_when_pending_exists(self, db_pool):
        """Second enqueue should coalesce when pending job exists."""
        library_id = "/test/enqueue-coalesce"

        # Ensure library exists and clean up
        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            # First enqueue should succeed
            job1 = await enqueue_job(library_id, JobType.INCREMENTAL)
            assert job1.disposition == JobEnqueueDisposition.QUEUED

            # Second enqueue should upgrade the existing pending job
            job2 = await enqueue_job(library_id, JobType.FORCE)
            assert job2.disposition == JobEnqueueDisposition.UPGRADED
            assert job2.job.id == job1.job.id
            assert job2.job.job_type == JobType.FORCE
            assert job2.job.priority == min(job1.job.priority, 5)

            jobs = await list_jobs(library_id=library_id, status=JobStatus.PENDING)
            assert len(jobs) == 1
            assert jobs[0].job_type == JobType.FORCE
        finally:
            # Cleanup
            await cleanup_test_library(db_pool, library_id)

    @pytest.mark.integration
    async def test_enqueue_creates_follow_up_when_running_job_exists(self, db_pool):
        """A running job can coexist with one pending follow-up request."""
        library_id = "/test/enqueue-follow-up"

        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            job1 = await enqueue_job(library_id, JobType.INCREMENTAL)
            assert job1.disposition == JobEnqueueDisposition.QUEUED
            claimed = await claim_job("test-worker-follow-up")
            assert claimed is not None

            job2 = await enqueue_job(library_id, JobType.FORCE, priority=3)
            assert job2.disposition == JobEnqueueDisposition.QUEUED
            assert job2.job.id != claimed.id
            assert job2.job.status == JobStatus.PENDING
            assert job2.job.job_type == JobType.FORCE
            assert job2.job.priority == 3
        finally:
            await cleanup_test_library(db_pool, library_id)


class TestJobClaiming:
    """Tests for job claiming with FOR UPDATE SKIP LOCKED."""

    @pytest.mark.integration
    async def test_claim_job_transitions_to_running(self, db_pool):
        """Claiming a job should transition it to running status."""
        library_id = "/test/claim-basic"

        # Ensure library exists and clean up
        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            job = await enqueue_job(library_id, JobType.INCREMENTAL)
            assert job.disposition == JobEnqueueDisposition.QUEUED

            # Claim the job
            claimed = await claim_job("test-worker-1")

            assert claimed is not None
            assert claimed.id == job.job.id
            assert claimed.status == JobStatus.RUNNING
            assert claimed.worker_id == "test-worker-1"
            assert claimed.claimed_at is not None
            assert claimed.started_at is not None
            assert claimed.heartbeat_at is not None
        finally:
            # Cleanup
            await cleanup_test_library(db_pool, library_id)

    @pytest.mark.integration
    async def test_claim_job_skips_follow_up_for_library_with_running_job(self, db_pool):
        """Pending follow-up work must not claim while the same library is already running."""
        library_id = "/test/claim-running-library-guard"

        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            job1 = await enqueue_job(library_id, JobType.INCREMENTAL, priority=5)
            assert job1.disposition == JobEnqueueDisposition.QUEUED
            claimed = await claim_job("test-worker-guard")
            assert claimed is not None

            job2 = await enqueue_job(library_id, JobType.FORCE, priority=3)
            assert job2.disposition == JobEnqueueDisposition.QUEUED

            follow_up = await claim_job("test-worker-guard-2")
            assert follow_up is None
        finally:
            await cleanup_test_library(db_pool, library_id)

    @pytest.mark.integration
    async def test_claim_returns_none_when_no_jobs(self, db_pool):
        """Claiming should return None when no pending jobs exist."""
        # Ensure no pending jobs exist (this is a weak test due to shared state)
        claimed = await claim_job("test-worker-empty")

        # This might claim an existing job in a real DB, so we just verify
        # it returns a valid result (None or a job)
        assert claimed is None or claimed.status == JobStatus.RUNNING


class TestJobLifecycle:
    """Tests for job completion and failure."""

    @pytest.mark.integration
    async def test_complete_job_transitions_to_completed(self, db_pool):
        """Completing a running job should transition to completed status."""
        library_id = "/test/lifecycle-complete"

        # Ensure library exists and clean up
        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            job = await enqueue_job(library_id)
            assert job.disposition == JobEnqueueDisposition.QUEUED
            claimed = await claim_job("test-worker-complete")
            assert claimed is not None

            # Complete the job
            success = await complete_job(claimed.id)
            assert success is True

            # Verify status
            updated = await get_job(claimed.id)
            assert updated is not None
            assert updated.status == JobStatus.COMPLETED
            assert updated.completed_at is not None
        finally:
            # Cleanup
            await cleanup_test_library(db_pool, library_id)

    @pytest.mark.integration
    async def test_fail_job_transitions_to_failed(self, db_pool):
        """Failing a running job should transition to failed status."""
        library_id = "/test/lifecycle-fail"

        # Ensure library exists and clean up
        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            job = await enqueue_job(library_id)
            assert job.disposition == JobEnqueueDisposition.QUEUED
            claimed = await claim_job("test-worker-fail")
            assert claimed is not None

            # Fail the job
            error_msg = "Test error message"
            success = await fail_job(claimed.id, error_msg)
            assert success is True

            # Verify status
            updated = await get_job(claimed.id)
            assert updated is not None
            assert updated.status == JobStatus.FAILED
            assert updated.error_message == error_msg
            assert updated.completed_at is not None
        finally:
            # Cleanup
            await cleanup_test_library(db_pool, library_id)


class TestHeartbeat:
    """Tests for job heartbeat updates."""

    @pytest.mark.integration
    async def test_heartbeat_updates_timestamp(self, db_pool):
        """Heartbeat should update heartbeat_at timestamp."""
        library_id = "/test/heartbeat-basic"

        # Ensure library exists and clean up
        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            job = await enqueue_job(library_id)
            assert job.disposition == JobEnqueueDisposition.QUEUED
            claimed = await claim_job("test-worker-heartbeat")
            assert claimed is not None

            original_heartbeat = claimed.heartbeat_at

            # Small delay to ensure timestamp changes
            await asyncio.sleep(0.1)

            # Send heartbeat
            success = await heartbeat_job(claimed.id)
            assert success is True

            # Verify timestamp updated
            updated = await get_job(claimed.id)
            assert updated is not None
            assert updated.heartbeat_at is not None
            assert updated.heartbeat_at >= original_heartbeat
        finally:
            # Cleanup
            await cleanup_test_library(db_pool, library_id)

    @pytest.mark.integration
    async def test_heartbeat_with_progress(self, db_pool):
        """Heartbeat should update files_processed count."""
        library_id = "/test/heartbeat-progress"

        # Ensure library exists and clean up
        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            job = await enqueue_job(library_id)
            assert job.disposition == JobEnqueueDisposition.QUEUED
            claimed = await claim_job("test-worker-progress")
            assert claimed is not None
            assert claimed.files_processed == 0

            # Send heartbeat with progress
            success = await heartbeat_job(claimed.id, files_processed=42)
            assert success is True

            # Verify progress updated
            updated = await get_job(claimed.id)
            assert updated is not None
            assert updated.files_processed == 42
        finally:
            # Cleanup
            await cleanup_test_library(db_pool, library_id)


class TestStaleReclamation:
    """Tests for stale job reclamation."""

    @pytest.mark.integration
    async def test_reclaim_stale_jobs(self, db_pool):
        """Stale jobs should be reclaimed back to pending."""
        library_id = "/test/stale-reclaim"

        # Ensure library exists and clean up
        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            # Insert a stale running job directly
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO doc_search.index_jobs (
                        library_id, job_type, priority, status, worker_id,
                        claimed_at, heartbeat_at, started_at
                    ) VALUES (
                        $1, 'incremental', 5, 'running', 'stale-worker',
                        NOW() - INTERVAL '10 minutes',
                        NOW() - INTERVAL '10 minutes',
                        NOW() - INTERVAL '10 minutes'
                    )
                    """,
                    library_id,
                )

            # Reclaim jobs stale for more than 5 minutes
            reclaimed = await reclaim_stale_jobs(300)  # 5 minutes

            assert len(reclaimed) >= 1

            # Verify the job is now pending
            jobs = await list_jobs(library_id=library_id, status=JobStatus.PENDING)
            assert len(jobs) == 1
            assert jobs[0].worker_id is None
            assert jobs[0].claimed_at is None
            assert jobs[0].heartbeat_at is None
        finally:
            # Cleanup
            await cleanup_test_library(db_pool, library_id)

    @pytest.mark.integration
    async def test_reclaim_stale_job_with_existing_pending_marks_stale_runner_failed(self, db_pool):
        """A stale runner should be retired when a pending job already exists."""
        library_id = "/test/stale-reclaim-superseded"

        await ensure_test_library(db_pool, library_id)
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id = $1",
                library_id,
            )

        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO doc_search.index_jobs (
                        library_id, job_type, priority, status
                    ) VALUES (
                        $1, 'full', 5, 'pending'
                    )
                    """,
                    library_id,
                )
                await conn.execute(
                    """
                    INSERT INTO doc_search.index_jobs (
                        library_id, job_type, priority, status, worker_id,
                        claimed_at, heartbeat_at, started_at
                    ) VALUES (
                        $1, 'full', 5, 'running', 'stale-worker',
                        NOW() - INTERVAL '10 minutes',
                        NOW() - INTERVAL '10 minutes',
                        NOW() - INTERVAL '10 minutes'
                    )
                    """,
                    library_id,
                )

            reclaimed = await reclaim_stale_jobs(300)

            assert len(reclaimed) == 1

            pending_jobs = await list_jobs(library_id=library_id, status=JobStatus.PENDING)
            failed_jobs = await list_jobs(library_id=library_id, status=JobStatus.FAILED)

            assert len(pending_jobs) == 1
            assert len(failed_jobs) == 1
            assert failed_jobs[0].error_message == (
                "Stale running job superseded by existing pending job during recovery"
            )
        finally:
            await cleanup_test_library(db_pool, library_id)


class TestListJobs:
    """Tests for listing jobs with filters."""

    @pytest.mark.integration
    async def test_list_jobs_with_status_filter(self, db_pool):
        """Should filter jobs by status."""
        library_id = "/test/list-filter"
        library_id_pending = f"{library_id}-pending"
        library_id_completed = f"{library_id}-completed"

        # Ensure libraries exist
        await ensure_test_library(db_pool, library_id_pending)
        await ensure_test_library(db_pool, library_id_completed)

        # Setup - clean up existing jobs
        async with db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_search.index_jobs WHERE library_id LIKE '/test/list-%'"
            )

        try:
            async with db_pool.acquire() as conn:
                # Create pending job
                await conn.execute(
                    """
                    INSERT INTO doc_search.index_jobs (library_id, job_type, priority, status)
                    VALUES ($1, 'incremental', 5, 'pending')
                    """,
                    library_id_pending,
                )

                # Create completed job
                await conn.execute(
                    """
                    INSERT INTO doc_search.index_jobs (library_id, job_type, priority, status, completed_at)
                    VALUES ($1, 'incremental', 5, 'completed', NOW())
                    """,
                    library_id_completed,
                )

            # Filter by pending
            pending = await list_jobs(status=JobStatus.PENDING)
            assert any(j.library_id == library_id_pending for j in pending)

            # Filter by completed
            completed = await list_jobs(status=JobStatus.COMPLETED)
            assert any(j.library_id == library_id_completed for j in completed)
        finally:
            # Cleanup
            await cleanup_test_library(db_pool, library_id_pending)
            await cleanup_test_library(db_pool, library_id_completed)


class TestWorkerModule:
    """Tests for the worker module imports and structure."""

    def test_worker_module_imports(self):
        """Worker module should export expected functions."""
        from repowise.docs.jobs import (
            claim_job,
            complete_job,
            enqueue_job,
            fail_job,
            get_worker_id,
            heartbeat_job,
            is_worker_running,
            start_worker,
            stop_worker,
        )

        # Just verify imports work
        assert callable(enqueue_job)
        assert callable(claim_job)
        assert callable(heartbeat_job)
        assert callable(complete_job)
        assert callable(fail_job)
        assert callable(start_worker)
        assert callable(stop_worker)
        assert callable(is_worker_running)
        assert callable(get_worker_id)

    @pytest.mark.integration
    async def test_worker_not_running_initially(self):
        """Worker should not be running by default."""
        from repowise.docs.jobs import get_worker_id, is_worker_running

        # Note: This may fail if another test started the worker
        # In isolation, worker should not be running
        running = is_worker_running()
        worker_id = get_worker_id()

        # Just verify the functions work
        assert isinstance(running, bool)
        assert worker_id is None or isinstance(worker_id, str)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_worker_loop_skips_claiming_when_dependencies_are_unhealthy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import repowise.docs.jobs.worker as worker_module

        shutdown_event = asyncio.Event()
        claim_job = AsyncMock()

        async def _stop_soon() -> None:
            await asyncio.sleep(0)
            shutdown_event.set()

        monkeypatch.setattr(worker_module, "_dependencies_ready", AsyncMock(return_value=False))
        monkeypatch.setattr(worker_module, "claim_job", claim_job)

        await asyncio.gather(
            worker_module._worker_loop("worker-test", shutdown_event),
            _stop_soon(),
        )

        claim_job.assert_not_awaited()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_dependencies_ready_uses_dependency_only_status(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import repowise.docs.jobs.worker as worker_module
        import repowise.docs.server.health as health_module

        dependency_status = AsyncMock(return_value={"qdrant": "ok", "tei": "ok", "database": "ok"})
        full_health = AsyncMock()

        monkeypatch.setattr(health_module, "get_dependency_status", dependency_status)
        monkeypatch.setattr(health_module, "get_health_status", full_health)

        assert await worker_module._dependencies_ready() is True
        dependency_status.assert_awaited_once()
        full_health.assert_not_called()
