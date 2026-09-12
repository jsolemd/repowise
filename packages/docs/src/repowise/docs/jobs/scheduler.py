"""Background scheduler for automatic documentation freshness checks.

This module provides a scheduler that periodically checks if indexed libraries
have new upstream commits and queues reindex jobs when needed.

## How It Works

```
Server Startup
      │
      └── start_scheduler()
              │
              └── Loop every BASE_CHECK_INTERVAL (default: 1 hour)
                      │
                      ├── For each "ready" library:
                      │     ├── Check if enough time has passed based on priority
                      │     │   (high=1h, medium=6h, low=24h)
                      │     ├── Get current_sha from database
                      │     ├── Get latest source ref from git or snapshot probe
                      │     └── If different → enqueue incremental job
                      │
                      └── Sleep BASE_CHECK_INTERVAL
```

## Priority-Based Check Intervals

Libraries are checked at different frequencies based on their priority:
- Priority 1-3 (high): Check every 1 hour
- Priority 4-6 (medium): Check every 6 hours (default behavior)
- Priority 7-10 (low): Check every 24 hours

## Configuration

- `DOC_SEARCH_FRESHNESS_CHECK_INTERVAL`: Base interval in seconds (default: 3600 = 1 hour)

## Usage

The scheduler is automatically started in the HTTP server lifespan.
You don't need to call it directly.
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from repowise.docs.config import get_settings
from repowise.docs.db import enqueue_job, list_libraries, update_library_status
from repowise.docs.jobs.policy import (
    DEFAULT_CHECK_INTERVAL,
    STALE_ERROR_RETRY_THRESHOLD,
    bootstrap_next_freshness_check_at,
    compute_next_freshness_check_at,
    stale_error_retry_interval_seconds,
    transient_error_retry_interval_seconds,
)
from repowise.docs.library.git_manager import (
    GitError,
    _is_permanent_error,
    _is_transient_error,
    clone_repo,
    get_source_ref,
)
from repowise.docs.library.models import (
    JobEnqueueDisposition,
    JobType,
    LibrarySourceType,
    LibraryState,
    LibraryStatus,
)

logger = logging.getLogger(__name__)

# Cache TTL for freshness checks (5 minutes)
# Avoids hitting upstream sources on every query
FRESHNESS_CACHE_TTL = 5 * 60  # 300 seconds


def _effective_last_check_time(lib: LibraryState) -> datetime | None:
    """Return the persisted scheduler cadence anchor for a library."""
    return lib.freshness_checked_at or lib.indexed_at


def _scheduled_check_at(lib: LibraryState, *, now: datetime | None = None) -> datetime:
    """Return the persisted or bootstrapped next freshness probe time."""
    if lib.next_freshness_check_at is not None:
        return lib.next_freshness_check_at
    return bootstrap_next_freshness_check_at(
        lib.library_id,
        lib.priority,
        _effective_last_check_time(lib),
        now=now,
    )


def _transient_retry_due_at(lib: LibraryState, *, now: datetime | None = None) -> datetime:
    """Return the persisted or bootstrapped next retry window for transient errors."""
    if lib.next_freshness_check_at is not None:
        return lib.next_freshness_check_at
    baseline = lib.freshness_checked_at or lib.updated_at or lib.indexed_at
    return bootstrap_next_freshness_check_at(
        lib.library_id,
        lib.priority,
        baseline,
        now=now,
        interval_seconds=transient_error_retry_interval_seconds(lib.priority),
    )


# Global state for scheduler control
_scheduler_task: asyncio.Task | None = None
_stop_event: asyncio.Event | None = None


class FreshnessResult(NamedTuple):
    """Cached freshness check result."""

    state: "FreshnessState"
    checked_at: datetime
    remote_sha: str | None = None
    error: str | None = None


class RepoSnapshot(NamedTuple):
    """Prepared source repository snapshot for a scheduler cycle."""

    repo_dir: Path | None
    error: str | None = None


class FreshnessState(StrEnum):
    """Outcome of a freshness check."""

    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


# In-memory cache: library_id -> FreshnessResult
# Protected by _freshness_cache_lock for atomic updates
_freshness_cache: dict[str, FreshnessResult] = {}
_freshness_cache_lock: asyncio.Lock | None = None
_last_scheduler_run_started_at: datetime | None = None
_last_scheduler_run_completed_at: datetime | None = None
_last_scheduler_summary: dict[str, int] = {
    "ready_libraries": 0,
    "checked": 0,
    "skipped": 0,
    "queued": 0,
    "unknown": 0,
    "transient_retries": 0,
    "stale_error_retries": 0,
}

# Track consecutive stale-error retries per library for exponential backoff.
_stale_error_retry_counts: dict[str, int] = {}

# Track consecutive check failures per library.
# When get_remote_head_sha returns None, we increment the counter.
# On success, we reset it. After CHECK_FAILURE_WARN_THRESHOLD consecutive
# failures, we log a WARNING each cycle so it doesn't go unnoticed.
CHECK_FAILURE_WARN_THRESHOLD = 3
_check_failure_counts: dict[str, int] = {}


def _get_cache_lock() -> asyncio.Lock:
    """Get or create the cache lock (lazy init for event loop safety)."""
    global _freshness_cache_lock
    if _freshness_cache_lock is None:
        _freshness_cache_lock = asyncio.Lock()
    return _freshness_cache_lock


def _source_refs_match(library: LibraryState, remote_sha: str) -> bool:
    """Compare upstream source refs against the stored current_sha format."""
    current_sha = (library.current_sha or "").strip()
    normalized_remote = remote_sha.strip()
    if not current_sha or not normalized_remote:
        return False
    if library.source_type != LibrarySourceType.SNAPSHOT:
        return current_sha == normalized_remote
    return current_sha == normalized_remote or current_sha.startswith(f"{normalized_remote}:")


def _supports_snapshot_automation(library_id: str) -> bool:
    from repowise.docs.scrapers.snapshot_runtime import supports_snapshot_automation

    return supports_snapshot_automation(library_id)


async def _get_snapshot_source_ref(library_id: str) -> str | None:
    from repowise.docs.scrapers.snapshot_runtime import get_snapshot_source_ref

    return await get_snapshot_source_ref(library_id)


async def _refresh_snapshot_library(library_id: str):
    from repowise.docs.scrapers.snapshot_runtime import refresh_snapshot_library

    return await refresh_snapshot_library(library_id)


async def check_library_freshness_details(
    library: LibraryState,
    use_cache: bool = True,
    repo_snapshot_cache: dict[tuple[str, str], RepoSnapshot] | None = None,
    prefer_full_checkout: bool = False,
) -> FreshnessResult:
    """
    Check if a library has new upstream commits.

    Results are cached for FRESHNESS_CACHE_TTL seconds to avoid
    hitting GitHub API on every query.

    Args:
        library: Full library state
        use_cache: Whether to use cached results (default: True)

    Returns:
        FreshnessState.STALE when a library needs reindexing, FRESH when up to
        date, and UNKNOWN when the remote could not be checked reliably.
    """
    global _freshness_cache
    library_id = library.library_id
    lock = _get_cache_lock()

    # Check cache first (read under lock for consistency)
    if use_cache:
        async with lock:
            if library_id in _freshness_cache:
                cached = _freshness_cache[library_id]
                age = (datetime.now(UTC) - cached.checked_at).total_seconds()
                if age < FRESHNESS_CACHE_TTL:
                    logger.debug(
                        "Using cached freshness for %s: state=%s",
                        library_id,
                        cached.state.value,
                    )
                    return cached

    if not library.current_sha:
        # Never indexed - definitely stale
        logger.info(f"Library {library_id} has no indexed SHA")
        result = FreshnessResult(
            state=FreshnessState.STALE,
            checked_at=datetime.now(UTC),
            remote_sha=None,
            error=None,
        )
    else:
        remote_sha = await _get_latest_source_ref(
            library,
            repo_snapshot_cache=repo_snapshot_cache,
            prefer_full_checkout=prefer_full_checkout,
        )
        if remote_sha is None:
            # Track consecutive failures so we can escalate logging
            count = _check_failure_counts.get(library_id, 0) + 1
            _check_failure_counts[library_id] = count
            if count >= CHECK_FAILURE_WARN_THRESHOLD:
                if library.source_type == LibrarySourceType.SNAPSHOT:
                    logger.warning(
                        "Library %s snapshot source unavailable for %d consecutive checks",
                        library_id,
                        count,
                    )
                else:
                    logger.warning(
                        "Library %s unreachable for %d consecutive checks (repo=%s, branch=%s) — verify repo exists and hasn't been renamed",
                        library_id,
                        count,
                        library.repo,
                        library.branch,
                    )
            else:
                logger.debug("Could not fetch remote SHA for %s", library_id)
            result = FreshnessResult(
                state=FreshnessState.UNKNOWN,
                checked_at=datetime.now(UTC),
                remote_sha=None,
                error="source_ref_unavailable",
            )
        elif not _source_refs_match(library, remote_sha):
            _check_failure_counts.pop(library_id, None)  # Reset on success
            logger.info(
                f"Library {library_id} is stale: "
                f"indexed={library.current_sha[:8]}, remote={remote_sha[:8]}"
            )
            result = FreshnessResult(
                state=FreshnessState.STALE,
                checked_at=datetime.now(UTC),
                remote_sha=remote_sha,
                error=None,
            )
        else:
            _check_failure_counts.pop(library_id, None)  # Reset on success
            logger.debug(f"Library {library_id} is fresh at {library.current_sha[:8]}")
            result = FreshnessResult(
                state=FreshnessState.FRESH,
                checked_at=datetime.now(UTC),
                remote_sha=remote_sha,
                error=None,
            )

    # Update cache (write under lock for atomicity)
    async with lock:
        _freshness_cache[library_id] = result

    return result


async def check_library_freshness(
    library: LibraryState,
    use_cache: bool = True,
) -> FreshnessState:
    """Return only the freshness state for callers that do not need metadata."""
    result = await check_library_freshness_details(
        library=library,
        use_cache=use_cache,
    )
    return result.state


async def _get_latest_source_ref(
    library: LibraryState,
    *,
    repo_snapshot_cache: dict[tuple[str, str], RepoSnapshot] | None,
    prefer_full_checkout: bool,
) -> str | None:
    if library.source_type == LibrarySourceType.SNAPSHOT:
        if _supports_snapshot_automation(library.library_id):
            return await _get_snapshot_source_ref(library.library_id)
        from repowise.docs.db import get_snapshot_state

        state = await get_snapshot_state(library.library_id)
        return state.source_ref if state is not None else None

    snapshot = await _get_repo_snapshot(
        library.repo,
        library.branch,
        prefer_full_checkout=prefer_full_checkout,
        repo_snapshot_cache=repo_snapshot_cache,
    )
    if snapshot.error or snapshot.repo_dir is None:
        return None
    try:
        return await get_source_ref(snapshot.repo_dir, library.source_subpath)
    except GitError:
        return None


async def _get_repo_snapshot(
    repo: str,
    branch: str,
    *,
    prefer_full_checkout: bool,
    repo_snapshot_cache: dict[tuple[str, str], RepoSnapshot] | None,
) -> RepoSnapshot:
    cache = repo_snapshot_cache if repo_snapshot_cache is not None else {}
    key = (repo, branch)
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        repo_dir = await clone_repo(
            repo,
            branch,
            prefer_full_checkout=prefer_full_checkout,
        )
        snapshot = RepoSnapshot(repo_dir=repo_dir, error=None)
    except Exception as exc:  # pragma: no cover - defensive
        snapshot = RepoSnapshot(repo_dir=None, error=str(exc))
    cache[key] = snapshot
    return snapshot


def clear_freshness_cache(library_id: str | None = None):
    """
    Clear the freshness cache.

    Args:
        library_id: Clear only this library, or all if None
    """
    global _freshness_cache
    # Direct dict operations are atomic in CPython (GIL)
    if library_id is None:
        _freshness_cache.clear()
    elif library_id in _freshness_cache:
        del _freshness_cache[library_id]


async def run_freshness_check():
    """
    Run a single freshness check for libraries based on their priority.

    Libraries are checked at different frequencies:
    - Priority 1-3 (high): every hour
    - Priority 4-6 (medium): every 6 hours
    - Priority 7-10 (low): every 24 hours

    This bypasses the cache to get fresh data from GitHub.

    Returns:
        Number of libraries queued for reindex
    """
    global _last_scheduler_run_started_at, _last_scheduler_run_completed_at, _last_scheduler_summary
    now_utc = datetime.now(UTC)
    _last_scheduler_run_started_at = now_utc

    libraries = await list_libraries(status=LibraryStatus.READY)
    queued_count = 0
    checked_count = 0
    skipped_count = 0
    unknown_count = 0
    repo_snapshot_cache: dict[tuple[str, str], RepoSnapshot] = {}
    git_libraries = [lib for lib in libraries if lib.source_type == LibrarySourceType.GIT]
    repo_requires_full_checkout = {
        (lib.repo, lib.branch): any(
            candidate.source_subpath
            for candidate in git_libraries
            if candidate.repo == lib.repo and candidate.branch == lib.branch
        )
        for lib in git_libraries
    }

    for lib in libraries:
        try:
            scheduled_at = _scheduled_check_at(lib, now=now_utc)
            if lib.next_freshness_check_at is None:
                await update_library_status(
                    lib.library_id,
                    next_freshness_check_at=scheduled_at,
                )
                lib = lib.model_copy(update={"next_freshness_check_at": scheduled_at})

            if scheduled_at > now_utc:
                skipped_count += 1
                logger.debug(
                    "Skipping %s (priority %d, next due %s)",
                    lib.name,
                    lib.priority,
                    scheduled_at.isoformat(),
                )
                continue

            checked_count += 1

            # Bypass cache for scheduled checks - we want fresh data
            freshness = await check_library_freshness_details(
                library=lib,
                use_cache=False,
                repo_snapshot_cache=repo_snapshot_cache,
                prefer_full_checkout=repo_requires_full_checkout.get((lib.repo, lib.branch), False),
            )
            next_check_at = compute_next_freshness_check_at(
                lib.library_id,
                lib.priority,
                freshness.checked_at,
            )
            await update_library_status(
                lib.library_id,
                freshness_checked_at=freshness.checked_at,
                next_freshness_check_at=next_check_at,
                last_freshness_state=freshness.state.value,
                last_remote_sha=freshness.remote_sha or "",
                last_freshness_error=freshness.error or "",
            )

            if freshness.state == FreshnessState.UNKNOWN:
                unknown_count += 1
                continue

            if freshness.state == FreshnessState.STALE:
                if lib.source_type == LibrarySourceType.SNAPSHOT and _supports_snapshot_automation(
                    lib.library_id
                ):
                    try:
                        result = await _refresh_snapshot_library(lib.library_id)
                    except Exception as exc:
                        message = f"snapshot_refresh_failed: {exc}"
                        logger.exception("Automatic snapshot refresh failed for %s", lib.library_id)
                        await update_library_status(
                            lib.library_id,
                            last_freshness_error=message,
                        )
                        continue

                    if result.queue_disposition in (
                        JobEnqueueDisposition.QUEUED.value,
                        JobEnqueueDisposition.UPGRADED.value,
                    ):
                        queued_count += 1
                    logger.info(
                        "Refreshed snapshot-backed docs for %s (changed=%s, queue=%s/%s)",
                        lib.name,
                        result.changed,
                        result.queued_job_type,
                        result.queue_disposition,
                    )
                    continue

                # Use library priority for the job (inverted: low priority number = high importance)
                job_priority = min(
                    10, max(5, lib.priority + 2)
                )  # Offset to not compete with manual triggers
                result = await enqueue_job(
                    lib.library_id,
                    JobType.INCREMENTAL,
                    priority=job_priority,
                )
                if result.disposition == JobEnqueueDisposition.QUEUED:
                    logger.info(
                        "Queued freshness update for %s (job %s, priority %d)",
                        lib.name,
                        result.job.id,
                        job_priority,
                    )
                    queued_count += 1
                    clear_freshness_cache(lib.library_id)
                elif result.disposition == JobEnqueueDisposition.UPGRADED:
                    logger.info(
                        "Upgraded pending freshness job for %s (job %s, priority %d)",
                        lib.name,
                        result.job.id,
                        result.job.priority,
                    )
                    queued_count += 1
                else:
                    logger.debug(f"Job coalesced for {lib.name}")

        except Exception as e:
            logger.error(f"Error checking freshness for {lib.library_id}: {e}")

    logger.debug(
        f"Freshness check: {checked_count} checked, {skipped_count} skipped, {queued_count} queued"
    )

    # ── Auto-retry transient error libraries ──────────────────────────────
    # Libraries stuck in ERROR state due to transient failures (DNS, timeouts)
    # get one retry attempt per scheduler cycle.
    error_libraries = await list_libraries(status=LibraryStatus.ERROR)
    transient_retry_count = 0

    for lib in error_libraries:
        if not lib.error_message:
            continue
        if not _is_transient_error(lib.error_message):
            continue
        retry_due_at = _transient_retry_due_at(lib, now=now_utc)
        if lib.next_freshness_check_at is None:
            await update_library_status(
                lib.library_id,
                next_freshness_check_at=retry_due_at,
            )
            lib = lib.model_copy(update={"next_freshness_check_at": retry_due_at})
        if retry_due_at > now_utc:
            continue

        try:
            result = await enqueue_job(lib.library_id, JobType.INCREMENTAL, priority=8)
            checked_at = datetime.now(UTC)
            next_retry_at = compute_next_freshness_check_at(
                lib.library_id,
                lib.priority,
                checked_at,
                interval_seconds=transient_error_retry_interval_seconds(lib.priority),
            )
            await update_library_status(
                lib.library_id,
                freshness_checked_at=checked_at,
                next_freshness_check_at=next_retry_at,
                last_freshness_state=FreshnessState.UNKNOWN.value,
                last_freshness_error=lib.error_message or "transient retry queued",
            )
            if result.disposition == JobEnqueueDisposition.QUEUED:
                logger.info(
                    f"Auto-retrying transient error library {lib.name}: {lib.error_message[:80]}"
                )
                transient_retry_count += 1
                queued_count += 1
            elif result.disposition == JobEnqueueDisposition.UPGRADED:
                logger.info(
                    "Upgraded pending retry job for transient error library %s",
                    lib.name,
                )
                transient_retry_count += 1
                queued_count += 1
            else:
                logger.debug(f"Retry job coalesced for {lib.name}")
        except Exception as e:
            logger.error(f"Error queueing retry for {lib.library_id}: {e}")

    if transient_retry_count:
        logger.info(f"Auto-retried {transient_retry_count} transient error libraries")

    # ── Staleness-based retry for unclassified errors ────────────────────
    # Any ERROR library that has been stuck longer than STALE_ERROR_RETRY_THRESHOLD
    # gets retried with exponential backoff, regardless of error classification.
    # This catches indexing pipeline failures, unrecognized network errors, etc.
    stale_error_retry_count = 0
    current_error_ids = {lib.library_id for lib in error_libraries}
    # Clear retry counters for libraries that recovered
    for lib_id in list(_stale_error_retry_counts):
        if lib_id not in current_error_ids:
            del _stale_error_retry_counts[lib_id]
    transient_retried_ids = {
        lib.library_id
        for lib in error_libraries
        if lib.error_message and _is_transient_error(lib.error_message)
    }

    for lib in error_libraries:
        if lib.library_id in transient_retried_ids:
            continue  # already handled above
        if _is_permanent_error(lib.error_message or ""):
            continue  # genuinely permanent — don't retry

        error_age = (now_utc - (lib.updated_at or lib.created_at)).total_seconds()
        if error_age < STALE_ERROR_RETRY_THRESHOLD:
            continue

        consecutive = _stale_error_retry_counts.get(lib.library_id, 0)
        retry_interval = stale_error_retry_interval_seconds(consecutive)
        retry_due_at = (lib.updated_at or lib.created_at) + timedelta(seconds=retry_interval)

        if lib.next_freshness_check_at is not None:
            retry_due_at = lib.next_freshness_check_at
        else:
            await update_library_status(
                lib.library_id,
                next_freshness_check_at=retry_due_at,
            )

        if retry_due_at > now_utc:
            continue

        try:
            result = await enqueue_job(lib.library_id, JobType.FORCE, priority=9)
            checked_at = datetime.now(UTC)
            _stale_error_retry_counts[lib.library_id] = consecutive + 1
            next_retry_at = checked_at + timedelta(
                seconds=stale_error_retry_interval_seconds(consecutive + 1)
            )
            await update_library_status(
                lib.library_id,
                freshness_checked_at=checked_at,
                next_freshness_check_at=next_retry_at,
                last_freshness_state=FreshnessState.UNKNOWN.value,
                last_freshness_error=f"stale_error_retry (attempt {consecutive + 1})",
            )
            if result.disposition in (JobEnqueueDisposition.QUEUED, JobEnqueueDisposition.UPGRADED):
                logger.info(
                    "Stale-error retry for %s (attempt %d, next in %s): %s",
                    lib.name,
                    consecutive + 1,
                    timedelta(seconds=stale_error_retry_interval_seconds(consecutive + 1)),
                    (lib.error_message or "")[:80],
                )
                stale_error_retry_count += 1
                queued_count += 1
            else:
                logger.debug(f"Stale-error retry job coalesced for {lib.name}")
        except Exception as e:
            logger.error(f"Error queueing stale-error retry for {lib.library_id}: {e}")

    if stale_error_retry_count:
        logger.info(f"Stale-error retried {stale_error_retry_count} libraries")

    _last_scheduler_run_completed_at = datetime.now(UTC)
    _last_scheduler_summary = {
        "ready_libraries": len(libraries),
        "checked": checked_count,
        "skipped": skipped_count,
        "queued": queued_count,
        "unknown": unknown_count,
        "transient_retries": transient_retry_count,
        "stale_error_retries": stale_error_retry_count,
    }

    return queued_count


async def scheduler_loop(check_interval: int = DEFAULT_CHECK_INTERVAL):
    """
    Main scheduler loop that runs freshness checks periodically.

    The scheduler runs every check_interval (default: 1 hour) and checks
    each library based on its priority setting.

    Args:
        check_interval: Base interval in seconds between scheduler runs
    """
    global _stop_event
    _stop_event = asyncio.Event()

    interval_mins = check_interval // 60
    logger.info(
        f"Scheduler started: running every {interval_mins} minutes with priority-based library checks"
    )

    while not _stop_event.is_set():
        try:
            logger.info("Running scheduled freshness check...")
            start = datetime.now(UTC)
            queued = await run_freshness_check()
            elapsed = (datetime.now(UTC) - start).total_seconds()

            logger.info(f"Freshness check complete: {queued} libraries queued in {elapsed:.1f}s")

            try:
                await asyncio.wait_for(
                    _stop_event.wait(),
                    timeout=check_interval,
                )
                break
            except TimeoutError:
                pass

        except Exception as e:
            logger.exception(f"Scheduler error: {e}")
            # Continue running despite errors
            await asyncio.sleep(60)  # Brief pause before retry

    logger.info("Scheduler stopped")


async def start_scheduler() -> asyncio.Task:
    """
    Start the background scheduler.

    Returns:
        The scheduler task
    """
    global _scheduler_task

    if _scheduler_task is not None and not _scheduler_task.done():
        logger.warning("Scheduler already running")
        return _scheduler_task

    settings = get_settings()
    check_interval = getattr(settings, "freshness_check_interval", DEFAULT_CHECK_INTERVAL)

    _scheduler_task = asyncio.create_task(scheduler_loop(check_interval))
    return _scheduler_task


async def stop_scheduler():
    """Stop the background scheduler gracefully."""
    global _scheduler_task, _stop_event

    if _stop_event is not None:
        _stop_event.set()

    if _scheduler_task is not None:
        try:
            await asyncio.wait_for(_scheduler_task, timeout=5.0)
        except TimeoutError:
            _scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _scheduler_task

        _scheduler_task = None

    logger.info("Scheduler stopped")


def is_scheduler_running() -> bool:
    """Check if the scheduler is currently running."""
    return _scheduler_task is not None and not _scheduler_task.done()


def get_unreachable_libraries() -> dict[str, int]:
    """Return libraries with consecutive check failures above threshold.

    Used by the health endpoint to surface libraries that have been
    silently unreachable (e.g., renamed repos, deleted branches).

    Returns:
        Dict mapping library_id to consecutive failure count.
    """
    return {
        lib_id: count
        for lib_id, count in _check_failure_counts.items()
        if count >= CHECK_FAILURE_WARN_THRESHOLD
    }


def get_scheduler_status() -> dict[str, object]:
    """Return runtime status and last-run summary for the freshness scheduler."""
    return {
        "running": is_scheduler_running(),
        "last_started_at": (
            _last_scheduler_run_started_at.isoformat() if _last_scheduler_run_started_at else None
        ),
        "last_completed_at": (
            _last_scheduler_run_completed_at.isoformat()
            if _last_scheduler_run_completed_at
            else None
        ),
        "summary": dict(_last_scheduler_summary),
    }
