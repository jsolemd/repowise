"""Shared cadence policy for docs freshness probes and transient retries."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

DEFAULT_CHECK_INTERVAL = 60 * 60

PRIORITY_INTERVALS = {
    "high": 1 * 60 * 60,
    "medium": 6 * 60 * 60,
    "low": 24 * 60 * 60,
}

_FRESHNESS_JITTER_RATIO = 0.2


def get_check_interval_for_priority(priority: int) -> int:
    """Return the steady-state freshness interval for a library priority."""
    if priority <= 3:
        return PRIORITY_INTERVALS["high"]
    if priority <= 6:
        return PRIORITY_INTERVALS["medium"]
    return PRIORITY_INTERVALS["low"]


def transient_error_retry_interval_seconds(priority: int) -> int:
    """Back off transient retry probes relative to the normal freshness cadence."""
    return max(DEFAULT_CHECK_INTERVAL, get_check_interval_for_priority(priority) // 2)


# Staleness-based retry: any ERROR library older than this gets retried
# regardless of whether the error is classified as transient.
STALE_ERROR_RETRY_THRESHOLD = 24 * 60 * 60  # 24 hours
STALE_ERROR_MAX_BACKOFF = 7 * 24 * 60 * 60  # 7 days


def stale_error_retry_interval_seconds(consecutive_failures: int) -> int:
    """Exponential backoff for unclassified error retries: 24h, 48h, 96h, ... capped at 7d."""
    interval = STALE_ERROR_RETRY_THRESHOLD * (2 ** max(0, consecutive_failures - 1))
    return min(interval, STALE_ERROR_MAX_BACKOFF)


def _stable_offset_seconds(library_id: str, window_seconds: int) -> int:
    if window_seconds <= 0:
        return 0
    digest = hashlib.sha256(library_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % window_seconds


def _jitter_seconds(library_id: str, interval_seconds: int) -> int:
    jitter_window = min(
        max(60, int(interval_seconds * _FRESHNESS_JITTER_RATIO)), DEFAULT_CHECK_INTERVAL
    )
    return _stable_offset_seconds(library_id, jitter_window)


def compute_next_freshness_check_at(
    library_id: str,
    priority: int,
    baseline: datetime,
    *,
    interval_seconds: int | None = None,
) -> datetime:
    """Return the next persisted probe time for a library."""
    interval = interval_seconds or get_check_interval_for_priority(priority)
    return baseline + timedelta(seconds=interval + _jitter_seconds(library_id, interval))


def bootstrap_next_freshness_check_at(
    library_id: str,
    priority: int,
    baseline: datetime | None,
    *,
    now: datetime | None = None,
    interval_seconds: int | None = None,
) -> datetime:
    """Seed a persisted schedule without bursting every library on restart."""
    now_utc = now or datetime.now(UTC)
    interval = interval_seconds or get_check_interval_for_priority(priority)

    if baseline is not None:
        scheduled = compute_next_freshness_check_at(
            library_id,
            priority,
            baseline,
            interval_seconds=interval,
        )
        if scheduled > now_utc:
            return scheduled

    bootstrap_window = min(interval, DEFAULT_CHECK_INTERVAL)
    return now_utc + timedelta(seconds=_stable_offset_seconds(library_id, bootstrap_window))
