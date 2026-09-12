"""Job queue module for doc-search.

This module provides a durable job queue for background indexing operations.
Jobs survive restarts and support concurrent workers with proper locking.

## Architecture

```
HTTP Trigger                    Worker Loop
    │                               │
    ▼                               ▼
enqueue_job() ──▶ index_jobs ◀── claim_job()
                     table          │
                                    ▼
                             index_library()
                                    │
                              heartbeat_job()
                                    │
                              complete_job()
                              or fail_job()
```

## Key Guarantees

1. **Queue coalescing**: Only one pending follow-up job per library; a running job
   can coexist with one stronger pending follow-up request
2. **Atomic claiming**: FOR UPDATE SKIP LOCKED prevents double-claiming
   (see `single_writer` for the process-level guard that keeps a second
   replica from running a second scheduler and recovery loop at all)
3. **Stale recovery**: Jobs without heartbeat are reclaimed automatically
4. **Progress tracking**: files_processed/files_total updated during indexing

## Usage

```python
from repowise.docs.jobs import enqueue_job, start_worker

# Enqueue a job (returns a disposition + effective job)
result = await enqueue_job("/langfuse/langfuse-docs", JobType.INCREMENTAL)

# Start worker (runs in background)
task = asyncio.create_task(start_worker("worker-1"))
```
"""

# Re-export job queue operations from db module
from repowise.docs.db import (
    claim_job,
    complete_job,
    enqueue_job,
    fail_job,
    get_job,
    heartbeat_job,
    list_jobs,
    reclaim_stale_jobs,
)

# Export recovery functions
from repowise.docs.jobs.recovery import (
    is_recovery_running,
    recover_orphaned_libraries,
    start_recovery_task,
    stop_recovery_task,
)

# Export scheduler functions
from repowise.docs.jobs.scheduler import (
    FreshnessState,
    check_library_freshness,
    clear_freshness_cache,
    get_scheduler_status,
    get_unreachable_libraries,
    is_scheduler_running,
    run_freshness_check,
    start_scheduler,
    stop_scheduler,
)

# Export the single-writer guard
from repowise.docs.jobs.single_writer import (
    DOCS_BACKGROUND_LOCK_KEY,
    acquire_background_lock,
    holds_background_lock,
    release_background_lock,
)

# Export worker functions
from repowise.docs.jobs.worker import (
    get_worker_id,
    is_worker_running,
    start_worker,
    stop_worker,
)
from repowise.docs.library.models import (
    IndexJob,
    JobEnqueueDisposition,
    JobEnqueueResult,
    JobStatus,
    JobType,
)

__all__ = [
    # Single-writer guard
    "DOCS_BACKGROUND_LOCK_KEY",
    "FreshnessState",
    # Types
    "IndexJob",
    "JobEnqueueDisposition",
    "JobEnqueueResult",
    "JobStatus",
    "JobType",
    "acquire_background_lock",
    "check_library_freshness",
    "claim_job",
    "clear_freshness_cache",
    "complete_job",
    # Queue operations
    "enqueue_job",
    "fail_job",
    "get_job",
    "get_scheduler_status",
    "get_unreachable_libraries",
    "get_worker_id",
    "heartbeat_job",
    "holds_background_lock",
    "is_recovery_running",
    "is_scheduler_running",
    "is_worker_running",
    "list_jobs",
    "reclaim_stale_jobs",
    "recover_orphaned_libraries",
    "release_background_lock",
    "run_freshness_check",
    # Recovery control
    "start_recovery_task",
    # Scheduler control
    "start_scheduler",
    # Worker control
    "start_worker",
    "stop_recovery_task",
    "stop_scheduler",
    "stop_worker",
]
