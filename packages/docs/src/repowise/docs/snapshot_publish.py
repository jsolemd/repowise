"""Native publish path for snapshot-backed documentation libraries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from repowise.docs import db
from repowise.docs.db_snapshots import SnapshotFileInput
from repowise.docs.jobs import clear_freshness_cache
from repowise.docs.library.git_manager import (
    clone_repo,
    get_library_root_path_from_repo_dir,
    get_source_ref,
    list_doc_files,
    resolve_repo_docs_path,
)
from repowise.docs.library.models import JobEnqueueDisposition, JobType, LibrarySourceType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotPublishResult:
    library_id: str
    changed: bool
    source_ref: str
    manifest_hash: str
    file_count: int
    queued_job_type: str | None = None
    queue_disposition: str | None = None


def collect_snapshot_files(output_dir: Path) -> list[SnapshotFileInput]:
    """Collect a logical docs tree into current-state snapshot rows."""
    if not output_dir.exists():
        raise FileNotFoundError(f"Snapshot output directory not found: {output_dir}")

    files: list[SnapshotFileInput] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        relative_path = path.relative_to(output_dir).as_posix()
        files.append(
            SnapshotFileInput(
                file_path=relative_path,
                content=path.read_text(encoding="utf-8"),
            )
        )
    return files


def _collect_snapshot_files_from_paths(
    root: Path, file_paths: list[Path]
) -> list[SnapshotFileInput]:
    return [
        SnapshotFileInput(
            file_path=path.as_posix(),
            content=(root / path).read_text(encoding="utf-8"),
        )
        for path in sorted(file_paths)
        if (root / path).is_file()
    ]


async def publish_snapshot_files(
    library_id: str,
    files: list[SnapshotFileInput],
    *,
    source_ref: str | None = None,
    queue_reindex: bool = True,
) -> SnapshotPublishResult:
    """Replace the current docs state for a snapshot-backed library and queue reindexing."""
    library = await db.get_library(library_id)
    if library is None:
        raise ValueError(f"Library not found: {library_id}")
    if library.source_type != LibrarySourceType.SNAPSHOT:
        raise ValueError(f"{library_id} is not configured as a snapshot-backed library")

    snapshot_state, changed = await db.replace_snapshot(
        library_id,
        files,
        source_ref=source_ref,
    )

    checked_at = datetime.now(UTC)
    remote_changed = library.current_sha != snapshot_state.source_ref
    await db.update_library_status(
        library_id,
        freshness_checked_at=checked_at,
        next_freshness_check_at=checked_at,
        last_remote_sha=snapshot_state.source_ref,
        last_freshness_state="stale" if remote_changed else "fresh",
        last_freshness_error="",
        file_count=snapshot_state.file_count,
        error_message="",
    )
    clear_freshness_cache(library_id)

    queued_job_type: str | None = None
    queue_disposition: str | None = None
    if changed and queue_reindex:
        job_type = JobType.FULL if not library.current_sha else JobType.INCREMENTAL
        result = await db.enqueue_job(
            library_id,
            job_type,
            priority=library.priority,
        )
        queued_job_type = job_type.value
        queue_disposition = result.disposition.value
        if result.disposition != JobEnqueueDisposition.COALESCED:
            logger.info(
                "Queued %s snapshot refresh for %s (%s)",
                job_type.value,
                library_id,
                result.disposition.value,
            )

    return SnapshotPublishResult(
        library_id=library_id,
        changed=changed,
        source_ref=snapshot_state.source_ref,
        manifest_hash=snapshot_state.manifest_hash,
        file_count=snapshot_state.file_count,
        queued_job_type=queued_job_type,
        queue_disposition=queue_disposition,
    )


async def publish_snapshot_directory(
    library_id: str,
    output_dir: Path,
    *,
    source_ref: str | None = None,
    queue_reindex: bool = True,
) -> SnapshotPublishResult:
    """Publish a directory tree as the current docs state for a snapshot-backed library."""
    files = collect_snapshot_files(output_dir)
    return await publish_snapshot_files(
        library_id,
        files,
        source_ref=source_ref,
        queue_reindex=queue_reindex,
    )


async def publish_snapshot_repo_source(
    library_id: str,
    *,
    repo: str,
    branch: str = "main",
    docs_path: str = "",
    source_subpath: str = "",
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    queue_reindex: bool = True,
) -> SnapshotPublishResult:
    """Publish a git-backed docs subtree into snapshot storage.

    This is primarily for one-time migrations from older git-backed doc sources
    into the native current-state snapshot model.
    """
    repo_dir = await clone_repo(repo, branch, docs_path, source_subpath, force=True)
    resolved_docs_path = await resolve_repo_docs_path(repo_dir, branch, docs_path, source_subpath)
    library_root = get_library_root_path_from_repo_dir(repo_dir, source_subpath)
    doc_files = list_doc_files(
        library_root,
        resolved_docs_path,
        include_patterns or ["**/*.md", "**/*.mdx", "**/*.rst", "**/*.adoc", "**/*.txt"],
        exclude_patterns or [],
    )
    source_ref = await get_source_ref(repo_dir, source_subpath)
    files = _collect_snapshot_files_from_paths(library_root, doc_files)
    return await publish_snapshot_files(
        library_id,
        files,
        source_ref=source_ref,
        queue_reindex=queue_reindex,
    )
