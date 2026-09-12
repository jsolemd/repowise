"""End-to-end indexing pipeline for a single library."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from repowise.docs.chunking import chunk_file
from repowise.docs.db import (
    delete_library_file,
    get_library,
    get_library_files,
    update_library_status,
    upsert_library,
    upsert_library_file,
)
from repowise.docs.indexer import (
    ChangeSet,
    compute_changes,
    delete_by_file,
    delete_files_outside_scope,
    delete_stale_file_chunks,
    ensure_collection,
    get_chunk_count,
    get_qdrant_client,
    upsert_chunks,
)
from repowise.docs.indexer.incremental import compute_files_info
from repowise.docs.jobs.policy import compute_next_freshness_check_at
from repowise.docs.library.git_manager import (
    GitError,
    clone_repo,
    get_head_sha,
    get_library_root_path_from_repo_dir,
    list_doc_files,
    resolve_default_branch,
    resolve_repo_docs_path,
)
from repowise.docs.library.models import LibraryConfig, LibraryStatus
from repowise.docs.library.source_runtime import (
    get_latest_source_ref,
    get_library_snapshot_files,
    get_snapshot_file_infos,
    is_snapshot_library,
)

logger = logging.getLogger(__name__)


def _partial_index_warning(errors: list[str]) -> str:
    preview = ", ".join(sorted(errors)[:5])
    return f"Skipped {len(errors)} file(s) during indexing: {preview}"


def _next_freshness_check(library_id: str, priority: int, checked_at: datetime) -> datetime:
    return compute_next_freshness_check_at(library_id, priority, checked_at)


async def _reconcile_index_scope(
    library_id: str,
    valid_file_paths: set[str],
    client,
) -> dict[str, int | list[str]]:
    """Prune indexed files that no longer belong to the discovered library scope."""
    if not valid_file_paths:
        return {"files_removed": 0, "chunks_removed": 0, "paths": []}
    return await delete_files_outside_scope(library_id, valid_file_paths, client)


async def _persist_library_config(
    library,
    *,
    branch: str | None = None,
    docs_path: str | None = None,
):
    """Persist discovered library config changes without resetting index state."""
    next_branch = branch if branch is not None else library.branch
    next_docs_path = docs_path if docs_path is not None else library.docs_path
    if next_branch == library.branch and next_docs_path == library.docs_path:
        return library

    config = LibraryConfig(
        source_type=library.source_type,
        library_id=library.library_id,
        repo=library.repo,
        name=library.name,
        description=library.description,
        source_subpath=library.source_subpath,
        docs_path=next_docs_path,
        branch=next_branch,
        include_patterns=list(library.include_patterns),
        exclude_patterns=list(library.exclude_patterns),
        priority=library.priority,
    )
    return await upsert_library(config)


async def index_library(library_id: str, force: bool = False) -> dict:
    """
    Full indexing pipeline for a library.

    Args:
        library_id: Library identifier (e.g., "/langfuse/langfuse-docs")
        force: If True, reindex all files regardless of hash

    Returns:
        Stats dict with files_processed, chunks_indexed, etc.
    """
    client = get_qdrant_client()

    # Ensure collection exists
    ensure_collection(client)

    # 1. Get library config from DB
    library = await get_library(library_id)
    if not library:
        raise ValueError(f"Library not found: {library_id}")

    logger.info(f"Indexing library: {library.name} ({library.repo})")

    # Update status to indexing
    result = await update_library_status(library_id, status=LibraryStatus.INDEXING)
    if not result:
        raise ValueError(f"Failed to set library {library_id} to INDEXING status")

    try:
        repo_head_sha: str | None = None
        branch = library.branch
        resolved_docs_path = library.docs_path
        library_root = None
        snapshot_files_by_path: dict[str, str] = {}
        current_files = None

        if is_snapshot_library(library):
            snapshot_files = await get_library_snapshot_files(library)
            if not snapshot_files:
                raise ValueError(f"Snapshot source for {library_id} has not been published yet")
            source_ref = await get_latest_source_ref(library)
            if source_ref is None:
                raise ValueError(f"Snapshot source ref missing for {library_id}")
            doc_files = [Path(file.file_path) for file in snapshot_files]
            snapshot_files_by_path = {file.file_path: file.content for file in snapshot_files}
            current_files = await get_snapshot_file_infos(library)
            logger.info(
                "Using snapshot source ref %s with %d files", source_ref[:8], len(snapshot_files)
            )
        else:
            # 2. Clone or fetch repo (with branch auto-correction)
            try:
                repo_dir = await clone_repo(
                    library.repo,
                    branch,
                    library.docs_path,
                    library.source_subpath,
                )
            except GitError as e:
                if "couldn't find remote ref" in e.stderr.lower():
                    # Wrong branch — try to auto-detect the correct one
                    logger.warning(
                        f"Branch '{branch}' not found for {library.repo}, "
                        f"attempting auto-detection..."
                    )
                    default_branch = await resolve_default_branch(library.repo)
                    if default_branch and default_branch != branch:
                        logger.info(
                            f"Auto-correcting branch for {library.repo}: "
                            f"'{branch}' → '{default_branch}'"
                        )
                        branch = default_branch
                        library = await _persist_library_config(library, branch=branch)
                        repo_dir = await clone_repo(
                            library.repo,
                            branch,
                            library.docs_path,
                            library.source_subpath,
                            force=True,
                        )
                    else:
                        raise
                else:
                    raise
            repo_head_sha = await get_head_sha(repo_dir)
            source_ref = await get_latest_source_ref(library, repo_dir=repo_dir)
            if source_ref is None:
                raise ValueError(f"Could not resolve source ref for {library_id}")
            logger.info(
                "At source ref: %s (repo head %s)",
                source_ref[:8],
                repo_head_sha[:8],
            )

            resolved_docs_path = await resolve_repo_docs_path(
                repo_dir,
                branch,
                library.docs_path,
                library.source_subpath,
            )
            if resolved_docs_path != library.docs_path:
                library = await _persist_library_config(library, docs_path=resolved_docs_path)
            library_root = get_library_root_path_from_repo_dir(repo_dir, library.source_subpath)

            doc_files = list_doc_files(
                library_root,
                resolved_docs_path,
                library.include_patterns,
                library.exclude_patterns,
            )
            logger.info(f"Found {len(doc_files)} doc files")

        stored_files = None
        doc_file_paths = {
            str(path.file_path if hasattr(path, "file_path") else path) for path in doc_files
        }
        scope_reconciliation = await _reconcile_index_scope(
            library_id,
            doc_file_paths,
            client,
        )

        if (
            not force
            and library.status == LibraryStatus.READY
            and library.current_sha
            and library.current_sha == source_ref
        ):
            stored_files = await get_library_files(library_id)
            stored_paths = {stored.file_path for stored in stored_files}
            if doc_file_paths == stored_paths:
                logger.info(
                    "Repository HEAD unchanged and indexed file set matched cache; "
                    "skipping hash scan and reindex"
                )
                chunk_count = await get_chunk_count(library_id, client)
                checked_at = datetime.now(UTC)
                result = await update_library_status(
                    library_id,
                    current_sha=source_ref,
                    status=LibraryStatus.READY,
                    chunk_count=chunk_count,
                    file_count=len(doc_files),
                    freshness_checked_at=checked_at,
                    last_freshness_state="fresh",
                    last_remote_sha=source_ref,
                    last_freshness_error="",
                    next_freshness_check_at=_next_freshness_check(
                        library_id,
                        library.priority,
                        checked_at,
                    ),
                    error_message="",
                )
                if not result:
                    raise ValueError(f"Failed to set library {library_id} to READY status")
                response = {
                    "library_id": library_id,
                    "commit_sha": repo_head_sha,
                    "source_ref": source_ref,
                    "status": "unchanged",
                    "files_found": len(doc_files),
                    "total_chunks": chunk_count,
                }
                if scope_reconciliation["files_removed"]:
                    response["files_pruned"] = scope_reconciliation["files_removed"]
                    response["chunks_pruned"] = scope_reconciliation["chunks_removed"]
                return response

        if current_files is None:
            current_files = compute_files_info(library_root, doc_files)

        # 4. Compute changes (hash-based)
        if force:
            # Force reindex: treat all files as added
            changes = ChangeSet(added={f.path for f in current_files})
            logger.info("Force reindex: processing all files")
        else:
            if stored_files is None:
                stored_files = await get_library_files(library_id)
            changes = compute_changes(current_files, stored_files)

        logger.info(f"Changes: {changes}")

        if not changes.has_changes and not force:
            logger.info("No changes detected, skipping indexing")
            chunk_count = await get_chunk_count(library_id, client)
            # Still need to update library status to READY (was set to INDEXING at start)
            checked_at = datetime.now(UTC)
            result = await update_library_status(
                library_id,
                current_sha=source_ref,
                status=LibraryStatus.READY,
                chunk_count=chunk_count,
                file_count=len(doc_files),
                freshness_checked_at=checked_at,
                last_freshness_state="fresh",
                last_remote_sha=source_ref,
                last_freshness_error="",
                next_freshness_check_at=_next_freshness_check(
                    library_id,
                    library.priority,
                    checked_at,
                ),
                error_message="",
            )
            if not result:
                raise ValueError(f"Failed to set library {library_id} to READY status")
            response = {
                "library_id": library_id,
                "commit_sha": repo_head_sha,
                "source_ref": source_ref,
                "status": "unchanged",
                "files_found": len(doc_files),
                "total_chunks": chunk_count,
            }
            if scope_reconciliation["files_removed"]:
                response["files_pruned"] = scope_reconciliation["files_removed"]
                response["chunks_pruned"] = scope_reconciliation["chunks_removed"]
            return response

        # 5. Process deletions
        for path in changes.deleted:
            await delete_by_file(library_id, path, client)
            await delete_library_file(library_id, path)
            logger.debug(f"Deleted chunks for {path}")

        # 6. Process additions and modifications
        total_chunks = 0
        errors = []
        successful_files = 0
        current_files_by_path = {file.path: file for file in current_files}

        for path in changes.added | changes.modified:
            if is_snapshot_library(library):
                content = snapshot_files_by_path.get(path)
                if content is None:
                    logger.error("Snapshot file missing for %s", path)
                    errors.append(str(path))
                    continue
            else:
                full_path = library_root / path
                try:
                    content = full_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    logger.warning(f"UTF-8 decode failed for {path}, trying latin-1")
                    content = full_path.read_text(encoding="latin-1")
                except Exception as e:
                    logger.error(f"Failed to read {path}: {e}")
                    errors.append(str(path))
                    continue

            try:
                chunks = chunk_file(
                    str(path),
                    content,
                    library_id,
                    repo_head_sha or source_ref,
                    library.repo or None,
                    library.branch,
                    library.source_subpath,
                )

                if chunks:
                    await upsert_chunks(chunks, client)
                    total_chunks += len(chunks)
                    logger.debug(f"Indexed {len(chunks)} chunks from {path}")

            except Exception as e:
                logger.error(f"Failed to chunk/index {path}: {e}")
                errors.append(str(path))
                continue

            # Delete stale leftovers only after the replacement chunks are safely written.
            if path in changes.modified or force:
                current_chunk_ids = [chunk.id for chunk in chunks]
                if current_chunk_ids:
                    await delete_stale_file_chunks(library_id, path, current_chunk_ids, client)
                else:
                    await delete_by_file(library_id, path, client)

            # Update file hash in DB
            file_info = current_files_by_path.get(path)
            if file_info:
                await upsert_library_file(
                    library_id,
                    path,
                    file_info.content_hash,
                    chunk_count=len(chunks) if chunks else 0,
                )
            successful_files += 1

        # 7. Update library status
        chunk_count = await get_chunk_count(library_id, client)
        partial_warning = _partial_index_warning(errors) if errors else ""
        if errors and successful_files == 0:
            raise RuntimeError(
                f"Indexing failed for {len(errors)} file(s): {', '.join(sorted(errors)[:5])}"
            )
        checked_at = datetime.now(UTC)
        result = await update_library_status(
            library_id,
            current_sha=source_ref,
            status=LibraryStatus.READY,
            chunk_count=chunk_count,
            file_count=len(doc_files),
            indexed_at=checked_at,
            freshness_checked_at=checked_at,
            last_freshness_state="fresh",
            last_remote_sha=source_ref,
            last_freshness_error="",
            next_freshness_check_at=_next_freshness_check(
                library_id,
                library.priority,
                checked_at,
            ),
            error_message=partial_warning,
        )
        if not result:
            raise ValueError(f"Failed to set library {library_id} to READY status")

        response = {
            "library_id": library_id,
            "commit_sha": repo_head_sha,
            "source_ref": source_ref,
            "status": "indexed",
            "files_found": len(doc_files),
            "files_added": len(changes.added),
            "files_modified": len(changes.modified),
            "files_deleted": len(changes.deleted),
            "chunks_indexed": total_chunks,
            "total_chunks": chunk_count,
            "errors": errors,
        }
        if partial_warning:
            response["warning"] = partial_warning
        if scope_reconciliation["files_removed"]:
            response["files_pruned"] = scope_reconciliation["files_removed"]
            response["chunks_pruned"] = scope_reconciliation["chunks_removed"]
        return response

    except Exception as e:
        # Update library status to error (best-effort, don't mask original error)
        try:
            await update_library_status(
                library_id,
                status=LibraryStatus.ERROR,
                error_message=str(e),
            )
        except Exception as status_err:
            logger.error(f"Failed to update library {library_id} to ERROR status: {status_err}")
        raise
