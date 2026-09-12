"""Source-provider helpers for git-backed and snapshot-backed docs libraries."""

from __future__ import annotations

from pathlib import Path

from repowise.docs.db_snapshots import get_snapshot_file, get_snapshot_state, list_snapshot_files
from repowise.docs.indexer.incremental import FileInfo
from repowise.docs.library.models import LibrarySourceType, LibraryState, SnapshotFile
from repowise.docs.tools.common import ToolError, read_text_with_fallback, resolve_repo_file_path

from .repository import get_library_root_path, get_source_ref


def is_snapshot_library(library: LibraryState) -> bool:
    return library.source_type == LibrarySourceType.SNAPSHOT


def get_git_library_root(library: LibraryState) -> Path:
    return get_library_root_path(library.repo, library.source_subpath)


async def get_library_file_content(library: LibraryState, file_path: str) -> str:
    """Load the current file content for either source type."""
    if is_snapshot_library(library):
        snapshot_file = await get_snapshot_file(library.library_id, file_path)
        if snapshot_file is None:
            raise ToolError(f"File not found: {file_path}")
        return snapshot_file.content

    repo_path = get_git_library_root(library)
    if not repo_path.exists():
        raise ToolError(
            "Repository not cached. Please trigger indexing first with update_doc_library."
        )

    try:
        full_path = resolve_repo_file_path(repo_path, file_path)
    except ToolError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ToolError(f"Invalid path: {file_path}") from exc

    if not full_path.exists():
        raise ToolError(f"File not found: {file_path}")
    if not full_path.is_file():
        raise ToolError(f"Path is not a file: {file_path}")
    return read_text_with_fallback(full_path)


async def get_library_snapshot_files(library: LibraryState) -> list[SnapshotFile]:
    if not is_snapshot_library(library):
        return []
    return await list_snapshot_files(library.library_id)


async def get_snapshot_file_infos(library: LibraryState) -> list[FileInfo]:
    snapshot_files = await get_library_snapshot_files(library)
    return [
        FileInfo(path=file.file_path, content_hash=file.content_hash) for file in snapshot_files
    ]


async def get_latest_source_ref(
    library: LibraryState,
    *,
    repo_dir: Path | None = None,
) -> str | None:
    """Return the latest available source ref for a library."""
    if is_snapshot_library(library):
        state = await get_snapshot_state(library.library_id)
        return state.source_ref if state else None
    if repo_dir is None:
        raise ValueError("repo_dir is required for git-backed source ref lookup")
    return await get_source_ref(repo_dir, library.source_subpath)
