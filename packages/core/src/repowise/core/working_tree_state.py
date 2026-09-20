"""Successful graph updates checkpoint the dirty files they actually covered.

Source publication has a separate lifecycle and must never mark the graph current.
The checkpoint lives in the existing repository state, shared by workspace hosts.
"""

from __future__ import annotations

import logging
from hashlib import file_digest
from pathlib import Path
from typing import Any

from .ingestion.change_detector import has_working_tree_changes
from .ingestion.traverser import is_candidate_source_path

_log = logging.getLogger(__name__)
_KEY = "working_tree_checkpoint"


def capture_working_tree(repo_path: Path) -> dict[str, Any] | None:
    """Read a conservative snapshot without parsing or rebuilding anything."""
    import git

    try:
        with git.Repo(repo_path) as repo:
            head = repo.head.commit
            paths = set(repo.untracked_files)
            for change in head.diff(None):
                paths.update(path for path in (change.a_path, change.b_path) if path)
            files: dict[str, Any] = {}
            for path in sorted(paths):
                if not is_candidate_source_path(path):
                    continue
                absolute = repo_path / path
                try:
                    before = absolute.stat()
                    with absolute.open("rb") as source:
                        digest = file_digest(source, "sha256").hexdigest()
                    after = absolute.stat()
                except FileNotFoundError:
                    files[path] = None
                    continue
                # A save while reading is not a usable checkpoint. mtime and
                # inode also expose edits restored to their previous contents.
                stamp = (before.st_mtime_ns, before.st_size, before.st_ino)
                if stamp != (after.st_mtime_ns, after.st_size, after.st_ino):
                    return None
                files[path] = [digest, *stamp]
            return {"head": head.hexsha, "files": files}
    except (OSError, ValueError, git.GitError) as exc:
        _log.warning("Working-tree checkpoint unavailable for %s: %s", repo_path, exc)
        return None


def working_tree_update_reason(repo_path: Path, state: dict[str, Any]) -> str | None:
    """Explain changed dirty content or cleanup, without rebuilding unchanged work."""
    previous = state.get(_KEY)
    current = capture_working_tree(repo_path)
    if current is None:
        if previous is not None:
            return "working-tree state unavailable"
        dirty = has_working_tree_changes(repo_path)
    else:
        if current == previous:
            return None
        dirty = bool(current["files"])
    if dirty:
        return "uncommitted changes"
    if previous is not None or state.get("working_tree_paths"):
        return "working-tree cleanup"
    return None


def checkpoint_working_tree(
    repo_path: Path, state: dict[str, Any], before: dict[str, Any] | None
) -> None:
    """Amend state only after success, and only if no save overlapped the run."""
    if before is not None and capture_working_tree(repo_path) == before:
        state[_KEY] = before
    else:
        state.pop(_KEY, None)
