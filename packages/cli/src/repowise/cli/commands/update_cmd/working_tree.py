"""Documentation's working-tree checkpoint, independent of source/graph updates.

A saved edit invalidates pages once. Later runs drain stale pages without
replaying the same HEAD-relative dirty diff and invalidating them again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repowise.core import working_tree_state
from repowise.core.ingestion.change_detector import FileDiff

DOCS_CHECKPOINT = "docs_working_tree_checkpoint"


def documentation_base_ref(state: dict[str, Any]) -> str | None:
    # A degraded run may already have advanced the source or legacy docs
    # pointer. The successful documentation checkpoint owns the retry range.
    previous = state.get(DOCS_CHECKPOINT) or {}
    return previous.get("head") or state.get("last_docs_commit") or state.get("last_sync_commit")


def _contents(snapshot: dict[str, Any]) -> dict[str, str | None]:
    return {
        path: value[0] if value is not None else None for path, value in snapshot["files"].items()
    }


def previous_documentation_paths(state: dict[str, Any]) -> set[str]:
    checkpoint = state.get(DOCS_CHECKPOINT) or {}
    return set(checkpoint.get("files", {}))


def filter_documentation_diffs(
    diffs: list[FileDiff],
    state: dict[str, Any],
    current: dict[str, Any] | None,
) -> list[FileDiff]:
    """Drop only paths proved unchanged since successful documentation work.

    A different HEAD or unavailable snapshot retains the conservative replay.
    A reverted path is absent from the current dirty inventory and must stay
    in the diff. File stamps guard concurrent saves at publication; content
    hashes decide whether an edit needs another invalidation.
    """
    previous = state.get(DOCS_CHECKPOINT)
    if not previous or current is None or previous.get("head") != current.get("head"):
        return diffs
    old_files = _contents(previous)
    new_files = _contents(current)
    unchanged = {
        path for path in old_files.keys() & new_files.keys() if old_files[path] == new_files[path]
    }
    return [diff for diff in diffs if diff.path not in unchanged]


def documentation_update_reason(repo_path: Path, state: dict[str, Any]) -> str | None:
    """Keep documentation eligible after the watcher advances its own state."""
    current = working_tree_state.capture_working_tree(repo_path)
    previous = state.get(DOCS_CHECKPOINT)
    if current is None:
        return "documentation working-tree state unavailable"
    if previous is None:
        if current["files"] or state.get("last_docs_commit") != current["head"]:
            return "documentation changes pending"
        return None
    if previous["head"] != current["head"] or _contents(previous) != _contents(current):
        return "documentation changes pending"
    return None


def checkpoint_documentation(
    repo_path: Path,
    state: dict[str, Any],
    before: dict[str, Any] | None,
) -> bool:
    """Checkpoint only a completed run that did not overlap a save.

    On interruption or a concurrent edit, retain the prior checkpoint and its
    paths so a subsequent revert/deletion cannot disappear from recovery.
    """
    if before is None or working_tree_state.capture_working_tree(repo_path) != before:
        return False
    state[DOCS_CHECKPOINT] = before
    return True
