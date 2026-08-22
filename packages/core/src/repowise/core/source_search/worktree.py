"""What the working tree has done to the corpus since it was built.

Every other freshness signal on :class:`~.status.SourceIndexStatus` is read
back from something the *build* wrote: the manifest's ``indexed_commit``, the
outbox rows, ``state.json``'s ``working_tree_paths``. That is enough to notice
a new commit, because ``HEAD`` moves in the repository rather than in the
build's record — but it is structurally blind to an edit or a deletion made
after the build, which leaves no trace in any of those places. The index then
serves chunks of a file the working tree has changed, or removed outright,
with nothing anywhere saying so.

This module supplies the missing half: one live ``git diff`` against ``HEAD``,
intersected with the paths the active generation actually holds. It answers
only about *indexed* paths on purpose. A file the corpus never covered cannot
make an answer wrong, and reporting one would turn every scratch file in a
checkout into a staleness alarm.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DIVERGENCE_DELETED",
    "DIVERGENCE_MODIFIED",
    "HOT_PATH_CACHE_TTL",
    "UNCHECKED",
    "WorkingTreeDivergence",
    "divergence_from_candidates",
    "reset_cache_for_tests",
    "working_tree_candidates",
]

#: How a path diverges. Consumers switch on these, never on a git status
#: letter — ``git diff`` has more of those than this distinction needs, and
#: which ones it emits depends on rename detection being configured on.
DIVERGENCE_MODIFIED = "modified"
DIVERGENCE_DELETED = "deleted"

#: Bounded like every other git call in this codebase. A repository slow
#: enough to blow this is reported as unverified, which weakens the trust
#: claim — the direction that stays honest — instead of stalling a search.
_GIT_TIMEOUT = 10.0

#: The read is ~3.5 ms on a 4k-file checkout with a warm stat cache and is
#: paid per status call *and* per search. The cache is only here to collapse
#: the fan-out of a single tool call (an answer runs several searches), so the
#: window is deliberately shorter than the gap between a human or agent
#: editing a file and asking about it. Freshness questions
#: (``get_index_status``) pass ``max_age=0`` and never read it.
HOT_PATH_CACHE_TTL = 1.0
_CACHE_MAX = 16

_cache: dict[Path, tuple[float, dict[str, str], str | None]] = {}
_cache_lock = threading.Lock()

# ``--no-optional-locks`` keeps a read from taking ``.git/index.lock``: this
# runs inside a status call, and a diagnostic must not contend with whatever
# git the developer is running in the same checkout. ``diff.relative`` is
# pinned off because a repository may configure it on, and every path this
# module compares against is repo-root-relative. Submodules are skipped
# outright — their directory entry is never an indexed path, so recursing into
# them buys nothing.
_GIT_DIFF_ARGV = (
    "git",
    "--no-optional-locks",
    "-c",
    "core.quotepath=false",
    "-c",
    "diff.relative=false",
    "diff",
    "--name-status",
    "-z",
    "--ignore-submodules=all",
    "HEAD",
    "--",
)


@dataclass(frozen=True, slots=True)
class WorkingTreeDivergence:
    """Indexed paths the working tree has changed since the corpus was built.

    ``checked`` is the field that separates "the working tree is clean" from
    "nobody looked". Both have empty path tuples, and a consumer that reads
    only the tuples would report the second as the first — which is the exact
    silence this module exists to end.
    """

    checked: bool
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    @property
    def diverged(self) -> bool:
        return bool(self.modified or self.deleted)

    @property
    def total(self) -> int:
        return len(self.modified) + len(self.deleted)

    @property
    def paths(self) -> frozenset[str]:
        return frozenset(self.modified) | frozenset(self.deleted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "modified": list(self.modified),
            "deleted": list(self.deleted),
            "unavailable_reason": self.unavailable_reason,
        }


#: The verdict for a caller that had no way to look — no lexical store to ask
#: which paths are indexed, or a status read that deliberately skipped it.
UNCHECKED = WorkingTreeDivergence(checked=False, unavailable_reason="working_tree_not_read")


def reset_cache_for_tests() -> None:
    """Forget the process cache. Only tests have a reason to."""

    with _cache_lock:
        _cache.clear()


def _parse_name_status(payload: str) -> dict[str, str]:
    """``git diff --name-status -z`` output as path -> divergence kind.

    The stream is NUL-separated fields, not NUL-separated records: a rename or
    copy is ``R100\\0old\\0new``, everything else is ``X\\0path``. Reading it as
    records would pair a rename's status letter with the wrong path and then
    treat the following path as a status letter.
    """

    fields = [field for field in payload.split("\0") if field]
    changes: dict[str, str] = {}
    index = 0
    while index < len(fields):
        code = fields[index]
        index += 1
        if not code:
            continue
        letter = code[0]
        if letter in {"R", "C"} and index + 1 < len(fields):
            old_path, new_path = fields[index], fields[index + 1]
            index += 2
            # The old path's chunks describe a file that is no longer there.
            # A copy leaves the source in place, so only a rename retires it.
            if letter == "R":
                changes[old_path] = DIVERGENCE_DELETED
            changes[new_path] = DIVERGENCE_MODIFIED
            continue
        if index >= len(fields):
            break
        path = fields[index]
        index += 1
        changes[path] = DIVERGENCE_DELETED if letter == "D" else DIVERGENCE_MODIFIED
    return changes


def _run_git_diff(repo_path: Path) -> tuple[dict[str, str], str | None]:
    # Refuse to answer from an enclosing repository. Git walks upwards to find
    # one, and a corpus rooted below a repo root would then be compared against
    # paths spelled from *that* root — which intersect nothing here, so the
    # divergence would come back empty and be reported as a clean tree. Both a
    # plain checkout (directory) and a linked worktree (file) have this entry.
    if not (repo_path / ".git").exists():
        return {}, "not_a_git_repository"
    try:
        result = subprocess.run(
            _GIT_DIFF_ARGV,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {}, "git_diff_timeout"
    except Exception as exc:
        return {}, f"git_diff_unavailable: {type(exc).__name__}"
    if result.returncode != 0:
        # No commits yet, not a repository, an unreadable object store. All of
        # them mean the same thing here: the live comparison did not happen.
        return {}, "git_diff_failed"
    return _parse_name_status(result.stdout), None


def _tracked_changes(repo_path: Path, *, max_age: float) -> tuple[dict[str, str], str | None]:
    if max_age > 0.0:
        now = time.monotonic()
        with _cache_lock:
            cached = _cache.get(repo_path)
            if cached is not None and now - cached[0] <= max_age:
                return dict(cached[1]), cached[2]
    changes, error = _run_git_diff(repo_path)
    if max_age > 0.0:
        with _cache_lock:
            if len(_cache) >= _CACHE_MAX:
                _cache.clear()
            _cache[repo_path] = (time.monotonic(), dict(changes), error)
    return changes, error


def _vanished_working_tree_paths(repo_path: Path, known: Mapping[str, str]) -> dict[str, str]:
    """Paths indexed *from the working tree* that are no longer on disk.

    ``git diff HEAD`` cannot see these: an untracked file the watcher indexed
    has no committed side to diverge from, so deleting it makes it invisible
    to git entirely while its chunks keep serving. ``state.json``'s
    ``working_tree_paths`` is the build's own record of exactly that set — the
    same list :meth:`ChangeDetector.stale_working_tree_diffs` re-diffs on the
    next update — which makes it the bounded candidate list for a stat probe.
    Only deletions are read off it. A recorded path that still exists may have
    had its edit reverted, but it may equally have been re-indexed from a
    commit since, and this module refuses to guess between them.
    """

    from repowise.core.workspace.update import read_repo_state

    recorded = read_repo_state(repo_path).get("working_tree_paths") or []
    vanished: dict[str, str] = {}
    for entry in recorded:
        if not isinstance(entry, str) or entry in known:
            continue
        if not (repo_path / entry).is_file():
            vanished[entry] = DIVERGENCE_DELETED
    return vanished


def working_tree_candidates(
    repo_path: Path | str,
    *,
    max_age: float = 0.0,
) -> tuple[dict[str, str], str | None]:
    """Every repo-relative path the working tree has diverged on, live.

    Returns the candidates and an error code, not a verdict: which of these
    matter depends on which are actually in the active generation, and only a
    caller holding the lexical store can answer that.
    """

    repo = Path(repo_path).resolve()
    changes, error = _tracked_changes(repo, max_age=max_age)
    if error is not None:
        return {}, error
    try:
        changes.update(_vanished_working_tree_paths(repo, changes))
    except Exception as exc:  # pragma: no cover - defensive
        return changes, f"working_tree_record_unreadable: {type(exc).__name__}"
    return changes, None


def divergence_from_candidates(
    candidates: Mapping[str, str],
    indexed: Iterable[str] | None,
    *,
    error: str | None = None,
) -> WorkingTreeDivergence:
    """Narrow live candidates to the paths the active generation serves.

    *indexed* is ``None`` when membership could not be established. That is
    not "nothing is indexed": it is the unchecked verdict, because a caller
    who reported an empty divergence from it would be claiming a clean
    corpus on the strength of a store it never opened.
    """

    if error is not None:
        return WorkingTreeDivergence(checked=False, unavailable_reason=error)
    if indexed is None:
        return UNCHECKED
    served = frozenset(indexed)
    modified: list[str] = []
    deleted: list[str] = []
    for path, kind in candidates.items():
        if path not in served:
            continue
        (deleted if kind == DIVERGENCE_DELETED else modified).append(path)
    return WorkingTreeDivergence(
        checked=True,
        modified=tuple(sorted(modified)),
        deleted=tuple(sorted(deleted)),
    )
