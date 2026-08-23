"""What ``--since <ref>`` resolves to, and the one shape of it that lies.

``ChangeDetector.get_changed_files`` resolves both ends of the range with
GitPython's ``repo.commit()``. Anything that raises there is swallowed into an
empty list, which every caller reads as "nothing changed" -- so a ref that
cannot be resolved is reported as a clean, up-to-date repo rather than as a
failure to look.

That is a real trap, not a hypothetical one: diffing from git's empty-tree sha
is the standard way to ask for "everything", it is a valid object, and it
silently produces no changes at all. These tests pin the trap and the recipe
that works around it, so neither can drift unnoticed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from repowise.core.ingestion.change_detector import ChangeDetector

#: git's empty tree. A real object in every repository, and not a commit.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    """A two-commit repository: one file, then a second one."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    _git(repo.parent, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")

    (repo / "src" / "widget.py").write_text("def render():\n    return 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "first")

    (repo / "src" / "extra.py").write_text("def spawn():\n    return 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    return repo


def test_a_commit_ref_diffs_the_range(tmp_path):
    """Control: an ordinary commit ref resolves and reports its range."""
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD~1")

    diffs = ChangeDetector(repo).get_changed_files(base, "HEAD")

    assert {fd.path for fd in diffs} == {"src/extra.py"}


def test_the_empty_tree_sha_reads_as_no_changes_rather_than_everything(tmp_path):
    """The trap, pinned.

    ``git diff <empty-tree> HEAD`` lists the whole tree, so the sha looks like
    the obvious way to force a full re-diff. It is not: ``repo.commit()``
    rejects it as a tree, the exception is swallowed, and the caller is told
    the repo is unchanged. Nothing distinguishes this from a genuinely clean
    range, which is what makes it cost a debugging session rather than a
    glance at a stack trace.

    This test asserts today's behaviour, NOT that the behaviour is right. A
    change that makes an unresolvable ref loud is an improvement and should
    replace this test -- see the note in ``get_changed_files``.
    """
    repo = _repo(tmp_path)

    # git itself has no trouble with the range.
    porcelain = _git(repo, "diff", "--name-only", EMPTY_TREE, "HEAD").splitlines()
    assert set(porcelain) == {"src/widget.py", "src/extra.py"}

    # The detector reports the opposite, and says nothing about why.
    assert ChangeDetector(repo).get_changed_files(EMPTY_TREE, "HEAD") == []


def test_a_dangling_empty_commit_is_the_full_tree_diff_that_works(tmp_path):
    """The recipe, pinned.

    Wrapping the empty tree in a commit gives a base that ``repo.commit()``
    resolves, so the range is the whole tree -- the supported way to force a
    full re-parse through the incremental path.
    """
    repo = _repo(tmp_path)
    empty_commit = _git(repo, "commit-tree", EMPTY_TREE, "-m", "empty base")

    diffs = ChangeDetector(repo).get_changed_files(empty_commit, "HEAD")

    assert {fd.path for fd in diffs} == {"src/widget.py", "src/extra.py"}


def test_a_ref_that_does_not_exist_also_reads_as_no_changes(tmp_path):
    """The documented fallback, and why the trap above is hard to narrow.

    A missing ref is indistinguishable, at this seam, from a ref that resolves
    to the wrong kind of object -- both arrive as an exception from
    ``repo.commit()``. The fallback is deliberate here: the workspace updater
    and the server job executor pass a *stored* pointer that a rebase or a gc
    can legitimately invalidate, and both rely on self-healing rather than
    failing. Any hardening has to keep that while still being loud about a
    ref a person typed.
    """
    repo = _repo(tmp_path)

    assert ChangeDetector(repo).get_changed_files("no-such-ref", "HEAD") == []
