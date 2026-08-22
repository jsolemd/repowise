"""Divergence that happens *after* the build, and who gets told about it.

Every other freshness signal is read back from something the build wrote, so
each of them is structurally incapable of noticing an edit or a deletion made
afterwards. These tests mutate a real checkout after a real corpus was built
over it and then ask both places staleness is spoken — the status verdict and
the search envelope — whether they can see it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from repowise.core.source_search import status as status_module
from repowise.core.source_search.chunks import SymbolRecord, build_symbol_chunk
from repowise.core.source_search.fts import SourceFTSIndex
from repowise.core.source_search.generation import GenerationRef
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    default_manifest_path,
    write_manifest,
)
from repowise.core.source_search.status import inspect_source_index
from repowise.core.source_search.worktree import (
    DIVERGENCE_DELETED,
    DIVERGENCE_MODIFIED,
    UNCHECKED,
    divergence_from_candidates,
    reset_cache_for_tests,
    working_tree_candidates,
)

_GENERATION = GenerationRef("generation-1", 1)
_FTS_REL = ".repowise/source_search/source_fts_v2.db"
_INDEXED = ("src/alpha.py", "src/beta.py")


@pytest.fixture(autouse=True)
def _no_cache():
    """No test may inherit another's live-git read."""

    reset_cache_for_tests()
    yield
    reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _quiet_outbox(monkeypatch):
    """Answer the update-queue question with a clean snapshot.

    These tests are about the working tree; letting the real query run would
    add an outbox finding whose content depends on whatever database the
    environment happens to resolve.
    """

    async def snapshot(*_args, **_kwargs):
        return status_module._SourceUpdateSnapshot(None, {}, 0, None)

    monkeypatch.setattr(status_module, "_source_update_snapshot", snapshot)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=Dev", "-c", "user.email=dev@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _chunk(path: str) -> Any:
    name = Path(path).stem
    return build_symbol_chunk(
        SymbolRecord(
            symbol_id=f"{path}::{name}",
            file_path=path,
            name=name,
            qualified_name=name,
            kind="function",
            signature="",
            docstring=None,
            start_line=1,
            end_line=2,
            language="python",
        ),
        [f"def {name}():", "    return 1"],
    )


@pytest.fixture
def indexed_repo(tmp_path: Path) -> Path:
    """A committed checkout with a published corpus over exactly two files."""

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    for path in _INDEXED:
        (repo / path).write_text(f"def {Path(path).stem}():\n    return 1\n", encoding="utf-8")
    (repo / "notes.md").write_text("# not indexed\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "seed")

    with SourceFTSIndex(repo / _FTS_REL, generation=_GENERATION) as fts:
        fts.index_chunks([_chunk(path) for path in _INDEXED])
    write_manifest(
        default_manifest_path(repo),
        SourceIndexManifest(
            recipe_fingerprint="recipe-1",
            corpus_hash="corpus-1",
            symbol_chunks=2,
            file_window_chunks=0,
            files_covered=2,
            indexed_commit=_git(repo, "rev-parse", "HEAD"),
            built_at="2026-08-21T16:00:00+00:00",
            embedder=EmbedderIdentity(provider="mock", model="mock-embedder", dims=8),
            generation_id=_GENERATION.generation_id,
            generation_sequence=_GENERATION.sequence,
            fts_path=_FTS_REL,
        ),
    )
    return repo


# ---------------------------------------------------------------------------
# The live read itself
# ---------------------------------------------------------------------------


async def test_an_uncommitted_edit_to_an_indexed_file_is_seen(indexed_repo):
    (indexed_repo / "src/alpha.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")

    status = await inspect_source_index(indexed_repo)

    assert status.working_tree.checked is True
    assert status.working_tree.modified == ("src/alpha.py",)
    assert status.working_tree.deleted == ()
    # The build's own record still says nothing, which is the whole point.
    assert status.stale_files == {}


async def test_a_removed_indexed_file_is_seen_staged_or_not(indexed_repo):
    _git(indexed_repo, "rm", "-q", "src/alpha.py")
    (indexed_repo / "src/beta.py").unlink()

    status = await inspect_source_index(indexed_repo)

    assert status.working_tree.deleted == ("src/alpha.py", "src/beta.py")
    assert status.working_tree.modified == ()
    assert status.working_tree.diverged is True


async def test_a_change_outside_the_corpus_is_not_divergence(indexed_repo):
    """Disclosure has to be about the corpus, not about the checkout.

    A tool that reported every dirty file would fire on the scratch notes in
    every working directory, and an alarm that is always on is one nobody
    reads by the time a real deletion arrives.
    """

    (indexed_repo / "notes.md").write_text("# edited\n", encoding="utf-8")
    (indexed_repo / "scratch.py").write_text("x = 1\n", encoding="utf-8")

    status = await inspect_source_index(indexed_repo)

    assert status.working_tree.checked is True
    assert status.working_tree.diverged is False


async def test_a_clean_checkout_is_checked_and_clean_not_merely_silent(indexed_repo):
    status = await inspect_source_index(indexed_repo)

    assert status.working_tree.checked is True
    assert status.working_tree.diverged is False
    assert status.working_tree.unavailable_reason is None


async def test_a_new_commit_does_not_masquerade_as_working_tree_divergence(indexed_repo):
    """``index_behind_head`` already covers this, and must keep covering it alone."""

    (indexed_repo / "src/alpha.py").write_text("def alpha():\n    return 3\n", encoding="utf-8")
    _git(indexed_repo, "add", "src/alpha.py")
    _git(indexed_repo, "commit", "-qm", "advance")

    status = await inspect_source_index(indexed_repo)

    assert status.working_tree.checked is True
    assert status.working_tree.diverged is False
    assert status.indexed_commit != _git(indexed_repo, "rev-parse", "HEAD")


async def test_a_vanished_untracked_indexed_file_is_a_deletion(indexed_repo):
    """The one divergence ``git diff`` cannot express.

    An untracked file the watcher indexed has no committed side, so deleting
    it makes it invisible to git while its chunks keep serving. The build's
    record of what it read out of the working tree is the bounded candidate
    list that makes the stat probe affordable.
    """

    with SourceFTSIndex(indexed_repo / _FTS_REL, generation=_GENERATION) as fts:
        fts.index_chunks([_chunk("src/untracked.py")])
    (indexed_repo / ".repowise" / "state.json").write_text(
        '{"working_tree_paths": ["src/untracked.py"]}', encoding="utf-8"
    )

    status = await inspect_source_index(indexed_repo)

    assert status.working_tree.deleted == ("src/untracked.py",)


async def test_a_directory_that_is_not_a_repository_is_unchecked_not_clean(tmp_path):
    """An empty verdict and an unasked question must not look the same."""

    candidates, error = working_tree_candidates(tmp_path)

    assert candidates == {}
    assert error == "not_a_git_repository"
    assert divergence_from_candidates(candidates, None, error=error).checked is False


def test_unresolvable_membership_is_unchecked_rather_than_empty():
    candidates = {"src/alpha.py": DIVERGENCE_DELETED}

    assert divergence_from_candidates(candidates, None) == UNCHECKED
    assert divergence_from_candidates(candidates, set()).checked is True
    assert divergence_from_candidates(candidates, {"src/alpha.py"}).deleted == ("src/alpha.py",)


def test_the_name_status_stream_is_read_as_fields_not_records():
    """A rename is three fields, not two; misreading it corrupts the rest."""

    from repowise.core.source_search.worktree import _parse_name_status

    parsed = _parse_name_status("M\0a.py\0R100\0old.py\0new.py\0D\0gone.py\0")

    assert parsed == {
        "a.py": DIVERGENCE_MODIFIED,
        "old.py": DIVERGENCE_DELETED,
        "new.py": DIVERGENCE_MODIFIED,
        "gone.py": DIVERGENCE_DELETED,
    }


async def test_a_freshness_question_is_never_answered_from_the_cache(indexed_repo):
    """The cache exists for the search fan-out, not for the tool that asks."""

    await inspect_source_index(indexed_repo)
    (indexed_repo / "src/alpha.py").unlink()

    immediate = await inspect_source_index(indexed_repo)
    cached = await inspect_source_index(indexed_repo, working_tree_max_age=60.0)
    (indexed_repo / "src/beta.py").unlink()
    still_cached = await inspect_source_index(indexed_repo, working_tree_max_age=60.0)

    assert immediate.working_tree.deleted == ("src/alpha.py",)
    assert cached.working_tree.deleted == ("src/alpha.py",)
    assert still_cached.working_tree.deleted == ("src/alpha.py",)


# ---------------------------------------------------------------------------
# The search envelope
# ---------------------------------------------------------------------------


class _Inner:
    """A coordinator that serves whatever the test says the corpus holds."""

    def __init__(self, files: list[str]) -> None:
        self._files = files

    async def search(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "results": [{"file": path, "target_path": path} for path in self._files],
            "confidence": "confident",
            "_meta": {},
        }


def _envelope_coordinator(repo: Path, files: list[str]) -> Any:
    from repowise.server.source_search_wiring import _StatusCoordinator

    fts = SourceFTSIndex(repo / _FTS_REL, generation=_GENERATION, read_only=True)
    return _StatusCoordinator(_Inner(files), repo, SimpleNamespace(), fts), fts


async def test_the_envelope_names_the_deleted_file_it_just_served(indexed_repo):
    """A deleted file served at full confidence is the silent-consumer failure.

    The envelope has to carry it, because a consumer reading a search response
    has no reason to make a second call to a status tool before believing what
    is in front of it.
    """

    coordinator, fts = _envelope_coordinator(indexed_repo, ["src/alpha.py"])
    _git(indexed_repo, "rm", "-q", "src/alpha.py")
    try:
        response = await coordinator.search("alpha")
    finally:
        fts.close()

    working_tree = response["_meta"]["source_search"]["working_tree"]
    assert working_tree["checked"] is True
    assert working_tree["deleted"] == 1
    assert working_tree["served_deleted"] == ["src/alpha.py"]
    assert working_tree["served_modified"] == []
    # Ranking is untouched: this is disclosure beside the answer, not a
    # different answer.
    assert response["confidence"] == "confident"


async def test_the_envelope_separates_divergence_it_served_from_divergence_it_did_not(
    indexed_repo,
):
    coordinator, fts = _envelope_coordinator(indexed_repo, ["src/beta.py"])
    (indexed_repo / "src/alpha.py").write_text("def alpha():\n    return 9\n", encoding="utf-8")
    try:
        response = await coordinator.search("beta")
    finally:
        fts.close()

    working_tree = response["_meta"]["source_search"]["working_tree"]
    assert working_tree["modified"] == 1
    assert working_tree["served_modified"] == []
    assert working_tree["served_deleted"] == []


async def test_a_clean_envelope_says_it_looked(indexed_repo):
    coordinator, fts = _envelope_coordinator(indexed_repo, ["src/alpha.py"])
    try:
        response = await coordinator.search("alpha")
    finally:
        fts.close()

    assert response["_meta"]["source_search"]["working_tree"] == {
        "checked": True,
        "modified": 0,
        "deleted": 0,
        "served_modified": [],
        "served_deleted": [],
        "unavailable_reason": None,
    }
