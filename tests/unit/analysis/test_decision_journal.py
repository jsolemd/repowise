"""Durability and concurrency contract for the decision JSONL authority."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from repowise.core.analysis.decisions.journal import (
    DECISIONS_JOURNAL_ENV,
    DecisionJournal,
    DecisionJournalConfigurationError,
)


class InjectedCrash(RuntimeError):
    pass


@pytest.fixture
def journal_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".repowise").mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, ".codeatlas/decisions.jsonl")
    return repo


def _record(
    journal: DecisionJournal,
    title: str = "Keep one authority",
    *,
    decision_id: str | None = None,
):
    return journal.record(
        decision_id=decision_id,
        title=title,
        decision=f"{title} in the canonical file.",
        why="Derived stores can always be rebuilt.",
        anchors=[{"file": "src/service.py", "symbol": "VALUE"}],
    )


def test_append_preserves_untouched_line_bytes(journal_repo: Path) -> None:
    journal = DecisionJournal(journal_repo)
    file_sha = hashlib.sha256((journal_repo / "src/service.py").read_bytes()).hexdigest()
    original = {
        "id": "dec-00000001",
        "title": "Existing formatting stays",
        "decision": "Preserve reviewable diffs.",
        "why": "A projection must not rewrite its authority.",
        "anchors": [{"file": "src/service.py", "symbol": None, "file_sha": file_sha}],
        "status": "active",
        "supersedes": None,
        "superseded_by": None,
        "recorded_at": "2026-08-17T00:00:00Z",
        "confirmed_at": "2026-08-17T00:00:00Z",
    }
    journal.path.parent.mkdir(parents=True)
    first_line = (json.dumps(original, separators=(", ", ": ")) + "\n").encode()
    journal.path.write_bytes(first_line)

    _record(journal, "Append without churn")

    assert journal.path.read_bytes().startswith(first_line)
    assert len(journal.list()) == 2


def test_reader_preserves_canonical_text_values_exactly(journal_repo: Path) -> None:
    journal = DecisionJournal(journal_repo)
    file_sha = hashlib.sha256((journal_repo / "src/service.py").read_bytes()).hexdigest()
    original = {
        "id": "dec-0000000f",
        "title": "  Deliberately spaced title  ",
        "decision": "First line\n  indented second line\n",
        "why": "  Preserve authored prose.  ",
        "anchors": [{"file": "src/service.py", "symbol": None, "file_sha": file_sha}],
        "status": "active",
        "supersedes": None,
        "superseded_by": None,
        "recorded_at": "2026-08-17T00:00:00Z",
        "confirmed_at": "2026-08-17T00:00:00Z",
    }
    journal.path.parent.mkdir(parents=True)
    journal.path.write_text(json.dumps(original) + "\n", encoding="utf-8")

    decision = journal.get("dec-0000000f")
    assert decision.title == original["title"]
    assert decision.decision == original["decision"]
    assert decision.why == original["why"]


def test_crash_before_replace_leaves_canonical_file_unchanged(journal_repo: Path) -> None:
    journal = DecisionJournal(journal_repo)
    baseline = _record(journal)
    before = journal.path.read_bytes()

    def crash(stage: str) -> None:
        if stage == "before_replace":
            raise InjectedCrash(stage)

    with pytest.raises(InjectedCrash, match="before_replace"):
        journal.confirm(baseline.id, crash_hook=crash)

    assert journal.path.read_bytes() == before
    assert journal.get(baseline.id).confirmed_at == baseline.confirmed_at


def test_locked_concurrent_writers_converge_without_loss(journal_repo: Path) -> None:
    def write(index: int) -> str:
        return _record(
            DecisionJournal(journal_repo),
            f"Concurrent decision {index}",
            decision_id=f"dec-{index:08x}",
        ).id

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(write, range(16)))

    decisions = DecisionJournal(journal_repo).list()
    assert len(decisions) == 16
    assert len(set(ids)) == 16
    assert {decision.id for decision in decisions} == set(ids)


def test_nonlocking_external_edit_is_detected_and_merged(journal_repo: Path) -> None:
    journal = DecisionJournal(journal_repo)
    first = _record(journal, "First")
    file_sha = hashlib.sha256((journal_repo / "src/service.py").read_bytes()).hexdigest()
    external = {
        "anchors": [{"file": "src/service.py", "file_sha": file_sha, "symbol": None}],
        "confirmed_at": "2026-08-17T00:00:02Z",
        "decision": "An external editor added this line.",
        "id": "dec-eeeeeeee",
        "recorded_at": "2026-08-17T00:00:02Z",
        "status": "active",
        "superseded_by": None,
        "supersedes": None,
        "title": "External edit",
        "why": "The next writer must merge, not erase it.",
    }
    injected = False

    def edit_once(stage: str) -> None:
        nonlocal injected
        if stage != "before_replace" or injected:
            return
        injected = True
        with journal.path.open("ab") as handle:
            handle.write(json.dumps(external, separators=(",", ":"), sort_keys=True).encode())
            handle.write(b"\n")
            handle.flush()

    second = journal.record(
        title="Cooperating write",
        decision="Retry against the externally changed bytes.",
        why="Optimistic hash checking preserves both edits.",
        anchors=[{"file": "src/service.py", "symbol": None}],
        crash_hook=edit_once,
    )

    ids = {decision.id for decision in journal.list()}
    assert ids == {first.id, "dec-eeeeeeee", second.id}


def test_invalid_external_edit_is_never_overwritten(journal_repo: Path) -> None:
    journal = DecisionJournal(journal_repo)
    _record(journal, "Valid first row")
    with journal.path.open("ab") as handle:
        handle.write(b'{"id":"not-a-decision"}\n')
    externally_edited = journal.path.read_bytes()

    with pytest.raises(ValueError, match="id must match"):
        _record(journal, "Must not erase the invalid hand edit")

    assert journal.path.read_bytes() == externally_edited


def test_supersession_updates_both_rows_in_one_rewrite(journal_repo: Path) -> None:
    journal = DecisionJournal(journal_repo)
    old = _record(journal, "Old choice")
    new = journal.record(
        title="New choice",
        decision="Use the replacement.",
        why="The old choice no longer meets the contract.",
        anchors=[{"file": "src/service.py", "symbol": None}],
        supersedes=old.id,
    )

    by_id = {decision.id: decision for decision in journal.list()}
    assert by_id[old.id].status == "superseded"
    assert by_id[old.id].superseded_by == new.id
    assert by_id[new.id].supersedes == old.id


def test_lock_path_cannot_escape_through_repo_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    (repo / ".repowise").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, ".codeatlas/decisions.jsonl")

    with pytest.raises(DecisionJournalConfigurationError, match="lock path"):
        DecisionJournal(repo)


def test_health_reports_a_lock_held_by_another_writer(journal_repo: Path) -> None:
    import fcntl
    import os

    journal = DecisionJournal(journal_repo)
    journal.lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(journal.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        assert journal.lock_acquirable() is False
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
