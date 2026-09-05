"""The decision-journal contract, as an absorption keeps or breaks it.

Journal mode moves the authority for a decision out of the database and into a
git-tracked JSONL. Every upstream release that touches decisions arrives not
knowing that, so this file states the contract in one place: what an absorption
must leave true about ids, statuses, supersession chains, lanes, and where a
write is allowed to land.

Reads ``tests/fixtures/decisions/journal_contract.jsonl`` — a small fixture with
the structural cases that matter, never a live journal. It was derived from a
real one and its prose rewritten, so the shapes are true and the content is not
anybody's decision.

Written during the v0.48.0 absorption, where upstream #2070's new lane
derivation read an acceptance table journal mode never writes and reported every
confirmed decision as an unreviewed candidate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy import select

from repowise.core.analysis.decisions.journal import (
    DECISIONS_JOURNAL_ENV,
    DecisionJournalMutationDisabledError,
)
from repowise.core.analysis.decisions.journal_projection import (
    DECISION_JOURNAL_SOURCE,
    confirm_journal_decision,
    record_journal_decision,
    refresh_decision_journal,
    supersede_journal_decision,
)
from repowise.core.persistence import crud
from repowise.core.persistence.models import DecisionRecord
from tests.unit.persistence.helpers import insert_repo

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "decisions" / "journal_contract.jsonl"


@pytest.fixture
async def journal_repo(async_session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    (root / ".codeatlas").mkdir(parents=True)
    (root / ".repowise").mkdir()
    shutil.copy(_FIXTURE, root / ".codeatlas" / "decisions.jsonl")
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, ".codeatlas/decisions.jsonl")
    repo = await insert_repo(
        async_session, name="repo", local_path=str(root), url="https://example.test/repo"
    )
    return repo, root


def _rows(root: Path) -> list[dict]:
    text = (root / ".codeatlas" / "decisions.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


async def _projected(async_session, repo_id: str) -> dict[str, DecisionRecord]:
    result = await async_session.execute(
        select(DecisionRecord).where(DecisionRecord.repository_id == repo_id)
    )
    return {record.id: record for record in result.scalars().all()}


async def test_ids_and_chains_survive_the_projection(
    async_session, journal_repo, in_memory_vector_store
) -> None:
    """Nothing re-keys a journal row, and no link in a chain is rewritten."""
    repo, root = journal_repo
    await refresh_decision_journal(
        async_session, repo.id, repo_root=root, vector_store=in_memory_vector_store
    )
    rows = _rows(root)
    projected = await _projected(async_session, repo.id)

    assert set(projected) == {row["id"] for row in rows}
    for row in rows:
        record = projected[row["id"]]
        assert record.id == row["id"]
        assert record.title == row["title"]
        assert record.supersedes == (row.get("supersedes") or None)
        assert record.superseded_by == (row.get("superseded_by") or None)
        assert record.source == DECISION_JOURNAL_SOURCE

    # Upstream derives an id from the identity a decision dedupes on. A journal
    # row must never be re-keyed to it on read: the id is in git, in `get_why`
    # answers, and in the decisions the fork has already written down.
    from repowise.core.persistence.crud.decisions import derive_decision_id

    for row in rows:
        derived = derive_decision_id(
            repo.id, row["title"], source=DECISION_JOURNAL_SOURCE, evidence_file=None
        )
        assert derived != row["id"], f"{row['id']} was re-keyed to {derived}"


async def test_an_id_that_names_a_live_row_is_never_redirected(
    async_session, journal_repo, in_memory_vector_store
) -> None:
    """`get_why id=dec-...` and every by-id route resolve to the row asked for."""
    repo, root = journal_repo
    await refresh_decision_journal(
        async_session, repo.id, repo_root=root, vector_store=in_memory_vector_store
    )
    from repowise.core.persistence.crud.authority import resolve_decision_id
    from repowise.server.routers.decisions import _live_decision_id

    for row in _rows(root):
        assert (await crud.get_decision(async_session, row["id"])).id == row["id"]
        assert await _live_decision_id(async_session, row["id"]) == row["id"]
        assert await resolve_decision_id(async_session, row["id"]) in (None, row["id"])


async def test_a_confirmed_row_governs_and_an_unconfirmed_one_does_not(
    async_session, journal_repo, in_memory_vector_store
) -> None:
    """The lane split, on the authority journal mode actually keeps.

    Upstream derives a lane from `decision_acceptances`, which journal mode never
    writes. `confirmed_at` is the projection of the journal's own confirmation
    and the projector is its only writer, so it is that row's equivalent — and
    without it every ratified decision reads to a model as an open candidate.
    """
    repo, root = journal_repo
    await refresh_decision_journal(
        async_session, repo.id, repo_root=root, vector_store=in_memory_vector_store
    )
    from repowise.core.analysis.decisions.lifecycle import is_governing
    from repowise.core.persistence.crud.authority import decision_currencies

    projected = await _projected(async_session, repo.id)
    currencies = await decision_currencies(async_session, repo.id, list(projected.values()))

    confirmed = {row["id"] for row in _rows(root) if row.get("confirmed_at")}
    assert confirmed, "fixture carries no confirmed row"
    for record_id in confirmed:
        assert currencies.get(record_id) is not None, f"{record_id} read as a candidate"
    for record_id in set(projected) - confirmed:
        assert currencies.get(record_id) is None

    # A confirmed, un-superseded row still binds; a superseded one is history.
    governing = {i for i, c in currencies.items() if is_governing(c)}
    assert governing
    assert all(projected[i].status != "superseded" for i in governing)


async def test_record_confirm_supersede_round_trips(
    async_session, journal_repo, in_memory_vector_store
) -> None:
    repo, root = journal_repo
    await refresh_decision_journal(
        async_session, repo.id, repo_root=root, vector_store=in_memory_vector_store
    )
    before = {row["id"] for row in _rows(root)}

    proposal = await record_journal_decision(
        async_session,
        repo.id,
        title="Contract probe",
        decision="Probe the journal seam",
        why="An absorption must leave this round trip working",
        anchors=[{"file": ".codeatlas/decisions.jsonl", "symbol": None}],
        confirmed=False,
        vector_store=in_memory_vector_store,
    )
    assert proposal.status == "proposed" and proposal.id.startswith("dec-")

    confirmed = await confirm_journal_decision(
        async_session, repo.id, proposal.id, vector_store=in_memory_vector_store
    )
    assert confirmed.status == "active"

    successor = await record_journal_decision(
        async_session,
        repo.id,
        title="Contract probe successor",
        decision="Replace the probe",
        why="The supersession leg of the same round trip",
        anchors=[{"file": ".codeatlas/decisions.jsonl", "symbol": None}],
        vector_store=in_memory_vector_store,
    )
    retired = await supersede_journal_decision(
        async_session,
        repo.id,
        proposal.id,
        superseded_by=successor.id,
        vector_store=in_memory_vector_store,
    )
    assert retired.status == "superseded"
    assert retired.superseded_by == successor.id
    assert before <= {row["id"] for row in _rows(root)}


async def test_machine_writes_stay_disabled_under_the_flag(async_session, journal_repo) -> None:
    from repowise.core.persistence.crud.decisions import (
        _journal_mode_enabled,
        _reject_sqlite_only_mutation,
    )

    assert _journal_mode_enabled() is True
    with pytest.raises(DecisionJournalMutationDisabledError):
        _reject_sqlite_only_mutation("upsert_decision")


async def test_no_write_escapes_the_journal(
    async_session, journal_repo, in_memory_vector_store
) -> None:
    """Acceptance events, manifests and review queues stay inside the JSONL.

    Upstream v0.48.0 added all three. Under the flag the journal is the only
    file a decision write may touch, or the authority has quietly moved.
    """
    repo, root = journal_repo
    await refresh_decision_journal(
        async_session, repo.id, repo_root=root, vector_store=in_memory_vector_store
    )
    before_files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    before_rows = len(_rows(root))

    record = await record_journal_decision(
        async_session,
        repo.id,
        title="Containment probe",
        decision="Record and confirm",
        why="Prove no write lands outside the journal",
        anchors=[{"file": ".codeatlas/decisions.jsonl", "symbol": None}],
        confirmed=False,
        vector_store=in_memory_vector_store,
    )
    await confirm_journal_decision(
        async_session, repo.id, record.id, vector_store=in_memory_vector_store
    )

    after_files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert sorted(set(after_files) - set(before_files)) == []
    assert len(_rows(root)) == before_rows + 1
