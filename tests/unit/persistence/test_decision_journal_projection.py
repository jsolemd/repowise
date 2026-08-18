"""Projection and write-through contract for the canonical decision journal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import delete, select

from repowise.core.analysis.decisions.evolution import (
    detect_supersessions_and_conflicts,
    run_update_evolution,
)
from repowise.core.analysis.decisions.journal import (
    DECISIONS_JOURNAL_ENV,
    DecisionJournal,
    DecisionJournalMutationDisabledError,
)
from repowise.core.analysis.decisions.journal_projection import (
    DECISION_JOURNAL_SOURCE,
    confirm_journal_decision,
    record_journal_decision,
    refresh_decision_journal,
)
from repowise.core.analysis.decisions.semantic_match import DECISION_VECTOR_PREFIX
from repowise.core.persistence import crud
from repowise.core.persistence.models import (
    DecisionEdge,
    DecisionNodeLink,
    DecisionRecord,
)
from tests.unit.persistence.helpers import insert_repo


class InjectedCrash(RuntimeError):
    pass


class ExplodingProvider:
    async def complete(self, *args, **kwargs):
        raise AssertionError("journal mode must not make an LLM call")


@pytest.fixture
async def journal_projection_repo(
    async_session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "repository"
    root.mkdir()
    (root / ".repowise").mkdir()
    (root / "src").mkdir()
    (root / "src" / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "replacement.py").write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, ".codeatlas/decisions.jsonl")
    repo = await insert_repo(
        async_session,
        name="journal-repo",
        local_path=str(root),
        url="https://example.test/journal-repo",
    )
    return repo, root


def _row(record: DecisionRecord) -> dict[str, object]:
    """Serialize authority-backed fields for a rebuild-parity assertion."""

    return {
        "id": record.id,
        "title": record.title,
        "decision": record.decision,
        "rationale": record.rationale,
        "status": record.status,
        "anchors": json.loads(record.anchors_json),
        "affected_files": json.loads(record.affected_files_json),
        "supersedes": record.supersedes,
        "superseded_by": record.superseded_by,
        "created_at": record.created_at.isoformat(),
        "confirmed_at": (
            record.confirmed_at.isoformat() if record.confirmed_at is not None else None
        ),
        "staleness_score": record.staleness_score,
    }


async def _projected_rows(async_session, repo_id: str) -> list[DecisionRecord]:
    result = await async_session.execute(
        select(DecisionRecord)
        .where(
            DecisionRecord.repository_id == repo_id,
            DecisionRecord.source == DECISION_JOURNAL_SOURCE,
        )
        .order_by(DecisionRecord.id)
    )
    return list(result.scalars().all())


async def test_refresh_is_exact_idempotent_and_rebuilds_from_jsonl(
    async_session,
    journal_projection_repo,
    in_memory_vector_store,
) -> None:
    repo, root = journal_projection_repo
    repo_id = repo.id
    journal = DecisionJournal(root)
    old = journal.record(
        decision_id="dec-00000001",
        title="Keep the journal canonical",
        decision="Project decisions out of the tracked JSONL file.",
        why="A rebuild must never lose governance history.",
        anchors=[{"file": "src/service.py", "symbol": "VALUE"}],
    )
    new = journal.record(
        decision_id="dec-00000002",
        title="Use a replacement implementation",
        decision="Move the implementation to replacement.py.",
        why="The replacement has the corrected contract.",
        anchors=[{"file": "src/replacement.py", "symbol": None}],
        supersedes=old.id,
    )

    first_health = await refresh_decision_journal(
        async_session,
        repo_id,
        vector_store=in_memory_vector_store,
    )
    assert first_health is not None
    assert first_health.projected_count == 2
    assert first_health.lock_acquirable is True
    first = [_row(record) for record in await _projected_rows(async_session, repo_id)]

    # A no-change refresh does not duplicate records, links, edges, or vectors.
    second_health = await refresh_decision_journal(
        async_session,
        repo_id,
        vector_store=in_memory_vector_store,
    )
    second = [_row(record) for record in await _projected_rows(async_session, repo_id)]
    assert second == first
    assert second_health is not None
    assert second_health.content_hash == first_health.content_hash
    assert set(await in_memory_vector_store.list_page_ids()) == {
        f"{DECISION_VECTOR_PREFIX}{old.id}",
        f"{DECISION_VECTOR_PREFIX}{new.id}",
    }

    # The content hash alone cannot prove the derived vector store survived a
    # rebuild. Missing canonical vectors are restored and stale ones removed.
    expected_vector_ids = set(await in_memory_vector_store.list_page_ids())
    await in_memory_vector_store.delete_many(sorted(expected_vector_ids))
    await in_memory_vector_store.embed_and_upsert(
        f"{DECISION_VECTOR_PREFIX}dec-deadbeef",
        "stale decision vector",
        {"title": "stale", "page_type": "decision_record"},
    )
    await refresh_decision_journal(
        async_session,
        repo_id,
        vector_store=in_memory_vector_store,
    )
    assert set(await in_memory_vector_store.list_page_ids()) == expected_vector_ids

    links = list(
        (
            await async_session.execute(
                select(DecisionNodeLink).where(DecisionNodeLink.repository_id == repo_id)
            )
        )
        .scalars()
        .all()
    )
    edges = list(
        (
            await async_session.execute(
                select(DecisionEdge).where(DecisionEdge.repository_id == repo_id)
            )
        )
        .scalars()
        .all()
    )
    assert {(link.decision_id, link.node_id) for link in links} == {
        (old.id, "src/service.py"),
        (new.id, "src/replacement.py"),
    }
    assert [(edge.src_decision_id, edge.dst_decision_id, edge.kind) for edge in edges] == [
        (new.id, old.id, "supersedes")
    ]

    # Simulate a derived-store loss. The next refresh reconstructs the exact set.
    await async_session.execute(delete(DecisionEdge).where(DecisionEdge.repository_id == repo_id))
    await async_session.execute(
        delete(DecisionNodeLink).where(DecisionNodeLink.repository_id == repo_id)
    )
    await async_session.execute(
        delete(DecisionRecord).where(
            DecisionRecord.repository_id == repo_id,
            DecisionRecord.source == DECISION_JOURNAL_SOURCE,
        )
    )
    await async_session.flush()
    async_session.expire_all()

    rebuilt_health = await refresh_decision_journal(
        async_session,
        repo_id,
        vector_store=in_memory_vector_store,
    )
    rebuilt = [_row(record) for record in await _projected_rows(async_session, repo_id)]
    assert rebuilt == first
    assert rebuilt_health is not None
    assert rebuilt_health.projected_count == len(journal.list()) == 2


async def test_refresh_recomputes_hash_staleness_and_confirm_restamps(
    async_session,
    journal_projection_repo,
) -> None:
    repo, root = journal_projection_repo
    repo_id = repo.id
    rec = await record_journal_decision(
        async_session,
        repo_id,
        title="Anchor the implementation",
        decision="Tie the rule to the exact implementation bytes.",
        why="Changes should visibly stale the decision.",
        anchors=[{"file": "src/service.py", "symbol": "VALUE"}],
    )
    decision_id = rec.id
    assert rec.staleness_score == 0.0

    (root / "src" / "service.py").write_text("VALUE = 99\n", encoding="utf-8")
    await refresh_decision_journal(async_session, repo_id)
    async_session.expire_all()
    stale = await async_session.get(DecisionRecord, decision_id)
    assert stale is not None
    assert stale.staleness_score == 1.0

    confirmed = await confirm_journal_decision(async_session, repo_id, decision_id)
    assert confirmed.staleness_score == 0.0
    anchor = json.loads(confirmed.anchors_json)[0]
    assert anchor["file_sha"] == DecisionJournal(root).get(decision_id).anchors[0].file_sha


async def test_exact_ids_survive_duplicate_title_and_anchor_shape(
    async_session,
    journal_projection_repo,
) -> None:
    repo, root = journal_projection_repo
    journal = DecisionJournal(root)
    for decision_id, decision_text in (
        ("dec-0000000a", "First independently identified rule."),
        ("dec-0000000b", "Second independently identified rule."),
    ):
        journal.record(
            decision_id=decision_id,
            title="Shared title",
            decision=decision_text,
            why="The canonical id, not a lossy natural key, owns identity.",
            anchors=[{"file": "src/service.py", "symbol": None}],
        )

    await refresh_decision_journal(async_session, repo.id)
    projected = await _projected_rows(async_session, repo.id)

    assert [record.id for record in projected] == ["dec-0000000a", "dec-0000000b"]
    assert {record.decision for record in projected} == {
        "First independently identified rule.",
        "Second independently identified rule.",
    }
    assert all(record.evidence_file is None for record in projected)


async def test_crash_before_rename_changes_neither_authority_nor_projection(
    async_session,
    journal_projection_repo,
) -> None:
    repo, root = journal_projection_repo
    journal_path = root / ".codeatlas" / "decisions.jsonl"

    def crash(stage: str) -> None:
        if stage == "before_replace":
            raise InjectedCrash(stage)

    with pytest.raises(InjectedCrash, match="before_replace"):
        await record_journal_decision(
            async_session,
            repo.id,
            title="Never partially write",
            decision="Rename only after the temporary file is durable.",
            why="A killed process must leave valid JSONL.",
            anchors=[{"file": "src/service.py", "symbol": None}],
            crash_hook=crash,
        )

    assert not journal_path.exists()
    assert await _projected_rows(async_session, repo.id) == []


async def test_crash_after_rename_heals_projection_on_next_refresh(
    async_session,
    journal_projection_repo,
) -> None:
    repo, root = journal_projection_repo

    def crash(stage: str) -> None:
        if stage == "after_rename":
            raise InjectedCrash(stage)

    with pytest.raises(InjectedCrash, match="after_rename"):
        await record_journal_decision(
            async_session,
            repo.id,
            title="Heal after durable rename",
            decision="Treat the database as a disposable projection.",
            why="A read can finish projection after a killed writer.",
            anchors=[{"file": "src/service.py", "symbol": None}],
            crash_hook=crash,
        )

    durable = DecisionJournal(root).list()
    assert len(durable) == 1
    assert await async_session.get(DecisionRecord, durable[0].id) is None

    await refresh_decision_journal(async_session, repo.id)
    projected = await async_session.get(DecisionRecord, durable[0].id)
    assert projected is not None
    assert projected.decision == durable[0].decision


async def test_machine_candidates_are_disabled_and_raw_crud_is_guarded(
    async_session,
    journal_projection_repo,
) -> None:
    repo, root = journal_projection_repo
    canonical = DecisionJournal(root).record(
        decision_id="dec-1234abcd",
        title="Curate decisions",
        decision="Do not let extractors write governance.",
        why="Machine candidates are not confirmed decisions.",
        anchors=[{"file": "src/service.py", "symbol": None}],
    )

    touched = await crud.bulk_upsert_decisions(
        async_session,
        repo.id,
        [
            {
                "title": "Machine proposal",
                "decision": "This must not be persisted.",
                "source": "inline_marker",
                "status": "proposed",
                "affected_files": ["src/service.py"],
            }
        ],
    )
    assert touched == []
    assert [record.id for record in await _projected_rows(async_session, repo.id)] == [canonical.id]

    with pytest.raises(DecisionJournalMutationDisabledError, match="canonical journal"):
        await crud.upsert_decision(
            async_session,
            repository_id=repo.id,
            title="SQLite-only mutation",
        )
    guarded_mutations = (
        crud.update_decision_metadata(
            async_session,
            canonical.id,
            affected_files=["src/replacement.py"],
        ),
        crud.update_decision_status(async_session, canonical.id, "deprecated"),
        crud.update_decision_by_id(
            async_session,
            canonical.id,
            title="SQLite-only title",
        ),
        crud.delete_decision(async_session, canonical.id),
    )
    for mutation in guarded_mutations:
        with pytest.raises(DecisionJournalMutationDisabledError, match="canonical journal"):
            await mutation


async def test_staleness_never_follows_a_changed_symlink_outside_the_repo(
    async_session,
    journal_projection_repo,
) -> None:
    repo, root = journal_projection_repo
    outside = root.parent / "outside.txt"
    outside.write_text("do not read beyond the repository\n", encoding="utf-8")
    (root / "src" / "escape.txt").symlink_to(outside)
    outside_sha = hashlib.sha256(outside.read_bytes()).hexdigest()
    journal_path = root / ".codeatlas" / "decisions.jsonl"
    journal_path.parent.mkdir()
    journal_path.write_text(
        json.dumps(
            {
                "id": "dec-bad0cafe",
                "title": "Unsafe external anchor",
                "decision": "Never follow this symlink.",
                "why": "Repository anchors cannot authorize outside reads.",
                "anchors": [
                    {
                        "file": "src/escape.txt",
                        "symbol": None,
                        "file_sha": outside_sha,
                    }
                ],
                "status": "active",
                "supersedes": None,
                "superseded_by": None,
                "recorded_at": "2026-08-17T00:00:00Z",
                "confirmed_at": "2026-08-17T00:00:00Z",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    await refresh_decision_journal(async_session, repo.id)
    record = await async_session.get(DecisionRecord, "dec-bad0cafe")
    assert record is not None
    assert record.staleness_score == 1.0


async def test_automatic_evolution_is_disabled_before_any_provider_call(
    async_session,
    journal_projection_repo,
) -> None:
    repo, _root = journal_projection_repo
    provider = ExplodingProvider()

    detected = await detect_supersessions_and_conflicts(
        async_session,
        repo.id,
        touched_ids=["dec-00000001"],
        vector_store=object(),
        provider=provider,
    )
    evolved = await run_update_evolution(
        async_session,
        repo.id,
        changed_files={"src/service.py"},
        evidence_by_file={"src/service.py": "changed"},
        provider=provider,
    )

    assert detected == {"supersedes": 0, "conflicts": 0, "flipped": 0}
    assert evolved == {
        "regen_files": set(),
        "superseded": 0,
        "amended": 0,
        "reaffirmed": 0,
    }
