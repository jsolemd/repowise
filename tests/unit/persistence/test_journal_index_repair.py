"""Journal authority must survive ordinary indexing's native SQL repairs."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from repowise.core.analysis.decisions.journal import DECISIONS_JOURNAL_ENV
from repowise.core.analysis.decisions.journal_projection import refresh_decision_journal
from repowise.core.persistence.database import create_engine, create_session_factory, init_db
from repowise.core.persistence.models import (
    DecisionAcceptance,
    DecisionCandidateMeta,
    DecisionEdge,
    DecisionEvidence,
    DecisionNodeLink,
    DecisionRecord,
    GraphNode,
    Repository,
)

FIELDS = (
    "id",
    "title",
    "decision",
    "rationale",
    "status",
    "supersedes",
    "superseded_by",
    "confirmed_at",
    "anchors_json",
    "affected_files_json",
    "affected_modules_json",
    "identity_quote",
    "scope_basis",
    "confidence",
    "verification",
)


@pytest.mark.parametrize(
    "repair",
    [
        "persist_analysis",
        "identity",
        "classification",
        "scope_prune",
        "links",
        "identity_plan",
        "classification_plan",
        "ranks",
        "confidence",
        "unretire",
        "purge_scope",
        "purge_source",
        "dedupe_plan",
    ],
)
@pytest.mark.parametrize("ambient_journal_mode", [True, False])
def test_native_index_repairs_preserve_journal_authority(
    tmp_path, monkeypatch, repair, ambient_journal_mode
):
    root = tmp_path / "repo"
    (root / ".codeatlas").mkdir(parents=True)
    src = Path(__file__).resolve().parents[2] / "fixtures/decisions/journal_contract.jsonl"
    journal = root / ".codeatlas/decisions.jsonl"
    shutil.copy2(src, journal)
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, ".codeatlas/decisions.jsonl")

    async def run():
        engine = create_engine("sqlite+aiosqlite:///" + str(tmp_path / "test.db"))
        try:
            await init_db(engine)
            factory = create_session_factory(engine)
            async with factory() as session:
                repo = Repository(id="journal-repair", name="repo", local_path=str(root))
                session.add(repo)
                await session.flush()
                await refresh_decision_journal(session, repo.id, repo_root=root)
                # Legacy native evidence must not become authority for a journal row.
                session.add(
                    DecisionEvidence(
                        id="legacy-evidence",
                        decision_id="dec-e19eb656",
                        source="session",
                        source_rank=0,
                        source_quote="A machine-derived replacement",
                        confidence=0.1,
                        verification="unverified",
                    )
                )
                if repair == "unretire":
                    edge = (
                        await session.scalars(
                            select(DecisionEdge).where(
                                DecisionEdge.src_decision_id == "dec-03d1785f",
                                DecisionEdge.dst_decision_id == "dec-713b1361",
                                DecisionEdge.kind == "supersedes",
                            )
                        )
                    ).one()
                    edge.evidence = "auto-detected: legacy detector"
                if repair in {"purge_scope", "purge_source"}:
                    # Native proposed rows can predate a projection refresh.
                    rec = await session.get(DecisionRecord, "dec-b2fb9e69")
                    rec.status = "proposed"
                    rec.evidence_file = "outside-index.md"
                for n in range(51):
                    session.add(
                        GraphNode(
                            id=f"file-{n}",
                            repository_id=repo.id,
                            node_id=f"src/unrelated-{n}.py",
                            node_type="file",
                        )
                    )
                await session.flush()

                async def rows():
                    records = (
                        await session.scalars(
                            select(DecisionRecord).where(DecisionRecord.repository_id == repo.id)
                        )
                    ).all()
                    return {r.id: tuple(getattr(r, f, None) for f in FIELDS) for r in records}

                async def projections():
                    return {
                        model.__tablename__: sorted(
                            (
                                tuple(getattr(row, c.name) for c in model.__table__.columns)
                                for row in (await session.scalars(select(model))).all()
                            ),
                            key=repr,
                        )
                        for model in (DecisionNodeLink, DecisionEdge, DecisionEvidence)
                    }

                before = await rows()
                before_projections = await projections()
                original = journal.read_bytes()
                if not ambient_journal_mode:
                    monkeypatch.delenv(DECISIONS_JOURNAL_ENV)
                from repowise.core.persistence import decision_migration as dm
                from repowise.core.persistence.crud import decisions as crud
                from repowise.core.persistence.decision_id_migration import (
                    IdMigrationPlan,
                    IdRowPlan,
                    apply_id_migration,
                )

                if repair == "persist_analysis":
                    from repowise.core.pipeline.persist import persist_analysis

                    await persist_analysis(
                        SimpleNamespace(
                            dead_code_report=None,
                            health_report=None,
                            decision_report=None,
                            git_metadata_list=[],
                            vector_store=None,
                        ),
                        session,
                        repo.id,
                    )
                elif repair == "identity":
                    await apply_id_migration(session, repo.id)
                elif repair == "identity_plan":
                    await apply_id_migration(
                        session,
                        repo.id,
                        plan=IdMigrationPlan(
                            rows=[IdRowPlan("dec-e19eb656", "native-id", "changed", "rewrite")],
                            pinned_quotes={"dec-e19eb656": "changed"},
                        ),
                    )
                elif repair == "classification_plan":
                    await dm.apply_migration(
                        session,
                        repo.id,
                        plan=dm.MigrationPlan(
                            rows=[
                                dm.RowPlan(
                                    "dec-e19eb656",
                                    "changed",
                                    "active",
                                    "journal",
                                    "candidate",
                                    "stale plan",
                                )
                            ]
                        ),
                    )
                elif repair == "dedupe_plan":
                    from repowise.core.persistence.decision_dedupe import (
                        DedupePlan,
                        FoldPlan,
                        apply_dedupe,
                    )

                    await apply_dedupe(
                        session,
                        repo.id,
                        vector_store=None,
                        plan=DedupePlan(
                            clusters=[
                                FoldPlan(
                                    canonical_id="dec-03d1785f",
                                    canonical_title="changed",
                                    folded=[("dec-b2fb9e69", "changed", 0.99)],
                                )
                            ]
                        ),
                    )
                elif repair == "ranks":
                    await crud.reconcile_source_ranks(session)
                elif repair == "confidence":
                    await crud.reconcile_decision_confidence(session)
                elif repair == "unretire":
                    await crud.unretire_auto_superseded(session)
                elif repair == "purge_scope":
                    await crud.purge_proposed_decisions_outside_files(session, repo.id, set())
                elif repair == "purge_source":
                    await crud.purge_proposed_decisions_by_source(session, repo.id, "journal")
                elif repair == "classification":
                    await dm.apply_migration(session, repo.id)
                elif repair == "scope_prune":
                    await dm.prune_unindexed_scope_files(session, repo.id)
                elif repair == "links":
                    await dm.backfill_decision_node_links(session, repo.id)
                await session.flush()
                assert await rows() == before, repair
                assert await projections() == before_projections, repair
                assert (await session.scalars(select(DecisionAcceptance))).all() == []
                assert (await session.scalars(select(DecisionCandidateMeta))).all() == []
                assert journal.read_bytes() == original
        finally:
            await engine.dispose()

    asyncio.run(run())
