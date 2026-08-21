"""Transactional wiring from authoritative symbol writes to the source outbox."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
from sqlalchemy import select

from repowise.core.ingestion import ASTParser, FileTraverser
from repowise.core.persistence.database import (
    create_engine,
    create_session_factory,
    resolve_db_url,
)
from repowise.core.persistence.models import Repository, SourceIndexUpdate
from repowise.core.pipeline.incremental import persist_incremental_index
from repowise.core.source_search import SOURCE_SEARCH_ENV
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    default_manifest_path,
    write_manifest,
)
from repowise.core.source_search.outbox import (
    PENDING,
    PUBLISHED,
    READY,
    enqueue_full_update,
    enqueue_incremental_update,
    suppress_incremental_paths,
)

_IDENTITY = EmbedderIdentity(provider="mock", model="mock", dims=8)


class _Graph:
    def __init__(self, path: str) -> None:
        self._graph = nx.DiGraph()
        self._graph.add_node(path, node_type="file")

    def graph(self):
        return self._graph

    def pagerank(self):
        return {}


def _parsed(repo: Path, path: str):
    info = next(item for item in FileTraverser(repo).traverse() if item.path == path)
    return ASTParser().parse_file(info, (repo / path).read_bytes())


async def _rows(repo: Path) -> list[SourceIndexUpdate]:
    engine = create_engine(resolve_db_url(repo))
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            return list(
                (
                    await session.execute(
                        select(SourceIndexUpdate).order_by(SourceIndexUpdate.sequence)
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()


def _diff(
    path: str,
    content_hash: str | None,
    *,
    parse_state: str = "window",
    status: str = "modified",
    old_path: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        status=status,
        old_path=old_path,
        new_parsed=None,
        parse_state=parse_state,
        content_hash=content_hash,
    )


async def _seed_active_ledger(
    repo: Path,
    batches: list[list[SimpleNamespace]],
    *,
    stale_files: dict[str, str] | None = None,
    final_full: bool = False,
    malformed_active: bool = False,
) -> SourceIndexManifest:
    from repowise.core.persistence import get_session, init_db, upsert_repository

    repo.mkdir(parents=True, exist_ok=True)
    engine = create_engine(resolve_db_url(repo))
    try:
        await init_db(engine)
        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            repository = await upsert_repository(
                session,
                name="repo",
                local_path=str(repo),
            )
            seeded: list[SourceIndexUpdate] = []
            for batch in batches:
                row = await enqueue_incremental_update(
                    session,
                    repository.id,
                    repo,
                    file_diffs=batch,
                    parsed_files=[],
                )
                assert row is not None
                seeded.append(row)
            if final_full:
                seeded.append(
                    await enqueue_full_update(
                        session,
                        repository.id,
                        repo,
                        parsed_files=[],
                    )
                )

            active = seeded[-1]
            manifest = SourceIndexManifest(
                recipe_fingerprint="recipe",
                corpus_hash="corpus",
                symbol_chunks=1,
                file_window_chunks=2,
                files_covered=3,
                indexed_commit=None,
                built_at="2026-08-21T00:00:00+00:00",
                embedder=_IDENTITY,
                generation_id=active.generation_id,
                generation_sequence=active.sequence,
                stale_files=dict(stale_files or {}),
            )
            for row in seeded:
                row.state = PUBLISHED
            active.artifact_json = json.dumps(manifest.to_dict(), sort_keys=True)
            if malformed_active:
                active.change_set_json = "{}"
            write_manifest(default_manifest_path(repo), manifest)
            return manifest
    finally:
        await engine.dispose()


async def _enqueue_diffs(
    repo: Path,
    diffs: list[SimpleNamespace],
    *,
    upstream_ready: bool = True,
    upstream_error: str | None = None,
) -> SourceIndexUpdate | None:
    from repowise.core.persistence import get_session

    engine = create_engine(resolve_db_url(repo))
    try:
        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            repository = (
                await session.execute(select(Repository).where(Repository.local_path == str(repo)))
            ).scalar_one()
            return await enqueue_incremental_update(
                session,
                repository.id,
                repo,
                file_diffs=diffs,
                parsed_files=[],
                upstream_ready=upstream_ready,
                upstream_error=upstream_error,
            )
    finally:
        await engine.dispose()


async def test_incremental_symbol_commit_enqueues_a_ready_change(tmp_path, monkeypatch):
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def run():\n    return 1\n")
    parsed = _parsed(repo, "src/app.py")
    diff = SimpleNamespace(
        path="src/app.py",
        status="modified",
        old_path=None,
        new_parsed=parsed,
    )

    await persist_incremental_index(
        repo,
        _Graph("src/app.py"),
        {},
        None,
        None,
        ["src/app.py"],
        current_graph_file_paths={"src/app.py"},
        file_diffs=[diff],
        parsed_files=[parsed],
    )

    rows = await _rows(repo)
    assert len(rows) == 1
    assert rows[0].state == "pending"
    assert rows[0].upstream_ready is True
    assert json.loads(rows[0].change_set_json)[0]["parse_state"] == "parsed"


async def test_symbol_failure_commits_a_blocked_change_instead_of_false_ready(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def run():\n    return 1\n")
    parsed = _parsed(repo, "src/app.py")
    diff = SimpleNamespace(
        path="src/app.py",
        status="modified",
        old_path=None,
        new_parsed=parsed,
    )

    from repowise.core.pipeline import persist as persist_module

    async def _fail_symbols(*args, **kwargs):
        raise RuntimeError("symbol transaction input rejected")

    monkeypatch.setattr(persist_module, "persist_incremental_symbols", _fail_symbols)
    await persist_incremental_index(
        repo,
        _Graph("src/app.py"),
        {},
        None,
        None,
        ["src/app.py"],
        current_graph_file_paths={"src/app.py"},
        file_diffs=[diff],
        parsed_files=[parsed],
    )

    rows = await _rows(repo)
    assert len(rows) == 1
    assert rows[0].state == "blocked"
    assert rows[0].upstream_ready is False
    assert "symbol transaction input rejected" in (rows[0].last_error or "")


async def test_watch_fast_lane_suppresses_only_paths_it_already_captured(tmp_path, monkeypatch):
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "fast.py").write_text("def fast():\n    return 1\n")
    (repo / "src" / "commit.py").write_text("def committed():\n    return 2\n")

    from repowise.core.persistence import get_session, init_db, upsert_repository

    engine = create_engine(resolve_db_url(repo))
    try:
        await init_db(engine)
        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            repository = await upsert_repository(session, name="repo", local_path=str(repo))
            diffs = [
                SimpleNamespace(path="src/fast.py", status="modified", old_path=None),
                SimpleNamespace(path="src/commit.py", status="modified", old_path=None),
            ]
            with suppress_incremental_paths({"src/fast.py"}):
                await enqueue_incremental_update(
                    session,
                    repository.id,
                    repo,
                    file_diffs=diffs,
                    parsed_files=[],
                )
    finally:
        await engine.dispose()

    rows = await _rows(repo)
    assert len(rows) == 1
    changes = json.loads(rows[0].change_set_json)
    assert [change["path"] for change in changes] == ["src/commit.py"]


async def test_restart_replay_of_active_shell_and_yaml_allocates_no_sequence(tmp_path):
    repo = tmp_path / "repo"
    shell = _diff("scripts/smoke.sh", "shell-hash", parse_state="window")
    yaml = _diff("docker-compose.yml", "yaml-hash", parse_state="window")
    await _seed_active_ledger(repo, [[shell], [yaml]])

    replay = await _enqueue_diffs(
        repo,
        [
            _diff(
                "scripts/smoke.sh",
                "shell-hash",
                parse_state="parsed",
                old_path="scripts/smoke.sh",
            ),
            _diff("docker-compose.yml", "yaml-hash", parse_state="parsed"),
        ],
    )

    assert replay is None
    assert len(await _rows(repo)) == 2


async def test_durable_suppression_keeps_unrelated_real_work(tmp_path):
    repo = tmp_path / "repo"
    await _seed_active_ledger(
        repo,
        [
            [_diff("scripts/smoke.sh", "shell-hash")],
            [_diff("docker-compose.yml", "yaml-hash")],
        ],
    )

    row = await _enqueue_diffs(
        repo,
        [
            _diff("scripts/smoke.sh", "shell-hash", parse_state="parsed"),
            _diff("src/real.py", "new-hash", parse_state="parsed"),
        ],
    )

    assert row is not None
    changes = json.loads(row.change_set_json)
    assert [change["path"] for change in changes] == ["src/real.py"]


async def test_different_hash_and_a_to_b_to_a_are_not_suppressed(tmp_path):
    repo = tmp_path / "repo"
    await _seed_active_ledger(repo, [[_diff("scripts/smoke.sh", "hash-b")]])

    row = await _enqueue_diffs(repo, [_diff("scripts/smoke.sh", "hash-a")])

    assert row is not None
    assert json.loads(row.change_set_json)[0]["content_hash"] == "hash-a"


@pytest.mark.parametrize("later_state", [PENDING, READY])
async def test_later_work_touching_a_candidate_prevents_suppression(
    tmp_path,
    later_state: str,
):
    repo = tmp_path / later_state
    await _seed_active_ledger(repo, [[_diff("scripts/smoke.sh", "active-hash")]])
    pending = await _enqueue_diffs(repo, [_diff("scripts/smoke.sh", "pending-hash")])
    assert pending is not None
    if later_state == READY:
        engine = create_engine(resolve_db_url(repo))
        try:
            from repowise.core.persistence import get_session

            factory = create_session_factory(engine)
            async with get_session(factory) as session:
                row = (
                    await session.execute(
                        select(SourceIndexUpdate).where(
                            SourceIndexUpdate.generation_id == pending.generation_id
                        )
                    )
                ).scalar_one()
                row.state = READY
        finally:
            await engine.dispose()

    replay = await _enqueue_diffs(repo, [_diff("scripts/smoke.sh", "active-hash")])

    assert replay is not None
    assert len(await _rows(repo)) == 3


async def test_manifest_visible_ready_row_can_witness_active_state(tmp_path):
    repo = tmp_path / "repo"
    active = await _seed_active_ledger(
        repo,
        [[_diff("scripts/smoke.sh", "shell-hash")]],
    )
    engine = create_engine(resolve_db_url(repo))
    try:
        from repowise.core.persistence import get_session

        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            row = (
                await session.execute(
                    select(SourceIndexUpdate).where(
                        SourceIndexUpdate.generation_id == active.generation_id
                    )
                )
            ).scalar_one()
            row.state = READY
    finally:
        await engine.dispose()

    replay = await _enqueue_diffs(repo, [_diff("scripts/smoke.sh", "shell-hash")])

    assert replay is None
    assert len(await _rows(repo)) == 1


async def test_coalesced_ready_rows_with_active_artifact_witness_state(tmp_path):
    repo = tmp_path / "repo"
    active = await _seed_active_ledger(
        repo,
        [
            [_diff("src/app.py", "app-hash", parse_state="parsed")],
            [_diff("infra/service.yaml", "yaml-hash")],
        ],
    )
    engine = create_engine(resolve_db_url(repo))
    try:
        from repowise.core.persistence import get_session

        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            rows = list(
                (
                    await session.execute(
                        select(SourceIndexUpdate).order_by(SourceIndexUpdate.sequence)
                    )
                )
                .scalars()
                .all()
            )
            assert len(rows) == 2
            artifact_json = json.dumps(active.to_dict(), sort_keys=True)
            for row in rows:
                row.state = READY
                row.artifact_json = artifact_json
    finally:
        await engine.dispose()

    replay = await _enqueue_diffs(
        repo,
        [_diff("src/app.py", "app-hash", parse_state="parsed")],
    )

    assert replay is None
    assert len(await _rows(repo)) == 2


async def test_coalesced_ready_row_with_mismatched_artifact_fails_open(tmp_path):
    repo = tmp_path / "repo"
    active = await _seed_active_ledger(
        repo,
        [
            [_diff("src/app.py", "app-hash", parse_state="parsed")],
            [_diff("infra/service.yaml", "yaml-hash")],
        ],
    )
    engine = create_engine(resolve_db_url(repo))
    try:
        from repowise.core.persistence import get_session

        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            rows = list(
                (
                    await session.execute(
                        select(SourceIndexUpdate).order_by(SourceIndexUpdate.sequence)
                    )
                )
                .scalars()
                .all()
            )
            assert len(rows) == 2
            rows[0].state = READY
            rows[0].artifact_json = "{}"
            rows[1].state = READY
            rows[1].artifact_json = json.dumps(active.to_dict(), sort_keys=True)
    finally:
        await engine.dispose()

    replay = await _enqueue_diffs(
        repo,
        [_diff("src/app.py", "app-hash", parse_state="parsed")],
    )

    assert replay is not None
    assert replay.sequence == 3


async def test_parser_mismatch_and_stale_manifest_fail_open(tmp_path):
    repo = tmp_path / "repo"
    await _seed_active_ledger(
        repo,
        [
            [_diff("scripts/smoke.sh", "shell-hash")],
            [_diff("docker-compose.yml", "yaml-hash")],
        ],
        stale_files={"docker-compose.yml": "latest parse failed"},
    )
    engine = create_engine(resolve_db_url(repo))
    try:
        from repowise.core.persistence import get_session

        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            first = (
                (
                    await session.execute(
                        select(SourceIndexUpdate).order_by(SourceIndexUpdate.sequence)
                    )
                )
                .scalars()
                .first()
            )
            assert first is not None
            first.parser_fingerprint = "different-parser"
    finally:
        await engine.dispose()

    row = await _enqueue_diffs(
        repo,
        [
            _diff("scripts/smoke.sh", "shell-hash"),
            _diff("docker-compose.yml", "yaml-hash"),
        ],
    )

    assert row is not None
    assert {item["path"] for item in json.loads(row.change_set_json)} == {
        "scripts/smoke.sh",
        "docker-compose.yml",
    }


async def test_non_successful_or_non_modified_candidates_fail_open(tmp_path):
    repo = tmp_path / "repo"
    await _seed_active_ledger(repo, [[_diff("src/app.py", "active-hash", parse_state="parsed")]])

    cases = [
        _diff("src/app.py", "active-hash", parse_state="failed"),
        _diff("src/app.py", "active-hash", parse_state="unindexed"),
        _diff(
            "src/app.py",
            "active-hash",
            parse_state="parsed",
            status="renamed",
            old_path="src/old.py",
        ),
        _diff("src/app.py", None, parse_state="deleted", status="deleted"),
    ]
    for candidate in cases:
        row = await _enqueue_diffs(repo, [candidate])
        assert row is not None


async def test_full_and_malformed_history_are_suppression_barriers(tmp_path):
    full_repo = tmp_path / "full"
    await _seed_active_ledger(
        full_repo,
        [[_diff("scripts/smoke.sh", "shell-hash")]],
        final_full=True,
    )
    assert await _enqueue_diffs(full_repo, [_diff("scripts/smoke.sh", "shell-hash")]) is not None

    malformed_repo = tmp_path / "malformed"
    await _seed_active_ledger(
        malformed_repo,
        [[_diff("scripts/smoke.sh", "shell-hash")]],
        malformed_active=True,
    )
    assert (
        await _enqueue_diffs(malformed_repo, [_diff("scripts/smoke.sh", "shell-hash")]) is not None
    )


async def test_manifest_movement_abandons_suppression(tmp_path, monkeypatch):
    from repowise.core.source_search import outbox as outbox_module

    repo = tmp_path / "repo"
    active = await _seed_active_ledger(
        repo,
        [[_diff("scripts/smoke.sh", "shell-hash")]],
    )
    moved = replace(
        active,
        generation_id="moved-generation",
        generation_sequence=active.generation_sequence + 1,
    )
    manifests = iter((active, moved))
    monkeypatch.setattr(outbox_module, "read_manifest", lambda _path: next(manifests))

    row = await _enqueue_diffs(repo, [_diff("scripts/smoke.sh", "shell-hash")])

    assert row is not None
    assert row.parent_generation_id == moved.generation_id


async def test_blocked_upstream_change_is_never_suppressed(tmp_path):
    repo = tmp_path / "repo"
    await _seed_active_ledger(repo, [[_diff("src/app.py", "active-hash", parse_state="parsed")]])

    row = await _enqueue_diffs(
        repo,
        [_diff("src/app.py", "active-hash", parse_state="parsed")],
        upstream_ready=False,
        upstream_error="symbol write failed",
    )

    assert row is not None
    assert row.state == "blocked"


def test_older_full_repo_parse_cannot_overwrite_a_newer_save(tmp_path):
    from repowise.core.pipeline.persist import _changed_file_symbols

    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    path = repo / "src" / "app.py"
    path.write_text("def before():\n    return 1\n")
    parsed = _parsed(repo, "src/app.py")
    path.write_text("def after():\n    return 2\n")

    reconcile_paths, symbols = _changed_file_symbols([parsed], ["src/app.py"])

    assert reconcile_paths == []
    assert symbols == []
