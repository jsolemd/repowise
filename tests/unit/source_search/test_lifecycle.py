"""Lifecycle invariants for the generation-published source corpus."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import FileLock
from sqlalchemy import delete, select

from repowise.core.ingestion import ASTParser, FileTraverser
from repowise.core.persistence.database import (
    create_engine,
    create_session_factory,
    get_session,
    init_db,
    resolve_db_url,
)
from repowise.core.persistence.models import Repository, SourceIndexUpdate, WikiSymbol
from repowise.core.pipeline.persist import persist_incremental_symbols
from repowise.core.providers.embedding.base import MockEmbedder
from repowise.core.source_search.fast_update import capture_source_changes
from repowise.core.source_search.fts import SourceFTSIndex
from repowise.core.source_search.generation import GenerationRef
from repowise.core.source_search.indexer import build_source_index
from repowise.core.source_search.lifecycle import (
    SourceIndexDeferredError,
    _select_plan,
    reconcile_source_index,
    record_source_index_error,
)
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    default_manifest_path,
    read_manifest,
)
from repowise.core.source_search.outbox import (
    PENDING,
    PUBLISHED,
    READY,
    SourceChange,
    SourceUpdateRecord,
    enqueue_incremental_update,
)
from repowise.core.source_search.status import inspect_source_index
from repowise.core.source_search.vector_store import SourceChunkVectorStore

pytest.importorskip("lancedb")

_IDENTITY = EmbedderIdentity(provider="mock", model="MockEmbedder", dims=8)
_APP_V1 = '''def alpha():
    return "oldquasar"


def stable():
    return "stablecomet"
'''
_APP_V2 = _APP_V1.replace("oldquasar", "newnebula")
_APP_V3 = _APP_V1.replace("oldquasar", "finalpulsar")


def test_manifest_absent_plan_is_a_full_build_without_stale_file_access() -> None:
    update = SourceUpdateRecord(
        sequence=1,
        generation_id="first",
        parent_generation_id=None,
        mode="incremental",
        state=PENDING,
        parser_fingerprint="parser",
        changes=(SourceChange("src/app.py", "modified", None, "hash", "parsed"),),
        upstream_ready=True,
        attempts=0,
        last_error=None,
        artifact=None,
    )

    plan = _select_plan([update], None, "recipe")

    assert plan is not None
    assert plan.full is True
    assert plan.stale_files == {}


class _WideEmbedder(MockEmbedder):
    dimensions = 16

    async def embed(self, texts: list[str]) -> list[list[float]]:
        narrow = await super().embed(texts)
        return [vector + vector for vector in narrow]


class _CountingEmbedder(MockEmbedder):
    def __init__(self) -> None:
        self.texts = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts += len(texts)
        return await super().embed(texts)


class _Crash(BaseException):
    pass


class _WikiVectors:
    def __init__(self) -> None:
        self._embedder = MockEmbedder()

    async def search_by_vector(self, vector, limit=10):
        return []


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _parse(repo: Path, path: str):
    info = next(item for item in FileTraverser(repo).traverse() if item.path == path)
    return ASTParser().parse_file(info, (repo / path).read_bytes())


async def _factory(repo: Path):
    engine = create_engine(resolve_db_url(repo))
    await init_db(engine)
    return engine, create_session_factory(engine)


async def _replace_symbols(session, repo_id: str, repo: Path, path: str) -> object:
    parsed = _parse(repo, path)
    await persist_incremental_symbols(session, repo_id, [parsed], [path])
    return parsed


async def _capture(
    repo: Path,
    *,
    path: str,
    status: str = "modified",
    old_path: str | None = None,
    parse: bool = True,
    mutate_symbols: bool = True,
) -> None:
    engine, factory = await _factory(repo)
    try:
        async with get_session(factory) as session:
            repository = (
                await session.execute(select(Repository).where(Repository.local_path == str(repo)))
            ).scalar_one()
            parsed = _parse(repo, path) if parse and status != "deleted" else None
            if old_path:
                await session.execute(
                    delete(WikiSymbol).where(
                        WikiSymbol.repository_id == repository.id,
                        WikiSymbol.file_path == old_path,
                    )
                )
            if status == "deleted":
                await session.execute(
                    delete(WikiSymbol).where(
                        WikiSymbol.repository_id == repository.id,
                        WikiSymbol.file_path == path,
                    )
                )
            elif mutate_symbols and parsed is not None:
                await persist_incremental_symbols(session, repository.id, [parsed], [path])
            diff = SimpleNamespace(
                path=path,
                status=status,
                old_path=old_path,
                new_parsed=parsed,
            )
            await enqueue_incremental_update(
                session,
                repository.id,
                repo,
                file_diffs=[diff],
                parsed_files=[parsed] if parsed is not None else [],
            )
    finally:
        await engine.dispose()


async def _updates(repo: Path) -> list[SourceIndexUpdate]:
    engine, factory = await _factory(repo)
    try:
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


def _fts(repo: Path, manifest) -> SourceFTSIndex:
    return SourceFTSIndex(
        repo / manifest.fts_path,
        generation=GenerationRef(manifest.generation_id, manifest.generation_sequence),
    )


def _vector(repo: Path, manifest, embedder=None) -> SourceChunkVectorStore:
    return SourceChunkVectorStore(
        str(repo / ".repowise" / "lancedb"),
        embedder=embedder or MockEmbedder(),
        table_name=manifest.lance_table,
        generation=GenerationRef(manifest.generation_id, manifest.generation_sequence),
    )


@pytest.fixture
async def lifecycle_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "infra").mkdir()
    (repo / "src" / "app.py").write_text(_APP_V1)
    (repo / "infra" / "service.yaml").write_text("service: durable_marker\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", "-A")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "seed")

    engine, factory = await _factory(repo)
    try:
        async with get_session(factory) as session:
            repository = Repository(id="r1", name="repo", local_path=str(repo))
            session.add(repository)
            await session.flush()
            await _replace_symbols(session, repository.id, repo, "src/app.py")
    finally:
        await engine.dispose()

    initial = await reconcile_source_index(
        repo,
        embedder=MockEmbedder(),
        embedder_identity=_IDENTITY,
        force_full=True,
    )
    assert initial.status == "published"
    return repo


@pytest.fixture
async def legacy_repo(tmp_path):
    """The exact A1 pairing: sequence-0 manifest + unversioned Lance table."""

    repo = tmp_path / "legacy-repo"
    (repo / "src").mkdir(parents=True)
    (repo / "infra").mkdir()
    (repo / "src" / "app.py").write_text(_APP_V1)
    (repo / "infra" / "service.yaml").write_text("service: durablecomet\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", "-A")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "seed")

    engine, factory = await _factory(repo)
    try:
        async with get_session(factory) as session:
            repository = Repository(id="r1", name="legacy-repo", local_path=str(repo))
            session.add(repository)
            await session.flush()
            await _replace_symbols(session, repository.id, repo, "src/app.py")
    finally:
        await engine.dispose()
    await build_source_index(
        repo,
        embedder=MockEmbedder(),
        embedder_identity=_IDENTITY,
    )
    manifest = read_manifest(default_manifest_path(repo))
    assert manifest is not None and manifest.generation_sequence == 0
    return repo


async def test_incremental_edit_is_atomically_visible_and_idempotent(lifecycle_repo):
    repo = lifecycle_repo
    old = read_manifest(default_manifest_path(repo))
    assert old is not None

    (repo / "src" / "app.py").write_text(_APP_V2)
    await _capture(repo, path="src/app.py")
    counter = _CountingEmbedder()
    result = await reconcile_source_index(
        repo, embedder=counter, embedder_identity=_IDENTITY
    )
    current = read_manifest(default_manifest_path(repo))
    assert current is not None
    assert current.generation_sequence > old.generation_sequence
    assert result.embedded == 1
    assert result.reused == 1
    assert counter.texts == 1

    with _fts(repo, old) as old_fts, _fts(repo, current) as current_fts:
        assert old_fts.query("oldquasar")
        assert not old_fts.query("newnebula")
        assert current_fts.query("newnebula")
        assert not current_fts.query("oldquasar")

    second = await reconcile_source_index(
        repo, embedder=counter, embedder_identity=_IDENTITY
    )
    assert second.status == "current"
    assert second.generation_id == current.generation_id
    assert counter.texts == 1
    store = _vector(repo, current)
    assert await store.count() == current.symbol_chunks + current.file_window_chunks
    await store.close()


async def test_fast_capture_replaces_a_tracked_excluded_file_window(lifecycle_repo):
    repo = lifecycle_repo
    old = read_manifest(default_manifest_path(repo))
    assert old is not None
    (repo / ".repowise" / "config.yaml").write_text('exclude_patterns:\n  - "**/*.yaml"\n')
    path = "infra/service.yaml"
    (repo / path).write_text("service: freshnebula\n")

    capture = await capture_source_changes(repo, [path])
    result = await reconcile_source_index(
        repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY
    )

    current = read_manifest(default_manifest_path(repo))
    assert current is not None
    assert capture.parsed == 0
    assert result.status == "published"
    assert current.generation_sequence == old.generation_sequence + 1
    with _fts(repo, current) as fts:
        fresh = fts.query("freshnebula")
        assert {hit.file_path for hit in fresh} == {path}
        assert all(hit.chunk_id.startswith(f"file:{path}:") for hit in fresh)
        assert not [hit for hit in fts.query("durable_marker") if hit.file_path == path]

    status = await inspect_source_index(repo, embedder=MockEmbedder())
    assert status.state == "current"
    assert status.stale_files == {}
    assert status.fts_chunks == status.vector_chunks == status.expected_chunks

    second = await reconcile_source_index(
        repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY
    )
    assert second.status == "current"
    assert second.generation_id == current.generation_id

    # A restarted watcher has lost its in-memory fast-path set. Re-observing
    # the same dirty tracked window must still be a durable no-op before an
    # outbox sequence is allocated.
    row_count = len(await _updates(repo))
    await capture_source_changes(repo, [path])
    assert len(await _updates(repo)) == row_count
    replay = await reconcile_source_index(
        repo,
        embedder=MockEmbedder(),
        embedder_identity=_IDENTITY,
    )
    replay_manifest = read_manifest(default_manifest_path(repo))
    assert replay.status == "current"
    assert replay_manifest is not None
    assert replay_manifest.generation_id == current.generation_id
    assert replay_manifest.generation_sequence == current.generation_sequence

    # A real byte change at the same path remains observable and advances once.
    (repo / path).write_text("service: finalquasar\n")
    await capture_source_changes(repo, [path])
    assert len(await _updates(repo)) == row_count + 1
    changed = await reconcile_source_index(
        repo,
        embedder=MockEmbedder(),
        embedder_identity=_IDENTITY,
    )
    changed_manifest = read_manifest(default_manifest_path(repo))
    assert changed.status == "published"
    assert changed_manifest is not None
    assert changed_manifest.generation_sequence == current.generation_sequence + 1
    with _fts(repo, changed_manifest) as fts:
        assert {hit.file_path for hit in fts.query("finalquasar")} == {path}


async def test_concurrent_reconciler_returns_busy_without_touching_generation(lifecycle_repo):
    repo = lifecycle_repo
    active = read_manifest(default_manifest_path(repo))
    assert active is not None
    lock_path = repo / ".repowise" / "source_search" / "reconcile.lock"

    with FileLock(lock_path):
        result = await reconcile_source_index(
            repo,
            embedder=MockEmbedder(),
            embedder_identity=_IDENTITY,
        )

    assert result.status == "busy"
    assert result.error == "another source-index reconciler is already running"
    assert read_manifest(default_manifest_path(repo)) == active


async def test_rest_reader_reopens_on_manifest_flip_and_reports_live_status(lifecycle_repo):
    from repowise.server.source_search_wiring import rest_coordinator

    repo = lifecycle_repo
    state = SimpleNamespace(
        db_url=resolve_db_url(repo),
        vector_store=_WikiVectors(),
        fts=None,
    )
    first = await rest_coordinator(state)
    assert first is not None
    first_manifest = read_manifest(default_manifest_path(repo))
    assert first_manifest is not None

    (repo / "src" / "app.py").write_text(_APP_V2)
    await _capture(repo, path="src/app.py")
    await reconcile_source_index(repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY)

    second = await rest_coordinator(state)
    assert second is not None and second is not first
    second_manifest = read_manifest(default_manifest_path(repo))
    assert second_manifest is not None
    assert second_manifest.generation_sequence > first_manifest.generation_sequence

    response = await second.search("newnebula", limit=3)
    assert any(result["file"] == "src/app.py" for result in response["results"])
    source_meta = response["_meta"]["source_search"]
    assert source_meta["generation_sequence"] == second_manifest.generation_sequence
    assert source_meta["status"] == "current"
    assert source_meta["pending_updates"] == 0

    await second.close()
    state.source_search_coordinator = None


async def test_manifest_publication_precedes_outbox_bookkeeping(lifecycle_repo):
    repo = lifecycle_repo
    previous = read_manifest(default_manifest_path(repo))
    assert previous is not None
    (repo / "src" / "app.py").write_text(_APP_V2)
    await _capture(repo, path="src/app.py")
    counter = _CountingEmbedder()

    def crash_after_manifest(boundary: str) -> None:
        if boundary == "after_manifest":
            raise _Crash(boundary)

    with pytest.raises(_Crash):
        await reconcile_source_index(
            repo,
            embedder=counter,
            embedder_identity=_IDENTITY,
            failure_injector=crash_after_manifest,
        )

    published = read_manifest(default_manifest_path(repo))
    assert published is not None
    assert published.generation_sequence == previous.generation_sequence + 1
    rows = await _updates(repo)
    assert rows[-1].generation_id == published.generation_id
    assert rows[-1].state == READY
    with _fts(repo, published) as fts:
        assert fts.query("newnebula")
        assert not fts.query("oldquasar")
    store = _vector(repo, published)
    assert await store.count() == published.symbol_chunks + published.file_window_chunks
    await store.close()

    # The manifest is the sole reader-publication pointer. Its generation is
    # current even during the brief, separately committed ledger bookkeeping
    # interval; the same-sequence READY row is not outstanding work.
    status = await inspect_source_index(repo, embedder=MockEmbedder())
    assert status.state == "current"
    assert status.generation_id == published.generation_id
    assert status.ready_updates == 0

    embedded_before_recovery = counter.texts
    recovered = await reconcile_source_index(
        repo,
        embedder=counter,
        embedder_identity=_IDENTITY,
    )
    assert recovered.status == "current"
    assert recovered.generation_id == published.generation_id
    assert recovered.generation_sequence == published.generation_sequence
    assert recovered.embedded == 0
    assert counter.texts == embedded_before_recovery
    assert (await _updates(repo))[-1].state == PUBLISHED

    repeated = await reconcile_source_index(repo, embedder=counter, embedder_identity=_IDENTITY)
    assert repeated.status == "current"
    assert repeated.generation_id == published.generation_id


async def test_coalesced_ready_replay_after_manifest_crash_is_a_noop(lifecycle_repo):
    repo = lifecycle_repo
    previous = read_manifest(default_manifest_path(repo))
    assert previous is not None
    (repo / "src" / "app.py").write_text(_APP_V2)
    await _capture(repo, path="src/app.py")
    (repo / "infra" / "service.yaml").write_text("service: coalescednebula\n")
    await capture_source_changes(repo, ["infra/service.yaml"])
    pending = await _updates(repo)
    assert [row.state for row in pending[-2:]] == [PENDING, PENDING]
    counter = _CountingEmbedder()

    def crash_after_manifest(boundary: str) -> None:
        if boundary == "after_manifest":
            raise _Crash(boundary)

    with pytest.raises(_Crash):
        await reconcile_source_index(
            repo,
            embedder=counter,
            embedder_identity=_IDENTITY,
            failure_injector=crash_after_manifest,
        )

    published = read_manifest(default_manifest_path(repo))
    assert published is not None
    assert published.generation_sequence == previous.generation_sequence + 2
    rows = await _updates(repo)
    assert [row.state for row in rows[-2:]] == [READY, READY]
    assert all(row.artifact_json for row in rows[-2:])
    assert all(json.loads(row.artifact_json) == published.to_dict() for row in rows[-2:])
    row_count = len(rows)
    embedded_before_recovery = counter.texts

    # A restarted capture can replay either member of the coalesced generation
    # before relational bookkeeping is healed. The active artifact proves that
    # both READY rows are already visible through the manifest.
    await _capture(repo, path="src/app.py")
    assert len(await _updates(repo)) == row_count

    recovered = await reconcile_source_index(
        repo,
        embedder=counter,
        embedder_identity=_IDENTITY,
    )
    recovered_manifest = read_manifest(default_manifest_path(repo))
    assert recovered.status == "current"
    assert recovered_manifest == published
    assert recovered.generation_id == published.generation_id
    assert recovered.generation_sequence == published.generation_sequence
    assert recovered.embedded == 0
    assert counter.texts == embedded_before_recovery
    assert [row.state for row in (await _updates(repo))[-2:]] == [PUBLISHED, PUBLISHED]


async def test_status_distinguishes_pending_from_degraded_work(lifecycle_repo):
    repo = lifecycle_repo
    current = await inspect_source_index(repo, embedder=MockEmbedder())
    assert current.state == "current"
    assert current.manifest_state == "ok"
    assert current.built_at
    assert current.embedder == _IDENTITY
    assert current.parser_fingerprint
    assert current.symbol_chunks + current.file_window_chunks == current.expected_chunks
    assert current.files_covered > 0
    assert current.fts_chunks == current.expected_chunks
    assert current.vector_chunks == current.expected_chunks

    (repo / "src" / "app.py").write_text(_APP_V2)
    await _capture(repo, path="src/app.py")
    pending = await inspect_source_index(repo, embedder=MockEmbedder())
    assert pending.state == "pending"
    assert pending.pending_updates == 1

    await record_source_index_error(repo, "ollama unavailable")
    degraded = await inspect_source_index(repo, embedder=MockEmbedder())
    assert degraded.state == "degraded"
    assert degraded.last_error == "ollama unavailable"


async def test_status_cannot_mistake_unreadable_manifest_for_missing(lifecycle_repo):
    manifest_path = default_manifest_path(lifecycle_repo)
    manifest_path.write_text("not json", encoding="utf-8")

    status = await inspect_source_index(lifecycle_repo, embedder=MockEmbedder())

    assert status.state == "inconsistent"
    assert status.manifest_state == "unreadable"
    assert status.manifest_error
    assert any("manifest unreadable" in error for error in status.integrity_errors)


async def test_status_read_does_not_recreate_a_missing_lance_store(lifecycle_repo):
    lance_path = lifecycle_repo / ".repowise" / "lancedb"
    displaced = lifecycle_repo / ".repowise" / "lancedb-displaced"
    lance_path.rename(displaced)

    status = await inspect_source_index(lifecycle_repo, embedder=MockEmbedder())

    assert status.state == "inconsistent"
    assert any("Lance store missing" in error for error in status.integrity_errors)
    assert not lance_path.exists()


@pytest.mark.parametrize(
    "stage",
    [
        "after_claim",
        "after_chunks",
        "after_fts",
        "after_vector",
        "after_verify",
        "after_ready",
        "after_manifest",
        "after_publish",
    ],
)
async def test_legacy_to_v2_migration_is_side_by_side_and_recoverable(legacy_repo, stage):
    repo = legacy_repo
    legacy = read_manifest(default_manifest_path(repo))
    assert legacy is not None
    legacy_store = _vector(repo, legacy)
    assert await legacy_store.count() == 3
    await legacy_store.close()

    def crash(boundary: str) -> None:
        if boundary == stage:
            raise _Crash(boundary)

    with pytest.raises(_Crash):
        await reconcile_source_index(
            repo,
            embedder=MockEmbedder(),
            embedder_identity=_IDENTITY,
            force_full=True,
            failure_injector=crash,
        )

    interrupted = read_manifest(default_manifest_path(repo))
    assert interrupted is not None
    if stage in {"after_manifest", "after_publish"}:
        assert interrupted.generation_sequence > 0
    else:
        assert interrupted.generation_sequence == 0
        assert interrupted.lance_table == "source_chunks"

    # The A1 table is never mutated by migration, on either side of the flip.
    legacy_store = _vector(repo, legacy)
    assert await legacy_store.count() == 3
    await legacy_store.close()

    await reconcile_source_index(repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY)
    current = read_manifest(default_manifest_path(repo))
    assert current is not None
    assert current.generation_sequence > 0
    assert current.lance_table != "source_chunks"
    assert current.fts_path.endswith("source_fts_v2.db")
    with _fts(repo, current) as fts:
        assert fts.count() == 3
        assert fts.query("oldquasar")
    current_store = _vector(repo, current)
    assert await current_store.count() == 3
    await current_store.close()


async def test_rename_then_delete_converges_without_ghosts(lifecycle_repo):
    repo = lifecycle_repo
    (repo / "src" / "app.py").rename(repo / "src" / "renamed.py")
    await _capture(
        repo,
        path="src/renamed.py",
        status="renamed",
        old_path="src/app.py",
    )
    await reconcile_source_index(repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY)
    renamed = read_manifest(default_manifest_path(repo))
    assert renamed is not None
    with _fts(repo, renamed) as fts:
        assert {hit.file_path for hit in fts.query("oldquasar")} == {"src/renamed.py"}
    store = _vector(repo, renamed)
    assert "src/app.py" not in await store.active_file_paths()
    await store.close()

    (repo / "src" / "renamed.py").unlink()
    await _capture(repo, path="src/renamed.py", status="deleted")
    await reconcile_source_index(repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY)
    deleted = read_manifest(default_manifest_path(repo))
    assert deleted is not None
    with _fts(repo, deleted) as fts:
        assert not fts.query("oldquasar")
    store = _vector(repo, deleted)
    assert "src/renamed.py" not in await store.active_file_paths()
    await store.close()


async def test_transient_parse_failure_keeps_last_good_chunks_and_marks_stale(lifecycle_repo):
    repo = lifecycle_repo
    (repo / "src" / "app.py").write_text("def alpha(:\n    brokenmarker\n")
    await _capture(repo, path="src/app.py", parse=False, mutate_symbols=False)
    await reconcile_source_index(repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY)
    stale = read_manifest(default_manifest_path(repo))
    assert stale is not None
    assert "src/app.py" in stale.stale_files
    with _fts(repo, stale) as fts:
        assert fts.query("oldquasar")
        assert not fts.query("brokenmarker")

    (repo / "src" / "app.py").write_text(_APP_V2)
    await _capture(repo, path="src/app.py")
    await reconcile_source_index(repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY)
    healed = read_manifest(default_manifest_path(repo))
    assert healed is not None
    assert "src/app.py" not in healed.stale_files
    with _fts(repo, healed) as fts:
        assert fts.query("newnebula")
        assert not fts.query("oldquasar")


@pytest.mark.parametrize(
    "stage",
    [
        "after_claim",
        "after_chunks",
        "after_fts",
        "after_vector",
        "after_verify",
        "after_ready",
        "after_manifest",
        "after_publish",
    ],
)
async def test_every_interruption_boundary_recovers_without_torn_visibility(
    lifecycle_repo, stage
):
    repo = lifecycle_repo
    old = read_manifest(default_manifest_path(repo))
    assert old is not None
    (repo / "src" / "app.py").write_text(_APP_V2)
    await _capture(repo, path="src/app.py")

    def crash(boundary: str) -> None:
        if boundary == stage:
            raise _Crash(boundary)

    with pytest.raises(_Crash):
        await reconcile_source_index(
            repo,
            embedder=MockEmbedder(),
            embedder_identity=_IDENTITY,
            failure_injector=crash,
        )

    interrupted = read_manifest(default_manifest_path(repo))
    assert interrupted is not None
    if stage in {"after_manifest", "after_publish"}:
        assert interrupted.generation_sequence > old.generation_sequence
    else:
        assert interrupted.generation_id == old.generation_id
        with _fts(repo, interrupted) as fts:
            assert fts.query("oldquasar")
            assert not fts.query("newnebula")

    await reconcile_source_index(repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY)
    recovered = read_manifest(default_manifest_path(repo))
    assert recovered is not None
    with _fts(repo, recovered) as fts:
        assert fts.query("newnebula")
        assert not fts.query("oldquasar")
        assert fts.count() == recovered.symbol_chunks + recovered.file_window_chunks
    store = _vector(repo, recovered)
    assert await store.count() == recovered.symbol_chunks + recovered.file_window_chunks
    await store.close()
    assert all(row.state == "published" for row in await _updates(repo))


async def test_superseded_abandoned_generation_never_leaks_forward(lifecycle_repo):
    repo = lifecycle_repo
    (repo / "src" / "app.py").write_text(_APP_V2)
    await _capture(repo, path="src/app.py")

    def crash_after_vector(boundary: str) -> None:
        if boundary == "after_vector":
            raise _Crash(boundary)

    with pytest.raises(_Crash):
        await reconcile_source_index(
            repo,
            embedder=MockEmbedder(),
            embedder_identity=_IDENTITY,
            failure_injector=crash_after_vector,
        )

    (repo / "src" / "app.py").write_text(_APP_V3)
    await _capture(repo, path="src/app.py")
    await reconcile_source_index(repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY)
    current = read_manifest(default_manifest_path(repo))
    assert current is not None
    with _fts(repo, current) as fts:
        assert fts.query("finalpulsar")
        assert not fts.query("newnebula")
        assert not fts.query("oldquasar")


async def test_changed_again_after_capture_defers_without_publishing(lifecycle_repo):
    repo = lifecycle_repo
    old = read_manifest(default_manifest_path(repo))
    (repo / "src" / "app.py").write_text(_APP_V2)
    await _capture(repo, path="src/app.py")
    (repo / "src" / "app.py").write_text(_APP_V3)

    with pytest.raises(SourceIndexDeferredError):
        await reconcile_source_index(
            repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY
        )
    assert read_manifest(default_manifest_path(repo)) == old

    await _capture(repo, path="src/app.py")
    await reconcile_source_index(repo, embedder=MockEmbedder(), embedder_identity=_IDENTITY)
    current = read_manifest(default_manifest_path(repo))
    assert current is not None
    with _fts(repo, current) as fts:
        assert fts.query("finalpulsar")
        assert not fts.query("newnebula")


async def test_model_recipe_change_builds_beside_and_flips_once(lifecycle_repo):
    repo = lifecycle_repo
    old = read_manifest(default_manifest_path(repo))
    assert old is not None
    wide = _WideEmbedder()
    identity = EmbedderIdentity(provider="mock", model="WideMock", dims=16)
    result = await reconcile_source_index(repo, embedder=wide, embedder_identity=identity)
    current = read_manifest(default_manifest_path(repo))
    assert current is not None
    assert result.status == "published"
    assert current.recipe_fingerprint != old.recipe_fingerprint
    assert current.lance_table != old.lance_table
    assert current.symbol_chunks == old.symbol_chunks
    assert current.file_window_chunks == old.file_window_chunks
    old_store = _vector(repo, old)
    assert await old_store.count() == old.symbol_chunks + old.file_window_chunks
    await old_store.close()
    new_store = _vector(repo, current, embedder=wide)
    assert await new_store.count() == current.symbol_chunks + current.file_window_chunks
    await new_store.close()


async def test_outbox_row_rolls_back_with_its_symbol_transaction(lifecycle_repo):
    repo = lifecycle_repo
    before = len(await _updates(repo))
    engine, factory = await _factory(repo)
    try:
        with pytest.raises(RuntimeError, match="rollback"):
            async with get_session(factory) as session:
                repository = (
                    await session.execute(
                        select(Repository).where(Repository.local_path == str(repo))
                    )
                ).scalar_one()
                parsed = _parse(repo, "src/app.py")
                await enqueue_incremental_update(
                    session,
                    repository.id,
                    repo,
                    file_diffs=[
                        SimpleNamespace(
                            path="src/app.py",
                            status="modified",
                            old_path=None,
                            new_parsed=parsed,
                        )
                    ],
                    parsed_files=[parsed],
                )
                raise RuntimeError("rollback")
    finally:
        await engine.dispose()
    assert len(await _updates(repo)) == before
