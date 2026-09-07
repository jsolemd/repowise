"""End-to-end sweep behaviour of the CLI ``persist_result`` path.

The normal single-repo ``repowise init`` persists the INDEX phase
incrementally during the run, so ``persist_result`` takes the
``index_persisted_incrementally=True`` branch. Before the fix that branch
called ``persist_analysis`` + ``persist_generation`` but never swept stale
structurally-keyed pages (community-N / scc / layer), so they survived every
re-index forever — and their FTS + vector embeddings leaked with them.

These tests drive ``persist_result`` against a real repo-local SQLite DB with
a pre-seeded stale ``module_page`` and assert it is swept from the DB, the FTS
index, and an attached vector store, while the run's current page survives.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from repowise.cli._repo_session import open_repo_db
from repowise.cli.commands.init_cmd.persistence import persist_result
from repowise.core.generation.models import GeneratedPage
from repowise.core.persistence import FullTextSearch, get_session
from repowise.core.persistence.models import Page
from repowise.core.persistence.vector_store.in_memory import InMemoryVectorStore
from repowise.core.providers.embedding.base import MockEmbedder


def _generated_page(page_type: str, target: str) -> GeneratedPage:
    now = datetime.now(UTC).isoformat()
    pid = f"{page_type}:{target}"
    return GeneratedPage(
        page_id=pid,
        page_type=page_type,
        title=target,
        content=f"content for {target}",
        source_hash="x" * 64,
        model_name="mock",
        provider_name="mock",
        input_tokens=1,
        output_tokens=1,
        cached_tokens=0,
        generation_level=1,
        target_path=target,
        created_at=now,
        updated_at=now,
    )


def _result(
    repo_name: str,
    generated_pages: list[GeneratedPage],
    *,
    vector_store=None,
    authoritative_page_types: set[str] | None = None,
    preserved_page_ids: set[str] | None = None,
) -> SimpleNamespace:
    """A minimal PipelineResult stand-in for the ``index_done`` branch.

    persist_analysis / persist_generation read these attributes; everything is
    falsy so they no-op except the page upsert + the sweep.
    """
    return SimpleNamespace(
        repo_name=repo_name,
        index_persisted_incrementally=True,
        generated_pages=generated_pages,
        tech_stack=None,
        vector_store=vector_store,
        dead_code_report=None,
        health_report=None,
        decision_report=None,
        git_metadata_list=[],
        knowledge_graph_result=None,
        parsed_files=[],
        authoritative_page_types=authoritative_page_types or set(),
        preserved_page_ids=preserved_page_ids or set(),
    )


async def _seed_stale_module_page(repo_path, page_id: str) -> str:
    """Insert a stale module_page directly and return the repo id."""
    engine, sf, repo_id = await open_repo_db(repo_path, repo_name="r")
    try:
        now = datetime.now(UTC)
        async with get_session(sf) as session:
            session.add(
                Page(
                    id=page_id,
                    repository_id=repo_id,
                    page_type="module_page",
                    title="stale",
                    content="stale body",
                    target_path="community-155",
                    source_hash="x" * 64,
                    model_name="mock",
                    provider_name="mock",
                    created_at=now,
                    updated_at=now,
                )
            )
    finally:
        await engine.dispose()
    return repo_id


async def test_persist_result_index_done_sweeps_stale_module_page(tmp_path):
    """The incremental-index branch must sweep stale structural pages.

    Pre-fix this fails: the ``index_done`` branch never called the sweep, so
    the pre-seeded ``module_page:community-155`` survived alongside the new
    ``module_page:community-75``.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    stale_id = "module_page:community-155"
    await _seed_stale_module_page(repo_path, stale_id)

    current = _generated_page("module_page", "community-75")
    await persist_result(_result("r", [current]), repo_path)

    engine, sf, _ = await open_repo_db(repo_path, repo_name="r")
    try:
        async with get_session(sf) as session:
            ids = (
                (await session.execute(select(Page.id).where(Page.page_type == "module_page")))
                .scalars()
                .all()
            )
        assert set(ids) == {"module_page:community-75"}
        assert stale_id not in ids
    finally:
        await engine.dispose()


async def test_persist_result_index_done_sweeps_fts_and_vector(tmp_path):
    """Swept ids are removed from FTS + vector store; current ids survive."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    stale_id = "module_page:community-155"
    await _seed_stale_module_page(repo_path, stale_id)

    # Seed the FTS index for the stale page through the same engine
    # persist_result will reuse.
    engine, _sf, _ = await open_repo_db(repo_path, repo_name="r")
    fts = FullTextSearch(engine)
    await fts.ensure_index()
    await fts.index(stale_id, "stale", "stale body about widgets")
    await engine.dispose()

    # Seed the vector store with both the stale page and the page the run keeps.
    store = InMemoryVectorStore(MockEmbedder())
    current = _generated_page("module_page", "community-75")
    await store.embed_batch(
        [
            (stale_id, "stale body about widgets", {}),
            (current.page_id, "current body", {}),
        ]
    )

    await persist_result(_result("r", [current], vector_store=store), repo_path)

    # Vector store: stale gone, current survives.
    assert await store.list_page_ids() == {"module_page:community-75"}

    # FTS: the stale page is no longer searchable; the current page is indexed.
    engine, _sf, _ = await open_repo_db(repo_path, repo_name="r")
    fts = FullTextSearch(engine)
    await fts.ensure_index()
    hits = {r.page_id for r in await fts.search("widgets", limit=10)}
    assert stale_id not in hits
    # The run's current page is freshly FTS-indexed (its content mentions its
    # target_path "community-75").
    current_hits = {r.page_id for r in await fts.search("community-75", limit=10)}
    assert current.page_id in current_hits
    await engine.dispose()


@pytest.mark.asyncio
async def test_persist_result_index_done_keeps_pages_when_no_module_run(tmp_path):
    """A run with no module pages must not wipe existing module pages."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    stale_id = "module_page:community-1"
    await _seed_stale_module_page(repo_path, stale_id)

    # Run produces only a file_page — module pages are an unproduced type.
    current = _generated_page("file_page", "src/app.py")
    await persist_result(_result("r", [current]), repo_path)

    engine, sf, _ = await open_repo_db(repo_path, repo_name="r")
    try:
        async with get_session(sf) as session:
            ids = (await session.execute(select(Page.id))).scalars().all()
        assert stale_id in ids
        assert "file_page:src/app.py" in ids
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_persist_result_curated_zero_modules_sweeps_stale(tmp_path):
    """Live mini-taskq regression through the CLI path.

    A curated run authoritative for ``module_page`` that emitted ZERO module
    pages (every module collapsed into its layer via wholeLayer) must still
    sweep the pre-curated ``module_page:community-0``.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    stale_id = "module_page:community-0"
    await _seed_stale_module_page(repo_path, stale_id)

    # Run emits only a layer page but is authoritative for module_page too.
    current = _generated_page("layer_page", "layer:Data")
    await persist_result(
        _result("r", [current], authoritative_page_types={"module_page", "layer_page"}),
        repo_path,
    )

    engine, sf, _ = await open_repo_db(repo_path, repo_name="r")
    try:
        async with get_session(sf) as session:
            ids = (await session.execute(select(Page.id))).scalars().all()
        assert stale_id not in ids
        assert "layer_page:layer:Data" in ids
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_persist_result_keeps_pages_a_resume_skipped(tmp_path):
    """Issue #1089, through the path the reporter would have taken.

    A provider outage kills part of an init. `repowise init --resume`
    regenerates the missing pages and skips the ones already written, so the
    skipped ones are absent from ``generated_pages``. They must survive the
    sweep, keep their vector embedding, and become searchable — the resume
    exists to protect exactly this content, and a page nobody can find is not
    protected in any sense the user would recognise.
    """
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    kept_id = "module_page:community-1"
    await _seed_stale_module_page(repo_path, kept_id)

    # No FTS row is seeded on purpose. A run killed mid-generation flushes
    # pages to SQL and nothing else, so this is the state a resume actually
    # finds: the page exists and search has never heard of it. Seeding one here
    # would assert against a situation the broken flow never produces.
    store = InMemoryVectorStore(MockEmbedder())
    regenerated = _generated_page("module_page", "community-2")
    await store.embed_batch([(kept_id, "kept body about widgets", {})])

    await persist_result(
        _result("r", [regenerated], vector_store=store, preserved_page_ids={kept_id}),
        repo_path,
    )

    engine, sf, _ = await open_repo_db(repo_path, repo_name="r")
    try:
        async with get_session(sf) as session:
            ids = (
                (await session.execute(select(Page.id).where(Page.page_type == "module_page")))
                .scalars()
                .all()
            )
        assert set(ids) == {kept_id, "module_page:community-2"}

        # Backfilled from SQL: the seeded row's content is "stale body", which
        # is all a resume has to go on for a page it did not generate.
        fts = FullTextSearch(engine)
        await fts.ensure_index()
        hits = {r.page_id for r in await fts.search("body", limit=10)}
        assert kept_id in hits
    finally:
        await engine.dispose()

    assert kept_id in await store.list_page_ids()


# ---------------------------------------------------------------------------
# Retired pages must lose their vectors too (update path)
#
# Vector deletes used to ride only on ``tombstoned_page_ids``. A swept page is
# the worse case of the two: a tombstone still has a row, while a swept page is
# deleted outright, and retrieval hydrates title and snippet from the vector
# store itself — so the orphan answers in full for a page that 404s.
# ---------------------------------------------------------------------------


class _RecordingStore:
    def __init__(self) -> None:
        self.deleted: list[list[str]] = []

    async def delete_many(self, page_ids: list[str]) -> None:
        self.deleted.append(list(page_ids))


def test_update_persist_deletes_vectors_for_tombstoned_and_swept_pages(tmp_path, monkeypatch):
    """The single-repo index-only path deletes the union, not just tombstones."""
    from repowise.cli.commands.update_cmd import incremental as update_incremental
    from repowise.cli.commands.update_cmd import persistence as update_persistence
    from repowise.core.pipeline import incremental as core_incremental
    from repowise.core.pipeline.prune_state import DeletedFilePruneOutcome

    repo_path = tmp_path / "repo"
    (repo_path / ".repowise").mkdir(parents=True)

    outcome = DeletedFilePruneOutcome(
        attempted=True,
        pruned_paths=0,
        tombstoned_page_ids=("file_page:src/gone.py",),
        swept_page_ids=("module_page:codeatlas/chunking", "file_page:src/gone.py"),
    )

    async def _fake_persist(*_args, **_kwargs):
        return outcome

    store = _RecordingStore()
    monkeypatch.setattr(core_incremental, "persist_incremental_index", _fake_persist)
    monkeypatch.setattr(update_incremental, "_build_update_vector_store", lambda *_a, **_kw: store)
    monkeypatch.setattr(update_persistence, "run_decay_health_rescore", lambda *_a, **_kw: False)

    import networkx as nx

    class _Builder:
        def graph(self):
            return nx.DiGraph()

    update_persistence._persist_index_only_update(
        repo_path,
        _Builder(),
        {},
        None,
        None,
        {},
        "head1234",
        0.0,
        [],
    )

    assert store.deleted == [["file_page:src/gone.py", "module_page:codeatlas/chunking"]], (
        "the swept page kept its vector"
    )


def test_workspace_vector_delete_covers_swept_pages(tmp_path, monkeypatch):
    """The workspace pass takes the same union, for repos on the core path."""
    from repowise.cli.commands.update_cmd import incremental as update_incremental
    from repowise.cli.commands.update_cmd.workspace import _remove_tombstoned_page_vectors
    from repowise.core.pipeline.prune_state import DeletedFilePruneOutcome

    (tmp_path / "a").mkdir()
    store = _RecordingStore()
    monkeypatch.setattr(update_incremental, "_build_update_vector_store", lambda *_a, **_kw: store)

    result = SimpleNamespace(
        alias="a",
        updated=True,
        prune_outcome=DeletedFilePruneOutcome(
            attempted=True,
            tombstoned_page_ids=("file_page:a.py",),
            swept_page_ids=("scc_page:cycle-1",),
        ),
    )
    ws_config = SimpleNamespace(repos=[SimpleNamespace(alias="a", path="a")])

    _remove_tombstoned_page_vectors(tmp_path, ws_config, [result])

    assert store.deleted == [["file_page:a.py", "scc_page:cycle-1"]]
