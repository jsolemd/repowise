"""A parser upgrade must refresh SQL before publishing unchanged source."""

from __future__ import annotations

import asyncio
import sqlite3
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from repowise.cli.main import cli
from repowise.core.ingestion.parse_cache import parser_fingerprint


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _seed(repo: Path, monkeypatch, *, old_witness: str | None = None) -> None:
    from repowise.cli.helpers import config_fingerprint, save_state
    from repowise.core.pipeline.full_index import index_repo_full

    repo.mkdir()
    (repo / "ChatThread.tsx").write_text(
        "/** Render the withheld-note without disclosing its text. */\n"
        "export function ChatLineView() { return null; }\n"
    )
    (repo / "other.py").write_text("def other():\n    return 1\n")
    (repo / ".gitignore").write_text(".repowise/\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "initial")
    monkeypatch.delenv("REPOWISE_SOURCE_SEARCH", raising=False)
    asyncio.run(index_repo_full(repo))
    save_state(
        repo,
        {
            "last_sync_commit": _git(repo, "rev-parse", "HEAD"),
            "docs_enabled": False,
            "config_fingerprint": config_fingerprint(repo),
        },
    )
    # Reproduce a legacy SQL parse even when the source manifest was already
    # rebuilt by the new parser. Git and the source bytes remain unchanged.
    with sqlite3.connect(repo / ".repowise/wiki.db") as conn:
        conn.execute("UPDATE wiki_symbols SET docstring = NULL WHERE name = 'ChatLineView'")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(repositories)")}
        if "symbols_parser_fingerprint" in columns:
            assert (
                conn.execute("SELECT symbols_parser_fingerprint FROM repositories").fetchone()[0]
                == parser_fingerprint()
            )
            conn.execute("UPDATE repositories SET symbols_parser_fingerprint = ?", (old_witness,))


def _assert_fresh(repo: Path) -> None:
    from repowise.core.source_search.lifecycle import _full_chunks

    with sqlite3.connect(repo / ".repowise/wiki.db") as conn:
        docstring = conn.execute(
            "SELECT docstring FROM wiki_symbols WHERE name = 'ChatLineView'"
        ).fetchone()[0]
        assert docstring == "Render the withheld-note without disclosing its text."
        assert (
            conn.execute("SELECT symbols_parser_fingerprint FROM repositories").fetchone()[0]
            == parser_fingerprint()
        )
    chunk = next(
        chunk for chunk in asyncio.run(_full_chunks(repo, None)) if chunk.name == "ChatLineView"
    )
    assert "withheld-note" in chunk.text


@pytest.mark.parametrize("dirty", [False, True], ids=["quiet", "concurrent-dirty-file"])
@pytest.mark.parametrize("old_witness", [None, "older-parser"], ids=["legacy", "parser-drift"])
def test_cli_refreshes_unchanged_symbols(tmp_path, monkeypatch, dirty, old_witness):
    repo = tmp_path / "repo"
    _seed(repo, monkeypatch, old_witness=old_witness)
    if dirty:
        (repo / "other.py").write_text("def other():\n    return 2\n")
    result = CliRunner().invoke(
        cli, ["update", str(repo), "--no-workspace", "--index-only", "--include-working-tree"]
    )
    assert result.exit_code == 0, result.output
    _assert_fresh(repo)


def test_core_workspace_quiet_update_refreshes_unchanged_symbols(tmp_path, monkeypatch):
    from repowise.core.workspace.update import update_single_repo_index

    repo = tmp_path / "repo"
    _seed(repo, monkeypatch)
    result = asyncio.run(update_single_repo_index(repo))
    assert result.updated
    _assert_fresh(repo)


def _source_runtime(repo, monkeypatch):
    from repowise.cli import source_search_runtime
    from repowise.cli.providers import embedders
    from repowise.core.providers.embedding.base import MockEmbedder
    from repowise.core.source_search.lifecycle import reconcile_source_index
    from repowise.core.source_search.manifest import identify_embedder

    embedder = MockEmbedder()
    identity = identify_embedder(embedder, provider="mock")
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")
    monkeypatch.setattr(embedders, "resolve_embedder_for_repo", lambda _: "mock")
    monkeypatch.setattr(embedders, "build_embedder", lambda *_: embedder)
    calls = []
    failures = []

    async def reconcile(path, **kwargs):
        calls.append(kwargs.get("force_full", False))

        def fail_at(stage):
            if failures and stage == "after_fts":
                raise RuntimeError("injected publication failure")

        return await reconcile_source_index(
            path,
            embedder=embedder,
            embedder_identity=identity,
            force_full=kwargs.get("force_full", False),
            failure_injector=fail_at,
        )

    monkeypatch.setattr(source_search_runtime, "reconcile_configured_source_index", reconcile)
    return reconcile, calls, failures


def _legacy_publication(repo, monkeypatch, reconcile):
    """Reproduce the old publisher certifying stale SQL under the new recipe."""
    from repowise.core.persistence import parser_state
    from repowise.core.source_search.manifest import default_manifest_path, read_manifest

    async def old_publisher_did_not_check_sql(*_args, **_kwargs):
        return False

    with monkeypatch.context() as legacy:
        legacy.setattr(
            parser_state, "symbol_parser_refresh_required", old_publisher_did_not_check_sql
        )
        asyncio.run(reconcile(repo, force_full=True))
    manifest = read_manifest(default_manifest_path(repo))
    assert manifest.recipe_fingerprint
    return manifest


def _assert_published_documentation(repo):
    from repowise.core.source_search.fts import SourceFTSIndex
    from repowise.core.source_search.generation import GenerationRef
    from repowise.core.source_search.manifest import default_manifest_path, read_manifest

    manifest = read_manifest(default_manifest_path(repo))
    fts = SourceFTSIndex(
        repo / manifest.fts_path,
        generation=GenerationRef(manifest.generation_id, manifest.generation_sequence),
    )
    assert any("ChatLineView" in hit.chunk_id for hit in fts.query("withheld"))
    return manifest


@pytest.mark.parametrize("publication_failure", [False, True], ids=["published", "retry-pending"])
def test_current_source_recipe_does_not_hide_stale_sql(tmp_path, monkeypatch, publication_failure):
    from repowise.cli import source_search_runtime
    from repowise.core.pipeline import incremental
    from repowise.core.update_lock import release_update_lock

    repo = tmp_path / "repo"
    _seed(repo, monkeypatch)
    reconcile, calls, failures = _source_runtime(repo, monkeypatch)
    before = _legacy_publication(repo, monkeypatch, reconcile)
    assert asyncio.run(source_search_runtime.configured_source_recipe_changes(repo)) == ()
    calls.clear()
    if publication_failure:
        failures.append(True)
    result = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace", "--index-only"])
    assert result.exit_code == 0, result.output
    # CliRunner does not exit its process; model the CLI's atexit release
    # before issuing the next independent command.
    release_update_lock(repo)
    _assert_fresh(repo)

    def must_not_parse_again(*_args, **_kwargs):
        raise AssertionError("current SQL parser must not require another full parse")

    monkeypatch.setattr(incremental, "build_repo_graph", must_not_parse_again)
    if publication_failure:
        assert asyncio.run(source_search_runtime.configured_source_pending_updates(repo))
        failures.clear()
        calls.clear()
        retry = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace", "--index-only"])
        assert retry.exit_code == 0, retry.output
        assert calls == [False], retry.output
        release_update_lock(repo)
    after = _assert_published_documentation(repo)
    assert after.generation_id != before.generation_id
    assert after.recipe_fingerprint == before.recipe_fingerprint
    calls.clear()
    noop = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace", "--index-only"])
    assert noop.exit_code == 0, noop.output
    assert calls == []


def test_source_outbox_failure_rolls_back_parser_repair(tmp_path, monkeypatch):
    from repowise.core.source_search import outbox

    repo = tmp_path / "repo"
    _seed(repo, monkeypatch, old_witness="older-parser")
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")

    async def fail_receipt(*_args, **_kwargs):
        raise RuntimeError("injected receipt failure")

    monkeypatch.setattr(outbox, "enqueue_full_update", fail_receipt)
    result = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace", "--index-only"])
    assert result.exit_code != 0
    assert "injected receipt failure" in str(result.exception)
    with sqlite3.connect(repo / ".repowise/wiki.db") as conn:
        assert (
            conn.execute(
                "SELECT docstring FROM wiki_symbols WHERE name = 'ChatLineView'"
            ).fetchone()[0]
            is None
        )
        assert (
            conn.execute("SELECT symbols_parser_fingerprint FROM repositories").fetchone()[0]
            == "older-parser"
        )
        assert conn.execute("SELECT COUNT(*) FROM source_index_updates").fetchone()[0] == 0


@pytest.mark.parametrize("writer", ["build", "reconcile"])
def test_direct_source_writer_defers_before_mutation_on_stale_sql(tmp_path, monkeypatch, writer):
    from repowise.core.providers.embedding.base import MockEmbedder
    from repowise.core.source_search.indexer import build_source_index
    from repowise.core.source_search.lifecycle import reconcile_source_index
    from repowise.core.source_search.manifest import identify_embedder

    repo = tmp_path / "repo"
    _seed(repo, monkeypatch)
    embedder = MockEmbedder()
    function = build_source_index if writer == "build" else reconcile_source_index
    with pytest.raises(RuntimeError, match="SQL symbols need a parser refresh"):
        asyncio.run(
            function(
                repo,
                embedder=embedder,
                embedder_identity=identify_embedder(embedder),
            )
        )
    with sqlite3.connect(repo / ".repowise/wiki.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM source_index_updates").fetchone()[0] == 0
        assert (
            conn.execute("SELECT symbols_parser_fingerprint FROM repositories").fetchone()[0]
            is None
        )
    from repowise.core.source_search.manifest import default_manifest_path

    assert not default_manifest_path(repo).exists()


def test_source_cli_explains_stale_sql_without_a_traceback(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _seed(repo, monkeypatch)
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")

    result = CliRunner().invoke(cli, ["source-index", "--repo", str(repo), "--embedder", "mock"])

    assert result.exit_code == 1
    assert "Error: Persisted SQL symbols need a parser refresh" in result.output
    assert "repowise update" in result.output


@pytest.mark.parametrize("source_state", ["error", "busy", "degraded"])
def test_deferred_source_retry_preserves_receipt_and_completes_no_content_heals(
    tmp_path, monkeypatch, source_state
):
    import json

    from repowise.cli.helpers import (
        load_state,
        read_update_pending,
        write_update_pending,
    )
    from repowise.cli.source_search_runtime import configured_source_pending_updates
    from repowise.core.ingestion import ASTParser
    from repowise.core.update_lock import release_update_lock, try_acquire_update_lock

    repo = tmp_path / "repo"
    _seed(repo, monkeypatch)
    _reconcile, calls, failures = _source_runtime(repo, monkeypatch)
    failures.append(True)
    first = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace", "--index-only"])
    assert first.exit_code == 0, first.output
    release_update_lock(repo)

    previous = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "--allow-empty", "-qm", "metadata only")
    head = _git(repo, "rev-parse", "HEAD")
    write_update_pending(repo, previous)
    calls.clear()
    reason = "injected publication failure"
    if source_state != "error":
        from repowise.cli import source_search_runtime
        from repowise.core.source_search.lifecycle import SourceLifecycleResult

        reason = f"injected {source_state} publication"

        async def unavailable_source(*_args, **kwargs):
            calls.append(kwargs.get("force_full", False))
            return SourceLifecycleResult(
                status=source_state,
                generation_id=None,
                generation_sequence=None,
                updates_consumed=0,
                embedded=0,
                reused=0,
                chunks=0,
                stale_files=0,
                total_seconds=0,
                error=reason,
            )

        monkeypatch.setattr(
            source_search_runtime, "reconcile_configured_source_index", unavailable_source
        )

    def unexpected_parse(*_args, **_kwargs):
        raise AssertionError("source retry must not parse current SQL")

    monkeypatch.setattr(ASTParser, "parse_file", unexpected_parse)
    retry = CliRunner().invoke(
        cli, ["update", str(repo), "--no-workspace", "--index-only", "--progress", "json"]
    )

    assert retry.exit_code == 0, retry.output
    done = next(json.loads(line) for line in retry.stdout.splitlines() if '"done"' in line)
    assert done["outcome"] == "noop"
    assert done["pages_generated"] == 0
    assert any(reason in item for item in done["degraded"])
    assert calls == [False]
    assert load_state(repo)["last_sync_commit"] == head
    assert load_state(repo)["last_docs_commit"] == previous
    assert read_update_pending(repo) is None
    assert asyncio.run(configured_source_pending_updates(repo))
    assert try_acquire_update_lock(repo, head) is None, "the early return must release its lock"
    release_update_lock(repo)


def test_source_retry_retires_its_exit_callbacks_before_a_successor_acquires(tmp_path, monkeypatch):
    import atexit

    from repowise.cli import source_search_runtime
    from repowise.core.pipeline.full_index import index_repo_full
    from repowise.core.update_lock import update_lock_path

    repo = tmp_path / "repo"
    _seed(repo, monkeypatch)
    asyncio.run(index_repo_full(repo))

    async def pending(*_args, **_kwargs):
        return True

    async def reconciled(*_args, **_kwargs):
        return None

    monkeypatch.setattr(source_search_runtime, "configured_source_pending_updates", pending)
    monkeypatch.setattr(source_search_runtime, "reconcile_configured_source_index", reconciled)
    callbacks = []
    retired = []

    def register(callback, *args, **kwargs):
        callbacks.append((callback, args, kwargs))
        return callback

    monkeypatch.setattr(atexit, "register", register)
    monkeypatch.setattr(atexit, "unregister", retired.append)
    result = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace", "--index-only"])
    assert result.exit_code == 0, result.output
    assert not update_lock_path(repo).exists()

    # A different process can acquire immediately after this early return,
    # while the workspace process that called us remains alive.
    lock = update_lock_path(repo)
    queued = repo / ".repowise" / ".update.queued"
    lock.write_text("successor lock")
    queued.write_text("successor queue marker")
    for callback, args, kwargs in callbacks:
        callback(*args, **kwargs)
    assert lock.read_text() == "successor lock"
    assert queued.read_text() == "successor queue marker"
    assert callbacks
    assert all(callback in retired for callback, _, _ in callbacks)


def test_legacy_blocked_full_intent_reopens_only_after_complete_sql_refresh(tmp_path, monkeypatch):
    from sqlalchemy import select

    from repowise.core.ingestion import ASTParser, FileTraverser
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        get_session,
        resolve_db_url,
    )
    from repowise.core.persistence.models import Repository
    from repowise.core.source_search.outbox import enqueue_full_update

    repo = tmp_path / "repo"
    _seed(repo, monkeypatch)
    reconcile, _, _ = _source_runtime(repo, monkeypatch)
    _legacy_publication(repo, monkeypatch, reconcile)
    parsed = [
        ASTParser().parse_file(info, Path(info.abs_path).read_bytes())
        for info in FileTraverser(repo).traverse()
    ]

    async def block():
        engine = create_engine(resolve_db_url(repo))
        try:
            async with get_session(create_session_factory(engine)) as session:
                repository = (await session.execute(select(Repository))).scalar_one()
                blocked = await enqueue_full_update(
                    session,
                    repository.id,
                    repo,
                    parsed_files=parsed,
                    upstream_ready=False,
                    upstream_error="legacy failed symbol write",
                )
                direct = await enqueue_full_update(
                    session,
                    repository.id,
                    repo,
                    parsed_files=parsed,
                )
                assert direct.generation_id == blocked.generation_id
                assert direct.state == "blocked"
                assert not direct.upstream_ready
                return blocked.generation_id, blocked.sequence, blocked.dedupe_key
        finally:
            await engine.dispose()

    identity = asyncio.run(block())
    result = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace", "--index-only"])
    assert result.exit_code == 0, result.output
    _assert_fresh(repo)
    publication = _assert_published_documentation(repo)
    assert publication.generation_id == identity[0]
    with sqlite3.connect(repo / ".repowise/wiki.db") as conn:
        row = conn.execute(
            "SELECT generation_id, sequence, dedupe_key, state, upstream_ready, last_error "
            "FROM source_index_updates WHERE generation_id = ?",
            (identity[0],),
        ).fetchone()
        assert row == (*identity, "published", 1, None)
        assert conn.execute("SELECT COUNT(*) FROM source_index_updates").fetchone()[0] == 2


def test_docs_writer_refreshes_unchanged_sql_symbols(tmp_path, monkeypatch):
    from repowise.cli.commands.update_cmd.persistence import _persist_full_update_async
    from repowise.core.ingestion import ASTParser, FileTraverser, GraphBuilder

    repo = tmp_path / "repo"
    _seed(repo, monkeypatch)
    parsed = [
        ASTParser().parse_file(info, Path(info.abs_path).read_bytes())
        for info in FileTraverser(repo).traverse()
    ]
    graph = GraphBuilder(repo)
    for pf in parsed:
        graph.add_file(pf)
    graph.build()
    graph.traversed_file_paths = {pf.file_info.path for pf in parsed}
    asyncio.run(
        _persist_full_update_async(
            repo_path=repo,
            repo_name=repo.name,
            generated_pages=[],
            file_diffs=[],
            git_meta_map={},
            new_decision_markers=[],
            decision_vector_store=None,
            provider=None,
            partial_health_report=None,
            dead_code_report=None,
            graph_builder=graph,
            knowledge_graph_result=None,
            degraded=[],
            parsed_files=parsed,
        )
    )
    _assert_fresh(repo)


def test_failed_symbol_reconcile_cannot_fall_through_to_parser_certification(tmp_path, monkeypatch):
    from repowise.core.persistence import crud

    repo = tmp_path / "repo"
    _seed(repo, monkeypatch)

    async def fail_symbols(*_args, **_kwargs):
        raise RuntimeError("injected symbol write failure")

    monkeypatch.setattr(crud, "reconcile_symbols_for_files", fail_symbols)
    result = CliRunner().invoke(cli, ["update", str(repo), "--no-workspace", "--index-only"])
    assert result.exit_code != 0
    assert "injected symbol write failure" in str(result.exception)
    with sqlite3.connect(repo / ".repowise/wiki.db") as conn:
        assert (
            conn.execute(
                "SELECT docstring FROM wiki_symbols WHERE name = 'ChatLineView'"
            ).fetchone()[0]
            is None
        )
        assert (
            conn.execute("SELECT symbols_parser_fingerprint FROM repositories").fetchone()[0]
            is None
        )
