"""Source-index trust, path diagnostics, and deliberate reindexing over MCP."""

from __future__ import annotations

import importlib
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from repowise.core.persistence.models import Repository
from repowise.core.source_search.chunks import (
    SymbolRecord,
    build_symbol_chunk,
    iter_file_windows,
)
from repowise.core.source_search.fts import SourceFTSIndex
from repowise.core.source_search.generation import GenerationRef
from repowise.core.source_search.manifest import EmbedderIdentity
from repowise.core.source_search.status import SourceIndexStatus
from repowise.server.services.job_queue import IndexJobQueueResult


def _status(**changes) -> SourceIndexStatus:
    status = SourceIndexStatus(
        state="current",
        generation_id="generation-3",
        generation_sequence=3,
        indexed_commit="indexed-head",
        recipe_fingerprint="recipe-1",
        pending_updates=0,
        blocked_updates=0,
        building_updates=0,
        ready_updates=0,
        manifest_state="ok",
        built_at="2026-08-21T16:00:00+00:00",
        published_at="2026-08-21T16:01:00+00:00",
        embedder=EmbedderIdentity(provider="mock", model="mock-embedder", dims=8),
        parser_fingerprint="parser-1",
        symbol_chunks=7,
        file_window_chunks=3,
        files_covered=5,
        expected_chunks=10,
        fts_chunks=10,
        vector_chunks=10,
        fts_path=".repowise/source_search/source_fts_v2.db",
    )
    return replace(status, **changes)


async def _wire_repo(
    *,
    module,
    monkeypatch,
    session,
    factory,
    repo_id: str,
    repo_path: Path,
    vector_store,
    fts,
) -> SimpleNamespace:
    repository = await session.get(Repository, repo_id)
    assert repository is not None
    repository.local_path = str(repo_path)
    repository.name = repo_path.name
    await session.commit()
    context = SimpleNamespace(
        alias="default",
        path=repo_path,
        session_factory=factory,
        vector_store=vector_store,
        fts=fts,
    )

    async def resolve(_repo: str | None) -> SimpleNamespace:
        return context

    monkeypatch.setattr(module, "_resolve_repo_context", resolve)
    return context


@pytest.mark.asyncio
async def test_status_verifies_both_stores_and_reports_exact_uncapped_queue_counts(
    tmp_path,
    monkeypatch,
    session,
    factory,
    repo_id,
    vector_store,
    fts,
) -> None:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    await _wire_repo(
        module=module,
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=repo_path,
        vector_store=vector_store,
        fts=fts,
    )
    observed: dict[str, object] = {}
    status = _status(
        pending_updates=100_003,
        building_updates=7,
        ready_updates=11,
        blocked_updates=13,
    )

    async def inspect(path, *, embedder, verify_stores):
        observed.update(path=Path(path), embedder=embedder, verify_stores=verify_stores)
        return status

    monkeypatch.setattr(module, "inspect_source_index", inspect)
    monkeypatch.setattr(module, "get_head_commit", lambda _path: "indexed-head")
    monkeypatch.setattr(module, "read_repo_state", lambda _path: {"working_tree_paths": []})
    monkeypatch.setattr(
        module,
        "_runtime_identities",
        lambda _ctx: (status.embedder, status.parser_fingerprint, None),
    )

    result = await module.get_index_status()

    assert observed == {
        "path": repo_path.resolve(),
        "embedder": vector_store._embedder,
        "verify_stores": True,
    }
    assert result["trust"] == {
        "search_results": "stale",
        "reasons": ["updates_outstanding", "updates_blocked"],
    }
    assert result["queue"] == {
        "pending": 100_003,
        "building": 7,
        "ready": 11,
        "blocked": 13,
        "total": 100_034,
        "unit": "source_index_update_rows_after_active_generation",
    }
    assert result["stores"] == {
        "manifest": {
            "state": "ok",
            "error": None,
            "chunks": {"symbol": 7, "file_window": 3, "total": 10},
            "files_covered": 5,
        },
        "fts_chunks": 10,
        "vector_chunks": 10,
        "parity": {"fts": True, "vector": True},
        "unit": "active_source_chunks",
    }
    assert result["degraded"] is False
    assert "degraded_reason" not in result
    assert "failed_legs" not in result


@pytest.mark.asyncio
async def test_status_reuses_standard_degradation_keys(
    tmp_path,
    monkeypatch,
    session,
    factory,
    repo_id,
    vector_store,
    fts,
) -> None:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    await _wire_repo(
        module=module,
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=repo_path,
        vector_store=vector_store,
        fts=fts,
    )
    status = _status(
        state="inconsistent",
        fts_chunks=9,
        integrity_errors=("FTS count mismatch: expected 10, found 9",),
    )
    monkeypatch.setattr(module, "inspect_source_index", AsyncMock(return_value=status))
    monkeypatch.setattr(module, "get_head_commit", lambda _path: "indexed-head")
    monkeypatch.setattr(module, "read_repo_state", lambda _path: {})
    monkeypatch.setattr(
        module,
        "_runtime_identities",
        lambda _ctx: (status.embedder, status.parser_fingerprint, None),
    )

    result = await module.get_index_status()

    assert result["degraded"] is True
    assert result["degraded_reason"] == "FTS count mismatch: expected 10, found 9"
    assert result["failed_legs"] == [
        {
            "leg": "source lexical",
            "error": "count_mismatch",
            "detail": "FTS count mismatch: expected 10, found 9",
        }
    ]


@pytest.mark.parametrize(
    ("status", "head", "expected", "reason"),
    [
        (_status(), "indexed-head", "trustworthy", None),
        (_status(), "new-head", "stale", "index_behind_head"),
        (
            _status(
                state="inconsistent",
                manifest_state="unreadable",
                manifest_error="JSONDecodeError",
                integrity_errors=("source manifest unreadable: JSONDecodeError",),
                fts_chunks=None,
                vector_chunks=None,
            ),
            "indexed-head",
            "unknown",
            "manifest_unreadable",
        ),
    ],
)
def test_search_trust_is_explicitly_tri_state(status, head, expected, reason) -> None:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")

    trust, reasons = module._trust_state(
        status,
        head_commit=head,
        runtime_embedder=status.embedder,
        runtime_parser=status.parser_fingerprint,
    )

    assert trust == expected
    if reason is None:
        assert reasons == []
    else:
        assert reason in reasons


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _symbol(path: str):
    body = "def indexed_symbol():\n    return 1\n"
    return build_symbol_chunk(
        SymbolRecord(
            symbol_id=f"{path}::indexed_symbol",
            file_path=path,
            name="indexed_symbol",
            qualified_name="indexed_symbol",
            kind="function",
            signature="def indexed_symbol()",
            docstring=None,
            start_line=1,
            end_line=2,
            language="python",
        ),
        body.splitlines(),
    )


def test_path_mode_uses_active_inventory_and_authoritative_policy_sites(
    tmp_path,
) -> None:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("def indexed_symbol():\n    return 1\n", encoding="utf-8")
    (repo / "deploy.yaml").write_text("services:\n  api: {}\n", encoding="utf-8")
    (repo / ".gitignore").write_text("scratch/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "src/app.py", "deploy.yaml", ".gitignore")
    _git(
        repo,
        "-c",
        "user.name=Dev",
        "-c",
        "user.email=dev@example.com",
        "commit",
        "-qm",
        "seed",
    )

    (repo / ".repowise").mkdir()
    (repo / ".repowise" / "config.yaml").write_text(
        'exclude_patterns:\n  - "ignored/**"\n', encoding="utf-8"
    )
    for path in ("ignored/tool.py", "scratch/tool.py", "local.yaml", "assets/blob.zzz"):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("value: 1\n", encoding="utf-8")

    generation = GenerationRef("generation-3", 3)
    fts_path = repo / ".repowise" / "source_search" / "source_fts_v2.db"
    chunks = [
        _symbol("src/app.py"),
        *iter_file_windows("deploy.yaml", "services:\n  api: {}\n"),
    ]
    with SourceFTSIndex(fts_path, generation=generation) as source_fts:
        source_fts.index_chunks(chunks)
    status = _status(
        symbol_chunks=1,
        file_window_chunks=1,
        files_covered=2,
        expected_chunks=2,
        fts_chunks=2,
        vector_chunks=2,
    )

    def diagnose(path: str) -> dict:
        base = {"mode": "status", "repo": {"path": str(repo)}}
        return module._path_mode_payload(base, status, {}, requested=path)["path"]

    symbol = diagnose("src/app.py")
    assert symbol["reason"] == "indexed"
    assert symbol["lanes"] == ["symbol"]
    assert symbol["chunks"]["total"] == 1

    window = diagnose("deploy.yaml")
    assert window["reason"] == "indexed"
    assert window["lanes"] == ["file_window"]
    assert window["chunks"]["total"] == 1

    config_excluded = diagnose("ignored/tool.py")
    assert config_excluded["reason"] == "config_excluded"
    assert config_excluded["eligibility"]["wiki_exclusion"]["source"] == "config"

    gitignored = diagnose("scratch/tool.py")
    assert gitignored["reason"] == "gitignored"
    assert gitignored["eligibility"]["wiki_exclusion"]["source"] == "gitignore"

    untracked = diagnose("local.yaml")
    assert untracked["reason"] == "untracked_window_only"
    assert untracked["untracked"] is True

    unknown = diagnose("assets/blob.zzz")
    assert unknown["reason"] == "unknown"
    assert "does not expose the deciding rule" in unknown["unknown_reason"]


@pytest.mark.asyncio
async def test_reindex_preview_is_read_only_and_confirmed_run_queues_index_only(
    tmp_path,
    monkeypatch,
    session,
    factory,
    repo_id,
    vector_store,
    fts,
) -> None:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    await _wire_repo(
        module=module,
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=repo_path,
        vector_store=vector_store,
        fts=fts,
    )
    status = _status(state="pending", pending_updates=1)
    index = {
        "trust": {"search_results": "stale", "reasons": ["updates_outstanding"]},
        "_meta": {},
    }
    monkeypatch.setattr(
        module,
        "_status_payload",
        AsyncMock(return_value=(index, status, {})),
    )
    monkeypatch.setattr(module, "_active_jobs", AsyncMock(return_value=[]))
    queue = AsyncMock(
        return_value=IndexJobQueueResult(
            status="accepted",
            job_id="job-1",
            job_state="pending",
            force=False,
            existing=False,
        )
    )
    monkeypatch.setattr(module, "queue_index_only_job", queue)

    preview = await module.reindex_repository()

    assert preview["status"] == "confirmation_required"
    assert preview["reindex"]["will_run"] is False
    assert preview["reindex"]["confirmation_required"] is True
    assert preview["reindex"]["cost"]["generative_calls"] == 0
    assert preview["reindex"]["cost"]["estimate"] == {
        "basis": "active_generation",
        "files": 5,
        "chunks": 10,
        "maximum_embeddings_for_active_generation": 10,
        "checkout_changes_may_change_totals": True,
    }
    queue.assert_not_awaited()

    accepted = await module.reindex_repository(confirm=True)

    assert accepted["status"] == "accepted"
    assert accepted["reindex"]["will_run"] is True
    assert accepted["reindex"]["job"] == {
        "id": "job-1",
        "state": "pending",
        "mode": "index_only",
        "force": False,
        "existing": False,
    }
    queue.assert_awaited_once()


@pytest.mark.asyncio
async def test_reindex_current_is_noop_unless_force_is_explicit(
    tmp_path,
    monkeypatch,
    session,
    factory,
    repo_id,
    vector_store,
    fts,
) -> None:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    await _wire_repo(
        module=module,
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=repo_path,
        vector_store=vector_store,
        fts=fts,
    )
    status = _status()
    index = {"trust": {"search_results": "trustworthy", "reasons": []}, "_meta": {}}
    monkeypatch.setattr(
        module,
        "_status_payload",
        AsyncMock(return_value=(index, status, {})),
    )
    monkeypatch.setattr(module, "_active_jobs", AsyncMock(return_value=[]))
    queue = AsyncMock(
        return_value=IndexJobQueueResult(
            status="accepted",
            job_id="job-force",
            job_state="pending",
            force=True,
            existing=False,
        )
    )
    monkeypatch.setattr(module, "queue_index_only_job", queue)

    current = await module.reindex_repository(confirm=True)
    assert current["status"] == "current"
    queue.assert_not_awaited()

    forced = await module.reindex_repository(confirm=True, force=True)
    assert forced["status"] == "accepted"
    assert forced["reindex"]["job"]["force"] is True
    assert queue.await_args.kwargs["force"] is True


def test_job_runtime_snapshots_repo_state_and_shares_process_registries() -> None:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    first_ctx = SimpleNamespace(
        session_factory=object(),
        fts=object(),
        vector_store=object(),
    )
    second_ctx = SimpleNamespace(
        session_factory=object(),
        fts=object(),
        vector_store=object(),
    )

    first = module._job_runtime(first_ctx)
    second = module._job_runtime(second_ctx)

    assert first is not second
    assert first.session_factory is first_ctx.session_factory
    assert first.fts is first_ctx.fts
    assert first.vector_store is first_ctx.vector_store
    assert second.session_factory is second_ctx.session_factory
    assert second.fts is second_ctx.fts
    assert second.vector_store is second_ctx.vector_store
    assert first.background_tasks is second.background_tasks
    assert first.job_tasks is second.job_tasks
    assert first.job_events is second.job_events
    assert first.job_cancel_tokens is second.job_cancel_tokens
    assert first.workspace_vector_stores is second.workspace_vector_stores


@pytest.mark.asyncio
async def test_tools_list_and_selection_keep_read_default_mutation_opt_in(
    tmp_path,
    monkeypatch,
) -> None:
    from repowise.core.registry import mcp_tool_registry
    from repowise.server.mcp_server import ensure_full_surface
    from repowise.server.mcp_server._tool_selection import (
        GENERATIVE_TOOL_NAMES,
        NO_GENERATIVE_ENV,
        resolve_enabled_tools,
    )

    mcp = ensure_full_surface()
    entries = {entry.name: entry for entry in mcp_tool_registry.entries()}
    listed = {tool.name for tool in await mcp.list_tools()}

    assert {"get_index_status", "reindex_repository"} <= listed
    assert entries["get_index_status"].default is True
    assert entries["reindex_repository"].default is False

    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    default = resolve_enabled_tools(entries.values(), is_workspace=False, repo_path=tmp_path)
    all_enabled = resolve_enabled_tools(
        entries.values(),
        is_workspace=False,
        override="all",
        repo_path=tmp_path,
    )
    assert "get_index_status" in default
    assert "reindex_repository" not in default
    assert "reindex_repository" in all_enabled
    assert not (GENERATIVE_TOOL_NAMES & all_enabled)
