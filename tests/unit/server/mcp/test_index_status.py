"""Source-index trust, path diagnostics, and deliberate reindexing over MCP."""

from __future__ import annotations

import importlib
import os
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from repowise.core.persistence.models import GenerationJob, Repository
from repowise.core.source_search.chunks import (
    SymbolRecord,
    build_symbol_chunk,
    iter_file_windows,
)
from repowise.core.source_search.fts import SourceFTSIndex
from repowise.core.source_search.generation import GenerationRef
from repowise.core.source_search.manifest import EmbedderIdentity
from repowise.core.source_search.status import (
    CODE_COUNT_MISMATCH,
    CODE_MISSING,
    CODE_UNREADABLE,
    COMPONENT_DENSE,
    COMPONENT_LEXICAL,
    COMPONENT_MANIFEST,
    COMPONENT_QUEUE,
    EVIDENCE_PRESERVING_CODES,
    INTEGRITY_CODES,
    IntegrityFinding,
    SourceIndexStatus,
)
from repowise.core.source_search.worktree import WorkingTreeDivergence
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
        working_tree=WorkingTreeDivergence(checked=True),
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
    assert "degradation_findings" not in result


@pytest.mark.asyncio
async def test_persisted_prune_refusal_degrades_trust_and_surfaces_a_finding(
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
    monkeypatch.setattr(module, "inspect_source_index", AsyncMock(return_value=status))
    monkeypatch.setattr(module, "get_head_commit", lambda _path: "indexed-head")
    monkeypatch.setattr(
        module,
        "read_repo_state",
        lambda _path: {
            "prune_refusals": {
                "from_commit": "prior-head",
                "findings": [
                    {
                        "table": "graph_nodes",
                        "candidate_paths": 448,
                        "persisted_paths": 776,
                    }
                ],
            }
        },
    )
    monkeypatch.setattr(
        module,
        "_runtime_identities",
        lambda _ctx: (status.embedder, status.parser_fingerprint, None),
    )

    result = await module.get_index_status()

    assert result["trust"] == {
        "search_results": "stale",
        "reasons": ["prune_refused: graph_nodes 448/776"],
    }
    assert result["degraded"] is True
    assert result["degradation_findings"] == [
        {
            "component": "deleted-file prune",
            "code": "prune_refused",
            "detail": (
                "Deleted-file prune refused for graph_nodes: 448 of 776 paths looked "
                "deleted, which reads as a broken run rather than a commit. Re-run with "
                "--accept-mass-deletion after confirming the deletion."
            ),
        }
    ]


@pytest.mark.asyncio
async def test_status_keeps_publication_findings_separate_from_retrieval_failures(
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
        integrity_findings=(
            IntegrityFinding(
                COMPONENT_LEXICAL,
                CODE_COUNT_MISMATCH,
                "FTS count mismatch: expected 10, found 9",
            ),
        ),
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
    assert result["degradation_findings"] == [
        {
            "component": "source lexical",
            "code": "count_mismatch",
            "detail": "FTS count mismatch: expected 10, found 9",
        }
    ]
    assert "failed_legs" not in result


@pytest.mark.parametrize(
    ("status", "head", "expected", "reason"),
    [
        (_status(), "indexed-head", "trustworthy", None),
        # An unread working tree is not a clean one: same empty path
        # tuples, opposite meaning, and only the read one supports
        # trustworthy.
        (
            _status(working_tree=WorkingTreeDivergence(checked=False)),
            "indexed-head",
            "unknown",
            "working_tree_unverified",
        ),
        (_status(), "new-head", "stale", "index_behind_head"),
        (
            _status(
                state="inconsistent",
                manifest_state="unreadable",
                manifest_error="JSONDecodeError",
                integrity_findings=(
                    IntegrityFinding(
                        COMPONENT_MANIFEST,
                        CODE_UNREADABLE,
                        "source manifest unreadable: JSONDecodeError",
                    ),
                ),
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
    (repo / "infra").mkdir()
    (repo / "infra" / "nginx.conf").write_text("server {}\n", encoding="utf-8")
    (repo / ".gitignore").write_text("scratch/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "src/app.py", "deploy.yaml", "infra/nginx.conf", ".gitignore")
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

    tracked_window = diagnose("infra/nginx.conf")
    assert tracked_window["reason"] == "eligible_not_indexed"
    assert tracked_window["eligibility"]["eligible"] is True
    assert tracked_window["eligibility"]["file_window"]["lane_eligible"] is True
    assert tracked_window["path_shape_candidate"] is False

    config_excluded = diagnose("ignored/tool.py")
    assert config_excluded["reason"] == "untracked_window_only"
    assert config_excluded["eligibility"]["parser"]["traversal_eligible"] is False

    gitignored = diagnose("scratch/tool.py")
    assert gitignored["reason"] == "untracked_window_only"
    assert gitignored["eligibility"]["parser"]["traversal_eligible"] is False

    untracked = diagnose("local.yaml")
    assert untracked["reason"] == "untracked_window_only"
    assert untracked["untracked"] is True

    unknown = diagnose("assets/blob.zzz")
    assert unknown["reason"] == "unknown"
    assert "deciding rule is unavailable" in unknown["unknown_reason"]


@pytest.mark.asyncio
async def test_active_jobs_discloses_corrupt_config_instead_of_guessing_sync(
    session,
    factory,
    repo_id,
) -> None:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    job = GenerationJob(
        repository_id=repo_id,
        status="running",
        config_json="{not-json",
    )
    session.add(job)
    await session.commit()

    jobs, total = await module._active_jobs(factory, repo_id)

    assert total == 1
    assert len(jobs) == 1
    assert jobs[0]["id"] == job.id
    assert jobs[0]["mode"] is None
    assert jobs[0]["config_error"] == "invalid_config_json"


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
    monkeypatch.setattr(module, "_active_jobs", AsyncMock(return_value=([], 0)))
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
    monkeypatch.setattr(module, "_active_jobs", AsyncMock(return_value=([], 0)))
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


# ---------------------------------------------------------------------------
# Trust is decided by the producer's codes, never by its prose
# ---------------------------------------------------------------------------


def _classify(status) -> tuple[str, list[str]]:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    return module._trust_state(
        status,
        head_commit=status.indexed_commit,
        runtime_embedder=status.embedder,
        runtime_parser=status.parser_fingerprint,
    )


@pytest.mark.parametrize(
    "code",
    sorted(INTEGRITY_CODES | {"a_fault_kind_this_consumer_has_never_seen"}),
)
def test_an_unrecognised_integrity_code_degrades_to_unknown_not_to_stale(code) -> None:
    """Drift on either side of the contract may only ever weaken the claim.

    ``EVIDENCE_PRESERVING_CODES`` is an allowlist. A code added to the
    producer without teaching this consumer about it lands outside the list
    and reports "unknown" — the honest "cannot tell" — rather than borrowing
    the stronger "stale" from the count mismatch sitting beside it.
    """

    status = _status(
        state="inconsistent",
        fts_chunks=9,
        integrity_findings=(IntegrityFinding(COMPONENT_LEXICAL, code, "any wording at all"),),
    )

    trust, _reasons = _classify(status)

    assert trust == ("stale" if code in EVIDENCE_PRESERVING_CODES else "unknown")


def test_rewording_an_integrity_message_cannot_move_the_verdict() -> None:
    """The old classifier read "missing"/"unreadable" out of free text.

    Both halves of that failure are pinned here: a store that is gone stays
    "unknown" however its message is phrased, and a message that merely
    contains the word "missing" cannot drag a count mismatch out of "stale".
    """

    for detail in ("FTS store missing: x", "the lexical store is not where the manifest says", ""):
        gone = _status(
            state="inconsistent",
            fts_chunks=None,
            integrity_findings=(IntegrityFinding(COMPONENT_LEXICAL, CODE_MISSING, detail),),
        )
        assert _classify(gone)[0] == "unknown"

    mismatch = _status(
        state="inconsistent",
        fts_chunks=9,
        integrity_findings=(
            IntegrityFinding(COMPONENT_LEXICAL, CODE_COUNT_MISMATCH, "rows missing: expected 10"),
        ),
    )
    trust, reasons = _classify(mismatch)
    assert trust == "stale"
    assert reasons == ["fts_count_mismatch"]


def test_hard_unknown_reasons_come_from_the_component_not_the_message() -> None:
    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    status = _status(
        state="inconsistent",
        fts_chunks=None,
        vector_chunks=None,
        integrity_findings=(
            # Deliberately worded so no substring names its own component.
            IntegrityFinding(COMPONENT_QUEUE, CODE_UNREADABLE, "could not read the update ledger"),
        ),
    )

    trust, reasons = module._trust_state(
        status,
        head_commit=status.indexed_commit,
        runtime_embedder=status.embedder,
        runtime_parser=status.parser_fingerprint,
        prune_refusals=(module.PruneRefusal("graph_nodes", 448, 776),),
    )

    assert trust == "unknown"
    assert "source_queue_unverified" in reasons
    assert "prune_refused: graph_nodes 448/776" in reasons


def test_every_declared_component_has_a_label_bound_to_the_name_it_displays() -> None:
    """A leg rename in the coordinator must reach this payload, not diverge from it."""

    from repowise.core.source_search.coordinator import LEG_SOURCE_DENSE, LEG_SOURCE_LEXICAL
    from repowise.core.source_search.status import INTEGRITY_COMPONENTS

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")

    assert set(module._COMPONENT_LABELS) == INTEGRITY_COMPONENTS
    assert module._COMPONENT_LABELS[COMPONENT_LEXICAL] == LEG_SOURCE_LEXICAL
    assert module._COMPONENT_LABELS[COMPONENT_DENSE] == LEG_SOURCE_DENSE


def test_an_unlabelled_component_keeps_the_producer_s_own_word() -> None:
    """Better an unfamiliar component name than a familiar wrong one."""

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    status = _status(
        state="inconsistent",
        integrity_findings=(IntegrityFinding("symbol_table", CODE_UNREADABLE, "boom"),),
    )

    assert module._degradation_findings(status) == [
        {"component": "source symbol_table", "code": "unreadable", "detail": "boom"}
    ]


# ---------------------------------------------------------------------------
# Bounded arrays, disclosed and reversible
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unbounded_arrays_are_capped_with_the_exact_total_and_a_restorable_tail(
    tmp_path,
    monkeypatch,
    session,
    factory,
    repo_id,
    vector_store,
    fts,
) -> None:
    """A grammar regression can mark every file stale; the payload must not blow up.

    The cap is disclosed rather than silent: the exact total sits beside the
    listed slice, and the dropped tail goes to the omission store so
    ``_meta.omitted`` can hand it back verbatim.
    """

    from repowise.core.distill.store import OmissionStore

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    (repo_path / ".repowise").mkdir(parents=True)
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
    stale = {f"src/stale_{index:04d}.py": "parser_failed" for index in range(120)}
    dirty = [f"src/dirty_{index:04d}.py" for index in range(120)]
    status = _status(state="degraded", stale_files=stale)
    monkeypatch.setattr(module, "inspect_source_index", AsyncMock(return_value=status))
    monkeypatch.setattr(module, "get_head_commit", lambda _path: "indexed-head")
    monkeypatch.setattr(module, "read_repo_state", lambda _path: {"working_tree_paths": dirty})
    monkeypatch.setattr(
        module,
        "_runtime_identities",
        lambda _ctx: (status.embedder, status.parser_fingerprint, None),
    )

    result = await module.get_index_status()

    generation = result["generation"]
    assert generation["stale_file_count"] == 120
    assert generation["stale_files_listed"] == 50
    assert len(generation["stale_files"]) == 50
    assert generation["uncommitted_indexed_path_count"] == 120
    assert generation["uncommitted_indexed_paths_listed"] == 50
    assert len(generation["uncommitted_indexed_paths"]) == 50

    omitted = result["_meta"]["omitted"]
    assert omitted["refs"]
    assert "repowise expand" in omitted["restore"]
    with OmissionStore.open_default(repo_path) as store:
        restored = "\n".join(store.get(ref) or "" for ref in omitted["refs"])
    assert "src/stale_0119.py" in restored
    assert "src/dirty_0119.py" in restored
    assert "src/stale_0000.py" not in restored


@pytest.mark.asyncio
async def test_a_payload_with_nothing_to_drop_writes_nothing_and_claims_nothing(
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
    (repo_path / ".repowise").mkdir(parents=True)
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
    monkeypatch.setattr(module, "inspect_source_index", AsyncMock(return_value=status))
    monkeypatch.setattr(module, "get_head_commit", lambda _path: "indexed-head")
    monkeypatch.setattr(module, "read_repo_state", lambda _path: {"working_tree_paths": ["a.py"]})
    monkeypatch.setattr(
        module,
        "_runtime_identities",
        lambda _ctx: (status.embedder, status.parser_fingerprint, None),
    )

    result = await module.get_index_status()

    assert result["generation"]["uncommitted_indexed_paths"] == ["a.py"]
    assert result["generation"]["uncommitted_indexed_path_count"] == 1
    assert "omitted" not in result["_meta"]
    assert not (repo_path / ".repowise" / "omissions").exists()


@pytest.mark.asyncio
async def test_active_jobs_are_capped_while_the_reported_count_stays_exact(
    session,
    factory,
    repo_id,
) -> None:
    """A capped list beside a capped total would understate a backlog."""

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    for _ in range(5):
        session.add(
            GenerationJob(
                repository_id=repo_id,
                status="pending",
                config_json='{"mode": "index_only"}',
            )
        )
    await session.commit()

    jobs, total = await module._active_jobs(factory, repo_id, limit=2)

    assert total == 5
    assert len(jobs) == 2


# ---------------------------------------------------------------------------
# Reads do not write; blocking work does not run on the loop
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_path_mode_reads_a_write_protected_lexical_store(tmp_path) -> None:
    """The store may be read-only on disk; a diagnostic must still answer.

    Opening it read-write — which is what the schema-applying constructor
    does — fails outright on a store the caller cannot write, so a tool
    documented as read-only would report "inventory unavailable" for a
    perfectly readable index.
    """

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def indexed_symbol():\n    return 1\n", encoding="utf-8")
    generation = GenerationRef("generation-3", 3)
    fts_path = repo / ".repowise" / "source_search" / "source_fts_v2.db"
    with SourceFTSIndex(fts_path, generation=generation) as source_fts:
        source_fts.index_chunks([_symbol("src/app.py")])
    before = fts_path.read_bytes()
    fts_path.chmod(0o444)
    status = _status(symbol_chunks=1, file_window_chunks=0, expected_chunks=1, fts_chunks=1)

    try:
        payload = module._path_mode_payload(
            {"mode": "status", "repo": {"path": str(repo)}},
            status,
            {},
            requested="src/app.py",
        )
    finally:
        fts_path.chmod(0o644)

    assert payload["path"]["indexed"] is True
    assert payload["path"]["chunks"]["total"] == 1
    assert payload["path"]["chunks"]["unknown_reason"] is None
    assert fts_path.read_bytes() == before


@pytest.mark.asyncio
async def test_path_mode_runs_its_blocking_work_off_the_event_loop(
    tmp_path,
    monkeypatch,
    session,
    factory,
    repo_id,
    vector_store,
    fts,
) -> None:
    """git, the ignore walk, and SQLite must not stall the MCP loop."""

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    (repo_path / "a.py").write_text("x = 1\n", encoding="utf-8")
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
    monkeypatch.setattr(module, "inspect_source_index", AsyncMock(return_value=status))
    monkeypatch.setattr(module, "get_head_commit", lambda _path: "indexed-head")
    monkeypatch.setattr(module, "read_repo_state", lambda _path: {"working_tree_paths": []})
    monkeypatch.setattr(
        module,
        "_runtime_identities",
        lambda _ctx: (status.embedder, status.parser_fingerprint, None),
    )
    observed: list[threading.Thread] = []
    for name in ("_repo_facts", "_path_mode_payload"):
        wrapped = getattr(module, name)

        def spy(*args, _wrapped=wrapped, **kwargs):
            observed.append(threading.current_thread())
            return _wrapped(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)

    result = await module.get_index_status(mode="path", path="a.py")

    assert result["mode"] == "path"
    assert len(observed) == 2
    assert all(thread is not threading.main_thread() for thread in observed)


def test_the_traversal_surface_is_built_once_until_a_rule_file_changes(
    tmp_path,
    monkeypatch,
) -> None:
    """Constructing a traverser per call re-read every ignore file and logged a line.

    Caching is only honest if an edit to a rule that governs the queried path
    misses the cache, so the key carries each rule file's identity.
    """

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text("scratch/\n", encoding="utf-8")
    built: list[int] = []
    real = module.FileTraverser

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "FileTraverser", counting)
    module._TRAVERSER_CACHE.clear()
    try:
        assert module._traversal_eligibility(repo, "src/app.py") == (True, None)
        assert module._traversal_eligibility(repo, "src/app.py") == (True, None)
        assert len(built) == 1

        (repo / ".gitignore").write_text("scratch/\nbuild/\n", encoding="utf-8")
        module._traversal_eligibility(repo, "src/app.py")
        assert len(built) == 2

        (repo / "src" / ".gitignore").write_text("app.py\n", encoding="utf-8")
        assert module._traversal_eligibility(repo, "src/app.py")[0] is False
        assert len(built) == 3
    finally:
        module._TRAVERSER_CACHE.clear()


# ---------------------------------------------------------------------------
# already_running names what is actually running
# ---------------------------------------------------------------------------


async def _reindex_with_jobs(module, monkeypatch, ctx_args, jobs) -> dict:
    await _wire_repo(module=module, monkeypatch=monkeypatch, **ctx_args)
    for mode in jobs:
        ctx_args["session"].add(
            GenerationJob(
                repository_id=ctx_args["repo_id"],
                status="running",
                config_json=f'{{"mode": "{mode}"}}',
            )
        )
    await ctx_args["session"].commit()
    status = _status()
    monkeypatch.setattr(
        module,
        "_status_payload",
        AsyncMock(
            return_value=(
                {"trust": {"search_results": "stale", "reasons": []}, "_meta": {}},
                status,
                {},
            )
        ),
    )
    return await module.reindex_repository()


@pytest.mark.asyncio
async def test_a_running_generative_job_is_not_presented_as_the_priced_reindex(
    tmp_path,
    monkeypatch,
    session,
    factory,
    repo_id,
    vector_store,
    fts,
) -> None:
    """``cost.generative_calls: 0`` describes the queued index_only job, nothing else."""

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    result = await _reindex_with_jobs(
        module,
        monkeypatch,
        {
            "session": session,
            "factory": factory,
            "repo_id": repo_id,
            "repo_path": repo_path,
            "vector_store": vector_store,
            "fts": fts,
        },
        ["full_resync"],
    )

    reindex = result["reindex"]
    assert result["status"] == "already_running"
    assert reindex["cost"]["generative_calls"] == 0
    assert reindex["job"]["mode"] == "full_resync"
    assert reindex["job_matches_reindex_mode"] is False
    assert "index_only" in reindex["job_mode_note"]


@pytest.mark.asyncio
async def test_a_running_index_only_job_is_the_one_reported_when_several_are_active(
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
    result = await _reindex_with_jobs(
        module,
        monkeypatch,
        {
            "session": session,
            "factory": factory,
            "repo_id": repo_id,
            "repo_path": repo_path,
            "vector_store": vector_store,
            "fts": fts,
        },
        ["full_resync", "index_only"],
    )

    reindex = result["reindex"]
    assert reindex["active_job_count"] == 2
    assert reindex["job"]["mode"] == "index_only"
    assert reindex["job_matches_reindex_mode"] is True
    assert "job_mode_note" not in reindex
