"""Retrieval quality over MCP: the report, the export, and the write it refuses."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repowise.core.persistence.models import Repository
from repowise.core.registry import mcp_tool_registry
from repowise.core.source_search.query_log import QueryEvent, QueryLog, TopEntry
from repowise.server.mcp_server import tool_query_report
from repowise.server.mcp_server.tool_query_report import get_query_quality



async def _wire(*, monkeypatch, session, factory, repo_id: str, repo_path: Path):
    repository = await session.get(Repository, repo_id)
    assert repository is not None
    repository.local_path = str(repo_path)
    repository.name = repo_path.name
    await session.commit()
    context = SimpleNamespace(
        alias="default",
        path=repo_path,
        session_factory=factory,
        vector_store=None,
        fts=None,
    )

    async def resolve(_repo: str | None) -> SimpleNamespace:
        return context

    monkeypatch.setattr(tool_query_report, "_resolve_repo_context", resolve)
    return context


def _seed(repo_path: Path) -> Path:
    """A log with one repeated no-match and one contradicted confident answer."""
    log_path = repo_path / ".repowise" / "source_search" / "query_log.jsonl"
    log = QueryLog(log_path)
    for index in range(3):
        log.append(
            QueryEvent(
                query="how do I configure kubernetes ingress",
                mode="hybrid",
                limit=5,
                latency_ms=9.0,
                confidence="no_match",
                result_count=0,
                no_match=True,
                ts=f"2026-08-17T09:0{index}:00.000+00:00",
                generation="aaaaaaaaaaaa",
            )
        )
    for owner in ("taskq/task.py", "taskq/queue.py"):
        log.append(
            QueryEvent(
                query="where does a task get its identity",
                mode="hybrid",
                limit=5,
                latency_ms=12.0,
                confidence="confident",
                result_count=2,
                top=[
                    TopEntry(
                        file=owner,
                        lane="source",
                        dense_cosine=0.77,
                        lexical_rank=1,
                        exact_name=True,
                        fused_score=0.02,
                    )
                ],
                selected_owner_file=owner,
                ts="2026-08-17T09:10:00.000+00:00",
                generation="aaaaaaaaaaaa",
            )
        )
    return log_path


@pytest.mark.asyncio
async def test_report_mode_buckets_the_log_and_names_its_caveats(
    monkeypatch, session, factory, repo_id, tmp_path
):
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    _seed(tmp_path)

    result = await get_query_quality(mode="report")
    buckets = result["report"]["buckets"]
    assert buckets["no_match"]["count"] == 3
    assert buckets["wrong_owner"]["count"] == 2
    assert buckets["no_match"]["offenders"][0]["occurrences"] == 3
    assert result["report"]["trend"]["window"] == "day"
    # Definitions travel with the numbers, so a reader of the payload never
    # has to guess what a bucket was claiming.
    assert "later signal contradicts" in buckets["wrong_owner"]["definition"]


@pytest.mark.asyncio
async def test_a_repository_that_never_searched_gets_an_answer_not_an_error(
    monkeypatch, session, factory, repo_id, tmp_path
):
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    result = await get_query_quality(mode="report")
    assert result["report"]["source"]["read_error"] == "log_not_found"
    assert "no query log exists" in result["next_action"].lower()


@pytest.mark.asyncio
async def test_export_mode_returns_cases_and_counts_the_pending_ones(
    monkeypatch, session, factory, repo_id, tmp_path
):
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    _seed(tmp_path)

    result = await get_query_quality(mode="export", bucket="wrong_owner")
    assert result["case_count"] == 1
    # The bucket exists because retrieval contradicted itself; which rival is
    # right is exactly what the log cannot say.
    assert result["pending_cases"] == 1
    assert result["cases"][0]["expect"]["kind"] == "todo"
    assert result["cases"][0]["expect"]["candidates"] == ["taskq/task.py", "taskq/queue.py"]


@pytest.mark.asyncio
async def test_export_writes_only_inside_the_eval_directory(
    monkeypatch, session, factory, repo_id, tmp_path
):
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    _seed(tmp_path)

    written = await get_query_quality(mode="export", bucket="no_match", write_to="week31.json")
    assert written["write"]["written"] is True
    target = tmp_path / ".repowise" / "source_search" / "eval" / "week31.json"
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["bucket"] == "no_match"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested",
    ["../../../escape.json", "/etc/passwd.json", "sub/../../out.json", "notjson.txt", ""],
)
async def test_a_write_outside_the_eval_directory_is_refused(
    monkeypatch, session, factory, repo_id, tmp_path, requested
):
    """An agent-facing tool must not be a write-anywhere primitive."""
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    _seed(tmp_path)

    result = await get_query_quality(mode="export", bucket="no_match", write_to=requested)
    assert result["write"]["written"] is False
    assert not (tmp_path / ".repowise" / "source_search" / "eval").exists()


@pytest.mark.asyncio
async def test_export_without_a_bucket_says_which_ones_exist(
    monkeypatch, session, factory, repo_id, tmp_path
):
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    _seed(tmp_path)

    result = await get_query_quality(mode="export")
    assert "bucket is required" in result["error"]
    assert result["valid_buckets"] == ["error", "wrong_owner", "no_match", "low_confidence"]


@pytest.mark.asyncio
async def test_an_unknown_argument_is_named_rather_than_applied(
    monkeypatch, session, factory, repo_id, tmp_path
):
    """A typo must not silently become a different report."""
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    _seed(tmp_path)

    result = await get_query_quality(mode="report", window="fortnight")
    assert result["report"]["trend"]["window"] == "day"
    assert result["ignored_arguments"] == [
        {"argument": "window", "values": ["fortnight"], "valid": ["day", "hour", "week"]}
    ]


@pytest.mark.asyncio
async def test_a_malformed_verdicts_file_changes_nothing(
    monkeypatch, session, factory, repo_id, tmp_path
):
    """A verdict overrides the whole detector, so a bad one is refused whole."""
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    _seed(tmp_path)
    eval_dir = tmp_path / ".repowise" / "source_search" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "v.json").write_text(json.dumps({"a query": 7}), encoding="utf-8")

    result = await get_query_quality(mode="report", verdicts="v.json")
    assert result["verdicts"]["applied"] is False
    assert "must map a query" in result["verdicts"]["reason"]
    assert result["report"]["totals"]["verdicts_applied"] == 0
    # And the self-contradiction verdict is still reported, unaltered.
    assert result["report"]["buckets"]["wrong_owner"]["count"] == 2


@pytest.mark.asyncio
async def test_a_good_verdicts_file_turns_the_bucket_into_a_measurement(
    monkeypatch, session, factory, repo_id, tmp_path
):
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    _seed(tmp_path)
    eval_dir = tmp_path / ".repowise" / "source_search" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "v.json").write_text(
        json.dumps({"where does a task get its identity": "taskq/task.py"}), encoding="utf-8"
    )

    result = await get_query_quality(mode="report", verdicts="v.json")
    assert "verdicts" not in result
    assert result["report"]["totals"]["verdicts_applied"] == 1
    offender = result["report"]["buckets"]["wrong_owner"]["offenders"][0]
    assert offender["conflict_basis"] == "verdict"
    assert offender["expected_owner"] == "taskq/task.py"


@pytest.mark.asyncio
async def test_run_mode_without_an_index_says_so_instead_of_raising(
    monkeypatch, session, factory, repo_id, tmp_path
):
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    _seed(tmp_path)
    await get_query_quality(mode="export", bucket="no_match", write_to="s.json")

    result = await get_query_quality(mode="run", suite="s.json")
    assert result["error"] == "no source index is available for this repository"
    assert "source-index" in result["next_action"]


@pytest.mark.asyncio
async def test_run_mode_reads_only_inside_the_eval_directory(
    monkeypatch, session, factory, repo_id, tmp_path
):
    """Loading any JSON on the host and echoing it back is exfiltration."""
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    outside = tmp_path.parent / "secrets.json"
    outside.write_text("{}", encoding="utf-8")

    result = await get_query_quality(mode="run", suite=str(outside))
    assert result["error"].startswith("outside_export_dir")


@pytest.mark.asyncio
async def test_run_mode_requires_a_suite(
    monkeypatch, session, factory, repo_id, tmp_path
):
    await _wire(
        monkeypatch=monkeypatch,
        session=session,
        factory=factory,
        repo_id=repo_id,
        repo_path=tmp_path,
    )
    result = await get_query_quality(mode="run")
    assert result["error"] == "suite is required when mode='run'"
