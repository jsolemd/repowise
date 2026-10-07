"""An anchored question cannot inherit a ruling explicitly scoped elsewhere."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from repowise.core.persistence.models import DecisionRecord
from repowise.server.mcp_server.tool_why.ranking import (
    _rank_keyword_matches,
    _score_keyword_matches,
)


def _record(id_: str, *, files=(), modules=(), basis="stated", text="pinned snapshot"):
    return SimpleNamespace(
        id=id_,
        title=text,
        decision=text,
        rationale="",
        context="",
        consequences_json="[]",
        tags_json="[]",
        evidence_file=None,
        status="active",
        affected_files_json=json.dumps(files),
        affected_modules_json=json.dumps(modules),
        scope_basis=basis,
    )


@pytest.mark.parametrize(
    ("query", "files", "target"),
    [
        (
            "read command one pinned generation snapshot",
            ["vibe/scripts/tmux_checkpoint.py"],
            "conceptatlas/atlas/cli.py",
        ),
        (
            "one query embedding for claim passages and held text",
            ["infra/repowise/docs/compose.yaml"],
            "conceptatlas/atlas/evidence_web/claim_search.py",
        ),
    ],
)
def test_exact_question_overlap_cannot_override_an_explicitly_different_scope(query, files, target):
    record = _record("elsewhere", files=files, text=query)
    assert _rank_keyword_matches([record], query, {target}) == []
    # The same record remains findable when no file was requested.
    assert _rank_keyword_matches([record], query, set()) == [record]


@pytest.mark.parametrize(
    ("files", "modules", "targets", "matches"),
    [
        (["src/cache.py"], [], {"src/cache.py"}, True),
        (["src/cache.py"], [], {"src/other.py", "src/cache.py"}, True),
        (["src/cache.py"], [], {"src/cache.py.old"}, False),
        (["src/cache.py"], [], {"src"}, True),
        ([], ["src/cache"], {"src/cache/store.py"}, True),
        ([], ["src/cache"], {"src/cache"}, True),
        ([], ["src/cache"], {"src/cache2/store.py"}, False),
        ([], ["src/cache"], {"src"}, True),
    ],
)
def test_scope_intersection_is_boundary_safe_and_accepts_any_target(
    files, modules, targets, matches
):
    record = _record("scoped", files=files, modules=modules)
    assert bool(_rank_keyword_matches([record], "pinned snapshot", targets)) is matches


@pytest.mark.parametrize("basis", ["repository", "commit_footprint", "session_proximity"])
def test_nonbinding_scopes_do_not_prove_a_contradiction(basis):
    record = _record("broad", files=["other/file.py"], basis=basis)
    assert _rank_keyword_matches([record], "pinned snapshot", {"src/cache.py"}) == [record]


def test_unscoped_records_remain_searchable():
    record = _record("unscoped")
    assert _rank_keyword_matches([record], "pinned snapshot", {"src/cache.py"}) == [record]


def test_scope_filter_does_not_recalculate_the_textual_floor_or_bypass_it():
    target = _record("target", files=["src/cache.py"], text="pinned snapshot")
    elsewhere = _record("elsewhere", files=["other.py"], text="pinned archive")
    records = [elsewhere, target]
    query = "pinned snapshot"
    unanchored = dict(
        (record.id, key) for key, record in _score_keyword_matches(records, query, set())
    )
    anchored = _score_keyword_matches(records, query, {"src/cache.py"})
    assert [record.id for _, record in anchored] == ["target"]
    assert anchored[0][0][0] == unanchored["target"][0]
    assert (
        _rank_keyword_matches(records, "zebrafish concurrency transactions", {"src/cache.py"}) == []
    )


async def _seed(session, repository_id, *, id_, files):
    session.add(
        DecisionRecord(
            id=id_,
            repository_id=repository_id,
            title="Zebrafish pinned snapshots",
            decision="Use a zebrafish pinned snapshot",
            rationale="One consistent read",
            affected_files_json=json.dumps(files),
            affected_modules_json="[]",
            scope_basis="stated",
            source="manual",
            status="active",
        )
    )
    await session.flush()


@pytest.mark.asyncio
async def test_a_scope_miss_keeps_the_requested_files_rationale(session, setup_mcp, tmp_path):
    from repowise.server.mcp_server import get_why

    await _seed(session, setup_mcp, id_="elsewhere", files=["other.py"])
    source = tmp_path / "src/cache.py"
    source.parent.mkdir()
    source.write_text(
        "# We pin zebrafish snapshots because reads must stay consistent.\nvalue = 1\n"
    )

    result = await get_why("why zebrafish pinned snapshots", targets=["src/cache.py"])

    assert result["decisions"] == []
    assert "src/cache.py" in result["target_context"]
    assert result["code_rationale"][0]["path"] == "src/cache.py"
    assert result["answer_basis"] == "rationale"


@pytest.mark.asyncio
async def test_supporting_semantic_hits_cannot_reintroduce_a_known_scope_miss(
    session, setup_mcp, monkeypatch
):
    from repowise.server.mcp_server import get_why
    from repowise.server.mcp_server.tool_why import search

    await _seed(session, setup_mcp, id_="target", files=["src/cache.py"])
    await _seed(session, setup_mcp, id_="elsewhere", files=["other.py"])

    async def semantic(*_args):
        return [
            SimpleNamespace(
                page_id=f"decision:{id_}", title=id_, snippet="snapshot evidence", score=0.9
            )
            for id_ in ("elsewhere", "unknown")
        ], []

    monkeypatch.setattr(search, "_semantic_lanes", semantic)
    result = await get_why(
        "why zebrafish pinned snapshots", targets=["src/cache.py"], include=["supporting"]
    )

    assert {row["id"] for row in result["decisions"]} == {"target", "unknown"}
    unknown = next(row for row in result["decisions"] if row["id"] == "unknown")
    assert unknown["authority"] == "candidate"
