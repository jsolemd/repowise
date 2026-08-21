"""get_symbol fetches a list of targets in one round trip.

Reading a card and then pulling four of its symbols used to be four calls.
Round-trip count, not payload size, is what dominates an agent's cost on this
tool, so the list form exists — and the single-target form has to keep
returning exactly what it always did, because every existing caller passes one.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from repowise.core.persistence.models import Repository, WikiSymbol

MODULE_SOURCE = '''"""Batch fixture."""


def alpha():
    return "a"


def beta():
    return "b"


def gamma():
    return "c"
'''


@pytest.fixture
def repo_on_disk(tmp_path, monkeypatch):
    import repowise.server.mcp_server as mcp_mod

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "batch.py").write_text(MODULE_SOURCE)
    monkeypatch.setattr(mcp_mod, "_repo_path", str(tmp_path))
    return tmp_path


@pytest.fixture
async def batch_symbols(session):
    repo = (await session.execute(select(Repository))).scalars().first()
    for i, (name, start, end) in enumerate((("alpha", 4, 5), ("beta", 8, 9), ("gamma", 12, 13))):
        session.add(
            WikiSymbol(
                id=f"batch{i}",
                repository_id=repo.id,
                file_path="pkg/batch.py",
                symbol_id=f"pkg/batch.py::{name}",
                name=name,
                qualified_name=f"pkg.batch.{name}",
                kind="function",
                signature=f"def {name}()",
                start_line=start,
                end_line=end,
                language="python",
            )
        )
    await session.flush()


@pytest.mark.asyncio
async def test_single_target_shape_is_unchanged(setup_mcp, repo_on_disk, batch_symbols):
    from repowise.server.mcp_server import get_symbol

    result = await get_symbol("pkg/batch.py::alpha")

    assert "results" not in result
    assert result["symbol_id"] == "pkg/batch.py::alpha"
    assert 'return "a"' in result["source"]


@pytest.mark.asyncio
async def test_batch_returns_one_result_per_target_in_order(setup_mcp, repo_on_disk, batch_symbols):
    from repowise.server.mcp_server import get_symbol

    result = await get_symbol(["pkg/batch.py::gamma", "pkg/batch.py::alpha", "beta"])

    assert result["count"] == 3
    assert result["resolved_count"] == 3
    assert [r["symbol_id"] for r in result["results"]] == [
        "pkg/batch.py::gamma",
        "pkg/batch.py::alpha",
        "pkg/batch.py::beta",
    ]
    assert [r["target"] for r in result["results"]] == [
        "pkg/batch.py::gamma",
        "pkg/batch.py::alpha",
        "beta",
    ]
    assert 'return "c"' in result["results"][0]["source"]
    # One envelope, not one per item: the freshness block is identical for
    # every entry and repeating it is pure duplication in the agent's context.
    assert "_meta" in result
    assert all("_meta" not in r for r in result["results"])


@pytest.mark.asyncio
async def test_batch_reports_a_miss_per_item_without_sinking_the_call(
    setup_mcp, repo_on_disk, batch_symbols
):
    from repowise.server.mcp_server import get_symbol

    result = await get_symbol(["pkg/batch.py::alpha", "pkg/batch.py::nope"])

    assert result["count"] == 2
    assert result["resolved_count"] == 1
    assert result["results"][0].get("error") is None
    assert "error" in result["results"][1]


@pytest.mark.asyncio
async def test_batch_carries_ambiguity_per_item(setup_mcp, repo_on_disk, batch_symbols, session):
    from repowise.server.mcp_server import get_symbol

    repo = (await session.execute(select(Repository))).scalars().first()
    session.add(
        WikiSymbol(
            id="batchdup",
            repository_id=repo.id,
            file_path="pkg/batch.py",
            symbol_id="pkg/batch.py::Shadow::alpha",
            name="alpha",
            qualified_name="pkg.batch.Shadow.alpha",
            kind="method",
            signature="def alpha(self)",
            start_line=4,
            end_line=5,
            language="python",
        )
    )
    await session.flush()

    result = await get_symbol(["beta", "alpha"])

    assert result["resolved_count"] == 1
    assert result["ambiguous_count"] == 1
    assert result["results"][0]["symbol_id"] == "pkg/batch.py::beta"
    assert result["results"][1]["status"] == "ambiguous"
    assert result["results"][1]["match_count"] == 2


@pytest.mark.asyncio
async def test_batch_over_the_cap_serves_what_fits_and_names_the_rest(
    setup_mcp, repo_on_disk, batch_symbols
):
    from repowise.server.mcp_server.tool_symbol import _MAX_BATCH_TARGETS, get_symbol

    targets = ["pkg/batch.py::alpha"] * (_MAX_BATCH_TARGETS + 3)
    result = await get_symbol(targets)

    assert result["count"] == _MAX_BATCH_TARGETS
    assert len(result["not_served"]) == 3
    assert str(_MAX_BATCH_TARGETS) in result["note"]


@pytest.mark.asyncio
async def test_id_alias_accepts_a_list_too(setup_mcp, repo_on_disk, batch_symbols):
    from repowise.server.mcp_server import get_symbol

    result = await get_symbol(id=["pkg/batch.py::alpha", "pkg/batch.py::beta"])

    assert result["count"] == 2
    assert [r["symbol_id"] for r in result["results"]] == [
        "pkg/batch.py::alpha",
        "pkg/batch.py::beta",
    ]


@pytest.mark.asyncio
async def test_empty_list_is_the_missing_argument_error(setup_mcp, repo_on_disk):
    from repowise.server.mcp_server import get_symbol

    result = await get_symbol([])

    assert "symbol_id is required" in result["error"]


@pytest.mark.asyncio
async def test_one_item_batch_preserves_single_call_freshness(
    setup_mcp, repo_on_disk, batch_symbols
):
    from repowise.server.mcp_server import get_symbol

    target = "pkg/batch.py::alpha"
    single = await get_symbol(target)
    batch = await get_symbol([target])
    freshness_keys = {
        "index_age_days",
        "indexed_commit",
        "live_head",
        "index_behind",
        "stale_warning",
    }

    assert {key: value for key, value in batch["_meta"].items() if key in freshness_keys} == {
        key: value for key, value in single["_meta"].items() if key in freshness_keys
    }
    assert all("_meta" not in item for item in batch["results"])


@pytest.mark.asyncio
async def test_batch_rebuilds_freshness_over_canonical_files_and_names_stale_targets(
    setup_mcp, repo_on_disk, batch_symbols, session, monkeypatch
):
    import repowise.server.mcp_server._meta as meta_mod
    from repowise.server.mcp_server import get_symbol

    other_path = repo_on_disk / "pkg" / "other.py"
    other_path.write_text('def delta():\n    return "d"\n')
    repo = (await session.execute(select(Repository))).scalars().first()
    repo.head_commit = "a" * 40
    session.add(
        WikiSymbol(
            id="batch-other",
            repository_id=repo.id,
            file_path="pkg/other.py",
            symbol_id="pkg/other.py::delta",
            name="delta",
            qualified_name="pkg.other.delta",
            kind="function",
            signature="def delta()",
            start_line=1,
            end_line=2,
            language="python",
        )
    )
    await session.flush()
    monkeypatch.setattr(meta_mod, "resolve_indexed_commit", lambda *_: "a" * 40)
    monkeypatch.setattr(meta_mod, "read_live_head", lambda *_: "b" * 40)
    monkeypatch.setattr(
        meta_mod,
        "_changed_files_between",
        lambda *_: frozenset({"pkg/batch.py"}),
    )

    result = await get_symbol(["alpha", "delta"])

    assert result["resolved_count"] == 2
    assert result["ambiguous_count"] == 0
    assert result["_meta"]["stale_warning"].startswith("A file this response serves changed")
    assert result["_meta"]["stale_targets"] == ["pkg/batch.py"]
    assert all("_meta" not in item for item in result["results"])
