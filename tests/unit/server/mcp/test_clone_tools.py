"""find_clones and find_patterns over a real index of the planted repository.

The graph rows here are produced by running ingestion over the toy tree
and persisting what it built, rather than by hand-writing nodes: a tool
that reads the index must be tested against the index the indexer
actually writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.persistence.models import GraphEdge, GraphNode
from tests.unit.analysis import toy_clone_repo as toy

@pytest.fixture
async def indexed_toy_repo(tmp_path: Path, session: AsyncSession, factory, repo_id: str) -> Path:
    """Plant the tree, index it for real, persist the graph, wire MCP state."""
    from repowise.core.ingestion import ASTParser, FileTraverser, GraphBuilder
    from repowise.server.mcp_server import _state

    root = toy.build(tmp_path / "toy")
    traverser, parser, builder = FileTraverser(root), ASTParser(), GraphBuilder()
    for file_info in traverser.traverse():
        builder.add_file(parser.parse_file(file_info, Path(file_info.abs_path).read_bytes()))
    graph = builder.build()

    for node_id, data in graph.nodes(data=True):
        session.add(
            GraphNode(
                repository_id=repo_id,
                node_id=node_id,
                node_type=data.get("node_type", "file"),
                language=data.get("language") or "",
                symbol_count=data.get("symbol_count", 0),
                is_test=bool(data.get("is_test")),
                is_entry_point=bool(data.get("is_entry_point")),
                kind=data.get("kind"),
                name=data.get("name"),
                qualified_name=data.get("qualified_name"),
                file_path=data.get("file_path"),
                start_line=data.get("start_line"),
                end_line=data.get("end_line"),
                visibility=data.get("visibility"),
                signature=data.get("signature"),
                parent_name=data.get("parent_name"),
                parent_symbol_id=data.get("parent_symbol_id"),
            )
        )
    for src, dst, data in graph.edges(data=True):
        session.add(
            GraphEdge(
                repository_id=repo_id,
                source_node_id=src,
                target_node_id=dst,
                edge_type=data.get("edge_type", "imports"),
            )
        )
    await session.commit()

    _state._session_factory = factory
    _state._repo_path = str(root)
    try:
        yield root
    finally:
        _state._session_factory = None
        _state._repo_path = None
        _state._registry = None


# ---------------------------------------------------------------------------
# find_clones
# ---------------------------------------------------------------------------


async def test_find_clones_returns_the_planted_pair(indexed_toy_repo: Path) -> None:
    from repowise.server.mcp_server.tool_clones import find_clones

    result = await find_clones()

    assert result["status"] == "complete"
    files = [{s["file"] for s in f["sites"]} for f in result["findings"]]
    assert {"alpha/report.py", "beta/report.py"} in files
    assert result["definition"]["exact"]
    assert "_meta" in result


async def test_find_clones_never_names_the_detector_cache(indexed_toy_repo: Path) -> None:
    from repowise.server.mcp_server.tool_clones import find_clones

    await find_clones()  # warm the artifacts
    (indexed_toy_repo / ".repowise" / "duplication_cache.pkl").write_bytes(b"corrupt")

    result = await find_clones()
    payload = json.dumps(result).lower()

    assert "cache_unreadable" in payload
    for leaked in (".pkl", "pickle", "duplication_cache", "duplication_pairs"):
        assert leaked not in payload


async def test_find_clones_near_leg_is_off_by_default(indexed_toy_repo: Path) -> None:
    from repowise.server.mcp_server.tool_clones import find_clones

    result = await find_clones()

    assert result["near_clones"] == {"enabled": False, "threshold": None}
    assert all(f["kind"] == "exact" for f in result["findings"])
    assert "near" not in result["definition"]


async def test_find_clones_discloses_a_missing_similarity_index(
    indexed_toy_repo: Path,
) -> None:
    """Asking for near clones without an index is disclosed, not silently ignored."""
    from repowise.server.mcp_server.tool_clones import find_clones

    result = await find_clones(include_near=True)

    codes = {d["code"] for d in result["degradations"]}
    assert "near_clones_unavailable" in codes
    assert result["near_clones"]["enabled"] is True
    assert result["near_clones"]["threshold"] == 0.86
    assert result["status"] == "partial"
    # The exact leg still answered.
    files = [{s["file"] for s in f["sites"]} for f in result["findings"]]
    assert {"alpha/report.py", "beta/report.py"} in files


@pytest.mark.parametrize(("given", "clamped"), [(0.1, 0.70), (1.5, 0.999)])
async def test_near_threshold_is_clamped_and_the_clamp_is_reported(
    indexed_toy_repo: Path, given: float, clamped: float
) -> None:
    from repowise.server.mcp_server.tool_clones import find_clones

    result = await find_clones(include_near=True, near_threshold=given)

    assert result["near_clones"]["threshold"] == clamped
    assert result["ignored_arguments"][0]["argument"] == "near_threshold"


async def test_find_clones_scopes_to_a_path(indexed_toy_repo: Path) -> None:
    from repowise.server.mcp_server.tool_clones import find_clones

    result = await find_clones(path="core")

    assert result["summary"]["scanned_files"] == 4
    assert result["findings"] == []


async def test_find_clones_clamps_the_limit(indexed_toy_repo: Path) -> None:
    from repowise.server.mcp_server.tool_clones import find_clones

    result = await find_clones(limit=500)

    assert "limit_note" in result
    assert len(result["findings"]) <= 40


async def test_find_clones_rejects_repo_all(indexed_toy_repo: Path) -> None:
    from repowise.server.mcp_server.tool_clones import find_clones

    result = await find_clones(repo="all")
    assert "error" in result


async def test_find_clones_on_an_empty_index_does_not_claim_zero_duplication(
    tmp_path: Path, factory, repo_id: str
) -> None:
    from repowise.server.mcp_server import _state
    from repowise.server.mcp_server.tool_clones import find_clones

    _state._session_factory = factory
    _state._repo_path = str(tmp_path)
    try:
        result = await find_clones()
    finally:
        _state._session_factory = None
        _state._repo_path = None

    assert result["status"] == "failed"
    assert result["findings"] == []
    assert "not a finding of zero duplication" in result["no_results_reason"]
    assert result["degradations"][0]["impact"] == "results_unavailable"


# ---------------------------------------------------------------------------
# find_patterns
# ---------------------------------------------------------------------------


async def test_find_patterns_without_a_name_returns_the_catalogue(
    indexed_toy_repo: Path,
) -> None:
    from repowise.server.mcp_server.tool_patterns import find_patterns

    result = await find_patterns()

    assert len(result["patterns"]) == 6
    assert result["available_patterns"] == [
        "duplicate_signatures",
        "orphan_exports",
        "hub_functions",
        "isolated_siblings",
        "reuse_candidates",
        "bridge_functions",
    ]
    assert "matches" not in result


async def test_unknown_pattern_returns_the_catalogue_not_an_empty_list(
    indexed_toy_repo: Path,
) -> None:
    from repowise.server.mcp_server.tool_patterns import find_patterns

    result = await find_patterns(pattern="hubs")

    assert "matches" not in result
    assert len(result["patterns"]) == 6
    assert "not one of the six patterns" in result["note"]


@pytest.mark.parametrize(
    "pattern",
    [
        "duplicate_signatures",
        "orphan_exports",
        "hub_functions",
        "isolated_siblings",
        "reuse_candidates",
        "bridge_functions",
    ],
)
async def test_every_pattern_response_carries_its_definition(
    indexed_toy_repo: Path, pattern: str
) -> None:
    from repowise.server.mcp_server.tool_patterns import find_patterns

    result = await find_patterns(pattern=pattern)

    definition = result["definition"]
    assert definition["name"] == pattern
    for field in ("question", "predicate", "ranking", "not_this", "params"):
        assert definition[field]


async def test_each_pattern_finds_its_plant(indexed_toy_repo: Path) -> None:
    from repowise.server.mcp_server.tool_patterns import find_patterns

    async def keys(pattern: str, **kwargs) -> list[str]:
        result = await find_patterns(pattern=pattern, **kwargs)
        return [m["key"] for m in result["matches"]]

    assert "computetotal/2" in await keys("duplicate_signatures")
    assert "core/util.py::orphan_helper" in await keys("orphan_exports")
    assert "core/util.py::shared_helper" in await keys("hub_functions", min_callers=2)
    assert "core/lonely.py::loner" in await keys("isolated_siblings")
    assert "core/util.py::shared_helper" in await keys("reuse_candidates")
    assert "core/gateway.py::bridge" in await keys("bridge_functions")


async def test_inapplicable_parameters_are_reported_not_dropped(
    indexed_toy_repo: Path,
) -> None:
    from repowise.server.mcp_server.tool_patterns import find_patterns

    result = await find_patterns(pattern="bridge_functions", min_callers=99)

    assert result["ignored_arguments"][0]["argument"] == "min_callers"
    assert "does not apply" in result["ignored_arguments"][0]["reason"]


async def test_applied_parameters_travel_with_the_definition(
    indexed_toy_repo: Path,
) -> None:
    from repowise.server.mcp_server.tool_patterns import find_patterns

    result = await find_patterns(pattern="hub_functions", min_callers=2)

    assert result["definition"]["params"]["min_callers"] == 2
    assert "ignored_arguments" not in result


async def test_find_patterns_rejects_repo_all(indexed_toy_repo: Path) -> None:
    from repowise.server.mcp_server.tool_patterns import find_patterns

    result = await find_patterns(pattern="hub_functions", repo="all")
    assert "error" in result


async def test_empty_index_is_stated_not_implied(tmp_path: Path, factory, repo_id: str) -> None:
    from repowise.server.mcp_server import _state
    from repowise.server.mcp_server.tool_patterns import find_patterns

    _state._session_factory = factory
    _state._repo_path = str(tmp_path)
    try:
        result = await find_patterns(pattern="hub_functions")
    finally:
        _state._session_factory = None
        _state._repo_path = None

    assert result["matches"] == []
    assert "no indexed symbols" in result["no_results_reason"]


async def test_responses_stay_inside_the_transport_budget(
    indexed_toy_repo: Path,
) -> None:
    """Both tools cap their own output rather than relying on the caller."""
    from repowise.server.mcp_server.tool_clones import find_clones
    from repowise.server.mcp_server.tool_patterns import find_patterns

    clones = await find_clones(limit=500)
    patterns = await find_patterns(pattern="orphan_exports", limit=500)

    assert len(clones["findings"]) <= 40
    assert len(patterns["matches"]) <= 30
    assert len(json.dumps(clones)) < 100_000
    assert len(json.dumps(patterns)) < 100_000


async def test_test_files_are_out_of_scope_in_both_legs(
    indexed_toy_repo: Path, session: AsyncSession, repo_id: str
) -> None:
    """A duplicate planted in a test file is invisible until include_tests is set."""
    from sqlalchemy import update

    from repowise.server.mcp_server.tool_clones import find_clones

    await session.execute(
        update(GraphNode)
        .where(GraphNode.repository_id == repo_id, GraphNode.node_id == "beta/report.py")
        .values(is_test=True)
    )
    await session.commit()

    default = await find_clones()
    opted_in = await find_clones(include_tests=True)

    assert [{s["file"] for s in f["sites"]} for f in default["findings"]] == []
    assert {"alpha/report.py", "beta/report.py"} in [
        {s["file"] for s in f["sites"]} for f in opted_in["findings"]
    ]
    assert default["summary"]["scanned_files"] < opted_in["summary"]["scanned_files"]


async def test_the_scope_that_ran_travels_with_the_answer(indexed_toy_repo: Path) -> None:
    """Two responses are only comparable if each says what it filtered on."""
    from repowise.server.mcp_server.tool_clones import find_clones

    result = await find_clones(path="core", min_lines=9, cross_directory_only=True)

    assert result["scope"] == {
        "path": "core",
        "min_lines": 9,
        "include_tests": False,
        "include_intra_file": True,
        "cross_directory_only": True,
    }


async def test_nonsense_min_lines_is_clamped(indexed_toy_repo: Path) -> None:
    from repowise.server.mcp_server.tool_clones import find_clones

    result = await find_clones(min_lines=-5)

    assert result["scope"]["min_lines"] == 1
