"""Contracts for calls-only paths and complete inbound dependents."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import event

from repowise.core.persistence.models import GraphEdge, GraphNode, WikiSymbol

_NOW = datetime(2026, 8, 17, 12, 0, 0, tzinfo=UTC)


def _file_node(
    key: str,
    path: str,
    *,
    pagerank: float = 0.0,
    is_test: bool = False,
) -> GraphNode:
    return GraphNode(
        id=f"file-{key}",
        repository_id="repo1",
        node_id=path,
        node_type="file",
        language="python",
        pagerank=pagerank,
        is_test=is_test,
        created_at=_NOW,
    )


def _symbol_rows(
    key: str,
    path: str,
    name: str,
    *,
    qualified_name: str | None = None,
    pagerank: float = 0.0,
    is_test: bool = False,
) -> tuple[GraphNode, WikiSymbol]:
    symbol_id = f"{path}::{name}"
    qualified = qualified_name or f"pkg.{key}.{name}"
    node = GraphNode(
        id=f"symbol-node-{key}",
        repository_id="repo1",
        node_id=symbol_id,
        node_type="symbol",
        language="python",
        pagerank=pagerank,
        is_test=is_test,
        kind="function",
        name=name,
        qualified_name=qualified,
        file_path=path,
        start_line=1,
        end_line=2,
        created_at=_NOW,
    )
    symbol = WikiSymbol(
        id=f"wiki-symbol-{key}",
        repository_id="repo1",
        file_path=path,
        symbol_id=symbol_id,
        name=name,
        qualified_name=qualified,
        kind="function",
        signature=f"def {name}()",
        start_line=1,
        end_line=2,
        language="python",
        created_at=_NOW,
        updated_at=_NOW,
    )
    return node, symbol


def _edge(
    key: str,
    source: str,
    target: str,
    edge_type: str,
) -> GraphEdge:
    return GraphEdge(
        id=f"edge-{key}",
        repository_id="repo1",
        source_node_id=source,
        target_node_id=target,
        edge_type=edge_type,
        created_at=_NOW,
    )


async def _seed(factory, *rows: Any) -> None:
    async with factory() as session:
        session.add_all(rows)
        await session.commit()


@pytest.mark.asyncio
async def test_calls_mode_never_uses_import_or_containment_edges(setup_mcp, factory):
    """A shorter mixed-layer route must not contaminate a calls chain."""
    from repowise.server.mcp_server import get_dependency_path

    entry_node, entry_symbol = _symbol_rows("entry", "pkg/entry.py", "b1_entry")
    middle_node, middle_symbol = _symbol_rows("middle", "pkg/middle.py", "b1_middle")
    sink_node, sink_symbol = _symbol_rows("sink", "pkg/sink.py", "b1_sink")
    await _seed(
        factory,
        entry_node,
        entry_symbol,
        middle_node,
        middle_symbol,
        sink_node,
        sink_symbol,
        _edge("entry-middle-call", entry_node.node_id, middle_node.node_id, "calls"),
        _edge("middle-sink-ref", middle_node.node_id, sink_node.node_id, "references"),
        # Both would make a one-hop path if calls mode admitted them.
        _edge("entry-sink-import", entry_node.node_id, sink_node.node_id, "imports"),
        _edge("entry-sink-defines", entry_node.node_id, sink_node.node_id, "defines"),
    )

    result = await get_dependency_path("b1_entry", "b1_sink", mode="calls")

    assert result["status"] == "found"
    assert result["distance"] == 2
    assert [hop["node"] for hop in result["path"]] == [
        entry_node.node_id,
        middle_node.node_id,
        sink_node.node_id,
    ]
    assert [hop["relationship"] for hop in result["path"][:-1]] == [
        "calls",
        "references",
    ]


@pytest.mark.asyncio
async def test_calls_mode_returns_ambiguity_candidates_without_picking_one(setup_mcp, factory):
    from repowise.server.mcp_server import get_dependency_path

    first_node, first_symbol = _symbol_rows("dup-a", "pkg/a.py", "b1_duplicate")
    second_node, second_symbol = _symbol_rows("dup-b", "pkg/b.py", "b1_duplicate")
    sink_node, sink_symbol = _symbol_rows("dup-sink", "pkg/sink.py", "b1_unique_sink")
    await _seed(
        factory,
        first_node,
        first_symbol,
        second_node,
        second_symbol,
        sink_node,
        sink_symbol,
    )

    result = await get_dependency_path("b1_duplicate", "b1_unique_sink", mode="calls")

    assert result["status"] == "ambiguous"
    assert result["endpoint"] == "source"
    assert result["match_count"] == 2
    assert result["path"] == []
    assert {candidate["symbol_id"] for candidate in result["candidates"]} == {
        first_node.node_id,
        second_node.node_id,
    }


@pytest.mark.asyncio
async def test_limit_paths_returns_distinct_shortest_chains_and_reports_more(setup_mcp, factory):
    from repowise.server.mcp_server import get_dependency_path

    symbols = [
        _symbol_rows(key, f"pkg/{key}.py", f"b1_{key}")
        for key in ("route_start", "route_a", "route_b", "route_c", "route_end")
    ]
    nodes = [pair[0] for pair in symbols]
    wiki_symbols = [pair[1] for pair in symbols]
    start, route_a, route_b, route_c, end = nodes
    await _seed(
        factory,
        *nodes,
        *wiki_symbols,
        _edge("route-start-a", start.node_id, route_a.node_id, "calls"),
        _edge("route-a-end", route_a.node_id, end.node_id, "calls"),
        _edge("route-start-b", start.node_id, route_b.node_id, "calls"),
        _edge("route-b-end", route_b.node_id, end.node_id, "calls"),
        _edge("route-start-c", start.node_id, route_c.node_id, "calls"),
        _edge("route-c-end", route_c.node_id, end.node_id, "calls"),
    )

    result = await get_dependency_path(
        start.node_id,
        end.node_id,
        mode="calls",
        limit_paths=2,
    )
    repeated = await get_dependency_path(
        start.node_id,
        end.node_id,
        mode="calls",
        limit_paths=2,
    )

    assert len(result["paths"]) == 2
    assert result["paths_truncated"] is True
    assert {path["distance"] for path in result["paths"]} == {2}
    chains = {tuple(hop["node"] for hop in path["path"]) for path in result["paths"]}
    assert len(chains) == 2
    assert result["path"] == result["paths"][0]["path"]
    assert repeated["paths"] == result["paths"]

    capped = await get_dependency_path(
        start.node_id,
        end.node_id,
        mode="calls",
        limit_paths=99,
    )
    assert capped["limit_paths"] == 5
    assert capped["limit_paths_requested"] == 99


@pytest.mark.asyncio
async def test_file_dependents_are_complete_ranked_paginated_and_test_filtered(
    setup_mcp, factory, engine
):
    from repowise.server.mcp_server import get_dependents

    target = _file_node("dep-target", "b1/target.py")
    direct_a = _file_node("dep-a", "b1/a.py", pagerank=0.2)
    direct_b = _file_node("dep-b", "b1/b.py", pagerank=0.1)
    transitive_c = _file_node("dep-c", "b1/c.py", pagerank=0.9)
    transitive_d = _file_node("dep-d", "b1/d.py", pagerank=0.3)
    test_bridge = _file_node("dep-test", "tests/test_b1_target.py", is_test=True)
    behind_test = _file_node("dep-behind-test", "b1/test_harness.py", pagerank=1.0)
    await _seed(
        factory,
        target,
        direct_a,
        direct_b,
        transitive_c,
        transitive_d,
        test_bridge,
        behind_test,
        _edge("dep-a-target", direct_a.node_id, target.node_id, "imports"),
        _edge("dep-b-target-import", direct_b.node_id, target.node_id, "imports"),
        _edge("dep-b-target-type", direct_b.node_id, target.node_id, "type_use"),
        _edge("dep-c-a", transitive_c.node_id, direct_a.node_id, "imports"),
        _edge("dep-d-a", transitive_d.node_id, direct_a.node_id, "imports"),
        _edge("dep-test-target", test_bridge.node_id, target.node_id, "imports"),
        _edge("dep-behind-test", behind_test.node_id, test_bridge.node_id, "imports"),
    )

    inbound_queries: list[str] = []

    def _record_inbound_query(_conn, _cursor, statement, _parameters, _context, _many):
        if "FROM graph_edges JOIN graph_nodes" in statement:
            inbound_queries.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", _record_inbound_query)
    try:
        first = await get_dependents(target.node_id, depth=2, limit=2)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _record_inbound_query)
    second = await get_dependents(target.node_id, depth=2, offset=2, limit=2)

    # One set-oriented inbound query per depth, never one query per node.
    assert len(inbound_queries) == 2
    assert first["total"] == 4
    assert first["counts_by_depth"] == {"1": 2, "2": 2}
    assert first["pagination"] == {
        "offset": 0,
        "limit": 2,
        "returned": 2,
        "has_more": True,
        "next_offset": 2,
    }
    # Two relation kinds beat every one-relation result despite lower PageRank.
    assert first["dependents"][0]["node"] == direct_b.node_id
    assert first["dependents"][0]["reference_count"] == 2
    page_one = {row["node"] for row in first["dependents"]}
    page_two = {row["node"] for row in second["dependents"]}
    assert page_one.isdisjoint(page_two)
    assert page_one | page_two == {
        direct_a.node_id,
        direct_b.node_id,
        transitive_c.node_id,
        transitive_d.node_id,
    }
    assert test_bridge.node_id not in page_one | page_two
    assert behind_test.node_id not in page_one | page_two

    with_tests = await get_dependents(target.node_id, depth=2, include_tests=True, limit=20)
    assert with_tests["total"] == 6
    assert {test_bridge.node_id, behind_test.node_id} <= {
        row["node"] for row in with_tests["dependents"]
    }


@pytest.mark.asyncio
async def test_dependents_total_is_not_silently_capped_to_the_page(setup_mcp, factory):
    from repowise.server.mcp_server import get_dependents

    target = _file_node("wide-target", "b1/wide_target.py")
    nodes = [_file_node(f"wide-{index}", f"b1/wide_{index:02d}.py") for index in range(31)]
    edges = [
        _edge(f"wide-{index}", node.node_id, target.node_id, "imports")
        for index, node in enumerate(nodes)
    ]
    await _seed(factory, target, *nodes, *edges)

    result = await get_dependents(target.node_id, limit=5)

    assert result["total"] == 31
    assert len(result["dependents"]) == 5
    assert result["pagination"]["has_more"] is True
    assert result["pagination"]["next_offset"] == 5


@pytest.mark.asyncio
async def test_symbol_dependents_use_symbol_relations_not_import_or_containment(setup_mcp, factory):
    from repowise.server.mcp_server import get_dependents

    target_node, target_symbol = _symbol_rows("sym-target", "pkg/target.py", "b1_handler")
    caller_node, caller_symbol = _symbol_rows("sym-caller", "pkg/caller.py", "b1_caller")
    ref_node, ref_symbol = _symbol_rows("sym-ref", "pkg/registry.py", "b1_registry")
    subclass_node, subclass_symbol = _symbol_rows("sym-sub", "pkg/sub.py", "B1Subclass")
    import_node, import_symbol = _symbol_rows("sym-import", "pkg/importer.py", "b1_importer")
    define_node, define_symbol = _symbol_rows("sym-define", "pkg/owner.py", "b1_owner")
    await _seed(
        factory,
        target_node,
        target_symbol,
        caller_node,
        caller_symbol,
        ref_node,
        ref_symbol,
        subclass_node,
        subclass_symbol,
        import_node,
        import_symbol,
        define_node,
        define_symbol,
        _edge("sym-call", caller_node.node_id, target_node.node_id, "calls"),
        _edge("sym-reference", ref_node.node_id, target_node.node_id, "references"),
        _edge("sym-extends", subclass_node.node_id, target_node.node_id, "extends"),
        _edge("sym-import", import_node.node_id, target_node.node_id, "imports"),
        _edge("sym-defines", define_node.node_id, target_node.node_id, "defines"),
    )

    result = await get_dependents("b1_handler", limit=20)

    nodes = {row["node"] for row in result["dependents"]}
    assert nodes == {caller_node.node_id, ref_node.node_id, subclass_node.node_id}
    assert import_node.node_id not in nodes
    assert define_node.node_id not in nodes


@pytest.mark.asyncio
async def test_dependents_ambiguous_symbol_returns_candidates(setup_mcp, factory):
    from repowise.server.mcp_server import get_dependents

    first_node, first_symbol = _symbol_rows("dep-dup-a", "pkg/one.py", "b1_dep_duplicate")
    second_node, second_symbol = _symbol_rows("dep-dup-b", "pkg/two.py", "b1_dep_duplicate")
    await _seed(factory, first_node, first_symbol, second_node, second_symbol)

    result = await get_dependents("b1_dep_duplicate")

    assert result["status"] == "ambiguous"
    assert result["match_count"] == 2
    assert result["dependents"] == []
    assert result["total"] == 0
    assert {candidate["symbol_id"] for candidate in result["candidates"]} == {
        first_node.node_id,
        second_node.node_id,
    }
