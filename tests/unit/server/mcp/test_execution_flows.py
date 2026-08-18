"""get_execution_flows: stable handles, a reproducible list, and the drill."""

from __future__ import annotations

import json

import pytest

from repowise.core.persistence.models import GraphEdge, GraphNode
from repowise.server.mcp_server._flow_ids import parse_flow_id

_ENTRY_A = "src/svc/alpha.py::run_alpha"
_ENTRY_B = "src/svc/beta.py::run_beta"
_HOP = "src/svc/alpha.py::_alpha_step"


def _symbol(node_id: str, *, score: float | None, start: int, end: int) -> dict:
    meta = {} if score is None else {"entry_point_score": score, "label": "svc"}
    return {
        "node_id": node_id,
        "node_type": "symbol",
        "language": "python",
        "community_id": 7,
        "community_meta_json": json.dumps(meta),
        "kind": "function",
        "name": node_id.split("::")[-1],
        "qualified_name": node_id.replace("/", ".").replace("::", ".").removesuffix(".py"),
        "file_path": node_id.split("::")[0],
        "start_line": start,
        "end_line": end,
    }


@pytest.fixture
async def flow_graph(session, populated_db):
    """Two entry points scored identically, and one call hop to walk.

    The tie is the point: it is the case where the stock ordering had nothing
    left to sort on and fell back to whatever the rows happened to be in.
    """
    rid = populated_db
    rows = [
        _symbol(_ENTRY_A, score=0.99, start=10, end=40),
        _symbol(_ENTRY_B, score=0.99, start=5, end=25),
        _symbol(_HOP, score=None, start=50, end=70),
    ]
    for index, row in enumerate(rows):
        session.add(GraphNode(id=f"flow-n{index}", repository_id=rid, **row))
    session.add(
        GraphEdge(
            id="flow-e0",
            repository_id=rid,
            source_node_id=_ENTRY_A,
            target_node_id=_HOP,
            edge_type="calls",
            confidence=0.95,
            resolution_origin="self_scope",
        )
    )
    await session.commit()
    return rid


async def _flows(**kwargs):
    from repowise.server.mcp_server.tool_flows import get_execution_flows

    return await get_execution_flows(**kwargs)


def _by_entry(result: dict, entry_point: str) -> dict:
    return next(f for f in result["flows"] if f["entry_point"] == entry_point)


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------


async def test_every_flow_carries_a_handle(setup_mcp, flow_graph):
    result = await _flows(top_n=2)
    assert result["flows"]
    for flow in result["flows"]:
        parsed = parse_flow_id(flow["flow_id"])
        assert parsed is not None
        assert parsed.entry_point == flow["entry_point"]
        assert parsed.generation == result["index_generation"]


async def test_the_same_request_returns_the_same_handles(setup_mcp, flow_graph):
    first = await _flows(top_n=5)
    second = await _flows(top_n=5)
    assert [f["flow_id"] for f in first["flows"]] == [f["flow_id"] for f in second["flows"]]


async def test_tied_scores_do_not_reorder_between_calls(setup_mcp, flow_graph):
    """Score alone is not a total order; two equal flows must not swap."""
    orders = {tuple(f["entry_point"] for f in (await _flows(top_n=2))["flows"]) for _ in range(5)}
    assert len(orders) == 1
    # And the tie breaks on the entry point, so the order is predictable.
    order = orders.pop()
    assert list(order) == sorted(order)


async def test_a_handle_does_not_depend_on_the_depth_it_was_seen_at(setup_mcp, flow_graph):
    """Depth is a view of a flow, not part of which flow it is."""
    shallow = _by_entry(await _flows(top_n=2, max_depth=1), _ENTRY_A)
    deep = _by_entry(await _flows(top_n=2, max_depth=8), _ENTRY_A)
    assert shallow["flow_id"] == deep["flow_id"]


# ---------------------------------------------------------------------------
# The drill
# ---------------------------------------------------------------------------


async def test_a_handle_re_requests_its_own_flow(setup_mcp, flow_graph):
    listed = _by_entry(await _flows(top_n=2), _ENTRY_A)
    drilled = await _flows(flow_id=listed["flow_id"])

    assert drilled["total_entry_points"] == 1
    assert drilled["flows"][0]["entry_point"] == _ENTRY_A
    assert drilled["flow_id"] == listed["flow_id"]
    assert drilled["generation_changed"] is False


async def test_the_drill_resolves_every_hop_to_a_file_and_lines(setup_mcp, flow_graph):
    listed = _by_entry(await _flows(top_n=2), _ENTRY_A)
    steps = (await _flows(flow_id=listed["flow_id"], depth=4))["flows"][0]["steps"]

    assert [s["index"] for s in steps] == list(range(len(steps)))
    assert steps[0]["node_id"] == _ENTRY_A
    assert steps[0]["file"] == "src/svc/alpha.py"
    assert (steps[0]["start_line"], steps[0]["end_line"]) == (10, 40)
    assert steps[0]["kind"] == "function"
    assert all(s["resolved"] for s in steps)

    hop = steps[1]
    assert hop["node_id"] == _HOP
    assert (hop["start_line"], hop["end_line"]) == (50, 70)
    # Provenance of the edge that reached this hop.
    assert hop["via"] == "self_scope"
    # The entry point was not reached by an edge, so it claims no provenance.
    assert "via" not in steps[0]


async def test_the_list_view_stays_lean(setup_mcp, flow_graph):
    """Steps are what a drill is for; ten flows' worth would bloat every list."""
    result = await _flows(top_n=2)
    assert all("steps" not in flow for flow in result["flows"])


async def test_the_drill_honours_its_own_depth(setup_mcp, flow_graph):
    listed = _by_entry(await _flows(top_n=2), _ENTRY_A)
    shallow = await _flows(flow_id=listed["flow_id"], depth=1)
    assert len(shallow["flows"][0]["steps"]) == 2  # entry point + one hop


# ---------------------------------------------------------------------------
# Handles that have outlived their index
# ---------------------------------------------------------------------------


async def test_a_handle_from_an_older_generation_still_answers_and_says_so(setup_mcp, flow_graph):
    stale = f"flow:000000000000:{_ENTRY_A}"
    result = await _flows(flow_id=stale)

    assert result["generation_changed"] is True
    assert result["requested_generation"] == "000000000000"
    assert result["index_generation"] != "000000000000"
    assert "generation_hint" in result
    # It answers rather than refusing — the trace below is freshly walked.
    assert result["flows"][0]["entry_point"] == _ENTRY_A
    assert result["flows"][0]["steps"]
    # And the flow is re-handed a current handle.
    assert parse_flow_id(result["flows"][0]["flow_id"]).generation == result["index_generation"]


async def test_a_handle_naming_a_vanished_entry_point_reports_both_facts(setup_mcp, flow_graph):
    result = await _flows(flow_id="flow:000000000000:src/gone.py::removed")
    assert "error" in result
    assert result["generation_changed"] is True
    assert result["requested_generation"] == "000000000000"
    assert result["current_generation"]


async def test_a_malformed_handle_is_refused_by_name(setup_mcp, flow_graph):
    result = await _flows(flow_id="nonsense")
    assert "Not a flow id" in result["error"]
    assert result["flow_id"] == "nonsense"
    assert "flows" not in result


# ---------------------------------------------------------------------------
# The stock envelope
# ---------------------------------------------------------------------------


async def test_the_existing_keys_are_untouched(setup_mcp, flow_graph):
    flow = _by_entry(await _flows(top_n=2), _ENTRY_A)
    assert set(flow) >= {
        "entry_point",
        "entry_point_name",
        "entry_point_score",
        "trace",
        "depth",
        "crosses_community",
        "communities_visited",
        "termination",
    }
