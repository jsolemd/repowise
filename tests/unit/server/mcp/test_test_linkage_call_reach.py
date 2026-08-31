"""The test-linkage graph rung reads the call graph, not just the import edge.

``get_risk`` used to carry two definitions of "a test reaches this file". PR
mode asked ``test_reachability``'s three-hop call walk; the per-file rung asked
whether a test file *imports* the file, one hop. Upstream measured both against
a real ``coverage run --contexts=test``: the import relation finds 19.5% of the
files a test provably executed, the call walk 27.7%, and the call walk is the
more precise of the two (91.7% against 72.1%). The gap is structural — a test
that calls a service that calls the changed helper imports neither of them, so
no import query can ever see it.

These tests pin the swap and the two things it must not move: the vocabulary
(call-graph reach is *proven* linkage, so it clears the gap; a matching
basename is not, so it never does) and the served directive keys.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from repowise.core.persistence.models import GraphEdge, GraphNode
from repowise.server.mcp_server._test_linkage import resolve_test_linkage

_NOW = datetime.now(UTC)

#: Three call hops from the test to the target, and no import edge anywhere on
#: the path: ``test_checkout_flow`` -> ``handle`` -> ``charge`` -> ``settle``.
#: The old rung called ``billing/ledger.py`` untested because nothing named
#: ``test_ledger.py`` exists and no test imports it.
LEDGER = "src/billing/ledger.py"
FLOW_TEST = "tests/test_checkout_flow.py"


async def _seed_call_chain(session, repo_id: str) -> None:
    """A test that only ever reaches the target through two intermediaries."""
    files = {
        LEDGER: False,
        "src/billing/charges.py": False,
        "src/api/checkout.py": False,
        FLOW_TEST: True,
    }
    for index, (path, is_test) in enumerate(files.items()):
        session.add(
            GraphNode(
                id=f"cr-file-{index}",
                repository_id=repo_id,
                node_id=path,
                node_type="file",
                language="python",
                is_test=is_test,
            )
        )
    symbols = {
        f"{LEDGER}::settle": LEDGER,
        "src/billing/charges.py::charge": "src/billing/charges.py",
        "src/api/checkout.py::handle": "src/api/checkout.py",
        f"{FLOW_TEST}::test_checkout_flow": FLOW_TEST,
    }
    for index, (symbol_id, owner) in enumerate(symbols.items()):
        session.add(
            GraphNode(
                id=f"cr-sym-{index}",
                repository_id=repo_id,
                node_id=symbol_id,
                node_type="symbol",
                language="python",
            )
        )
        session.add(
            GraphEdge(
                id=f"cr-def-{index}",
                repository_id=repo_id,
                source_node_id=owner,
                target_node_id=symbol_id,
                edge_type="defines",
                created_at=_NOW,
            )
        )
    calls = [
        (f"{FLOW_TEST}::test_checkout_flow", "src/api/checkout.py::handle"),
        ("src/api/checkout.py::handle", "src/billing/charges.py::charge"),
        ("src/billing/charges.py::charge", f"{LEDGER}::settle"),
    ]
    for index, (caller, callee) in enumerate(calls):
        session.add(
            GraphEdge(
                id=f"cr-call-{index}",
                repository_id=repo_id,
                source_node_id=caller,
                target_node_id=callee,
                edge_type="calls",
                resolution_origin="import_binding",
                created_at=_NOW,
            )
        )
    await session.flush()


@pytest.mark.asyncio
async def test_three_call_hops_are_proven_linkage_not_a_possibility(
    setup_mcp, populated_db, session
):
    """The recall the swap was for: reached at depth 3, and reported as proof."""
    await _seed_call_chain(session, populated_db)

    linkage = await resolve_test_linkage(session, populated_db, LEDGER)

    assert linkage.basis == "graph"
    assert linkage.via == "call-graph"
    assert linkage.tested is True
    assert linkage.tests == [FLOW_TEST]

    payload = linkage.as_payload()
    assert payload["guarding_tests"] == [FLOW_TEST]
    assert payload["test_linkage_via"] == "call-graph"
    assert "possible_tests" not in payload


@pytest.mark.asyncio
async def test_the_same_file_reads_untested_without_the_call_graph(
    setup_mcp, populated_db, session
):
    """Same fixture minus the call edges: nothing else in the ladder sees it.

    This is the old rung's answer, and the reason the recall table reads 19.5%.
    No test imports ``ledger.py`` and nothing is named ``test_ledger.py``, so
    every other rung is silent — the call walk is the only one that can speak.
    """
    await _seed_call_chain(session, populated_db)
    for index in range(3):
        await session.delete(await session.get(GraphEdge, f"cr-call-{index}"))
    await session.flush()

    linkage = await resolve_test_linkage(session, populated_db, LEDGER)

    assert linkage.basis == "none"
    assert linkage.tested is False


@pytest.mark.asyncio
async def test_both_get_risk_modes_now_name_the_same_guard(setup_mcp, populated_db, session):
    """The two definitions of "tested" that this unit collapsed into one.

    PR mode has read the call walk since upstream added it; the per-file rung
    had not. Assert they now agree on the file that only the call walk can see,
    and that each keeps its own basis vocabulary while doing so.
    """
    from repowise.server.mcp_server import get_context, get_risk

    await _seed_call_chain(session, populated_db)

    dossier = (await get_risk([LEDGER]))["targets"][LEDGER]
    card = (await get_context([LEDGER]))["targets"][LEDGER]

    assert dossier["test_gap"] is False
    assert dossier["guarding_tests"] == [FLOW_TEST]
    assert card["tested"] is True
    assert card["test_linkage_basis"] == dossier["test_linkage_basis"] == "graph"
    assert card["test_linkage_via"] == dossier["test_linkage_via"] == "call-graph"

    pr = await get_risk([LEDGER], changed_files=[LEDGER])
    guarding = pr["pr_blast_radius"]["guarding_tests"]
    assert FLOW_TEST in guarding["tests_to_run"]


@pytest.mark.asyncio
async def test_a_matching_basename_alone_is_still_only_possible(setup_mcp, populated_db, session):
    """The B1 ruling survives the swap: naming similarity is never proof.

    ``test_ledger.py`` names the file and reaches nothing in the graph. It has
    to stay ``possible_tests`` on the ``naming`` basis, keep ``tested`` False,
    and leave ``test_gap`` standing — a run-list entry, not a guard.
    """
    from repowise.server.mcp_server import get_context, get_risk

    await _seed_call_chain(session, populated_db)
    for index in range(3):
        await session.delete(await session.get(GraphEdge, f"cr-call-{index}"))
    session.add(
        GraphNode(
            id="cr-named-only",
            repository_id=populated_db,
            node_id="tests/test_ledger.py",
            node_type="file",
            language="python",
            is_test=True,
        )
    )
    await session.flush()

    linkage = await resolve_test_linkage(session, populated_db, LEDGER)
    assert linkage.basis == "naming"
    assert linkage.via is None
    assert linkage.tested is False

    payload = linkage.as_payload()
    assert payload["possible_tests"] == ["tests/test_ledger.py"]
    assert "guarding_tests" not in payload
    assert "test_linkage_via" not in payload

    dossier = (await get_risk([LEDGER]))["targets"][LEDGER]
    card = (await get_context([LEDGER]))["targets"][LEDGER]
    assert dossier["test_gap"] is True
    assert card["tested"] is False
    assert card["possible_tests"] == ["tests/test_ledger.py"]


@pytest.mark.asyncio
async def test_the_import_tier_still_answers_where_the_call_walk_cannot(
    setup_mcp, populated_db, session
):
    """Nothing that was proven before may come back as merely possible.

    ``src/auth/service.py`` is guarded by a test that imports it and calls
    nothing the graph records. The call walk is silent, so the import relation
    this rung has always used has to answer — same basis, same guard, and the
    tier says which one spoke.
    """
    linkage = await resolve_test_linkage(session, populated_db, "src/auth/service.py")

    assert linkage.basis == "graph"
    assert linkage.via == "import-graph"
    assert linkage.tested is True
    assert linkage.tests == ["tests/test_service.py"]


@pytest.mark.asyncio
async def test_the_walk_needs_all_three_hops(setup_mcp, populated_db, session):
    """The reach really is three hops, so the depth is what buys it.

    Pinned against the shared walk directly: at ``call_depth=2`` the chain runs
    out one call short of the test and the target is unanswered, at 3 it lands.
    Without this the first test would pass just as well on a one-hop fixture.
    """
    from repowise.core.analysis.test_reachability import tests_reaching_by_tier

    await _seed_call_chain(session, populated_db)

    shallow = await tests_reaching_by_tier(
        session, populated_db, [LEDGER], call_depth=2, import_depth=0
    )
    deep = await tests_reaching_by_tier(
        session, populated_db, [LEDGER], call_depth=3, import_depth=0
    )

    assert LEDGER not in shallow
    assert deep[LEDGER].tests == [FLOW_TEST]
    assert deep[LEDGER].via == "call-graph"


@pytest.mark.asyncio
async def test_a_failing_call_walk_falls_through_instead_of_accusing(
    setup_mcp, populated_db, session, monkeypatch
):
    """A broken walk must not become "this file is untested".

    It degrades to the tier below rather than raising or reporting a gap. The
    fixture is the file the call walk *does* answer, with an import edge from a
    second test added underneath it, so the assertion can tell a degraded walk
    from a walk that was never patched: unpatched this file reports
    ``call-graph`` and ``test_checkout_flow``.
    """
    import repowise.core.analysis.test_reachability as reachability

    await _seed_call_chain(session, populated_db)
    session.add(
        GraphNode(
            id="cr-importer",
            repository_id=populated_db,
            node_id="tests/test_ledger_import.py",
            node_type="file",
            language="python",
            is_test=True,
        )
    )
    session.add(
        GraphEdge(
            id="cr-import-edge",
            repository_id=populated_db,
            source_node_id="tests/test_ledger_import.py",
            target_node_id=LEDGER,
            edge_type="imports",
            created_at=_NOW,
        )
    )
    await session.flush()

    healthy = await resolve_test_linkage(session, populated_db, LEDGER)
    assert healthy.via == "call-graph" and healthy.tests == [FLOW_TEST]

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("graph_edges unavailable")

    monkeypatch.setattr(reachability, "tests_reaching_by_tier", _boom)

    degraded = await resolve_test_linkage(session, populated_db, LEDGER)

    assert degraded.basis == "graph"
    assert degraded.via == "import-graph"
    assert degraded.tested is True
    assert degraded.tests == ["tests/test_ledger_import.py"]


@pytest.mark.asyncio
async def test_pr_mode_basis_vocabulary_is_untouched(setup_mcp, populated_db, session):
    """``tests_to_run_basis`` keeps upstream's domain and upstream's meaning.

    The swap moved the per-file rung's evidence source. It did not touch
    ``test_impact``, so a graph-derived recommendation is still ``inferred``
    there and a coverage-derived one still ``measured``. Nothing that was
    measured became inferred, and no graph reach was promoted to measured.
    """
    from repowise.server.mcp_server import get_risk

    await _seed_call_chain(session, populated_db)

    result = await get_risk([LEDGER], changed_files=[LEDGER])
    directive = result["directive"]

    assert {"may_break", "missing_cochanges", "tests_to_run"} <= directive.keys()
    assert directive["tests_to_run_basis"] in {"measured", "inferred", "none"}
    # No coverage map in the fixture, so the only evidence is the graph walk —
    # and the graph walk is inferred on this surface, by upstream's design.
    assert directive["tests_to_run_basis"] == "inferred"
    assert isinstance(directive["tests_to_run"], list)
    assert isinstance(directive["may_break"], list)
    assert isinstance(directive["missing_cochanges"], list)
