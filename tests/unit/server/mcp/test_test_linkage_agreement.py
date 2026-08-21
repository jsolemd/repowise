"""get_context and get_risk answer "does this file have tests" identically.

They used to answer it from two unrelated sources — the health engine's
index-time paired-filename column on one side, a live substring scan of the
test nodes on the other — and contradicted each other on the fixture repo and
on a real 751-file index alike. An agent that reads both in one turn cannot
tell which is lying, so the fix was one owner of the question
(``_test_linkage``) and both tools reading it.

The fixture repo is the contradiction in miniature: ``src/auth/service.py``
carries ``has_test_file=False`` in health metrics, while ``tests/test_service.py``
is a test node that imports it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from repowise.server.mcp_server._test_linkage import resolve_test_linkage

SERVICE = "src/auth/service.py"
TEST_FILE = "tests/test_service.py"
UNTESTED = "src/auth/middleware.py"


@pytest.mark.asyncio
async def test_context_and_risk_agree_on_every_shared_file(setup_mcp, health_data):
    from repowise.server.mcp_server import get_context, get_risk

    files = [SERVICE, "src/db/models.py", UNTESTED]
    context = await get_context(files, include=["health"])
    risk = await get_risk(files)

    for path in files:
        card = context["targets"][path]
        dossier = risk["targets"][path]
        assert card["tested"] is not dossier["test_gap"], path
        assert card.get("guarding_tests") == dossier.get("guarding_tests"), path
        assert card.get("possible_tests") == dossier.get("possible_tests"), path
        assert card["test_linkage_basis"] == dossier["test_linkage_basis"], path


@pytest.mark.asyncio
async def test_the_file_that_used_to_disagree_now_reports_its_guard(setup_mcp, health_data):
    from repowise.server.mcp_server import get_context, get_risk

    card = (await get_context([SERVICE], include=["health"]))["targets"][SERVICE]
    dossier = (await get_risk([SERVICE]))["targets"][SERVICE]

    # The import edge from the test file is the evidence; the paired-filename
    # heuristic that used to drive get_context never saw it.
    assert card["tested"] is True
    assert card["test_linkage_basis"] == "graph"
    assert card["guarding_tests"] == [TEST_FILE]
    assert dossier["test_gap"] is False
    # ...and the health block still reports its own column, unchanged, which is
    # exactly the value that used to be read as the answer to this question.
    assert card["health"]["has_test_file"] is False


@pytest.mark.asyncio
async def test_a_file_nothing_tests_says_so_in_both_tools(setup_mcp, health_data):
    from repowise.server.mcp_server import get_context, get_risk

    card = (await get_context([UNTESTED]))["targets"][UNTESTED]
    dossier = (await get_risk([UNTESTED]))["targets"][UNTESTED]

    assert card["tested"] is False
    assert card["test_linkage_basis"] == "none"
    assert "guarding_tests" not in card
    assert dossier["test_gap"] is True


@pytest.mark.asyncio
async def test_a_test_file_is_never_its_own_gap(setup_mcp, populated_db, session):
    linkage = await resolve_test_linkage(session, populated_db, TEST_FILE)

    assert linkage.basis == "self"
    assert linkage.tested is True


@pytest.mark.asyncio
async def test_naming_rung_matches_the_basename_not_a_substring(setup_mcp, populated_db, session):
    from repowise.core.persistence.models import GraphNode

    # ``test_service_helpers.py`` is not the paired test for ``service.py``.
    # The old substring scan counted it, which is how a file with no test at
    # all was reported as tested.
    session.add(
        GraphNode(
            id="gn-helpers",
            repository_id=populated_db,
            node_id="tests/test_middleware_helpers.py",
            node_type="file",
            language="python",
            is_test=True,
        )
    )
    await session.flush()

    linkage = await resolve_test_linkage(session, populated_db, UNTESTED)
    assert linkage.basis == "none"

    session.add(
        GraphNode(
            id="gn-paired",
            repository_id=populated_db,
            node_id="tests/test_middleware.py",
            node_type="file",
            language="python",
            is_test=True,
        )
    )
    await session.flush()

    linkage = await resolve_test_linkage(session, populated_db, UNTESTED)
    assert linkage.basis == "naming"
    assert linkage.tests == ["tests/test_middleware.py"]
    assert linkage.tested is False
    assert linkage.as_payload()["possible_tests"] == ["tests/test_middleware.py"]
    assert "guarding_tests" not in linkage.as_payload()


@pytest.mark.asyncio
async def test_coverage_outranks_the_graph(setup_mcp, populated_db, session):
    from repowise.core.persistence.crud import save_test_coverage

    await save_test_coverage(
        session,
        populated_db,
        [
            SimpleNamespace(
                test_id="tests/test_coverage_proven.py::test_it",
                test_file="tests/test_coverage_proven.py",
                file_path=SERVICE,
                covered_lines=[10, 11],
            )
        ],
        source_format="pytest-cov",
    )
    await session.flush()

    linkage = await resolve_test_linkage(session, populated_db, SERVICE)
    assert linkage.basis == "coverage"
    assert linkage.tests == ["tests/test_coverage_proven.py"]


@pytest.mark.asyncio
async def test_naming_only_evidence_does_not_clear_the_risk_gap(setup_mcp, populated_db, session):
    from repowise.core.persistence.models import GraphNode
    from repowise.server.mcp_server import get_context, get_risk

    session.add(
        GraphNode(
            id="gn-possible-middleware",
            repository_id=populated_db,
            node_id="tests/test_middleware.py",
            node_type="file",
            language="python",
            is_test=True,
        )
    )
    await session.flush()

    card = (await get_context([UNTESTED]))["targets"][UNTESTED]
    dossier = (await get_risk([UNTESTED]))["targets"][UNTESTED]

    assert card["tested"] is False
    assert dossier["test_gap"] is True
    assert card["possible_tests"] == ["tests/test_middleware.py"]
    assert dossier["possible_tests"] == card["possible_tests"]
    assert "guarding_tests" not in card
    assert "guarding_tests" not in dossier


@pytest.mark.parametrize(
    ("test_file", "subject"),
    [
        ("tests/test_chunker.py", "chunker"),
        ("pkg/chunker_test.go", "chunker"),
        ("src/chunker.spec.ts", "chunker"),
        ("src/chunker.test.tsx", "chunker"),
        ("src/main/ChunkerTest.java", "Chunker"),
        ("src/ChunkerSpecs.cs", "Chunker"),
        ("src/chunker.py", None),
        ("conftest.py", None),
    ],
)
def test_subject_stem_reads_conventions_from_the_registry(test_file, subject):
    """Every marker comes from the language registry, so ecosystems come free.

    A local ``("test_", "_test", ".spec.")`` list would need editing per
    ecosystem; core.test_paths exists because fourteen such copies disagreed.
    """
    from repowise.server.mcp_server._test_linkage import subject_stem

    assert subject_stem(test_file) == subject


@pytest.mark.asyncio
async def test_naming_rung_pairs_a_go_style_test(setup_mcp, populated_db, session):
    from repowise.core.persistence.models import GraphNode

    session.add(
        GraphNode(
            id="gn-go",
            repository_id=populated_db,
            node_id="internal/middleware_test.go",
            node_type="file",
            language="go",
            is_test=True,
        )
    )
    await session.flush()

    linkage = await resolve_test_linkage(session, populated_db, UNTESTED)
    assert linkage.basis == "naming"
    assert linkage.tests == ["internal/middleware_test.go"]
    assert linkage.tested is False


@pytest.mark.asyncio
async def test_naming_rung_suppresses_cross_package_stem_collisions(
    setup_mcp, populated_db, session
):
    from repowise.core.persistence.models import GraphNode

    session.add_all(
        [
            GraphNode(
                id="gn-concept-graph",
                repository_id=populated_db,
                node_id="conceptatlas/atlas/graph.py",
                node_type="file",
                language="python",
                is_test=False,
            ),
            GraphNode(
                id="gn-code-graph",
                repository_id=populated_db,
                node_id="codeatlas/doc_search/graph.py",
                node_type="file",
                language="python",
                is_test=False,
            ),
            GraphNode(
                id="gn-code-test-graph",
                repository_id=populated_db,
                node_id="codeatlas/tests/doc_search/test_graph.py",
                node_type="file",
                language="python",
                is_test=True,
            ),
        ]
    )
    await session.flush()

    linkage = await resolve_test_linkage(session, populated_db, "conceptatlas/atlas/graph.py")

    assert linkage.basis == "none"
    assert linkage.tested is False
    assert "possible_tests" not in linkage.as_payload()
