"""The six graph-backed patterns, each against its own planted instance.

The graph under test comes from a real traverse → parse → resolve pass
over the planted tree, so what these assert on are edges a resolver
actually produced, not a hand-written adjacency that agrees with the
patterns by construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repowise.core.analysis.patterns import (
    PATTERN_NAMES,
    SymbolGraph,
    UnknownPatternError,
    catalogue,
    definition_of,
    run_pattern,
)
from repowise.core.analysis.patterns.duplicate_signatures import (
    normalize_name,
    parameter_count,
)
from repowise.core.analysis.patterns.hub_functions import _percentile_floor

from . import toy_clone_repo as toy


@pytest.fixture
def graph(tmp_path: Path) -> SymbolGraph:
    return toy.symbol_graph(toy.build(tmp_path / "toy"))


def _keys(result) -> list[str]:
    return [m.key for m in result.matches]


# ---------------------------------------------------------------------------
# Catalogue contract
# ---------------------------------------------------------------------------


def test_all_six_patterns_are_registered() -> None:
    assert set(PATTERN_NAMES) == {
        "duplicate_signatures",
        "orphan_exports",
        "hub_functions",
        "isolated_siblings",
        "reuse_candidates",
        "bridge_functions",
    }


@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_every_response_carries_its_definition(name: str, graph: SymbolGraph) -> None:
    payload = run_pattern(name, graph).as_dict()

    definition = payload["definition"]
    assert definition["name"] == name
    # Each field is what stops the caller supplying their own definition:
    # what was asked, the exact rule, how it was ordered, what it is not,
    # and the parameter values this particular run used.
    for field in ("question", "predicate", "ranking", "not_this"):
        assert definition[field].strip(), f"{name}.{field} is empty"
    assert isinstance(definition["params"], dict)
    assert definition["params"], f"{name} reported no parameters"


def test_definitions_state_their_effective_parameters(graph: SymbolGraph) -> None:
    """An overridden threshold travels with the answer it changed."""
    payload = run_pattern("hub_functions", graph, min_callers=2).as_dict()

    assert payload["definition"]["params"]["min_callers"] == 2
    assert payload["definition"]["params"]["effective_caller_threshold"] >= 2


def test_unknown_pattern_names_the_catalogue(graph: SymbolGraph) -> None:
    with pytest.raises(UnknownPatternError) as excinfo:
        run_pattern("hub_funktions", graph)

    assert excinfo.value.available == PATTERN_NAMES
    assert "duplicate_signatures" in str(excinfo.value)


def test_catalogue_serves_every_definition() -> None:
    entries = catalogue()
    assert [e["name"] for e in entries] == list(PATTERN_NAMES)
    assert all(e["not_this"] for e in entries)


def test_misspelled_parameters_fail_loudly(graph: SymbolGraph) -> None:
    """A typo'd threshold must not silently return the default-threshold answer."""
    with pytest.raises(TypeError):
        run_pattern("hub_functions", graph, min_caller=2)


# ---------------------------------------------------------------------------
# The six plants
# ---------------------------------------------------------------------------


def test_duplicate_signatures_finds_the_planted_pair(graph: SymbolGraph) -> None:
    result = run_pattern("duplicate_signatures", graph)

    assert _keys(result) == ["computetotal/2"]
    match = result.matches[0].as_dict()
    assert match["arity"] == 2
    assert match["file_count"] == 2
    assert {d["file"] for d in match["declarations"]} == {"alpha/report.py", "beta/report.py"}


def test_orphan_exports_finds_the_planted_orphan(graph: SymbolGraph) -> None:
    result = run_pattern("orphan_exports", graph)

    assert "core/util.py::orphan_helper" in _keys(result)
    match = next(m for m in result.matches if m.key == "core/util.py::orphan_helper")
    assert match.as_dict()["callers"] == 0


def test_orphan_exports_excludes_used_symbols(graph: SymbolGraph) -> None:
    result = run_pattern("orphan_exports", graph)

    # Called from two other directories, so it can never be an orphan.
    assert "core/util.py::shared_helper" not in _keys(result)


def test_hub_functions_finds_the_planted_hub(graph: SymbolGraph) -> None:
    result = run_pattern("hub_functions", graph, min_callers=2)

    assert _keys(result) == ["core/util.py::shared_helper"]
    match = result.matches[0].as_dict()
    assert match["callers"] == 2
    assert match["caller_directories"] == 2


def test_isolated_siblings_finds_the_planted_loner(graph: SymbolGraph) -> None:
    result = run_pattern("isolated_siblings", graph)

    assert "core/lonely.py::loner" in _keys(result)
    match = next(m for m in result.matches if m.key == "core/lonely.py::loner")
    assert match.as_dict()["siblings"] == 2
    # Its file-mates touch each other, so neither of them is isolated.
    assert "core/lonely.py::used_one" not in _keys(result)
    assert "core/lonely.py::used_two" not in _keys(result)


def test_reuse_candidates_finds_the_planted_shared_helper(graph: SymbolGraph) -> None:
    result = run_pattern("reuse_candidates", graph)

    assert _keys(result) == ["core/util.py::shared_helper"]
    match = result.matches[0].as_dict()
    assert sorted(match["external_directories"]) == ["alpha", "beta"]
    assert match["external_directory_count"] == 2


def test_bridge_functions_finds_the_planted_bridge(graph: SymbolGraph) -> None:
    result = run_pattern("bridge_functions", graph)

    assert _keys(result) == ["core/gateway.py::bridge"]
    match = result.matches[0].as_dict()
    assert match["bridged_file_pairs"] == [
        {"from_file": "alpha/report.py", "to_file": "core/sink.py"}
    ]


def test_bridge_is_not_reported_once_the_two_files_talk_directly(tmp_path: Path) -> None:
    """The bridge claim is 'traffic must pass through here'; make it false."""
    root = toy.build(tmp_path / "toy")
    (root / "alpha/report.py").write_text(
        (root / "alpha/report.py").read_text() + "\n\ndef alpha_direct(payload):\n"
        "    from core.sink import sink_write\n\n"
        "    return sink_write(payload)\n",
        encoding="utf-8",
    )

    result = run_pattern("bridge_functions", toy.symbol_graph(root))

    assert "core/gateway.py::bridge" not in _keys(result)


# ---------------------------------------------------------------------------
# Shape and filtering rules
# ---------------------------------------------------------------------------


def test_synthetic_module_scopes_are_never_reported(graph: SymbolGraph) -> None:
    """They own a file's top-level calls; they are edge endpoints, not findings."""
    for name in PATTERN_NAMES:
        keys = _keys(run_pattern(name, graph))
        assert not any("__module__" in key for key in keys), name


def test_module_scope_still_counts_as_a_tie_to_the_file(tmp_path: Path) -> None:
    """A helper invoked from top level is used by its file, not isolated in it."""
    root = toy.build(tmp_path / "toy")
    (root / "core/lonely.py").write_text(
        (root / "core/lonely.py").read_text() + '\n\nSTAMP = loner("x", {})\n',
        encoding="utf-8",
    )

    result = run_pattern("isolated_siblings", toy.symbol_graph(root))

    assert "core/lonely.py::loner" not in _keys(result)


def test_results_are_limited_and_the_truncation_is_stated(graph: SymbolGraph) -> None:
    full = run_pattern("orphan_exports", graph)
    assert len(full.matches) > 1

    payload = full.as_dict(limit=1)
    assert len(payload["matches"]) == 1
    assert payload["summary"]["total_matches"] == len(full.matches)
    assert payload["summary"]["truncated"] is True


def test_ordering_is_deterministic(graph: SymbolGraph) -> None:
    first = _keys(run_pattern("orphan_exports", graph))
    second = _keys(run_pattern("orphan_exports", graph))
    assert first == second


# ---------------------------------------------------------------------------
# Graph view construction
# ---------------------------------------------------------------------------


def test_symbol_test_membership_comes_from_the_file(tmp_path: Path) -> None:
    """``is_test`` is stamped on file nodes, so symbols must inherit it."""
    nodes = [
        {"node_id": "tests/test_x.py", "node_type": "file", "is_test": True},
        {
            "node_id": "tests/test_x.py::helper",
            "node_type": "symbol",
            "file_path": "tests/test_x.py",
            "name": "helper",
            "kind": "function",
            "visibility": "public",
            "start_line": 1,
            "end_line": 9,
        },
    ]
    graph = SymbolGraph.from_rows(nodes, [])

    assert graph.symbols["tests/test_x.py::helper"].is_test is True
    assert graph.candidates() == []
    assert len(graph.candidates(include_tests=True)) == 1


def test_unknown_edge_types_are_ignored_not_counted() -> None:
    nodes = [
        {"node_id": "a.py", "node_type": "file"},
        {
            "node_id": "a.py::f",
            "node_type": "symbol",
            "file_path": "a.py",
            "name": "f",
            "kind": "function",
            "visibility": "public",
            "start_line": 1,
            "end_line": 5,
        },
        {
            "node_id": "a.py::g",
            "node_type": "symbol",
            "file_path": "a.py",
            "name": "g",
            "kind": "function",
            "visibility": "public",
            "start_line": 7,
            "end_line": 11,
        },
    ]
    edges = [
        {"source_node_id": "a.py::f", "target_node_id": "a.py::g", "edge_type": "invented_later"},
    ]
    graph = SymbolGraph.from_rows(nodes, edges)

    assert graph.in_degree("a.py::g") == 0


# ---------------------------------------------------------------------------
# Signature parsing
# ---------------------------------------------------------------------------


def test_normalize_name_folds_case_and_separators() -> None:
    assert normalize_name("parseConfig") == normalize_name("parse_config")
    assert normalize_name("Parse-Config") == "parseconfig"
    assert normalize_name("parse_config2") == "parseconfig2"


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        ("def f()", 0),
        ("def f(a)", 1),
        ("def f(a, b)", 2),
        ("def f(a, b=(1, 2))", 2),
        ("def f(a: dict[str, int], b: list[int])", 2),
        ("def f(   )", 0),
        ("func f(a int, b int) error", 2),
        ("no parentheses here", None),
    ],
)
def test_parameter_count(signature: str, expected: int | None) -> None:
    assert parameter_count(signature) == expected


def test_unparseable_signatures_group_only_with_each_other() -> None:
    nodes = [
        {"node_id": "a.py", "node_type": "file"},
        {"node_id": "b.py", "node_type": "file"},
        {
            "node_id": "a.py::thing",
            "node_type": "symbol",
            "file_path": "a.py",
            "name": "thing",
            "kind": "function",
            "visibility": "public",
            "signature": "thing",
            "start_line": 1,
            "end_line": 3,
        },
        {
            "node_id": "b.py::thing",
            "node_type": "symbol",
            "file_path": "b.py",
            "name": "thing",
            "kind": "function",
            "visibility": "public",
            "signature": "def thing(a, b)",
            "start_line": 1,
            "end_line": 3,
        },
    ]
    result = run_pattern("duplicate_signatures", SymbolGraph.from_rows(nodes, []))

    assert result.matches == ()


def test_percentile_floor_is_nearest_rank() -> None:
    assert _percentile_floor([], 95.0) == 0
    assert _percentile_floor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 100.0) == 10
    assert _percentile_floor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50.0) == 5
    assert _percentile_floor([4], 95.0) == 4


def test_definition_of_rejects_unknown_names() -> None:
    with pytest.raises(UnknownPatternError):
        definition_of("nope")


def test_a_scope_that_never_coheres_reports_nothing(tmp_path: Path) -> None:
    """Edge sparsity is not misfiling — see the measurement in _scope_coheres."""
    root = toy.build(tmp_path / "toy")
    (root / "core/loose.py").write_text(
        "def one(a):\n    return a + 1\n\n\n"
        "def two(b):\n    return b + 2\n\n\n"
        "def three(c):\n    return c + 3\n",
        encoding="utf-8",
    )

    keys = [m.key for m in run_pattern("isolated_siblings", toy.symbol_graph(root)).matches]

    # None of the three touch each other, so the file tells us nothing about
    # any one of them. The planted loner still reports: its siblings cohere.
    assert not any(key.startswith("core/loose.py") for key in keys)
    assert "core/lonely.py::loner" in keys


def test_kinds_that_cannot_call_are_out_of_scope_by_default(tmp_path: Path) -> None:
    """A constant in a cohesive module is data, not a misfiling."""
    root = toy.build(tmp_path / "toy")
    (root / "core/lonely.py").write_text(
        (root / "core/lonely.py").read_text() + "\n\nLONELY_CONSTANT = 42\n",
        encoding="utf-8",
    )
    graph = toy.symbol_graph(root)

    default = [m.key for m in run_pattern("isolated_siblings", graph).matches]
    widened = [
        m.key
        for m in run_pattern(
            "isolated_siblings", graph, kinds=["function", "method", "class", "constant"]
        ).matches
    ]

    assert "core/lonely.py::LONELY_CONSTANT" not in default
    assert "core/lonely.py::LONELY_CONSTANT" in widened
