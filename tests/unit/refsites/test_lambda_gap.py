"""The store records occurrences the dependency graph structurally cannot.

This is the fixture the whole package exists for. ``widget.ts`` declares
``makeUser`` inside an anonymous arrow passed to ``describe`` and calls it
inside another arrow passed to ``it``. The parse layer deletes a symbol whose
AST ancestor is an anonymous callable, so ``makeUser`` never becomes a symbol,
never becomes a graph node, and the call to it never becomes an edge. The name
is plainly in the file; the graph cannot see it.

The second half of the file measures the weaker version of the same loss: a
graph edge is unique per ``(source, target, type)``, so repeated calls collapse
and the edge count undercounts the call sites.
"""

from __future__ import annotations

from pathlib import Path

from repowise.core.ingestion.graph.builder import GraphBuilder
from repowise.core.ingestion.parser import ASTParser
from repowise.core.ingestion.traverser import FileTraverser
from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.taxonomy import TIER_AST, TIER_TEXTUAL, ReferenceKind


def _build_graph(repo: Path):
    traverser = FileTraverser(repo)
    parser = ASTParser()
    builder = GraphBuilder(repo)
    sources: dict[str, bytes] = {}
    for file_info in traverser.traverse():
        raw = Path(file_info.abs_path).read_bytes()
        sources[file_info.path] = raw
        builder.add_file(parser.parse_file(file_info, raw))
    builder.set_source_map(sources)
    return builder.build()


def _call_edges(graph, *, into: str) -> list[tuple[str, str]]:
    return [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data.get("edge_type") == "calls" and target == into
    ]


def test_graph_has_no_node_and_no_edge_for_a_lambda_scoped_symbol(toy_repo):
    """Baseline: the gap is real, not assumed."""
    graph = _build_graph(toy_repo)

    assert not [node for node in graph.nodes if "makeUser" in node]
    assert not [
        (source, target)
        for source, target, data in graph.edges(data=True)
        if data.get("edge_type") == "calls" and "makeUser" in f"{source}{target}"
    ]


def test_store_records_both_lambda_scoped_sites(toy_repo):
    """The store keeps the call the graph drops, and recovers the definition."""
    result = extract_repository(toy_repo)
    sites = sorted(
        (site for site in result.sites if site.name == "makeUser"),
        key=lambda site: site.start_line,
    )

    assert [(site.start_line, str(site.kind), site.tier) for site in sites] == [
        # The declaration, deleted by the parse layer, recovered by the sweep.
        (10, str(ReferenceKind.IDENTIFIER), TIER_TEXTUAL),
        # The call, which the parse layer *does* see and the graph then drops
        # because the callee symbol no longer exists to be an edge endpoint.
        (12, str(ReferenceKind.CALL), TIER_AST),
    ]
    assert all(site.range_exact for site in sites)
    # Nothing defines it as far as the index is concerned, so both are unbound.
    assert all(site.target_symbol_id is None for site in sites)
    # And both are honest about that: neither claims rewrite confidence.
    assert all(site.confidence <= 0.25 for site in sites)


def test_graph_undercounts_call_sites_that_the_store_keeps(toy_repo):
    """Six edges for eleven occurrences — the collapse, measured."""
    graph = _build_graph(toy_repo)
    result = extract_repository(toy_repo)

    edges = _call_edges(graph, into="calc.ts::computeTotal")
    sites = [
        site
        for site in result.sites
        if site.target_symbol_id == "calc.ts::computeTotal"
        and site.kind in (ReferenceKind.CALL, ReferenceKind.IDENTIFIER)
    ]

    assert len(edges) == 6
    assert len(sites) == 7
    # The extra one is the second call on widget.ts line 7, which the parse
    # layer's same-line dedup discarded before the graph ever saw it.
    extra = [site for site in sites if site.occurrence_index > 0]
    assert [(site.file_path, site.start_line) for site in extra] == [("widget.ts", 7)]


def test_store_attributes_a_lambda_call_without_inventing_a_scope(toy_repo):
    """A call inside a lambda keeps the enclosing symbol the parser assigned.

    ``HANDLERS = {"a": lambda v: compute_total(v)}`` makes the graph draw an
    edge whose *caller is a constant*. The store records the same attribution
    rather than silently improving it — the site's value is its position, and
    misreporting the scope would be a new claim we cannot support.
    """
    result = extract_repository(toy_repo)
    site = next(
        site
        for site in result.sites
        if site.file_path == "app.py" and site.name == "compute_total" and site.start_line == 9
    )

    assert site.enclosing_symbol_id == "app.py::HANDLERS"
    assert site.target_symbol_id == "helpers.py::compute_total"
    assert site.range_exact
