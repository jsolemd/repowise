"""``bridge_functions`` — the symbols two files can only reach each other through.

A bridge is not "a symbol with many neighbours". It is a symbol whose
callers and callees live in files that do not otherwise talk: remove it,
or change its contract, and a path that exists only through it is gone.

The disjointness test runs at *file* granularity rather than directory
granularity on purpose. At directory granularity almost every bridge is
masked, because some unrelated symbol in the caller's directory usually
calls something in the callee's directory, and the pair stops looking
disconnected. Files are the level at which the claim stays true.
"""

from __future__ import annotations

from ._definition import PatternDefinition, PatternMatch, PatternResult
from ._graph import SymbolGraph

DEFINITION = PatternDefinition(
    name="bridge_functions",
    question="Which symbols are the only path between two otherwise unconnected files?",
    predicate=(
        "A symbol with at least one caller and at least one callee outside its own file, "
        "where the caller files and callee files are disjoint sets, and at least one "
        "(caller file, callee file) pair has no direct symbol-level call or reference "
        "edge between them. Traffic from that caller file to that callee file must pass "
        "through this symbol."
    ),
    ranking="Number of bridged file pairs, then inbound plus outbound degree, descending.",
    not_this=(
        "Not hub_functions. A hub is wide; a bridge is narrow and load-bearing. A symbol "
        "with two callers and one callee can be a bridge and will never be a hub."
    ),
    defaults={"include_tests": False},
)


def run(
    graph: SymbolGraph,
    *,
    include_tests: bool = False,
) -> PatternResult:
    candidates = graph.candidates(include_tests=include_tests)

    matches: list[PatternMatch] = []
    for sym in candidates:
        caller_files = tuple(f for f in graph.files_of(graph.callers_of(sym.id)) if f != sym.file)
        callee_files = tuple(f for f in graph.files_of(graph.callees_of(sym.id)) if f != sym.file)
        if not caller_files or not callee_files:
            continue
        if set(caller_files) & set(callee_files):
            continue

        bridged = [
            {"from_file": a, "to_file": b}
            for a in caller_files
            for b in callee_files
            if (a, b) not in graph.file_use_pairs
        ]
        if not bridged:
            continue

        degree = graph.in_degree(sym.id) + graph.out_degree(sym.id)
        matches.append(
            PatternMatch(
                key=sym.id,
                evidence={
                    **sym.location(),
                    "bridged_file_pairs": bridged[:10],
                    "bridged_pair_count": len(bridged),
                    "callers": graph.in_degree(sym.id),
                    "callees": graph.out_degree(sym.id),
                },
                rank=(float(len(bridged)), float(degree)),
            )
        )

    matches.sort(key=lambda m: (-m.rank[0], -m.rank[1], m.key))
    return PatternResult(
        definition=DEFINITION,
        params={"include_tests": include_tests},
        matches=tuple(matches),
        scanned_symbols=len(candidates),
    )
