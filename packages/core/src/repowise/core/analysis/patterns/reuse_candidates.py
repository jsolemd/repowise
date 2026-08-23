"""``reuse_candidates`` — the surface that has already proven reusable.

Before writing a helper, the useful question is not "what is popular" but
"what has already been adopted outside the neighbourhood it was written
in". Demand from several directories is the evidence that a symbol
generalises; volume from one directory is not.
"""

from __future__ import annotations

from ._definition import PatternDefinition, PatternMatch, PatternResult
from ._graph import SymbolGraph

DEFINITION = PatternDefinition(
    name="reuse_candidates",
    question="Which existing symbols should I call instead of writing a new helper?",
    predicate=(
        "An exported symbol (public, protected or internal) called or referenced from at "
        "least min_directories distinct directories other than its own. Callers inside "
        "the symbol's own directory are counted and reported but do not count toward the "
        "threshold, because local use is not evidence that a symbol generalises."
    ),
    ranking=(
        "Distinct external caller directories, then total callers, descending — spread "
        "first, volume second."
    ),
    not_this=(
        "Not hub_functions, which ranks raw inbound volume regardless of where it comes "
        "from. A symbol can be both; a symbol with five hundred callers all in one "
        "package is a hub and not a reuse candidate."
    ),
    defaults={"min_directories": 2, "include_tests": False},
)


def run(
    graph: SymbolGraph,
    *,
    min_directories: int = 2,
    include_tests: bool = False,
) -> PatternResult:
    candidates = graph.candidates(include_tests=include_tests)

    matches: list[PatternMatch] = []
    for sym in candidates:
        if not sym.exported:
            continue
        inbound = graph.callers_of(sym.id)
        if not inbound:
            continue
        directories = graph.directories_of(inbound)
        external = tuple(d for d in directories if d != sym.directory)
        if len(external) < min_directories:
            continue
        matches.append(
            PatternMatch(
                key=sym.id,
                evidence={
                    **sym.location(),
                    "signature": sym.signature,
                    "external_directories": list(external),
                    "external_directory_count": len(external),
                    "callers": len(inbound),
                    "local_callers": sum(
                        1 for cid in inbound if graph.symbols[cid].directory == sym.directory
                    ),
                },
                rank=(float(len(external)), float(len(inbound))),
            )
        )

    matches.sort(key=lambda m: (-m.rank[0], -m.rank[1], m.key))
    return PatternResult(
        definition=DEFINITION,
        params={"min_directories": min_directories, "include_tests": include_tests},
        matches=tuple(matches),
        scanned_symbols=len(candidates),
    )
