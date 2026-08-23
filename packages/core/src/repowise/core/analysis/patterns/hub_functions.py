"""``hub_functions`` — the symbols with the widest inbound blast radius.

Two thresholds, both stated, because either alone misreads a repository.
A bare percentile makes every small repo produce "hubs" that have three
callers; a bare absolute floor makes a large repo produce hundreds. A
symbol qualifies only when it clears both.
"""

from __future__ import annotations

from ._definition import PatternDefinition, PatternMatch, PatternResult
from ._graph import SymbolGraph

DEFINITION = PatternDefinition(
    name="hub_functions",
    question="Which symbols would a change here reach the most callers of?",
    predicate=(
        "A symbol whose inbound call/reference count is at least min_callers AND at or "
        "above the percentile-th percentile of inbound counts across all symbols that "
        "have at least one caller. Both thresholds must hold: the percentile keeps the "
        "answer relative to this repository, the floor keeps a three-caller symbol in a "
        "small repository from being called a hub."
    ),
    ranking="Inbound count, then the number of distinct caller files, descending.",
    not_this=(
        "Not reuse_candidates. This ranks raw inbound volume — the blast radius of a "
        "change. A symbol called two hundred times from inside its own directory is a "
        "hub and is not a reuse candidate."
    ),
    defaults={"min_callers": 5, "percentile": 95.0, "include_tests": False},
)


def _percentile_floor(values: list[int], percentile: float) -> int:
    """Smallest inbound count that sits at or above *percentile* of *values*.

    Nearest-rank, so the result is always an observed value and the answer
    does not depend on an interpolation convention.
    """
    if not values:
        return 0
    ordered = sorted(values)
    pct = min(max(percentile, 0.0), 100.0)
    rank = max(1, -(-int(len(ordered) * pct) // 100))  # ceil(n * p / 100)
    return ordered[min(rank, len(ordered)) - 1]


def run(
    graph: SymbolGraph,
    *,
    min_callers: int = 5,
    percentile: float = 95.0,
    include_tests: bool = False,
) -> PatternResult:
    candidates = graph.candidates(include_tests=include_tests)
    degrees = [graph.in_degree(s.id) for s in candidates]
    floor = _percentile_floor([d for d in degrees if d > 0], percentile)
    threshold = max(min_callers, floor)

    matches: list[PatternMatch] = []
    for sym in candidates:
        inbound = graph.callers_of(sym.id)
        if len(inbound) < threshold:
            continue
        caller_files = graph.files_of(inbound)
        matches.append(
            PatternMatch(
                key=sym.id,
                evidence={
                    **sym.location(),
                    "callers": len(inbound),
                    "caller_files": len(caller_files),
                    "caller_directories": len(graph.directories_of(inbound)),
                },
                rank=(float(len(inbound)), float(len(caller_files))),
            )
        )

    matches.sort(key=lambda m: (-m.rank[0], -m.rank[1], m.key))
    return PatternResult(
        definition=DEFINITION,
        params={
            "min_callers": min_callers,
            "percentile": percentile,
            "include_tests": include_tests,
            "effective_caller_threshold": threshold,
        },
        matches=tuple(matches),
        scanned_symbols=len(candidates),
    )
