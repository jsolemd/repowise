"""``orphan_exports`` — public symbols nothing in the index uses.

The raw graph predicate, stated plainly: declared for outside use, and no
resolved caller or referencer anywhere. That is a weaker claim than "safe
to delete" and the response says so, because the gap between the two is
where every false positive lives — dynamic dispatch, reflective loading,
a public API consumed by a repository this index does not contain.
"""

from __future__ import annotations

from ._definition import PatternDefinition, PatternMatch, PatternResult
from ._graph import SymbolGraph

DEFINITION = PatternDefinition(
    name="orphan_exports",
    question="Which symbols are declared for outside use but never used?",
    predicate=(
        "A symbol whose visibility is public, protected or internal, defined in a file "
        "that is not an entry point, with zero inbound calls/references edges in the "
        "indexed graph. Test files are excluded unless include_tests is set, and so are "
        "function-local definitions and the synthetic module-level scopes."
    ),
    ranking="Symbol size in lines, descending — the largest unused surface first.",
    not_this=(
        "Not get_dead_code's unused_export, which discounts this same predicate by "
        "runtime-load risk, entry-point proximity and confidence tiering before it "
        "reports. This is the undiscounted graph fact: it will list symbols that "
        "get_dead_code deliberately withholds. Treat it as a starting set to verify, "
        "not a deletion list."
    ),
    defaults={
        "include_tests": False,
        "min_lines": 1,
        "visibility": ["public", "protected", "internal"],
    },
)


def run(
    graph: SymbolGraph,
    *,
    include_tests: bool = False,
    min_lines: int = 1,
    visibility: tuple[str, ...] | list[str] | None = None,
) -> PatternResult:
    allowed = {v.lower() for v in (visibility or DEFINITION.defaults["visibility"])}
    candidates = graph.candidates(include_tests=include_tests)

    matches: list[PatternMatch] = []
    for sym in candidates:
        if sym.visibility.lower() not in allowed:
            continue
        if sym.lines < min_lines:
            continue
        file_node = graph.files.get(sym.file)
        if file_node is not None and file_node.is_entry_point:
            continue
        if graph.in_degree(sym.id):
            continue
        matches.append(
            PatternMatch(
                key=sym.id,
                evidence={
                    **sym.location(),
                    "lines": sym.lines,
                    "signature": sym.signature,
                    "callers": 0,
                },
                rank=(float(sym.lines),),
            )
        )

    matches.sort(key=lambda m: (-m.rank[0], m.key))
    return PatternResult(
        definition=DEFINITION,
        params={
            "include_tests": include_tests,
            "min_lines": min_lines,
            "visibility": sorted(allowed),
        },
        matches=tuple(matches),
        scanned_symbols=len(candidates),
    )
