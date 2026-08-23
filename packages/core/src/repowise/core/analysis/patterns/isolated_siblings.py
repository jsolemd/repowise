"""``isolated_siblings`` — symbols that share a scope but not a purpose.

A file (or a class) is a claim that the things inside it belong together.
When a symbol has several siblings and touches none of them — and the
file's own top-level code does not touch it either — the claim is false
for that symbol, and the file is quietly two files.

Top-level use counts as a tie. Ingestion parks a file's top-level calls
on a synthetic ``__module__`` scope; a helper invoked from module scope is
used by its file even though no sibling names it, and reporting it as
isolated would be wrong.
"""

from __future__ import annotations

from ._definition import PatternDefinition, PatternMatch, PatternResult
from ._graph import SymbolGraph

DEFINITION = PatternDefinition(
    name="isolated_siblings",
    question="Which symbols sit in a scope whose other members they never touch?",
    predicate=(
        "A symbol sharing its parent scope (its enclosing class, or its file when it has "
        "no enclosing class) with at least min_siblings other reportable symbols, having "
        "no call or reference edge in either direction to any of them, and not used from "
        "the file's top-level scope either — in a scope that does cohere without it, "
        "meaning at least one of those siblings uses another. That last clause is what "
        "separates a misfiled symbol from a file the resolver simply produced no edges "
        "for, such as a constants or type-declaration module. Only kinds that can call "
        "their neighbours are considered: a constant or a type declaration sitting in a "
        "cohesive module is data, not a misfiling, and including those kinds made the "
        "predicate match a third of every symbol in a real repository."
    ),
    ranking="Sibling count, descending — the more it fails to touch, the stronger the signal.",
    not_this=(
        "Not orphan_exports. An isolated sibling can be heavily used from outside the "
        "file; what it lacks is any relationship to the code it was filed with. The two "
        "patterns overlap only when a symbol is both unused and misfiled."
    ),
    defaults={
        "min_siblings": 2,
        "include_tests": False,
        "kinds": ["function", "method", "class"],
    },
)


def _scope_coheres(graph: SymbolGraph, sibling_ids: set[str], scope_ids: set[str]) -> bool:
    """Does the rest of the scope hold together without the candidate?

    Without this the pattern measures edge *sparsity* rather than
    misfiling. Measured on a 40k-symbol index: dropping this clause makes
    the predicate match 54% of every symbol in the repository, because a
    file of constants or type declarations has no intra-file edges at all
    and every one of its members then reads as isolated. Requiring the
    scope to demonstrate cohesion somewhere else is what makes one
    member's lack of it mean something.
    """
    return any(scope_ids.intersection(graph.callees_of(other)) - {other} for other in sibling_ids)


def run(
    graph: SymbolGraph,
    *,
    min_siblings: int = 2,
    include_tests: bool = False,
    kinds: tuple[str, ...] | list[str] | None = None,
) -> PatternResult:
    allowed = {k.lower() for k in (kinds or DEFINITION.defaults["kinds"])}
    candidates = [
        s for s in graph.candidates(include_tests=include_tests) if s.kind.lower() in allowed
    ]

    matches: list[PatternMatch] = []
    for sym in candidates:
        siblings = graph.siblings_of(sym)
        if len(siblings) < min_siblings:
            continue

        # Top-level use of this symbol is a tie to its file, so the file's
        # synthetic module scope joins the sibling set for edge purposes
        # only — it is never itself reported and never counted above.
        sibling_ids = {s.id for s in siblings}
        scope_ids = set(sibling_ids)
        module_scope = graph.module_scope_of(sym.file)
        if module_scope is not None:
            scope_ids.add(module_scope)

        touched = scope_ids.intersection(graph.callees_of(sym.id)) | scope_ids.intersection(
            graph.callers_of(sym.id)
        )
        if touched:
            continue
        if not _scope_coheres(
            graph, sibling_ids | ({module_scope} if module_scope else set()), scope_ids
        ):
            continue

        matches.append(
            PatternMatch(
                key=sym.id,
                evidence={
                    **sym.location(),
                    "scope": sym.parent_id,
                    "siblings": len(siblings),
                    "sibling_names": [s.name for s in siblings][:10],
                    "external_callers": graph.in_degree(sym.id),
                },
                rank=(float(len(siblings)),),
            )
        )

    matches.sort(key=lambda m: (-m.rank[0], m.key))
    return PatternResult(
        definition=DEFINITION,
        params={
            "min_siblings": min_siblings,
            "include_tests": include_tests,
            "kinds": sorted(allowed),
        },
        matches=tuple(matches),
        scanned_symbols=len(candidates),
    )
