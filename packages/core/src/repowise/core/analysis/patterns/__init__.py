"""Six named, graph-backed pattern queries over the symbol graph.

Each pattern is a filter with a stated predicate, not a score. The point
of naming them is that a caller can ask for one by name and get back both
the matches *and* the rule that produced them, so the answer cannot be
read against a definition the query did not use.

Usage::

    from repowise.core.analysis.patterns import SymbolGraph, run_pattern

    graph = SymbolGraph.from_rows(nodes, edges)
    result = run_pattern("hub_functions", graph, min_callers=3)
    payload = result.as_dict(limit=20)   # carries definition + matches

An unknown name raises :class:`UnknownPatternError`, which carries the
catalogue: a caller that guessed wrong gets the six real names back
rather than an empty match list that reads as "nothing found".
"""

from __future__ import annotations

from typing import Any

from . import (
    bridge_functions,
    duplicate_signatures,
    hub_functions,
    isolated_siblings,
    orphan_exports,
    reuse_candidates,
)
from ._definition import PatternDefinition, PatternMatch, PatternResult
from ._graph import EXPORTED_VISIBILITY, USE_EDGE_TYPES, FileNode, Symbol, SymbolGraph

_MODULES = {
    "duplicate_signatures": duplicate_signatures,
    "orphan_exports": orphan_exports,
    "hub_functions": hub_functions,
    "isolated_siblings": isolated_siblings,
    "reuse_candidates": reuse_candidates,
    "bridge_functions": bridge_functions,
}

#: Declaration order, which is also the order the catalogue is served in.
PATTERN_NAMES: tuple[str, ...] = tuple(_MODULES)

__all__ = [
    "EXPORTED_VISIBILITY",
    "PATTERN_NAMES",
    "USE_EDGE_TYPES",
    "FileNode",
    "PatternDefinition",
    "PatternMatch",
    "PatternResult",
    "Symbol",
    "SymbolGraph",
    "UnknownPatternError",
    "catalogue",
    "definition_of",
    "run_pattern",
]


class UnknownPatternError(LookupError):
    """Raised for a pattern name outside the catalogue."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Unknown pattern {name!r}. Available: {', '.join(PATTERN_NAMES)}.")
        self.name = name
        self.available = PATTERN_NAMES


def definition_of(name: str) -> PatternDefinition:
    module = _MODULES.get(name)
    if module is None:
        raise UnknownPatternError(name)
    return module.DEFINITION


def catalogue() -> list[dict[str, Any]]:
    """Every pattern's definition, at its declared defaults."""
    return [definition_of(name).as_dict() for name in PATTERN_NAMES]


def run_pattern(name: str, graph: SymbolGraph, **params: Any) -> PatternResult:
    """Run one named pattern over *graph*.

    Unknown keyword parameters are rejected by the pattern's own signature
    rather than silently ignored, so a misspelled threshold fails loudly
    instead of returning the default-threshold answer under the caller's
    intended one.
    """
    module = _MODULES.get(name)
    if module is None:
        raise UnknownPatternError(name)
    return module.run(graph, **params)
