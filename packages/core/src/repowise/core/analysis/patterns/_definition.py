"""The stated definition that travels with every pattern response.

A named pattern is only useful if the caller knows what it asserted. A
response that says ``hub_functions: [a, b, c]`` and nothing else invites
the reader to supply their own definition of "hub", and the one they
supply will not be the one the query ran. So every pattern carries the
predicate it evaluated, the ranking it applied, the parameter values it
ran with, and — the field that does the most work in practice —
``not_this``, naming the adjacent surface it is most likely to be
confused with.

``params`` is filled in per call, not per pattern: the definition a
response carries must describe *that* run, including any threshold the
caller overrode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PatternDefinition:
    """What one named pattern asserts about the symbols it returns."""

    name: str
    question: str
    predicate: str
    ranking: str
    not_this: str
    #: Parameter name -> default. Overridden values are merged in per call.
    defaults: dict[str, Any]

    def as_dict(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        effective = dict(self.defaults)
        if params:
            effective.update(params)
        return {
            "name": self.name,
            "question": self.question,
            "predicate": self.predicate,
            "ranking": self.ranking,
            "not_this": self.not_this,
            "params": effective,
        }


@dataclass(frozen=True, slots=True)
class PatternMatch:
    """One symbol (or symbol group) that satisfied the predicate."""

    key: str
    evidence: dict[str, Any]
    #: Sort key, descending. Higher is a stronger instance of the pattern.
    rank: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, **self.evidence}


@dataclass(frozen=True, slots=True)
class PatternResult:
    """A pattern run: its definition, its matches, and what it scanned."""

    definition: PatternDefinition
    params: dict[str, Any]
    matches: tuple[PatternMatch, ...]
    scanned_symbols: int
    truncated: bool = False

    def as_dict(self, *, limit: int | None = None) -> dict[str, Any]:
        shown = self.matches if limit is None else self.matches[:limit]
        return {
            "pattern": self.definition.name,
            "definition": self.definition.as_dict(self.params),
            "matches": [m.as_dict() for m in shown],
            "summary": {
                "total_matches": len(self.matches),
                "returned": len(shown),
                "scanned_symbols": self.scanned_symbols,
                "truncated": self.truncated
                or (limit is not None and len(shown) < len(self.matches)),
            },
        }
