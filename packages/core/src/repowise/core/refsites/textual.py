"""Tier B — the textual sweep, split out to keep ``extraction`` under the cap.

Everything here operates on positions the AST pass did *not* claim, and is
bounded by :meth:`Universe.attests`: the sweep may only look for names the
parse layer already saw **in the same file**, so it cannot invent a reference,
only relocate one it could not place structurally.

The per-file bound is load-bearing, not a detail. Bounding by the
repository-wide name set instead — which sounds equivalent — was measured on a
3,697-file repository at 976k sites, 73% of them textual, because identifiers
like ``path``, ``name`` and ``value`` are somebody's symbol name somewhere and
so nearly every token in the tree matched. Requiring the name to be attested
in this file keeps every recovery below and drops the coincidences.

It exists for three losses Tier A cannot reach:

* the parse layer dedups call sites on ``(line, target, receiver)``, so a pair
  of calls to the same function on one line arrives as one;
* a symbol declared inside an anonymous callable is deleted before it reaches
  ``ParsedFile.symbols``, and its declaration site goes with it;
* a name reached only through a string literal has no AST representation at all.

Everything it produces sits on the rubric's textual floors. That is not
pessimism, it is the honest reading: a lexer knows where a name is spelled and
cannot know whether it means the symbol you asked about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from repowise.core.refsites.coverage import is_sweepable
from repowise.core.refsites.taxonomy import TIER_TEXTUAL, ReferenceKind, ReferenceOrigin

if TYPE_CHECKING:  # pragma: no cover - typing only
    from repowise.core.refsites.extraction import _Placement
    from repowise.core.refsites.universe import Universe


class _Host(Protocol):
    """What ``sweep`` needs from the object it is mixed into.

    Spelled out rather than left implicit because a mixin's ``self`` is
    otherwise untyped: this is the contract ``_FileExtractor`` has to keep, and
    the reason the split between the two files is safe. The last two entries
    are the mixin's own methods — ``sweep`` calls them through ``self``, so
    they belong to the same contract.
    """

    path: str
    language: str
    universe: Universe
    _claimed: set[tuple[int, int]]
    _code: dict[str, list[tuple[int, int]]]
    _strings: list[tuple[str, int, int, int, int]]

    def _emit(
        self,
        name: str,
        kind: ReferenceKind,
        placement: _Placement,
        origin: ReferenceOrigin,
        target: str | None,
        enclosing: str | None,
        tier: str = ...,
    ) -> None: ...

    def _textual_origin(self, name: str) -> ReferenceOrigin: ...

    def _string_refs(self) -> None: ...


class TextualSweepMixin:
    """Recovers positions the AST pass could not attest."""

    def _textual_origin(self: _Host, name: str) -> ReferenceOrigin:
        """In-scope names outrank strangers that merely share a spelling.

        A file that defines or imports the name is far more likely to mean
        *that* binding than a same-named symbol from elsewhere in the tree, so
        the two get separate floors rather than one averaged guess.
        """
        in_scope = name in self.universe.file_symbols.get(
            self.path, {}
        ) or name in self.universe.file_bindings.get(self.path, {})
        return (
            ReferenceOrigin.TEXTUAL_IDENTIFIER_IN_SCOPE
            if in_scope
            else ReferenceOrigin.TEXTUAL_IDENTIFIER
        )

    def sweep(self: _Host) -> None:
        """Emit a site for every unclaimed occurrence of a known name."""
        from repowise.core.refsites.extraction import _Placement

        if not is_sweepable(self.language):
            return
        for name, positions in self._code.items():
            if not self.universe.attests(self.path, name):
                continue
            for line, col in positions:
                if (line, col) in self._claimed:
                    continue
                self._claimed.add((line, col))
                resolution = self.universe.resolve(name, self.path)
                self._emit(
                    name=name,
                    kind=ReferenceKind.IDENTIFIER,
                    placement=_Placement(line, line, col, col + len(name), True),
                    origin=self._textual_origin(name),
                    target=resolution.symbol_id,
                    enclosing=None,
                    tier=TIER_TEXTUAL,
                )
        self._string_refs()

    def _string_refs(self: _Host) -> None:
        """Emit a site for a string literal that spells a known name.

        Matches the whole literal, or its last dotted segment so that
        ``"app.handlers.compute_total"`` still points at ``compute_total``.
        A literal spanning lines keeps its line range but reports no columns.
        """
        from repowise.core.refsites.extraction import _Placement

        for text, line, end_line, start_col, _end_col in self._strings:
            body = text.strip()
            if self.universe.attests(self.path, body):
                name, offset = body, text.find(body)
            elif "." in body and self.universe.attests(self.path, body.rsplit(".", 1)[-1]):
                name = body.rsplit(".", 1)[-1]
                offset = text.rfind(name)
            else:
                continue
            single_line = line == end_line
            resolution = self.universe.resolve(name, self.path)
            self._emit(
                name=name,
                kind=ReferenceKind.STRING_REF,
                placement=_Placement(
                    line,
                    end_line,
                    start_col + offset if single_line else 0,
                    start_col + offset + len(name) if single_line else 0,
                    single_line,
                ),
                origin=ReferenceOrigin.TEXTUAL_STRING,
                target=resolution.symbol_id,
                enclosing=None,
                tier=TIER_TEXTUAL,
            )


__all__ = ["TextualSweepMixin"]
