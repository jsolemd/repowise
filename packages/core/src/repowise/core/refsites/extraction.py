"""Turn one ``ParsedFile`` into reference sites.

Two tiers, kept apart on purpose.

**Tier A — AST-attested.** One site per fact the parse layer already
established: definitions, call sites, receivers, import statements and their
bindings, type references, heritage. The parse layer gives a line and a name
but never a column, so each fact is *placed* by finding its identifier in the
code regions of that line. Placement claims positions left to right, so N
facts on one line take the first N positions of that name.

**Tier B — textual sweep.** Every remaining code identifier, and every string
literal, whose text is a name the parse layer attested somewhere in the
repository. It runs after Tier A over the positions Tier A did not claim, and
lives in :mod:`repowise.core.refsites.textual` — which is where the three
losses it recovers are documented.
"""

from __future__ import annotations

from dataclasses import dataclass

from repowise.core.ingestion.models import ParsedFile
from repowise.core.refsites.records import ReferenceSiteRecord
from repowise.core.refsites.sweep import REGION_CODE, lex_config_for, scan
from repowise.core.refsites.taxonomy import TIER_AST, ReferenceKind, ReferenceOrigin
from repowise.core.refsites.textual import TextualSweepMixin
from repowise.core.refsites.universe import Universe


@dataclass(slots=True)
class _Placement:
    """Where a site landed, and whether we actually found it in the text."""

    start_line: int
    end_line: int
    start_col: int
    end_col: int
    exact: bool


class _FileExtractor(TextualSweepMixin):
    """Per-file state: identifier positions, what has been claimed, the output.

    Tier A lives here; Tier B is :class:`TextualSweepMixin`, split out only to
    keep both files under the project's line ceiling.
    """

    def __init__(self, parsed: ParsedFile, source: str, universe: Universe) -> None:
        self.parsed = parsed
        self.path = parsed.file_info.path
        self.language = parsed.file_info.language
        self.source = source
        self.universe = universe
        self.sites: list[ReferenceSiteRecord] = []
        # Split once. ``_statement_spans`` runs per import statement and walks
        # every line; re-splitting inside it made import placement quadratic in
        # (imports x lines) on files that have many of both.
        self._lines: list[str] = source.split("\n")
        self._claimed: set[tuple[int, int]] = set()
        self._code: dict[str, list[tuple[int, int]]] = {}
        self._strings: list[tuple[str, int, int, int, int]] = []
        if lex_config_for(self.language) is not None:
            for token in scan(source, self.language):
                if token.kind == "identifier" and token.region == REGION_CODE:
                    self._code.setdefault(token.text, []).append((token.line, token.start_col))
                elif token.kind == "string":
                    self._strings.append(
                        (token.text, token.line, token.end_line, token.start_col, token.end_col)
                    )
            for positions in self._code.values():
                positions.sort()

    # -- placement ---------------------------------------------------------

    def _place(self, name: str, start_line: int, end_line: int | None = None) -> _Placement:
        """Claim the first unclaimed code position of *name* in the line range.

        Falls back to a line-only placement when the name is not literally
        present — an aliased import, a language we cannot lex, or a parse-layer
        line that points at the enclosing construct rather than the token.
        ``exact=False`` is how that fallback stays visible downstream instead
        of masquerading as a column of zero. The fallback collapses to the
        *start* line rather than spanning the searched range: for a definition
        that range is the whole body, and claiming a rename would touch every
        line of a function is a worse answer than claiming one line and
        admitting the column is unknown.
        """
        last = end_line if end_line is not None else start_line
        for line, col in self._code.get(name, ()):
            if start_line <= line <= last and (line, col) not in self._claimed:
                self._claimed.add((line, col))
                return _Placement(line, line, col, col + len(name), True)
        return _Placement(start_line, start_line, 0, 0, False)

    def _emit(
        self,
        name: str,
        kind: ReferenceKind,
        placement: _Placement,
        origin: ReferenceOrigin,
        target: str | None,
        enclosing: str | None,
        tier: str = TIER_AST,
    ) -> None:
        self.sites.append(
            ReferenceSiteRecord(
                file_path=self.path,
                language=self.language,
                name=name,
                kind=kind,
                start_line=placement.start_line,
                end_line=placement.end_line,
                start_col=placement.start_col,
                end_col=placement.end_col,
                range_exact=placement.exact,
                origin=origin,
                tier=tier,
                target_symbol_id=target,
                enclosing_symbol_id=enclosing,
            )
        )

    # -- tier A ------------------------------------------------------------

    def definitions(self) -> None:
        for symbol in self.parsed.symbols:
            self._emit(
                name=symbol.name,
                kind=ReferenceKind.DEFINITION,
                placement=self._place(symbol.name, symbol.start_line, symbol.end_line),
                origin=ReferenceOrigin.DEFINITION,
                target=symbol.id,
                enclosing=symbol.parent_symbol_id,
            )

    def calls(self) -> None:
        for site, kind in (
            *((call, ReferenceKind.CALL) for call in self.parsed.calls),
            *((ref, ReferenceKind.REFERENCE) for ref in self.parsed.references),
        ):
            resolution = self.universe.resolve(
                site.target_name, self.path, enclosing_symbol_id=site.caller_symbol_id
            )
            self._emit(
                name=site.target_name,
                kind=kind,
                placement=self._place(site.target_name, site.line),
                origin=resolution.origin,
                target=resolution.symbol_id,
                enclosing=site.caller_symbol_id,
            )
            if site.receiver_name and site.receiver_name.isidentifier():
                receiver = self.universe.resolve(
                    site.receiver_name, self.path, enclosing_symbol_id=site.caller_symbol_id
                )
                self._emit(
                    name=site.receiver_name,
                    kind=ReferenceKind.RECEIVER,
                    placement=self._place(site.receiver_name, site.line),
                    origin=receiver.origin,
                    target=receiver.symbol_id,
                    enclosing=site.caller_symbol_id,
                )

    def _statement_spans(self, raw: str) -> list[tuple[int, int, int]]:
        """Every ``(start_line, end_line, indent)`` at which *raw* is a statement.

        Returning *every* occurrence is the point: the parse layer dedups
        imports by raw text, so two byte-identical import statements in one
        file collapse to a single ``Import``. Matching is anchored to the first
        non-space character of a line, so a mention inside a longer expression
        cannot masquerade as a statement.
        """
        if not raw:
            return []
        height = raw.count("\n")
        head = raw.split("\n", 1)[0]

        anchored: list[tuple[int, int, int]] = []
        loose: list[tuple[int, int, int]] = []
        offset = 0
        for index, line_text in enumerate(self._lines, start=1):
            indent = len(line_text) - len(line_text.lstrip())
            if self.source.startswith(raw, offset + indent):
                anchored.append((index, index + height, indent))
            else:
                column = line_text.find(head)
                if column >= 0 and self.source.startswith(raw, offset + column):
                    loose.append((index, index + height, column))
            offset += len(line_text) + 1

        # ``Import.raw_statement`` is the captured node's text, which for a
        # destructuring ``require`` is the declarator rather than the whole
        # statement — so it never starts its line. The unanchored matches are
        # the fallback for exactly that, and are used only when no statement
        # anchored anywhere in the file, so a real statement always wins.
        return anchored or loose

    def imports(self) -> None:
        """Emit one site per import statement, plus one per name it binds.

        An ``IMPORT`` site's ``name`` is the *module path*, not a symbol name —
        a different namespace sharing a column. The store's by-name lookups
        exclude the kind for that reason; a module rename is the query that
        wants them.
        """
        for statement in self.parsed.imports:
            spans = self._statement_spans(statement.raw_statement)
            if not spans:
                # The raw text is not literally in the source — a projected SFC
                # region, or a statement the query normalised. Record it on
                # line 1 with ``exact=False`` rather than dropping it.
                self._emit(
                    name=statement.module_path,
                    kind=ReferenceKind.IMPORT,
                    placement=_Placement(1, 1, 0, 0, False),
                    origin=ReferenceOrigin.IMPORT_BINDING,
                    target=None,
                    enclosing=None,
                )
                self._import_bindings(statement, 1, len(self._lines))
                continue
            for start_line, end_line, indent in spans:
                self._emit(
                    name=statement.module_path,
                    kind=ReferenceKind.IMPORT,
                    placement=_Placement(
                        start_line,
                        end_line,
                        indent,
                        indent + len(statement.raw_statement.split("\n", 1)[0]),
                        True,
                    ),
                    origin=ReferenceOrigin.IMPORT_BINDING,
                    target=None,
                    enclosing=None,
                )
                self._import_bindings(statement, start_line, end_line)

    def _import_bindings(self, statement, start_line: int, end_line: int) -> None:
        kind = ReferenceKind.REEXPORT if statement.is_reexport else ReferenceKind.IMPORT_BINDING
        bound = self.universe.file_bindings.get(self.path, {})
        names = (
            [binding.local_name for binding in statement.bindings if not binding.is_module_alias]
            if statement.bindings
            else [name for name in statement.imported_names if name != "*"]
        )
        for name in names:
            resolution = bound.get(name)
            self._emit(
                name=name,
                kind=kind,
                placement=self._place(name, start_line, end_line),
                origin=resolution.origin if resolution else ReferenceOrigin.UNRESOLVED,
                target=resolution.symbol_id if resolution else None,
                enclosing=None,
            )

    def types_and_heritage(self) -> None:
        for type_ref in self.parsed.type_refs:
            resolution = self.universe.resolve(type_ref.type_name, self.path)
            self._emit(
                name=type_ref.type_name,
                kind=ReferenceKind.TYPE_REF,
                placement=self._place(type_ref.type_name, type_ref.line),
                origin=resolution.origin,
                target=resolution.symbol_id,
                enclosing=None,
            )
        for relation in self.parsed.heritage:
            resolution = self.universe.resolve(relation.parent_name, self.path)
            self._emit(
                name=relation.parent_name,
                kind=ReferenceKind.HERITAGE,
                placement=self._place(relation.parent_name, relation.line),
                origin=resolution.origin,
                target=resolution.symbol_id,
                enclosing=None,
            )


def _finalise(sites: list[ReferenceSiteRecord]) -> tuple[ReferenceSiteRecord, ...]:
    """Deduplicate by position and number same-line siblings.

    ``occurrence_index`` is assigned here rather than at emit time so it means
    the same thing for both tiers: the nth occurrence of this name on this
    line, left to right. A non-zero index on a call site is the visible
    fingerprint of a position the parse layer's same-line dedup discarded.
    """
    unique: dict[tuple[str, int, int, str, str], ReferenceSiteRecord] = {}
    for site in sites:
        unique.setdefault(site.position_key, site)

    ordered = sorted(
        unique.values(),
        key=lambda site: (site.file_path, site.start_line, site.start_col, site.name, site.kind),
    )
    counters: dict[tuple[str, int, str], int] = {}
    numbered: list[ReferenceSiteRecord] = []
    for site in ordered:
        key = (site.file_path, site.start_line, site.name)
        index = counters.get(key, 0)
        counters[key] = index + 1
        numbered.append(site if index == 0 else _with_index(site, index))
    return tuple(numbered)


def _with_index(site: ReferenceSiteRecord, index: int) -> ReferenceSiteRecord:
    return ReferenceSiteRecord(
        file_path=site.file_path,
        language=site.language,
        name=site.name,
        kind=site.kind,
        start_line=site.start_line,
        end_line=site.end_line,
        start_col=site.start_col,
        end_col=site.end_col,
        range_exact=site.range_exact,
        origin=site.origin,
        tier=site.tier,
        target_symbol_id=site.target_symbol_id,
        enclosing_symbol_id=site.enclosing_symbol_id,
        occurrence_index=index,
        extractor_version=site.extractor_version,
    )


def extract_file(
    parsed: ParsedFile,
    source: str,
    universe: Universe,
    *,
    sweep_textual: bool = True,
) -> tuple[ReferenceSiteRecord, ...]:
    """Extract every reference site in one parsed file.

    ``sweep_textual=False`` keeps Tier A only. On a 3,697-file repository the
    textual tier is 37% of all rows, and a deployment that only wants sites it
    can act on mechanically has no use for them. It costs the recoveries that
    tier exists for — the same-line dedup pair, the declaration deleted inside
    an anonymous callable, the name spelled in a string — so a rename preview
    built on a Tier-A-only store must be read as less complete, not cleaner.
    """
    extractor = _FileExtractor(parsed, source, universe)
    extractor.definitions()
    extractor.calls()
    extractor.imports()
    extractor.types_and_heritage()
    if sweep_textual:
        extractor.sweep()
    return _finalise(extractor.sites)


__all__ = ["extract_file"]
