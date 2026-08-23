"""The name universe a reference site is resolved against.

Resolution here is deliberately a *narrower* problem than the call resolver's.
The call resolver has to decide which edge to draw and must not draw a wrong
one, so it declines rather than guesses. A reference site has somewhere to put
a guess — the confidence rubric — so it can record a weak binding instead of
dropping the occurrence. The tiers below are ordered by what could defeat them:

``same_file``
    Nothing in another file can shadow a file-local definition.
``import_binding``
    The file's own import brought this name in, and the module it names
    resolved to a repository file that defines it.
``import_scoped``
    The file imports the name, but the module is external or unresolvable.
    Exactly one repository symbol answers to it, so that is the binding.
``unique_global``
    Nothing brought the name into scope explicitly, but only one symbol in the
    repository has it. True until the repository grows a second one.
``ambiguous_global`` / ``unresolved``
    No single answer. The site is still recorded; ``target_symbol_id`` stays
    NULL and the origin says which of the two failures happened.

The universe is also what bounds the textual sweep: :attr:`Universe.names`
is every name the parse layer attested somewhere, and the sweep may look for
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from repowise.core.ingestion.graph._stem import build_stem_map
from repowise.core.ingestion.models import ParsedFile, Symbol
from repowise.core.ingestion.resolvers import ResolverContext, resolve_import
from repowise.core.refsites.taxonomy import ReferenceOrigin


@dataclass(frozen=True, slots=True)
class Resolution:
    """Which symbol a name denotes from a given file, and how sure we are."""

    symbol_id: str | None
    origin: ReferenceOrigin

    @property
    def bound(self) -> bool:
        return self.symbol_id is not None


@dataclass
class Universe:
    """Every name the parse layer attested, indexed for resolution."""

    #: symbol id -> Symbol.
    symbols: dict[str, Symbol] = field(default_factory=dict)
    #: bare name -> symbol ids that carry it, repository-wide.
    names: dict[str, list[str]] = field(default_factory=dict)
    #: file path -> {bare name -> best symbol id} for symbols defined in that
    #: file. "Best" is the outermost one, which is what a name at file scope
    #: reaches; :attr:`file_symbol_candidates` keeps the rest.
    file_symbols: dict[str, dict[str, str]] = field(default_factory=dict)
    #: file path -> {bare name -> every symbol id in that file with that name}.
    file_symbol_candidates: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    #: file path -> {local name -> Resolution} contributed by that file's imports.
    file_bindings: dict[str, dict[str, Resolution]] = field(default_factory=dict)
    #: file path -> language, so a site can record the language without the FileInfo.
    file_language: dict[str, str] = field(default_factory=dict)
    #: file path -> every name the parse layer attested *in that file*: symbols
    #: it defines, names its imports bind, and the targets and receivers of the
    #: calls and references it makes. This, not the repository-wide name set,
    #: is what bounds the textual sweep — see :meth:`attests`.
    file_attested: dict[str, set[str]] = field(default_factory=dict)

    def _prefer(self, candidates: list[str], enclosing_symbol_id: str | None) -> str:
        """Pick the candidate the reference's own scope actually reaches.

        One file can define ``render`` twice — as a method of a class and as a
        module-level function — and which one a bare ``render()`` means depends
        entirely on where it is written. Ignoring that and taking whichever was
        parsed first is not a small error: ``same_file`` is the second-highest
        tier in the rubric, so a wrong pick here is a 0.95-confidence false
        positive, the exact thing a rename must never be handed.
        """
        if len(candidates) == 1:
            return candidates[0]

        enclosing = self.symbols.get(enclosing_symbol_id or "")
        if enclosing is not None and enclosing.parent_symbol_id:
            # Inside a method: a sibling under the same owner wins.
            sibling = next(
                (
                    candidate
                    for candidate in candidates
                    if self.symbols[candidate].parent_symbol_id == enclosing.parent_symbol_id
                ),
                None,
            )
            if sibling is not None:
                return sibling

        # Otherwise only an outermost definition is in scope for a bare name.
        top_level = next(
            (
                candidate
                for candidate in candidates
                if self.symbols[candidate].parent_symbol_id is None
            ),
            None,
        )
        return top_level if top_level is not None else candidates[0]

    def resolve(
        self, name: str, file_path: str, *, enclosing_symbol_id: str | None = None
    ) -> Resolution:
        """Resolve *name* as seen from *file_path*, inside *enclosing_symbol_id*."""
        candidates = self.file_symbol_candidates.get(file_path, {}).get(name)
        if candidates:
            return Resolution(
                self._prefer(candidates, enclosing_symbol_id), ReferenceOrigin.SAME_FILE
            )

        bound = self.file_bindings.get(file_path, {}).get(name)
        if bound is not None and bound.bound:
            return bound

        global_candidates = self.names.get(name)
        if not global_candidates:
            return Resolution(None, ReferenceOrigin.UNRESOLVED)
        if len(global_candidates) == 1:
            return Resolution(global_candidates[0], ReferenceOrigin.UNIQUE_GLOBAL)
        return Resolution(None, ReferenceOrigin.AMBIGUOUS_GLOBAL)

    def knows(self, name: str) -> bool:
        """True when the parse layer attested *name* somewhere in the repository."""
        return name in self.names

    def attests(self, file_path: str, name: str) -> bool:
        """True when the parse layer attested *name* **inside** *file_path*.

        This is the bound on the textual sweep, and it is per file on purpose.
        Bounding by the repository-wide name set sounds equivalent and is not:
        measured on a 3,697-file repository it produced 976k sites, 73% of them
        textual, because common identifiers like ``path``, ``name`` and
        ``value`` are symbol names *somewhere*, so nearly every token in the
        tree matched something. Requiring the parser to have seen the name in
        this file keeps exactly the recoveries the sweep exists for — a second
        call the same-line dedup dropped, a declaration deleted inside an
        anonymous callable, a name spelled in a string — and discards the
        coincidences.
        """
        return name in self.file_attested.get(file_path, ())


def _resolver_context(repo_path: Path, parsed: list[ParsedFile]) -> ResolverContext:
    """Build the context the shared import resolvers expect.

    Deliberately partial. ``tsconfig_resolver``, ``go_modules`` and the
    workspace indexes that ``GraphBuilder`` populates are left unset, so a
    path-alias import simply fails to resolve and its binding falls to the
    ``import_scoped`` or ``unique_global`` tier. That is a confidence
    downgrade, never a wrong answer, and it keeps this module off
    ``GraphBuilder``'s private build sequence.
    """
    path_set = {parsed_file.file_info.path for parsed_file in parsed}
    return ResolverContext(
        path_set=path_set,
        stem_map=build_stem_map(path_set),
        graph=nx.DiGraph(),
        repo_path=repo_path,
        has_sfc_files=any(p.endswith((".vue", ".svelte", ".astro")) for p in path_set),
        parsed_files={parsed_file.file_info.path: parsed_file for parsed_file in parsed},
    )


def _binding_targets(imported: ParsedFile) -> list[tuple[str, str, str]]:
    """Yield ``(local_name, exported_name, module_path)`` for one file's imports.

    ``Import.bindings`` carries aliasing when the grammar supplied it;
    ``imported_names`` is the fallback for the languages whose queries only
    capture bare names. A wildcard import binds nothing nameable and is skipped.
    """
    out: list[tuple[str, str, str]] = []
    for statement in imported.imports:
        if statement.bindings:
            for binding in statement.bindings:
                if binding.is_module_alias:
                    continue
                exported = binding.exported_name or binding.local_name
                out.append((binding.local_name, exported, statement.module_path))
            continue
        for name in statement.imported_names:
            if name == "*":
                continue
            out.append((name, name, statement.module_path))
    return out


def build_universe(repo_path: Path, parsed: list[ParsedFile]) -> Universe:
    """Index *parsed* into a resolvable name universe."""
    universe = Universe()

    for parsed_file in parsed:
        path = parsed_file.file_info.path
        universe.file_language[path] = parsed_file.file_info.language
        per_file = universe.file_symbol_candidates.setdefault(path, {})
        attested = universe.file_attested.setdefault(path, set())
        for symbol in parsed_file.symbols:
            universe.symbols[symbol.id] = symbol
            universe.names.setdefault(symbol.name, []).append(symbol.id)
            per_file.setdefault(symbol.name, []).append(symbol.id)
            attested.add(symbol.name)

    # Every name the parser attested as *referenced* also belongs in the
    # universe even when nothing defines it — that is what lets an unresolved
    # call keep its row instead of being filtered out for having no target.
    for parsed_file in parsed:
        attested = universe.file_attested.setdefault(parsed_file.file_info.path, set())
        for call in (*parsed_file.calls, *parsed_file.references):
            universe.names.setdefault(call.target_name, [])
            attested.add(call.target_name)
            if call.receiver_name:
                universe.names.setdefault(call.receiver_name, [])
                attested.add(call.receiver_name)

    for symbol_ids in universe.names.values():
        symbol_ids.sort()

    # Collapse the per-file candidates to the one an *outside* reference would
    # reach. An import binds a module-level name, never a method, so the
    # outermost definition is the right answer for every cross-file lookup and
    # for the in-scope check the textual sweep makes.
    for path, per_name in universe.file_symbol_candidates.items():
        collapsed = universe.file_symbols.setdefault(path, {})
        for name, symbol_ids in per_name.items():
            collapsed[name] = universe._prefer(symbol_ids, None)

    context = _resolver_context(repo_path, parsed)
    for parsed_file in parsed:
        path = parsed_file.file_info.path
        language = parsed_file.file_info.language
        bindings = universe.file_bindings.setdefault(path, {})
        attested = universe.file_attested.setdefault(path, set())
        for local_name, exported_name, module_path in _binding_targets(parsed_file):
            universe.names.setdefault(local_name, [])
            attested.add(local_name)
            bindings.setdefault(
                local_name,
                _resolve_binding(universe, context, path, language, exported_name, module_path),
            )

    return universe


def _resolve_binding(
    universe: Universe,
    context: ResolverContext,
    importer_path: str,
    language: str,
    exported_name: str,
    module_path: str,
) -> Resolution:
    """Bind one imported name to a symbol, best tier first."""
    target_file = resolve_import(module_path, importer_path, language, context)
    if target_file is not None:
        in_target = universe.file_symbols.get(target_file, {}).get(exported_name)
        if in_target is not None:
            return Resolution(in_target, ReferenceOrigin.IMPORT_BINDING)

    candidates = universe.names.get(exported_name) or []
    if len(candidates) == 1:
        return Resolution(candidates[0], ReferenceOrigin.IMPORT_SCOPED)
    if len(candidates) > 1:
        return Resolution(None, ReferenceOrigin.AMBIGUOUS_GLOBAL)
    return Resolution(None, ReferenceOrigin.UNRESOLVED)


__all__ = ["Resolution", "Universe", "build_universe"]
