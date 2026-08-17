"""Resolve a symbol target string to WikiSymbol rows.

Both ``get_symbol`` and ``get_context`` accept symbol targets, and an agent
that reads an id out of one response naturally pastes it into the other. That
only works if the two agree on what an id *is*, so the parsing, the separator
normalisation and the lookup ladder live here once rather than once per tool.

Three target forms are accepted, in the order an agent tends to have them:

  1. ``"{file_path}::{Name}"`` — the canonical id another response handed back.
  2. ``"Class.method"`` / ``"module.Class.method"`` — a qualified name read out
     of source or a stack trace, with no path attached.
  3. ``"reconcile_symbols_for_files"`` — a bare name, which is what an agent
     actually has after reading a call site.

Forms 2 and 3 are the ones the agent has *before* it has talked to the index,
so refusing them costs a round trip (and an agent that gets "not found" for a
name it can see in the source stops trusting the tool). They resolve through
the same ladder, and a query matching several symbols returns every match:
the caller decides, and nothing is silently picked for it.

Separator normalisation is the load-bearing part. Languages disagree about
what goes between the segments of a qualified name. Python and TypeScript
write ``Class.method``, C++ and Rust write ``Class::method``, and some tools
emit ``Class/method``. The index stores exactly one of those forms, so a
lookup that matches the caller's string verbatim resolves or misses depending
on which convention the caller happened to use. Every separator form of a
name is therefore tried, and only the name is rewritten: file paths are
matched as given, since ``.`` and ``/`` are meaningful inside them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import or_, select

from repowise.core.persistence.models import WikiSymbol
from repowise.core.persistence.sql import LIKE_ESCAPE, escape_like

# Separators used between name segments AFTER the file path.
NAME_SEPARATORS = (".", "::", "/")

#: Hard cap on the rows any single name rung returns. A bare ``__init__`` or
#: ``main`` matches hundreds of symbols; the caller needs enough of them to see
#: that the query was vague, not all of them.
NAME_MATCH_LIMIT = 200

#: Matches a tool enumerates when a target is ambiguous. The exact total always
#: ships alongside as ``match_count``; this caps only the list, because a bare
#: ``__init__`` matches 123 symbols on the SoleMD.Infra mirror and 123 entries
#: is the budget spent proving the question was vague rather than answering it.
#: Shared so get_symbol and get_context list the same ambiguity the same way.
MAX_AMBIGUITY_CANDIDATES = 20

#: A trailing source-file extension. Lowercase by convention in every language
#: repowise indexes, which is what separates ``mod.py`` (a path) from ``Foo.Bar``
#: (a nested scope) in the segment before ``::``.
_FILE_EXTENSION_RE = re.compile(r"\.[a-z0-9]{1,6}$")


def asserts_file_path(target: str) -> bool:
    """Whether *target* names a specific file before its ``::``.

    A caller that wrote ``nope/wrong.py::alpha`` asserted a file, and matching
    ``alpha`` in some other file would answer a question they did not ask —
    they get ``suggestions`` and choose. ``Class::method`` asserts no such
    thing: the first segment is a scope, and resolving it by name is the only
    way to answer at all.
    """
    head, sep, _ = target.partition("::")
    if not sep:
        return False
    return "/" in head or "\\" in head or bool(_FILE_EXTENSION_RE.search(head))


def parse_symbol_id(symbol_id: str) -> tuple[str | None, str | None]:
    """Split a ``"{path}::{name}"`` id. Either side may be None if missing.

    Tolerant of double-colons in qualified names like ``"Foo::Bar::baz"`` by
    splitting on the FIRST ``"::"`` only — the first segment is always the
    file path. Returns (file_path, name) where name may itself contain ``"::"``
    for nested qualified forms.
    """
    if not symbol_id or "::" not in symbol_id:
        return symbol_id or None, None
    file_part, _, name_part = symbol_id.partition("::")
    return (file_part or None, name_part or None)


def name_segments(name: str) -> list[str]:
    """Split a qualified name into its atomic segments, separator-agnostically."""
    segments = [name]
    for sep in NAME_SEPARATORS:
        segments = [part for seg in segments for part in seg.split(sep)]
    return [s for s in segments if s]


def name_variants(name: str) -> list[str]:
    """Generate all separator variants of a qualified name segment.

    Given ``"App.update_template_context"`` we yield the same name with every
    supported separator between segments, so a DB storing ``"App::method"``
    still resolves when the caller passed dot-form (or vice versa).

    Operates only on the *name* (post file-path), never on the path itself.
    """
    if not name:
        return []
    # Split on any of the known separators to get atomic segments.
    segments = name_segments(name)
    if not segments:
        return [name]
    variants: list[str] = []
    seen: set[str] = set()
    for sep in NAME_SEPARATORS:
        v = sep.join(segments)
        if v not in seen:
            seen.add(v)
            variants.append(v)
    # Also include the original as-is in case it used a mixed separator.
    if name not in seen:
        variants.append(name)
    return variants


def symbol_id_variants(symbol_id: str) -> list[str]:
    """Generate ``{file_path}::{name_variant}`` for every name separator form."""
    file_path, name = parse_symbol_id(symbol_id)
    if not file_path or not name:
        return [symbol_id]
    out: list[str] = []
    seen: set[str] = set()
    for nv in name_variants(name):
        sid = f"{file_path}::{nv}"
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    if symbol_id not in seen:
        out.append(symbol_id)
    return out


def bare_name(name: str) -> str:
    """Return the last name segment regardless of separator style."""
    tail = name
    for sep in NAME_SEPARATORS:
        tail = tail.rsplit(sep, 1)[-1]
    return tail


def order_candidates(rows: list[WikiSymbol], queried_file_path: str | None) -> list[WikiSymbol]:
    """Deterministically order a candidate list, best match first.

    Priority for the head slot:
      1. file_path matches the file_path embedded in the queried symbol_id
      2. deterministic tiebreak on the (id) primary key (ascending)

    Ambiguous lookups (len > 1) are NOT collapsed here — the caller decides
    whether to serve every candidate or just the head. The remainder is
    ordered by source position for readability.
    """
    if len(rows) <= 1:
        return rows

    def _head_key(r: WikiSymbol) -> tuple:
        file_match = 0 if (queried_file_path and r.file_path == queried_file_path) else 1
        return (file_match, r.id or "")

    head = min(rows, key=_head_key)
    rest = sorted(
        (r for r in rows if r is not head),
        key=lambda r: (r.file_path or "", r.start_line or 0, r.id or ""),
    )
    return [head, *rest]


def order_name_candidates(rows: list[WikiSymbol]) -> list[WikiSymbol]:
    """Order rows matched by *name* rather than by id, in source order.

    A name query carries no file to prefer, so ``order_candidates``' head slot
    would fall through to the row uuid — stable, but arbitrary, and arbitrary
    is the wrong thing to put first in a list a human reads to disambiguate.
    Source order (file, then line) groups a class with its methods and makes
    the same query render the same way twice.
    """
    return sorted(rows, key=lambda r: (r.file_path or "", r.start_line or 0, r.symbol_id or ""))


async def _match_name_rung(session, repo_id: str, condition, limit: int) -> list[WikiSymbol]:
    """Run one rung of the name ladder and return its rows."""
    res = await session.execute(
        select(WikiSymbol)
        .where(WikiSymbol.repository_id == repo_id, condition)
        .order_by(WikiSymbol.file_path, WikiSymbol.start_line, WikiSymbol.symbol_id)
        .limit(limit)
    )
    return list(res.scalars().all())


def _eq(column, value: str, *, ignore_case: bool):
    """Equality on *column*, case-folded when asked.

    ``ilike`` with every metacharacter escaped is equality that ignores case,
    and it is the portable spelling — ``lower()`` on an indexed column is not
    sargable on every backend, and SQLite's own ``=`` is case-sensitive for the
    non-ASCII identifiers this has to keep working for.
    """
    if not ignore_case:
        return column == value
    return column.ilike(escape_like(value), escape=LIKE_ESCAPE)


#: One rung: its name, the SQL that finds candidates, and the Python predicate
#: that decides whether a returned row is a case-*exact* match. Both halves are
#: needed because SQL equality and SQL ``LIKE`` disagree about case across (and
#: within) backends — SQLite's ``LIKE`` is case-insensitive for ASCII while its
#: ``=`` is not — so the case-sensitive pass has to confirm in Python what the
#: query only narrowed.
_NameRung = tuple[str, object, "Callable[[WikiSymbol], bool]"]


def _name_rungs(query: str, *, ignore_case: bool) -> list[_NameRung]:
    """The ladder for a path-less name query.

    Ordered strictly most- to least-specific, so an exact qualified name is
    never outranked by a leaf-name match that happens to be found first:

      * ``qualified_name`` — the whole query is the stored qualified name.
      * ``name`` — a single-segment query is a bare symbol name. Qualified
        names are folded in here because a module-level function stores the
        same string in both columns, and splitting them would order two
        spellings of one match against each other.
      * ``qualified_suffix`` — the query is the tail of a stored qualified
        name (``Class.method`` under ``pkg.mod.Class.method``), matched on a
        separator boundary so ``Class.method`` cannot match ``XClass.method``.
      * ``leaf_name`` — last resort for a qualified query whose index stores
        names unqualified: match the final segment alone. Broad by
        construction, which is why it runs only after everything else missed.
    """
    variants = name_variants(query)
    variant_set = set(variants)
    segments = name_segments(query)

    if len(segments) <= 1:
        return [
            (
                "name",
                or_(
                    _eq(WikiSymbol.name, query, ignore_case=ignore_case),
                    *(_eq(WikiSymbol.qualified_name, v, ignore_case=ignore_case) for v in variants),
                ),
                lambda r: r.name == query or (r.qualified_name or "") in variant_set,
            )
        ]

    suffixes = tuple(f"{sep}{variant}" for sep in NAME_SEPARATORS for variant in variants)
    suffix_terms = [
        WikiSymbol.qualified_name.ilike(f"%{escape_like(s)}", escape=LIKE_ESCAPE)
        if ignore_case
        else WikiSymbol.qualified_name.like(f"%{escape_like(s)}", escape=LIKE_ESCAPE)
        for s in suffixes
    ]
    leaf = bare_name(query)
    return [
        (
            "qualified_name",
            or_(*(_eq(WikiSymbol.qualified_name, v, ignore_case=ignore_case) for v in variants)),
            lambda r: (r.qualified_name or "") in variant_set,
        ),
        (
            "qualified_suffix",
            or_(*suffix_terms),
            lambda r: (r.qualified_name or "").endswith(suffixes),
        ),
        (
            "leaf_name",
            _eq(WikiSymbol.name, leaf, ignore_case=ignore_case),
            lambda r: r.name == leaf,
        ),
    ]


async def resolve_name_rows(
    session,
    repo_id: str,
    query: str,
    *,
    limit: int = NAME_MATCH_LIMIT,
) -> tuple[list[WikiSymbol], str | None]:
    """Resolve a path-less name query. Returns ``(rows, rung)``.

    Walks :func:`_name_rungs` case-sensitively and returns the first rung that
    produced rows. Only when the whole ladder comes back empty is it walked
    again case-insensitively — a case-folded match is a real answer for an
    agent that typed ``authservice`` for ``AuthService``, but it must never
    outrank an exact one in a language where ``Foo`` and ``foo`` are two
    different symbols. The case-insensitive rung is reported with a ``"_ci"``
    suffix so the caller can say how the match was reached.

    The case-sensitive pass re-checks each row in Python rather than trusting
    the SQL, because ``LIKE`` is case-insensitive on SQLite: without it the
    suffix rung would quietly serve a case-folded match as an exact one, and
    ``Foo.bar`` next to ``foo.Bar`` would read as ambiguous when only one of
    them was asked for.
    """
    if not query:
        return [], None
    for ignore_case in (False, True):
        for rung, condition, is_exact in _name_rungs(query, ignore_case=ignore_case):
            rows = await _match_name_rung(session, repo_id, condition, limit)
            if not ignore_case:
                rows = [r for r in rows if is_exact(r)]
            if rows:
                return order_name_candidates(rows), (f"{rung}_ci" if ignore_case else rung)
    return [], None


@dataclass(frozen=True)
class SymbolMatch:
    """The outcome of resolving one symbol target, with how it was reached.

    ``rows`` is every symbol the winning rung matched — never pre-collapsed to
    one. ``status`` is what the caller reports: a lookup matching several
    symbols is ``"ambiguous"``, and answering it with a single body would be a
    guess dressed as an answer.
    """

    query: str
    rows: list[WikiSymbol] = field(default_factory=list)
    #: Which rung matched: ``symbol_id`` / ``file_qualified_name`` /
    #: ``file_name`` / ``path_suffix`` for path-qualified targets, or one of
    #: the name rungs for path-less ones. A ``_ci`` suffix means the match was
    #: reached only by case-folding, which is worth reporting: an agent that
    #: asked for ``helper`` and got ``HELPER`` should be told why.
    rung: str | None = None
    #: True when the winning rung matched more rows than the cap allowed.
    truncated: bool = False

    @property
    def status(self) -> str:
        """``resolved`` | ``ambiguous`` | ``not_found`` — what the tool reports."""
        if not self.rows:
            return "not_found"
        return "ambiguous" if len(self.rows) > 1 else "resolved"

    def candidates(self) -> list[dict]:
        """The structured ambiguity payload: one metadata entry per match."""
        return [
            {
                "symbol_id": r.symbol_id,
                "file": r.file_path,
                "name": r.name,
                "qualified_name": r.qualified_name,
                "kind": r.kind,
                "start_line": r.start_line,
            }
            for r in self.rows
        ]


async def resolve_symbol_match(
    session,
    repo_id: str,
    target: str,
    *,
    limit: int = NAME_MATCH_LIMIT,
) -> SymbolMatch:
    """Resolve *target* in any accepted form and report how it resolved.

    The path-qualified ladder runs first (see :func:`resolve_symbol_rows`);
    when it misses, the whole target is retried as a path-less name, unless
    the target :func:`asserts_file_path`. That fallback is what makes
    ``Class::method`` work: ``parse_symbol_id`` reads the first segment as a
    file path, which resolves nothing, and the name ladder then matches the
    qualified tail.
    """
    rows, rung = await _resolve_path_qualified(session, repo_id, target)
    if rows or asserts_file_path(target):
        return SymbolMatch(query=target, rows=rows, rung=rung)

    name_rows, name_rung = await resolve_name_rows(session, repo_id, target, limit=limit)
    return SymbolMatch(
        query=target,
        rows=name_rows,
        rung=name_rung,
        truncated=len(name_rows) >= limit,
    )


async def resolve_symbol_rows(session, repo_id: str, symbol_id: str) -> list[WikiSymbol]:
    """Look up a symbol by id, qualified name, or bare name.

    Returns every row the first matching lookup stage produced, best match
    first (see :func:`order_candidates`); ``[]`` when nothing matched. A
    multi-row result means the target is genuinely ambiguous — overloads,
    re-exports, conditional defs, or simply a common leaf name — and the
    caller decides how to present that rather than having a guess made for it.
    :func:`resolve_symbol_match` is the same ladder with the rung and the
    ambiguity status attached.

    Language-agnostic: the qualified-name portion of the target is normalized
    across ``.``, ``::`` and ``/`` separators before matching, so callers can
    pass any of ``Class.method``, ``Class::method``, or ``Class/method`` and
    still resolve. Only the name part is normalized — file paths are never
    rewritten.
    """
    return (await resolve_symbol_match(session, repo_id, symbol_id)).rows


async def _resolve_path_qualified(
    session, repo_id: str, symbol_id: str
) -> tuple[list[WikiSymbol], str | None]:
    """The path-qualified half of the ladder. Returns ``(rows, rung)``."""
    file_path, name = parse_symbol_id(symbol_id)

    # 1. Exact symbol_id — try every separator variant.
    res = await session.execute(
        select(WikiSymbol).where(
            WikiSymbol.repository_id == repo_id,
            WikiSymbol.symbol_id.in_(symbol_id_variants(symbol_id)),
        )
    )
    rows = list(res.scalars().all())
    if rows:
        return order_candidates(rows, file_path), "symbol_id"

    if not name:
        return [], None

    variants = name_variants(name)

    # 2. Match on (file_path, qualified_name) across name variants.
    if file_path:
        res = await session.execute(
            select(WikiSymbol).where(
                WikiSymbol.repository_id == repo_id,
                WikiSymbol.file_path == file_path,
                WikiSymbol.qualified_name.in_(variants),
            )
        )
        rows = list(res.scalars().all())
        if rows:
            return order_candidates(rows, file_path), "file_qualified_name"

        # 3. Match on (file_path, name) — last segment of qualified name.
        res = await session.execute(
            select(WikiSymbol).where(
                WikiSymbol.repository_id == repo_id,
                WikiSymbol.file_path == file_path,
                WikiSymbol.name == bare_name(name),
            )
        )
        rows = list(res.scalars().all())
        if rows:
            return order_candidates(rows, file_path), "file_name"

    # 4. Suffix file-path match — the caller passed a bare filename or partial
    #    path ("answer.py::get_answer") instead of the full indexed path.
    #    Resolve against any file whose path ends with that segment on a "/"
    #    boundary, on the bare leaf name, so a remembered filename is not a
    #    dead end.
    if file_path and name:
        esc = escape_like(file_path.strip("/").replace("\\", "/"))
        res = await session.execute(
            select(WikiSymbol).where(
                WikiSymbol.repository_id == repo_id,
                WikiSymbol.name == bare_name(name),
                or_(
                    WikiSymbol.file_path == file_path.strip("/").replace("\\", "/"),
                    WikiSymbol.file_path.like(f"%/{esc}", escape=LIKE_ESCAPE),
                ),
            )
        )
        rows = list(res.scalars().all())
        if rows:
            return order_candidates(rows, file_path), "path_suffix"

    return [], None
