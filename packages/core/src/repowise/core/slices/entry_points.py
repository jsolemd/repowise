"""Query → entry points: where a task starts reading the code.

Two ways in, and they are not the same question.

* An **explicit** entry point ("src/auth/service.py", "src/auth/service.py::login",
  or a bare unique symbol name) is *resolved*: it either names a graph node or
  it does not, and a name that matches several symbols is ambiguous rather than
  silently collapsed to whichever row arrived first.
* A **task query** ("fix the login token refresh") is *nominated*: every node
  whose name, qualified name or path carries a query term becomes a candidate,
  and the candidates are scored.

Candidacy for the second is where :mod:`repowise.core.entry_candidacy` is
reused, and reused for exactly what it owns. ``not_an_execution_start`` answers
"can a reader enter the system here at all" from a path and a language alone —
a ``server.json`` describes the system and a resolver's nested ``index.py``
dispatches within it, so neither is where a task starts. Both remain perfectly
good slice *members* if something reaches them; they are just not seeds. The
module's own docstring makes the split explicit: candidacy and ordering are two
questions, and it owns only the first. Ordering is the scoring below.

Everything here runs on the SQL graph store — ``graph_nodes`` rows written by
ingestion. No embedder, no vector store, no second index: an entry point that
only a semantic model can find is not reproducible across two calls, and a
slice that cannot be rebuilt from its recorded seeds is not extendable.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.entry_candidacy import not_an_execution_start
from repowise.core.exclusion import is_excluded
from repowise.core.persistence.models import GraphNode
from repowise.core.persistence.sql import LIKE_ESCAPE, escape_like
from repowise.core.slices.errors import EntryPointsUnresolvedError
from repowise.core.slices.models import EntryCandidate
from repowise.core.slices.nodes import is_test_node, node_path

#: Words that carry no locating power in a task sentence. Deliberately short:
#: an over-eager stop list drops the one domain noun that matters ("index",
#: "store", "handler" are all real module names in real repos).
# fmt: off
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "into", "when",
        "where", "what", "which", "how", "why", "add", "fix", "make", "use",
        "using", "need", "needs", "want", "should", "would", "could", "can",
        "does", "did", "are", "was", "were", "has", "have", "had", "its",
        "our", "your", "their", "not", "but", "all", "any", "some", "new",
        "old", "code", "file", "files", "please", "then", "than", "also",
    }
)
# fmt: on

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")

#: Per-term row cap on each candidate query. A term like "get" matches
#: thousands of symbols; the cap keeps one vague word from deciding the whole
#: candidate set, and the ORDER BY makes the cut ranked rather than arbitrary.
_TERM_ROW_CAP = 200

#: Weights for the candidate score, before normalisation. Exact name match
#: dominates on purpose: a task naming a symbol means that symbol.
_W_EXACT_NAME = 3.0
_W_NAME_SUBSTRING = 1.5
_W_QUALIFIED = 1.0
_W_PATH = 1.0
_W_DECLARED_ENTRY = 1.5
_W_CENTRALITY = 1.0

#: Multipliers, applied after the additive terms.
_TEST_DAMPING = 0.35
_NON_EXECUTION_START_DAMPING = 0.4


def query_terms(task: str) -> list[str]:
    """Locating terms from a task sentence, longest first, deduplicated.

    Identifiers are also split on their case boundaries, so a task mentioning
    ``refreshToken`` matches a symbol named ``refresh`` and a path containing
    ``token``. The full identifier is kept too and, being longer, sorts first.
    """
    seen: dict[str, None] = {}
    for word in _WORD_RE.findall(task or ""):
        lowered = word.lower()
        if len(lowered) >= 3 and lowered not in _STOP_WORDS:
            seen.setdefault(lowered, None)
        for part in _CAMEL_RE.findall(word):
            part_lower = part.lower()
            if len(part_lower) >= 3 and part_lower not in _STOP_WORDS:
                seen.setdefault(part_lower, None)
        for part in word.split("_"):
            part_lower = part.lower()
            if len(part_lower) >= 3 and part_lower not in _STOP_WORDS:
                seen.setdefault(part_lower, None)
    return sorted(seen, key=lambda t: (-len(t), t))


async def _rows_for_term(
    session: AsyncSession,
    repo_id: str,
    term: str,
    *,
    include_tests: bool,
) -> list[GraphNode]:
    pattern = f"%{escape_like(term)}%"
    conditions = [
        GraphNode.node_id.like(pattern, escape=LIKE_ESCAPE),
        GraphNode.name.like(pattern, escape=LIKE_ESCAPE),
        GraphNode.qualified_name.like(pattern, escape=LIKE_ESCAPE),
    ]
    stmt = select(GraphNode).where(
        GraphNode.repository_id == repo_id,
        GraphNode.node_type.in_(("file", "symbol")),
        or_(*conditions),
    )
    if not include_tests:
        stmt = stmt.where(GraphNode.is_test.is_(False))
    # The cut has to be ranked or a vague term returns whichever rows the
    # planner walked first; pagerank then node_id keeps it deterministic.
    stmt = stmt.order_by(GraphNode.pagerank.desc(), GraphNode.node_id).limit(_TERM_ROW_CAP)
    result = await session.execute(stmt)
    return list(result.scalars().all())


def _score_candidate(
    node: GraphNode,
    terms: list[str],
    *,
    max_pagerank: float,
    include_tests: bool,
) -> tuple[float, list[str], list[str]]:
    """Raw score, matched terms and human reasons for one candidate node."""
    path = node_path(node)
    name = (node.name or "").lower()
    qualified = (node.qualified_name or "").lower()
    lowered_path = path.lower()

    raw = 0.0
    matched: list[str] = []
    reasons: list[str] = []

    for term in terms:
        hit = False
        if name and name == term:
            raw += _W_EXACT_NAME
            reasons.append(f"name is exactly '{term}'")
            hit = True
        elif name and term in name:
            raw += _W_NAME_SUBSTRING
            reasons.append(f"name contains '{term}'")
            hit = True
        if qualified and term in qualified and (not name or term not in name):
            raw += _W_QUALIFIED
            reasons.append(f"qualified name contains '{term}'")
            hit = True
        if term in lowered_path:
            raw += _W_PATH
            reasons.append(f"path contains '{term}'")
            hit = True
        if hit:
            matched.append(term)

    if raw <= 0.0:
        return 0.0, [], []

    if node.is_entry_point:
        raw += _W_DECLARED_ENTRY
        reasons.append("ingestion flagged this file as an execution entry point")

    if max_pagerank > 0.0:
        raw += _W_CENTRALITY * (float(node.pagerank or 0.0) / max_pagerank)

    if is_test_node(node) and not include_tests:
        raw *= _TEST_DAMPING
        reasons.append("test material, demoted as a starting point")

    # entry_candidacy owns exactly this question: can a reader enter the system
    # here at all. Damped rather than dropped — a config file or a re-export
    # barrel is a legitimate slice member, just never the place a task starts.
    if not_an_execution_start(path, node.language or ""):
        raw *= _NON_EXECUTION_START_DAMPING
        reasons.append("not an execution start (config/markup or a nested glue module)")

    return raw, matched, reasons


async def nominate_entry_points(
    session: AsyncSession,
    repo_id: str,
    task: str,
    *,
    limit: int = 5,
    include_tests: bool = False,
    exclude_spec: Any = None,
) -> list[EntryCandidate]:
    """Score every node a task query names, best first.

    Returns an empty list when the task yields no usable terms or nothing
    matches; the caller decides whether that is an error, because an explicit
    entry point may still have been supplied alongside the query.
    """
    terms = query_terms(task)
    if not terms:
        return []

    by_node: dict[str, GraphNode] = {}
    for term in terms:
        for node in await _rows_for_term(session, repo_id, term, include_tests=include_tests):
            by_node.setdefault(node.node_id, node)

    if exclude_spec is not None:
        by_node = {
            nid: node
            for nid, node in by_node.items()
            if not is_excluded(node_path(node), exclude_spec)
        }
    if not by_node:
        return []

    max_pagerank = max((float(n.pagerank or 0.0) for n in by_node.values()), default=0.0)

    scored: list[tuple[float, GraphNode, list[str], list[str]]] = []
    for node in by_node.values():
        raw, matched, reasons = _score_candidate(
            node, terms, max_pagerank=max_pagerank, include_tests=include_tests
        )
        if raw > 0.0:
            scored.append((raw, node, matched, reasons))
    if not scored:
        return []

    top_raw = max(raw for raw, _, _, _ in scored)
    scored.sort(key=lambda row: (-row[0], row[1].node_id))

    return [
        EntryCandidate(
            node_id=node.node_id,
            node_type=node.node_type,
            file_path=node_path(node),
            name=node.name,
            kind=node.kind,
            language=node.language or "",
            score=raw / top_raw if top_raw else 0.0,
            matched_terms=tuple(matched),
            reasons=tuple(reasons),
            is_entry_point=bool(node.is_entry_point),
            pagerank=float(node.pagerank or 0.0),
        )
        for raw, node, matched, reasons in scored[:limit]
    ]


async def resolve_explicit_entry_points(
    session: AsyncSession,
    repo_id: str,
    requested: list[str],
) -> tuple[list[EntryCandidate], list[str], list[dict[str, Any]]]:
    """Resolve caller-named entry points exactly.

    Returns ``(resolved, unresolved, near_misses)``. An exact ``node_id`` wins,
    which keeps a file path out of symbol-name heuristics. A bare name resolves
    only when it is unique in the repo; several matches stay unresolved and
    every match is reported as a near miss so the caller can pick one, because
    picking for them is how a slice ends up rooted in the wrong function.
    """
    resolved: list[EntryCandidate] = []
    unresolved: list[str] = []
    near_misses: list[dict[str, Any]] = []

    for raw_target in requested:
        target = raw_target.strip()
        if not target:
            continue
        exact = await session.execute(
            select(GraphNode).where(
                GraphNode.repository_id == repo_id,
                GraphNode.node_id == target,
            )
        )
        node = exact.scalar_one_or_none()
        matched_by = "node_id"

        if node is None:
            by_name = await session.execute(
                select(GraphNode)
                .where(
                    GraphNode.repository_id == repo_id,
                    GraphNode.node_type == "symbol",
                    GraphNode.name == target,
                )
                .order_by(GraphNode.pagerank.desc(), GraphNode.node_id)
                .limit(10)
            )
            matches = list(by_name.scalars().all())
            if len(matches) == 1:
                node = matches[0]
                matched_by = "symbol_name"
            elif len(matches) > 1:
                unresolved.append(target)
                near_misses.extend(
                    {
                        "requested": target,
                        "node": m.node_id,
                        "file": node_path(m),
                        "why": "symbol name is ambiguous; name one of these node ids",
                    }
                    for m in matches
                )
                continue

        if node is None:
            unresolved.append(target)
            continue

        resolved.append(
            EntryCandidate(
                node_id=node.node_id,
                node_type=node.node_type,
                file_path=node_path(node),
                name=node.name,
                kind=node.kind,
                language=node.language or "",
                score=1.0,
                matched_terms=(),
                reasons=(f"requested explicitly, matched by {matched_by}",),
                is_entry_point=bool(node.is_entry_point),
                pagerank=float(node.pagerank or 0.0),
            )
        )

    return resolved, unresolved, near_misses


async def resolve_entry_points(
    session: AsyncSession,
    repo_id: str,
    *,
    task: str,
    entry_points: list[str] | None = None,
    limit: int = 5,
    include_tests: bool = False,
    exclude_spec: Any = None,
) -> list[EntryCandidate]:
    """The one call a slice build makes. Raises rather than returning nothing.

    **An explicit entry point suppresses nomination entirely.** Naming a
    starting node means "start here", and folding query-nominated seeds in
    beside it changes the shape of the resulting slice in a way the caller did
    not ask for: every extra seed sits at distance 0, so the walk's whole
    distance field — and therefore its ranking and its frontier — is measured
    from somewhere the caller never named. The task is still used, for
    ranking: :func:`query_terms` weights members the task's own words mention,
    wherever the walk found them.

    Callers who genuinely want both pass both as ``entry_points``.
    """
    if entry_points:
        resolved, unresolved, near_misses = await resolve_explicit_entry_points(
            session, repo_id, entry_points
        )
        if not resolved:
            raise EntryPointsUnresolvedError(
                task=task,
                requested=entry_points,
                unresolved=unresolved,
                near_misses=near_misses,
            )
        return resolved

    nominated = await nominate_entry_points(
        session,
        repo_id,
        task,
        limit=max(1, limit),
        include_tests=include_tests,
        exclude_spec=exclude_spec,
    )
    if not nominated:
        raise EntryPointsUnresolvedError(task=task, requested=None, unresolved=[])
    return nominated


def count_query_hits(terms: list[str], *fields: str | None) -> int:
    """How many of *terms* appear in any of *fields*.

    Used to give reached members the same query signal seeds get. Without it
    the ranking's ``query`` component is zero for everything the walk found,
    and a member the task names by name ranks below one it never mentions.
    """
    if not terms:
        return 0
    haystack = " ".join(f.lower() for f in fields if f)
    return sum(1 for term in terms if term in haystack)
