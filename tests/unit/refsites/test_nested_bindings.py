"""Which function-local bindings become addressable nested definitions.

This file exists because the question "why does ``weaveTo`` have no definition
row?" has two very different answers that look identical from the store, and
telling them apart by reading rows is not possible:

1. the extraction cannot see function-nested arrow bindings at all, or
2. the extraction sees them fine, and *this particular* binding sits under an
   anonymous callback, where A3's parentage rule refuses to attribute it.

The answer is (2), and every claim below is a hand check against the fixture
text in this module. ``TS_NESTING`` is written so the two cases sit side by
side: ``arrowConst`` is declared directly in ``outer``'s body and becomes
``outer::arrowConst``; ``refused`` is declared inside ``xs.map((x) => {…})``
and becomes nothing at all. One token of difference in the ancestor chain,
opposite outcomes — that is the boundary, stated as a test rather than as a
comment someone has to find.

The refusal is owned by ``_finalize_symbol_parentage`` in
:mod:`repowise.core.ingestion.parser`: a callable ancestor that maps to no
named extracted candidate suppresses the nested symbol rather than leaping
across it to a more distant named owner. ``test_lambda_gap.py`` measures what
that costs on the graph side; this file fixes what it means for reference
sites, so a future change to the query files or the parentage rule has to
come past an explicit statement of the current contract.

Two binding forms are recorded here as *known gaps* rather than as behaviour
anyone chose per-site. Both are properties of the ``.scm`` query files, and
both are noted at their tests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.store import SqlReferenceSiteStore
from repowise.core.refsites.taxonomy import TIER_AST, TIER_TEXTUAL, ReferenceKind
from repowise.core.source_search.worktree import WorkingTreeDivergence
from repowise.server import source_search_wiring as w

# ---------------------------------------------------------------------------
# The fixture: one named function containing every binding form that matters
# ---------------------------------------------------------------------------

#: Line numbers are asserted below, so this text is positional. ``outer`` is a
#: plain named function declaration; everything interesting is inside it.
TS_NESTING = """\
export function outer(xs: number[]) {
  const arrowConst = (a: number) => a + 1;
  let arrowLet = (b: number) => b + 2;
  var arrowVar = (c: number) => c + 3;
  const funcExpr = function (d: number) { return d + 4; };
  function nestedDecl(e: number) { return e + 5; }
  const notAFunction = 41;
  const deep = () => {
    const grandchild = (g: number) => g + 6;
    return grandchild(1);
  };
  const mapped = xs.map((x) => {
    const refused = (h: number) => h + 7;
    return refused(x);
  });
  return [arrowConst, arrowLet, arrowVar, funcExpr, nestedDecl, notAFunction, deep, mapped];
}

const fileScope = (z: number) => z;
"""

TS_NESTING_PATH = "nesting.ts"

#: The Python side of the same question, used for the cross-language parity
#: claim. A nested ``def`` is a symbol; a lambda bound to a name is not.
PY_NESTING = """\
def outer(xs):
    def nested(a):
        return a + 1

    helper = lambda b: b + 2
    return nested(1), helper(2)
"""

PY_NESTING_PATH = "nesting.py"


@pytest.fixture
def nesting_repo(tmp_path: Path) -> Path:
    """A repo whose only property is the nesting in the two files above."""
    (tmp_path / TS_NESTING_PATH).write_text(TS_NESTING)
    (tmp_path / PY_NESTING_PATH).write_text(PY_NESTING)
    (tmp_path / ".git").mkdir()
    return tmp_path


def _definitions(result, path: str) -> dict[str, str]:
    """``name -> target_symbol_id`` for every DEFINITION site in *path*."""
    return {
        site.name: site.target_symbol_id
        for site in result.sites
        if site.file_path == path and site.kind is ReferenceKind.DEFINITION
    }


def _sites_named(result, path: str, name: str) -> list:
    return sorted(
        (
            site
            for site in result.sites
            if site.file_path == path and site.name == name
        ),
        key=lambda site: (site.start_line, site.start_col),
    )


# ---------------------------------------------------------------------------
# What DOES become a nested definition
# ---------------------------------------------------------------------------


def test_a_function_nested_arrow_const_is_a_nested_definition(nesting_repo):
    """``const f = (…) => …`` inside a named function is addressable.

    This is the claim the whole file turns on: the chain is
    ``<file>::<Parent>::<name>``, matching the served ``symbol_path``, and it
    is produced by the ordinary AST tier — no sweep, no fallback.
    """
    result = extract_repository(nesting_repo)
    site = _sites_named(result, TS_NESTING_PATH, "arrowConst")[0]

    assert site.kind is ReferenceKind.DEFINITION
    assert site.target_symbol_id == "nesting.ts::outer::arrowConst"
    assert site.enclosing_symbol_id == "nesting.ts::outer"
    assert site.tier == TIER_AST
    assert site.start_line == 2
    assert site.range_exact


def test_a_nested_let_arrow_binding_is_recorded_the_same_way(nesting_repo):
    """``let`` is not a second-class declaration keyword here."""
    result = extract_repository(nesting_repo)
    site = _sites_named(result, TS_NESTING_PATH, "arrowLet")[0]

    assert site.kind is ReferenceKind.DEFINITION
    assert site.target_symbol_id == "nesting.ts::outer::arrowLet"
    assert site.enclosing_symbol_id == "nesting.ts::outer"


def test_a_nested_function_declaration_produces_the_same_chain(nesting_repo):
    """Parity: the declaration form and the arrow-binding form agree.

    Worth pinning separately because the two reach the symbol table through
    different query patterns and could drift into different id shapes without
    anything else noticing.
    """
    result = extract_repository(nesting_repo)
    site = _sites_named(result, TS_NESTING_PATH, "nestedDecl")[0]

    assert site.kind is ReferenceKind.DEFINITION
    assert site.target_symbol_id == "nesting.ts::outer::nestedDecl"
    assert site.enclosing_symbol_id == "nesting.ts::outer"


def test_nesting_composes_to_more_than_one_level(nesting_repo):
    """An arrow inside an arrow inside a function keeps the whole chain.

    ``deep`` is itself a nested arrow binding, so ``grandchild`` proves the
    parent chain is built by walking real ancestry rather than by special-
    casing one level of nesting.
    """
    result = extract_repository(nesting_repo)
    definitions = _definitions(result, TS_NESTING_PATH)

    assert definitions["deep"] == "nesting.ts::outer::deep"
    assert definitions["grandchild"] == "nesting.ts::outer::deep::grandchild"


def test_a_file_scope_arrow_const_is_unchanged_in_shape(nesting_repo):
    """The file-scope case keeps its two-segment id and gains no parent."""
    result = extract_repository(nesting_repo)
    site = _sites_named(result, TS_NESTING_PATH, "fileScope")[0]

    assert site.kind is ReferenceKind.DEFINITION
    assert site.target_symbol_id == "nesting.ts::fileScope"
    assert site.enclosing_symbol_id is None


def test_a_call_to_a_nested_binding_resolves_to_it(nesting_repo):
    """The definition is not decorative: calls bind to the nested chain.

    Without this, a nested definition row would be an id nothing points at,
    and ``get_reference_sites`` on the helper would still come back empty.
    """
    result = extract_repository(nesting_repo)
    calls = [
        site
        for site in _sites_named(result, TS_NESTING_PATH, "grandchild")
        if site.kind is ReferenceKind.CALL
    ]

    assert calls
    assert all(
        site.target_symbol_id == "nesting.ts::outer::deep::grandchild" for site in calls
    )


# ---------------------------------------------------------------------------
# What does NOT — the ratified refusal
# ---------------------------------------------------------------------------


def test_a_binding_under_an_anonymous_callback_is_refused(nesting_repo):
    """The A3 boundary, stated positively: no definition, no invented parent.

    ``refused`` is declared inside ``xs.map((x) => {…})``. Its only path to a
    named ancestor crosses an anonymous arrow, and the parentage rule declines
    to leap across it to ``outer``. What survives is the textual sweep's
    identifier row with no target — the name is plainly in the file, and the
    index says exactly that much about it and nothing more.

    This is the shape of the real ``ConnectiveThreads.tsx`` ``weaveTo``, which
    sits inside ``active.map((thread) => {…})``. Its absence from the store is
    this rule firing, not a gap in extraction.
    """
    result = extract_repository(nesting_repo)
    sites = _sites_named(result, TS_NESTING_PATH, "refused")

    assert ReferenceKind.DEFINITION not in {site.kind for site in sites}
    assert all(site.target_symbol_id is None for site in sites)

    declaration = sites[0]
    assert declaration.kind is ReferenceKind.IDENTIFIER
    assert declaration.tier == TIER_TEXTUAL
    # No enclosing symbol either: attributing it to ``outer`` would be the
    # same leap by another name.
    assert declaration.enclosing_symbol_id is None


def test_the_refusal_is_scoped_to_the_ancestor_not_to_the_binding_form(nesting_repo):
    """``arrowConst`` and ``refused`` are the same construct, one nesting apart.

    Stated as one assertion so the contrast cannot be broken by a change that
    only looks at one of them.
    """
    result = extract_repository(nesting_repo)
    definitions = _definitions(result, TS_NESTING_PATH)

    assert "arrowConst" in definitions
    assert "refused" not in definitions


def test_a_non_callable_local_binding_is_not_a_definition(nesting_repo):
    """``const notAFunction = 41`` stays a local variable, not a symbol.

    The anchored query that admits module constants is deliberately restricted
    to file scope; without that restriction every ``const x = useMemo(...)``
    in the repository would enter the index as a definition.
    """
    result = extract_repository(nesting_repo)

    assert "notAFunction" not in _definitions(result, TS_NESTING_PATH)


# ---------------------------------------------------------------------------
# Known query-file gaps, recorded rather than discovered again
# ---------------------------------------------------------------------------


def test_a_var_arrow_binding_is_not_recorded_at_any_scope(nesting_repo):
    """KNOWN GAP: ``var f = (…) => …`` produces no definition, nested or not.

    The arrow-binding patterns in ``queries/typescript.scm`` match
    ``(lexical_declaration …)``, which covers ``const`` and ``let`` only;
    ``var`` parses as ``variable_declaration`` and matches nothing. This is a
    query-file gap, so closing it changes the parser fingerprint and forces a
    full re-parse of every corpus — it is not a per-call-site decision.

    Asserted rather than skipped so that closing the gap fails here loudly and
    is updated deliberately.
    """
    result = extract_repository(nesting_repo)

    assert "arrowVar" not in _definitions(result, TS_NESTING_PATH)


def test_a_nested_function_expression_binding_is_not_recorded(nesting_repo):
    """KNOWN GAP: ``const f = function (…) {…}`` is file-scope only.

    ``(function_expression)`` appears in the query's binding list only under
    the ``(program …)`` anchor, so the same text nested inside a function body
    matches nothing. The anchor is the anti-flooding decision described at
    ``test_a_non_callable_local_binding_is_not_a_definition``; lifting it for
    function expressions alone is a query-file change with the same
    fingerprint cost as the ``var`` gap above.
    """
    result = extract_repository(nesting_repo)

    assert "funcExpr" not in _definitions(result, TS_NESTING_PATH)


# ---------------------------------------------------------------------------
# Cross-language parity, and the identifier-row question it settles
# ---------------------------------------------------------------------------


def test_python_nested_defs_and_lambda_bindings_split_the_same_way(nesting_repo):
    """TypeScript's two outcomes are Python's two outcomes.

    A nested ``def`` is a definition on the ``outer::nested`` chain; a name
    bound to a lambda is not a symbol and survives only as a textual
    identifier with no target. The TS behaviour above is therefore the
    existing cross-language contract, not a dialect of its own.
    """
    result = extract_repository(nesting_repo)
    definitions = _definitions(result, PY_NESTING_PATH)

    assert definitions["nested"] == "nesting.py::outer::nested"
    assert "helper" not in definitions

    helper = _sites_named(result, PY_NESTING_PATH, "helper")[0]
    assert helper.kind is ReferenceKind.IDENTIFIER
    assert helper.target_symbol_id is None


@pytest.mark.parametrize(
    ("path", "name", "line"),
    [(TS_NESTING_PATH, "arrowConst", 2), (PY_NESTING_PATH, "nested", 2)],
)
def test_a_recorded_definition_leaves_no_rival_identifier_row(
    nesting_repo, path, name, line
):
    """A nested definition is one row, not a definition plus a textual twin.

    This is the identifier-row question, and it is settled by construction
    rather than by a rule anyone maintains: the textual sweep only claims
    positions Tier A did not, so a definition at the binding site consumes
    that position and the sweep never re-emits it. Asserted in both languages
    because a divergence would mean the two tiers had stopped agreeing about
    what a claimed position is.
    """
    result = extract_repository(nesting_repo)
    at_declaration = [
        site for site in _sites_named(result, path, name) if site.start_line == line
    ]

    assert len(at_declaration) == 1
    assert at_declaration[0].kind is ReferenceKind.DEFINITION


# ---------------------------------------------------------------------------
# End to end: the served row's contains_symbols
# ---------------------------------------------------------------------------


async def test_definitions_in_range_returns_the_nested_binding(
    async_session, repo_id, nesting_repo
):
    """The store method the naming pass calls sees the nested arrow binding.

    ``_name_contained_definitions`` can only name what ``definitions_in_range``
    returns, so this is the step between "the extractor recorded it" and "the
    served row says it".
    """
    store = SqlReferenceSiteStore(async_session)
    await store.replace_repository(repo_id, extract_repository(nesting_repo))
    await async_session.commit()

    sites = await store.definitions_in_range(repo_id, TS_NESTING_PATH, 1, 200)
    chains = {site.target_symbol_id for site in sites}

    assert "nesting.ts::outer::arrowConst" in chains
    # And the refused binding is absent here too, which is why a row covering
    # these lines can never be named with it.
    assert not any(site.name == "refused" for site in sites)


class _Inner:
    """A coordinator that has already ranked; only its response matters here."""

    def __init__(self, response: dict) -> None:
        self._response = response

    async def search(self, *_args, **_kwargs) -> dict:
        return self._response


def _patch_status(monkeypatch) -> None:
    """Stand in for the index status read with a clean working tree."""
    from repowise.core.source_search import status as status_mod

    async def fake_inspect(*_args, **_kwargs):
        return SimpleNamespace(
            state="ready",
            generation_id="g1",
            generation_sequence=1,
            pending_updates=0,
            blocked_updates=0,
            building_updates=0,
            ready_updates=0,
            stale_files=(),
            working_tree=WorkingTreeDivergence(checked=True),
            last_error=None,
            degraded=False,
        )

    monkeypatch.setattr(status_mod, "inspect_source_index", fake_inspect)


@pytest.fixture
def session_factory(async_engine):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    return async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)


async def test_a_served_window_names_the_parent_and_its_nested_arrow(
    monkeypatch, async_session, repo_id, session_factory, nesting_repo
):
    """The end-to-end claim: ``["outer", "outer::arrowConst"]`` on a served row.

    This is the shape the ``ConnectiveThreads.tsx`` row was expected to have.
    It is reachable whenever the helper is declared directly in the named
    function's body — and the same assertion records that the binding under
    the anonymous callback is still not named, from the served row rather than
    from the extractor, which is where a reader would actually notice it.
    """
    store = SqlReferenceSiteStore(async_session)
    await store.replace_repository(repo_id, extract_repository(nesting_repo))
    await async_session.commit()
    _patch_status(monkeypatch)

    row = {
        "file": TS_NESTING_PATH,
        "target_path": TS_NESTING_PATH,
        "name": TS_NESTING_PATH,
        "kind": "file_window",
        "source": "file_window",
        "snippet": TS_NESTING,
        "relevance_score": 0.5,
        "evidence": {},
        "start_line": 1,
        "end_line": 20,
    }
    coordinator = w._StatusCoordinator(
        _Inner({"results": [row], "mode": "hybrid", "confidence": "confident", "_meta": {}}),
        nesting_repo,
        object(),
        object(),
        session_factory,
    )

    response = await coordinator.search("how does outer build its helpers")
    named = response["results"][0]["contains_symbols"]

    assert "outer" in named
    assert "outer::arrowConst" in named
    assert not any(name.endswith("refused") for name in named)
