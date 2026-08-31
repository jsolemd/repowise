"""One answer to "what tests guard this file", shared by every tool that asks.

``get_context`` and ``get_risk`` both report on a file's tests, and used to do
it from two unrelated sources: the context card carried the health engine's
index-time ``has_test_file`` (an exact paired-filename check — is there a
``test_foo.py`` next to ``foo.py``), while ``get_risk`` computed ``test_gap``
live from a ``LIKE '%test_foo%'`` scan of the test nodes. The substring scan
matches ``test_foo_helpers.py`` and the paired check does not, so on the
SoleMD.Infra mirror the two tools gave opposite answers about the same file 49
times out of 751 — and an agent that reads both in one turn has no way to tell
which one is lying.

Neither was reading the evidence the index already holds. A test file that
imports a source file is a recorded edge, and coverage ingestion records the
exact test-to-file map. This module walks those, strongest evidence first, and
both tools answer from it:

  1. ``self``     — the file is itself test material; it needs no guard.
  2. ``coverage`` — a coverage run proved these tests execute this file.
  3. ``graph``    — the dependency graph records a test reaching it. Two
                    tiers, strongest first: a test whose symbols can execute
                    into this file's symbols within three call hops
                    (``call-graph``), else a test file that imports it
                    (``import-graph``). Both are recorded edges, so both are
                    proof; the tier travels with the answer as
                    ``test_linkage_via``.
  4. ``naming``   — no test references it, but an unambiguous file named for
                    it exists. Possible evidence only: it never clears a gap.
  5. ``none``     — nothing found.

The basis travels with the answer, because "12 tests import this" and "a file
called test_foo.py exists somewhere" are not the same claim and an agent
deciding whether it can refactor safely needs to know which one it has.

Why the graph rung leads with the call graph
--------------------------------------------
It did not, and that was a measured recall hole. This rung shipped asking one
question — "does a test file import this one?" — while ``pr_blast`` had already
moved to ``test_reachability``'s three-hop call walk, so ``get_risk``'s two
modes carried two different definitions of "a test reaches this". Upstream
measured both against a real ``coverage run --contexts=test``: the 1-hop import
relation finds 19.5% of the files a test provably executed, the 3-hop call walk
finds 27.7% *and* is more precise (91.7% against 72.1%). The import relation
structurally cannot see transitive execution — a test that calls a service that
calls the changed helper imports neither.

So the call walk runs first and the import relation is kept as the second tier,
seeded only where the call walk stayed silent. That is upstream's own tier
discipline (``tests_reaching_by_tier``), and its measurement says the weaker
tier must not speak over the stronger one: unioning them answers no additional
file and costs 1.7 points of precision on which tests are named.

The second tier is *this module's* import query, not upstream's, and that is
deliberate. Upstream's walks ``FILE_DEPENDENCY_EDGE_TYPES``; this one walks
every edge that is not containment or co-change, which is a superset. Keeping
it as the fallback makes the swap monotone: every file this rung called
``graph`` before still does, so recall can only rise and nothing that was
proven becomes possible.

What does *not* change is the vocabulary. A three-hop call edge is a recorded
edge, not a name match, so it stays on the proven side: ``basis`` is ``graph``,
``tested`` is True, and the tests land in ``guarding_tests``. Naming similarity
remains the ``naming`` rung, remains ``possible_tests``, and still never clears
a gap.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.ingestion.models import NON_DEPENDENCY_EDGE_TYPES
from repowise.core.persistence.models import GraphEdge, GraphNode

#: Cap on definitive or possible test paths named in a payload. The count is
#: always exact; the list is a run-list, and a hundred paths is not one.
MAX_TEST_LINKS = 10


@dataclass(frozen=True)
class TestLinkage:
    """What guards *file_path*, and on what evidence."""

    file_path: str
    #: Guarding or possible test files, sorted and capped at MAX_TEST_LINKS.
    tests: list[str] = field(default_factory=list)
    #: Total linked test paths found, before the cap.
    total: int = 0
    #: ``self`` | ``coverage`` | ``graph`` | ``naming`` | ``none``.
    basis: str = "none"
    #: Which tier of the ``graph`` basis answered: ``call-graph`` (a test can
    #: execute into this file) or ``import-graph`` (a test file references it).
    #: None on every other basis. Both tiers are proof; the distinction is how
    #: strong the proof is, and an agent choosing what to run first wants it.
    via: str | None = None

    @property
    def tested(self) -> bool:
        """Whether authoritative evidence proves this file is tested."""
        return self.basis in {"self", "coverage", "graph"}

    def as_payload(self) -> dict:
        """The additive block both tools attach to a file's card."""
        out: dict = {"tested": self.tested, "test_linkage_basis": self.basis}
        if self.via:
            out["test_linkage_via"] = self.via
        if self.tests:
            if self.tested:
                out["guarding_tests"] = self.tests
                out["guarding_test_count"] = self.total
            else:
                out["possible_tests"] = self.tests
                out["possible_test_count"] = self.total
        return out


def subject_stem(test_filename: str) -> str | None:
    """The source stem *test_filename* names itself the test for, or None.

    ``test_chunker.py`` → ``chunker``; ``chunker_test.go`` → ``chunker``;
    ``chunker.spec.ts`` → ``chunker``; ``ChunkerTest.java`` → ``Chunker``.

    Every marker comes from the language registry, the same source
    ``core.test_paths`` reads to decide what a test file *is*. A local list of
    ``("test_", "_test", ".spec.", …)`` would be the exact copy that module
    exists to prevent: it has to be edited per ecosystem, and the two copies
    that were edited per extension are what let #288 regress twice. Adding an
    ecosystem is a row on a language spec, and this rung follows for free.
    """
    from repowise.core.ingestion.languages.registry import REGISTRY

    filename = posixpath.basename(test_filename.replace("\\", "/"))
    stem, ext = posixpath.splitext(filename)
    if not stem:
        return None

    lowered = stem.lower()
    for infix in REGISTRY.test_infixes():
        # ``.spec.`` / ``.test.`` sit inside the name, so splitext already left
        # the marker on the stem: "chunker.spec.ts" → stem "chunker.spec".
        marker = infix.rstrip(".")
        if lowered.endswith(marker):
            return stem[: -len(marker)]
    for prefix in REGISTRY.test_stem_prefixes():
        if lowered.startswith(prefix):
            return stem[len(prefix) :]
    for suffix in REGISTRY.test_stem_suffixes():
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    camel_re = REGISTRY.camel_test_res_by_extension().get(ext.lower())
    if camel_re is not None:
        match = camel_re.search(stem)
        if match is not None:
            return stem[: match.start()]
    return None


async def _is_test_node(session: AsyncSession, repo_id: str, file_path: str) -> bool:
    res = await session.execute(
        select(GraphNode.is_test)
        .where(GraphNode.repository_id == repo_id, GraphNode.node_id == file_path)
        .limit(1)
    )
    return res.scalar_one_or_none() is True


async def _coverage_tests(session: AsyncSession, repo_id: str, file_path: str) -> list[str]:
    from repowise.core.persistence.crud import tests_covering

    rows = await tests_covering(session, repo_id, file_path)
    return sorted({r["test_file"] for r in rows if r.get("test_file")})


async def _call_graph_tests(session: AsyncSession, repo_id: str, file_path: str) -> list[str]:
    """Test files whose code can execute into *file_path*, within three hops.

    One call into ``test_reachability``, which is the same walk ``pr_blast``,
    ``get_change_risk`` and the health engine already answer this question
    with — so ``get_risk``'s two modes stop carrying two definitions of "a test
    reaches this file". ``import_depth=0`` switches off that module's own
    weaker tier: this module keeps its own, below, and running both would ask
    the same question twice with two different edge-type sets.

    Returns the uncapped list. ``ReachedBy.tests`` is already trimmed to
    ``MAX_TESTS_PER_TARGET`` and the caller needs the true total before its own
    cap, so ``all_tests`` is the one to read.

    Degrades to "no signal" rather than raising: a failed walk falls through to
    the import tier and, past that, to naming. It must never be the thing that
    turns a tested file into an accusation, and it must never invent one
    either.
    """
    from repowise.core.analysis.test_reachability import tests_reaching_by_tier

    try:
        found = await tests_reaching_by_tier(session, repo_id, [file_path], import_depth=0)
    except Exception:
        return []
    reached = found.get(file_path)
    if reached is None:
        return []
    return sorted(reached.all_tests or tuple(reached.tests))


async def _import_graph_tests(session: AsyncSession, repo_id: str, file_path: str) -> list[str]:
    """Test files that depend on *file_path* — the ``tested_by`` relation.

    Derived from the dependency edges rather than stored, because the
    knowledge-graph export derives it the same way (a dependency edge whose
    source is a test and whose target is not becomes ``tested_by``) and a
    second stored copy would be a second thing to keep in sync.

    The weaker of the two graph tiers, and reached only where the call walk
    said nothing. Kept rather than replaced by ``test_reachability``'s import
    walk because this query's edge set is the wider one — everything that is
    not containment or co-change, against that walk's
    ``FILE_DEPENDENCY_EDGE_TYPES`` — so keeping it is what makes the swap
    strictly additive.
    """
    res = await session.execute(
        select(GraphEdge.source_node_id)
        .join(
            GraphNode,
            (GraphNode.repository_id == GraphEdge.repository_id)
            & (GraphNode.node_id == GraphEdge.source_node_id),
        )
        .where(
            GraphEdge.repository_id == repo_id,
            GraphEdge.target_node_id == file_path,
            GraphEdge.edge_type.notin_(NON_DEPENDENCY_EDGE_TYPES),
            GraphNode.is_test.is_(True),
        )
        .distinct()
    )
    return sorted({row[0] for row in res.all() if row[0]})


async def _naming_index(
    session: AsyncSession, repo_id: str
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The two maps the naming rung needs, from one scan.

    Returns ``(tests_by_subject, sources_by_stem)``: every test file keyed by
    the source stem its name claims, and every non-test file keyed by its own
    stem. The rung needs both — one to find the candidate, the other to decide
    whether the candidate's name is ambiguous — and it used to read them with
    two separate full scans of ``graph_nodes`` per file, which is the cost that
    made the rung the module's most expensive one.

    Fetched whole and matched in Python: the mapping is one pass over the file
    nodes, while the equivalent SQL is a per-target ``OR`` of ``LIKE '%/name'``
    patterns, none of them indexable — 6.9 ms a call on the SoleMD.Infra mirror
    against 2.5 ms for this.

    The predicate is the union of what the two scans asked for, not a narrowing
    of it: test nodes are taken on ``is_test`` alone, because ingestion is free
    to classify a test that is not a ``file`` node and dropping it would make a
    guarded file read as unguarded.
    """
    res = await session.execute(
        select(GraphNode.node_id, GraphNode.is_test).where(
            GraphNode.repository_id == repo_id,
            or_(
                GraphNode.is_test.is_(True),
                and_(GraphNode.node_type == "file", GraphNode.is_test.is_not(True)),
            ),
        )
    )
    by_subject: dict[str, list[str]] = {}
    sources: dict[str, list[str]] = {}
    for node_id, is_test in res.all():
        if not node_id:
            continue
        if is_test:
            subject = subject_stem(node_id)
            if subject:
                by_subject.setdefault(subject.lower(), []).append(node_id)
            continue
        normalized = node_id.replace("\\", "/")
        stem = posixpath.splitext(posixpath.basename(normalized))[0]
        if stem:
            sources.setdefault(stem.lower(), []).append(normalized)
    return by_subject, sources


async def _named_tests(session: AsyncSession, repo_id: str, file_path: str) -> list[str]:
    """Test nodes whose filename names *file_path*'s stem as their subject.

    Matched on the whole stem, never as a substring: ``LIKE '%test_search%'``
    also matches ``tests/test_search_ranking.py`` in an unrelated package, and
    a false "this is tested" is the one answer that must not be guessed.
    """
    stem = posixpath.splitext(posixpath.basename(file_path.replace("\\", "/")))[0]
    if not stem:
        return []
    by_subject, sources_by_stem = await _naming_index(session, repo_id)
    tests = sorted(set(by_subject.get(stem.lower(), [])))
    if not tests:
        return []

    # A basename convention cannot attribute ``test_graph.py`` when several
    # unrelated source trees contain ``graph.py``. Suppress the candidate
    # rather than making a package-affinity guess: naming evidence is already
    # weaker than a graph/coverage edge, and a false guard is the costly error.
    normalized_target = file_path.replace("\\", "/")
    same_stem_sources = set(sources_by_stem.get(stem.lower(), ()))
    if any(source != normalized_target for source in same_stem_sources):
        return []
    return tests


async def _graph_tests(
    session: AsyncSession, repo_id: str, file_path: str
) -> tuple[list[str], str]:
    """The graph rung's answer and which tier gave it.

    Call walk first, import relation only where it stayed silent. Upstream's
    measurement is the reason for the order and for the fallback rather than a
    union: falling back answers one more target at the stronger tier's
    precision, unioning answers no more and costs 1.7 points of it.
    """
    tests = await _call_graph_tests(session, repo_id, file_path)
    if tests:
        return tests, "call-graph"
    return await _import_graph_tests(session, repo_id, file_path), "import-graph"


async def resolve_test_linkage(session: AsyncSession, repo_id: str, file_path: str) -> TestLinkage:
    """The single authoritative answer to "what tests guard *file_path*".

    Walks the evidence strongest-first and stops at the first rung that
    produces anything, so the reported ``basis`` is always the best evidence
    available rather than whichever check happened to run.
    """
    if not file_path:
        return TestLinkage(file_path=file_path)

    if await _is_test_node(session, repo_id, file_path):
        return TestLinkage(file_path=file_path, basis="self")

    def _found(tests: list[str], basis: str, via: str | None = None) -> TestLinkage:
        return TestLinkage(
            file_path=file_path,
            tests=tests[:MAX_TEST_LINKS],
            total=len(tests),
            basis=basis,
            via=via,
        )

    coverage = await _coverage_tests(session, repo_id, file_path)
    if coverage:
        return _found(coverage, "coverage")

    graph, via = await _graph_tests(session, repo_id, file_path)
    if graph:
        return _found(graph, "graph", via)

    named = await _named_tests(session, repo_id, file_path)
    if named:
        return _found(named, "naming")

    return TestLinkage(file_path=file_path)
