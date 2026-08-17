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
  3. ``graph``    — these test files import it (the graph's ``tested_by``
                    relation, derived here from the import edges the
                    knowledge-graph export labels the same way).
  4. ``naming``   — no test references it, but a file named for it exists.
                    A convention, not evidence, and labelled as such.
  5. ``none``     — nothing found.

The basis travels with the answer, because "12 tests import this" and "a file
called test_foo.py exists somewhere" are not the same claim and an agent
deciding whether it can refactor safely needs to know which one it has.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.ingestion.models import NON_DEPENDENCY_EDGE_TYPES
from repowise.core.persistence.models import GraphEdge, GraphNode

#: Cap on the test files named in a payload. The count is always exact; the
#: list is a run-list, and a hundred paths is not one.
MAX_GUARDING_TESTS = 10


@dataclass(frozen=True)
class TestLinkage:
    """What guards *file_path*, and on what evidence."""

    file_path: str
    #: Test files that guard this one, sorted, capped at MAX_GUARDING_TESTS.
    tests: list[str] = field(default_factory=list)
    #: Total guarding tests found, before the cap.
    total: int = 0
    #: ``self`` | ``coverage`` | ``graph`` | ``naming`` | ``none``.
    basis: str = "none"

    @property
    def tested(self) -> bool:
        """Whether anything tests this file. ``test_gap`` is the negation."""
        return self.basis != "none"

    def as_payload(self) -> dict:
        """The additive block both tools attach to a file's card."""
        out: dict = {"tested": self.tested, "test_linkage_basis": self.basis}
        if self.tests:
            out["guarding_tests"] = self.tests
            out["guarding_test_count"] = self.total
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


async def _graph_tests(session: AsyncSession, repo_id: str, file_path: str) -> list[str]:
    """Test files that depend on *file_path* — the ``tested_by`` relation.

    Derived from the dependency edges rather than stored, because the
    knowledge-graph export derives it the same way (a dependency edge whose
    source is a test and whose target is not becomes ``tested_by``) and a
    second stored copy would be a second thing to keep in sync.
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


async def _tests_by_subject(session: AsyncSession, repo_id: str) -> dict[str, list[str]]:
    """Every test file in the repo, keyed by the source stem it names.

    Fetched whole and matched in Python: the mapping is one pass over the test
    nodes, while the equivalent SQL is a per-target ``OR`` of ``LIKE '%/name'``
    patterns, none of them indexable — 6.9 ms a call on the SoleMD.Infra mirror
    against 2.5 ms for this.
    """
    res = await session.execute(
        select(GraphNode.node_id).where(
            GraphNode.repository_id == repo_id,
            GraphNode.is_test.is_(True),
        )
    )
    by_subject: dict[str, list[str]] = {}
    for (node_id,) in res.all():
        if not node_id:
            continue
        subject = subject_stem(node_id)
        if subject:
            by_subject.setdefault(subject.lower(), []).append(node_id)
    return by_subject


async def _named_tests(session: AsyncSession, repo_id: str, file_path: str) -> list[str]:
    """Test nodes whose filename names *file_path*'s stem as their subject.

    Matched on the whole stem, never as a substring: ``LIKE '%test_search%'``
    also matches ``tests/test_search_ranking.py`` in an unrelated package, and
    a false "this is tested" is the one answer that must not be guessed.
    """
    stem = posixpath.splitext(posixpath.basename(file_path.replace("\\", "/")))[0]
    if not stem:
        return []
    return sorted(set((await _tests_by_subject(session, repo_id)).get(stem.lower(), [])))


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

    for basis, finder in (
        ("coverage", _coverage_tests),
        ("graph", _graph_tests),
        ("naming", _named_tests),
    ):
        tests = await finder(session, repo_id, file_path)
        if tests:
            return TestLinkage(
                file_path=file_path,
                tests=tests[:MAX_GUARDING_TESTS],
                total=len(tests),
                basis=basis,
            )

    return TestLinkage(file_path=file_path)
