"""The source-search lane in workspace mode: scoped, federated, honest.

The C3 federated probes measured three defects on the served surface: the
source lane bound to one repo path (inert at a workspace root, one corpus
answering every ``repo=`` at a member repo), federated wiki fusion collapsing
to workspace-config order, and candidate paths losing their repo. These pin
the replacements. Every test here fails on the pre-federation code — that is
the point of the exact orderings asserted.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from repowise.server.mcp_server._source_federation import workspace_source_search


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubCoordinator:
    def __init__(self, envelope: dict[str, Any] | Exception):
        self._envelope = envelope
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, *, limit: int, mode: str, base_meta: dict) -> dict:
        self.calls.append({"query": query, "limit": limit, "mode": mode})
        if isinstance(self._envelope, Exception):
            raise self._envelope
        # A fresh copy per call — the federation layer mutates rows in place.
        import copy

        return copy.deepcopy(self._envelope)


class _StubRegistry:
    """Mimics RepoRegistry.resolve_repo_param/get semantics for two repos."""

    def __init__(self, aliases: list[str], default: str):
        self._aliases = aliases
        self._default = default
        self.contexts = {
            a: SimpleNamespace(alias=a, path=f"/ws/{a}", vector_store=object())
            for a in aliases
        }

    def get_all_aliases(self) -> list[str]:
        return list(self._aliases)

    def get_default_alias(self) -> str:
        return self._default

    def resolve_repo_param(self, repo):
        if repo is None:
            return self._default
        if repo == "all":
            return self.get_all_aliases()
        if repo not in self._aliases:
            raise ValueError(f"Unknown repo '{repo}'")
        return repo

    async def get(self, alias: str):
        return self.contexts[alias]


def _evidence(
    cosine: float | None,
    exact: bool,
    coverage: tuple[float, float] | None,
) -> dict[str, Any]:
    """One evidence dict in the shape the coordinator serves it.

    ``coverage`` is A3's ``(concept_coverage, content_concept_coverage)`` pair.
    Omitting it omits both keys, which is the real shape of a response whose
    co-location evidence never arrived — not a pair of zeros.
    """
    evidence: dict[str, Any] = {"dense_cosine": cosine, "exact_name": exact, "lane": "source"}
    if coverage is not None:
        evidence["concept_coverage"] = coverage[0]
        evidence["content_concept_coverage"] = coverage[1]
    return evidence


def _envelope(
    files: list[tuple[str, float]],
    confidence: str,
    *,
    owner: str | None = None,
    cosine: float | None = None,
    exact: bool = False,
    coverage: tuple[float, float] | None = None,
    owner_coverage: tuple[float, float] | None = None,
) -> dict[str, Any]:
    results = [
        {
            "file": f,
            "target_path": f,
            "relevance_score": score,
            "evidence": _evidence(cosine, exact, coverage),
        }
        for f, score in files
    ]
    env: dict[str, Any] = {
        "results": results,
        "mode": "concept",
        "confidence": confidence,
        "selected_owner": (
            {
                "file": owner,
                "reason": "top evidence",
                "evidence": _evidence(
                    cosine, exact, coverage if owner_coverage is None else owner_coverage
                ),
            }
            if owner
            else None
        ),
        "_meta": {"source_search": {"generation_id": f"gen-{confidence}"}},
        "candidates": [{"path": f} for f, _ in files],
    }
    return env


def _wire(monkeypatch, coordinators: dict[str, _StubCoordinator | None]) -> None:
    async def fake_context_coordinator(ctx):
        return coordinators.get(ctx.alias)

    monkeypatch.setattr(
        "repowise.server.source_search_wiring.context_coordinator",
        fake_context_coordinator,
    )


_META = lambda: {"base": True}  # noqa: E731


# ---------------------------------------------------------------------------
# Scoped calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoped_call_routes_to_that_repos_coordinator(monkeypatch):
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    alpha = _StubCoordinator(_envelope([("a.py", 1.0)], "confident", owner="a.py"))
    beta = _StubCoordinator(_envelope([("b.py", 1.0)], "confident", owner="b.py"))
    _wire(monkeypatch, {"alpha": alpha, "beta": beta})

    resp = await workspace_source_search(
        "where is b", limit=5, mode="concept", repo="beta",
        registry=registry, build_meta=_META,
    )

    assert beta.calls and not alpha.calls
    assert all(row["repo"] == "beta" for row in resp["results"])
    assert all(c["repo"] == "beta" for c in resp["candidates"])
    assert resp["selected_owner"]["repo"] == "beta"
    assert resp["_meta"]["source_search"]["answered_from"] == "beta"


@pytest.mark.asyncio
async def test_repo_omitted_scopes_to_the_default_repo(monkeypatch):
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    alpha = _StubCoordinator(_envelope([("a.py", 1.0)], "caution"))
    beta = _StubCoordinator(_envelope([("b.py", 1.0)], "confident", owner="b.py"))
    _wire(monkeypatch, {"alpha": alpha, "beta": beta})

    resp = await workspace_source_search(
        "anything", limit=5, mode="concept", repo=None,
        registry=registry, build_meta=_META,
    )

    assert alpha.calls and not beta.calls
    assert resp["_meta"]["source_search"]["answered_from"] == "alpha"


@pytest.mark.asyncio
async def test_scoped_call_without_an_index_falls_through(monkeypatch):
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    _wire(monkeypatch, {"alpha": None, "beta": None})

    resp = await workspace_source_search(
        "anything", limit=5, mode="concept", repo="alpha",
        registry=registry, build_meta=_META,
    )
    assert resp is None


# ---------------------------------------------------------------------------
# Federated calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_federated_winner_block_leads_regardless_of_config_order(monkeypatch):
    # alpha is first in the workspace config; beta has the stronger envelope.
    # Config-order fusion put alpha first here — that is the measured defect.
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    alpha = _StubCoordinator(_envelope([("a1.py", 0.9), ("a2.py", 0.8)], "caution", cosine=0.5))
    beta = _StubCoordinator(
        _envelope([("b1.py", 0.9), ("b2.py", 0.8)], "confident", owner="b1.py", cosine=0.9)
    )
    _wire(monkeypatch, {"alpha": alpha, "beta": beta})

    resp = await workspace_source_search(
        "who owns it", limit=4, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert [r["repo"] for r in resp["results"]] == ["beta", "beta", "alpha", "alpha"]
    assert resp["confidence"] == "confident"
    assert resp["selected_owner"] == {
        "file": "b1.py",
        "reason": "top evidence",
        "evidence": {"dense_cosine": 0.9, "exact_name": False, "lane": "source"},
        "repo": "beta",
    }
    assert resp["_meta"]["source_search"]["repo_order"] == ["beta", "alpha"]
    assert resp["_meta"]["source_search"]["federated"] is True


@pytest.mark.asyncio
async def test_two_confident_repos_with_different_owners_demote_to_caution(monkeypatch):
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    alpha = _StubCoordinator(
        _envelope([("lib/x.ts", 1.0)], "confident", owner="lib/x.ts", cosine=0.91)
    )
    beta = _StubCoordinator(
        _envelope([("lib/x.ts", 1.0)], "confident", owner="lib/x.ts", cosine=0.90)
    )
    _wire(monkeypatch, {"alpha": alpha, "beta": beta})

    resp = await workspace_source_search(
        "the x helper", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["confidence"] == "caution"
    competing = {(c["repo"], c["file"]) for c in resp["competing_owners"]}
    assert competing == {("alpha", "lib/x.ts"), ("beta", "lib/x.ts")}
    # The winner's owner is still named — demotion is uncertainty, not amnesia.
    assert resp["selected_owner"]["repo"] == "alpha"
    assert "note" in resp
    # Identical relative paths in two repos are two candidates, each tagged.
    cands = {(c["repo"], c["path"]) for c in resp["candidates"]}
    assert cands == {("alpha", "lib/x.ts"), ("beta", "lib/x.ts")}


@pytest.mark.asyncio
async def test_federation_never_lifts_confidence(monkeypatch):
    registry = _StubRegistry(["alpha", "beta"], default="alpha")

    # All caution stays caution (absent-subject safety must survive federation).
    _wire(monkeypatch, {
        "alpha": _StubCoordinator(_envelope([("a.py", 0.5)], "caution")),
        "beta": _StubCoordinator(_envelope([("b.py", 0.4)], "caution")),
    })
    resp = await workspace_source_search(
        "no such subsystem", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )
    assert resp["confidence"] == "caution"

    # All no_match stays no_match, and says so.
    _wire(monkeypatch, {
        "alpha": _StubCoordinator(_envelope([], "no_match")),
        "beta": _StubCoordinator(_envelope([], "no_match")),
    })
    resp = await workspace_source_search(
        "no such subsystem", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )
    assert resp["confidence"] == "no_match"
    assert "note" in resp


@pytest.mark.asyncio
async def test_missing_index_is_disclosed_and_the_rest_still_answer(monkeypatch):
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    beta = _StubCoordinator(_envelope([("b.py", 1.0)], "confident", owner="b.py"))
    _wire(monkeypatch, {"alpha": None, "beta": beta})

    resp = await workspace_source_search(
        "who owns it", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp is not None
    assert resp["_meta"]["source_search"]["repos"]["alpha"] == {
        "unavailable": "source index unavailable"
    }
    assert [r["repo"] for r in resp["results"]] == ["beta"]


@pytest.mark.asyncio
async def test_a_failing_leg_is_disclosed_not_fatal(monkeypatch):
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    alpha = _StubCoordinator(RuntimeError("lance exploded"))
    beta = _StubCoordinator(_envelope([("b.py", 1.0)], "caution"))
    _wire(monkeypatch, {"alpha": alpha, "beta": beta})

    resp = await workspace_source_search(
        "who owns it", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp is not None
    assert resp["_meta"]["source_search"]["repos"]["alpha"] == {
        "unavailable": "search failed: RuntimeError"
    }


@pytest.mark.asyncio
async def test_every_coordinator_missing_falls_through(monkeypatch):
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    _wire(monkeypatch, {"alpha": None, "beta": None})

    resp = await workspace_source_search(
        "who owns it", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )
    assert resp is None


# ---------------------------------------------------------------------------
# Envelope strength: which repo's owner is named for the subject
#
# Every envelope below carries the evidence the live workspace actually
# returned for that query (G4 federated probe set, pulled per repo from the
# scoped calls). The numbers are measurements, not illustrations, which is why
# they are odd.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_named_owner_takes_the_answer_from_a_confident_term_sponge(monkeypatch):
    """The archetype bleed: "docx template engine".

    infra's ``boost_make.py`` is a frozenset of authoring words that literally
    contains docx/template/engine, so it matches every concept in its body and
    its corpus honestly calls it confident. Make holds the only docx engine in
    the workspace, and ``docx_template.py`` matches the same subject through
    its *path* — coverage 0.7105 with none of it content-carried.

    Ranking confidence first hands a Make question to infra. The federated
    answer is Make's, and it is ``caution`` — composing down from infra's
    confident, never up from Make's caution.
    """
    registry = _StubRegistry(["infra", "make"], default="infra")
    infra = _StubCoordinator(
        _envelope(
            [("codeatlas/code_search/boost_make.py", 0.01639)],
            "confident",
            owner="codeatlas/code_search/boost_make.py",
            cosine=0.4326,
            coverage=(1.0, 1.0),
        )
    )
    make = _StubCoordinator(
        _envelope(
            [("src/make/engines/docx_template.py", 0.0161)],
            "caution",
            owner="src/make/engines/docx_template.py",
            cosine=0.5736,
            coverage=(0.7105, 0.0),
        )
    )
    _wire(monkeypatch, {"infra": infra, "make": make})

    resp = await workspace_source_search(
        "docx template engine", limit=10, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["_meta"]["source_search"]["repo_order"] == ["make", "infra"]
    assert resp["selected_owner"]["file"] == "src/make/engines/docx_template.py"
    assert resp["selected_owner"]["repo"] == "make"
    # Federation composes downward: the winner's own class, never infra's.
    assert resp["confidence"] == "caution"


@pytest.mark.asyncio
async def test_a_partly_named_owner_still_declares_the_subject(monkeypatch):
    """"health check endpoint" — graph's ``routes/health.py`` carries the
    subject partly in its path (coverage 0.6412, content 0.2507) against
    infra's confident ``transport/handlers.py`` at a saturated 1.0/1.0. The
    declaration is about the split between path and body, not about the path
    carrying everything."""
    registry = _StubRegistry(["graph", "infra"], default="infra")
    infra = _StubCoordinator(
        _envelope(
            [("codeatlas/code_search/transport/handlers.py", 0.02)],
            "confident",
            owner="codeatlas/code_search/transport/handlers.py",
            cosine=0.5287,
            coverage=(1.0, 1.0),
        )
    )
    graph = _StubCoordinator(
        _envelope(
            [("apps/api/app/routes/health.py", 0.02)],
            "caution",
            owner="apps/api/app/routes/health.py",
            cosine=0.5034,
            coverage=(0.6412, 0.2507),
        )
    )
    _wire(monkeypatch, {"graph": graph, "infra": infra})

    resp = await workspace_source_search(
        "health check endpoint", limit=10, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["_meta"]["source_search"]["repo_order"] == ["graph", "infra"]
    assert resp["selected_owner"]["file"] == "apps/api/app/routes/health.py"


@pytest.mark.asyncio
async def test_one_query_word_in_a_path_is_not_a_declaration(monkeypatch):
    """The false positive the coverage floor exists to stop.

    "dramatiq broker wired to Redis with retry middleware and shutdown
    notifications" is an eight-concept query. infra's
    ``neo4j/writer_retry.py`` matches exactly one of them, "retry", through
    its path — coverage 0.0724, all of it path-carried. Graph's ``broker.py``
    is the real owner at coverage 0.7641. A bare "the path carries something"
    test hands this to infra; the engine's own no-match floor is what refuses
    to call one incidental token a declaration.
    """
    registry = _StubRegistry(["graph", "infra"], default="graph")
    infra = _StubCoordinator(
        _envelope(
            [("codeatlas/code_search/neo4j/writer_retry.py", 0.01)],
            "caution",
            owner="codeatlas/code_search/neo4j/writer_retry.py",
            cosine=0.4331,
            coverage=(0.0724, 0.0),
        )
    )
    graph = _StubCoordinator(
        _envelope(
            [("apps/worker/app/broker.py", 0.02)],
            "caution",
            owner="apps/worker/app/broker.py",
            cosine=0.6035,
            coverage=(0.7641, 0.7641),
        )
    )
    _wire(monkeypatch, {"graph": graph, "infra": infra})

    resp = await workspace_source_search(
        "dramatiq broker wired to Redis with retry middleware and shutdown "
        "notifications",
        limit=10, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["_meta"]["source_search"]["repo_order"] == ["graph", "infra"]
    assert resp["selected_owner"]["file"] == "apps/worker/app/broker.py"


@pytest.mark.asyncio
async def test_the_inverse_control_a_repo_full_of_another_repos_words(monkeypatch):
    """The inverse control: infra's ``boost_graph_ui.py`` is a helper whose
    *content* is web-authoring vocabulary, and it is the correct owner of a
    question about that helper. Graph's ``SearchSection.tsx`` picks up "search"
    in its path at coverage 0.2173. Any rule that simply prefers the repo whose
    path matches loses this case; the answer stays with infra on cosine."""
    registry = _StubRegistry(["graph", "infra"], default="infra")
    infra = _StubCoordinator(
        _envelope(
            [("codeatlas/code_search/boost_graph_ui.py", 0.02)],
            "caution",
            owner="codeatlas/code_search/boost_graph_ui.py",
            cosine=0.5909,
            coverage=(0.6696, 0.6696),
        )
    )
    graph = _StubCoordinator(
        _envelope(
            [("apps/web/features/graph/components/explore/info/SearchSection.tsx", 0.01)],
            "caution",
            owner="apps/web/features/graph/components/explore/info/SearchSection.tsx",
            cosine=0.4235,
            coverage=(0.2173, 0.1617),
        )
    )
    _wire(monkeypatch, {"graph": graph, "infra": infra})

    resp = await workspace_source_search(
        "helper deciding whether a search query is asking about web UI "
        "authoring from anchor terms like hero, cta and card",
        limit=10, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["_meta"]["source_search"]["repo_order"] == ["infra", "graph"]
    assert resp["selected_owner"]["file"] == "codeatlas/code_search/boost_graph_ui.py"


@pytest.mark.asyncio
async def test_declaration_is_read_from_the_owner_not_from_a_neighbour(monkeypatch):
    """Evidence from another candidate cannot make *owner* trustworthy — the
    coordinator's own rule, and it has to survive federation. Alpha's top row
    is named for the subject while the file alpha actually nominated is not;
    beta's nominee is. Alpha also holds the better cosine."""
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    alpha = _StubCoordinator(
        _envelope(
            [("alpha/retrieval_profile.py", 0.9)],
            "caution",
            owner="alpha/server.py",
            cosine=0.87,
            coverage=(0.9, 0.0),
            owner_coverage=(0.9, 0.9),
        )
    )
    beta = _StubCoordinator(
        _envelope(
            [("beta/retrieval_profile.py", 0.9)],
            "caution",
            owner="beta/retrieval_profile.py",
            cosine=0.44,
            coverage=(0.9, 0.0),
        )
    )
    _wire(monkeypatch, {"alpha": alpha, "beta": beta})

    resp = await workspace_source_search(
        "retrieval profile", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["_meta"]["source_search"]["repo_order"] == ["beta", "alpha"]
    assert resp["selected_owner"]["file"] == "beta/retrieval_profile.py"


@pytest.mark.asyncio
async def test_envelopes_without_coverage_evidence_rank_exactly_as_before(monkeypatch):
    """The degradation guard. An evidence dict with no coverage keys is the
    real shape of a response whose co-location evidence never arrived, and
    with no repo declaring anything the order must fall back to confidence
    then cosine — what it was before the declaration test existed."""
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    _wire(monkeypatch, {
        "alpha": _StubCoordinator(
            _envelope([("a.py", 0.9)], "caution", owner="a.py", cosine=0.41)
        ),
        "beta": _StubCoordinator(
            _envelope([("b.py", 0.9)], "caution", owner="b.py", cosine=0.83)
        ),
    })

    resp = await workspace_source_search(
        "who owns it", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )
    assert resp["_meta"]["source_search"]["repo_order"] == ["beta", "alpha"]


# ---------------------------------------------------------------------------
# The federated window: no answering repo is starved out of sight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_winner_block_does_not_fill_the_whole_window(monkeypatch):
    """The second measured defect on the SEO query: limit=10 and all ten rows
    came from one repo, so the other repo's correct owner was never in the
    window at all. The winner keeps the head and its order; the loser is owed
    its top row, and that row is the one it ranked first."""
    registry = _StubRegistry(["infra", "web"], default="infra")
    web = _StubCoordinator(
        _envelope(
            [(f"src/app/w{i}.ts", 1.0 - i / 100) for i in range(10)],
            "caution",
            owner="src/app/w0.ts",
            cosine=0.39,
            coverage=(1.0, 0.67),
        )
    )
    infra = _StubCoordinator(
        _envelope(
            [(f"codeatlas/i{i}.py", 1.0 - i / 100) for i in range(10)],
            "caution",
            owner="codeatlas/i0.py",
            cosine=0.46,
            coverage=(0.33, 0.0),
        )
    )
    _wire(monkeypatch, {"infra": infra, "web": web})

    resp = await workspace_source_search(
        "seo metadata endpoints", limit=10, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert [r["repo"] for r in resp["results"]] == ["web"] * 9 + ["infra"]
    assert [r["file"] for r in resp["results"][:9]] == [f"src/app/w{i}.ts" for i in range(9)]
    # The reserved slot holds infra's own top row, not whatever fell off last.
    assert resp["results"][-1]["file"] == "codeatlas/i0.py"


@pytest.mark.asyncio
async def test_every_answering_repo_is_seated_when_the_window_can_hold_them(monkeypatch):
    """The bound: ``limit`` at least the number of answering repos means every
    answering repo has at least its top row in the window."""
    registry = _StubRegistry(["alpha", "beta", "gamma"], default="alpha")
    _wire(monkeypatch, {
        "alpha": _StubCoordinator(
            _envelope([(f"a{i}.py", 0.9) for i in range(5)], "confident", owner="a0.py",
                      cosine=0.8, coverage=(1.0, 0.9))
        ),
        "beta": _StubCoordinator(
            _envelope([(f"b{i}.py", 0.9) for i in range(5)], "caution", cosine=0.5,
                      coverage=(0.5, 0.3))
        ),
        "gamma": _StubCoordinator(
            _envelope([(f"g{i}.py", 0.9) for i in range(5)], "caution", cosine=0.4,
                      coverage=(0.4, 0.2))
        ),
    })

    resp = await workspace_source_search(
        "who owns it", limit=3, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert [(r["repo"], r["file"]) for r in resp["results"]] == [
        ("alpha", "a0.py"), ("beta", "b0.py"), ("gamma", "g0.py")
    ]


@pytest.mark.asyncio
async def test_a_window_too_small_for_everyone_pays_the_strongest_first(monkeypatch):
    """Below the bound the window cannot seat every repo, so it says who gives
    way: the winner never loses its top row, and the remaining slots go in
    envelope order — gamma, the weakest envelope, is the one left out."""
    registry = _StubRegistry(["alpha", "beta", "gamma"], default="alpha")
    _wire(monkeypatch, {
        "alpha": _StubCoordinator(
            _envelope([(f"a{i}.py", 0.9) for i in range(4)], "confident", owner="a0.py",
                      cosine=0.8, coverage=(1.0, 0.9))
        ),
        "beta": _StubCoordinator(
            _envelope([(f"b{i}.py", 0.9) for i in range(4)], "caution", cosine=0.5,
                      coverage=(0.5, 0.3))
        ),
        "gamma": _StubCoordinator(
            _envelope([(f"g{i}.py", 0.9) for i in range(4)], "caution", cosine=0.4,
                      coverage=(0.4, 0.2))
        ),
    })

    resp = await workspace_source_search(
        "who owns it", limit=2, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert [(r["repo"], r["file"]) for r in resp["results"]] == [
        ("alpha", "a0.py"), ("beta", "b0.py")
    ]


@pytest.mark.asyncio
async def test_a_repo_with_no_rows_is_owed_no_slot(monkeypatch):
    """A reserved slot is for an answering repo. A repo that returned nothing
    is not owed one, and must not cost the winner a row to hold it open."""
    registry = _StubRegistry(["alpha", "beta", "gamma"], default="alpha")
    _wire(monkeypatch, {
        "alpha": _StubCoordinator(
            _envelope([(f"a{i}.py", 0.9) for i in range(5)], "confident", owner="a0.py",
                      cosine=0.8, coverage=(1.0, 0.9))
        ),
        "beta": _StubCoordinator(_envelope([], "no_match")),
        "gamma": _StubCoordinator(
            _envelope([(f"g{i}.py", 0.9) for i in range(3)], "caution", cosine=0.4,
                      coverage=(0.4, 0.2))
        ),
    })

    resp = await workspace_source_search(
        "who owns it", limit=3, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert [(r["repo"], r["file"]) for r in resp["results"]] == [
        ("alpha", "a0.py"), ("alpha", "a1.py"), ("gamma", "g0.py")
    ]


@pytest.mark.asyncio
async def test_a_reserved_row_already_in_the_head_is_not_seated_twice(monkeypatch):
    """The winner is short, so the head spills into the runner-up's block and
    seats the very row that block was owed. That reservation is already paid:
    it must not be charged again, and the freed slot goes back to the head."""
    registry = _StubRegistry(["alpha", "beta", "gamma"], default="alpha")
    _wire(monkeypatch, {
        "alpha": _StubCoordinator(
            _envelope([("a0.py", 0.9)], "confident", owner="a0.py", cosine=0.8,
                      coverage=(1.0, 0.9))
        ),
        "beta": _StubCoordinator(
            _envelope([(f"b{i}.py", 0.9) for i in range(6)], "caution", cosine=0.5,
                      coverage=(0.5, 0.3))
        ),
        "gamma": _StubCoordinator(
            _envelope([(f"g{i}.py", 0.9) for i in range(3)], "caution", cosine=0.4,
                      coverage=(0.4, 0.2))
        ),
    })

    resp = await workspace_source_search(
        "who owns it", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert [(r["repo"], r["file"]) for r in resp["results"]] == [
        ("alpha", "a0.py"),
        ("beta", "b0.py"), ("beta", "b1.py"), ("beta", "b2.py"),
        ("gamma", "g0.py"),
    ]


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_mode_routes_source_search_through_federation(monkeypatch):
    from repowise.server.mcp_server import _state, tool_search

    sentinel = {"results": [], "confidence": "caution", "_meta": {}}

    async def fake_ws(query, **kwargs):
        return sentinel

    async def fail_singleton():
        raise AssertionError("workspace mode must never touch the singleton")

    monkeypatch.setattr(tool_search, "source_search_enabled", lambda: True)
    monkeypatch.setattr(tool_search, "mcp_coordinator", fail_singleton)
    monkeypatch.setattr(
        "repowise.server.mcp_server._source_federation.workspace_source_search", fake_ws
    )
    monkeypatch.setattr(_state, "_registry", _StubRegistry(["alpha"], default="alpha"))

    result = await tool_search.search_codebase("how does the retry queue drain?")
    assert result is sentinel


@pytest.mark.asyncio
async def test_single_repo_mode_still_uses_the_singleton(monkeypatch):
    from repowise.server.mcp_server import _state, tool_search

    sentinel = {"results": [], "confidence": "caution", "_meta": {}}

    class _Coord:
        async def search(self, query, *, limit, mode, base_meta):
            return sentinel

    async def fake_singleton():
        return _Coord()

    def fail_ws(*a, **k):
        raise AssertionError("single-repo mode must never route through federation")

    monkeypatch.setattr(tool_search, "source_search_enabled", lambda: True)
    monkeypatch.setattr(tool_search, "mcp_coordinator", fake_singleton)
    monkeypatch.setattr(
        "repowise.server.mcp_server._source_federation.workspace_source_search", fail_ws
    )
    monkeypatch.setattr(_state, "_registry", None)
    monkeypatch.setattr(tool_search, "_build_meta", lambda **k: {})

    result = await tool_search.search_codebase("how does the retry queue drain?")
    assert result is sentinel


# ---------------------------------------------------------------------------
# Wiki-lane federated fusion (the fallback path when the source lane is off)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wiki_federated_merge_ranks_by_relevance_not_config_order(monkeypatch):
    from repowise.server.mcp_server import tool_search

    contexts = [SimpleNamespace(alias="alpha"), SimpleNamespace(alias="beta")]

    canned = {
        "alpha": [
            {"page_id": "file_page:a_weak.py", "page_type": "file_page",
             "target_path": "a_weak.py", "relevance_score": 0.2},
        ],
        "beta": [
            {"page_id": "file_page:b_strong.py", "page_type": "file_page",
             "target_path": "b_strong.py", "relevance_score": 0.9},
        ],
    }

    async def fake_all_contexts():
        return contexts

    async def fake_single(ctx, query, limit, page_type, kind=None):
        import copy

        return copy.deepcopy(canned[ctx.alias])

    monkeypatch.setattr(tool_search, "_resolve_all_contexts", fake_all_contexts)
    monkeypatch.setattr(tool_search, "_search_single_repo", fake_single)
    monkeypatch.setattr(tool_search, "_build_meta", lambda **k: {})

    resp = await tool_search._federated_search("q", 5, None)
    ordered = [(r["repo"], r["target_path"]) for r in resp["results"]]
    assert ordered == [("beta", "b_strong.py"), ("alpha", "a_weak.py")]


@pytest.mark.asyncio
async def test_wiki_federated_rrf_tie_breaks_on_raw_evidence(monkeypatch):
    from repowise.server.mcp_server import tool_search

    contexts = [SimpleNamespace(alias="alpha"), SimpleNamespace(alias="beta")]

    # Equal fused scores — the structural cross-repo tie. Beta's hit carries
    # the stronger dense cosine, so beta must lead; config-order fusion (or a
    # bare stable sort) puts alpha first.
    canned = {
        "alpha": [
            {"page_id": "file_page:twin.py", "page_type": "file_page",
             "target_path": "twin.py", "relevance_score": 0.5, "_best_cosine": 0.41},
        ],
        "beta": [
            {"page_id": "file_page:twin.py", "page_type": "file_page",
             "target_path": "twin.py", "relevance_score": 0.5, "_best_cosine": 0.83},
        ],
    }

    async def fake_all_contexts():
        return contexts

    async def fake_single(ctx, query, limit, page_type, kind=None):
        import copy

        return copy.deepcopy(canned[ctx.alias])

    monkeypatch.setattr(tool_search, "_resolve_all_contexts", fake_all_contexts)
    monkeypatch.setattr(tool_search, "_search_single_repo", fake_single)
    monkeypatch.setattr(tool_search, "_build_meta", lambda **k: {})

    resp = await tool_search._federated_search("q", 5, None)
    assert [r["repo"] for r in resp["results"]] == ["beta", "alpha"]
    # The raw signals are ranking internals, not response surface.
    assert all("_best_cosine" not in r for r in resp["results"])
    # Identical relative paths in two repos are two candidates, each tagged.
    cands = {(c["repo"], c["path"]) for c in resp["candidates"]}
    assert cands == {("alpha", "twin.py"), ("beta", "twin.py")}


def test_file_candidates_keeps_both_repos_twins():
    from repowise.server.mcp_server._page_paths import file_candidates

    hits = [
        {"page_type": "file_page", "target_path": "package.json", "repo": "alpha"},
        {"page_type": "file_page", "target_path": "package.json", "repo": "beta"},
    ]
    assert file_candidates(hits, limit=5) == [
        {"path": "package.json", "repo": "alpha"},
        {"path": "package.json", "repo": "beta"},
    ]


def test_file_candidates_single_repo_shape_unchanged():
    from repowise.server.mcp_server._page_paths import file_candidates

    hits = [
        {"page_type": "file_page", "target_path": "a.py"},
        {"page_type": "file_page", "target_path": "a.py"},
        {"page_type": "file_page", "target_path": "b.py"},
    ]
    assert file_candidates(hits, limit=5) == [{"path": "a.py"}, {"path": "b.py"}]
