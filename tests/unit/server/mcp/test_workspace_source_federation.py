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
# Envelope strength: whose owner actually carries the subject
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_caution_tie_resolves_on_owner_subject_evidence_not_cosine(monkeypatch):
    """The measured SEO defect, with the evidence the probe actually returned.

    "Where are the SEO metadata endpoints for robots, sitemap, and web app
    manifest defined?" — web's ``src/app/robots.ts`` is the owner and carries
    robots/sitemap/manifest in itself; infra's ``semantic_file_scope.py`` is
    semantically nearby noise that carries none of them. Both envelopes are
    caution with no exact name, so the pre-fix key fell straight to dense
    cosine, where infra's embedding happened to sit closer — and infra's
    block led a question only web can answer.
    """
    registry = _StubRegistry(["infra", "web"], default="infra")
    infra = _StubCoordinator(
        _envelope(
            [("codeatlas/code_search/tools/semantic_file_scope.py", 0.71)],
            "caution",
            owner="codeatlas/code_search/tools/semantic_file_scope.py",
            cosine=0.4612,
            coverage=(0.3333, 0.0),
        )
    )
    web = _StubCoordinator(
        _envelope(
            [("src/app/robots.ts", 0.68)],
            "caution",
            owner="src/app/robots.ts",
            cosine=0.3874,
            coverage=(1.0, 0.6667),
        )
    )
    _wire(monkeypatch, {"infra": infra, "web": web})

    resp = await workspace_source_search(
        "Where are the SEO metadata endpoints for robots, sitemap, and web app "
        "manifest defined?",
        limit=10, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["_meta"]["source_search"]["repo_order"] == ["web", "infra"]
    assert resp["results"][0]["repo"] == "web"
    assert resp["selected_owner"]["file"] == "src/app/robots.ts"
    assert resp["selected_owner"]["repo"] == "web"


@pytest.mark.asyncio
async def test_an_owner_whose_case_is_its_path_loses_to_one_that_carries_it(monkeypatch):
    """A3's boundary, applied across corpora.

    Alpha's owner matches every concept in the query and carries none of them
    — the whole match is its filename. Beta's owner matches half and carries
    that half. Alpha wins on both permissive coverage and cosine, and still
    must not lead: a candidate whose entire case is its path may corroborate
    a subject, never constitute one.
    """
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    alpha = _StubCoordinator(
        _envelope(
            [("retry_queue_drain_worker.py", 0.9)],
            "caution",
            owner="retry_queue_drain_worker.py",
            cosine=0.88,
            coverage=(1.0, 0.0),
        )
    )
    beta = _StubCoordinator(
        _envelope(
            [("workers/pump.py", 0.9)],
            "caution",
            owner="workers/pump.py",
            cosine=0.42,
            coverage=(0.5, 0.5),
        )
    )
    _wire(monkeypatch, {"alpha": alpha, "beta": beta})

    resp = await workspace_source_search(
        "how does the retry queue drain?", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["_meta"]["source_search"]["repo_order"] == ["beta", "alpha"]
    assert resp["selected_owner"]["file"] == "workers/pump.py"


@pytest.mark.asyncio
async def test_coverage_outranks_cosine_once_both_owners_carry_the_subject(monkeypatch):
    """Both owners clear A3's content gate, so the question becomes how much
    of the subject each one covers — not which corpus the shared embedder
    placed nearer. Alpha holds the far better cosine and still loses."""
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    alpha = _StubCoordinator(
        _envelope([("a.py", 0.9)], "caution", owner="a.py", cosine=0.91, coverage=(0.4, 0.4))
    )
    beta = _StubCoordinator(
        _envelope([("b.py", 0.9)], "caution", owner="b.py", cosine=0.30, coverage=(0.9, 0.2))
    )
    _wire(monkeypatch, {"alpha": alpha, "beta": beta})

    resp = await workspace_source_search(
        "who owns it", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["_meta"]["source_search"]["repo_order"] == ["beta", "alpha"]


@pytest.mark.asyncio
async def test_coverage_is_read_from_the_owner_not_from_a_neighbour(monkeypatch):
    """Evidence from another candidate cannot make *owner* trustworthy — the
    coordinator's own rule, and it has to survive federation. Alpha's window
    is full of the subject while the file alpha nominated carries none of it;
    beta's nominee carries it. Alpha also holds the better cosine."""
    registry = _StubRegistry(["alpha", "beta"], default="alpha")
    alpha = _StubCoordinator(
        _envelope(
            [("neighbour.py", 0.9)],
            "caution",
            owner="named_for_the_topic.py",
            cosine=0.87,
            coverage=(1.0, 1.0),
            owner_coverage=(1.0, 0.0),
        )
    )
    beta = _StubCoordinator(
        _envelope([("b.py", 0.9)], "caution", owner="b.py", cosine=0.44, coverage=(0.6, 0.4))
    )
    _wire(monkeypatch, {"alpha": alpha, "beta": beta})

    resp = await workspace_source_search(
        "who owns it", limit=5, mode="concept", repo="all",
        registry=registry, build_meta=_META,
    )

    assert resp["_meta"]["source_search"]["repo_order"] == ["beta", "alpha"]
    assert resp["selected_owner"]["file"] == "b.py"


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
