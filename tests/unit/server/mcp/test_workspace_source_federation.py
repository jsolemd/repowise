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


def _envelope(
    files: list[tuple[str, float]],
    confidence: str,
    *,
    owner: str | None = None,
    cosine: float | None = None,
    exact: bool = False,
) -> dict[str, Any]:
    results = [
        {
            "file": f,
            "target_path": f,
            "relevance_score": score,
            "evidence": {"dense_cosine": cosine, "exact_name": exact, "lane": "source"},
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
                "evidence": {"dense_cosine": cosine, "exact_name": exact, "lane": "source"},
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
