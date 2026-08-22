"""Cross-repo merges must carry the per-repo noise partition.

Every list a workspace merge receives has already been ranked by
``_search_single_repo``, which ends in ``_sort_demoting_noise`` — a *class*
partition that puts decision records and test pages below every real page for a
plain implementation query, precisely because the scores those two classes win
beat the multiplicative down-weights that precede it. A merge that re-orders
those lists on score alone therefore undoes the demotion, once per repo.

Three merges do it, and this file holds all three to the same rule: the wiki
federation (``_federated_search``), the hybrid lane's cross-repo concept
re-sort (``_structured_search``), and the CLI's ``--all`` fan-out.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from repowise.server.mcp_server import tool_search


def _page(path: str, score: float, page_type: str = "file_page") -> dict:
    return {
        "page_id": f"{page_type}:{path}",
        "title": path,
        "page_type": page_type,
        "target_path": path,
        "relevance_score": score,
        "snippet": "",
    }


# The shape ``_search_single_repo`` returns for a plain implementation query:
# the real page first at a LOWER score than the noise ranked beneath it.
_DEMOTED = {
    "alpha": [_page("alpha/impl.py", 4.0), _page("alpha/why.md", 9.5, "decision_record")],
    "beta": [_page("beta/impl.py", 3.0), _page("beta/tests/test_impl.py", 8.5)],
}

_QUERY = "how does the queue drain"


def _fake_repos(monkeypatch, canned: dict[str, list[dict]], *, stub_meta: bool = True):
    """Stand in for the per-repo searches with lists already partitioned."""
    contexts = [
        SimpleNamespace(
            alias=alias,
            path=f"/ws/{alias}",
            session_factory=alias,
            fts=object(),
            vector_store=object(),
            vector_store_ready=None,
        )
        for alias in canned
    ]

    async def all_contexts():
        return contexts

    async def one_repo(ctx, query, limit, page_type, kind=None):
        return [dict(row) for row in canned[ctx.alias]]

    monkeypatch.setattr(tool_search, "_resolve_all_contexts", all_contexts)
    monkeypatch.setattr(tool_search, "_search_single_repo", one_repo)
    if stub_meta:
        monkeypatch.setattr(tool_search, "_build_meta", lambda **kwargs: {})
    # No repo rows by default: the ranking tests are about order, and a real
    # freshness lookup would need a Repository per context to say anything.
    # ``raising=False`` so this file still collects against a build that has no
    # freshness lookup at all — the ranking tests are then answering for
    # themselves rather than erroring on a missing name.
    monkeypatch.setattr(tool_search, "_federated_freshness", _no_freshness, raising=False)
    return contexts


async def _no_freshness(contexts, output):
    return {}


async def test_the_federated_merge_keeps_every_repo_s_noise_demotion(monkeypatch):
    _fake_repos(monkeypatch, _DEMOTED)

    response = await tool_search._federated_search(_QUERY, 5, None)
    paths = [row["target_path"] for row in response["results"]]

    assert paths[:2] == ["alpha/impl.py", "beta/impl.py"]
    assert set(paths[2:]) == {"alpha/why.md", "beta/tests/test_impl.py"}


async def test_the_federated_merge_still_ranks_real_pages_by_relevance(monkeypatch):
    """The partition leads; inside a class the score still decides."""
    _fake_repos(
        monkeypatch,
        {
            "alpha": [_page("alpha/weak.py", 1.0)],
            "beta": [_page("beta/strong.py", 9.0)],
        },
    )

    response = await tool_search._federated_search(_QUERY, 5, None)

    assert [row["target_path"] for row in response["results"]] == [
        "beta/strong.py",
        "alpha/weak.py",
    ]


async def test_a_why_shaped_query_still_ranks_decisions_across_repos(monkeypatch):
    """The classifier is per query, and the merge must use the caller's."""
    _fake_repos(monkeypatch, _DEMOTED)

    response = await tool_search._federated_search("why does the queue drain", 5, None)

    assert response["results"][0]["target_path"] == "alpha/why.md"


# ---------------------------------------------------------------------------
# The hybrid lane re-sorts concept pages across repos too
# ---------------------------------------------------------------------------


async def test_the_hybrid_concept_resort_keeps_the_noise_demotion(monkeypatch):
    contexts = _fake_repos(monkeypatch, _DEMOTED)

    async def contexts_for(repo):
        return contexts

    async def no_symbols(ctx, query, limit, symbol_kind=None, kind=None):
        return []

    monkeypatch.setattr(tool_search, "_contexts_for", contexts_for)
    monkeypatch.setattr(tool_search, "search_symbols_single", no_symbols)

    response = await tool_search._structured_search(
        _QUERY, 5, None, None, None, "all", "hybrid", None
    )
    paths = [row["target_path"] for row in response["results"]]

    assert paths[:2] == ["alpha/impl.py", "beta/impl.py"]


# ---------------------------------------------------------------------------
# H2 — ``repo`` is a workspace argument
# ---------------------------------------------------------------------------


def _single_repo_server(monkeypatch):
    """A single-repo server (no registry) with a two-hit corpus."""

    class _Hit:
        def __init__(self, page_id, score, page_type="file_page"):
            self.page_id = page_id
            self.title = page_id
            self.page_type = page_type
            self.snippet = "snippet"
            self.score = score

    async def fake_fts(ctx, query, limit):
        return [_Hit("file_page:src/queue.py", 9.0), _Hit("decision:d", 8.0, "decision_record")]

    async def fake_vector(ctx, query, limit):
        return [_Hit("file_page:src/queue.py", 0.81), _Hit("file_page:src/drain.py", 0.77)]

    async def fake_wait(ctx):
        return None

    async def fake_load(session, output, with_git=False):
        return {row["page_id"]: row["page_id"].split(":", 1)[-1] for row in output}, set(), {}

    async def fake_get_repo(session, repo=None):
        return SimpleNamespace(name="r", indexed_commit="abc", last_indexed_at=None)

    @contextlib.asynccontextmanager
    async def fake_session(session_factory):
        yield object()

    ctx = SimpleNamespace(
        alias="default",
        path="/repo",
        session_factory=None,
        fts=object(),
        vector_store=object(),
        vector_store_ready=None,
    )

    async def all_contexts():
        return [ctx]

    async def one_context(repo=None):
        return ctx

    for name, value in (
        ("_safe_fts", fake_fts),
        ("_safe_vector", fake_vector),
        ("_wait_for_vector_store", fake_wait),
        ("_load_page_info", fake_load),
        ("_get_repo", fake_get_repo),
        ("get_session", fake_session),
        ("_resolve_all_contexts", all_contexts),
        ("_resolve_repo_context", one_context),
        ("_build_meta", lambda **kwargs: {}),
        ("_get_exclude_spec", lambda path: None),
        ("source_search_enabled", lambda: False),
        ("_is_workspace_mode", lambda: False),
    ):
        monkeypatch.setattr(tool_search, name, value)


async def test_repo_all_against_a_single_repo_server_means_omitting_it(monkeypatch):
    """Not "a federation of one". ``repo`` is documented workspace-only, and
    all of the one repo is what an unqualified search already returns — so the
    two spellings must not be able to produce different answers."""
    _single_repo_server(monkeypatch)

    omitted = await tool_search.search_codebase(_QUERY, mode="concept")
    everything = await tool_search.search_codebase(_QUERY, mode="concept", repo="all")

    assert everything == omitted


async def test_a_single_repo_response_carries_no_repo_tag(monkeypatch):
    """The tag is a workspace field; stamping "default" on it says nothing and
    invites a consumer to branch on a distinction that does not exist here."""
    _single_repo_server(monkeypatch)

    response = await tool_search.search_codebase(_QUERY, mode="concept", repo="all")

    assert not any("repo" in row for row in response["results"])
    assert not any("repo" in candidate for candidate in response.get("candidates") or [])


async def test_the_federated_lane_is_unreachable_without_a_registry(monkeypatch):
    _single_repo_server(monkeypatch)

    async def explode(*args, **kwargs):
        raise AssertionError("_federated_search reached in single-repo mode")

    monkeypatch.setattr(tool_search, "_federated_search", explode)

    await tool_search.search_codebase(_QUERY, mode="concept", repo="all")


# ---------------------------------------------------------------------------
# L6 — the CLI fan-out is the same merge
# ---------------------------------------------------------------------------


def _fan_out_results(monkeypatch, canned: dict[str, list[dict]], query: str = _QUERY):
    """Run ``--all`` over canned per-repo payloads and return the merged rows."""
    from repowise.cli.commands import search_cmd

    emitted: dict = {}

    def fake_run_search(repo_path, q, limit, tool_mode):
        return {"results": [dict(row) for row in canned[repo_path.name]], "mode": "concept"}

    monkeypatch.setattr(search_cmd, "_run_search", fake_run_search)
    monkeypatch.setattr(search_cmd, "emit_json", lambda payload: emitted.update(payload))

    search_cmd._fan_out(
        [Path(f"/ws/{name}") for name in canned],
        query,
        5,
        "semantic",
        "concept",
        "json",
        False,
        search_cmd._notices,
    )
    return emitted.get("results") or []


def test_the_cli_fan_out_keeps_every_repo_s_noise_demotion(monkeypatch):
    rows = _fan_out_results(monkeypatch, _DEMOTED)
    paths = [row["path"] for row in rows]

    assert paths[:2] == ["alpha/impl.py", "beta/impl.py"]
    assert set(paths[2:]) == {"alpha/why.md", "beta/tests/test_impl.py"}


def test_the_cli_fan_out_demotes_a_repo_whose_whole_answer_is_noise(monkeypatch):
    """Rank fusion hid this: merging on rank alone preserved each repo's own
    order, so it looked partition-aware for as long as every repo had a real
    page to lead with. A repo that returned only a decision record led with it
    at rank 0 — ahead of every real page another repo ranked below its own."""
    rows = _fan_out_results(
        monkeypatch,
        {
            "alpha": [_page("alpha/why.md", 9.5, "decision_record")],
            "beta": [_page("beta/impl.py", 4.0), _page("beta/util.py", 3.0)],
        },
    )

    assert [row["path"] for row in rows] == [
        "beta/impl.py",
        "beta/util.py",
        "alpha/why.md",
    ]


def test_the_cli_fan_out_ranks_real_pages_by_relevance_not_by_repo_order(monkeypatch):
    """Cross-corpus RRF gave every repo's rank-N the same score, so the merge
    resolved to the order the repos happened to be visited in."""
    rows = _fan_out_results(
        monkeypatch,
        {
            "alpha": [_page("alpha/weak.py", 1.0)],
            "beta": [_page("beta/strong.py", 9.0)],
        },
    )

    assert [row["path"] for row in rows] == ["beta/strong.py", "alpha/weak.py"]


@pytest.mark.parametrize("mode", ["concept"])
def test_the_cli_fan_out_still_returns_every_repo(monkeypatch, mode):
    rows = _fan_out_results(monkeypatch, _DEMOTED)
    assert {row["repo"] for row in rows} == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# M9 — a workspace answer has to say how fresh each corpus behind it was
# ---------------------------------------------------------------------------


def _repo_row(tmp_path, name: str, *, age_days: int):
    from datetime import UTC, datetime, timedelta

    local = tmp_path / name
    local.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        name=name,
        local_path=str(local),
        head_commit="a" * 40,
        updated_at=datetime.now(UTC) - timedelta(days=age_days),
    )


def test_federated_freshness_reports_every_repo_and_the_oldest_age(tmp_path):
    from repowise.server.mcp_server._meta import federated_freshness

    meta = federated_freshness(
        [
            ("alpha", _repo_row(tmp_path, "alpha", age_days=1), []),
            ("beta", _repo_row(tmp_path, "beta", age_days=42), []),
        ]
    )

    assert set(meta["repo_freshness"]) == {"alpha", "beta"}
    assert meta["index_age_days"] == 42, "the workspace is as stale as its stalest corpus"


def test_federated_freshness_keeps_each_repos_own_commit_out_of_the_rollup(tmp_path):
    """One repo's SHA must not be published as though it described the whole
    workspace — the flaw the CLI's ``_worst_freshness`` admits to in its own
    comment. Each SHA stays with its owner in ``repos``."""
    from repowise.server.mcp_server._meta import federated_freshness

    meta = federated_freshness([("alpha", _repo_row(tmp_path, "alpha", age_days=1), [])])

    assert "indexed_commit" not in meta
    assert "live_head" not in meta
    assert meta["repo_freshness"]["alpha"]["indexed_commit"] == "a" * 12


def test_federated_freshness_of_nothing_is_nothing():
    from repowise.server.mcp_server._meta import federated_freshness

    assert federated_freshness([]) == {}


def test_an_unreadable_repo_row_still_names_the_repo_it_consulted():
    from repowise.server.mcp_server._meta import federated_freshness

    meta = federated_freshness([("alpha", None, [])])

    assert meta["repo_freshness"] == {"alpha": {}}
    assert "index_behind" not in meta, "absence must mean not evaluated, never false"


async def test_a_federated_response_carries_the_freshness_envelope(monkeypatch, tmp_path):
    """The end of the wire: ``_federated_search`` built its meta bare, so the
    one response shape whose rows differ most in staleness carried no signal."""
    real_freshness = getattr(tool_search, "_federated_freshness", None)
    assert real_freshness is not None, "the federated lane has no freshness lookup"
    _fake_repos(monkeypatch, _DEMOTED, stub_meta=False)
    # Put the real one back: this is the test that exercises it end to end.
    monkeypatch.setattr(tool_search, "_federated_freshness", real_freshness)

    rows = {
        "alpha": _repo_row(tmp_path, "alpha", age_days=2),
        "beta": _repo_row(tmp_path, "beta", age_days=9),
    }
    opened: list[str] = []

    async def fake_get_repo(session, repo=None):
        return rows[opened.pop(0)]

    @contextlib.asynccontextmanager
    async def fake_session(session_factory):
        opened.append(session_factory)
        yield object()

    monkeypatch.setattr(tool_search, "_get_repo", fake_get_repo)
    monkeypatch.setattr(tool_search, "get_session", fake_session)

    response = await tool_search._federated_search(_QUERY, 5, None)

    assert set(response["_meta"]["repo_freshness"]) == {"alpha", "beta"}
    assert response["_meta"]["index_age_days"] == 9
