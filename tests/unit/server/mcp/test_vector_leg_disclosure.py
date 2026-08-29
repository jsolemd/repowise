"""A vector leg that fell over must not look like a corpus with nothing to say.

The deployment this was written for pins ``REPOWISE_EMBEDDER=ollama``, and
ollama is a container that is down at boot every morning. Nothing fails at
initialisation — ``OllamaEmbedder.__init__`` never touches the network — so the
process-level ``embedder_degraded`` stays false, and the connection error
arrives per query, inside ``_safe_vector``, where it was caught at DEBUG and
turned into ``[]``. The caller then served full-text hits in the ordinary
envelope: a normal-looking result set from a search that ran one of its two
legs, every morning, silently.

These pin the disclosure. The results are still served — the leg is fail-soft
by design and that is not in question — but the response now says which leg did
not run, why, and that the failure is this query's rather than the install's.

Scope: the stock wiki lane (``_fused_retrieve``). The source-search coordinator
classifies its own leg failures (fd466d79) and builds its own envelope, so
these tests turn that lane off rather than testing it twice.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from repowise.server.mcp_server import tool_search

_QUERY = "authentication service"

#: Every served field of every row the healthy fixture ranks. Hardcoded rather
#: than derived: this is the roster-stability assertion, and a roster computed
#: from the code it is checking cannot fail. It passes unchanged against the
#: code from before the disclosure existed, which is what makes "zero ranking
#: change" a measurement rather than a claim — scores included, since the
#: freshness tiebreak and the fused scale are what a careless edit here moves.
_HEALTHY_ROSTER = [
    {
        "title": "Auth Service",
        "page_type": "file_page",
        "snippet": "Auth Service",
        "relevance_score": 6.03,
        "sources": ["fts", "vector"],
        "target_path": "src/auth/service.py",
        "confidence_score": 1.0,
    },
    {
        "title": "DB Models",
        "page_type": "file_page",
        "snippet": "DB Models",
        # 2.9508 before v0.46.0; upstream's rerank_by_context_coverage(floor=0.5)
        # now halves a fused score whose row covers none of the query's terms
        # ("authentication service" vs a DB-models page). Order unchanged; the
        # fork's disclosure code is not what moved it (program ticket F37).
        "relevance_score": 1.4754,
        "sources": ["vector"],
        "target_path": "src/db/models.py",
        "confidence_score": 0.98,
    },
]

_DISCLOSURE_KEYS = ("retrieval_degraded", "retrieval_degraded_reason", "semantic_search")


@pytest.fixture(autouse=True)
def _stock_lane(monkeypatch):
    """Keep these on the stock wiki lane whatever the ambient env pins."""
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "0")


def _hit(page_id: str, title: str, target_path: str, score: float):
    from repowise.core.persistence.search import SearchResult

    return SearchResult(
        page_id=page_id,
        title=title,
        page_type="file_page",
        target_path=target_path,
        score=score,
        snippet=title,
        search_type="vector",
    )


async def _vector_rows(query, limit=10):
    return [
        _hit("file_page:src/auth/service.py", "Auth Service", "src/auth/service.py", 0.61),
        _hit("file_page:src/db/models.py", "DB Models", "src/db/models.py", 0.44),
    ]


async def _fts_rows(query, limit=10):
    return [_hit("file_page:src/auth/service.py", "Auth Service", "src/auth/service.py", 0.52)]


async def _no_rows(query, limit=10):
    return []


def _install(mcp_mod, vector_search, fts_search=_fts_rows) -> None:
    """Point both retrievers at canned rows so the roster is a fixture.

    FTS is stubbed for the same reason the federated test in ``test_search``
    stubs it: the shared in-memory engine isolates per connection, so a live
    handle shadows the seeded rows for the follow-up session.
    """
    mcp_mod._vector_store.search = vector_search
    mcp_mod._fts.search = fts_search


def _roster(response: dict) -> list[dict]:
    """The served rows, whole. ``page_id`` is already dropped as derivable."""
    return response["results"]


# ---------------------------------------------------------------------------
# The failure the deployment actually hits
# ---------------------------------------------------------------------------


async def test_a_dead_embedding_backend_is_disclosed_per_request(setup_mcp):
    """The morning-after case: ollama is down, the query embeds against nothing."""
    import repowise.server.mcp_server as mcp_mod
    from repowise.server.mcp_server import search_codebase

    async def refused(query, limit=10):
        raise ConnectionRefusedError(111, "Connection refused")

    # The deployment's own state: ollama is configured and initialised cleanly,
    # because nothing in its constructor touches the network. Everything that
    # latches at startup says the install is healthy, and it is the query that
    # is not. ``setup_mcp`` resets this on teardown.
    mcp_mod._embedder_status = {"active": "ollama", "requested": "ollama", "degraded": False}

    _install(mcp_mod, refused)
    response = await search_codebase(_QUERY)

    # Fail-soft is preserved: the lexical leg still answers.
    assert response["results"], "full-text results must still be served"
    assert all(row["sources"] == ["fts"] for row in response["results"])

    meta = response["_meta"]
    assert meta["retrieval_degraded"] == ["vector"]
    assert meta["semantic_search"] is False
    reason = meta["retrieval_degraded_reason"]
    assert "ConnectionRefusedError" in reason, "the cause has to travel with the disclosure"
    assert "full-text only" in reason
    # An install-level claim latched at startup must not be rewritten by one
    # query's failure; that is what the separate key is for.
    assert meta["embedder_degraded"] is False


async def test_a_timeout_discloses_the_same_way(setup_mcp, monkeypatch):
    """A budget overrun drops exactly as much of the answer as an error does."""
    import repowise.server.mcp_server as mcp_mod
    from repowise.server.mcp_server import search_codebase
    from repowise.server.mcp_server._helpers import _VECTOR_TIMEOUT_ENV

    monkeypatch.setenv(_VECTOR_TIMEOUT_ENV, "0.01")

    async def slow(query, limit=10):
        await asyncio.sleep(1.0)
        return []

    _install(mcp_mod, slow)
    response = await search_codebase(_QUERY)

    assert response["results"]
    meta = response["_meta"]
    assert meta["retrieval_degraded"] == ["vector"]
    assert meta["semantic_search"] is False
    assert "timed out" in meta["retrieval_degraded_reason"]


async def test_an_empty_answer_from_a_half_read_corpus_still_says_so(setup_mcp):
    """The case with the most riding on it. Zero results is a claim about the
    repository — the one answer a half-read corpus is least entitled to give —
    and an agent reads it as "not in the codebase" unless told otherwise.

    Still a success envelope, not an error: the lexical leg ran and answered,
    so this is not fd466d79's every-leg-failed case, which is the only one that
    forfeits the right to a result set.
    """
    import repowise.server.mcp_server as mcp_mod
    from repowise.server.mcp_server import search_codebase

    async def refused(query, limit=10):
        raise ConnectionRefusedError(111, "Connection refused")

    _install(mcp_mod, refused, _no_rows)
    response = await search_codebase(_QUERY)

    assert response["results"] == []
    assert "error" not in response
    assert response["_meta"]["retrieval_degraded"] == ["vector"]
    assert response["_meta"]["semantic_search"] is False


async def test_hybrid_mode_discloses_from_its_own_envelope(setup_mcp):
    """``_structured_search`` builds a second envelope; it must say this too."""
    import repowise.server.mcp_server as mcp_mod
    from repowise.server.mcp_server import search_codebase

    async def refused(query, limit=10):
        raise ConnectionRefusedError(111, "Connection refused")

    _install(mcp_mod, refused)
    response = await search_codebase("where is the auth service handled", mode="hybrid")

    assert response["mode"] == "hybrid"
    assert response["_meta"]["retrieval_degraded"] == ["vector"]


# ---------------------------------------------------------------------------
# The healthy path pays nothing
# ---------------------------------------------------------------------------


async def test_a_whole_search_is_quiet_and_ranks_exactly_as_before(setup_mcp):
    """Roster pinned to the fixture, envelope free of every disclosure key."""
    import repowise.server.mcp_server as mcp_mod
    from repowise.server.mcp_server import search_codebase

    _install(mcp_mod, _vector_rows)
    response = await search_codebase(_QUERY)

    assert _roster(response) == _HEALTHY_ROSTER
    for key in _DISCLOSURE_KEYS:
        assert key not in response["_meta"], f"a healthy response must not carry {key}"


async def test_the_disclosure_clears_on_the_next_good_query(setup_mcp):
    """Recovery is per-request: embedding is per-call and nothing latches."""
    import repowise.server.mcp_server as mcp_mod
    from repowise.server.mcp_server import search_codebase

    async def refused(query, limit=10):
        raise ConnectionRefusedError(111, "Connection refused")

    _install(mcp_mod, refused)
    broken = await search_codebase(_QUERY)
    assert broken["_meta"]["retrieval_degraded"] == ["vector"]

    # The container came back. No restart, no reindex, same process.
    _install(mcp_mod, _vector_rows)
    recovered = await search_codebase(_QUERY)

    for key in _DISCLOSURE_KEYS:
        assert key not in recovered["_meta"], f"stale {key} survived the recovery"
    assert _roster(recovered) == _HEALTHY_ROSTER


# ---------------------------------------------------------------------------
# What counts as a failure
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, store):
        self.vector_store = store
        self.fts = None
        self.vector_store_ready = None


async def test_a_keyless_index_is_not_a_failed_leg():
    """A permanent configuration, already reported once per response by _meta."""
    from repowise.core.persistence.vector_store.in_memory import InMemoryVectorStore
    from repowise.core.providers.embedding.base import KeylessEmbedder

    tool_search._begin_retrieval_record()
    assert (
        await tool_search._safe_vector(_Ctx(InMemoryVectorStore(KeylessEmbedder())), "q", 5) == []
    )
    assert tool_search._retrieval_disclosure() == {}


async def test_a_server_with_no_vector_store_is_not_a_failed_leg():
    tool_search._begin_retrieval_record()
    assert await tool_search._safe_vector(_Ctx(None), "q", 5) == []
    assert tool_search._retrieval_disclosure() == {}


async def test_the_class_a_stopped_ollama_actually_raises_reaches_the_disclosure():
    """The chain, end to end, in the classes the deployment really uses.

    ``OllamaEmbedder.embed`` posts to ``/api/embed`` with no handler around it,
    and ``LanceDBVectorStore.search`` embeds the query inline with none either,
    so a refused connection leaves the store as an ``httpx.ConnectError``.
    That is a transport error and not a ``TimeoutError``, which is why the
    branch it lands in has to disclose as loudly as the timeout branch does.
    """
    import httpx

    class _RefusingStore:
        embedder = object()

        async def search(self, query, limit=None, **kw):
            raise httpx.ConnectError("[Errno 111] Connection refused")

    tool_search._begin_retrieval_record()
    assert await tool_search._safe_vector(_Ctx(_RefusingStore()), "q", 5) == []

    disclosure = tool_search._retrieval_disclosure()
    assert disclosure["retrieval_degraded"] == ["vector"]
    assert disclosure["semantic_search"] is False
    assert "ConnectError" in disclosure["retrieval_degraded_reason"]


async def test_a_repo_that_answered_cannot_bury_a_repo_that_did_not():
    """Federated: the leg ran over part of the workspace, so the answer is partial.

    ``_record_leg`` is last-wins, which would let repo B's success erase repo
    A's failure and leave the response claiming a whole semantic lane.
    """
    tool_search._begin_retrieval_record()
    tool_search._record_vector_leg("error", "ConnectionRefusedError: refused")
    tool_search._record_vector_leg("ok")

    disclosure = tool_search._retrieval_disclosure()
    assert disclosure["retrieval_degraded"] == ["vector"]
    assert "ConnectionRefusedError" in disclosure["retrieval_degraded_reason"]


async def test_the_federated_envelope_carries_both_freshness_and_the_disclosure():
    """The one meta site that already had extras. Neither may displace the other."""

    async def one_repo(ctx, query, limit, page_type, kind=None):
        tool_search._record_vector_leg("error", "ConnectionRefusedError: refused")
        return [
            {
                "page_id": "file_page:alpha/impl.py",
                "title": "Impl",
                "page_type": "file_page",
                "target_path": "alpha/impl.py",
                "relevance_score": 4.0,
                "snippet": "",
            }
        ]

    async def all_contexts():
        return [
            SimpleNamespace(
                alias="alpha",
                path="/ws/alpha",
                session_factory=None,
                fts=object(),
                vector_store=object(),
                vector_store_ready=None,
            )
        ]

    async def freshness(contexts, output):
        return {"repo_freshness": {"alpha": {}}}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tool_search, "_resolve_all_contexts", all_contexts)
        mp.setattr(tool_search, "_search_single_repo", one_repo)
        mp.setattr(tool_search, "_federated_freshness", freshness)
        tool_search._begin_retrieval_record()
        response = await tool_search._federated_search(_QUERY, 5, None)

    assert response["results"], "federation still serves what the lexical leg found"
    meta = response["_meta"]
    assert meta["repo_freshness"] == {"alpha": {}}
    assert meta["retrieval_degraded"] == ["vector"]


async def test_a_ranting_backend_cannot_bloat_every_degraded_response():
    """The cause is quoted, not pasted: some drivers put a stack trace in str()."""
    tool_search._begin_retrieval_record()
    tool_search._record_vector_leg("error", "RuntimeError: " + "x" * 5000)

    reason = tool_search._retrieval_disclosure()["retrieval_degraded_reason"]
    assert len(reason) < 500
    assert "RuntimeError" in reason


async def test_the_leg_outcome_uses_the_vocabulary_get_answer_reads():
    """One record, one spelling: ``degraded_legs`` computes both surfaces."""
    from repowise.server.mcp_server import _answer_pipeline as pipeline

    tool_search._begin_retrieval_record()
    tool_search._record_vector_leg("timeout", "timed out after 0.01s")

    assert pipeline.retrieval_legs() == {"vector": "timeout"}
    assert pipeline.degraded_legs(pipeline.retrieval_legs()) == ["vector"]


# ---------------------------------------------------------------------------
# The CLI reads the same envelope
# ---------------------------------------------------------------------------
#
# ``repowise search --mode semantic`` is an adapter over this tool and warns off
# ``semantic_search: false``. Setting that key for a transient failure would have
# made it print a cause and a repair that are both false — no embedder is
# configured, run a reindex — for an install whose embedder is configured and
# whose index is fine. Pinned here rather than in the CLI suite for the reason
# ``test_keyless_vector_leg`` gives: what is being tested is the chain across
# the seam, and it breaks from either end.


class _Notices:
    def __init__(self) -> None:
        self.said: list[str] = []

    def print(self, *args: object) -> None:
        self.said.append(" ".join(str(a) for a in args))


def test_cli_names_the_transient_cause_and_prescribes_no_reindex() -> None:
    from repowise.cli.commands.search_cmd import _warn_if_lexical_only

    tool_search._begin_retrieval_record()
    tool_search._record_vector_leg("error", "ConnectionRefusedError: [Errno 111] refused")
    payload = {"_meta": dict(tool_search._retrieval_disclosure())}

    notices = _Notices()
    _warn_if_lexical_only(payload, "semantic", notices)

    said = " ".join(notices.said)
    assert "ConnectionRefusedError" in said
    assert "Showing full-text results instead." in said
    assert "reindex" not in said, "the index is fine; only the leg fell over"
    assert "No embedder configured" not in said


def test_cli_still_calls_a_missing_embedder_what_it_is() -> None:
    """The configuration branch is untouched by the transient one."""
    from repowise.cli.commands.search_cmd import _warn_if_lexical_only

    notices = _Notices()
    _warn_if_lexical_only({"_meta": {"semantic_search": False}}, "semantic", notices)

    said = " ".join(notices.said)
    assert "No embedder configured" in said
    assert "reindex" in said
