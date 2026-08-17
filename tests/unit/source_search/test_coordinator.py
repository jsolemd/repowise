"""The retrieval coordinator: fusion, ranking, the owner, and the confidence.

Every store is a fake here, and deliberately so. What is under test is the
*policy* — which leg contributes what, what outranks what, and when the
response is entitled to claim it found something — and none of that is a
property of LanceDB or FTS5. The stores' own round-trips are covered by
``test_vector_store`` and ``test_fts``.
"""

from __future__ import annotations

from typing import Any

import pytest

from repowise.core.source_search.coordinator import (
    AGREEMENT_DENSE_COSINE,
    CONFIDENT_DENSE_COSINE,
    NO_MATCH_DENSE_COSINE,
    SourceSearchCoordinator,
)
from repowise.core.source_search.query_log import QueryLog
from repowise.core.source_search.vector_store import SourceChunkHit, SourceChunkRecord

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Embedder:
    """A fixed vector: what the legs do with it is what is under test."""

    dimensions = 4

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _SourceVectors:
    def __init__(
        self, hits: list[SourceChunkHit], records: dict[str, SourceChunkRecord] | None = None
    ):
        self._hits = hits
        self._records = records or {}

    async def search_by_vector(self, vector: Any, limit: int = 20) -> list[SourceChunkHit]:
        return self._hits[:limit]

    async def fetch_by_chunk_ids(self, chunk_ids: Any) -> dict[str, SourceChunkRecord]:
        return {cid: self._records[cid] for cid in chunk_ids if cid in self._records}


class _SourceFTS:
    def __init__(self, hits: list[Any]) -> None:
        self._hits = hits

    def query(self, match: str, limit: int = 20) -> list[Any]:
        return self._hits[:limit]


class _WikiVectors:
    def __init__(self, hits: list[Any]) -> None:
        self._hits = hits

    async def search_by_vector(self, vector: Any, limit: int = 10) -> list[Any]:
        return self._hits[:limit]


class _WikiFTS:
    def __init__(self, hits: list[Any]) -> None:
        self._hits = hits

    async def search(self, query: str, limit: int = 10) -> list[Any]:
        return self._hits[:limit]


class _FTSHit:
    """The shape ``SourceFTSIndex.query`` returns."""

    def __init__(self, chunk_id: str, file_path: str, score: float = 5.0) -> None:
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.score = score


class _PageHit:
    """The shape both wiki retrievers return (``SearchResult``)."""

    def __init__(
        self,
        page_id: str,
        target_path: str,
        score: float,
        page_type: str = "file_page",
        title: str = "",
        snippet: str = "",
    ) -> None:
        self.page_id = page_id
        self.target_path = target_path
        self.score = score
        self.page_type = page_type
        self.title = title or target_path
        self.snippet = snippet


def _hit(
    name: str,
    path: str,
    score: float,
    *,
    kind: str = "function",
    is_test: bool = False,
    source: str = "symbol",
    snippet: str = "",
) -> SourceChunkHit:
    return SourceChunkHit(
        chunk_id=f"{path}::{name}",
        file_path=path,
        name=name,
        kind=kind,
        start_line=1,
        end_line=9,
        is_test=is_test,
        source=source,
        content_hash="h",
        snippet=snippet or f"def {name}(): ...",
        score=score,
    )


def _record(hit: SourceChunkHit) -> SourceChunkRecord:
    return SourceChunkRecord(
        chunk_id=hit.chunk_id,
        file_path=hit.file_path,
        name=hit.name,
        kind=hit.kind,
        start_line=hit.start_line,
        end_line=hit.end_line,
        is_test=hit.is_test,
        source=hit.source,
        content_hash=hit.content_hash,
        snippet=hit.snippet,
    )


def _coordinator(
    tmp_path,
    *,
    source_dense: list[SourceChunkHit] | None = None,
    source_lexical: list[Any] | None = None,
    wiki_dense: list[Any] | None = None,
    wiki_lexical: list[Any] | None = None,
    records: dict[str, SourceChunkRecord] | None = None,
    embedder: Any = None,
) -> SourceSearchCoordinator:
    return SourceSearchCoordinator(
        repo_path=tmp_path,
        embedder=embedder or _Embedder(),
        source_vectors=_SourceVectors(source_dense or [], records),  # type: ignore[arg-type]
        source_fts=_SourceFTS(source_lexical or []),  # type: ignore[arg-type]
        wiki_vectors=_WikiVectors(wiki_dense or []),
        wiki_fts=_WikiFTS(wiki_lexical or []),
        query_log=QueryLog(tmp_path / "log.jsonl"),
    )


def _files(response: dict) -> list[str]:
    return [item["file"] for item in response["results"]]


# ---------------------------------------------------------------------------
# The two legs
# ---------------------------------------------------------------------------


async def test_the_dense_leg_merges_both_corpora_by_raw_cosine(tmp_path):
    """One ranking, ordered by cosine, regardless of which store produced it."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[_hit("alpha", "src/a.py", 0.60), _hit("gamma", "src/c.py", 0.40)],
        wiki_dense=[_PageHit("file_page:src/b.py", "src/b.py", 0.50)],
    )
    response = await coordinator.search("how does alpha work", limit=5)
    assert _files(response) == ["src/a.py", "src/b.py", "src/c.py"]
    assert [item["evidence"]["lane"] for item in response["results"]] == [
        "source",
        "wiki",
        "source",
    ]


async def test_the_query_is_embedded_once_for_both_dense_stores(tmp_path):
    """Two embeddings of one query are the same vector at twice the cost."""
    embedder = _Embedder()
    coordinator = _coordinator(
        tmp_path,
        embedder=embedder,
        source_dense=[_hit("alpha", "src/a.py", 0.6)],
        wiki_dense=[_PageHit("file_page:src/b.py", "src/b.py", 0.5)],
    )
    await coordinator.search("anything at all", limit=5)
    assert embedder.calls == 1


async def test_a_lexical_only_source_hit_arrives_with_its_metadata(tmp_path):
    """BM25 returns ids; the response still owes the caller lines and a body."""
    lexical_only = _hit("solo", "src/solo.py", 0.0, snippet="def solo(): ...")
    coordinator = _coordinator(
        tmp_path,
        source_lexical=[_FTSHit(lexical_only.chunk_id, lexical_only.file_path)],
        records={lexical_only.chunk_id: _record(lexical_only)},
    )
    response = await coordinator.search("solo behaviour", limit=5)
    result = response["results"][0]
    assert result["file"] == "src/solo.py"
    assert result["name"] == "solo"
    assert result["kind"] == "function"
    assert result["snippet"] == "def solo(): ..."
    assert (result["start_line"], result["end_line"]) == (1, 9)
    assert result["evidence"]["lexical_rank"] == 1
    assert result["evidence"]["dense_cosine"] is None


async def test_a_failing_leg_does_not_take_the_other_one_down(tmp_path):
    """One leg failing is a degraded answer; both failing is no answer."""

    class _Broken:
        async def search_by_vector(self, vector: Any, limit: int = 10) -> list[Any]:
            raise RuntimeError("wiki store is down")

    coordinator = _coordinator(tmp_path, source_dense=[_hit("alpha", "src/a.py", 0.6)])
    coordinator._wiki_vectors = _Broken()  # type: ignore[assignment]
    response = await coordinator.search("how does alpha work", limit=5)
    assert _files(response) == ["src/a.py"]


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


async def test_rrf_ranks_a_two_leg_hit_above_a_better_single_leg_one(tmp_path):
    """Agreement between independent retrievers is the signal RRF is buying."""
    both = _hit("agreed", "src/agreed.py", 0.50)
    dense_only = _hit("lonely", "src/lonely.py", 0.95)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[dense_only, both],
        source_lexical=[_FTSHit(both.chunk_id, both.file_path)],
    )
    response = await coordinator.search("agreed behaviour", limit=5)
    assert _files(response) == ["src/agreed.py", "src/lonely.py"]

    top, second = response["results"]
    assert top["evidence"]["lexical_rank"] == 1
    assert second["evidence"]["lexical_rank"] is None
    assert top["relevance_score"] > second["relevance_score"]


async def test_the_weights_are_the_documented_ones(tmp_path):
    """0.7 dense + 0.3 lexical over ``60 + rank``, ranks counted from one."""
    hit = _hit("alpha", "src/a.py", 0.6)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[hit],
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path)],
    )
    response = await coordinator.search("alpha behaviour", limit=5)
    assert response["results"][0]["relevance_score"] == pytest.approx(0.7 / 61 + 0.3 / 61, abs=1e-6)


# ---------------------------------------------------------------------------
# Test demotion
# ---------------------------------------------------------------------------


async def test_tests_fall_behind_product_code(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("thing", "tests/test_thing.py", 0.90, is_test=True),
            _hit("thing", "src/thing.py", 0.50),
        ],
    )
    response = await coordinator.search("how thing behaves", limit=5)
    assert _files(response) == ["src/thing.py", "tests/test_thing.py"]


async def test_a_query_about_tests_ranks_them_naturally(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("thing", "tests/test_thing.py", 0.90, is_test=True),
            _hit("thing", "src/thing.py", 0.50),
        ],
    )
    response = await coordinator.search("which fixture covers thing", limit=5)
    assert _files(response) == ["tests/test_thing.py", "src/thing.py"]


async def test_a_repair_query_naming_a_test_still_wants_the_code(tmp_path):
    """ "fix the failing thing test" names a symptom, not a subject."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("thing", "tests/test_thing.py", 0.90, is_test=True),
            _hit("thing", "src/thing.py", 0.50),
        ],
    )
    response = await coordinator.search("fix the failing thing test", limit=5)
    assert _files(response) == ["src/thing.py", "tests/test_thing.py"]


async def test_demotion_reorders_and_never_drops(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[_hit("only", "tests/test_only.py", 0.9, is_test=True)],
    )
    response = await coordinator.search("how only behaves", limit=5)
    assert _files(response) == ["tests/test_only.py"]


# ---------------------------------------------------------------------------
# The exact-identifier router
# ---------------------------------------------------------------------------


async def test_a_bare_identifier_puts_its_definition_first(tmp_path):
    """Dense retrieval answers a name with its neighbourhood; this corrects it."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("neighbour", "src/near.py", 0.90, snippet="calls merged_with(other)"),
            _hit("unrelated", "src/far.py", 0.80),
            _hit("merged_with", "src/target.py", 0.40),
        ],
    )
    response = await coordinator.search("merged_with", limit=5)
    assert _files(response) == ["src/target.py", "src/near.py", "src/far.py"]
    assert response["results"][0]["evidence"]["exact_name"] is True
    assert response["results"][1]["evidence"]["exact_name"] is False


async def test_the_router_matches_the_last_dotted_segment(tmp_path):
    """``Class.method`` and a stored ``Class::method`` are one name."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("other", "src/other.py", 0.9),
            _hit("method", "src/target.py", 0.1, kind="method"),
        ],
    )
    response = await coordinator.search("Class.method", limit=5)
    assert _files(response) == ["src/target.py", "src/other.py"]


async def test_the_router_is_case_insensitive(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("other", "src/other.py", 0.9),
            _hit("HostHeaderClient", "src/target.py", 0.1, kind="class"),
        ],
    )
    response = await coordinator.search("hostheaderclient", limit=5)
    assert _files(response) == ["src/target.py", "src/other.py"]


async def test_a_phrase_is_not_routed_as_an_identifier(tmp_path):
    """Two words are a topic. Fusion order stands."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("other", "src/other.py", 0.9),
            _hit("merged_with", "src/target.py", 0.1),
        ],
    )
    response = await coordinator.search("merged_with rows", limit=5)
    assert _files(response) == ["src/other.py", "src/target.py"]


# ---------------------------------------------------------------------------
# Dedupe and the window
# ---------------------------------------------------------------------------


async def test_one_file_yields_one_result(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("first", "src/a.py", 0.90),
            _hit("second", "src/a.py", 0.80),
            _hit("third", "src/b.py", 0.70),
        ],
    )
    response = await coordinator.search("how a works", limit=5)
    assert _files(response) == ["src/a.py", "src/b.py"]
    assert response["results"][0]["name"] == "first"


async def test_the_limit_cuts_the_window_but_not_the_candidates(tmp_path):
    """Candidates reach past the cut, so a short window still names files."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[_hit(f"s{i}", f"src/{i}.py", 0.9 - i / 100) for i in range(6)],
    )
    response = await coordinator.search("how the thing works", limit=2)
    assert len(response["results"]) == 2
    assert len(response["candidates"]) == 2
    assert response["candidates"][0] == {"path": "src/0.py"}


async def test_a_page_that_names_no_file_never_becomes_a_path(tmp_path):
    """A module page's target is a group key; an SCC page's is a hash."""
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[
            _PageHit("module_page:src/pkg", "src/pkg", 0.90, page_type="module_page"),
            _PageHit("scc_page:scc-abc", "scc-abc", 0.85, page_type="scc_page"),
            _PageHit("file_page:src/real.py", "src/real.py", 0.40),
        ],
    )
    response = await coordinator.search("how the package fits together", limit=5)
    assert [c["path"] for c in response["candidates"]] == ["src/real.py"]
    assert response["selected_owner"]["file"] == "src/real.py"


# ---------------------------------------------------------------------------
# Owner selection
# ---------------------------------------------------------------------------


async def test_the_owner_is_named_with_the_evidence_for_it(tmp_path):
    hit = _hit("alpha", "src/a.py", 0.6)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[hit],
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path)],
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["selected_owner"] == {
        "file": "src/a.py",
        "reason": "dense+lexical agreement",
    }


async def test_an_exact_name_hit_says_so(tmp_path):
    coordinator = _coordinator(tmp_path, source_dense=[_hit("merged_with", "src/a.py", 0.6)])
    response = await coordinator.search("merged_with", limit=5)
    assert response["selected_owner"]["reason"] == "exact name match"


async def test_a_definition_is_preferred_over_a_closely_scored_wiki_page(tmp_path):
    """Inside the band the shape decides; the page keeps its place in results."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[_hit("alpha", "src/a.py", 0.50, kind="class")],
        wiki_dense=[_PageHit("file_page:src/b.py", "src/b.py", 0.51)],
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["selected_owner"]["file"] == "src/a.py"
    assert "preferred over a closely-scored wiki page" in response["selected_owner"]["reason"]
    assert set(_files(response)) == {"src/a.py", "src/b.py"}


async def test_a_clear_winner_outside_the_band_is_not_second_guessed(tmp_path):
    """The fusion is allowed to be right about a page the shape rules dislike."""
    page = _PageHit("file_page:src/page.py", "src/page.py", 0.90)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[_hit("alpha", "src/a.py", 0.10)],
        wiki_dense=[page],
        wiki_lexical=[page],
    )
    response = await coordinator.search("how the page thing works", limit=5)
    assert response["selected_owner"]["file"] == "src/page.py"


async def test_no_results_means_no_owner(tmp_path):
    coordinator = _coordinator(tmp_path)
    response = await coordinator.search("nothing at all like this", limit=5)
    assert response["results"] == []
    assert response["selected_owner"] is None


# ---------------------------------------------------------------------------
# Confidence — absolute thresholds, no window normalisation
# ---------------------------------------------------------------------------


async def test_a_strong_cosine_alone_is_confident(tmp_path):
    coordinator = _coordinator(
        tmp_path, source_dense=[_hit("alpha", "src/a.py", CONFIDENT_DENSE_COSINE)]
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["confidence"] == "confident"


async def test_just_under_the_threshold_is_caution(tmp_path):
    coordinator = _coordinator(
        tmp_path, source_dense=[_hit("alpha", "src/a.py", CONFIDENT_DENSE_COSINE - 0.01)]
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["confidence"] == "caution"


async def test_a_middling_cosine_plus_lexical_agreement_is_confident(tmp_path):
    hit = _hit("alpha", "src/a.py", AGREEMENT_DENSE_COSINE)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[hit],
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path)],
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["confidence"] == "confident"


async def test_an_exact_name_is_confident_however_weak_the_cosine(tmp_path):
    coordinator = _coordinator(tmp_path, source_dense=[_hit("merged_with", "src/a.py", 0.05)])
    response = await coordinator.search("merged_with", limit=5)
    assert response["confidence"] == "confident"


async def test_weak_cosine_with_nothing_else_is_no_match(tmp_path):
    """The absent-topic rule: nearest noise is not an answer."""
    coordinator = _coordinator(
        tmp_path, source_dense=[_hit("alpha", "src/a.py", NO_MATCH_DENSE_COSINE - 0.01)]
    )
    response = await coordinator.search("croissant lamination fold schedule", limit=5)
    assert response["confidence"] == "no_match"
    assert "No indexed match" in response["note"]


async def test_an_empty_result_set_is_no_match(tmp_path):
    coordinator = _coordinator(tmp_path)
    response = await coordinator.search("croissant lamination fold schedule", limit=5)
    assert response["confidence"] == "no_match"


async def test_a_wiki_owner_no_source_chunk_corroborates_is_only_caution(tmp_path):
    """Prose about code, with no code behind it, is not a confident answer."""
    page = _PageHit("file_page:src/page.py", "src/page.py", 0.90)
    coordinator = _coordinator(
        tmp_path,
        # Both legs, so the page wins by far more than the owner policy's band
        # and the shape rules cannot promote the chunk over it.
        wiki_dense=[page],
        wiki_lexical=[page],
        source_dense=[_hit("elsewhere", "src/other.py", 0.20)],
    )
    response = await coordinator.search("how the page thing works", limit=5)
    assert response["selected_owner"]["file"] == "src/page.py"
    assert response["confidence"] == "caution"


async def test_a_wiki_owner_the_source_corpus_agrees_with_is_confident(tmp_path):
    """Corroboration is read before the dedupe that would always destroy it."""
    page = _PageHit("file_page:src/page.py", "src/page.py", 0.90)
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[page],
        wiki_lexical=[page],
        source_dense=[_hit("thing", "src/page.py", 0.85, source="file_window")],
    )
    response = await coordinator.search("how the page thing works", limit=5)
    assert response["selected_owner"]["file"] == "src/page.py"
    assert response["confidence"] == "confident"
    # One file, one result: the chunk and the page about it collapsed.
    assert _files(response) == ["src/page.py"]


async def test_confidence_is_absolute_not_relative_to_the_window(tmp_path):
    """A window of uniformly weak hits stays weak, however it is normalised."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[_hit(f"s{i}", f"src/{i}.py", 0.20 - i / 1000) for i in range(5)],
    )
    response = await coordinator.search("scuba cylinder hydrostatic inspection", limit=5)
    assert response["confidence"] == "no_match"
    assert len(response["results"]) == 5


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


async def test_the_envelope_carries_every_contract_key(tmp_path):
    hit = _hit("alpha", "src/a.py", 0.6)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[hit],
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path)],
    )
    response = await coordinator.search("how alpha works", limit=5, mode="hybrid")

    assert set(response) >= {
        "results",
        "candidates",
        "selected_owner",
        "confidence",
        "mode",
        "_meta",
    }
    assert response["mode"] == "hybrid"

    result = response["results"][0]
    assert set(result) >= {
        "file",
        "name",
        "kind",
        "snippet",
        "start_line",
        "end_line",
        "relevance_score",
        "evidence",
    }
    assert set(result["evidence"]) == {"dense_cosine", "lexical_rank", "exact_name", "lane"}
    assert response["candidates"] == [{"path": "src/a.py"}]


async def test_the_host_meta_survives_and_the_generation_is_added(tmp_path):
    from repowise.core.source_search.manifest import (
        EmbedderIdentity,
        SourceIndexManifest,
        default_manifest_path,
        write_manifest,
    )

    write_manifest(
        default_manifest_path(tmp_path),
        SourceIndexManifest(
            recipe_fingerprint="r",
            corpus_hash="c0ffee1234567890",
            symbol_chunks=7,
            file_window_chunks=3,
            files_covered=4,
            indexed_commit="abc123",
            built_at="2026-08-17T00:00:00+00:00",
            embedder=EmbedderIdentity(provider="ollama", model="embeddinggemma", dims=768),
        ),
    )
    coordinator = _coordinator(tmp_path, source_dense=[_hit("alpha", "src/a.py", 0.6)])
    response = await coordinator.search(
        "how alpha works", limit=5, base_meta={"indexed_commit": "abc123", "cached": True}
    )

    assert response["_meta"]["cached"] is True
    assert response["_meta"]["source_search"] == {
        "generation": "c0ffee123456",
        "indexed_commit": "abc123",
        "symbol_chunks": 7,
        "file_window_chunks": 3,
    }
    assert response["_meta"]["timing_ms"] >= 0


async def test_a_repository_with_no_manifest_still_answers(tmp_path):
    coordinator = _coordinator(tmp_path, source_dense=[_hit("alpha", "src/a.py", 0.6)])
    response = await coordinator.search("how alpha works", limit=5)
    assert response["_meta"]["source_search"]["generation"] is None
    assert response["results"]


# ---------------------------------------------------------------------------
# The query log
# ---------------------------------------------------------------------------


async def test_every_query_writes_one_event(tmp_path):
    import json

    hit = _hit("alpha", "src/a.py", 0.6)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[hit],
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path)],
    )
    await coordinator.search("how alpha works", limit=3, mode="concept")

    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["query"] == "how alpha works"
    assert event["mode"] == "concept"
    assert event["limit"] == 3
    assert event["result_count"] == 1
    assert event["confidence"] == "confident"
    assert event["selected_owner_file"] == "src/a.py"
    assert event["no_match"] is False
    assert event["latency_ms"] >= 0
    assert event["top"][0]["file"] == "src/a.py"
    assert event["top"][0]["lane"] == "source"
    assert event["top"][0]["lexical_rank"] == 1


async def test_a_log_that_cannot_be_written_does_not_fail_the_search(tmp_path):
    """A search that succeeded and then failed to record itself has succeeded."""
    coordinator = _coordinator(tmp_path, source_dense=[_hit("alpha", "src/a.py", 0.6)])
    # A directory where the file should be: every open() for append raises.
    (tmp_path / "log.jsonl").mkdir()
    response = await coordinator.search("how alpha works", limit=5)
    assert response["results"]
