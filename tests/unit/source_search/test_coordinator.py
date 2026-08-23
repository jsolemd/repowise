"""The retrieval coordinator: fusion, ranking, the owner, and the confidence.

Every store is a fake here, and deliberately so. What is under test is the
*policy* — which leg contributes what, what outranks what, and when the
response is entitled to claim it found something — and none of that is a
property of LanceDB or FTS5. The stores' own round-trips are covered by
``test_vector_store`` and ``test_fts``.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from repowise.core.source_search.coordinator import (
    AGREEMENT_DENSE_COSINE,
    CONFIDENT_DENSE_COSINE,
    NO_MATCH_DENSE_COSINE,
    SourceSearchCoordinator,
    _Item,
    _query_intent,
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
    def __init__(
        self,
        hits: list[Any],
        term_files: dict[str, set[str]] | None = None,
        active_paths: set[str] | None = None,
    ) -> None:
        self._hits = hits
        self._term_files = term_files
        self._active_paths = set(active_paths or ())
        self._active_paths.update(hit.file_path for hit in hits if hit.file_path)

    def query(self, match: str, limit: int = 20) -> list[Any]:
        return self._hits[:limit]

    def active_file_paths(self) -> list[str]:
        if self._term_files is not None:
            self._active_paths.update(path for paths in self._term_files.values() for path in paths)
        return sorted(self._active_paths)

    def term_file_evidence(self, terms: list[str] | tuple[str, ...]) -> dict[str, frozenset[str]]:
        if self._term_files is not None:
            return {term: frozenset(self._term_files.get(term, set())) for term in terms}
        # Every fake lexical hit stands in for a chunk matching this fake
        # query. Fine-grained co-location cases pass an explicit mapping.
        matched = frozenset(hit.file_path for hit in self._hits if hit.file_path)
        return {term: matched for term in terms}


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
    chunk_id: str | None = None,
    start_line: int = 1,
    end_line: int = 9,
) -> SourceChunkHit:
    return SourceChunkHit(
        chunk_id=chunk_id or f"{path}::{name}",
        file_path=path,
        name=name,
        kind=kind,
        start_line=start_line,
        end_line=end_line,
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
    source_term_files: dict[str, set[str]] | None = None,
    source_active_paths: set[str] | None = None,
) -> SourceSearchCoordinator:
    return SourceSearchCoordinator(
        repo_path=tmp_path,
        embedder=embedder or _Embedder(),
        source_vectors=_SourceVectors(source_dense or [], records),  # type: ignore[arg-type]
        source_fts=_SourceFTS(  # type: ignore[arg-type]
            source_lexical or [], source_term_files, source_active_paths
        ),
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


async def test_the_served_score_agrees_with_the_served_order(tmp_path):
    """The router reorders; a consumer that re-sorts by score must not undo it."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("neighbour", "src/near.py", 0.90),
            _hit("merged_with", "src/target.py", 0.40),
        ],
    )
    response = await coordinator.search("merged_with", limit=5)
    scores = [item["relevance_score"] for item in response["results"]]
    assert _files(response) == ["src/target.py", "src/near.py"]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[1]


# ---------------------------------------------------------------------------
# Identifiers carried inside prose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # snake_case, in all the casings it is written in
        ("fix the failing merged_with case", ["merged_with"]),
        ("where is HTTP_CALLS built", ["HTTP_CALLS"]),
        ("what sets _private_helper", ["_private_helper"]),
        # camelCase and CamelCase
        ("how does parseConfig read the file", ["parseConfig"]),
        ("where is HostHeaderClient constructed", ["HostHeaderClient"]),
        # dotted and ::-qualified
        ("what does Class.method return", ["Class.method"]),
        ("trace atlas::refresh through the run", ["atlas::refresh"]),
        # several at once, deduplicated, in the order written
        (
            "does parseConfig call merged_with or parseConfig again",
            ["parseConfig", "merged_with"],
        ),
    ],
)
def test_identifier_shaped_tokens_are_extracted_from_prose(query, expected):
    from repowise.core.source_search.coordinator import _embedded_identifiers

    assert _embedded_identifiers(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        # Plain English, including words a looser rule would claim.
        "how does the vault refresh skip unchanged files",
        "why is the coverage baseline rejected",
        "Obsidian vault notes written atomically for the Windows watcher",
        "route a clicked WSL file link to the right editor",
        # Sentence punctuation is not a qualified name, and neither are initials.
        "what happens next. the run continues",
        "the U.S. tax year boundary",
        # An acronym is one token, not a camel hump.
        "call the PostgREST endpoint",
    ],
)
def test_plain_prose_names_no_identifier(query):
    from repowise.core.source_search.coordinator import _embedded_identifiers

    assert _embedded_identifiers(query) == []


async def test_an_identifier_inside_prose_promotes_its_definition(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("other", "src/other.py", 0.90),
            _hit("merged_with", "src/target.py", 0.10),
        ],
    )
    response = await coordinator.search("fix the failing merged_with case", limit=5)
    assert _files(response) == ["src/target.py", "src/other.py"]
    assert response["results"][0]["evidence"]["exact_name"] is True
    assert response["results"][1]["evidence"]["exact_name"] is False


async def test_the_embedded_arm_names_itself_in_the_owner_reason(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("other", "src/other.py", 0.90),
            _hit("merged_with", "src/target.py", 0.10),
        ],
    )
    response = await coordinator.search("fix the failing merged_with case", limit=5)
    assert response["selected_owner"]["file"] == "src/target.py"
    assert response["selected_owner"]["reason"] == "embedded identifier match"
    assert response["selected_owner"]["evidence"]["exact_name"] is True
    assert response["confidence"] == "confident"


async def test_the_embedded_arm_does_not_promote_on_mere_mention(tmp_path):
    """Half a subsystem mentions any given helper; only a name match is evidence."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("other", "src/other.py", 0.90, snippet="calls merged_with(row)"),
            _hit("unrelated", "src/far.py", 0.10),
        ],
    )
    response = await coordinator.search("fix the failing merged_with case", limit=5)
    assert _files(response) == ["src/other.py", "src/far.py"]
    assert all(item["evidence"]["exact_name"] is False for item in response["results"])


async def test_a_whole_query_identifier_still_takes_the_stronger_router(tmp_path):
    """The two routers are exclusive, and the bare-identifier one keeps its tiers."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("other", "src/other.py", 0.90),
            _hit("mentions", "src/near.py", 0.50, snippet="calls merged_with(row)"),
            _hit("merged_with", "src/target.py", 0.10),
        ],
    )
    response = await coordinator.search("merged_with", limit=5)
    # Name match, then the containment tier the embedded arm deliberately lacks.
    assert _files(response) == ["src/target.py", "src/near.py", "src/other.py"]
    assert response["selected_owner"]["reason"] == "exact name match"


async def test_an_embedded_identifier_matches_a_wiki_symbol_page(tmp_path):
    """A page's name is its title; the symbol comes off the page id instead."""
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[
            _PageHit("file_page:src/other.py", "src/other.py", 0.90),
            _PageHit(
                "symbol_spotlight:src/target.py::pkg.mod.merged_with",
                "src/target.py::pkg.mod.merged_with",
                0.10,
                page_type="symbol_spotlight",
                title="Symbol: pkg.mod.merged_with",
            ),
        ],
    )
    response = await coordinator.search("fix the failing merged_with case", limit=5)
    assert _files(response) == ["src/target.py", "src/other.py"]
    assert response["results"][0]["evidence"]["exact_name"] is True


@pytest.mark.parametrize(
    ("identifier", "name"),
    [
        # The motivating shape: named by the part that carries the meaning.
        ("wants_tests", "query_wants_tests"),
        ("wants_tests", "_query_wants_tests"),
        # Three spellings of one boundary, on either side of the comparison.
        ("wants_tests", "queryWantsTests"),
        ("wantsTests", "query_wants_tests"),
        ("wants.tests", "query_wants_tests"),
        # More than one segment of prefix is still a suffix.
        ("wants_tests", "the_query_wants_tests"),
    ],
)
def test_a_boundary_aligned_tail_matches(identifier, name):
    from repowise.core.source_search.coordinator import _segments, _suffix_matches

    assert _suffix_matches(_segments(identifier), _segments(name))


@pytest.mark.parametrize(
    ("identifier", "name", "why"),
    [
        # The boundary requirement: nine shared characters, no shared segment.
        ("ants_tests", "query_wants_tests", "not aligned to a boundary"),
        # One segment would match half the corpus.
        ("tests", "query_wants_tests", "single segment"),
        ("py", "refresh_py", "single segment"),
        # Equal length is the same name — the full-name tier's claim, not this one.
        ("wants_tests", "wants_tests", "same name, stronger tier owns it"),
        # A tail is a tail, not a head and not a middle.
        ("query_wants", "query_wants_tests", "prefix, not suffix"),
        ("wants_tests", "wants_tests_helper", "suffix of the wrong end"),
        # Segments must be contiguous.
        ("query_tests", "query_wants_tests", "segments not adjacent"),
    ],
)
def test_a_tail_that_is_not_a_boundary_suffix_does_not_match(identifier, name, why):
    from repowise.core.source_search.coordinator import _segments, _suffix_matches

    assert not _suffix_matches(_segments(identifier), _segments(name)), why


async def test_a_suffix_match_ranks_below_a_full_name_match(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("unrelated", "src/other.py", 0.90),
            _hit("query_wants_tests", "src/suffix.py", 0.50),
            _hit("wants_tests", "src/exact.py", 0.10),
        ],
    )
    response = await coordinator.search("fix the wants_tests classifier", limit=5)
    assert _files(response) == ["src/exact.py", "src/suffix.py", "src/other.py"]


async def test_a_suffix_match_does_not_claim_an_exact_name(tmp_path):
    """``query_wants_tests`` is not the name ``wants_tests``. Say what is true."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("unrelated", "src/other.py", 0.90),
            _hit("query_wants_tests", "src/suffix.py", 0.10),
        ],
    )
    response = await coordinator.search("fix the wants_tests classifier", limit=5)
    assert _files(response) == ["src/suffix.py", "src/other.py"]
    assert response["results"][0]["evidence"]["exact_name"] is False
    assert response["selected_owner"]["file"] == "src/suffix.py"
    assert response["selected_owner"]["reason"] == "embedded identifier suffix match"
    assert response["selected_owner"]["evidence"]["exact_name"] is False


async def test_a_suffix_match_alone_does_not_make_the_answer_confident(tmp_path):
    """Confidence still comes from the cosine and the lexical agreement."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[_hit("query_wants_tests", "src/suffix.py", 0.35)],
    )
    response = await coordinator.search("fix the wants_tests classifier", limit=5)
    assert response["selected_owner"]["reason"] == "embedded identifier suffix match"
    assert response["confidence"] == "caution"


async def test_a_single_segment_identifier_promotes_nothing_by_tail(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("unrelated", "src/other.py", 0.90),
            _hit("run_tests", "src/runner.py", 0.10),
        ],
    )
    # ``parseTests`` is one identifier of two segments; its tail ``tests`` alone
    # must not drag in every name that happens to end with it.
    response = await coordinator.search("fix the parseTests helper", limit=5)
    assert _files(response) == ["src/other.py", "src/runner.py"]


# ---------------------------------------------------------------------------
# Dedupe keeps the item that can show its lines
# ---------------------------------------------------------------------------


async def test_a_chunk_takes_a_pages_slot_without_taking_its_rank(tmp_path):
    """One file to open; serve the item that can point at the lines."""
    page = _PageHit("file_page:src/a.py", "src/a.py", 0.90)
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[page],
        wiki_lexical=[page],
        source_dense=[
            _hit("thing", "src/a.py", 0.50),
            _hit("other", "src/b.py", 0.80),
        ],
    )
    response = await coordinator.search("how the thing works", limit=5)
    # src/a.py keeps the rank its page earned, and arrives with line bounds.
    assert _files(response) == ["src/a.py", "src/b.py"]
    top = response["results"][0]
    assert top["evidence"]["lane"] == "source"
    assert (top["start_line"], top["end_line"]) == (1, 9)


async def test_the_owner_is_chosen_before_the_citation_is_upgraded(tmp_path):
    """Upgrading first would hide that a file is page-backed, and the owner
    policy's preference for a chunk over a page would silently stop firing."""
    coordinator = _coordinator(
        tmp_path,
        # The page ranks first, but only a page names src/page.py.
        wiki_dense=[_PageHit("file_page:src/page.py", "src/page.py", 0.90)],
        source_dense=[
            _hit("real", "src/chunk.py", 0.80, kind="function"),
            _hit("aside", "src/page.py", 0.10),
        ],
    )
    response = await coordinator.search("how the thing works", limit=5)
    # The chunk-backed file wins the owner, because at the moment that was
    # decided src/page.py was still visibly a page.
    assert response["selected_owner"]["file"] == "src/chunk.py"
    assert "preferred over a closely-scored wiki page" in response["selected_owner"]["reason"]
    # And src/page.py is still re-cited onto the lines it does have.
    served = {item["file"]: item for item in response["results"]}
    assert served["src/page.py"]["evidence"]["lane"] == "source"
    assert "start_line" in served["src/page.py"]


async def test_the_page_keeps_its_slot_when_no_chunk_names_the_file(tmp_path):
    """Nothing to upgrade to: the page is the only thing that names src/a.py."""
    page = _PageHit("file_page:src/a.py", "src/a.py", 0.90)
    coordinator = _coordinator(
        tmp_path,
        # Both legs, so the owner policy's shape rules cannot promote the
        # unrelated chunk over it and the dedupe is what the test is watching.
        wiki_dense=[page],
        wiki_lexical=[page],
        source_dense=[_hit("other", "src/b.py", 0.50)],
    )
    response = await coordinator.search("how the thing works", limit=5)
    assert _files(response) == ["src/a.py", "src/b.py"]
    assert response["results"][0]["evidence"]["lane"] == "wiki"
    assert "start_line" not in response["results"][0]


async def test_lines_are_never_bought_with_name_evidence(tmp_path):
    """An exact-name page outranks a chunk that matched nothing in its file."""
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[
            _PageHit(
                "symbol_spotlight:src/a.py::pkg.merged_with",
                "src/a.py::pkg.merged_with",
                0.90,
                page_type="symbol_spotlight",
                title="Symbol: pkg.merged_with",
            )
        ],
        source_dense=[_hit("unrelated_helper", "src/a.py", 0.50)],
    )
    response = await coordinator.search("merged_with", limit=5)
    top = response["results"][0]
    assert top["file"] == "src/a.py"
    assert top["evidence"]["lane"] == "wiki"
    assert top["evidence"]["exact_name"] is True


async def test_a_short_trailing_segment_is_not_a_name(tmp_path):
    """``refresh.py`` must not make every symbol called ``py`` an exact match."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("other", "src/other.py", 0.90),
            _hit("py", "src/py.py", 0.10),
        ],
    )
    response = await coordinator.search("fix the refresh.py fingerprint check", limit=5)
    assert _files(response) == ["src/other.py", "src/py.py"]
    assert all(item["evidence"]["exact_name"] is False for item in response["results"])


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


async def test_prose_naming_no_identifier_leaves_the_fusion_alone(tmp_path):
    """Plain words are a topic. Neither router fires; fusion order stands."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("other", "src/other.py", 0.9),
            _hit("target", "src/target.py", 0.1),
        ],
    )
    response = await coordinator.search("how the vault refresh skips work", limit=5)
    assert _files(response) == ["src/other.py", "src/target.py"]
    assert all(item["evidence"]["exact_name"] is False for item in response["results"])


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
    assert response["selected_owner"]["file"] == "src/a.py"
    assert response["selected_owner"]["reason"] == "dense+lexical agreement"
    assert response["selected_owner"]["evidence"]["concept_coverage"] == 1.0


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


async def test_exact_file_qualified_id_owns_only_its_named_path(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit(
                "inner",
                "src/other.py",
                0.90,
                chunk_id="src/other.py::Outer::inner",
            ),
            _hit(
                "inner",
                "src/target.py",
                0.80,
                chunk_id="src/target.py::Outer::inner",
            ),
        ],
    )

    response = await coordinator.search("src/target.py::Outer::inner", limit=5)

    assert response["selected_owner"]["file"] == "src/target.py"
    assert response["selected_owner"]["reason"].startswith("owner policy: exact path/full-ID owner")


@pytest.mark.parametrize(
    "query",
    ["canonicalize_workspace_copy_path", "query_wants_tests", "build_symbol_uid"],
)
def test_bare_identifiers_are_not_misclassified_as_exact_paths(query):
    intent = _query_intent(query)

    assert intent.identifier == query
    assert intent.exact_target is None


async def test_exact_wiki_declaration_proxy_survives_source_declaration_policy(tmp_path):
    page = _PageHit(
        "symbol_spotlight:src/target.py::canonical_name",
        "src/target.py::canonical_name",
        0.90,
        page_type="symbol_spotlight",
        title="Symbol: canonical_name",
    )
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[page],
        source_dense=[
            _hit("usage", "src/usage.py", 0.80, snippet="canonical_name()"),
            _hit("canonical_name", "src/target.py", 0.70),
        ],
    )

    response = await coordinator.search("canonical_name", limit=5)

    assert response["selected_owner"]["file"] == "src/target.py"
    assert response["selected_owner"]["reason"] == "exact name match"
    assert response["results"][0]["file"] == "src/target.py"
    assert response["results"][0]["evidence"]["lane"] == "source"


async def test_filename_like_prose_does_not_claim_exact_path_ownership(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("routing", "src/current.py", 0.90),
            _hit("config", "src/config.py", 0.80),
        ],
    )

    response = await coordinator.search("how config.py controls routing", limit=5)

    assert "exact path/full-ID owner" not in response["selected_owner"]["reason"]


async def test_explicit_test_discovery_selects_a_close_test_owner(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("worker", "src/worker.py", 0.90),
            _hit("worker", "tests/test_worker.py", 0.80, is_test=True),
        ],
    )

    response = await coordinator.search("where are the worker fixtures", limit=5)

    assert response["selected_owner"]["file"] == "tests/test_worker.py"
    assert response["selected_owner"]["reason"].startswith("owner policy: explicit test owner")


async def test_repair_intent_treats_a_close_failing_test_as_the_symptom(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("worker", "tests/test_worker.py", 0.90, is_test=True),
            _hit("worker", "src/worker.py", 0.80),
        ],
    )

    response = await coordinator.search("fix the failing worker test", limit=5)

    assert response["selected_owner"]["file"] == "src/worker.py"
    assert response["selected_owner"]["reason"].startswith("owner policy: symptom test demotion")


async def test_complete_subject_evidence_beats_a_close_single_token_neighbor(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("neo4j_writer", "src/graph.py", 0.90),
            _hit("neo4j_cashflow", "src/combined.py", 0.80),
        ],
        source_term_files={
            "neo4j": {"src/graph.py", "src/combined.py"},
            "cashflow": {"src/combined.py"},
        },
    )

    response = await coordinator.search("how neo4j cashflow behaves", limit=5)

    assert response["selected_owner"]["file"] == "src/combined.py"
    assert response["selected_owner"]["reason"].startswith(
        "owner policy: co-located subject completeness"
    )


async def test_embedded_identifier_bypasses_prose_completeness(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("walk_notes", "src/complete.py", 0.90),
            _hit("walk_vault", "src/declaration.py", 0.80),
        ],
        source_term_files={
            "walk": {"src/complete.py", "src/declaration.py"},
            "vault": {"src/complete.py", "src/declaration.py"},
            "underscore": {"src/complete.py"},
            "directory": {"src/complete.py"},
        },
    )

    response = await coordinator.search("fix the walk_vault underscore directory test", limit=5)

    assert response["selected_owner"]["file"] == "src/declaration.py"
    assert "co-located subject completeness" not in response["selected_owner"]["reason"]


async def test_test_subject_bypasses_prose_completeness_between_test_files(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("test_pack_spans", "tests/test_packing.py", 0.90, is_test=True),
            _hit("test_other_exhaustion", "tests/test_other.py", 0.80, is_test=True),
        ],
        source_term_files={
            "pack": {"tests/test_packing.py", "tests/test_other.py"},
            "spans": {"tests/test_packing.py", "tests/test_other.py"},
            "budget": {"tests/test_packing.py", "tests/test_other.py"},
            "exhaustion": {"tests/test_other.py"},
        },
    )

    response = await coordinator.search("show the tests for pack spans budget exhaustion", limit=5)

    assert response["selected_owner"]["file"] == "tests/test_packing.py"
    assert "co-located subject completeness" not in response["selected_owner"]["reason"]


async def test_implementation_intent_prefers_close_source_over_generated_prose(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[_hit("alpha", "src/alpha.py", 0.80)],
        wiki_dense=[
            _PageHit(
                "file_page:docs/alpha.md",
                "docs/alpha.md",
                0.90,
                snippet="alpha",
            )
        ],
        source_term_files={"alpha": {"src/alpha.py", "docs/alpha.md"}},
    )

    response = await coordinator.search("how is alpha implemented", limit=5)

    assert response["selected_owner"]["file"] == "src/alpha.py"
    assert response["selected_owner"]["reason"].startswith("owner policy: source-owner bias")


async def test_explicit_docs_intent_does_not_apply_source_owner_bias(tmp_path):
    page = _PageHit(
        "file_page:docs/alpha.md",
        "docs/alpha.md",
        0.90,
        snippet="alpha",
    )
    coordinator = _coordinator(
        tmp_path,
        source_dense=[_hit("alpha", "src/alpha.py", 0.80)],
        wiki_dense=[page],
        source_term_files={"alpha": {"src/alpha.py", "docs/alpha.md"}},
    )

    response = await coordinator.search("where is alpha documented", limit=5)

    assert response["selected_owner"]["file"] == "docs/alpha.md"
    assert "source-owner bias" not in response["selected_owner"]["reason"]


async def test_declaration_owns_an_implementation_query_over_close_usage(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("request_usage", "src/caller.py", 0.90, kind="reference"),
            _hit("request_router", "src/router.py", 0.80, kind="function"),
        ],
        source_term_files={
            "request": {"src/caller.py", "src/router.py"},
            "routing": {"src/caller.py", "src/router.py"},
        },
    )

    response = await coordinator.search("where request routing is implemented", limit=5)

    assert response["selected_owner"]["file"] == "src/router.py"
    assert response["selected_owner"]["reason"].startswith("owner policy: declaration over usage")


async def test_operational_intent_preserves_a_close_file_window_owner(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("healthcheck", "src/health.py", 0.90, kind="function"),
            _hit(
                "compose.yaml",
                "deploy/compose.yaml",
                0.80,
                kind="file_window",
                source="file_window",
            ),
        ],
        source_term_files={
            "compose": {"src/health.py", "deploy/compose.yaml"},
            "healthcheck": {"src/health.py", "deploy/compose.yaml"},
        },
    )

    response = await coordinator.search("where compose service healthcheck is configured", limit=5)

    assert response["selected_owner"]["file"] == "deploy/compose.yaml"
    assert response["selected_owner"]["reason"].startswith(
        "owner policy: operational file preservation"
    )


async def test_bare_code_identifier_does_not_trigger_operational_file_policy(tmp_path):
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit(
                "compose.yaml",
                "deploy/compose.yaml",
                0.90,
                kind="file_window",
                source="file_window",
            ),
            _hit("healthcheck", "src/health.py", 0.80, kind="function"),
        ],
    )

    response = await coordinator.search("healthcheck", limit=5)

    assert response["selected_owner"]["file"] == "src/health.py"
    assert "operational file preservation" not in response["selected_owner"]["reason"]


def test_true_owner_evidence_ties_use_path_line_and_stable_key() -> None:
    later = _Item(
        key="source:z",
        lane="source",
        file="src/z.py",
        name="topic",
        kind="reference",
        snippet="topic",
        source="symbol",
        fused_score=0.01,
    )
    earlier = _Item(
        key="source:a",
        lane="source",
        file="src/a.py",
        name="topic",
        kind="reference",
        snippet="topic",
        source="symbol",
        fused_score=0.01,
    )

    owner, reason = SourceSearchCoordinator._select_owner(
        [later, earlier], intent=_query_intent("ordinary topic")
    )

    assert owner is earlier
    assert reason.startswith("owner policy: deterministic tie")


async def test_no_results_means_no_owner(tmp_path):
    coordinator = _coordinator(tmp_path)
    response = await coordinator.search("nothing at all like this", limit=5)
    assert response["results"] == []
    assert response["selected_owner"] is None


# ---------------------------------------------------------------------------
# Confidence — absolute thresholds, no window normalisation
# ---------------------------------------------------------------------------


async def test_a_strong_cosine_alone_is_only_caution(tmp_path):
    coordinator = _coordinator(
        tmp_path, source_dense=[_hit("alpha", "src/a.py", CONFIDENT_DENSE_COSINE)]
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["results"][0]["evidence"]["concept_coverage"] == 1.0
    assert response["confidence"] == "caution"


async def test_just_under_the_threshold_is_caution(tmp_path):
    coordinator = _coordinator(
        tmp_path, source_dense=[_hit("alpha", "src/a.py", CONFIDENT_DENSE_COSINE - 0.01)]
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["confidence"] == "caution"


async def test_strong_dense_complete_concepts_and_own_lexical_evidence_are_confident(tmp_path):
    hit = _hit("alpha", "src/a.py", CONFIDENT_DENSE_COSINE)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[hit],
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path)],
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["results"][0]["evidence"]["concept_coverage"] == 1.0
    assert response["confidence"] == "confident"


async def test_middling_dense_and_lexical_without_cross_corpus_corroboration_is_caution(
    tmp_path,
):
    hit = _hit("alpha", "src/a.py", AGREEMENT_DENSE_COSINE)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[hit],
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path)],
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["confidence"] == "caution"


async def test_middling_dense_complete_concepts_and_same_path_corroboration_are_confident(
    tmp_path,
):
    hit = _hit("alpha", "src/a.py", AGREEMENT_DENSE_COSINE)
    page = _PageHit("file_page:src/a.py", "src/a.py", AGREEMENT_DENSE_COSINE - 0.01)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[hit],
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path)],
        wiki_dense=[page],
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["results"][0]["evidence"]["same_path_corroborated"] is True
    assert response["confidence"] == "confident"


async def test_vocabulary_split_across_files_cannot_be_pooled_into_confidence(tmp_path):
    owner = _hit("neo4j_writer", "src/graph.py", 0.60)
    neighbour = _hit("cashflow_projection", "src/finance.py", 0.55)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[owner, neighbour],
        source_lexical=[_FTSHit(owner.chunk_id, owner.file_path)],
        source_term_files={
            "neo4j": {"src/graph.py"},
            "cashflow": {"src/finance.py"},
        },
    )

    response = await coordinator.search("neo4j cashflow", limit=5)

    assert response["selected_owner"]["file"] == "src/graph.py"
    assert response["results"][0]["evidence"]["concept_coverage"] == 0.5
    assert response["confidence"] == "caution"


async def test_complete_subject_concepts_on_one_owner_can_be_confident(tmp_path):
    owner = _hit("neo4j_cashflow", "src/combined.py", 0.60)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[owner],
        source_lexical=[_FTSHit(owner.chunk_id, owner.file_path)],
        source_term_files={
            "neo4j": {"src/combined.py"},
            "cashflow": {"src/combined.py"},
        },
    )

    response = await coordinator.search("neo4j cashflow", limit=5)

    assert response["results"][0]["evidence"]["concept_coverage"] == 1.0
    assert response["confidence"] == "confident"


async def test_a_file_named_for_the_queried_role_owns_that_role(tmp_path):
    """`page.tsx` beats a component that merely reads more like the word "page".

    Measured on SoleMD.Graph: "which page renders the map view" selected
    WikiPageView.tsx over app/map/page.tsx, with the right answer already
    retrieved at rank 2. Dense and lexical evidence cannot express this — by
    every textual measure WikiPageView looks *more* like "page" than page.tsx
    does — so it has to be an owner-selection rule.
    """
    named = _hit("Page", "app/map/page.tsx", 0.61)
    lookalike = _hit("WikiPageView", "features/wiki/WikiPageView.tsx", 0.62)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[lookalike, named],
        source_lexical=[
            _FTSHit(lookalike.chunk_id, lookalike.file_path),
            _FTSHit(named.chunk_id, named.file_path),
        ],
        source_term_files={"page": {"app/map/page.tsx", "features/wiki/WikiPageView.tsx"}},
    )

    response = await coordinator.search("which page renders the map view", limit=5)

    assert response["selected_owner"]["file"] == "app/map/page.tsx"


async def test_the_role_rule_is_inert_when_the_query_names_no_role(tmp_path):
    """Most queries name no role, and those must be decided exactly as before."""
    a = _hit("renderMap", "features/map/render.ts", 0.62)
    b = _hit("Page", "app/map/page.tsx", 0.61)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[a, b],
        source_lexical=[_FTSHit(a.chunk_id, a.file_path), _FTSHit(b.chunk_id, b.file_path)],
        source_term_files={"render": {"features/map/render.ts"}},
    )

    response = await coordinator.search("what renders the map", limit=5)

    assert response["selected_owner"]["file"] == "features/map/render.ts"


async def test_the_role_rule_is_inert_when_no_candidate_carries_the_role_name(tmp_path):
    """A repo with no file named for the role is unaffected rather than distorted."""
    a = _hit("handleSearch", "server/search.ts", 0.62)
    b = _hit("SearchBox", "ui/SearchBox.tsx", 0.61)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[a, b],
        source_lexical=[_FTSHit(a.chunk_id, a.file_path), _FTSHit(b.chunk_id, b.file_path)],
        source_term_files={"search": {"server/search.ts", "ui/SearchBox.tsx"}},
    )

    response = await coordinator.search("which page runs the search", limit=5)

    assert response["selected_owner"]["file"] == "server/search.ts"


def test_the_role_stem_strips_every_extension():
    """`route.test.ts` is still named for the role `route`.

    Whether a test file may own the answer is the test-demotion stage's
    decision, not something to smuggle in by leaving `.test` on the stem.
    """
    from repowise.core.source_search.coordinator import _basename_stem

    assert _basename_stem("app/api/x/route.ts") == "route"
    assert _basename_stem("app/api/x/route.test.ts") == "route"
    assert _basename_stem("Page.TSX") == "page"
    assert _basename_stem("noslash") == "noslash"


async def test_a_concept_supplied_only_by_the_filename_cannot_make_a_symbol_confident(
    tmp_path,
):
    """A thin symbol inherits its file's name, and that is not evidence about it.

    Every chunk opens with ``# File: <path>`` and a dotted qualified name that
    restates the path, so a two-line helper in ``…cashflow_chart_controller.js``
    matches ``cashflow`` while containing nothing of the sort.  Before A3 the
    corpus had no chunks thin enough for that header to outweigh a body; A3 added
    544 local symbols and one of them was confidently named the owner of an
    absent subject on exactly this evidence.  Retrieval may still rank it — a
    cashflow file is a fair guess for a cashflow query — but the confident claim
    is about the chunk, and the chunk cannot show it.
    """
    owner = _hit("line", "src/cashflow_chart_controller.js", 0.60)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[owner],
        source_lexical=[_FTSHit(owner.chunk_id, owner.file_path)],
        source_term_files={"cashflow": {"src/cashflow_chart_controller.js"}},
    )

    response = await coordinator.search("cashflow", limit=5)

    evidence = response["results"][0]["evidence"]
    assert evidence["concept_coverage"] == 1.0, "ranking still believes the filename"
    assert evidence["content_concept_coverage"] == 0.0, "the chunk itself carries nothing"
    assert response["confidence"] == "caution"


async def test_a_file_window_may_be_confident_from_its_own_path(tmp_path):
    """A file window *is* its file, so its path is the subject, not an accident.

    The guard above must not cost the operational-coverage family, where the
    answer genuinely is "that file" and the filename is how such a file declares
    itself.
    """
    owner = _hit(
        "compose.tailnet.yaml",
        "infra/compose.tailnet.yaml",
        0.60,
        source="file_window",
    )
    coordinator = _coordinator(
        tmp_path,
        source_dense=[owner],
        source_lexical=[_FTSHit(owner.chunk_id, owner.file_path)],
        source_term_files={"tailnet": {"infra/compose.tailnet.yaml"}},
    )

    response = await coordinator.search("tailnet", limit=5)

    assert response["results"][0]["evidence"]["content_concept_coverage"] == 1.0
    assert response["confidence"] == "confident"


async def test_a_symbol_named_for_its_concept_keeps_its_confidence(tmp_path):
    """The guard tests where the concept lives, not whether the path mentions it."""
    owner = _hit("cashflow_projection", "src/cashflow.py", 0.60)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[owner],
        source_lexical=[_FTSHit(owner.chunk_id, owner.file_path)],
        source_term_files={"cashflow": {"src/cashflow.py"}},
    )

    response = await coordinator.search("cashflow", limit=5)

    assert response["results"][0]["evidence"]["content_concept_coverage"] == 1.0
    assert response["confidence"] == "confident"


async def test_a_different_candidates_strong_cosine_cannot_make_the_owner_confident(tmp_path):
    """The file named as owner must carry the evidence behind the claim."""
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("symptom", "tests/test_symptom.py", 0.90, is_test=True),
            _hit("implementation", "src/implementation.py", 0.35),
        ],
    )

    response = await coordinator.search("fix the failing symptom test", limit=5)

    assert response["selected_owner"]["file"] == "src/implementation.py"
    assert response["results"][1]["evidence"]["dense_cosine"] == 0.9
    assert response["results"][0]["evidence"]["dense_cosine"] == 0.35
    assert response["confidence"] == "caution"


async def test_symptom_demotion_cannot_cross_the_owner_score_band(tmp_path):
    """A two-leg test that decisively wins retrieval stays the owner."""
    lexical = _hit("symptom", "tests/test_symptom.py", 0.0, is_test=True)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[
            _hit("implementation", "src/implementation.py", AGREEMENT_DENSE_COSINE),
            _hit("symptom", "tests/test_symptom.py", 0.20, is_test=True),
        ],
        source_lexical=[_FTSHit(lexical.chunk_id, lexical.file_path)],
    )

    response = await coordinator.search("fix the failing symptom test", limit=5)

    assert response["selected_owner"]["file"] == "tests/test_symptom.py"
    implementation = next(
        result for result in response["results"] if result["file"] == "src/implementation.py"
    )
    assert implementation["evidence"]["lexical_rank"] is None
    assert "symptom test demotion" not in response["selected_owner"]["reason"]
    assert response["confidence"] == "caution"


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
        source_term_files={"page": {"src/page.py"}, "thing": {"src/page.py"}},
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
        source_term_files={"page": {"src/page.py"}, "thing": {"src/page.py"}},
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
    assert set(result["evidence"]) == {
        "dense_cosine",
        "lexical_rank",
        "exact_name",
        "lane",
        "concept_coverage",
        "content_concept_coverage",
        "corpus_file_count",
        "same_path_corroborated",
        "concepts",
    }
    assert response["candidates"] == [{"path": "src/a.py"}]


async def test_a_wiki_result_carries_the_page_id_it_was_retrieved_under(tmp_path):
    """A page that names no file has nothing else to identify it by.

    ``module_page``/``scc_page``/``repo_overview`` targets are curated ids, not
    paths, so those results serve ``file: ""`` by design. Without the page id a
    consumer holds a title it cannot resolve to anything.
    """
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[
            _PageHit(
                "module_page:src/pkg",
                "src/pkg",
                0.90,
                page_type="module_page",
                title="Module: src/pkg",
            ),
        ],
    )
    response = await coordinator.search("how the package fits together", limit=5)
    result = response["results"][0]
    assert result["page_id"] == "module_page:src/pkg"
    assert result["file"] == ""


async def test_a_symbol_spotlight_keeps_its_qualified_id_intact(tmp_path):
    """Carried verbatim, never rebuilt.

    ``f"{kind}:{file}"`` would yield ``symbol_spotlight:src/a.py`` here — a
    different, perfectly well-formed id belonging to another page — because
    this response serves the *file* a spotlight names, not its ``::``-qualified
    target.
    """
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[
            _PageHit(
                "symbol_spotlight:src/a.py::Foo",
                "src/a.py::Foo",
                0.90,
                page_type="symbol_spotlight",
                title="Symbol: pkg.mod.Foo",
            ),
        ],
    )
    response = await coordinator.search("how the thing works", limit=5)
    result = response["results"][0]
    assert result["page_id"] == "symbol_spotlight:src/a.py::Foo"
    assert result["file"] == "src/a.py"
    assert f"{result['kind']}:{result['file']}" != result["page_id"]


async def test_a_source_result_has_no_page_id_key_at_all(tmp_path):
    """Absent, not null: a chunk is identified by its file and its lines."""
    coordinator = _coordinator(tmp_path, source_dense=[_hit("alpha", "src/a.py", 0.6)])
    response = await coordinator.search("how alpha works", limit=5)
    result = response["results"][0]
    assert "page_id" not in result
    assert (result["file"], result["start_line"], result["end_line"]) == ("src/a.py", 1, 9)


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
# A store that is broken, not slow
# ---------------------------------------------------------------------------


class _SchemaError(Exception):
    """Stands in for ``LanceError(Schema)``: raised on every read, forever."""


class _BrokenSourceVectors:
    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or _SchemaError("Schema error: no such column: vector")

    async def search_by_vector(self, vector: Any, limit: int = 20) -> list:
        raise self._exc

    async def fetch_by_chunk_ids(self, chunk_ids: Any) -> dict:
        raise self._exc


class _BrokenSourceFTS:
    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or _SchemaError("no such table: source_fts")

    def query(self, match: str, limit: int = 20) -> list:
        raise self._exc


class _SlowSourceVectors:
    async def search_by_vector(self, vector: Any, limit: int = 20) -> list:
        raise TimeoutError("store did not answer in time")

    async def fetch_by_chunk_ids(self, chunk_ids: Any) -> dict:
        return {}


async def test_a_broken_source_store_is_reported_not_swallowed(tmp_path, caplog):
    """The incident: a schema error read as an honest empty source lane.

    The wiki lane still answers, so the response is real — but it must say the
    source corpus was not read, and it must not sound sure of itself.
    """
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[_PageHit("file_page:src/a.py", "src/a.py", 0.95)],
    )
    coordinator._source_vectors = _BrokenSourceVectors()
    coordinator._source_fts = _BrokenSourceFTS()

    with caplog.at_level(logging.ERROR, logger="repowise.core.source_search.coordinator"):
        response = await coordinator.search("how the thing works", limit=5)

    source_meta = response["_meta"]["source_search"]
    assert source_meta["degraded"] is True
    assert "source dense" in source_meta["degraded_reason"]
    assert "source lexical" in source_meta["degraded_reason"]
    assert "_SchemaError" in source_meta["degraded_reason"]
    assert {f["leg"] for f in source_meta["failed_legs"]} == {"source dense", "source lexical"}

    # The healthy lane still answers.
    assert _files(response) == ["src/a.py"]
    assert "status" not in response

    # A 0.95 cosine would otherwise be confidently correct.
    assert response["confidence"] == "caution"

    # One structured record per failed leg.
    records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert {getattr(r, "leg", None) for r in records} >= {"source dense", "source lexical"}


async def test_a_slow_store_is_still_just_swallowed(tmp_path):
    """A timeout is a thinner answer, not a broken one — no degraded flag.

    The source lexical leg still names the file, but its evidence belongs to a
    different candidate than the dense wiki owner.  The timeout is not reported
    as a durable degradation; the owner-bound profile independently keeps the
    thinner answer at caution.
    """
    coordinator = _coordinator(
        tmp_path,
        wiki_dense=[_PageHit("file_page:src/a.py", "src/a.py", 0.95)],
        source_lexical=[_FTSHit("src/a.py::thing", "src/a.py")],
    )
    coordinator._source_vectors = _SlowSourceVectors()

    response = await coordinator.search("how the thing works", limit=5)
    source_meta = response["_meta"]["source_search"]
    assert "degraded" not in source_meta
    assert "failed_legs" not in source_meta
    assert response["confidence"] == "caution"
    assert _files(response) == ["src/a.py"]


async def test_a_hard_failure_never_lets_the_answer_claim_an_absence(tmp_path):
    """``no_match`` asserts the corpus has nothing. A half-read corpus cannot."""
    coordinator = _coordinator(tmp_path, wiki_dense=[])
    coordinator._source_vectors = _BrokenSourceVectors()
    coordinator._source_fts = _BrokenSourceFTS()

    response = await coordinator.search("croissant lamination fold schedule", limit=5)
    assert response["results"] == []
    assert response["confidence"] == "caution"
    assert response["_meta"]["source_search"]["degraded"] is True


async def test_every_leg_failing_returns_an_error_not_an_empty_answer(tmp_path, caplog):
    """Zero results would be a claim about the repository. Nothing was read."""

    class _BrokenWikiVectors:
        async def search_by_vector(self, vector: Any, limit: int = 10) -> list:
            raise OSError("lancedb: input/output error")

    class _BrokenWikiFTS:
        async def search(self, query: str, limit: int = 10) -> list:
            raise _SchemaError("no such table: page_fts")

    coordinator = _coordinator(tmp_path)
    coordinator._source_vectors = _BrokenSourceVectors()
    coordinator._source_fts = _BrokenSourceFTS()
    coordinator._wiki_vectors = _BrokenWikiVectors()
    coordinator._wiki_fts = _BrokenWikiFTS()

    with caplog.at_level(logging.ERROR, logger="repowise.core.source_search.coordinator"):
        response = await coordinator.search("how the thing works", limit=5)

    assert response["status"] == "error"
    assert response["error"]["code"] == "source_search_unavailable"
    assert {f["leg"] for f in response["error"]["failed_legs"]} == {
        "source dense",
        "source lexical",
        "wiki dense",
        "wiki lexical",
    }
    assert "NOT" in response["error"]["message"]  # not because nothing matched
    assert response["results"] == []
    assert response["selected_owner"] is None
    assert response["confidence"] == "caution"  # never no_match: nothing was read
    assert response["_meta"]["source_search"]["degraded"] is True
    assert any("every retrieval leg failed" in r.getMessage() for r in caplog.records)


async def test_a_missing_wiki_index_is_not_a_failure(tmp_path):
    """No store is a configuration; it must not count towards "all legs lost"."""
    coordinator = SourceSearchCoordinator(
        repo_path=tmp_path,
        embedder=_Embedder(),
        source_vectors=_SourceVectors([_hit("alpha", "src/a.py", 0.6)]),  # type: ignore[arg-type]
        source_fts=_SourceFTS([]),  # type: ignore[arg-type]
        wiki_vectors=None,
        wiki_fts=None,
        query_log=QueryLog(tmp_path / "log.jsonl"),
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert "status" not in response
    assert "degraded" not in response["_meta"]["source_search"]
    assert _files(response) == ["src/a.py"]


async def test_a_broken_source_store_with_no_wiki_at_all_is_an_error(tmp_path):
    """Both of this deployment's legs are the source ones, and both are down."""
    coordinator = SourceSearchCoordinator(
        repo_path=tmp_path,
        embedder=_Embedder(),
        source_vectors=_BrokenSourceVectors(),  # type: ignore[arg-type]
        source_fts=_BrokenSourceFTS(),  # type: ignore[arg-type]
        wiki_vectors=None,
        wiki_fts=None,
        query_log=QueryLog(tmp_path / "log.jsonl"),
    )
    response = await coordinator.search("how alpha works", limit=5)
    assert response["status"] == "error"
    assert response["error"]["code"] == "source_search_unavailable"


async def test_a_dead_embedder_takes_both_dense_legs_down_by_name(tmp_path):
    """The legs are what failed, whatever the cause — and the cause is kept."""

    class _DeadEmbedder:
        dimensions = 4

        async def embed(self, texts: list[str]) -> list[list[float]]:
            raise ConnectionError("ollama: connection refused")

    coordinator = _coordinator(tmp_path, embedder=_DeadEmbedder())
    coordinator._wiki_fts = _WikiFTS([_PageHit("file_page:src/a.py", "src/a.py", 4.0)])

    response = await coordinator.search("how the thing works", limit=5)
    source_meta = response["_meta"]["source_search"]
    assert {f["leg"] for f in source_meta["failed_legs"]} == {"source dense", "wiki dense"}
    assert "query embedding failed" in source_meta["degraded_reason"]
    assert "connection refused" in source_meta["degraded_reason"]
    # The lexical lane still answered, so this is a degraded answer, not an error.
    assert "status" not in response
    assert _files(response) == ["src/a.py"]
    assert response["confidence"] == "caution"


async def test_losing_only_the_chunk_metadata_lookup_degrades_without_erroring(tmp_path):
    """It hydrates evidence; it is not a lane, so the corpus is still read."""
    lexical_only = _hit("solo", "src/solo.py", 0.0)

    class _HalfBroken(_SourceVectors):
        async def fetch_by_chunk_ids(self, chunk_ids: Any) -> dict:
            raise _SchemaError("Schema error: no such column: snippet")

    coordinator = _coordinator(
        tmp_path,
        source_lexical=[_FTSHit(lexical_only.chunk_id, lexical_only.file_path)],
    )
    coordinator._source_vectors = _HalfBroken([])
    response = await coordinator.search("solo behaviour", limit=5)

    source_meta = response["_meta"]["source_search"]
    assert source_meta["degraded"] is True
    assert [f["leg"] for f in source_meta["failed_legs"]] == ["source chunk metadata"]
    assert "status" not in response
    assert response["confidence"] == "caution"


async def test_a_healthy_search_says_nothing_about_degradation(tmp_path):
    """The healthy envelope is byte-identical to the pre-degradation one."""
    hit = _hit("alpha", "src/a.py", 0.6)
    coordinator = _coordinator(
        tmp_path,
        source_dense=[hit],
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path)],
    )
    response = await coordinator.search("how alpha works", limit=5)
    source_meta = response["_meta"]["source_search"]
    # The absence of these is the claim, not the exact key set: a repository
    # with a manifest also carries its chunk counts here, and asserting the
    # whole shape would fail for a reason that has nothing to do with health.
    assert "degraded" not in source_meta
    assert "failed_legs" not in source_meta
    assert "degraded_reason" not in source_meta
    assert "status" not in response
    assert "error" not in response
    assert response["confidence"] == "confident"


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
    assert event["top"][0]["concept_coverage"] == 1.0
    assert event["top"][0]["same_path_corroborated"] is False
    assert event["selected_owner_evidence"] == {
        "dense_cosine": 0.6,
        "lexical_rank": 1,
        "exact_name": False,
        "lane": "source",
        "concept_coverage": 1.0,
        "content_concept_coverage": 1.0,
        "corpus_file_count": 1,
        "same_path_corroborated": False,
        "concepts": [
            {
                "token": "alpha",
                "document_frequency": 1,
                "matched": True,
                "content_carried": True,
            }
        ],
    }


async def test_a_log_that_cannot_be_written_does_not_fail_the_search(tmp_path):
    """A search that succeeded and then failed to record itself has succeeded."""
    coordinator = _coordinator(tmp_path, source_dense=[_hit("alpha", "src/a.py", 0.6)])
    # A directory where the file should be: every open() for append raises.
    (tmp_path / "log.jsonl").mkdir()
    response = await coordinator.search("how alpha works", limit=5)
    assert response["results"]


# ---------------------------------------------------------------------------
# Naming the symbol a row actually lands in
# ---------------------------------------------------------------------------
#
# A nested definition is indexed as its own chunk with a full ``::`` chain, so
# on a bare-identifier query it is promoted and served under its own name. On a
# conceptual query it is not: the enclosing chunk wins the fusion, per-file
# deduplication keeps one row, and the helper that made the file relevant
# disappears from the answer. These lock the naming that puts it back — and
# that it is only naming, with nothing kept or ordered differently.
#
# The fixture shapes are drawn from real cases: a helper closure inside a
# ranking function, a local function inside a builder component, and a guard
# closure inside a reducer.


def _nested_pair(
    path: str,
    outer: str,
    inner: str,
    *,
    outer_span: tuple[int, int] = (100, 140),
    inner_span: tuple[int, int] = (110, 118),
    outer_score: float = 0.72,
    inner_score: float = 0.55,
) -> tuple[SourceChunkHit, SourceChunkHit]:
    """An enclosing symbol chunk and the nested one defined inside it.

    Scored so the enclosing chunk wins the fusion, which is the conceptual-query
    shape this section is about: both are retrieved, one survives dedupe.
    """
    enclosing = _hit(
        outer,
        path,
        outer_score,
        chunk_id=f"{path}::{outer}",
        start_line=outer_span[0],
        end_line=outer_span[1],
        snippet=f"def {outer}(...):\n    def {inner}(...): ...",
    )
    nested = _hit(
        inner,
        path,
        inner_score,
        chunk_id=f"{path}::{outer}::{inner}",
        start_line=inner_span[0],
        end_line=inner_span[1],
        snippet=f"def {inner}(...): ...",
    )
    return enclosing, nested


def _row_for(response: dict, path: str) -> dict:
    for row in response["results"]:
        if row["file"] == path:
            return row
    raise AssertionError(f"no served row for {path}: {_files(response)}")


async def test_a_symbol_row_names_the_symbol_it_is(tmp_path):
    """``symbol_path`` carries the chain, so a nested row says which one it is."""
    path = "codeatlas/code_search/diversify.py"
    enclosing, nested = _nested_pair(path, "mmr_diversify", "_sim")
    coordinator = _coordinator(
        tmp_path,
        source_dense=[nested, enclosing],
        records={hit.chunk_id: _record(hit) for hit in (enclosing, nested)},
    )

    response = await coordinator.search("_sim", limit=5)

    row = _row_for(response, path)
    assert row["symbol_path"] == "mmr_diversify::_sim"
    # The row *is* the helper, so it contains nothing further down.
    assert "contains_symbols" not in row


async def test_an_enclosing_row_names_the_nested_rival_it_displaced(tmp_path):
    """The helper closure shape: `_sim` inside `mmr_diversify`, conceptual query.

    Both chunks are retrieved and both are for one file, so deduplication keeps
    the enclosing one and drops the helper. The served row has to say the
    helper is in there.
    """
    path = "codeatlas/code_search/diversify.py"
    enclosing, nested = _nested_pair(path, "mmr_diversify", "_sim")
    coordinator = _coordinator(
        tmp_path,
        source_dense=[enclosing, nested],
        source_lexical=[_FTSHit(enclosing.chunk_id, path), _FTSHit(nested.chunk_id, path)],
        records={hit.chunk_id: _record(hit) for hit in (enclosing, nested)},
    )

    response = await coordinator.search(
        "how are ranked results diversified to avoid near duplicates", limit=5
    )

    row = _row_for(response, path)
    assert row["symbol_path"] == "mmr_diversify"
    assert row["contains_symbols"] == ["mmr_diversify::_sim"]


async def test_a_local_function_in_a_builder_is_named_on_the_builder(tmp_path):
    """The tsx shape: `weaveTo` declared inside `buildThreads`."""
    path = "src/features/about/ConnectiveThreads.tsx"
    enclosing, nested = _nested_pair(
        path, "buildThreads", "weaveTo", outer_span=(40, 90), inner_span=(55, 70)
    )
    coordinator = _coordinator(
        tmp_path,
        source_dense=[enclosing, nested],
        source_lexical=[_FTSHit(enclosing.chunk_id, path), _FTSHit(nested.chunk_id, path)],
        records={hit.chunk_id: _record(hit) for hit in (enclosing, nested)},
    )

    response = await coordinator.search(
        "how are the connective threads between sections built", limit=5
    )

    assert _row_for(response, path)["contains_symbols"] == ["buildThreads::weaveTo"]


async def test_a_guard_closure_in_a_reducer_is_named_on_the_reducer(tmp_path):
    """The reducer shape: a `valid` closure inside `folderFanReducer`."""
    path = "src/features/folders/folder-fan-state.ts"
    enclosing, nested = _nested_pair(
        path, "folderFanReducer", "valid", outer_span=(10, 80), inner_span=(20, 30)
    )
    coordinator = _coordinator(
        tmp_path,
        source_dense=[enclosing, nested],
        source_lexical=[_FTSHit(enclosing.chunk_id, path), _FTSHit(nested.chunk_id, path)],
        records={hit.chunk_id: _record(hit) for hit in (enclosing, nested)},
    )

    response = await coordinator.search("what keeps the folder fan state consistent", limit=5)

    assert _row_for(response, path)["contains_symbols"] == ["folderFanReducer::valid"]


async def test_a_file_window_names_the_definitions_inside_its_span(tmp_path):
    """A window names no symbol of its own, so containment is the only proof."""
    path = "codeatlas/code_search/queries_flows.py"
    window = _hit(
        "queries_flows.py",
        path,
        0.71,
        kind="file_window",
        source="file_window",
        chunk_id=f"file:{path}:1-160",
        start_line=1,
        end_line=160,
        snippet="# file_window: 1-160",
    )
    nested = _hit(
        "_choose_flow_name",
        path,
        0.5,
        chunk_id=f"{path}::detect_and_persist_flows::_choose_flow_name",
        start_line=110,
        end_line=118,
    )
    coordinator = _coordinator(
        tmp_path,
        source_dense=[window, nested],
        source_lexical=[_FTSHit(window.chunk_id, path), _FTSHit(nested.chunk_id, path)],
        records={hit.chunk_id: _record(hit) for hit in (window, nested)},
    )

    response = await coordinator.search("how is a flow given its name", limit=5)

    row = _row_for(response, path)
    assert "symbol_path" not in row
    assert row["contains_symbols"] == ["detect_and_persist_flows::_choose_flow_name"]


async def test_a_nested_row_that_already_wins_keeps_its_own_name(tmp_path):
    """The already-passing shape must not regress into being called its parent."""
    path = "codeatlas/code_search/queries_flows.py"
    enclosing, nested = _nested_pair(
        path,
        "detect_and_persist_flows",
        "_choose_flow_name",
        outer_score=0.4,
        inner_score=0.9,
    )
    coordinator = _coordinator(
        tmp_path,
        source_dense=[nested, enclosing],
        source_lexical=[_FTSHit(nested.chunk_id, path)],
        records={hit.chunk_id: _record(hit) for hit in (enclosing, nested)},
    )

    response = await coordinator.search("_choose_flow_name", limit=5)

    row = _row_for(response, path)
    assert row["name"] == "_choose_flow_name"
    assert row["symbol_path"] == "detect_and_persist_flows::_choose_flow_name"


async def test_a_ts_local_function_that_already_wins_keeps_its_own_name(tmp_path):
    """The other already-passing shape: `gaussian` inside `clusterBallSampler`."""
    path = "src/graph/layout/position-initializers.ts"
    enclosing, nested = _nested_pair(
        path, "clusterBallSampler", "gaussian", outer_score=0.4, inner_score=0.9
    )
    coordinator = _coordinator(
        tmp_path,
        source_dense=[nested, enclosing],
        source_lexical=[_FTSHit(nested.chunk_id, path)],
        records={hit.chunk_id: _record(hit) for hit in (enclosing, nested)},
    )

    response = await coordinator.search("gaussian", limit=5)

    row = _row_for(response, path)
    assert row["symbol_path"] == "clusterBallSampler::gaussian"


async def test_a_disambiguated_id_does_not_carry_its_discriminator(tmp_path):
    """Two same-named nested symbols get ``~<hash>`` ids; the name is the name."""
    path = "src/a.py"
    nested = _hit(
        "inner",
        path,
        0.8,
        chunk_id=f"{path}::outer::inner~deadbeef",
        start_line=10,
        end_line=20,
    )
    coordinator = _coordinator(
        tmp_path, source_dense=[nested], records={nested.chunk_id: _record(nested)}
    )

    response = await coordinator.search("inner", limit=5)

    assert _row_for(response, path)["symbol_path"] == "outer::inner"


async def test_a_destructor_keeps_its_leading_tilde(tmp_path):
    """``~Foo`` is a name, not a discriminator: stripping it would erase the symbol."""
    path = "src/a.cpp"
    dtor = _hit("~Foo", path, 0.8, chunk_id=f"{path}::Foo::~Foo", start_line=10, end_line=20)
    coordinator = _coordinator(
        tmp_path, source_dense=[dtor], records={dtor.chunk_id: _record(dtor)}
    )

    response = await coordinator.search("how is Foo torn down", limit=5)

    assert _row_for(response, path)["symbol_path"] == "Foo::~Foo"


async def test_a_neighbour_is_not_claimed_as_something_the_row_contains(tmp_path):
    """Line containment alone is not enough when the row names a symbol of its own."""
    path = "src/a.py"
    outer = _hit("Outer", path, 0.8, chunk_id=f"{path}::Outer", start_line=10, end_line=50)
    # Inside Outer's lines, but the parser says it belongs to Other. One of the
    # two facts must be wrong; naming it anyway would publish the wrong one.
    stranger = _hit(
        "thing", path, 0.5, chunk_id=f"{path}::Other::thing", start_line=20, end_line=30
    )
    coordinator = _coordinator(
        tmp_path,
        source_dense=[outer, stranger],
        source_lexical=[_FTSHit(outer.chunk_id, path), _FTSHit(stranger.chunk_id, path)],
        records={hit.chunk_id: _record(hit) for hit in (outer, stranger)},
    )

    response = await coordinator.search("what does the outer thing do", limit=5)

    assert "contains_symbols" not in _row_for(response, path)


async def test_the_contained_list_is_capped_in_rank_order(tmp_path):
    """A class chunk can enclose more retrieved methods than a row should list."""
    from repowise.core.source_search.coordinator import MAX_CONTAINED_SYMBOLS

    path = "src/big.py"
    outer = _hit("Big", path, 0.99, chunk_id=f"{path}::Big", start_line=1, end_line=500)
    members = [
        _hit(
            f"m{index}",
            path,
            0.9 - index / 100.0,
            chunk_id=f"{path}::Big::m{index}",
            start_line=10 + index,
            end_line=11 + index,
        )
        for index in range(MAX_CONTAINED_SYMBOLS + 4)
    ]
    coordinator = _coordinator(
        tmp_path,
        source_dense=[outer, *members],
        source_lexical=[_FTSHit(hit.chunk_id, path) for hit in (outer, *members)],
        records={hit.chunk_id: _record(hit) for hit in (outer, *members)},
    )

    response = await coordinator.search("what does the big class hold", limit=5)

    contained = _row_for(response, path)["contains_symbols"]
    assert len(contained) == MAX_CONTAINED_SYMBOLS
    # Rank order, so the cap keeps the most relevant names rather than the
    # topmost lines in the file.
    assert contained[0] == "Big::m0"


async def test_naming_changes_nothing_about_which_rows_are_kept_or_their_order(
    tmp_path, monkeypatch
):
    """The pin: strip the two added keys and the response is the one it was.

    Ranking calibration is measured, and this path is only allowed to *name*.
    Running the same fused candidate set with the naming pass disabled and then
    enabled must differ in nothing else — not the rows, not their order, not a
    score, not the owner.
    """
    hits = [
        *_nested_pair("codeatlas/code_search/diversify.py", "mmr_diversify", "_sim"),
        *_nested_pair(
            "src/features/about/ConnectiveThreads.tsx",
            "buildThreads",
            "weaveTo",
            outer_span=(40, 90),
            inner_span=(55, 70),
            outer_score=0.61,
            inner_score=0.5,
        ),
        _hit("rank_files", "codeatlas/code_search/rank.py", 0.44, start_line=1, end_line=30),
    ]
    records = {hit.chunk_id: _record(hit) for hit in hits}
    query = "how are ranked results diversified to avoid near duplicates"

    def _strip(response: dict) -> dict:
        for row in response["results"]:
            row.pop("symbol_path", None)
            row.pop("contains_symbols", None)
        response["_meta"].pop("timing_ms", None)
        return response

    monkeypatch.setattr(
        SourceSearchCoordinator, "_name_contained_symbols", staticmethod(lambda *_: None)
    )
    without = _strip(
        await _coordinator(
            tmp_path,
            source_dense=hits,
            source_lexical=[_FTSHit(hit.chunk_id, hit.file_path) for hit in hits],
            records=records,
        ).search(query, limit=5)
    )
    monkeypatch.undo()
    live = await _coordinator(
        tmp_path,
        source_dense=hits,
        source_lexical=[_FTSHit(hit.chunk_id, hit.file_path) for hit in hits],
        records=records,
    ).search(query, limit=5)
    # Proves the fixture actually exercises the naming path, so the equality
    # below is a statement about a live case rather than about two no-ops.
    assert any("contains_symbols" in row for row in live["results"])

    assert _strip(live) == without


async def test_a_hex_named_destructor_is_not_mistaken_for_a_discriminator(tmp_path):
    """``deadbeef::~deadbeef`` is a destructor, not an eight-hex discriminator.

    The pathological overlap between the two spellings. Getting it wrong here
    does not drop the name, which would be survivable — it *mangles* it, eating
    the tail and serving ``deadbeef::`` as though a trailing separator were a
    symbol path. What separates the two is the boundary: the parser appends a
    discriminator to a complete id, so it never follows a ``::``.
    """
    path = "src/a.cpp"
    dtor = _hit(
        "~deadbeef",
        path,
        0.8,
        chunk_id=f"{path}::deadbeef::~deadbeef",
        start_line=10,
        end_line=20,
    )
    coordinator = _coordinator(
        tmp_path, source_dense=[dtor], records={dtor.chunk_id: _record(dtor)}
    )

    response = await coordinator.search("how is deadbeef torn down", limit=5)

    assert _row_for(response, path)["symbol_path"] == "deadbeef::~deadbeef"


async def test_a_disambiguated_destructor_keeps_the_destructor(tmp_path):
    """Both spellings at once: the discriminator goes, the tilde name stays."""
    path = "src/a.cpp"
    dtor = _hit(
        "~Foo",
        path,
        0.8,
        chunk_id=f"{path}::Foo::~Foo~cafebabe",
        start_line=10,
        end_line=20,
    )
    coordinator = _coordinator(
        tmp_path, source_dense=[dtor], records={dtor.chunk_id: _record(dtor)}
    )

    response = await coordinator.search("how is Foo torn down", limit=5)

    assert _row_for(response, path)["symbol_path"] == "Foo::~Foo"


async def test_a_page_that_names_no_file_never_yields_a_symbol_path(tmp_path):
    """A module or SCC page's target is a group key, and a ``::`` in it is not a chain.

    These serve ``file: ""`` by design. Reading a name out of such a target
    would invent a symbol the page never claimed to be about, so the missing
    file is a reason to refuse rather than a check to skip.
    """
    from repowise.core.source_search.coordinator import symbol_path_of

    assert symbol_path_of("nofile::Foo", "") == ""
    assert symbol_path_of("group::key::thing", "") == ""
    # Still read when the row does stand behind the file the id names.
    assert symbol_path_of("a/b.py::Outer::inner", "a/b.py") == "Outer::inner"


async def test_a_re_seated_row_is_the_one_that_gets_named(tmp_path):
    """The ordering pin: naming runs after the line-evidence upgrade.

    A wiki page for the file outranks both chunks, so deduplication keeps the
    page and the served slot belongs to it. ``_upgrade_line_evidence`` then
    re-seats that slot onto the enclosing chunk, because a page carries prose
    and no line bounds. Only after that is the served row the one a reader
    opens — so only after that can it be named.

    Naming one statement earlier still passes every other test in this file:
    the page it would annotate has no span, the annotation is skipped, and the
    chunk that actually reaches the caller arrives bare. This is the case that
    tells the difference.
    """
    path = "codeatlas/code_search/diversify.py"
    enclosing, nested = _nested_pair(path, "mmr_diversify", "_sim")
    page = _PageHit(
        page_id=f"file_page:{path}",
        target_path=path,
        score=0.99,
        page_type="file_page",
        title=f"File: {path}",
        snippet="How ranked results are diversified.",
    )
    # No lexical hits for the chunks, deliberately. With them the chunks
    # outrank the page on the fusion, dedupe keeps a chunk outright, and there
    # is no re-seat left to order anything against.
    coordinator = _coordinator(
        tmp_path,
        source_dense=[enclosing, nested],
        wiki_dense=[page],
        records={hit.chunk_id: _record(hit) for hit in (enclosing, nested)},
    )

    response = await coordinator.search(
        "how are ranked results diversified to avoid near duplicates", limit=5
    )

    row = _row_for(response, path)
    # The re-seat happened: dedupe kept the page, which carries no lines, and
    # the slot now holds the chunk.
    assert row["start_line"] == 100
    assert row["symbol_path"] == "mmr_diversify"
    # And the row that survived the re-seat is the one carrying the name.
    assert row["contains_symbols"] == ["mmr_diversify::_sim"]
