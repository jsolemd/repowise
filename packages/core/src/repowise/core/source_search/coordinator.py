"""One retrieval over two corpora, and an honest account of what it found.

The wiki index holds prose *about* a repository; the source index holds the
repository's own text. Asked separately they answer different questions and a
caller has to know which one to ask. Asked together they answer the question
people actually have — "where does this live" — provided the two rankings can
be merged without one scale swamping the other.

They can, in two different ways:

* **Dense.** Both corpora are embedded by the same embedder, so a cosine from
  ``source_chunks`` and a cosine from ``wiki_pages`` mean the same thing and
  merge by raw score into one ranking. This is why the query is embedded once
  here rather than by each store: two embeddings of one query are the same
  vector at twice the cost, and nothing downstream may assume otherwise.
* **Lexical.** Both legs are FTS5 BM25, but over different column sets, corpus
  sizes and MATCH expressions, so their raw scores are *not* comparable. They
  merge by interleaving each table's own ranking — see :func:`_merge_lexical`.

The two rankings then fuse by weighted RRF, which is rank-based and therefore
indifferent to whatever the underlying scales were.

Everything after fusion exists because a ranked list is not yet an answer:

* Tests are demoted, not dropped, unless the query is about tests.
* A query that is one bare identifier is a lookup, not a topic, so definitions
  of that name are pulled to the front of whatever the fusion produced.
* Results are deduplicated per file, because two chunks of one file are one
  file to open.
* One result is named as the **owner**, with the evidence for the choice.
* The response carries a **confidence** derived from absolute evidence —
  cosine values and exact-name hits — never from normalising the window
  against itself. A window normalised against itself is always confident about
  something, which is exactly the failure this replaces: on a query the corpus
  cannot answer, relative scoring reports its nearest noise as its best answer.

This class takes every store it uses as a constructor argument. The MCP server
and the REST API hold their handles in different places and build them at
different times, and a coordinator that reached for either process's globals
could only ever run in one of them.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from repowise.core.providers.embedding.base import Embedder

from .chunks import SOURCE_FILE_WINDOW, SOURCE_SYMBOL
from .fts import SourceFTSIndex
from .manifest import default_manifest_path, read_manifest
from .query_log import QueryEvent, QueryLog, TopEntry, default_query_log_path
from .vector_store import SourceChunkVectorStore

__all__ = [
    "AGREEMENT_LEXICAL_DEPTH",
    "CONFIDENT_DENSE_COSINE",
    "DENSE_WEIGHT",
    "EXACT_VIA_EMBEDDED",
    "EXACT_VIA_QUERY",
    "LANE_SOURCE",
    "LANE_WIKI",
    "LEG_FETCH",
    "LEXICAL_WEIGHT",
    "MIN_SUFFIX_SEGMENTS",
    "MIN_TAIL_CHARS",
    "NO_MATCH_DENSE_COSINE",
    "OWNER_SCORE_BAND",
    "RRF_K",
    "SOURCE_WIKI_PAGE",
    "SourceSearchCoordinator",
    "WikiFullTextSearch",
    "WikiVectorStore",
]

log = logging.getLogger(__name__)

#: Which corpus a result came from.
LANE_SOURCE = "source"
LANE_WIKI = "wiki"

#: The ``source`` value stamped on a wiki result, alongside the source index's
#: own ``symbol`` / ``file_window``. Three values, one field, so the owner
#: policy can rank chunk kinds against page kinds without a second predicate.
SOURCE_WIKI_PAGE = "wiki_page"

#: How deep each leg reads before fusion. Deep enough that a hit ranked well by
#: one leg and poorly by the other still meets itself in the fusion — which is
#: the entire mechanism — and shallow enough to stay one bounded query per leg.
LEG_FETCH = 100

#: RRF smoothing constant. 60 is the value the fusion literature settled on and
#: the one the stock wiki fusion already uses (``_answer_pipeline._RRF_K``), so
#: two fused rankings in one product do not disagree about rank spacing.
RRF_K = 60

#: Leg weights. Dense leads because the corpus is prose-and-code answering
#: questions phrased as prose; lexical is the corrective that keeps an exact
#: token from being embedded away into its neighbourhood.
DENSE_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3

# -- Confidence thresholds ---------------------------------------------------
#
# ABSOLUTE, and that is the whole point. They are cosine values against this
# embedder (ollama/embeddinggemma, 768d), measured on the frozen SoleMD.Infra
# corpus: the 5th percentile of a *correct* answer's cosine sits at ~0.43, and
# the 95th percentile of the best cosine for a topic the corpus does not cover
# sits at ~0.39. So 0.43 is where a hit is good enough to stand alone, and the
# band below it is where a second, independent signal has to agree before the
# response claims anything.
#
# Nothing here normalises against the window. A relative rule reports the best
# of ten wrong answers as a good answer, and the absent-topic half of the dev
# set exists to catch exactly that.

#: At or above this cosine, one leg is enough.
CONFIDENT_DENSE_COSINE = 0.43

#: Between this and :data:`CONFIDENT_DENSE_COSINE`, dense needs the lexical leg
#: to name the same file before the response is confident.
AGREEMENT_DENSE_COSINE = 0.38

#: How far into the lexical ranking that agreement may be found.
AGREEMENT_LEXICAL_DEPTH = 10

#: Below this cosine, with no exact name and nothing lexical, there is no
#: answer here — say so rather than serving the nearest noise.
NO_MATCH_DENSE_COSINE = 0.30

#: How close a result's fused score must be to the best one for the owner
#: policy to prefer it on shape (non-test, definition, source over wiki).
#: Beyond this the fusion is saying something the policy should not overrule.
OWNER_SCORE_BAND = 0.10

#: Confidence values, in the order a caller should treat them as degrading.
CONFIDENT = "confident"
CAUTION = "caution"
NO_MATCH = "no_match"

#: A query that is one bare identifier and nothing else. Bounded length so a
#: long dotted path of a sentence's worth of words cannot present as a symbol.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:]{1,127}$")

#: An identifier-shaped token carried *inside* a longer query. Three shapes, and
#: the point of all three is that no plain English word can wear them: a word
#: with an underscore in it, a word with an internal capital hump, or a dotted
#: or ``::``-qualified name. Prose is therefore free to contain as many words as
#: it likes without any of them presenting as a symbol.
#:
#: The snake and camel arms are ``tool_search._IDENT_TOKEN_RE``'s, with the camel
#: arm widened to accept a lowercase first letter so ``parseConfig`` is found and
#: not only ``ParseConfig``. The dotted arm is new here, and requires every
#: segment to be at least two characters so that "e.g." and initials do not read
#: as qualified names.
_EMBEDDED_IDENTIFIER_RE = re.compile(
    r"\b(?:"
    r"_*[A-Za-z0-9]+_[A-Za-z0-9_]+"
    r"|[A-Za-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+"
    r"|[A-Za-z_][A-Za-z0-9_]+(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]+)+"
    r")\b"
)

#: Shortest trailing dotted segment allowed to stand in for a whole name. Two
#: characters is a file extension or an initial far more often than it is a
#: symbol, and matching every ``py`` in the corpus is noise, not evidence.
MIN_TAIL_CHARS = 3

#: Fewest segments an identifier must have before it may match a longer name by
#: its tail. A one-segment identifier would match half the corpus — ``tests``
#: ends ``wants_tests``, ``run_tests``, ``skip_tests`` — which is a category,
#: not an answer. Two segments is where the tail stops being a common word and
#: starts being a name.
MIN_SUFFIX_SEGMENTS = 2

#: Splits a camel hump, taken before anything is lowercased: run the rule after
#: folding case and there are no humps left to find. Same order, and the same
#: reason, as :func:`repowise.core.source_search.fts.tokenize`.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: Everything that is not part of a segment — underscores, dots, colons, the
#: spaces the camel rule inserts.
_SEGMENT_BREAK = re.compile(r"[^A-Za-z0-9]+")

#: How an item came by its exact-name evidence. The whole query *being* the
#: identifier is a stronger claim than the query merely carrying it, and the two
#: reorder by different rules, so the response says which one it was.
EXACT_VIA_QUERY = "query"
EXACT_VIA_EMBEDDED = "embedded"

#: A query that names test material. On its own this is a query *about* tests,
#: and demoting them would be answering a different question.
_TEST_QUERY_RE = re.compile(r"\b(test|tests|testing|pytest|fixture|mock|spec)\b", re.IGNORECASE)

#: A query reporting something broken. Paired with a test word it flips the
#: reading: "fix the failing route test" names a test as the *symptom* and asks
#: for the code under it, which is the shape of every bug report an agent is
#: handed. Without this the demotion switch reads the symptom as the subject and
#: hands back the test file the caller is already looking at.
_REPAIR_QUERY_RE = re.compile(
    r"\b(fix|fixes|fixing|fixed|failing|fails|failure|failures|broken|breaks|bug|"
    r"crash|crashes|regression|debug|wrong|incorrect)\b",
    re.IGNORECASE,
)

#: Symbol kinds that define a thing rather than mention it. Preferred as owners
#: when the fusion cannot separate two results on score.
_DEFINITION_KINDS = frozenset({"class", "function", "method", "interface", "struct", "type"})

#: Owner preference among chunk kinds: the lines themselves beat a window that
#: merely contains them, which beats prose about them.
_SOURCE_RANK = {SOURCE_SYMBOL: 0, SOURCE_FILE_WINDOW: 1, SOURCE_WIKI_PAGE: 2}

#: Page types whose ``target_path`` is a repository-relative file. Every other
#: type is named by a curated id that reads like a path and is not one — a
#: module page by a directory-shaped group key, an SCC page by ``scc-<hash>``,
#: an overview by the repository's name. Ranking them is fine; serving one as
#: an openable file is not, and this response promises openable files.
#:
#: The same set is written down in ``server/mcp_server/_page_paths.py``, which
#: is where it belongs for the stock tools. It is restated rather than imported
#: because core must not import the server package; the two should be collapsed
#: into this module's copy when the page helpers move down.
_FILE_BACKED_PAGE_TYPES = frozenset({"file_page", "symbol_spotlight", "api_contract", "infra_page"})


class WikiVectorStore(Protocol):
    """The one method this coordinator needs from the wiki page vector store."""

    async def search_by_vector(self, vector: list[float], limit: int = 10) -> list[Any]: ...


class WikiFullTextSearch(Protocol):
    """The one method this coordinator needs from the wiki full-text index."""

    async def search(self, query: str, limit: int = 10) -> list[Any]: ...


@dataclass(slots=True, eq=False)
class _Item:
    """One candidate, carrying every signal that ranked it.

    Mutable because it is filled in by two legs arriving separately: the first
    leg to surface a candidate creates it, the second adds its evidence to the
    one already there. Compared by identity, not by value: two chunks can hold
    the same fields and still be two answers, and the ranking moves items about
    by position.
    """

    key: str
    lane: str
    file: str
    name: str
    kind: str
    snippet: str
    source: str
    #: The bare symbol name to match an identifier against, when the item has
    #: one. Separate from *name* because a wiki page's name is its title —
    #: "Symbol: a.b.c.foo", "File: a/b.py" — which is a sentence, not a name,
    #: and comparing an identifier against it can only ever fail. Taken
    #: structurally from the page id rather than parsed out of the title, and
    #: empty for a page that names no symbol.
    match_name: str = ""
    start_line: int | None = None
    end_line: int | None = None
    is_test: bool = False
    dense_cosine: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    #: "" when no identifier matched this item's name, else which router
    #: matched it — see :data:`EXACT_VIA_QUERY` / :data:`EXACT_VIA_EMBEDDED`.
    exact_via: str = ""
    #: Whether an embedded identifier matched the *tail* of this item's name.
    #: Deliberately not folded into ``exact_via``: ``query_wants_tests`` is not
    #: the name ``wants_tests``, and a response that said ``exact_name: true``
    #: for it would be claiming something it cannot show.
    suffix_match: bool = False
    fused_score: float = 0.0

    @property
    def exact_name(self) -> bool:
        """Whether an identifier matched this item's name, however it was found."""
        return bool(self.exact_via)

    def evidence(self) -> dict[str, Any]:
        return {
            "dense_cosine": round(self.dense_cosine, 4) if self.dense_cosine is not None else None,
            "lexical_rank": self.lexical_rank,
            "exact_name": self.exact_name,
            "lane": self.lane,
        }

    def to_result(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "file": self.file,
            # Mirrored under the name the stock tool serves paths as, so a
            # consumer that already knows how to open a search hit does not
            # need a second code path for this one.
            "target_path": self.file,
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "snippet": self.snippet,
            "relevance_score": round(self.fused_score, 6),
            "evidence": self.evidence(),
        }
        if self.start_line is not None and self.end_line is not None:
            out["start_line"] = self.start_line
            out["end_line"] = self.end_line
        return out


def _is_test_query(query: str) -> bool:
    """Whether *query* asks about tests, so test material ranks naturally.

    Naming a test is not the same as asking for one. A repair-shaped query
    names the failing test because that is where the failure showed up, and
    what it wants is the code under it — so the two signals together mean the
    opposite of the test signal alone.
    """
    return bool(_TEST_QUERY_RE.search(query)) and not _REPAIR_QUERY_RE.search(query)


def _identifier_query(query: str) -> str | None:
    """*query* as a bare identifier, or None when it is not one."""
    stripped = query.strip()
    if not stripped or any(ch.isspace() for ch in stripped):
        return None
    return stripped if _IDENTIFIER_RE.match(stripped) else None


def _embedded_identifiers(query: str) -> list[str]:
    """Identifier-shaped tokens *query* carries inside natural language.

    Order-preserving and deduplicated, so a query that names one symbol twice
    does not weight it twice. Empty for prose that names none, which is the
    common case and the reason this is cheap to ask.
    """
    seen: set[str] = set()
    out: list[str] = []
    for token in _EMBEDDED_IDENTIFIER_RE.findall(query):
        folded = token.lower()
        if folded not in seen:
            seen.add(folded)
            out.append(token)
    return out


def _norm_identifier(text: str) -> str:
    """Fold the separators that spell one name three ways.

    ``Class::method``, ``Class.method`` and ``CLASS.METHOD`` are the same name
    to a reader and to the symbol index, which stores whichever the language
    writes.
    """
    return text.strip().lower().replace("::", ".")


def _segments(name: str) -> tuple[str, ...]:
    """*name* split into its lowercase parts, however it spells the boundaries.

    ``query_wants_tests``, ``queryWantsTests`` and ``query.wants.tests`` are one
    name written three ways, and all three segment to
    ``("query", "wants", "tests")``. Camel humps are cut first, because folding
    case first would leave none to cut.
    """
    spaced = _CAMEL_BOUNDARY.sub(" ", name.strip())
    return tuple(part.lower() for part in _SEGMENT_BREAK.split(spaced) if part)


def _suffix_matches(identifier: Sequence[str], name: Sequence[str]) -> bool:
    """Whether *name* ends with *identifier* on a segment boundary.

    This is what lets ``wants_tests`` find ``query_wants_tests`` — the thing a
    person means when they name a helper by the part of it that carries the
    meaning. Comparing whole segments rather than characters is what makes it a
    boundary match and not a substring one: ``ants_tests`` shares nine trailing
    characters with ``query_wants_tests`` and none of its segments, so it is
    correctly no match at all.

    Strictly longer, because an equal-length match is the same name, and that
    is the stronger claim the full-name tier above this one already makes.
    """
    width = len(identifier)
    return (
        width >= MIN_SUFFIX_SEGMENTS
        and len(name) > width
        and tuple(name[-width:]) == tuple(identifier)
    )


def _name_matchers(identifier: str) -> set[str]:
    """Every normalised name *identifier* should be considered a match for.

    The identifier itself, plus its last dotted segment — an agent writing
    ``Class.method`` is asking after ``method``, which is what the symbol index
    stores. The segment is dropped when it is shorter than
    :data:`MIN_TAIL_CHARS`: ``refresh.py`` would otherwise ask every symbol
    named ``py`` to present itself as an exact match.
    """
    normalised = _norm_identifier(identifier)
    matchers = {normalised}
    tail = normalised.rsplit(".", 1)[-1]
    if len(tail) >= MIN_TAIL_CHARS:
        matchers.add(tail)
    return matchers


def _page_file(page_type: str, target_path: str) -> str:
    """The openable file behind a page, or ``""`` when it names none.

    A symbol spotlight's target is ``file.py::Symbol`` — a page id built from a
    file path, so the file is recoverable by splitting. A module page's target
    is a structural group key that *looks* like a directory, and an SCC page's
    is a hash: no amount of string manipulation makes either openable, so they
    yield nothing rather than a plausible-looking path that errors on open.
    """
    if page_type not in _FILE_BACKED_PAGE_TYPES:
        return ""
    return (target_path or "").split("::", 1)[0].strip()


def _is_test_related(path: str) -> bool:
    """Whether *path* is test material, by the rule that stamped the corpus.

    Deferred import for the reason :mod:`.chunks` defers it: this module is
    reachable from a process that has not paid for the ingestion package.
    """
    if not path:
        return False
    from ..test_paths import is_test_related_path

    return bool(is_test_related_path(path))


def _merge_lexical(
    source_hits: Sequence[Any], wiki_hits: Sequence[Any]
) -> list[tuple[str, Any, str]]:
    """One lexical ranking from two BM25 tables, by rank rather than by score.

    Both legs return a negated FTS5 ``bm25()``, which looks comparable and is
    not. The two tables differ in every input that scale depends on: column
    count and length (``source_fts`` indexes two short columns, ``page_fts``
    four), document size (a chunk against a whole generated page), corpus size,
    and the expression itself — the wiki arm builds a *selectivity-filtered*
    MATCH while the source arm ORs every token. BM25 normalises against its own
    corpus' average document length and term frequencies, so these are two
    different scales wearing the same units, and merging them by score hands
    the ranking to whichever table happens to produce larger numbers. On the
    SoleMD.Infra corpus that is 8,200 chunks against ~1,000 pages, and the
    spread is wide enough to decide the whole window.

    Interleaving asks each table only what it can answer — the order of its
    own hits — and lets RRF weigh the merged position against the dense leg.
    Returns ``(lane, hit, key)`` triples in merged order.
    """
    merged: list[tuple[str, Any, str]] = []
    for index in range(max(len(source_hits), len(wiki_hits))):
        if index < len(source_hits):
            hit = source_hits[index]
            merged.append((LANE_SOURCE, hit, f"{LANE_SOURCE}:{hit.chunk_id}"))
        if index < len(wiki_hits):
            hit = wiki_hits[index]
            merged.append((LANE_WIKI, hit, f"{LANE_WIKI}:{hit.page_id}"))
    return merged


class SourceSearchCoordinator:
    """Hybrid source+wiki retrieval, owner selection, and confidence.

    Constructed per process, reused across queries. Every store handle is
    passed in: the MCP server resolves them from its lifespan state and the
    REST app from ``app.state``, and neither layout is visible from here.

    Not safe to share across threads — the source FTS index is a SQLite
    connection bound to the thread that opened it, which is the event-loop
    thread in both hosts.
    """

    def __init__(
        self,
        *,
        repo_path: Path | str,
        embedder: Embedder,
        source_vectors: SourceChunkVectorStore,
        source_fts: SourceFTSIndex,
        wiki_vectors: WikiVectorStore | None = None,
        wiki_fts: WikiFullTextSearch | None = None,
        query_log: QueryLog | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self._embedder = embedder
        self._source_vectors = source_vectors
        self._source_fts = source_fts
        self._wiki_vectors = wiki_vectors
        self._wiki_fts = wiki_fts
        self._query_log = query_log or QueryLog(default_query_log_path(self.repo_path))
        self._manifest_meta: dict[str, Any] | None = None

    # -- public surface ---------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        mode: str = "hybrid",
        base_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Retrieve, rank, name an owner, and say how sure it is.

        *base_meta* is the host's own ``_meta`` envelope; this adds its
        ``source_search`` block to it rather than replacing it, so a response
        served through the MCP tool keeps the freshness and embedder fields
        every other tool emits.
        """
        started = time.perf_counter()
        items, lexical_files = await self._retrieve(query)
        # Read before the deduplication, which keeps one item per file and so
        # can only ever leave one lane standing for it. Corroboration is a
        # question about what was *retrieved*, not about what survived ranking.
        source_files = {
            item.file for item in items.values() if item.lane == LANE_SOURCE and item.file
        }
        ranked = self._rank(query, items)
        deduped = self._dedupe_by_file(ranked)
        window = deduped[:limit]
        owner, reason = self._select_owner(window, wants_tests=_is_test_query(query))
        if owner is not None and deduped and deduped[0] is not owner:
            # The owner is the answer, so it leads the list the caller reads
            # and the candidate paths derived from it. Identity, not equality:
            # two results can hold identical fields and be different answers.
            deduped = [owner, *(item for item in deduped if item is not owner)]
            window = deduped[:limit]
        confidence = self._classify(window, lexical_files, owner, source_files)
        # Last, and after the owner: see :meth:`_upgrade_line_evidence`.
        self._upgrade_line_evidence(window, ranked)
        latency_ms = (time.perf_counter() - started) * 1000.0

        response = self._envelope(
            query=query,
            mode=mode,
            limit=limit,
            window=window,
            deduped=deduped,
            owner=owner,
            reason=reason,
            confidence=confidence,
            latency_ms=latency_ms,
            base_meta=base_meta,
        )
        self._record(query, mode, limit, latency_ms, confidence, window, owner)
        return response

    # -- retrieval --------------------------------------------------------

    async def _retrieve(self, query: str) -> tuple[dict[str, _Item], list[str]]:
        """Both legs, fused. Returns the candidates and the lexical top files.

        The lexical top files come back separately because the confidence rule
        asks a question the ranked items cannot answer on their own: whether a
        file the dense leg likes was *also* found lexically, near the top,
        independently of whether that file survived ranking.
        """
        dense_source, dense_wiki = await self._dense_legs(query)
        lexical_source, lexical_wiki = await self._lexical_legs(query)

        items: dict[str, _Item] = {}
        records = await self._chunk_records(dense_source, lexical_source)

        merged_dense = _merge_dense(dense_source, dense_wiki)
        for rank, (lane, hit, key) in enumerate(merged_dense, start=1):
            item = self._item_for(items, lane, hit, key, records)
            if item is None:
                continue
            item.dense_rank = rank
            item.dense_cosine = float(hit.score)

        lexical_files: list[str] = []
        for rank, (lane, hit, key) in enumerate(_merge_lexical(lexical_source, lexical_wiki), 1):
            item = self._item_for(items, lane, hit, key, records)
            if item is None:
                continue
            item.lexical_rank = rank
            if rank <= AGREEMENT_LEXICAL_DEPTH and item.file:
                lexical_files.append(item.file)
            # The lexical leg cuts its snippet around what matched; the dense
            # leg has no query text at the point its row was written and can
            # only serve the stored opener. Prefer the one that shows evidence.
            snippet = str(getattr(hit, "snippet", None) or "")
            if lane == LANE_WIKI and snippet:
                item.snippet = snippet

        for item in items.values():
            item.fused_score = _rrf(item.dense_rank, item.lexical_rank)
        return items, lexical_files

    async def _dense_legs(self, query: str) -> tuple[list[Any], list[Any]]:
        """Top-*LEG_FETCH* from each vector store, on one embedding of *query*."""
        try:
            vectors = await self._embedder.embed([query])
        except Exception:
            log.debug("source-search dense leg: embedding failed", exc_info=True)
            return [], []
        if not vectors:
            return [], []
        vector = [float(v) for v in vectors[0]]
        source, wiki = await asyncio.gather(
            _safe(self._source_vectors.search_by_vector(vector, limit=LEG_FETCH), "source dense"),
            _safe(
                self._wiki_vectors.search_by_vector(vector, limit=LEG_FETCH)
                if self._wiki_vectors is not None
                else None,
                "wiki dense",
            ),
        )
        return source, wiki

    async def _lexical_legs(self, query: str) -> tuple[list[Any], list[Any]]:
        """Top-*LEG_FETCH* from each BM25 index.

        The source index tokenizes *query* with the same function that built
        its token stream — that is :meth:`SourceFTSIndex.query`'s contract, and
        the reason a query is handed to it as free text rather than pre-split.
        """
        try:
            source = self._source_fts.query(query, limit=LEG_FETCH)
        except Exception:
            log.debug("source-search lexical leg: source FTS failed", exc_info=True)
            source = []
        wiki = await _safe(
            self._wiki_fts.search(query, limit=LEG_FETCH) if self._wiki_fts is not None else None,
            "wiki lexical",
        )
        return source, wiki

    async def _chunk_records(
        self, dense_source: Sequence[Any], lexical_source: Sequence[Any]
    ) -> dict[str, Any]:
        """Metadata for every source chunk the lexical leg found alone.

        A BM25 hit is an id and a file path. The dense leg already carries the
        rest, so only the ids it did not return are fetched, in one query.
        """
        known = {hit.chunk_id for hit in dense_source}
        missing = [hit.chunk_id for hit in lexical_source if hit.chunk_id not in known]
        if not missing:
            return {}
        try:
            return await self._source_vectors.fetch_by_chunk_ids(missing)
        except Exception:
            log.debug("source-search: chunk metadata lookup failed", exc_info=True)
            return {}

    def _item_for(
        self,
        items: dict[str, _Item],
        lane: str,
        hit: Any,
        key: str,
        records: dict[str, Any],
    ) -> _Item | None:
        """The candidate for *key*, created from *hit* the first time it is seen."""
        existing = items.get(key)
        if existing is not None:
            return existing
        item = (
            self._source_item(hit, key, records)
            if lane == LANE_SOURCE
            else self._wiki_item(hit, key)
        )
        if item is not None:
            items[key] = item
        return item

    @staticmethod
    def _source_item(hit: Any, key: str, records: dict[str, Any]) -> _Item | None:
        """A source chunk as a candidate, from a dense hit or a fetched record."""
        record = records.get(hit.chunk_id, hit)
        file_path = getattr(record, "file_path", "") or getattr(hit, "file_path", "")
        name = getattr(record, "name", "")
        if not name and not file_path:
            return None
        return _Item(
            key=key,
            lane=LANE_SOURCE,
            file=file_path,
            name=name,
            match_name=name,
            kind=getattr(record, "kind", ""),
            snippet=getattr(record, "snippet", ""),
            source=getattr(record, "source", "") or SOURCE_SYMBOL,
            start_line=getattr(record, "start_line", None),
            end_line=getattr(record, "end_line", None),
            is_test=bool(getattr(record, "is_test", False)) or _is_test_related(file_path),
        )

    @staticmethod
    def _wiki_item(hit: Any, key: str) -> _Item | None:
        """A wiki page as a candidate. Its ``kind`` is the page type."""
        page_type = getattr(hit, "page_type", "") or ""
        target_path = getattr(hit, "target_path", "") or ""
        file_path = _page_file(page_type, target_path)
        # A symbol spotlight's page id is ``file.py::Symbol``, so the symbol is
        # recoverable from the id itself. Read from there rather than from the
        # title, which is prose ("Symbol: a.b.c.foo") and would need parsing
        # that a retitling could silently break.
        _, sep, symbol = target_path.partition("::")
        return _Item(
            key=key,
            lane=LANE_WIKI,
            file=file_path,
            name=getattr(hit, "title", "") or "",
            match_name=symbol.rsplit(".", 1)[-1] if sep else "",
            kind=page_type,
            snippet=getattr(hit, "snippet", "") or "",
            source=SOURCE_WIKI_PAGE,
            is_test=_is_test_related(file_path),
        )

    # -- ranking ----------------------------------------------------------

    def _rank(self, query: str, items: dict[str, _Item]) -> list[_Item]:
        """Fused order, then test demotion, then the exact-identifier router."""
        ranked = sorted(items.values(), key=_fused_sort_key)
        if not _is_test_query(query):
            # Stable, so the fused order survives inside each group. Demotion,
            # not removal: a test is rarely the best first read, and sometimes
            # it is the only place a behaviour is written down.
            ranked.sort(key=lambda item: item.is_test)
        identifier = _identifier_query(query)
        if identifier:
            return self._route_exact(identifier, ranked)
        if embedded := _embedded_identifiers(query):
            return self._route_embedded(embedded, ranked)
        return ranked

    @staticmethod
    def _route_exact(identifier: str, ranked: list[_Item]) -> list[_Item]:
        """Definitions of *identifier* first, then mentions, then the rest.

        A bare identifier is a lookup. Dense retrieval answers it with the
        neighbourhood the name lives in — plausible, adjacent, and not the
        definition — so the name match is applied as an ordering over what
        fusion returned rather than as a separate search.
        """
        wanted = _name_matchers(identifier)
        literal = identifier.lower()

        buckets: list[tuple[int, _Item]] = []
        for item in ranked:
            name = _norm_identifier(item.match_name)
            if name and name in wanted:
                item.exact_via = EXACT_VIA_QUERY
                buckets.append((0, item))
            elif literal in item.snippet.lower() or literal in item.file.lower():
                buckets.append((1, item))
            else:
                buckets.append((2, item))
        buckets.sort(key=lambda pair: pair[0])
        return [item for _, item in buckets]

    @staticmethod
    def _route_embedded(identifiers: Sequence[str], ranked: list[_Item]) -> list[_Item]:
        """Definitions of an identifier the query *mentions*, ahead of the rest.

        Three tiers: the name *is* the identifier, the name *ends with* it on a
        segment boundary, then everything else.

        Deliberately weaker than :meth:`_route_exact`, in one specific way: it
        has no "mentions the literal somewhere" tier. When the whole query is
        one identifier, containment is the second-best evidence there is,
        because there is nothing else to go on. When the query is a sentence
        that happens to name a symbol, the sentence is carrying most of the
        intent and containment is nearly free — half the files in a subsystem
        mention any given helper — so promoting on it would let a weak signal
        overrule the fusion. A name match is not free, and neither is a
        boundary-aligned tail; those are the two this keeps.

        The partition is stable, so the fused order, the test demotion and
        everything else that ranked these items survives inside every tier.
        """
        wanted: set[str] = set()
        for identifier in identifiers:
            wanted |= _name_matchers(identifier)
        tails = [_segments(identifier) for identifier in identifiers]

        tiers: list[list[_Item]] = [[], [], []]
        for item in ranked:
            name = _norm_identifier(item.match_name)
            if name and name in wanted:
                item.exact_via = EXACT_VIA_EMBEDDED
                tiers[0].append(item)
                continue
            segments = _segments(item.match_name)
            if segments and any(_suffix_matches(tail, segments) for tail in tails):
                item.suffix_match = True
                tiers[1].append(item)
                continue
            tiers[2].append(item)
        return [item for tier in tiers for item in tier]

    @staticmethod
    def _dedupe_by_file(ranked: Sequence[_Item]) -> list[_Item]:
        """Best item per file, in rank order.

        Items with no file behind them (a decision record, a repo overview)
        are never collapsed together: they are distinct answers that happen to
        share the absence of a path.
        """
        seen: set[str] = set()
        out: list[_Item] = []
        for item in ranked:
            if item.file:
                if item.file in seen:
                    continue
                seen.add(item.file)
            out.append(item)
        return out

    @staticmethod
    def _upgrade_line_evidence(window: list[_Item], ranked: Sequence[_Item]) -> None:
        """Re-seat each served file on an item that can point at its lines.

        A wiki page and the chunk it describes are one file to open, and when
        the page ranks first the file arrives with prose about itself and no
        line bounds. A chunk further down for the same file takes the page's
        slot without taking its position: nothing is re-ranked, only re-cited.

        Runs *after* the owner is chosen, and that ordering is the whole
        subtlety. The owner policy prefers a chunk to a page, which is a
        judgement about which file to open — and it can only make it while it
        can still see that a file is page-backed. Upgrading first erases that,
        every file starts looking chunk-backed, and the preference silently
        stops firing (measured: one behavioural case lost rank 1 to exactly
        this). Choose the owner on what retrieval found, then improve the
        citation.

        Never at the cost of name evidence: an exact-name page keeps its slot
        against a chunk that matched nothing, because that trades the stronger
        claim for the better citation.
        """
        best_with_lines: dict[str, _Item] = {}
        for item in ranked:
            if item.file and item.start_line is not None:
                best_with_lines.setdefault(item.file, item)
        for position, item in enumerate(window):
            candidate = best_with_lines.get(item.file) if item.file else None
            if candidate is not None and _shows_lines_better(item, candidate):
                window[position] = candidate

    # -- owner and confidence ---------------------------------------------

    @staticmethod
    def _select_owner(
        window: Sequence[_Item], *, wants_tests: bool = False
    ) -> tuple[_Item | None, str]:
        """The result to open first, and a phrase naming the evidence for it.

        Preference applies only inside :data:`OWNER_SCORE_BAND` of the best
        fused score. Outside it the fusion is making a claim the shape rules
        should not overturn: a wiki page that beat every chunk by a clear
        margin is the answer, whatever its shape.

        *wants_tests* carries the same reading of the query the demotion uses.
        Without it the policy would hand a test-focused query a non-test owner
        it deliberately ranked below the test the caller asked for.
        """
        if not window:
            return None, ""
        best = window[0].fused_score
        floor = best * (1.0 - OWNER_SCORE_BAND)
        band = [item for item in window if item.fused_score >= floor] or [window[0]]
        winner = min(
            enumerate(band), key=lambda pair: _owner_sort_key(pair[1], pair[0], wants_tests)
        )[1]

        if winner.exact_via == EXACT_VIA_QUERY:
            reason = "exact name match"
        elif winner.exact_via == EXACT_VIA_EMBEDDED:
            reason = "embedded identifier match"
        elif winner.suffix_match:
            reason = "embedded identifier suffix match"
        elif winner.dense_rank is not None and winner.lexical_rank is not None:
            reason = "dense+lexical agreement"
        elif winner.dense_rank is not None:
            reason = "dense only"
        else:
            reason = "lexical only"

        displaced = window[0]
        if displaced is not winner:
            reason += f"; preferred over a closely-scored {_shape_phrase(displaced)}"
        return winner, reason

    @staticmethod
    def _classify(
        window: Sequence[_Item],
        lexical_files: Sequence[str],
        owner: _Item | None,
        source_files: set[str],
    ) -> str:
        """Confidence from absolute evidence only — never a normalised window."""
        if not window:
            return NO_MATCH
        cosines = [item.dense_cosine for item in window if item.dense_cosine is not None]
        best_dense = max(cosines) if cosines else 0.0
        exact = any(item.exact_name for item in window)
        any_lexical = any(item.lexical_rank is not None for item in window)
        top_lexical = set(lexical_files[:AGREEMENT_LEXICAL_DEPTH])
        agreement = any(
            item.dense_cosine is not None
            and item.dense_cosine >= AGREEMENT_DENSE_COSINE
            and item.file
            and item.file in top_lexical
            for item in window
        )

        if best_dense < NO_MATCH_DENSE_COSINE and not exact and not any_lexical:
            return NO_MATCH
        if not (exact or best_dense >= CONFIDENT_DENSE_COSINE or agreement):
            return CAUTION
        if _uncorroborated_page(source_files, owner):
            return CAUTION
        return CONFIDENT

    # -- response ---------------------------------------------------------

    def _envelope(
        self,
        *,
        query: str,
        mode: str,
        limit: int,
        window: Sequence[_Item],
        deduped: Sequence[_Item],
        owner: _Item | None,
        reason: str,
        confidence: str,
        latency_ms: float,
        base_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        meta = dict(base_meta or {})
        meta["timing_ms"] = round(latency_ms, 2)
        meta["source_search"] = self._source_meta()

        results = [item.to_result() for item in window]
        _clamp_monotone(results)
        response: dict[str, Any] = {
            "results": results,
            "mode": mode,
            "confidence": confidence,
            "_meta": meta,
        }
        candidates = _candidates(deduped, limit)
        if candidates:
            response["candidates"] = candidates
        # Always present, even as null: a caller that reads this key to decide
        # what to open must be able to tell "nothing to open" from "the field
        # is missing on this response shape".
        response["selected_owner"] = (
            {"file": owner.file, "reason": reason} if owner is not None and owner.file else None
        )
        if confidence == NO_MATCH:
            response["note"] = (
                f"No indexed match for {query!r}. The corpus covers this repository at "
                "the commit named in _meta.source_search — a question outside it has no "
                "answer here, and the results below (if any) are nearest neighbours, not "
                "evidence."
            )
        return response

    def _source_meta(self) -> dict[str, Any]:
        """The generation this response was served from, read once per process.

        Cached because the manifest is written by a rebuild, and a rebuild
        replaces the stores this coordinator holds anyway — a process that
        outlives one is already answering from the index it opened.
        """
        if self._manifest_meta is None:
            manifest = read_manifest(default_manifest_path(self.repo_path))
            self._manifest_meta = (
                {
                    "generation": manifest.corpus_hash[:12],
                    "indexed_commit": manifest.indexed_commit,
                    "symbol_chunks": manifest.symbol_chunks,
                    "file_window_chunks": manifest.file_window_chunks,
                }
                if manifest is not None
                else {"generation": None, "indexed_commit": None}
            )
        return dict(self._manifest_meta)

    def _record(
        self,
        query: str,
        mode: str,
        limit: int,
        latency_ms: float,
        confidence: str,
        window: Sequence[_Item],
        owner: _Item | None,
    ) -> None:
        """Append the query-quality event. Never raises — see :mod:`.query_log`."""
        self._query_log.append(
            QueryEvent(
                query=query,
                mode=mode,
                limit=limit,
                latency_ms=round(latency_ms, 2),
                confidence=confidence,
                result_count=len(window),
                top=[
                    TopEntry(
                        file=item.file,
                        lane=item.lane,
                        dense_cosine=(
                            round(item.dense_cosine, 4) if item.dense_cosine is not None else None
                        ),
                        lexical_rank=item.lexical_rank,
                        exact_name=item.exact_name,
                        fused_score=round(item.fused_score, 6),
                    )
                    for item in window
                ],
                selected_owner_file=owner.file if owner is not None else None,
                no_match=confidence == NO_MATCH,
            )
        )


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


async def _safe(awaitable: Any, label: str) -> list[Any]:
    """Await *awaitable*, returning ``[]`` on anything it raises.

    One leg failing is a degraded answer; two legs failing because one of them
    raised is no answer at all.
    """
    if awaitable is None:
        return []
    try:
        return list(await awaitable)
    except Exception:
        log.debug("source-search leg %s failed", label, exc_info=True)
        return []


def _merge_dense(
    source_hits: Sequence[Any], wiki_hits: Sequence[Any]
) -> list[tuple[str, Any, str]]:
    """One dense ranking from both stores, by raw cosine.

    Sound because both tables were written by the same embedder, which is an
    invariant of the recipe fingerprint rather than a hope: a rebuild under a
    different embedder invalidates the source index outright.
    """
    merged: list[tuple[str, Any, str]] = [
        (LANE_SOURCE, hit, f"{LANE_SOURCE}:{hit.chunk_id}") for hit in source_hits
    ]
    merged += [(LANE_WIKI, hit, f"{LANE_WIKI}:{hit.page_id}") for hit in wiki_hits]
    merged.sort(key=lambda entry: -float(entry[1].score))
    return merged


def _rrf(dense_rank: int | None, lexical_rank: int | None) -> float:
    """Weighted reciprocal-rank fusion. A leg that missed contributes nothing."""
    score = 0.0
    if dense_rank is not None:
        score += DENSE_WEIGHT / (RRF_K + dense_rank)
    if lexical_rank is not None:
        score += LEXICAL_WEIGHT / (RRF_K + lexical_rank)
    return score


def _fused_sort_key(item: _Item) -> tuple[float, int, int, str]:
    """Fused score descending, with a total order so results are reproducible."""
    return (
        -item.fused_score,
        item.dense_rank if item.dense_rank is not None else 1 << 30,
        item.lexical_rank if item.lexical_rank is not None else 1 << 30,
        item.key,
    )


def _owner_sort_key(item: _Item, position: int, wants_tests: bool) -> tuple[int, ...]:
    """Owner preference: openable, name evidence, product code, shape, rank.

    Openability leads because the owner is a file to open. A page named by a
    group key can be a perfectly good *result* and can never be the answer to
    "what do I open", so it is passed over however well it ranked.

    Name evidence is graded rather than boolean, so the owner agrees with the
    order the routers produced: the name itself, then a boundary-aligned tail,
    then no name evidence at all.
    """
    return (
        0 if item.file else 1,
        0 if item.exact_name else (1 if item.suffix_match else 2),
        0 if wants_tests else (1 if item.is_test else 0),
        _SOURCE_RANK.get(item.source, 3),
        0 if item.kind in _DEFINITION_KINDS else 1,
        position,
    )


def _shows_lines_better(kept: _Item, candidate: _Item) -> bool:
    """Whether *candidate* should take *kept*'s slot for their shared file.

    Only to gain line bounds, and only when nothing else is given up: an
    incumbent carrying name evidence keeps its slot against a candidate
    carrying none, because "this is the symbol you named" outranks "here are
    some lines from the same file".
    """
    if kept.start_line is not None or candidate.start_line is None:
        return False
    return not kept.exact_name or candidate.exact_name


def _uncorroborated_page(source_files: set[str], owner: _Item | None) -> bool:
    """Whether the owner rests on generated prose that no source chunk backs.

    A wiki page is written *about* code. When it is the answer and the
    repository's own text never named the same file, what the response has is a
    description and no lines behind it — which is precisely the case where a
    confident answer sends a reader to a page that paraphrases something that
    has since moved, or that was never quite what they asked about.

    This is the same corroboration rule the dense/lexical agreement arm already
    applies, moved up a level: there, two retrievers over one corpus have to
    agree; here, two corpora do. It demotes to caution rather than reordering,
    because the page may well be right — it is the *claim* that is unsupported,
    not the hit.

    *source_files* is every file the source lane retrieved, taken before the
    per-file deduplication. After it the answer would always be "no": dedupe
    keeps one item per file, so a page that outranked its own file's chunks is
    the only thing left holding that path.
    """
    if owner is None or owner.lane != LANE_WIKI or not owner.file:
        return False
    return owner.file not in source_files


def _shape_phrase(item: _Item) -> str:
    """How to describe a result that the owner policy passed over."""
    if item.is_test:
        return "test file"
    if item.source == SOURCE_WIKI_PAGE:
        return "wiki page"
    if item.source == SOURCE_FILE_WINDOW:
        return "file window"
    return "chunk"


#: Gap held between two clamped scores, at the precision they are served with.
#: Equal scores would let a consumer's own sort reorder a list this one had a
#: reason to order.
_SCORE_EPSILON = 1e-6


def _clamp_monotone(results: list[dict[str, Any]]) -> None:
    """Hold ``relevance_score`` non-increasing along the order actually served.

    Three stages reorder the fused ranking after the scores are computed: the
    test demotion, the exact-identifier router and the owner policy. All three
    are deliberate, and all three leave a result whose fused score is higher
    than the one above it — so a consumer that re-sorts by score sees a
    different list from the one it was handed, and the reorder is silently
    undone by whoever trusts the number over the order.

    The order is what carries the ranking decisions, so it wins and the score
    is clamped to agree with it. The unclamped fusion is still recoverable: the
    per-result ``evidence`` carries the raw cosine and lexical rank it was built
    from, and the query log records the true fused score.

    Same resolution ``FullTextSearch`` reached for the same problem across two
    incomparable BM25 expressions.
    """
    ceiling: float | None = None
    for result in results:
        score = float(result["relevance_score"])
        if ceiling is not None and score >= ceiling:
            score = ceiling
        result["relevance_score"] = round(score, 6)
        ceiling = round(score - _SCORE_EPSILON, 6)


def _candidates(deduped: Sequence[_Item], limit: int) -> list[dict[str, str]]:
    """Openable files, best first, in the shape the stock tool already serves.

    Drawn from the full deduplicated ranking rather than the returned window,
    so a search whose window is spent on pages that name no file can still say
    which files to open.
    """
    out: list[dict[str, str]] = []
    for item in deduped:
        if not item.file:
            continue
        out.append({"path": item.file})
        if len(out) >= limit:
            break
    return out
