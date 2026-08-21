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
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from repowise.core.providers.embedding.base import Embedder

from .chunks import SOURCE_FILE_WINDOW, SOURCE_SYMBOL, language_for_path, window_eligible
from .fts import SourceFTSIndex, tokenize
from .manifest import default_manifest_path, read_manifest
from .query_log import QueryEvent, QueryLog, TopEntry, default_query_log_path
from .vector_store import SourceChunkVectorStore

__all__ = [
    "AGREEMENT_LEXICAL_DEPTH",
    "CONFIDENT_CONCEPT_COVERAGE",
    "CONFIDENT_DENSE_COSINE",
    "DENSE_WEIGHT",
    "ERROR_ALL_LEGS_FAILED",
    "EXACT_VIA_EMBEDDED",
    "EXACT_VIA_QUERY",
    "LANE_SOURCE",
    "LANE_WIKI",
    "LEG_FETCH",
    "LEG_SOURCE_DENSE",
    "LEG_SOURCE_LEXICAL",
    "LEG_WIKI_DENSE",
    "LEG_WIKI_LEXICAL",
    "LEXICAL_WEIGHT",
    "MIN_SUFFIX_SEGMENTS",
    "MIN_TAIL_CHARS",
    "NO_MATCH_CONCEPT_COVERAGE",
    "NO_MATCH_DENSE_COSINE",
    "OWNER_SCORE_BAND",
    "RRF_K",
    "SOURCE_WIKI_PAGE",
    "LegFailure",
    "QueryIntent",
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

#: Dense floor for a prose owner that also carries complete concept and lexical
#: evidence.  Dense is never sufficient by itself.
CONFIDENT_DENSE_COSINE = 0.43

#: Between this and :data:`CONFIDENT_DENSE_COSINE`, dense needs the lexical leg
#: to name the same file before the response is confident.
AGREEMENT_DENSE_COSINE = 0.38

#: How far into the lexical ranking that agreement may be found.
AGREEMENT_LEXICAL_DEPTH = 10

#: Minimum IDF-weighted share of informative query concepts that must occur in
#: the selected owner file before prose can be called confident.
CONFIDENT_CONCEPT_COVERAGE = 0.80

#: Below this share, concept evidence cannot keep an otherwise weak result from
#: being an honest no-match.
NO_MATCH_CONCEPT_COVERAGE = 0.50

#: Terms appearing in more of the indexed file universe than this are context,
#: not discriminating subject evidence.  Intent/grammar words are removed
#: separately; this catches corpus-specific common nouns without a repo-specific
#: stop-word list.
MAX_CONCEPT_FILE_FRACTION = 0.20

#: A percentage is meaningful only on a real corpus.  Tiny fixtures and newly
#: initialized repositories keep their terms; otherwise every one-file term is
#: simultaneously "100% common" and the profile contains no evidence at all.
MIN_FILES_FOR_FREQUENCY_FILTER = 20

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

#: The four retrieval legs, named once so a failure report, a reason string and
#: a test all spell them the same way.
LEG_SOURCE_DENSE = "source dense"
LEG_WIKI_DENSE = "wiki dense"
LEG_SOURCE_LEXICAL = "source lexical"
LEG_WIKI_LEXICAL = "wiki lexical"

#: Not a lane of its own — it hydrates chunks the lexical leg found alone — so
#: losing it degrades the evidence without losing the corpus.
LEG_CHUNK_METADATA = "source chunk metadata"

#: Corpus/file co-location evidence used only for confidence.  Losing it does
#: not erase retrieval, but it makes a confident claim impossible.
LEG_SOURCE_EVIDENCE = "source confidence evidence"

RETRIEVAL_LEGS = frozenset({LEG_SOURCE_DENSE, LEG_WIKI_DENSE, LEG_SOURCE_LEXICAL, LEG_WIKI_LEXICAL})

#: Errors that mean "slow", not "broken". A store that timed out is expected to
#: answer the next query, so it keeps the swallow-and-carry-on idiom the stock
#: retrievers use. Everything else — a schema mismatch, an IO error, an
#: AttributeError from a bad call — is a condition that will still be true on
#: the next query and every query after it, and must reach the caller.
#:
#: ``CancelledError`` is not here and must not be: it derives from
#: ``BaseException``, so ``except Exception`` never sees it and a cancelled
#: request stays cancelled.
_SOFT_LEG_ERRORS: tuple[type[Exception], ...] = (TimeoutError,)

#: How much of an exception's message travels in the envelope. Enough to
#: recognise the fault, bounded because it reaches an agent's context window.
_FAILURE_DETAIL_CHARS = 160

#: Error code for a search that reached no corpus at all.
ERROR_ALL_LEGS_FAILED = "source_search_unavailable"


@dataclass(frozen=True, slots=True)
class LegFailure:
    """One retrieval leg that did not answer, and whether it can be expected to.

    *hard* is the whole point. A soft failure is a slow store and the response
    is simply thinner; a hard failure means the response is describing a corpus
    it could not read, and saying nothing about that is how a broken index
    spends twenty-one minutes looking like an empty one.
    """

    leg: str
    error: str
    detail: str
    hard: bool

    def to_dict(self) -> dict[str, Any]:
        return {"leg": self.leg, "error": self.error, "detail": self.detail}


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

# Raw runs are retained long enough for a CamelCase service/identifier such as
# ``CodeAtlas`` to contribute its path spelling (``codeatlas``).  The FTS
# tokenizer's split parts are still used for ordinary words.
_QUERY_TOKEN_RUN = re.compile(r"[A-Za-z0-9]+")

#: How an item came by its exact-name evidence. The whole query *being* the
#: identifier is a stronger claim than the query merely carrying it, and the two
#: reorder by different rules, so the response says which one it was.
EXACT_VIA_QUERY = "query"
EXACT_VIA_EMBEDDED = "embedded"

#: A query that names test material. On its own this is a query *about* tests,
#: and demoting them would be answering a different question.
_TEST_QUERY_RE = re.compile(
    r"\b(test|tests|testing|pytest|fixture|fixtures|mock|mocks|spec|specs)\b",
    re.IGNORECASE,
)

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

#: A request for explanatory/generated prose rather than the implementation
#: artifact itself.  This is intentionally explicit: ordinary "how does it
#: work" behavior questions still ask for source ownership.
_DOCS_QUERY_RE = re.compile(
    r"\b(doc|docs|documentation|documented|wiki|readme|guide|overview|"
    r"explain|explains|explained|explanation)\b",
    re.IGNORECASE,
)

#: Operational artifacts are declarations in their own right.  These tokens
#: keep a compose/YAML/shell/config window from losing to a nearby code symbol
#: merely because only the latter has a language grammar.
_OPERATIONAL_QUERY_RE = re.compile(
    r"\b(compose|dockerfile|makefile|ya?ml|shell|bash|powershell|script|"
    r"config|configuration|service|timer|healthcheck|health[ -]check|storage|file)\b",
    re.IGNORECASE,
)

#: Query shapes that ask for the implementation owner.  Repair is carried as a
#: separate typed bit and folded into this one by :func:`_query_intent`.
_IMPLEMENTATION_QUERY_RE = re.compile(
    r"\b(how|where|why|implement|implements|implemented|implementation|"
    r"behavio[u]?r|behaves|source|code|logic|handler|function|class|method|"
    r"define|defines|defined|definition|declaration|owner|owns)\b",
    re.IGNORECASE,
)

#: Symbol kinds that define a thing rather than mention it. Preferred as owners
#: when the fusion cannot separate two results on score.
_DEFINITION_KINDS = frozenset({"class", "function", "method", "interface", "struct", "type"})

# Query framing and generic programming nouns do not identify the subject.  A
# corpus-frequency filter below removes repository-specific common terms; this
# set removes language whose frequency can look deceptively low in source text
# ("which", "where", "failing") despite carrying no ownership evidence.
_INTENT_TOKENS = frozenset(
    {
        "about",
        "actual",
        "adapter",
        "after",
        "against",
        "and",
        "are",
        "around",
        "as",
        "assembled",
        "back",
        "before",
        "build",
        "builds",
        "behavior",
        "behaviors",
        "behaves",
        "behaviour",
        "behaviours",
        "broken",
        "by",
        "code",
        "collapse",
        "collapsed",
        "configure",
        "configures",
        "control",
        "controls",
        "cover",
        "covers",
        "debug",
        "decide",
        "decides",
        "does",
        "during",
        "emit",
        "emits",
        "event",
        "events",
        "failing",
        "fails",
        "failure",
        "file",
        "files",
        "find",
        "filtering",
        "fix",
        "for",
        "from",
        "function",
        "generate",
        "generated",
        "handler",
        "how",
        "in",
        "implementation",
        "implemented",
        "index",
        "indexed",
        "indexing",
        "integrate",
        "integrates",
        "inside",
        "into",
        "is",
        "its",
        "layer",
        "leave",
        "locate",
        "mine",
        "mines",
        "module",
        "normalize",
        "of",
        "old",
        "on",
        "one",
        "or",
        "onto",
        "owner",
        "owns",
        "persist",
        "persisted",
        "persists",
        "process",
        "processes",
        "pytest",
        "regression",
        "render",
        "renders",
        "replace",
        "replaces",
        "report",
        "reports",
        "represent",
        "represents",
        "resolve",
        "resolves",
        "restore",
        "restored",
        "return",
        "returns",
        "route",
        "script",
        "select",
        "selects",
        "service",
        "shell",
        "show",
        "state",
        "status",
        "store",
        "stores",
        "switch",
        "system",
        "test",
        "tested",
        "testing",
        "tests",
        "that",
        "the",
        "their",
        "them",
        "this",
        "through",
        "to",
        "under",
        "update",
        "updates",
        "used",
        "uses",
        "using",
        "very",
        "what",
        "when",
        "where",
        "whether",
        "which",
        "while",
        "with",
        "without",
        "write",
        "writer",
        "writes",
        "written",
        "work",
        "works",
    }
)

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


@dataclass(frozen=True, slots=True)
class _ConceptEvidence:
    """One informative query concept, measured over the indexed file universe."""

    token: str
    document_frequency: int
    weight: float
    matched: bool
    #: Whether the candidate carries this concept in something other than its
    #: own file path.  A chunk's indexed text opens with ``# File: <path>`` and
    #: a dotted qualified name that restates that same path, so every chunk in
    #: ``…/solemd_retirement_cashflow_chart_controller.js`` matches the token
    #: ``cashflow`` no matter what it contains.  Harmless while chunks were fat
    #: enough for their bodies to outweigh that header; decisive once A3 added
    #: two-line local symbols, where the repeated path is most of the token
    #: stream and BM25 is effectively scoring the filename.  Ranking may still
    #: use the permissive match — a cashflow file *is* a reasonable hit for a
    #: cashflow query — but a confident ownership claim may not, because the
    #: path is a fact about the file and the claim is about this chunk.
    content_carried: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "document_frequency": self.document_frequency,
            "matched": self.matched,
            "content_carried": self.content_carried,
        }


@dataclass(frozen=True, slots=True)
class _CandidateEvidence:
    """All confidence evidence bound to one candidate and its owner file."""

    dense_cosine: float | None
    dense_rank: int | None
    lexical_rank: int | None
    exact_via: str
    suffix_match: bool
    same_path_corroborated: bool
    corpus_file_count: int
    concepts: tuple[_ConceptEvidence, ...]

    @property
    def exact_name(self) -> bool:
        return bool(self.exact_via)

    @property
    def concept_coverage(self) -> float:
        total = sum(concept.weight for concept in self.concepts)
        if total <= 0:
            return 0.0
        matched = sum(concept.weight for concept in self.concepts if concept.matched)
        return matched / total

    @property
    def content_concept_coverage(self) -> float:
        """Coverage counting only concepts the candidate carries itself.

        The confidence gate reads this rather than :attr:`concept_coverage`.
        Retrieval is allowed to believe a filename; a confident ownership claim
        has to be able to point at the chunk.
        """
        total = sum(concept.weight for concept in self.concepts)
        if total <= 0:
            return 0.0
        matched = sum(
            concept.weight
            for concept in self.concepts
            if concept.matched and concept.content_carried
        )
        return matched / total

    def to_dict(self, *, lane: str) -> dict[str, Any]:
        return {
            "dense_cosine": (
                round(self.dense_cosine, 4) if self.dense_cosine is not None else None
            ),
            "lexical_rank": self.lexical_rank,
            "exact_name": self.exact_name,
            "lane": lane,
            "concept_coverage": round(self.concept_coverage, 4),
            "content_concept_coverage": round(self.content_concept_coverage, 4),
            "corpus_file_count": self.corpus_file_count,
            "same_path_corroborated": self.same_path_corroborated,
            "concepts": [concept.to_dict() for concept in self.concepts],
        }


@dataclass(frozen=True, slots=True)
class QueryIntent:
    """One shared, typed reading of a source-search query.

    Ranking and owner selection consume this same object.  Keeping intent in a
    named value prevents the old failure mode where test demotion, exact routing,
    and owner selection each re-parsed the sentence with subtly different rules.
    """

    identifier: str | None
    embedded_identifiers: tuple[str, ...]
    exact_target: str | None
    wants_tests: bool
    repair: bool
    docs: bool
    operational: bool
    implementation: bool

    @property
    def exact_artifact(self) -> bool:
        """Whether the complete query names a path or file-qualified symbol."""

        return self.exact_target is not None

    @property
    def is_prose(self) -> bool:
        """Whether semantic subject evidence, rather than any identifier, applies."""

        return (
            self.identifier is None and not self.embedded_identifiers and self.exact_target is None
        )


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
    #: Stable source chunk id or wiki target id.  Kept separately from ``key``
    #: so exact file-qualified ownership never has to parse a lane prefix.
    target_id: str = ""
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
    #: A wiki page's own primary key, carried verbatim from the retriever.
    #: Empty for a source chunk, which is identified by its file and lines.
    page_id: str = ""
    fused_score: float = 0.0
    evidence_profile: _CandidateEvidence | None = None

    @property
    def exact_name(self) -> bool:
        """Whether an identifier matched this item's name, however it was found."""
        return bool(self.exact_via)

    def evidence(self) -> dict[str, Any]:
        if self.evidence_profile is not None:
            return self.evidence_profile.to_dict(lane=self.lane)
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
        if self.lane == LANE_WIKI and self.page_id:
            # The only identity a page type outside ``_FILE_BACKED_PAGE_TYPES``
            # has. A module page, an SCC page or a repo overview serves
            # ``file: ""`` by design — it names no file to open — and without
            # this a consumer holds a title it cannot resolve to anything.
            #
            # Carried verbatim, never rebuilt. A page id *looks* like
            # ``f"{page_type}:{target_path}"`` and rebuilding it from the two
            # fields beside it is wrong in a way that does not announce itself:
            # this response serves the file a symbol spotlight names, not its
            # ``a/b.py::Foo`` target, so the rebuild yields a different,
            # perfectly well-formed id belonging to another page.
            out["page_id"] = self.page_id
        return out


def _is_test_query(query: str) -> bool:
    """Whether *query* asks about tests, so test material ranks naturally.

    Naming a test is not the same as asking for one. A repair-shaped query
    names the failing test because that is where the failure showed up, and
    what it wants is the code under it — so the two signals together mean the
    opposite of the test signal alone.
    """
    return _query_intent(query).wants_tests


def _query_intent(query: str) -> QueryIntent:
    """Parse *query* once for every ranking and owner-policy consumer."""

    identifier = _identifier_query(query)
    repair = bool(_REPAIR_QUERY_RE.search(query))
    docs = bool(_DOCS_QUERY_RE.search(query))
    operational = bool(_OPERATIONAL_QUERY_RE.search(query))
    return QueryIntent(
        identifier=identifier,
        embedded_identifiers=tuple(_embedded_identifiers(query)) if identifier is None else (),
        exact_target=_exact_target_query(query),
        wants_tests=bool(_TEST_QUERY_RE.search(query)) and not repair,
        repair=repair,
        docs=docs,
        operational=operational,
        implementation=(repair or bool(_IMPLEMENTATION_QUERY_RE.search(query))) and not docs,
    )


def _exact_target_query(query: str) -> str | None:
    """Return a normalized exact path/file-qualified ID, never a prose mention.

    Exactness is deliberately whole-query-only.  A sentence containing
    ``config.py`` has mentioned a filename-like word; it has not asserted that
    file as the answer.  Both POSIX and Windows separators are accepted because
    the stored corpus uses repository-relative POSIX paths on every platform.
    """

    stripped = query.strip()
    if not stripped or any(character.isspace() for character in stripped):
        return None
    normalized = stripped.replace("\\", "/")
    path, separator, symbol = normalized.partition("::")
    while path.startswith("./"):
        path = path[2:]
    if not path or not _looks_like_source_path(path):
        return None
    return path + (f"::{symbol}" if separator and symbol else "")


def _looks_like_source_path(value: str) -> bool:
    """Whether a whole-query token has the shape of an indexed source path."""

    if "/" in value:
        return True
    return bool(language_for_path(value)) or window_eligible(value, indexed_symbols=0)


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


def _concept_tokens(query: str) -> tuple[str, ...]:
    """Informative query concepts in the source index's own token language.

    There is deliberately no second stemmer here.  Evidence normalized
    differently from the corpus cannot establish that the corpus contains the
    query subject.  Corpus-common terms are removed later, once their exact
    active-generation document frequency is known.
    """

    concepts: list[str] = []
    for raw in _QUERY_TOKEN_RUN.findall(query):
        split = tokenize(raw)
        has_camel_boundary = bool(_CAMEL_BOUNDARY.search(raw))
        candidates = [raw.lower()] if has_camel_boundary and len(split) > 1 else split
        concepts.extend(token for token in candidates if token not in _INTENT_TOKENS)
    return tuple(dict.fromkeys(concepts))


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
        self._path_token_files: dict[str, frozenset[str]] | None = None

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
        intent = _query_intent(query)
        items, failures = await self._retrieve(query)
        hard = [failure for failure in failures if failure.hard]
        if hard and _all_legs_lost(hard, self._attempted_legs()):
            # Nothing was read. An empty result set here would be a claim about
            # the repository, and the only true statement available is that the
            # search did not happen.
            return self._error_envelope(query, mode, limit, hard, started, base_meta)
        # Read before the deduplication, which keeps one item per file and so
        # can only ever leave one lane standing for it. Corroboration is a
        # question about what was *retrieved*, not about what survived ranking.
        source_files = {
            item.file for item in items.values() if item.lane == LANE_SOURCE and item.file
        }
        ranked = self._rank(items, intent)
        evidence_failure = None
        if not any(failure.hard and failure.leg == LEG_SOURCE_LEXICAL for failure in failures):
            evidence_failure = self._profile_candidates(query, ranked, source_files)
        if evidence_failure is not None:
            failures.append(evidence_failure)
        hard = [failure for failure in failures if failure.hard]
        deduped = self._dedupe_by_file(ranked)
        window = deduped[:limit]
        owner, reason = self._select_owner(window, intent=intent)
        if owner is not None and deduped and deduped[0] is not owner:
            # The owner is the answer, so it leads the list the caller reads
            # and the candidate paths derived from it. Identity, not equality:
            # two results can hold identical fields and be different answers.
            deduped = [owner, *(item for item in deduped if item is not owner)]
            window = deduped[:limit]
        confidence = self._classify(window, owner, source_files)
        if hard:
            # Pinned, not capped. Capping alone would leave ``no_match``
            # untouched, and ``no_match`` is the one answer a half-read corpus
            # is least entitled to give: it asserts an absence, which is
            # exactly what a silently dead lane manufactures.
            confidence = CAUTION
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
            hard_failures=hard,
        )
        self._record(query, mode, limit, latency_ms, confidence, window, owner)
        return response

    # -- retrieval --------------------------------------------------------

    def _attempted_legs(self) -> set[str]:
        """The retrieval legs this deployment actually has a store for.

        A missing wiki index is a configuration, not a fault, so it must never
        count towards "everything failed".
        """
        legs = {LEG_SOURCE_DENSE, LEG_SOURCE_LEXICAL}
        if self._wiki_vectors is not None:
            legs.add(LEG_WIKI_DENSE)
        if self._wiki_fts is not None:
            legs.add(LEG_WIKI_LEXICAL)
        return legs

    async def _retrieve(self, query: str) -> tuple[dict[str, _Item], list[LegFailure]]:
        """Both legs, fused. Returns candidates and every leg that did not answer."""
        dense_source, dense_wiki, dense_failures = await self._dense_legs(query)
        lexical_source, lexical_wiki, lexical_failures = await self._lexical_legs(query)
        failures = [*dense_failures, *lexical_failures]

        items: dict[str, _Item] = {}
        records, records_failure = await self._chunk_records(dense_source, lexical_source)
        if records_failure is not None:
            failures.append(records_failure)

        merged_dense = _merge_dense(dense_source, dense_wiki)
        for rank, (lane, hit, key) in enumerate(merged_dense, start=1):
            item = self._item_for(items, lane, hit, key, records)
            if item is None:
                continue
            item.dense_rank = rank
            item.dense_cosine = float(hit.score)

        for rank, (lane, hit, key) in enumerate(_merge_lexical(lexical_source, lexical_wiki), 1):
            item = self._item_for(items, lane, hit, key, records)
            if item is None:
                continue
            item.lexical_rank = rank
            # The lexical leg cuts its snippet around what matched; the dense
            # leg has no query text at the point its row was written and can
            # only serve the stored opener. Prefer the one that shows evidence.
            snippet = str(getattr(hit, "snippet", None) or "")
            if lane == LANE_WIKI and snippet:
                item.snippet = snippet

        for item in items.values():
            item.fused_score = _rrf(item.dense_rank, item.lexical_rank)
        return items, failures

    async def _dense_legs(self, query: str) -> tuple[list[Any], list[Any], list[LegFailure]]:
        """Top-*LEG_FETCH* from each vector store, on one embedding of *query*.

        An embedder that raises takes both dense legs with it, so the failure
        is recorded against each leg that had a store rather than against the
        embedder alone — that is what "the dense lane is gone" means to
        everything downstream, and it keeps the total-failure test honest.
        """
        try:
            vectors = await self._embedder.embed([query])
        except Exception as exc:
            failure = _classify_failure("query embedding", exc)
            legs = [LEG_SOURCE_DENSE] + ([LEG_WIKI_DENSE] if self._wiki_vectors else [])
            return (
                [],
                [],
                [
                    LegFailure(
                        leg=leg,
                        error=failure.error,
                        detail=f"query embedding failed: {failure.detail}",
                        hard=failure.hard,
                    )
                    for leg in legs
                ],
            )
        if not vectors:
            return [], [], []
        vector = [float(v) for v in vectors[0]]
        (source, source_failure), (wiki, wiki_failure) = await asyncio.gather(
            _run_leg(
                self._source_vectors.search_by_vector(vector, limit=LEG_FETCH), LEG_SOURCE_DENSE
            ),
            _run_leg(
                self._wiki_vectors.search_by_vector(vector, limit=LEG_FETCH)
                if self._wiki_vectors is not None
                else None,
                LEG_WIKI_DENSE,
            ),
        )
        return source, wiki, [f for f in (source_failure, wiki_failure) if f is not None]

    async def _lexical_legs(self, query: str) -> tuple[list[Any], list[Any], list[LegFailure]]:
        """Top-*LEG_FETCH* from each BM25 index.

        The source index tokenizes *query* with the same function that built
        its token stream — that is :meth:`SourceFTSIndex.query`'s contract, and
        the reason a query is handed to it as free text rather than pre-split.
        """
        source, source_failure = _run_sync_leg(
            lambda: self._source_fts.query(query, limit=LEG_FETCH), LEG_SOURCE_LEXICAL
        )
        wiki, wiki_failure = await _run_leg(
            self._wiki_fts.search(query, limit=LEG_FETCH) if self._wiki_fts is not None else None,
            LEG_WIKI_LEXICAL,
        )
        return source, wiki, [f for f in (source_failure, wiki_failure) if f is not None]

    async def _chunk_records(
        self, dense_source: Sequence[Any], lexical_source: Sequence[Any]
    ) -> tuple[dict[str, Any], LegFailure | None]:
        """Metadata for every source chunk the lexical leg found alone.

        A BM25 hit is an id and a file path. The dense leg already carries the
        rest, so only the ids it did not return are fetched, in one query.
        """
        known = {hit.chunk_id for hit in dense_source}
        missing = [hit.chunk_id for hit in lexical_source if hit.chunk_id not in known]
        if not missing:
            return {}, None
        try:
            return await self._source_vectors.fetch_by_chunk_ids(missing), None
        except Exception as exc:
            return {}, _classify_failure(LEG_CHUNK_METADATA, exc)

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
            target_id=str(getattr(hit, "chunk_id", "") or getattr(record, "chunk_id", "")),
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
            target_id=target_path,
            match_name=symbol.rsplit(".", 1)[-1] if sep else "",
            kind=page_type,
            snippet=getattr(hit, "snippet", "") or "",
            source=SOURCE_WIKI_PAGE,
            page_id=str(getattr(hit, "page_id", "") or ""),
            is_test=_is_test_related(file_path),
        )

    # -- owner-bound evidence ---------------------------------------------

    def _profile_candidates(
        self,
        query: str,
        ranked: Sequence[_Item],
        source_files: set[str],
    ) -> LegFailure | None:
        """Attach one immutable, co-located evidence profile to every item.

        The source index answers two facts from the same active generation:
        which files contain each query concept, and how rare that concept is
        across the corpus.  A selected file must contain the complete subject
        itself; evidence from different files is never pooled.  File-path
        tokens join the indexed body/name evidence because operational owners
        often declare themselves in their path rather than in an AST symbol.
        """

        raw_concepts = _concept_tokens(query)
        try:
            active_paths = self._source_fts.active_file_paths()
            indexed_term_files = self._source_fts.term_file_evidence(raw_concepts)
        except Exception as exc:
            return _classify_failure(LEG_SOURCE_EVIDENCE, exc)

        if self._path_token_files is None:
            mutable: dict[str, set[str]] = {}
            for path in active_paths:
                for token in set(tokenize(path)):
                    mutable.setdefault(token, set()).add(path)
            self._path_token_files = {token: frozenset(paths) for token, paths in mutable.items()}

        corpus_file_count = max(len(active_paths), 1)
        concept_files: dict[str, frozenset[str]] = {}
        for token in raw_concepts:
            files = frozenset(
                set(indexed_term_files.get(token, frozenset()))
                | set(self._path_token_files.get(token, frozenset()))
            )
            if (
                corpus_file_count < MIN_FILES_FOR_FREQUENCY_FILTER
                or len(files) / corpus_file_count <= MAX_CONCEPT_FILE_FRACTION
            ):
                concept_files[token] = files

        lanes_by_file: dict[str, set[str]] = {}
        for item in ranked:
            if item.file:
                lanes_by_file.setdefault(item.file, set()).add(item.lane)

        for item in ranked:
            own_tokens = set(tokenize(" ".join((item.file, item.match_name, item.snippet))))
            # Split out what the candidate's own path supplies.  A file window
            # *is* its file, so for one of those the path is the subject rather
            # than an accident of location; for a symbol chunk it is neither.
            path_tokens = set(tokenize(item.file))
            # ``match_name`` only, never falling back to ``name``.  A wiki
            # page's name is its title — "File: a/b/cashflow_controller.js" —
            # which restates the path just as the chunk header does, so the
            # fallback would hand the path straight back as though the
            # candidate had earned it.
            name_tokens = set(tokenize(item.match_name))
            file_level = item.source == SOURCE_FILE_WINDOW
            concepts: list[_ConceptEvidence] = []
            for token, files in concept_files.items():
                document_frequency = len(files)
                weight = math.log((corpus_file_count + 1) / (document_frequency + 1)) + 1.0
                matched = (bool(item.file) and item.file in files) or token in own_tokens
                content_carried = matched and (
                    file_level or token not in path_tokens or token in name_tokens
                )
                concepts.append(
                    _ConceptEvidence(
                        token=token,
                        document_frequency=document_frequency,
                        weight=weight,
                        matched=matched,
                        content_carried=content_carried,
                    )
                )
            item.evidence_profile = _CandidateEvidence(
                dense_cosine=item.dense_cosine,
                dense_rank=item.dense_rank,
                lexical_rank=item.lexical_rank,
                exact_via=item.exact_via,
                suffix_match=item.suffix_match,
                same_path_corroborated=(
                    bool(item.file)
                    and item.file in source_files
                    and len(lanes_by_file.get(item.file, set())) > 1
                ),
                corpus_file_count=corpus_file_count,
                concepts=tuple(concepts),
            )
        return None

    # -- ranking ----------------------------------------------------------

    def _rank(self, items: dict[str, _Item], intent: QueryIntent) -> list[_Item]:
        """Fused order, then test demotion, then the exact-identifier router."""
        ranked = sorted(items.values(), key=_fused_sort_key)
        if not intent.wants_tests and not intent.repair:
            # Stable, so the fused order survives inside each group. Demotion,
            # not removal: a test is rarely the best first read, and sometimes
            # it is the only place a behaviour is written down.
            ranked.sort(key=lambda item: item.is_test)
        if intent.identifier:
            return self._route_exact(intent.identifier, ranked)
        if intent.embedded_identifiers:
            return self._route_embedded(intent.embedded_identifiers, ranked)
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
    def _select_owner(window: Sequence[_Item], *, intent: QueryIntent) -> tuple[_Item | None, str]:
        """Select an owner through named, intent-bounded policy stages.

        Preference applies only inside :data:`OWNER_SCORE_BAND` of the best
        fused score. Outside it the fusion is making a claim the shape rules
        should not overturn: a wiki page that beat every chunk by a clear
        margin is the answer, whatever its shape.

        Stages are deliberately explicit rather than encoded in one tuple.  A
        stage narrows only when its query shape applies and a matching candidate
        exists.  Retrieval score then decides among the survivors; path, line,
        and stable key are used only for a true score tie.
        """
        if not window:
            return None, ""
        best = window[0].fused_score
        floor = best * (1.0 - OWNER_SCORE_BAND)
        band = [item for item in window if item.fused_score >= floor] or [window[0]]
        candidates = [item for item in band if item.file] or band
        applied_rule = ""

        def narrow(rule: str, matching: Sequence[_Item]) -> None:
            nonlocal candidates, applied_rule
            selected = list(matching)
            if not selected or len(selected) == len(candidates):
                return
            displaced_incumbent = candidates[0] not in selected
            candidates = selected
            if displaced_incumbent and not applied_rule:
                applied_rule = rule

        # 1. An exact artifact names its owner directly.  Whole-query matching
        # is what keeps a filename-like word in prose from claiming this rule.
        narrow(
            "exact path/full-ID owner",
            [item for item in candidates if _exact_owner_match(item, intent)],
        )

        # 2–3. Test material is the owner only when tests are the subject.  A
        # repair query names a failing test as the observation site instead.
        if intent.wants_tests:
            narrow("explicit test owner", [item for item in candidates if item.is_test])
        elif intent.repair:
            narrow("symptom test demotion", [item for item in candidates if not item.is_test])

        # 4. Complete subject evidence must be co-located on the proposed owner.
        # A small relative advantage is not called "complete": the same 0.80
        # floor that licenses confident prose is the minimum semantic boundary.
        if intent.is_prose and not intent.wants_tests:
            complete = [
                item
                for item in candidates
                if item.evidence_profile is not None
                and item.evidence_profile.concept_coverage >= CONFIDENT_CONCEPT_COVERAGE
            ]
            if complete:
                best_coverage = max(
                    item.evidence_profile.concept_coverage  # type: ignore[union-attr]
                    for item in complete
                )
                narrow(
                    "co-located subject completeness",
                    [
                        item
                        for item in complete
                        if item.evidence_profile is not None
                        and math.isclose(
                            item.evidence_profile.concept_coverage,
                            best_coverage,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ],
                )

        # 5. Generated prose corroborates implementation; it does not own an
        # implementation-shaped request.  Explicit docs/wiki intent opts out.
        if intent.implementation and not intent.docs:
            narrow(
                "source-owner bias",
                [item for item in candidates if item.lane == LANE_SOURCE],
            )

        # 6. A declaration can own behavior only when it carries the subject.
        # Operational artifacts, docs, and explicit tests each have a stronger
        # query-specific owner class and deliberately bypass this stage.
        declaration_shape = bool(intent.identifier or intent.embedded_identifiers) or (
            intent.implementation and not intent.operational
        )
        if declaration_shape and not intent.docs and not intent.wants_tests:
            if intent.identifier or intent.embedded_identifiers:
                # A symbol spotlight is an exact declaration proxy even though
                # its lane is generated prose.  Per-file dedupe may have kept it
                # in place of the same file's source chunk; rejecting it here
                # would discard the strongest name evidence before the later
                # citation upgrade can put the owner back on source lines.
                exact_declarations = [item for item in candidates if item.exact_name]
                suffix_declarations = [item for item in candidates if item.suffix_match]
            else:
                exact_declarations = []
                suffix_declarations = []
            declarations = exact_declarations or suffix_declarations
            if not declarations:
                declarations = [
                    item
                    for item in candidates
                    if item.source == SOURCE_SYMBOL
                    and item.kind in _DEFINITION_KINDS
                    and _declaration_carries_subject(item)
                ]
            narrow("declaration over usage", declarations)

        # 7. A raw operational file is the declaration-equivalent owner for its
        # own behavior.  A bare code identifier that happens to be named
        # ``healthcheck`` is still a symbol lookup, not operational prose.
        if intent.operational and intent.identifier is None and not intent.docs:
            narrow(
                "operational file preservation",
                [item for item in candidates if item.source == SOURCE_FILE_WINDOW],
            )

        winner = _policy_tie_break(candidates)
        if (
            not applied_rule
            and winner is not candidates[0]
            and math.isclose(
                winner.fused_score,
                candidates[0].fused_score,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            applied_rule = "deterministic tie"
        reason = f"owner policy: {applied_rule}" if applied_rule else _owner_evidence_reason(winner)

        displaced = window[0]
        if displaced is not winner:
            reason += f"; preferred over a closely-scored {_shape_phrase(displaced)}"
        return winner, reason

    @staticmethod
    def _classify(
        window: Sequence[_Item],
        owner: _Item | None,
        source_files: set[str],
    ) -> str:
        """Confidence from the selected owner's own absolute evidence.

        Evidence from another candidate cannot make *owner* trustworthy.  The
        previous implementation pooled the best dense score, any exact-name
        hit, and file-level lexical agreement across the whole result window.
        That let one file win ownership on shape while a different file's
        evidence silently granted it confidence.  This is a correctness bug,
        not a threshold problem: the claim and the evidence must name the same
        candidate.

        The conservative ``no_match`` decision still considers the complete
        window.  It is a claim that the corpus has no plausible answer, not a
        claim about which candidate should own one.
        """
        if not window or owner is None:
            return NO_MATCH
        profiles = [item.evidence_profile for item in window if item.evidence_profile is not None]
        if owner.evidence_profile is None or len(profiles) != len(window):
            # Retrieval succeeded but the co-location evidence did not.  That
            # can support a candidate list, never a confident ownership claim
            # or a corpus-wide absence.
            return CAUTION
        cosines = [item.dense_cosine for item in window if item.dense_cosine is not None]
        best_dense = max(cosines) if cosines else 0.0
        any_exact = any(profile.exact_name for profile in profiles)
        best_coverage = max((profile.concept_coverage for profile in profiles), default=0.0)
        meaningful_lexical = any(
            profile.lexical_rank is not None
            and profile.concept_coverage >= NO_MATCH_CONCEPT_COVERAGE
            for profile in profiles
        )
        profile = owner.evidence_profile
        owner_dense = owner.dense_cosine or 0.0

        if (
            best_dense < NO_MATCH_DENSE_COSINE
            and not any_exact
            and best_coverage < NO_MATCH_CONCEPT_COVERAGE
            and not meaningful_lexical
        ):
            return NO_MATCH
        if _uncorroborated_page(source_files, owner):
            return CAUTION
        if profile.exact_name and owner.file:
            return CONFIDENT
        lexical_agreement = (
            profile.lexical_rank is not None and profile.lexical_rank <= AGREEMENT_LEXICAL_DEPTH
        )
        # A filename may corroborate a subject the candidate already carries; it
        # may not *be* the subject.  Coverage keeps its permissive reading, so a
        # file honestly named for its topic still counts — ``identity.py`` is
        # real evidence about identity.  What is refused is a candidate whose
        # entire case is its path: a two-line helper in a richly-named file, or
        # that file's wiki page, matching every concept it was asked about while
        # containing none of them.  Requiring one concept to be carried by the
        # candidate itself separates the two without punishing the first.
        complete_subject = (
            profile.concept_coverage >= CONFIDENT_CONCEPT_COVERAGE
            and profile.content_concept_coverage > 0.0
        )
        if complete_subject and lexical_agreement and owner_dense >= CONFIDENT_DENSE_COSINE:
            return CONFIDENT
        if (
            complete_subject
            and lexical_agreement
            and owner_dense >= AGREEMENT_DENSE_COSINE
            and profile.same_path_corroborated
        ):
            return CONFIDENT
        return CAUTION

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
        hard_failures: Sequence[LegFailure] = (),
    ) -> dict[str, Any]:
        meta = dict(base_meta or {})
        meta["timing_ms"] = round(latency_ms, 2)
        source_meta = self._source_meta()
        if hard_failures:
            # Only when something is wrong: a healthy response is byte-identical
            # to the one this served before degradation reporting existed.
            source_meta["degraded"] = True
            source_meta["degraded_reason"] = _degraded_reason(hard_failures)
            source_meta["failed_legs"] = [failure.to_dict() for failure in hard_failures]
        meta["source_search"] = source_meta

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
            {"file": owner.file, "reason": reason, "evidence": owner.evidence()}
            if owner is not None and owner.file
            else None
        )
        if confidence == NO_MATCH:
            response["note"] = (
                f"No indexed match for {query!r}. The corpus covers this repository at "
                "the commit named in _meta.source_search — a question outside it has no "
                "answer here, and the results below (if any) are nearest neighbours, not "
                "evidence."
            )
        return response

    def _error_envelope(
        self,
        query: str,
        mode: str,
        limit: int,
        hard_failures: Sequence[LegFailure],
        started: float,
        base_meta: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """The response for a search that reached no corpus at all.

        Deliberately not an empty result set. Zero results is a statement about
        the repository — "nothing here matches" — and this response cannot make
        it, because it read nothing. It carries ``status: "error"`` so no
        consumer can mistake it for an answer, and ``confidence: "caution"``
        rather than ``no_match`` for the same reason: ``no_match`` is the
        assertion that would repeat the incident this exists to prevent.
        """
        reason = _degraded_reason(hard_failures)
        latency_ms = (time.perf_counter() - started) * 1000.0
        log.error(
            "source-search served no corpus: every retrieval leg failed (%s)",
            reason,
            extra={"query": query, "failed_legs": [f.leg for f in hard_failures]},
        )
        meta = dict(base_meta or {})
        meta["timing_ms"] = round(latency_ms, 2)
        source_meta = self._source_meta()
        source_meta["degraded"] = True
        source_meta["degraded_reason"] = reason
        source_meta["failed_legs"] = [failure.to_dict() for failure in hard_failures]
        meta["source_search"] = source_meta

        response: dict[str, Any] = {
            "results": [],
            "candidates": [],
            "selected_owner": None,
            "confidence": CAUTION,
            "mode": mode,
            "status": "error",
            "error": {
                "code": ERROR_ALL_LEGS_FAILED,
                "message": (
                    "Every retrieval leg failed, so this search read no corpus. "
                    "The result set is empty because nothing was searched, NOT "
                    f"because nothing matched. Cause: {reason}"
                ),
                "failed_legs": [failure.to_dict() for failure in hard_failures],
            },
            "_meta": meta,
        }
        self._record(query, mode, limit, latency_ms, CAUTION, [], None)
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
                        concept_coverage=(
                            round(item.evidence_profile.concept_coverage, 4)
                            if item.evidence_profile is not None
                            else None
                        ),
                        same_path_corroborated=(
                            item.evidence_profile.same_path_corroborated
                            if item.evidence_profile is not None
                            else None
                        ),
                    )
                    for item in window
                ],
                selected_owner_file=owner.file if owner is not None else None,
                selected_owner_evidence=owner.evidence() if owner is not None else None,
                no_match=confidence == NO_MATCH,
            )
        )


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


def _classify_failure(leg: str, exc: Exception) -> LegFailure:
    """Record *exc* against *leg*, and log it at the volume it deserves.

    A hard failure logs at ERROR with the leg on the record, because it is an
    operational event someone has to see; a timeout logs at WARNING and is
    otherwise carried on from.
    """
    detail = (str(exc).strip() or type(exc).__name__)[:_FAILURE_DETAIL_CHARS]
    hard = not isinstance(exc, _SOFT_LEG_ERRORS)
    extra = {"leg": leg, "error_type": type(exc).__name__, "detail": detail}
    if hard:
        log.error("source-search leg %s failed: %s", leg, detail, exc_info=True, extra=extra)
    else:
        log.warning("source-search leg %s timed out", leg, extra=extra)
    return LegFailure(leg=leg, error=type(exc).__name__, detail=detail, hard=hard)


async def _run_leg(awaitable: Any, leg: str) -> tuple[list[Any], LegFailure | None]:
    """Await *awaitable*, returning its rows and how it failed if it did.

    ``None`` for the awaitable means the leg has no store behind it — absent,
    not broken, and never reported as a failure.
    """
    if awaitable is None:
        return [], None
    try:
        return list(await awaitable), None
    except Exception as exc:
        return [], _classify_failure(leg, exc)


def _run_sync_leg(call: Any, leg: str) -> tuple[list[Any], LegFailure | None]:
    """:func:`_run_leg` for the synchronous FTS index."""
    try:
        return list(call()), None
    except Exception as exc:
        return [], _classify_failure(leg, exc)


def _degraded_reason(failures: Sequence[LegFailure]) -> str:
    """One line naming every leg that hard-failed and what it raised."""
    return "; ".join(f"{f.leg} failed ({f.error}: {f.detail})" for f in failures)


def _all_legs_lost(hard: Sequence[LegFailure], attempted: set[str]) -> bool:
    """Whether every retrieval leg that had a store behind it hard-failed.

    Legs with no store are not counted: a deployment with no wiki index is not
    a broken one. When this is true the response has read nothing at all, and
    reporting an empty result set would be inventing an absence.
    """
    if not attempted:
        return False
    return attempted <= {f.leg for f in hard if f.leg in RETRIEVAL_LEGS}


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


def _exact_owner_match(item: _Item, intent: QueryIntent) -> bool:
    """Whether *item* is the exact path/full-ID owner asserted by *intent*."""

    if intent.exact_target is not None:
        query_path, query_symbol = _target_parts(intent.exact_target)
        item_path, item_symbol = _target_parts(item.target_id)
        normalized_file = item.file.replace("\\", "/").removeprefix("./").casefold()
        if query_path != normalized_file and query_path != item_path:
            return False
        return not query_symbol or query_symbol == item_symbol

    identifier = intent.identifier
    if identifier is None or not ("." in identifier or "::" in identifier):
        return False
    _path, target_symbol = _target_parts(item.target_id)
    return bool(target_symbol) and target_symbol == _norm_identifier(identifier)


def _target_parts(value: str) -> tuple[str, str]:
    """Normalize a stored/query path and optional qualified symbol separately."""

    normalized = value.strip().replace("\\", "/")
    path, separator, symbol = normalized.partition("::")
    while path.startswith("./"):
        path = path[2:]
    return path.casefold(), _norm_identifier(symbol) if separator else ""


def _declaration_carries_subject(item: _Item) -> bool:
    """Whether a declaration has evidence for the subject it would own."""

    if item.exact_name or item.suffix_match:
        return True
    return (
        item.evidence_profile is not None
        and item.evidence_profile.concept_coverage >= CONFIDENT_CONCEPT_COVERAGE
    )


def _policy_tie_break(candidates: Sequence[_Item]) -> _Item:
    """Keep semantic rank; path/line/key only for its true score ties.

    The ranking immediately before owner selection already includes the exact
    router and the generic non-test demotion.  Re-maximizing the raw fused score
    here would silently undo both.  The first surviving candidate is therefore
    the retrieval anchor; deterministic ordering applies only to peers carrying
    that same score.
    """

    best_score = candidates[0].fused_score
    tied = [
        item
        for item in candidates
        if math.isclose(item.fused_score, best_score, rel_tol=0.0, abs_tol=1e-12)
    ]
    return min(
        tied,
        key=lambda item: (
            item.file.casefold(),
            item.start_line if item.start_line is not None else 1 << 30,
            item.key,
        ),
    )


def _owner_evidence_reason(item: _Item) -> str:
    """Describe the evidence when no semantic owner rule displaced retrieval."""

    if item.exact_via == EXACT_VIA_QUERY:
        return "exact name match"
    if item.exact_via == EXACT_VIA_EMBEDDED:
        return "embedded identifier match"
    if item.suffix_match:
        return "embedded identifier suffix match"
    if item.dense_rank is not None and item.lexical_rank is not None:
        return "dense+lexical agreement"
    if item.dense_rank is not None:
        return "dense only"
    return "lexical only"


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
