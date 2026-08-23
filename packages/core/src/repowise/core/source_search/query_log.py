"""What each retrieval was asked, and what evidence it had for its answer.

One JSON object per query, appended to ``.repowise/source_search/query_log.jsonl``.
It records the *evidence* behind a result — which leg found it, at what cosine,
at what lexical rank — rather than the result text, so a later reader can ask
why retrieval was confident without re-running anything. The reader is
:mod:`.query_report`.

The record grows by addition only. A field may be appended with a default that
reproduces the old behaviour; no field may be removed, renamed, or have its
meaning changed, because lines written by an older build stay in the file
forever and the reporter has to keep understanding them.

Every write is fire-and-forget. A search that succeeded and then failed to
record itself has still succeeded, and a full disk, a read-only checkout or a
concurrent writer must not turn into a failed query. :meth:`QueryLog.append`
therefore swallows everything it can raise, and is the only place in this
package that does.

Appends are line-atomic by construction: one ``write()`` of one line under
``O_APPEND``, which is what keeps two processes sharing a repository from
interleaving halves of a record.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .fts import SOURCE_SEARCH_DIRNAME

__all__ = [
    "FAILED_LEGS_LOGGED",
    "QUERY_LOG_FILENAME",
    "STATUS_ERROR",
    "STATUS_OK",
    "TOP_EVENTS_LOGGED",
    "QueryEvent",
    "QueryLog",
    "TopEntry",
    "default_query_log_path",
]

QUERY_LOG_FILENAME = "query_log.jsonl"

#: A search that reached at least one corpus, whatever it then found.
STATUS_OK = "ok"

#: A search that reached no corpus at all, so its empty result set is a
#: statement about the retrieval stack and not about the repository. Mirrors
#: the ``status`` key the coordinator's error envelope puts on the response:
#: the log has to be able to tell the two apart, because on the wire they are
#: otherwise the same record — ``caution``, zero results, no owner.
STATUS_ERROR = "error"

#: How many ranked results each record keeps. Enough to see whether the right
#: answer was in the window and merely mis-ranked — the distinction between a
#: retrieval problem and a ranking problem — and bounded so one query cannot
#: write a kilobyte per call.
TOP_EVENTS_LOGGED = 10

#: How many failed legs one record keeps. There are six named legs, so this
#: cannot truncate a real incident; the bound exists so a caller passing an
#: unbounded list cannot make one line arbitrarily large.
FAILED_LEGS_LOGGED = 8

_log = logging.getLogger(__name__)


def default_query_log_path(repo_path: Path | str) -> Path:
    """Path to *repo_path*'s query log (not created here)."""
    return Path(repo_path) / ".repowise" / SOURCE_SEARCH_DIRNAME / QUERY_LOG_FILENAME


@dataclass(frozen=True, slots=True)
class TopEntry:
    """One ranked result, reduced to the evidence that put it there."""

    file: str
    lane: str
    dense_cosine: float | None
    lexical_rank: int | None
    exact_name: bool
    fused_score: float
    concept_coverage: float | None = None
    same_path_corroborated: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "lane": self.lane,
            "dense_cosine": self.dense_cosine,
            "lexical_rank": self.lexical_rank,
            "exact_name": self.exact_name,
            "fused_score": self.fused_score,
            "concept_coverage": self.concept_coverage,
            "same_path_corroborated": self.same_path_corroborated,
        }


@dataclass(frozen=True, slots=True)
class QueryEvent:
    """One coordinator query, as it goes on the wire.

    ``ts`` defaults to now rather than being passed in, so a caller cannot
    forget it and a test can still pin it.
    """

    query: str
    mode: str
    limit: int
    latency_ms: float
    confidence: str
    result_count: int
    top: list[TopEntry] = field(default_factory=list)
    selected_owner_file: str | None = None
    selected_owner_evidence: dict[str, Any] | None = None
    no_match: bool = False
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="milliseconds"))

    #: Whether this search read a corpus at all. Defaults to :data:`STATUS_OK`
    #: so an existing caller is unchanged and an existing line — which has no
    #: such key — reads back as the healthy value it was written under.
    status: str = STATUS_OK

    #: The coordinator's error code when *status* is :data:`STATUS_ERROR`.
    error_code: str | None = None

    #: The legs that hard-failed, as ``{"leg", "error", "detail"}`` dicts. A
    #: search can serve results with some legs dead, so this is not implied by
    #: *status*: it is the difference between "answered from half a corpus" and
    #: "answered from all of it", which nothing else in the record can express.
    failed_legs: list[dict[str, Any]] = field(default_factory=list)

    #: The corpus generation this answer was served from. It is what lets a
    #: reader tell an unstable ranker from a corpus that legitimately moved:
    #: two different owners for one query mean a defect only if the corpus
    #: underneath them was the same. ``None`` when the producer did not supply
    #: one, which every line written before this field existed did not.
    generation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "query": self.query,
            "mode": self.mode,
            "limit": self.limit,
            "latency_ms": self.latency_ms,
            "confidence": self.confidence,
            "result_count": self.result_count,
            "top": [entry.to_dict() for entry in self.top[:TOP_EVENTS_LOGGED]],
            "selected_owner_file": self.selected_owner_file,
            "selected_owner_evidence": self.selected_owner_evidence,
            "no_match": self.no_match,
            "status": self.status,
            "error_code": self.error_code,
            "failed_legs": list(self.failed_legs[:FAILED_LEGS_LOGGED]),
            "generation": self.generation,
        }


class QueryLog:
    """Append-only JSONL sink for :class:`QueryEvent`.

    Holds a path, not a handle: the file is opened per append. A retrieval
    costs tens of milliseconds against two indexes, so an open cannot be the
    expensive part, and a long-lived handle would have to be closed by a
    lifespan that currently has no reason to know this exists.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def append(self, event: QueryEvent) -> bool:
        """Record *event*. Returns whether it was written; never raises.

        The boolean is for tests and for a future reporting surface. No caller
        in the search path reads it, which is the point — there is no failure
        here that a search should react to.
        """
        try:
            line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            return True
        except Exception:
            # Debug, not warning: a repository whose log cannot be written
            # would otherwise emit one line per query forever, which is a
            # louder failure than the one it is reporting.
            _log.debug("source-search query log append failed", exc_info=True)
            return False
