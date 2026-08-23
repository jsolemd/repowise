"""Service boundary over the duplication detector.

The detector in :mod:`repowise.core.analysis.health.duplication` is built
for the health engine: it wants ``ParsedFile`` objects, it returns line
geometry weighted for a biomarker, and it accelerates itself with two
pickled artifacts inside the repository's own ``.repowise`` directory.
None of that should reach a query surface. This module is the seam.

What crosses the boundary outward is a :class:`CloneReport`: normalized
findings, a scan status, and typed degradations. What never crosses it is
the detector's on-disk cache — not its contents, not its filenames, not
its paths. Consumers ask this service; nothing else opens those files.
Two guarantees follow from that and are worth stating because they are
what the boundary is *for*:

* **No stale serving.** This service holds no findings cache of its own.
  Every scan re-derives from the file bytes on disk; the detector's cache
  is keyed by content hash, so a file that changed cannot be answered
  from it. Freshness is a property of content addressing here, not of an
  invalidation rule someone has to remember to fire.
* **No empty success.** A scan that could not run returns
  ``status="failed"`` with a degradation and a stated reason, never an
  empty finding list that reads as "this repository has no duplication".

The optional near-clone leg is a second, explicitly gated source: dense
similarity over the source-search index, for paraphrased duplication the
token-level detector cannot see. It is off unless asked for, its findings
are tagged ``kind="near"``, and they are never merged silently into the
exact set.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import structlog

from repowise.core.analysis.health.duplication import (
    DEFAULT_MIN_LINES,
    DEFAULT_WINDOW_TOKENS,
    ClonePair,
    DuplicationLimits,
    detect_clones,
)

log = structlog.get_logger(__name__)

#: Cosine floor for the near-clone leg.
#:
#: Code embeddings are dense and same-repository functions share a great
#: deal of surface — imports, idiom, house naming — so unrelated pairs sit
#: far above zero and a "high" threshold in the abstract is a low one
#: here. 0.86 is set to sit above that ambient band while still admitting
#: a renamed, reordered reimplementation of the same routine. It is a
#: parameter, and the value that ran is reported back with the findings.
DEFAULT_NEAR_THRESHOLD = 0.86

#: Nearest neighbours requested per chunk from the index. Small on
#: purpose: a chunk with more than a handful of near-duplicates is a
#: template or a generated file, and the pattern surfaces describe that
#: better than a hundred pairwise findings would.
DEFAULT_NEAR_NEIGHBOURS = 8

#: Chunks considered by the near-clone leg before it declares itself
#: truncated. Each one costs an index query.
DEFAULT_NEAR_MAX_CHUNKS = 4000

#: Wall-clock budget for the near-clone pass. The chunk cap bounds the
#: *number* of index queries but not their cost, and an approximate
#: nearest-neighbour search over a cold index is not fast. ``0`` disables
#: the deadline, which is what the deterministic tests use.
DEFAULT_NEAR_TIME_BUDGET_SECS = 20.0

# The detector's persisted artifacts. Mirrored rather than imported: both
# are module-private names inside the detector, and importing a private
# name would make a rename over there an ImportError over here. A guard
# test asserts these still match, so a rename breaks a test instead of
# silently disabling the health probe.
_CACHE_ARTIFACTS: tuple[str, ...] = ("duplication_cache.pkl", "duplication_pairs.pkl")

# Envelope magic written by ``repowise.core.cache_seal``. Checking it is an
# O(1) read that catches truncation, plaintext pickles and foreign files.
_SEAL_MAGIC = b"RWCH1"
_SEAL_HEADER_LEN = len(_SEAL_MAGIC) + 32


# ---------------------------------------------------------------------------
# Typed degradation
# ---------------------------------------------------------------------------


class DegradationCode(StrEnum):
    """Closed vocabulary for everything that can go wrong in a scan."""

    CACHE_UNREADABLE = "cache_unreadable"
    SCAN_FAILED = "scan_failed"
    SCAN_TRUNCATED = "scan_truncated"
    SCAN_DEADLINE = "scan_deadline"
    SCAN_SKIPPED_REPETITIVE = "scan_skipped_repetitive"
    FILES_UNREADABLE = "files_unreadable"
    FILES_SKIPPED_GENERATED = "files_skipped_generated"
    FILES_SKIPPED_TOO_LARGE = "files_skipped_too_large"
    NEAR_CLONES_UNAVAILABLE = "near_clones_unavailable"
    NEAR_CLONES_FAILED = "near_clones_failed"
    NEAR_CLONES_TRUNCATED = "near_clones_truncated"


class DegradationImpact(StrEnum):
    """What the degradation did to the findings beside it."""

    #: The findings are complete; something slower or less direct produced them.
    COMPLETE = "results_complete"
    #: Real findings, but the scan did not cover everything it was asked to.
    PARTIAL = "results_partial"
    #: No findings were produced. An empty list here does not mean "none exist".
    UNAVAILABLE = "results_unavailable"


@dataclass(frozen=True, slots=True)
class Degradation:
    """One typed disclosure attached to a report.

    ``detail`` is prose for a reader and is deliberately free of any path,
    filename or internal artifact name — the boundary discloses that a
    cache was unusable, never where it lives.
    """

    code: DegradationCode
    scope: str  # "exact" | "near" | "scan"
    impact: DegradationImpact
    detail: str
    count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code.value,
            "scope": self.scope,
            "impact": self.impact.value,
            "detail": self.detail,
        }
        if self.count is not None:
            out["count"] = self.count
        return out


# ---------------------------------------------------------------------------
# Normalized findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CloneSite:
    """One end of a clone, as repo-relative path and 1-based inclusive lines."""

    file: str
    start_line: int
    end_line: int

    @property
    def lines(self) -> int:
        return max(0, self.end_line - self.start_line + 1)

    @property
    def directory(self) -> str:
        head, _, tail = self.file.rpartition("/")
        return head if tail else "."

    def overlaps(self, other: CloneSite) -> bool:
        return (
            self.file == other.file
            and self.start_line <= other.end_line
            and other.start_line <= self.end_line
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "lines": self.lines,
        }


@dataclass(frozen=True, slots=True)
class CloneFinding:
    """One duplicated region, in the one shape both legs report in.

    Exact and near findings differ by ``kind`` and ``similarity`` and by
    nothing else in the record layout, so a consumer never branches on
    which detector produced a finding to read it.
    """

    id: str
    kind: str  # "exact" | "near"
    sites: tuple[CloneSite, CloneSite]
    similarity: float
    lines: int
    tokens: int = 0
    co_change_count: int = 0
    label: str = ""

    @property
    def intra_file(self) -> bool:
        return self.sites[0].file == self.sites[1].file

    @property
    def cross_directory(self) -> bool:
        return self.sites[0].directory != self.sites[1].directory

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "similarity": round(self.similarity, 4),
            "lines": self.lines,
            "sites": [site.as_dict() for site in self.sites],
            "intra_file": self.intra_file,
            "cross_directory": self.cross_directory,
        }
        if self.tokens:
            out["tokens"] = self.tokens
        if self.co_change_count:
            out["co_change_count"] = self.co_change_count
        if self.label:
            out["label"] = self.label
        return out


def _finding_id(kind: str, sites: tuple[CloneSite, CloneSite]) -> str:
    """Stable, content-addressed id so a finding can be named across calls."""
    parts = "|".join(f"{s.file}:{s.start_line}-{s.end_line}" for s in sites)
    return hashlib.sha256(f"{kind}|{parts}".encode()).hexdigest()[:12]


def _order_sites(a: CloneSite, b: CloneSite) -> tuple[CloneSite, CloneSite]:
    """One canonical orientation per pair, so a pair is reported once."""
    return (a, b) if (a.file, a.start_line) <= (b.file, b.start_line) else (b, a)


@dataclass(frozen=True, slots=True)
class CloneReport:
    """The result of one scan: findings plus everything that qualifies them."""

    findings: tuple[CloneFinding, ...]
    status: str  # "complete" | "partial" | "failed"
    scanned_files: int
    degradations: tuple[Degradation, ...] = ()
    near_clones_enabled: bool = False
    near_threshold: float | None = None
    duration_ms: float = 0.0

    @property
    def no_results_reason(self) -> str | None:
        """Why an empty list is empty — ``None`` only when it truly is a clean scan."""
        if self.findings:
            return None
        if self.status == "failed":
            return (
                "The duplication scan did not run to completion, so this is not a "
                "finding of zero duplication. See degradations."
            )
        if self.status == "partial":
            return (
                "The scan covered only part of the repository and found no duplication "
                "in what it covered. See degradations."
            )
        if self.scanned_files == 0:
            return "No files were in scope for this scan."
        return None

    def as_dict(self, *, limit: int | None = None) -> dict[str, Any]:
        shown = self.findings if limit is None else self.findings[:limit]
        exact = sum(1 for f in self.findings if f.kind == "exact")
        payload: dict[str, Any] = {
            "status": self.status,
            "findings": [f.as_dict() for f in shown],
            "summary": {
                "total_findings": len(self.findings),
                "returned": len(shown),
                "exact": exact,
                "near": len(self.findings) - exact,
                "scanned_files": self.scanned_files,
                "duplicated_lines": sum(
                    max(f.sites[0].lines, f.sites[1].lines) for f in self.findings
                ),
                "truncated": len(shown) < len(self.findings),
            },
            "near_clones": {
                "enabled": self.near_clones_enabled,
                "threshold": self.near_threshold,
            },
            "degradations": [d.as_dict() for d in self.degradations],
        }
        reason = self.no_results_reason
        if reason:
            payload["no_results_reason"] = reason
        return payload


# ---------------------------------------------------------------------------
# Near-clone leg
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NearCloneChunk:
    """One indexed unit the near-clone leg can pair."""

    id: str
    file: str
    name: str
    kind: str
    start_line: int
    end_line: int
    is_test: bool = False

    @property
    def lines(self) -> int:
        return max(0, self.end_line - self.start_line + 1)

    def site(self) -> CloneSite:
        return CloneSite(file=self.file, start_line=self.start_line, end_line=self.end_line)


@runtime_checkable
class NearCloneIndex(Protocol):
    """The slice of a similarity index this leg needs.

    Deliberately two methods and no vectors: nearest-neighbour search is
    the index's job, not this module's, so the leg never holds an
    embedding and never has an opinion about how one is produced.
    """

    async def chunks(self) -> Sequence[NearCloneChunk]:
        """Every pairable unit in the active index."""
        ...

    async def neighbours(self, chunk_id: str, *, limit: int) -> Sequence[tuple[str, float]]:
        """``(chunk_id, cosine similarity)`` neighbours of *chunk_id*."""
        ...


class SourceIndexNearClones:
    """:class:`NearCloneIndex` over the source-search corpus.

    Reads through the source index's own public query surface — the
    stored vectors and its vector search — so this leg has no knowledge
    of how the corpus is chunked, embedded or generation-gated.
    """

    def __init__(self, store: Any) -> None:
        self._store = store
        self._vectors: dict[str, list[float]] = {}

    @classmethod
    def for_repo(cls, repo_root: Path | str) -> SourceIndexNearClones | None:
        """Bind to *repo_root*'s active source-index generation, if it has one.

        ``None`` for "no usable index", whether that is because none was
        built or because the one on disk cannot be read. Both reach the
        caller as the same disclosure, which is the honest one: the
        near-clone leg has nothing to search.
        """
        try:
            return cls._bind(Path(repo_root))
        except Exception as exc:
            log.debug("clone_service_source_index_bind_failed", error=str(exc))
            return None

    @classmethod
    def _bind(cls, repo: Path) -> SourceIndexNearClones | None:
        from repowise.core.source_search.manifest import default_manifest_path, read_manifest
        from repowise.core.source_search.vector_store import SourceChunkVectorStore

        manifest = read_manifest(default_manifest_path(repo))
        if manifest is None:
            return None
        from repowise.core.source_search.generation import GenerationRef

        store = SourceChunkVectorStore(
            str(repo / ".repowise" / "lancedb"),
            # Read-only use: the embedder is only consulted when writing or
            # when embedding a text query, and this leg does neither — it
            # searches with vectors the index already holds.
            embedder=None,  # type: ignore[arg-type]
            table_name=manifest.lance_table,
            generation=GenerationRef(manifest.generation_id, manifest.generation_sequence),
        )
        return cls(store)

    async def chunks(self) -> Sequence[NearCloneChunk]:
        stored = await self._store.stored_vectors()
        self._vectors = {cid: sv.vector for cid, sv in stored.items()}
        records = await self._store.fetch_by_chunk_ids(list(self._vectors))
        out: list[NearCloneChunk] = []
        for chunk_id, record in records.items():
            # File-window chunks are overlapping line slices, so pairing
            # them reports the same duplication several times over. Symbol
            # chunks are the unit a reader can act on.
            if record.source != "symbol":
                continue
            out.append(
                NearCloneChunk(
                    id=chunk_id,
                    file=record.file_path,
                    name=record.name,
                    kind=record.kind,
                    start_line=record.start_line,
                    end_line=record.end_line,
                    is_test=record.is_test,
                )
            )
        out.sort(key=lambda c: (c.file, c.start_line, c.id))
        return out

    async def neighbours(self, chunk_id: str, *, limit: int) -> Sequence[tuple[str, float]]:
        vector = self._vectors.get(chunk_id)
        if vector is None:
            return ()
        hits = await self._store.search_by_vector(vector, limit=limit + 1)
        return tuple((hit.chunk_id, float(hit.score)) for hit in hits if hit.chunk_id != chunk_id)


async def pair_near_clones(
    index: NearCloneIndex,
    *,
    threshold: float = DEFAULT_NEAR_THRESHOLD,
    min_lines: int = DEFAULT_MIN_LINES,
    neighbours: int = DEFAULT_NEAR_NEIGHBOURS,
    max_chunks: int = DEFAULT_NEAR_MAX_CHUNKS,
    time_budget_secs: float = DEFAULT_NEAR_TIME_BUDGET_SECS,
    include_tests: bool = False,
    path_prefix: str | None = None,
    exclude: Sequence[CloneFinding] = (),
) -> tuple[list[CloneFinding], bool]:
    """Pair chunks above *threshold* that the exact leg did not already report.

    Returns the findings and whether the pass was cut short — by the chunk
    budget or by the wall-clock one. Every chunk costs the index a
    nearest-neighbour query, so an unbounded pass over a large corpus is
    the one way this leg could wedge a tool call; both budgets exist to
    make that impossible, and either one firing is disclosed rather than
    quietly shortening the answer.

    ``path_prefix`` confines the pass the way the exact leg is confined: a
    chunk outside the prefix is never offered, so a pair cannot span the
    scope boundary — and a pair the exact leg dropped as out of scope
    cannot come back mislabeled ``near``.

    Pairs are symmetric, so each is emitted once, in canonical site order.
    """
    all_chunks = list(await index.chunks())
    eligible = [
        c
        for c in all_chunks
        if _in_scope(c.file, path_prefix)
        and c.lines >= min_lines
        and (include_tests or not c.is_test)
    ]
    truncated = len(eligible) > max_chunks
    eligible = eligible[:max_chunks]
    by_id = {c.id: c for c in eligible}

    deadline = time.monotonic() + time_budget_secs if time_budget_secs > 0 else None
    seen: set[tuple[str, str]] = set()
    findings: list[CloneFinding] = []
    for chunk in eligible:
        if deadline is not None and time.monotonic() > deadline:
            truncated = True
            break
        for other_id, score in await index.neighbours(chunk.id, limit=neighbours):
            if score < threshold:
                continue
            other = by_id.get(other_id)
            if other is None or other.id == chunk.id:
                continue
            key = (chunk.id, other.id) if chunk.id <= other.id else (other.id, chunk.id)
            if key in seen:
                continue
            seen.add(key)
            sites = _order_sites(chunk.site(), other.site())
            # Two chunks that overlap in the same file are a symbol and its
            # own enclosing scope, not a duplicate of it.
            if sites[0].overlaps(sites[1]):
                continue
            if _already_reported(sites, exclude):
                continue
            names = sorted({chunk.name, other.name})
            findings.append(
                CloneFinding(
                    id=_finding_id("near", sites),
                    kind="near",
                    sites=sites,
                    similarity=float(score),
                    lines=max(sites[0].lines, sites[1].lines),
                    label=" ~ ".join(names) if len(names) > 1 else names[0],
                )
            )

    findings.sort(key=lambda f: (-f.similarity, -f.lines, f.id))
    return findings, truncated


def _already_reported(sites: tuple[CloneSite, CloneSite], existing: Sequence[CloneFinding]) -> bool:
    """True when an exact finding already covers both ends of this pair."""
    for found in existing:
        a, b = found.sites
        if (sites[0].overlaps(a) and sites[1].overlaps(b)) or (
            sites[0].overlaps(b) and sites[1].overlaps(a)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScanFile:
    """One file offered to the scan, in the detector's input contract."""

    path: str
    abs_path: str
    language: str


@dataclass
class CloneService:
    """Serves normalized clone findings over the duplication detector.

    Stateless with respect to findings by design — see the module
    docstring. The only state is where to scan and where the detector may
    keep its own caches, and the latter never leaves this object.
    """

    files: Sequence[ScanFile]
    cache_dir: Path | None = None
    git_meta_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    window_tokens: int = DEFAULT_WINDOW_TOKENS
    limits: DuplicationLimits | None = None

    @classmethod
    def from_paths(
        cls,
        repo_root: Path | str,
        entries: Iterable[tuple[str, str]],
        *,
        cache_dir: Path | None = None,
        git_meta_map: dict[str, dict[str, Any]] | None = None,
    ) -> CloneService:
        """Build from ``(repo-relative path, language)`` pairs.

        This is the shape a persisted index already has, so the query
        surface does not have to re-walk or re-parse the tree to ask a
        duplication question.
        """
        root = Path(repo_root)
        files = [
            ScanFile(path=path, abs_path=str(root / path), language=language)
            for path, language in entries
        ]
        return cls(
            files=files,
            cache_dir=cache_dir if cache_dir is not None else root / ".repowise",
            git_meta_map=git_meta_map or {},
        )

    # -- scanning --------------------------------------------------------

    def scan(
        self,
        *,
        min_lines: int = DEFAULT_MIN_LINES,
        path_prefix: str | None = None,
        include_intra_file: bool = True,
        cross_directory_only: bool = False,
    ) -> CloneReport:
        """Run the exact-clone leg and normalize what comes back."""
        started = time.perf_counter()
        selected = [f for f in self.files if _in_scope(f.path, path_prefix)]
        degradations: list[Degradation] = list(self._probe_cache_health())

        if not selected:
            return CloneReport(
                findings=(),
                status="complete",
                scanned_files=0,
                degradations=tuple(degradations),
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        try:
            report = detect_clones(
                [_parsed_shim(f) for f in selected],
                self.git_meta_map,
                window_tokens=self.window_tokens,
                min_lines=min_lines,
                limits=self.limits,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            # An empty list here would read as "no duplication". Say the
            # scan failed instead, and say it in the status as well as in
            # the degradation, so a caller that only reads one sees it.
            log.debug("clone_service_scan_failed", error=str(exc))
            degradations.append(
                Degradation(
                    code=DegradationCode.SCAN_FAILED,
                    scope="exact",
                    impact=DegradationImpact.UNAVAILABLE,
                    detail=(
                        "The duplication scan raised and produced no findings. This is "
                        "not a report of zero duplication."
                    ),
                )
            )
            return CloneReport(
                findings=(),
                status="failed",
                scanned_files=len(selected),
                degradations=tuple(degradations),
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        degradations.extend(_diagnostic_degradations(report.diagnostics))
        findings = tuple(
            _normalize_pair(pair)
            for pair in report.pairs
            if _pair_wanted(pair, include_intra_file, cross_directory_only)
        )
        findings = tuple(sorted(findings, key=lambda f: (-f.lines, -f.tokens, f.id)))

        return CloneReport(
            findings=findings,
            status=_status_for(degradations),
            scanned_files=len(selected),
            degradations=tuple(degradations),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def scan_with_near_clones(
        self,
        index: NearCloneIndex | None,
        *,
        min_lines: int = DEFAULT_MIN_LINES,
        path_prefix: str | None = None,
        include_intra_file: bool = True,
        cross_directory_only: bool = False,
        threshold: float = DEFAULT_NEAR_THRESHOLD,
        neighbours: int = DEFAULT_NEAR_NEIGHBOURS,
        max_chunks: int = DEFAULT_NEAR_MAX_CHUNKS,
        time_budget_secs: float = DEFAULT_NEAR_TIME_BUDGET_SECS,
        include_tests: bool = False,
    ) -> CloneReport:
        """The exact leg, plus dense near-clones the token detector missed.

        Only reachable by asking: :meth:`scan` is the default surface and
        never consults a similarity index. Near findings are appended
        after the exact ones and stay tagged ``kind="near"``.
        """
        base = self.scan(
            min_lines=min_lines,
            path_prefix=path_prefix,
            include_intra_file=include_intra_file,
            cross_directory_only=cross_directory_only,
        )
        degradations = list(base.degradations)

        if index is None:
            degradations.append(
                Degradation(
                    code=DegradationCode.NEAR_CLONES_UNAVAILABLE,
                    scope="near",
                    impact=DegradationImpact.PARTIAL,
                    detail=(
                        "Near-clone detection was requested but this repository has no "
                        "source-similarity index. Exact clone findings are unaffected."
                    ),
                )
            )
            return CloneReport(
                findings=base.findings,
                status=_status_for(degradations),
                scanned_files=base.scanned_files,
                degradations=tuple(degradations),
                near_clones_enabled=True,
                near_threshold=threshold,
                duration_ms=base.duration_ms,
            )

        try:
            near, truncated = await pair_near_clones(
                index,
                threshold=threshold,
                min_lines=min_lines,
                neighbours=neighbours,
                max_chunks=max_chunks,
                time_budget_secs=time_budget_secs,
                include_tests=include_tests,
                path_prefix=path_prefix,
                exclude=base.findings,
            )
        except Exception as exc:
            log.debug("clone_service_near_failed", error=str(exc))
            degradations.append(
                Degradation(
                    code=DegradationCode.NEAR_CLONES_FAILED,
                    scope="near",
                    impact=DegradationImpact.PARTIAL,
                    detail=(
                        "The near-clone pass raised and contributed nothing. Exact clone "
                        "findings are unaffected."
                    ),
                )
            )
            near, truncated = [], False

        if truncated:
            degradations.append(
                Degradation(
                    code=DegradationCode.NEAR_CLONES_TRUNCATED,
                    scope="near",
                    impact=DegradationImpact.PARTIAL,
                    detail=(
                        "The near-clone pass stopped before covering every eligible "
                        "symbol — it reached either its symbol budget or its time "
                        "budget — and covered the ones it reached in path order. Exact "
                        "findings are complete; narrow the scan with path_prefix for "
                        "full near-clone coverage."
                    ),
                )
            )

        if cross_directory_only:
            near = [f for f in near if f.cross_directory]
        if not include_intra_file:
            near = [f for f in near if not f.intra_file]

        return CloneReport(
            findings=base.findings + tuple(near),
            status=_status_for(degradations),
            scanned_files=base.scanned_files,
            degradations=tuple(degradations),
            near_clones_enabled=True,
            near_threshold=threshold,
            duration_ms=base.duration_ms,
        )

    # -- cache health ----------------------------------------------------

    def _probe_cache_health(self) -> list[Degradation]:
        """Disclose an unusable detector cache without naming or opening it.

        The detector degrades correctly on its own — an unreadable cache
        costs it a recompute, not correctness — but it degrades *silently*,
        which leaves a caller unable to tell a healthy index from one
        something has been writing garbage into. This checks the seal
        envelope, which is a constant-size read.

        The check catches truncation, unsigned or legacy formats and
        non-cache content. It does not catch a well-formed file with a
        forged authentication tag: the detector refuses that too and
        recomputes, so the findings stay correct, but this probe cannot
        distinguish it from a cold cache and will not report it.
        """
        if self.cache_dir is None:
            return []
        damaged = 0
        for name in _CACHE_ARTIFACTS:
            path = Path(self.cache_dir) / name
            try:
                if not path.is_file():
                    continue
                with path.open("rb") as fh:
                    header = fh.read(len(_SEAL_MAGIC))
                if header != _SEAL_MAGIC or path.stat().st_size < _SEAL_HEADER_LEN:
                    damaged += 1
            except OSError:
                damaged += 1
        if not damaged:
            return []
        return [
            Degradation(
                code=DegradationCode.CACHE_UNREADABLE,
                scope="scan",
                impact=DegradationImpact.COMPLETE,
                detail=(
                    "An incremental duplication cache for this repository is unreadable "
                    "and was bypassed. Findings were recomputed from source and are "
                    "complete; the scan was slower than it needed to be. Re-running the "
                    "indexer rewrites it."
                ),
                count=damaged,
            )
        ]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parsed_shim(f: ScanFile) -> Any:
    """The detector's input contract, which is three attributes deep."""
    return SimpleNamespace(
        file_info=SimpleNamespace(path=f.path, abs_path=f.abs_path, language=f.language),
        symbols=[],
    )


def _in_scope(path: str, prefix: str | None) -> bool:
    if not prefix:
        return True
    normalized = prefix.rstrip("/")
    return path == normalized or path.startswith(f"{normalized}/")


def _pair_wanted(pair: ClonePair, include_intra_file: bool, cross_directory_only: bool) -> bool:
    if not include_intra_file and pair.is_intra_file:
        return False
    if cross_directory_only:
        a = CloneSite(pair.file_a, pair.a_start_line, pair.a_end_line)
        b = CloneSite(pair.file_b, pair.b_start_line, pair.b_end_line)
        if a.directory == b.directory:
            return False
    return True


def _normalize_pair(pair: ClonePair) -> CloneFinding:
    sites = _order_sites(
        CloneSite(pair.file_a, pair.a_start_line, pair.a_end_line),
        CloneSite(pair.file_b, pair.b_start_line, pair.b_end_line),
    )
    return CloneFinding(
        id=_finding_id("exact", sites),
        kind="exact",
        sites=sites,
        similarity=1.0,
        lines=max(sites[0].lines, sites[1].lines),
        tokens=pair.token_count,
        co_change_count=pair.co_change_count,
    )


def _diagnostic_degradations(diagnostics: dict[str, Any]) -> list[Degradation]:
    """Translate the detector's guard counters into typed disclosures."""
    out: list[Degradation] = []
    if diagnostics.get("window_budget_hit"):
        out.append(
            Degradation(
                code=DegradationCode.SCAN_TRUNCATED,
                scope="exact",
                impact=DegradationImpact.PARTIAL,
                detail=(
                    "The scan reached its repository-wide window budget and stopped "
                    "early. Files after the cut were not compared. Narrow the scan with "
                    "path_prefix to cover them."
                ),
            )
        )
    if diagnostics.get("timed_out"):
        out.append(
            Degradation(
                code=DegradationCode.SCAN_DEADLINE,
                scope="exact",
                impact=DegradationImpact.PARTIAL,
                detail=(
                    "The scan hit its wall-clock deadline before comparing every "
                    "candidate region. Findings are real but incomplete."
                ),
            )
        )
    degenerate = int(diagnostics.get("degenerate_buckets") or 0)
    if degenerate:
        out.append(
            Degradation(
                code=DegradationCode.SCAN_SKIPPED_REPETITIVE,
                scope="exact",
                impact=DegradationImpact.PARTIAL,
                detail=(
                    f"{degenerate} group(s) of identically-shaped regions were too large "
                    "to compare pairwise and were skipped. These are usually boilerplate "
                    "(license headers, import blocks), but real duplication inside them "
                    "was not reported."
                ),
                count=degenerate,
            )
        )
    for key, code, noun in (
        ("skipped_unreadable", DegradationCode.FILES_UNREADABLE, "could not be read"),
        (
            "skipped_minified",
            DegradationCode.FILES_SKIPPED_GENERATED,
            "were skipped as minified or generated",
        ),
        (
            "skipped_token_cap",
            DegradationCode.FILES_SKIPPED_TOO_LARGE,
            "exceeded the per-file size cap",
        ),
    ):
        count = int(diagnostics.get(key) or 0)
        if count:
            out.append(
                Degradation(
                    code=code,
                    scope="exact",
                    impact=DegradationImpact.PARTIAL,
                    detail=f"{count} file(s) in scope {noun} and were not compared.",
                    count=count,
                )
            )
    return out


def _status_for(degradations: Sequence[Degradation]) -> str:
    impacts = {d.impact for d in degradations}
    if DegradationImpact.UNAVAILABLE in impacts:
        return "failed"
    if DegradationImpact.PARTIAL in impacts:
        return "partial"
    return "complete"
