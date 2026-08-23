"""The clone service boundary: normalization, degradation, and the three mutation legs.

Every test here runs the real detector over a real planted tree — see
``toy_clone_repo`` — because the properties under test (a finding
disappearing when its source does, a cache that cannot be trusted being
disclosed) are properties of the whole path, and a mocked detector would
assert them of nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repowise.core.analysis.clone_service import (
    DEFAULT_NEAR_THRESHOLD,
    CloneService,
    CloneSite,
    DegradationCode,
    DegradationImpact,
    SourceIndexNearClones,
    pair_near_clones,
)
from repowise.core.analysis.health.duplication.pair_index import _INDEX_FILENAME
from repowise.core.analysis.health.duplication.token_cache import _CACHE_FILENAME

from . import toy_clone_repo as toy


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return toy.build(tmp_path / "toy")


def _service(root: Path) -> CloneService:
    return CloneService.from_paths(root, toy.scan_entries(root))


def _has_pair(report, a: str, b: str) -> bool:
    want = {a, b}
    return any({s.file for s in f.sites} == want for f in report.findings)


# ---------------------------------------------------------------------------
# The planted exact clone
# ---------------------------------------------------------------------------


def test_exact_clone_plant_is_found(repo: Path) -> None:
    report = _service(repo).scan()

    assert report.status == "complete"
    assert _has_pair(report, "alpha/report.py", "beta/report.py")
    assert report.no_results_reason is None


def test_findings_are_normalized(repo: Path) -> None:
    """One canonical orientation, a stable id, and one shape for both legs."""
    finding = _service(repo).scan().findings[0]

    # Sites are ordered, so a pair is reported once and not once per direction.
    assert (finding.sites[0].file, finding.sites[0].start_line) <= (
        finding.sites[1].file,
        finding.sites[1].start_line,
    )
    # The id is content-addressed: the same geometry re-derives the same id.
    assert finding.id == _service(repo).scan().findings[0].id
    assert len(finding.id) == 12
    assert finding.kind == "exact"
    assert finding.similarity == 1.0
    assert finding.lines > 0
    assert finding.cross_directory is True
    assert finding.intra_file is False


def test_scan_scopes_to_a_path_prefix(repo: Path) -> None:
    report = _service(repo).scan(path_prefix="alpha")

    assert report.scanned_files == 1
    assert report.findings == ()
    # One file in scope and no partner to pair with is a real clean result,
    # not a failure, so no reason is attached.
    assert report.no_results_reason is None


def test_cross_directory_only_filters(repo: Path) -> None:
    report = _service(repo).scan(cross_directory_only=True)
    assert all(f.cross_directory for f in report.findings)


# ---------------------------------------------------------------------------
# Mutation leg (a): the plant is removed
# ---------------------------------------------------------------------------


def test_removing_the_clone_removes_the_finding(repo: Path) -> None:
    """No stale serving: the second scan reads a warm cache and still sees the edit."""
    first = _service(repo).scan()
    assert _has_pair(first, "alpha/report.py", "beta/report.py")
    # The first scan populated the detector's on-disk artifacts; the second
    # must not be answerable from them.
    assert (repo / ".repowise" / _CACHE_FILENAME).is_file()

    toy.mutate_remove_clone(repo)
    second = _service(repo).scan()

    assert not _has_pair(second, "alpha/report.py", "beta/report.py")
    assert second.status == "complete"
    assert second.findings == ()


# ---------------------------------------------------------------------------
# Mutation leg (b): the cache is corrupt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("artifact", [_CACHE_FILENAME, _INDEX_FILENAME])
@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(b"not a sealed cache at all, just bytes" * 8, id="garbage"),
        pytest.param(b"", id="empty"),
        pytest.param(b"RWCH1", id="truncated-header"),
    ],
)
def test_corrupt_cache_degrades_with_a_typed_disclosure(
    repo: Path, artifact: str, corrupt: bytes
) -> None:
    service = _service(repo)
    service.scan()  # warm the artifacts so there is something to corrupt
    (repo / ".repowise" / artifact).write_bytes(corrupt)

    report = _service(repo).scan()

    codes = {d.code for d in report.degradations}
    assert DegradationCode.CACHE_UNREADABLE in codes
    disclosure = next(d for d in report.degradations if d.code == DegradationCode.CACHE_UNREADABLE)
    # Correctness survives: the findings are still complete, just recomputed.
    assert disclosure.impact is DegradationImpact.COMPLETE
    assert report.status == "complete"
    assert _has_pair(report, "alpha/report.py", "beta/report.py")


def test_corrupt_cache_never_produces_an_empty_success(repo: Path, monkeypatch) -> None:
    """A scan that cannot run says so, in the status and in the reason."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr("repowise.core.analysis.clone_service.detect_clones", _boom)
    report = _service(repo).scan()

    assert report.findings == ()
    assert report.status == "failed"
    assert DegradationCode.SCAN_FAILED in {d.code for d in report.degradations}
    assert report.no_results_reason is not None
    assert "not a report of zero duplication" in report.as_dict()["degradations"][0]["detail"]


def test_cache_internals_never_reach_the_response(repo: Path) -> None:
    """The pickle cache stays private: no name, no path, no extension, anywhere."""
    service = _service(repo)
    service.scan()
    (repo / ".repowise" / _CACHE_FILENAME).write_bytes(b"corrupt")
    (repo / ".repowise" / _INDEX_FILENAME).write_bytes(b"corrupt")

    payload = json.dumps(_service(repo).scan().as_dict()).lower()

    assert DegradationCode.CACHE_UNREADABLE.value in payload  # the disclosure happened
    for leaked in (".pkl", "pickle", ".repowise", "duplication_cache", "duplication_pairs"):
        assert leaked not in payload, f"{leaked!r} leaked through the boundary"


def test_probe_filenames_still_match_the_detector() -> None:
    """The boundary mirrors two private detector constants; keep them honest.

    A rename inside the detector must break this test rather than silently
    turn the cache-health probe into a no-op.
    """
    from repowise.core.analysis.clone_service import _CACHE_ARTIFACTS

    assert set(_CACHE_ARTIFACTS) == {_CACHE_FILENAME, _INDEX_FILENAME}


def test_missing_cache_is_not_a_degradation(repo: Path) -> None:
    report = _service(repo).scan()
    assert not any(d.code is DegradationCode.CACHE_UNREADABLE for d in report.degradations)


# ---------------------------------------------------------------------------
# Mutation leg (c): near clones are off unless asked for
# ---------------------------------------------------------------------------


def test_near_clones_are_off_by_default(repo: Path) -> None:
    """The default surface never consults a similarity index at all."""
    index = toy.oracle_index(repo)
    report = _service(repo).scan()

    assert report.near_clones_enabled is False
    assert report.near_threshold is None
    assert all(f.kind == "exact" for f in report.findings)
    assert index.neighbour_calls == 0
    assert report.as_dict()["near_clones"] == {"enabled": False, "threshold": None}


async def test_near_clones_surface_the_paraphrased_plant(repo: Path) -> None:
    index = toy.oracle_index(repo)
    report = await _service(repo).scan_with_near_clones(index)

    assert report.near_clones_enabled is True
    assert report.near_threshold == DEFAULT_NEAR_THRESHOLD

    near = [f for f in report.findings if f.kind == "near"]
    assert len(near) == 1
    files = {s.file for s in near[0].sites}
    assert files == {"core/util.py", "gamma/paraphrase.py"}
    assert near[0].similarity == pytest.approx(0.93)

    # The exact plant is still there, still tagged exact, and never doubled.
    exact = [f for f in report.findings if f.kind == "exact"]
    assert len(exact) == 1
    assert {s.file for s in exact[0].sites} == {"alpha/report.py", "beta/report.py"}


async def test_near_leg_drops_pairs_the_exact_leg_already_reported(repo: Path) -> None:
    """The compute_total pair scores 0.99 in the oracle and must not be re-emitted."""
    report = await _service(repo).scan_with_near_clones(toy.oracle_index(repo))

    reported = [{s.file for s in f.sites} for f in report.findings if f.kind == "near"]
    assert {"alpha/report.py", "beta/report.py"} not in reported


async def test_near_pass_respects_the_path_prefix(repo: Path) -> None:
    """The prefix confines BOTH legs.

    The near pass pairing the whole index leaked out-of-scope findings — and
    re-emitted the exact plant the prefix had dropped, mislabeled ``near``,
    because the exclusion list is computed from the already-filtered exact leg.
    """
    report = await _service(repo).scan_with_near_clones(
        toy.oracle_index(repo), path_prefix="alpha"
    )

    assert all(s.file.startswith("alpha/") for f in report.findings for s in f.sites)
    reported = [{s.file for s in f.sites} for f in report.findings]
    assert {"alpha/report.py", "beta/report.py"} not in reported
    assert {"core/util.py", "gamma/paraphrase.py"} not in reported


async def test_below_threshold_pairs_are_not_reported(repo: Path) -> None:
    report = await _service(repo).scan_with_near_clones(toy.oracle_index(repo))

    near_files = [{s.file for s in f.sites} for f in report.findings if f.kind == "near"]
    assert {"core/util.py", "core/lonely.py"} not in near_files

    # Lowering the threshold under the 0.71 pair admits it, which is what
    # makes the default a gate rather than an accident of the fixture.
    lowered = await _service(repo).scan_with_near_clones(toy.oracle_index(repo), threshold=0.70)
    lowered_files = [{s.file for s in f.sites} for f in lowered.findings if f.kind == "near"]
    assert {"core/util.py", "core/lonely.py"} in lowered_files


async def test_absent_similarity_index_is_disclosed_not_silent(repo: Path) -> None:
    report = await _service(repo).scan_with_near_clones(None)

    assert report.near_clones_enabled is True
    assert DegradationCode.NEAR_CLONES_UNAVAILABLE in {d.code for d in report.degradations}
    assert report.status == "partial"
    # The exact leg is untouched by the near leg's absence.
    assert _has_pair(report, "alpha/report.py", "beta/report.py")


async def test_failing_similarity_index_does_not_take_the_scan_down(repo: Path) -> None:
    class Broken:
        async def chunks(self):
            raise RuntimeError("index unavailable")

        async def neighbours(self, chunk_id, *, limit):  # pragma: no cover - unreachable
            return ()

    report = await _service(repo).scan_with_near_clones(Broken())

    assert DegradationCode.NEAR_CLONES_FAILED in {d.code for d in report.degradations}
    assert report.status == "partial"
    assert _has_pair(report, "alpha/report.py", "beta/report.py")


# ---------------------------------------------------------------------------
# The pairing rules, isolated from the scan
# ---------------------------------------------------------------------------


async def test_pairing_is_symmetric_and_deduplicated(repo: Path) -> None:
    findings, truncated = await pair_near_clones(toy.oracle_index(repo), threshold=0.5)

    assert truncated is False
    keys = [frozenset(s.file for s in f.sites) for f in findings]
    assert len(keys) == len(set(keys))


async def test_pairing_respects_the_minimum_size_floor(repo: Path) -> None:
    index = toy.oracle_index(repo)
    small = await pair_near_clones(index, threshold=0.5, min_lines=200)
    assert small == ([], False)


async def test_pairing_truncation_is_disclosed(repo: Path) -> None:
    findings, truncated = await pair_near_clones(
        toy.oracle_index(repo), threshold=0.5, max_chunks=1
    )
    assert truncated is True
    assert findings == []


async def test_truncated_near_pass_is_disclosed_on_the_report(repo: Path) -> None:
    report = await _service(repo).scan_with_near_clones(toy.oracle_index(repo), max_chunks=1)

    assert DegradationCode.NEAR_CLONES_TRUNCATED in {d.code for d in report.degradations}
    assert report.status == "partial"


# ---------------------------------------------------------------------------
# The source-index adapter
# ---------------------------------------------------------------------------


class _FakeRecord:
    def __init__(self, chunk_id: str, source: str = "symbol") -> None:
        self.chunk_id = chunk_id
        self.file_path = "a.py"
        self.name = chunk_id
        self.kind = "function"
        self.start_line = 1
        self.end_line = 20
        self.is_test = False
        self.source = source
        self.content_hash = chunk_id
        self.snippet = ""


class _FakeHit(_FakeRecord):
    def __init__(self, chunk_id: str, score: float) -> None:
        super().__init__(chunk_id)
        self.score = score


class _FakeStore:
    def __init__(self) -> None:
        self.searched: list[list[float]] = []

    async def stored_vectors(self):
        from repowise.core.source_search.vector_store import StoredVector

        return {
            "sym-1": StoredVector(content_hash="h1", vector=[1.0, 0.0]),
            "win-1": StoredVector(content_hash="h2", vector=[0.0, 1.0]),
        }

    async def fetch_by_chunk_ids(self, chunk_ids):
        return {
            "sym-1": _FakeRecord("sym-1"),
            "win-1": _FakeRecord("win-1", source="file_window"),
        }

    async def search_by_vector(self, vector, limit=20):
        self.searched.append(list(vector))
        return [_FakeHit("sym-1", 1.0), _FakeHit("sym-2", 0.91)]


async def test_source_index_adapter_serves_symbol_chunks_only() -> None:
    """File-window chunks overlap each other, so pairing them says the same thing twice."""
    adapter = SourceIndexNearClones(_FakeStore())

    chunks = await adapter.chunks()

    assert [c.id for c in chunks] == ["sym-1"]


async def test_source_index_adapter_drops_the_self_hit() -> None:
    store = _FakeStore()
    adapter = SourceIndexNearClones(store)
    await adapter.chunks()

    neighbours = await adapter.neighbours("sym-1", limit=5)

    assert [n[0] for n in neighbours] == ["sym-2"]
    assert store.searched == [[1.0, 0.0]]


async def test_source_index_adapter_is_silent_on_an_unknown_chunk() -> None:
    adapter = SourceIndexNearClones(_FakeStore())
    assert await adapter.neighbours("never-indexed", limit=5) == ()


def test_adapter_returns_none_without_a_source_index(tmp_path: Path) -> None:
    assert SourceIndexNearClones.for_repo(tmp_path) is None


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------


def test_clone_site_overlap_is_file_scoped() -> None:
    a = CloneSite("x.py", 10, 20)
    assert a.overlaps(CloneSite("x.py", 20, 30))
    assert not a.overlaps(CloneSite("x.py", 21, 30))
    assert not a.overlaps(CloneSite("y.py", 10, 20))


def test_empty_scope_is_a_clean_result_not_a_failure(tmp_path: Path) -> None:
    report = CloneService.from_paths(tmp_path, []).scan()

    assert report.status == "complete"
    assert report.scanned_files == 0
    assert report.no_results_reason == "No files were in scope for this scan."


async def test_near_pass_stops_at_its_time_budget(repo: Path) -> None:
    """Every chunk costs an index query, so the pass must be able to give up."""
    import asyncio

    class Slow:
        def __init__(self, chunks):
            self._chunks = chunks

        async def chunks(self):
            return self._chunks

        async def neighbours(self, chunk_id, *, limit):
            await asyncio.sleep(0.05)
            return ()

    index = Slow(toy.symbol_chunks(repo))
    findings, truncated = await pair_near_clones(index, min_lines=1, time_budget_secs=0.01)

    assert truncated is True
    assert findings == []


async def test_zero_time_budget_disables_the_deadline(repo: Path) -> None:
    findings, truncated = await pair_near_clones(
        toy.oracle_index(repo), threshold=0.9, time_budget_secs=0
    )

    assert truncated is False
    # Both oracle pairs above 0.9: the paraphrase and, with no exact
    # findings passed to suppress it, the compute_total pair.
    assert {frozenset(s.file for s in f.sites) for f in findings} == {
        frozenset({"core/util.py", "gamma/paraphrase.py"}),
        frozenset({"alpha/report.py", "beta/report.py"}),
    }
