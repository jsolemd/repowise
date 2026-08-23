"""Turning a bucket of bad queries into a runnable regression suite.

The hard part of this format is not its shape, it is what a case is entitled
to assert. A row lands in the no-match bucket precisely because nobody knows
the right answer for it; writing the *observed* owner into an ``expect`` block
would freeze the defect into a test that passes forever because it asserts the
bug. So the exporter never invents ground truth. It emits only assertions that
follow from what the log actually established, and where nothing follows it
emits :data:`EXPECT_TODO` — a case that reports ``pending`` and is neither a
pass nor a failure until a human answers it.

Two export intents, because "a bad week becomes next week's suite" has two
readings and they want opposite files:

* ``goal`` (the default) asserts the behaviour after the fix. The suite is red
  today and turns green when retrieval improves. This is the reading the
  buckets are for.
* ``guard`` asserts today's behaviour as a floor. The suite is green today and
  turns red if retrieval gets worse. This is the reading for a bucket you have
  decided not to fix yet but refuse to let deteriorate.

Neither intent is a default that runs at execution time: every case carries
its own explicit ``expect`` block, so a fixture file means the same thing to
whoever reads it as it does to the runner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .buckets import (
    BUCKET_ERROR,
    BUCKET_LOW_CONFIDENCE,
    BUCKET_NO_MATCH,
    BUCKET_WRONG_OWNER,
    CONFIDENCE_CAUTION,
    CONFIDENCE_CONFIDENT,
    CONFIDENCE_NO_MATCH,
)
from .records import QueryRecord
from .report import QualityReport

__all__ = [
    "EXPECT_KINDS",
    "EXPECT_TODO",
    "FIXTURE_KIND",
    "FIXTURE_SCHEMA_VERSION",
    "INTENTS",
    "CaseOutcome",
    "FixtureCase",
    "FixtureRun",
    "FixtureSuite",
    "SearchCallable",
    "export_bucket",
    "load_suite",
    "run_suite",
    "write_suite",
]

FIXTURE_SCHEMA_VERSION = 1

#: Stamped into every file so a loader can refuse a document that merely
#: happens to be JSON with a ``cases`` key.
FIXTURE_KIND = "repowise.source_search.query_eval"

EXPECT_TODO = "todo"
EXPECT_OWNER = "owner"
EXPECT_ANY_OF = "any_of"
EXPECT_NOT_OWNER = "not_owner"
EXPECT_NO_MATCH = "no_match"
EXPECT_NO_MATCH_RESOLVES = "no_match_resolves"
EXPECT_MIN_CONFIDENCE = "min_confidence"
EXPECT_SUCCEEDS = "succeeds"

EXPECT_KINDS = (
    EXPECT_TODO,
    EXPECT_OWNER,
    EXPECT_ANY_OF,
    EXPECT_NOT_OWNER,
    EXPECT_NO_MATCH,
    EXPECT_NO_MATCH_RESOLVES,
    EXPECT_MIN_CONFIDENCE,
    EXPECT_SUCCEEDS,
)

INTENTS = ("goal", "guard")

OUTCOME_PASS = "pass"
OUTCOME_FAIL = "fail"
OUTCOME_PENDING = "pending"
OUTCOME_ERROR = "error"

#: Confidence as an ordered scale, so ``min_confidence`` has something to
#: compare against. A value the coordinator has never emitted sorts below
#: everything, which makes an unknown confidence fail a floor rather than
#: satisfy it.
_CONFIDENCE_RANK = {CONFIDENCE_NO_MATCH: 0, CONFIDENCE_CAUTION: 1, CONFIDENCE_CONFIDENT: 2}


class SearchCallable(Protocol):
    """What the runner needs: the coordinator's own ``search`` signature."""

    async def __call__(
        self, query: str, *, limit: int = 5, mode: str = "hybrid"
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FixtureCase:
    """One query, what it did, and what it is expected to do."""

    id: str
    query: str
    mode: str
    limit: int
    bucket: str
    expect: dict[str, Any]
    observed: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "mode": self.mode,
            "limit": self.limit,
            "bucket": self.bucket,
            "expect": self.expect,
            "observed": self.observed,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> FixtureCase:
        # Typed as Any and checked here so every rejection leaves through
        # ValueError. Callers catch that; a string in the cases list reaching
        # ``.get`` would leave through AttributeError instead and slip past them.
        if not isinstance(payload, Mapping):
            raise ValueError(f"case entry is {type(payload).__name__}, not an object")
        expect = payload.get("expect")
        if not isinstance(expect, Mapping) or "kind" not in expect:
            raise ValueError(f"case {payload.get('id')!r} has no expect.kind")
        if expect["kind"] not in EXPECT_KINDS:
            raise ValueError(
                f"case {payload.get('id')!r} has unknown expect.kind {expect['kind']!r}"
            )
        query = payload.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError(f"case {payload.get('id')!r} has no query")
        return cls(
            id=str(payload.get("id") or query[:40]),
            query=query,
            mode=str(payload.get("mode") or "hybrid"),
            limit=int(payload.get("limit") or 5),
            bucket=str(payload.get("bucket") or "unknown"),
            expect=dict(expect),
            observed=dict(payload.get("observed") or {}),
            provenance=dict(payload.get("provenance") or {}),
        )


@dataclass(frozen=True, slots=True)
class FixtureSuite:
    """A whole exported file."""

    cases: tuple[FixtureCase, ...]
    bucket: str
    intent: str
    source: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "kind": FIXTURE_KIND,
            "bucket": self.bucket,
            "intent": self.intent,
            "generated_at": self.generated_at,
            "source": self.source,
            "notes": list(self.notes),
            "cases": [case.to_dict() for case in self.cases],
        }


def _observed(records: Sequence[QueryRecord]) -> dict[str, Any]:
    """What the log saw, kept beside the expectation but never used as one."""
    owners: list[str | None] = []
    confidences: list[str] = []
    for record in records:
        if record.selected_owner_file not in owners:
            owners.append(record.selected_owner_file)
        if record.confidence not in confidences:
            confidences.append(record.confidence)
    top_files: list[str] = []
    for record in records:
        for name in record.top_files:
            if name not in top_files:
                top_files.append(name)
    return {
        "occurrences": len(records),
        "served_owners": owners,
        "confidences": confidences,
        "top_files": top_files[:10],
        "no_match": any(record.no_match for record in records),
        "status": sorted({record.status for record in records}),
    }


def _expectation(
    bucket: str,
    intent: str,
    records: Sequence[QueryRecord],
    conflict_owners: Sequence[str | None],
    expected_owner: str | None,
) -> dict[str, Any]:
    """The one assertion this bucket and intent actually support.

    Every arm that cannot be derived falls through to ``todo`` rather than
    guessing. A ``todo`` costs a human one line to resolve; a wrong assertion
    costs a suite its credibility.
    """
    if expected_owner is not None:
        # A verdict is ground truth and outranks both intents: whichever way
        # the suite is pointed, the right owner is the right owner.
        return {
            "kind": EXPECT_OWNER,
            "owner": expected_owner,
            "why": "an external verdict names this owner",
        }

    if bucket == BUCKET_NO_MATCH:
        if intent == "guard":
            return {
                "kind": EXPECT_NO_MATCH,
                "why": "hold the honest no-match: it must not become a confident wrong answer",
            }
        return {
            "kind": EXPECT_NO_MATCH_RESOLVES,
            "why": "the corpus should cover this question; fails until it does",
        }

    if bucket == BUCKET_ERROR:
        # There is no 'guard' reading of an outage. Locking in "this query
        # must keep failing" is never the thing anyone wants, so both intents
        # assert the fix.
        return {
            "kind": EXPECT_SUCCEEDS,
            "why": "a search that reached no corpus must reach one",
        }

    if bucket == BUCKET_LOW_CONFIDENCE:
        if intent == "guard":
            return {
                "kind": EXPECT_MIN_CONFIDENCE,
                "confidence": CONFIDENCE_CAUTION,
                "why": "hold the floor: this answer must not decay to a no-match or an error",
            }
        return {
            "kind": EXPECT_MIN_CONFIDENCE,
            "confidence": CONFIDENCE_CONFIDENT,
            "why": "the evidence should support an ownership claim; fails until it does",
        }

    if bucket == BUCKET_WRONG_OWNER:
        # The rivals are known; which of them is right is exactly what is not.
        # Naming them in the case is what turns a human's job into picking one.
        return {
            "kind": EXPECT_TODO,
            "candidates": list(conflict_owners),
            "why": (
                "retrieval contradicted itself here. Replace this with "
                f"{EXPECT_OWNER!r}, {EXPECT_ANY_OF!r} or {EXPECT_NOT_OWNER!r} once the "
                "correct owner is known."
            ),
        }

    return {"kind": EXPECT_TODO, "why": "no assertion follows from the log alone"}


def export_bucket(
    report: QualityReport,
    bucket: str,
    *,
    intent: str = "goal",
    limit: int | None = None,
) -> FixtureSuite:
    """Turn one bucket's rows into a suite of cases.

    Rows are grouped by normalized query, so a question asked forty times is
    one case with an occurrence count rather than forty duplicate cases.
    """
    if intent not in INTENTS:
        raise ValueError(f"intent must be one of {INTENTS}, got {intent!r}")

    groups: dict[str, list[QueryRecord]] = {}
    for record in report.bucket_records(bucket):
        groups.setdefault(record.normalized_query, []).append(record)

    # Most-repeated first, then by query text, so a truncated export keeps the
    # worst offenders and two runs over one log produce the same file.
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    if limit is not None:
        ordered = ordered[:limit]

    cases: list[FixtureCase] = []
    for index, (_key, members) in enumerate(ordered, start=1):
        first = members[0]
        classification = report.classifications.get(first.line_number)
        conflict = classification.conflict if classification else None
        stamps = sorted(record.ts_raw for record in members if record.ts_raw)
        cases.append(
            FixtureCase(
                id=f"{bucket}-{index:04d}",
                query=first.query,
                mode=first.mode or "hybrid",
                limit=first.limit or 5,
                bucket=bucket,
                expect=_expectation(
                    bucket,
                    intent,
                    members,
                    conflict.owners if conflict else (),
                    conflict.expected_owner if conflict else None,
                ),
                observed=_observed(members),
                provenance={
                    "log_lines": [record.line_number for record in members],
                    "first_seen": stamps[0] if stamps else None,
                    "last_seen": stamps[-1] if stamps else None,
                    "generations": sorted(
                        {record.generation for record in members if record.generation}
                    ),
                },
            )
        )

    notes = [
        f"Exported from the {bucket!r} bucket with intent {intent!r}.",
        "Cases whose expect.kind is 'todo' report as pending: they are questions for a "
        "human, not assertions, and they neither pass nor fail.",
        *report.caveats(),
    ]
    return FixtureSuite(
        cases=tuple(cases),
        bucket=bucket,
        intent=intent,
        source={
            "query_log": str(report.parsed.path) if report.parsed.path else None,
            "rows_analysed": len(report.parsed.records),
            "rows_in_bucket": sum(len(members) for _key, members in groups.items()),
            "distinct_queries_in_bucket": len(groups),
        },
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        notes=tuple(notes),
    )


def write_suite(suite: FixtureSuite, path: Path | str) -> Path:
    """Write *suite* to *path*.

    ``sort_keys`` and a trailing newline so two exports of the same log are
    byte-identical and a committed fixture diffs cleanly when a human edits
    one ``expect`` block.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(suite.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_suite(path: Path | str) -> FixtureSuite:
    """Read a suite back. Raises on a document that is not one."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixture file is not a JSON object")
    if payload.get("kind") != FIXTURE_KIND:
        raise ValueError(f"not a {FIXTURE_KIND} document (kind={payload.get('kind')!r})")
    version = payload.get("schema_version")
    if version != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"fixture schema_version {version!r} is not supported "
            f"(this build reads {FIXTURE_SCHEMA_VERSION})"
        )
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("fixture file has no cases list")
    return FixtureSuite(
        cases=tuple(FixtureCase.from_dict(case) for case in raw_cases),
        bucket=str(payload.get("bucket") or "unknown"),
        intent=str(payload.get("intent") or "goal"),
        source=dict(payload.get("source") or {}),
        generated_at=str(payload.get("generated_at") or ""),
        notes=tuple(payload.get("notes") or ()),
    )


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """What one case did when it was run."""

    id: str
    query: str
    outcome: str
    expect: dict[str, Any]
    observed: dict[str, Any]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "outcome": self.outcome,
            "expect": self.expect,
            "observed": self.observed,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class FixtureRun:
    """Every case's outcome, plus the one number a caller gates on."""

    outcomes: tuple[CaseOutcome, ...]
    bucket: str
    intent: str

    @property
    def failed(self) -> bool:
        """Pending never fails a run. An unanswered question is not a defect."""
        return any(outcome.outcome in (OUTCOME_FAIL, OUTCOME_ERROR) for outcome in self.outcomes)

    def counts(self) -> dict[str, int]:
        tally = {
            OUTCOME_PASS: 0,
            OUTCOME_FAIL: 0,
            OUTCOME_PENDING: 0,
            OUTCOME_ERROR: 0,
        }
        for outcome in self.outcomes:
            tally[outcome.outcome] += 1
        return tally

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "intent": self.intent,
            "counts": self.counts(),
            "failed": self.failed,
            "cases": [outcome.to_dict() for outcome in self.outcomes],
        }


def _observe(response: Mapping[str, Any]) -> dict[str, Any]:
    owner = response.get("selected_owner")
    owner_file = owner.get("file") if isinstance(owner, Mapping) else None
    results = response.get("results")
    return {
        "owner": owner_file,
        "confidence": response.get("confidence"),
        "status": response.get("status", "ok"),
        "result_count": len(results) if isinstance(results, list) else 0,
        "no_match": response.get("confidence") == CONFIDENCE_NO_MATCH,
    }


def _judge(expect: Mapping[str, Any], seen: Mapping[str, Any]) -> tuple[str, str]:
    kind = expect.get("kind")
    owner = seen["owner"]
    confidence = seen["confidence"]

    if kind == EXPECT_TODO:
        return OUTCOME_PENDING, "no expectation recorded yet"

    if seen["status"] == "error" and kind != EXPECT_SUCCEEDS:
        # An error answers no question about ranking. Reporting it as a failed
        # owner assertion would send the reader to the ranker for a problem in
        # the index.
        return OUTCOME_ERROR, f"search returned status=error; {kind} was not evaluated"

    if kind == EXPECT_SUCCEEDS:
        if seen["status"] == "error":
            return OUTCOME_FAIL, "search still reached no corpus"
        return OUTCOME_PASS, "search reached a corpus"

    if kind == EXPECT_OWNER:
        expected = expect.get("owner")
        if owner == expected:
            return OUTCOME_PASS, f"owner is {expected!r}"
        return OUTCOME_FAIL, f"expected owner {expected!r}, got {owner!r}"

    if kind == EXPECT_ANY_OF:
        allowed = list(expect.get("owners") or ())
        if owner in allowed:
            return OUTCOME_PASS, f"owner {owner!r} is allowed"
        return OUTCOME_FAIL, f"expected one of {allowed!r}, got {owner!r}"

    if kind == EXPECT_NOT_OWNER:
        denied = list(expect.get("owners") or ())
        if owner not in denied:
            return OUTCOME_PASS, f"owner {owner!r} is not one of the rejected files"
        return OUTCOME_FAIL, f"owner {owner!r} is rejected by this case"

    if kind == EXPECT_NO_MATCH:
        if seen["no_match"]:
            return OUTCOME_PASS, "still an honest no-match"
        return OUTCOME_FAIL, f"expected a no-match, got confidence {confidence!r} owner {owner!r}"

    if kind == EXPECT_NO_MATCH_RESOLVES:
        if not seen["no_match"]:
            return OUTCOME_PASS, f"no longer a no-match (confidence {confidence!r})"
        return OUTCOME_FAIL, "still a no-match"

    if kind == EXPECT_MIN_CONFIDENCE:
        floor = expect.get("confidence")
        have = _CONFIDENCE_RANK.get(str(confidence), -1)
        want = _CONFIDENCE_RANK.get(str(floor), -1)
        if want < 0:
            return OUTCOME_ERROR, f"unknown confidence floor {floor!r}"
        if have >= want:
            return OUTCOME_PASS, f"confidence {confidence!r} meets the {floor!r} floor"
        return OUTCOME_FAIL, f"confidence {confidence!r} is below the {floor!r} floor"

    return OUTCOME_ERROR, f"unknown expect.kind {kind!r}"


async def run_suite(suite: FixtureSuite, search: SearchCallable) -> FixtureRun:
    """Execute every case against *search* and judge it.

    A case that raises is an ``error`` outcome carrying the exception, not a
    failure and not a crash: one query that blows up must not decide the fate
    of the other thirty-nine, and it must not be reported as a ranking defect
    either.
    """
    outcomes: list[CaseOutcome] = []
    for case in suite.cases:
        if case.expect.get("kind") == EXPECT_TODO:
            outcomes.append(
                CaseOutcome(
                    id=case.id,
                    query=case.query,
                    outcome=OUTCOME_PENDING,
                    expect=dict(case.expect),
                    observed={},
                    message="no expectation recorded yet; not run",
                )
            )
            continue
        try:
            response = await search(case.query, limit=case.limit, mode=case.mode)
        except Exception as exc:
            outcomes.append(
                CaseOutcome(
                    id=case.id,
                    query=case.query,
                    outcome=OUTCOME_ERROR,
                    expect=dict(case.expect),
                    observed={},
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        seen = _observe(response if isinstance(response, Mapping) else {})
        outcome, message = _judge(case.expect, seen)
        outcomes.append(
            CaseOutcome(
                id=case.id,
                query=case.query,
                outcome=outcome,
                expect=dict(case.expect),
                observed=seen,
                message=message,
            )
        )
    return FixtureRun(outcomes=tuple(outcomes), bucket=suite.bucket, intent=suite.intent)
