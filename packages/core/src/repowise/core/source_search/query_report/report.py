"""Aggregating the query log into the four quality buckets.

Counts, rates, a trend over time windows, and the worst offenders per bucket.
Everything the pass could not use is declared rather than dropped: unreadable
lines, undated rows, rows whose corpus generation is unknown, and error rows
that had to be inferred rather than read. A quality report that quietly
excluded the rows it could not parse would be measuring the wrong denominator
in exactly the situation it exists to detect.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .buckets import (
    BUCKET_DEFINITIONS,
    BUCKET_ERROR,
    BUCKET_HEALTHY,
    BUCKET_LOW_CONFIDENCE,
    BUCKET_ORDER,
    BUCKET_WRONG_OWNER,
    BUCKETS,
    Classification,
    classify_all,
)
from .records import ParsedLog, QueryRecord, parse_log

__all__ = [
    "DEFAULT_OFFENDERS",
    "MAX_OFFENDER_LINES",
    "MAX_QUERY_CHARS",
    "REPORT_SCHEMA_VERSION",
    "TIME_WINDOWS",
    "Offender",
    "QualityReport",
    "build_report",
    "report_from_path",
]

REPORT_SCHEMA_VERSION = 1

#: Offenders listed per bucket unless the caller asks for more. The exact
#: group count sits beside the list, so this cap never hides the size of a
#: problem — only the tail of its membership.
DEFAULT_OFFENDERS = 10

#: Queries are echoed back so a reader can recognise them. Long ones are cut,
#: because a report is not a copy of the log and one pathological query must
#: not be able to dominate the payload.
MAX_QUERY_CHARS = 300

#: Malformed lines quoted in the report. The count is exact regardless.
MAX_MALFORMED_SAMPLE = 10

#: Log lines listed per offender. Its own cap rather than the offender limit:
#: how many groups to show and how many occurrences of one group to locate are
#: different questions, and one query can appear thousands of times. The exact
#: occurrence count sits beside the list, so this bounds the payload without
#: understating the problem.
MAX_OFFENDER_LINES = 20

TIME_WINDOWS = ("hour", "day", "week")


def _clip(text: str) -> str:
    return text[:MAX_QUERY_CHARS] + ("…" if len(text) > MAX_QUERY_CHARS else "")


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _floor(moment: datetime, window: str) -> datetime:
    # A naive timestamp is read as UTC because that is what the writer emits;
    # guessing local time would silently shift every row in a log carried
    # between machines.
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    aware = aware.astimezone(UTC)
    if window == "hour":
        return aware.replace(minute=0, second=0, microsecond=0)
    day = aware.replace(hour=0, minute=0, second=0, microsecond=0)
    if window == "day":
        return day
    return day - timedelta(days=day.weekday())


@dataclass(frozen=True, slots=True)
class Offender:
    """One repeated question inside a bucket, with what it was served."""

    query: str
    occurrences: int
    owners: tuple[str | None, ...]
    confidences: tuple[str, ...]
    best_dense_cosine: float | None
    first_seen: str | None
    last_seen: str | None
    line_numbers: tuple[int, ...]
    lines_total: int
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": _clip(self.query),
            "occurrences": self.occurrences,
            "served_owners": list(self.owners),
            "confidences": list(self.confidences),
            "best_dense_cosine": self.best_dense_cosine,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "log_lines": list(self.line_numbers),
            "log_lines_listed": len(self.line_numbers),
            "log_lines_total": self.lines_total,
        }
        payload.update(self.detail)
        return payload


def _group_key(
    record: QueryRecord,
    classification: Classification,
    bucket: str,
) -> tuple[Any, ...]:
    """What counts as "the same offence" for grouping.

    Everywhere but wrong-owner it is the question, so a query asked forty times
    is one offender with a count and not forty rows.

    Wrong-owner also keys on the incident, because one question can contradict
    itself twice over two different corpus generations and those are two
    incidents with two sets of rivals. Merging them would report the second
    one's occurrences under the first one's rival list. Verdict-basis rows are
    keyed on the question alone: a verdict deliberately spans generations, so
    splitting by one would fragment a single finding.
    """
    if bucket != BUCKET_WRONG_OWNER or classification.conflict is None:
        return (record.normalized_query,)
    conflict = classification.conflict
    if conflict.basis == "verdict":
        return (record.normalized_query, "verdict")
    return (record.normalized_query, conflict.basis, conflict.generation)


def _offenders(
    records: Sequence[QueryRecord],
    classifications: Mapping[int, Classification],
    bucket: str,
    limit: int,
) -> tuple[list[Offender], int]:
    groups: dict[tuple[Any, ...], list[QueryRecord]] = defaultdict(list)
    for record in records:
        classification = classifications.get(record.line_number)
        if classification is not None and classification.bucket == bucket:
            groups[_group_key(record, classification, bucket)].append(record)

    offenders: list[Offender] = []
    for members in groups.values():
        owners: list[str | None] = []
        confidences: list[str] = []
        for record in members:
            if record.selected_owner_file not in owners:
                owners.append(record.selected_owner_file)
            if record.confidence not in confidences:
                confidences.append(record.confidence)
        cosines = [
            record.best_dense_cosine for record in members if record.best_dense_cosine is not None
        ]
        # Ordered by the parsed instant, not by the raw string. Lexical order on
        # ISO text is only chronological while every row carries the same UTC
        # offset, which is true of what this writer emits and not a property the
        # reader should depend on.
        dated = sorted(
            (record for record in members if record.ts is not None and record.ts_raw),
            key=lambda record: record.ts,  # type: ignore[arg-type,return-value]
        )
        stamps = [record.ts_raw for record in dated]
        detail: dict[str, Any] = {}
        if bucket == BUCKET_ERROR:
            bases = sorted(
                {classifications[record.line_number].error_basis or "unknown" for record in members}
            )
            legs = sorted({str(leg.get("leg")) for record in members for leg in record.failed_legs})
            detail["error_basis"] = bases
            detail["failed_legs"] = legs
            detail["error_codes"] = sorted(
                {record.error_code for record in members if record.error_code}
            )
        if bucket == BUCKET_WRONG_OWNER:
            conflict = classifications[members[0].line_number].conflict
            detail["conflict_basis"] = conflict.basis if conflict else None
            detail["rival_owners"] = list(conflict.owners) if conflict else []
            detail["expected_owner"] = conflict.expected_owner if conflict else None
            detail["generation"] = conflict.generation if conflict else None
            detail["generation_known"] = conflict.generation_known if conflict else False
        offenders.append(
            Offender(
                query=members[0].query,
                occurrences=len(members),
                owners=tuple(owners),
                confidences=tuple(confidences),
                best_dense_cosine=max(cosines) if cosines else None,
                first_seen=stamps[0] if stamps else None,
                last_seen=stamps[-1] if stamps else None,
                line_numbers=tuple(record.line_number for record in members[:MAX_OFFENDER_LINES]),
                lines_total=len(members),
                detail=detail,
            )
        )

    def severity(offender: Offender) -> tuple:
        # Frequency first in every bucket: a question the corpus fails once is
        # an anecdote, one it fails forty times is the thing to fix. The
        # tiebreakers differ by what "worse" means in that bucket, and the
        # final key is the query text so ties never reorder between runs.
        if bucket == BUCKET_LOW_CONFIDENCE:
            # Thinner evidence is worse; unknown cosine sorts as thinnest.
            thinness = -1.0 if offender.best_dense_cosine is None else offender.best_dense_cosine
            return (-offender.occurrences, thinness, offender.query)
        if bucket == BUCKET_WRONG_OWNER:
            return (
                -offender.occurrences,
                -len(offender.detail.get("rival_owners") or ()),
                offender.query,
            )
        return (-offender.occurrences, offender.query)

    offenders.sort(key=severity)
    return offenders[:limit], len(groups)


def _trend(
    records: Sequence[QueryRecord],
    classifications: Mapping[int, Classification],
    window: str,
) -> tuple[list[dict[str, Any]], int]:
    points: dict[datetime, Counter] = defaultdict(Counter)
    undated = 0
    for record in records:
        if record.ts is None:
            undated += 1
            continue
        classification = classifications.get(record.line_number)
        bucket = classification.bucket if classification else BUCKET_HEALTHY
        slot = points[_floor(record.ts, window)]
        slot["total"] += 1
        slot[bucket] += 1
    series = []
    for start in sorted(points):
        counts = points[start]
        total = counts["total"]
        entry: dict[str, Any] = {
            "window_start": start.isoformat(),
            "total": total,
        }
        for bucket in BUCKET_ORDER:
            entry[bucket] = counts[bucket]
        for bucket in BUCKETS:
            entry[f"{bucket}_rate"] = _rate(counts[bucket], total)
        series.append(entry)
    return series, undated


@dataclass
class QualityReport:
    """The aggregated view, plus everything it could not account for."""

    parsed: ParsedLog
    classifications: dict[int, Classification]
    window: str
    offender_limit: int
    verdicts_applied: int

    def bucket_records(self, bucket: str) -> list[QueryRecord]:
        return [
            record
            for record in self.parsed.records
            if (classification := self.classifications.get(record.line_number)) is not None
            and classification.bucket == bucket
        ]

    def caveats(self) -> list[str]:
        """What this report knows it cannot see.

        Computed, not boilerplate: every entry here is triggered by something
        actually present in this log, so an empty list is a real statement.
        """
        notes: list[str] = []
        total = len(self.parsed.records)
        if self.parsed.read_error:
            notes.append(
                f"The log could not be fully read ({self.parsed.read_error}); "
                "every count below covers only what was readable."
            )
        if self.parsed.truncated and not self.parsed.read_error:
            notes.append(
                f"Reading stopped at line {self.parsed.lines_read}; later queries are not counted."
            )
        if self.parsed.malformed:
            notes.append(
                f"{len(self.parsed.malformed)} line(s) could not be parsed and are excluded "
                "from every rate. They are listed under 'malformed'."
            )
        without_generation = sum(1 for record in self.parsed.records if record.generation is None)
        if without_generation and total:
            notes.append(
                f"{without_generation} of {total} rows carry no corpus generation, so an owner "
                "disagreement among them cannot be told apart from a legitimate reindex. "
                "Those wrong-owner rows may be over-reported."
            )
        inferred = sum(
            1
            for classification in self.classifications.values()
            if classification.error_basis == "inferred"
        )
        if inferred:
            notes.append(
                f"{inferred} error row(s) were inferred from the empty-caution signature rather "
                "than read from a status field, because they predate it. If the coordinator ever "
                "serves an empty caution for an unrelated reason, that arm over-reports."
            )
        if total and not any(record.failed_legs for record in self.parsed.records):
            declared = any(record.status == "error" for record in self.parsed.records)
            if not declared:
                notes.append(
                    "No row in this log declares a status or failed legs, so every error verdict "
                    "here is inferred. Populating QueryEvent.status at the call site removes the "
                    "inference."
                )
        undated = sum(1 for record in self.parsed.records if record.ts is None)
        if undated:
            notes.append(
                f"{undated} row(s) have no usable timestamp; they are counted in the totals but "
                "absent from the trend."
            )
        warned = self.parsed.warned
        if warned:
            notes.append(
                f"{warned} row(s) had an unusable non-deciding field and were kept with a "
                "warning; see 'field_warnings'."
            )
        return notes

    def to_dict(self) -> dict[str, Any]:
        records = self.parsed.records
        total = len(records)
        counts = Counter(classification.bucket for classification in self.classifications.values())
        buckets: dict[str, Any] = {}
        for bucket in BUCKET_ORDER:
            entry: dict[str, Any] = {
                "count": counts.get(bucket, 0),
                "rate": _rate(counts.get(bucket, 0), total),
                "definition": BUCKET_DEFINITIONS[bucket],
            }
            if bucket != BUCKET_HEALTHY:
                offenders, groups = _offenders(
                    records, self.classifications, bucket, self.offender_limit
                )
                entry["distinct_queries"] = groups
                entry["offenders_listed"] = len(offenders)
                entry["offenders"] = [offender.to_dict() for offender in offenders]
            buckets[bucket] = entry

        series, undated = _trend(records, self.classifications, self.window)
        degraded = [
            record
            for record in records
            if (classification := self.classifications.get(record.line_number)) is not None
            and classification.degraded
        ]
        leg_counter: Counter = Counter()
        for record in degraded:
            for leg in record.failed_legs:
                leg_counter[str(leg.get("leg"))] += 1
        warning_counter: Counter = Counter()
        for record in records:
            warning_counter.update(record.warnings)

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "source": {
                "path": str(self.parsed.path) if self.parsed.path else None,
                "lines_read": self.parsed.lines_read,
                "blank_lines": self.parsed.blank_lines,
                "truncated": self.parsed.truncated,
                "read_error": self.parsed.read_error,
            },
            "totals": {
                "analysed": total,
                "malformed": len(self.parsed.malformed),
                "with_field_warnings": self.parsed.warned,
                "verdicts_applied": self.verdicts_applied,
            },
            "buckets": buckets,
            "trend": {
                "window": self.window,
                "undated_rows": undated,
                "points": series,
            },
            "degraded": {
                "count": len(degraded),
                "rate": _rate(len(degraded), total),
                "by_leg": dict(sorted(leg_counter.items())),
                "definition": (
                    "An answer was served while at least one retrieval leg was dead. Not an "
                    "error — the corpus was read, but not all of it."
                ),
            },
            "malformed": {
                "count": len(self.parsed.malformed),
                "rate": _rate(len(self.parsed.malformed), total + len(self.parsed.malformed)),
                "by_reason": dict(
                    sorted(Counter(line.reason for line in self.parsed.malformed).items())
                ),
                "sample": [
                    {
                        "line": line.line_number,
                        "reason": line.reason,
                        "detail": line.detail,
                        "excerpt": line.excerpt,
                    }
                    for line in self.parsed.malformed[:MAX_MALFORMED_SAMPLE]
                ],
                "sample_listed": min(len(self.parsed.malformed), MAX_MALFORMED_SAMPLE),
            },
            "field_warnings": dict(sorted(warning_counter.items())),
            "caveats": self.caveats(),
        }


def build_report(
    parsed: ParsedLog,
    *,
    window: str = "day",
    offender_limit: int = DEFAULT_OFFENDERS,
    verdicts: Mapping[str, str | None] | None = None,
) -> QualityReport:
    """Classify a parsed log and wrap it in a report."""
    if window not in TIME_WINDOWS:
        raise ValueError(f"window must be one of {TIME_WINDOWS}, got {window!r}")
    classifications = classify_all(parsed.records, verdicts=verdicts)
    applied = (
        sum(
            1
            for classification in classifications.values()
            if classification.conflict is not None and classification.conflict.basis == "verdict"
        )
        if verdicts
        else 0
    )
    return QualityReport(
        parsed=parsed,
        classifications=classifications,
        window=window,
        offender_limit=offender_limit,
        verdicts_applied=applied,
    )


def report_from_path(
    path: Path | str,
    *,
    window: str = "day",
    offender_limit: int = DEFAULT_OFFENDERS,
    max_lines: int | None = None,
    verdicts: Mapping[str, str | None] | None = None,
) -> QualityReport:
    """Read *path* and report on it. Never raises for a bad or missing log."""
    return build_report(
        parse_log(path, max_lines=max_lines),
        window=window,
        offender_limit=offender_limit,
        verdicts=verdicts,
    )
