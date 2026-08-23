"""Reading the source-search query log: quality buckets and eval fixtures.

:mod:`..query_log` writes one JSON object per retrieval. This package is its
only reader. It answers two questions:

* **How is retrieval doing?** :func:`report_from_path` aggregates the log into
  the four quality buckets — no-match, low-confidence, wrong-owner, error —
  with counts, rates, a trend over time windows, and the worst offenders in
  each. Everything it could not account for (unparseable lines, undated rows,
  rows with no corpus generation) is declared rather than dropped.
* **What should stop happening?** :func:`export_bucket` turns any bucket into
  a fixture suite that :func:`run_suite` executes, so a bad week of queries
  becomes next week's regression suite.

The reader never writes to the log and never opens a retrieval store: it reads
one text file. That is what makes it safe to run against a live repository.
"""

from __future__ import annotations

from .buckets import (
    BUCKET_DEFINITIONS,
    BUCKET_ERROR,
    BUCKET_HEALTHY,
    BUCKET_LOW_CONFIDENCE,
    BUCKET_NO_MATCH,
    BUCKET_WRONG_OWNER,
    BUCKETS,
    CONFIDENCE_CAUTION,
    CONFIDENCE_CONFIDENT,
    CONFIDENCE_NO_MATCH,
    Classification,
    OwnerConflict,
    classify,
    classify_all,
    detect_owner_conflicts,
    error_basis,
)
from .fixtures import (
    EXPECT_KINDS,
    FIXTURE_KIND,
    FIXTURE_SCHEMA_VERSION,
    INTENTS,
    CaseOutcome,
    FixtureCase,
    FixtureRun,
    FixtureSuite,
    export_bucket,
    load_suite,
    run_suite,
    write_suite,
)
from .records import (
    MalformedLine,
    ParsedLog,
    QueryRecord,
    TopRecord,
    normalize_query,
    parse_line,
    parse_log,
)
from .report import (
    DEFAULT_OFFENDERS,
    MAX_OFFENDER_LINES,
    REPORT_SCHEMA_VERSION,
    TIME_WINDOWS,
    Offender,
    QualityReport,
    build_report,
    report_from_path,
)

__all__ = [
    "BUCKETS",
    "BUCKET_DEFINITIONS",
    "BUCKET_ERROR",
    "BUCKET_HEALTHY",
    "BUCKET_LOW_CONFIDENCE",
    "BUCKET_NO_MATCH",
    "BUCKET_WRONG_OWNER",
    "CONFIDENCE_CAUTION",
    "CONFIDENCE_CONFIDENT",
    "CONFIDENCE_NO_MATCH",
    "DEFAULT_OFFENDERS",
    "EXPECT_KINDS",
    "FIXTURE_KIND",
    "FIXTURE_SCHEMA_VERSION",
    "INTENTS",
    "MAX_OFFENDER_LINES",
    "REPORT_SCHEMA_VERSION",
    "TIME_WINDOWS",
    "CaseOutcome",
    "Classification",
    "FixtureCase",
    "FixtureRun",
    "FixtureSuite",
    "MalformedLine",
    "Offender",
    "OwnerConflict",
    "ParsedLog",
    "QualityReport",
    "QueryRecord",
    "TopRecord",
    "build_report",
    "classify",
    "classify_all",
    "detect_owner_conflicts",
    "error_basis",
    "export_bucket",
    "load_suite",
    "normalize_query",
    "parse_line",
    "parse_log",
    "report_from_path",
    "run_suite",
    "write_suite",
]
