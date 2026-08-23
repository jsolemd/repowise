"""Reading the query log back: the four buckets, and what they refuse to claim.

The bucket assertions are written against a synthetic log built here rather
than a recorded one, because the interesting cases — a confident answer
contradicted by a later confident answer over the *same* corpus, and the same
thing over a *different* corpus — differ by one field and would be almost
impossible to catch in captured traffic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from repowise.core.source_search.coordinator import (
    CAUTION,
    CONFIDENT,
    NO_MATCH,
    SourceSearchCoordinator,
)
from repowise.core.source_search.query_log import QueryEvent, QueryLog, TopEntry
from repowise.core.source_search.query_report import (
    BUCKET_ERROR,
    BUCKET_HEALTHY,
    BUCKET_LOW_CONFIDENCE,
    BUCKET_NO_MATCH,
    BUCKET_WRONG_OWNER,
    CONFIDENCE_CAUTION,
    CONFIDENCE_CONFIDENT,
    CONFIDENCE_NO_MATCH,
    FIXTURE_KIND,
    MAX_OFFENDER_LINES,
    export_bucket,
    load_suite,
    parse_line,
    parse_log,
    report_from_path,
    run_suite,
    write_suite,
)
from repowise.core.source_search.vector_store import SourceChunkHit

BASE = datetime(2026, 8, 17, 9, 0, tzinfo=UTC)
GEN_A = "aaaaaaaaaaaa"
GEN_B = "bbbbbbbbbbbb"


# ---------------------------------------------------------------------------
# Synthetic log
# ---------------------------------------------------------------------------


def _ts(minutes: int) -> str:
    return (BASE + timedelta(minutes=minutes)).isoformat(timespec="milliseconds")


def _top(file: str, cosine: float, rank: int | None = 3, exact: bool = False) -> TopEntry:
    return TopEntry(
        file=file,
        lane="source",
        dense_cosine=cosine,
        lexical_rank=rank,
        exact_name=exact,
        fused_score=0.0134,
        concept_coverage=0.62,
        same_path_corroborated=True,
    )


def _event(**overrides: Any) -> QueryEvent:
    fields: dict[str, Any] = {
        "query": "where is the retry policy",
        "mode": "hybrid",
        "limit": 5,
        "latency_ms": 11.0,
        "confidence": CONFIDENT,
        "result_count": 3,
        "top": [_top("taskq/retry.py", 0.81, 1, True)],
        "selected_owner_file": "taskq/retry.py",
        "no_match": False,
        "ts": _ts(0),
        "generation": GEN_A,
    }
    fields.update(overrides)
    return QueryEvent(**fields)


def _no_match(query: str, minute: int, generation: str = GEN_A) -> QueryEvent:
    return _event(
        query=query,
        confidence=NO_MATCH,
        result_count=0,
        top=[],
        selected_owner_file=None,
        no_match=True,
        ts=_ts(minute),
        generation=generation,
    )


def _caution(query: str, owner: str, minute: int, cosine: float = 0.34) -> QueryEvent:
    return _event(
        query=query,
        confidence=CAUTION,
        result_count=4,
        top=[_top(owner, cosine, 7)],
        selected_owner_file=owner,
        ts=_ts(minute),
    )


def _write(path, events, extra_lines: tuple[str, ...] = ()):
    log = QueryLog(path)
    for event in events:
        assert log.append(event) is True
    if extra_lines:
        with path.open("a", encoding="utf-8") as handle:
            for line in extra_lines:
                handle.write(line + "\n")
    return path


@pytest.fixture
def busy_log(tmp_path):
    """One log covering every bucket, plus the shapes that must NOT bucket."""
    path = tmp_path / "query_log.jsonl"
    events = [
        # healthy: four identical confident answers
        *[_event(ts=_ts(i)) for i in range(4)],
        # no_match: one question the corpus keeps failing, plus a one-off
        *[_no_match("how do I configure kubernetes ingress", 10 + i) for i in range(5)],
        _no_match("what is the licence of this project", 20),
        # low_confidence
        *[_caution("which module owns storage", "taskq/storage.py", 30 + i) for i in range(3)],
        # degraded but served: legs died, an answer came back anyway
        _event(
            query="who calls enqueue",
            confidence=CAUTION,
            result_count=2,
            top=[_top("taskq/queue.py", 0.41, 2)],
            selected_owner_file="taskq/queue.py",
            ts=_ts(35),
            failed_legs=[{"leg": "wiki dense", "error": "TimeoutError", "detail": "slow"}],
        ),
        # wrong_owner: same question, same generation, two confident answers
        _event(
            query="where does a task get its identity",
            top=[_top("taskq/task.py", 0.77, 1, True)],
            selected_owner_file="taskq/task.py",
            ts=_ts(40),
        ),
        _event(
            query="Where does a task get its identity",
            top=[_top("taskq/queue.py", 0.74, 1, True)],
            selected_owner_file="taskq/queue.py",
            ts=_ts(41),
        ),
        # same question, DIFFERENT generation: the corpus moved, not a defect
        _event(
            query="where does a task get its identity",
            top=[_top("taskq/identity.py", 0.79, 1, True)],
            selected_owner_file="taskq/identity.py",
            ts=_ts(42),
            generation=GEN_B,
        ),
        # two cautious answers that disagree: unsure, not wrong
        _caution("where is the entry point", "taskq/cli.py", 45),
        _caution("where is the entry point", "taskq/__init__.py", 46),
        # error: declared
        _event(
            query="how does the scheduler back off",
            confidence=CAUTION,
            result_count=0,
            top=[],
            selected_owner_file=None,
            ts=_ts(50),
            status="error",
            error_code="source_search_unavailable",
            failed_legs=[{"leg": "source dense", "error": "OperationalError", "detail": "gone"}],
        ),
        # second day, so the trend has more than one point
        *[_no_match("how do I configure kubernetes ingress", 1440 + i, GEN_B) for i in range(3)],
    ]
    return _write(path, events)


# ---------------------------------------------------------------------------
# The contract with the producer
# ---------------------------------------------------------------------------


def test_the_confidence_vocabulary_matches_the_coordinators():
    """These three strings are restated in the reporter, not imported.

    Importing the coordinator would drag the retrieval stack into a pass that
    reads a text file. This test is the price of that: a rename over there
    fails here instead of silently reclassifying every row in every report.
    """
    assert (CONFIDENCE_CONFIDENT, CONFIDENCE_CAUTION, CONFIDENCE_NO_MATCH) == (
        CONFIDENT,
        CAUTION,
        NO_MATCH,
    )


def test_a_record_written_before_the_new_fields_still_reads(tmp_path):
    """The log-format compatibility promise, as a test.

    Lines written by an older build stay in the file forever. Every field
    added since must read back as the value that line meant when it was
    written — not as an unknown third state.
    """
    legacy = json.dumps(
        {
            "ts": _ts(1),
            "query": "why is the outbox empty",
            "mode": "hybrid",
            "limit": 5,
            "latency_ms": 2.2,
            "confidence": "confident",
            "result_count": 1,
            "top": [],
            "selected_owner_file": "taskq/outbox.py",
            "selected_owner_evidence": None,
            "no_match": False,
        }
    )
    record = parse_line(legacy, 1)
    assert record.query == "why is the outbox empty"
    assert record.status == "ok"
    assert record.error_code is None
    assert record.failed_legs == ()
    assert record.generation is None


def test_the_new_fields_round_trip_through_the_writer(tmp_path):
    path = _write(
        tmp_path / "q.jsonl",
        [
            _event(
                status="error",
                error_code="source_search_unavailable",
                failed_legs=[{"leg": "source dense", "error": "OperationalError", "detail": "x"}],
                generation=GEN_A,
            )
        ],
    )
    record = parse_log(path).records[0]
    assert record.status == "error"
    assert record.error_code == "source_search_unavailable"
    assert record.failed_legs[0]["leg"] == "source dense"
    assert record.generation == GEN_A


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------


def _counts(report) -> dict[str, int]:
    return {name: entry["count"] for name, entry in report.to_dict()["buckets"].items()}


def test_every_bucket_is_populated_and_the_counts_partition_the_log(busy_log):
    report = report_from_path(busy_log)
    counts = _counts(report)
    payload = report.to_dict()
    assert counts[BUCKET_NO_MATCH] == 9
    assert counts[BUCKET_LOW_CONFIDENCE] == 6
    assert counts[BUCKET_WRONG_OWNER] == 2
    assert counts[BUCKET_ERROR] == 1
    assert counts[BUCKET_HEALTHY] == 5
    # Precedence assigns each row exactly one bucket, so the parts sum to the
    # whole and no rate is inflated by double counting.
    assert sum(counts.values()) == payload["totals"]["analysed"] == 23


def test_a_moved_corpus_is_not_a_wrong_owner(busy_log):
    """The same question answered differently after a reindex is allowed.

    ``taskq/identity.py`` answers the identity question under generation B
    while A said ``taskq/task.py``. Charging retrieval for that would charge
    it for the repository's own history.
    """
    report = report_from_path(busy_log)
    wrong = {record.line_number for record in report.bucket_records(BUCKET_WRONG_OWNER)}
    identity_row = next(
        record
        for record in report.parsed.records
        if record.selected_owner_file == "taskq/identity.py"
    )
    assert identity_row.line_number not in wrong
    assert identity_row.generation == GEN_B


def test_two_cautious_answers_may_disagree_without_being_wrong(busy_log):
    """An answer that never claimed to be sure is allowed to change.

    Counting this as a wrong owner would charge the most serious bucket with
    the least serious evidence, and double-count what low-confidence already
    measures.
    """
    report = report_from_path(busy_log)
    entry_rows = [
        record
        for record in report.parsed.records
        if record.normalized_query == "where is the entry point"
    ]
    assert len(entry_rows) == 2
    assert {row.selected_owner_file for row in entry_rows} == {
        "taskq/cli.py",
        "taskq/__init__.py",
    }
    buckets = {report.classifications[row.line_number].bucket for row in entry_rows}
    assert buckets == {BUCKET_LOW_CONFIDENCE}


def test_a_wrong_owner_names_its_rivals_and_its_generation(busy_log):
    payload = report_from_path(busy_log).to_dict()
    offender = payload["buckets"][BUCKET_WRONG_OWNER]["offenders"][0]
    assert offender["query"] == "where does a task get its identity"
    assert offender["conflict_basis"] == "self_contradiction"
    assert offender["rival_owners"] == ["taskq/task.py", "taskq/queue.py"]
    assert offender["generation"] == GEN_A
    # Self-contradiction knows the answers disagree, never which one is right.
    assert offender["expected_owner"] is None


def test_a_verdict_beats_the_disagreement_it_explains(busy_log):
    """Ground truth settles the question, and vindicates the row that was right.

    Without this, the self-contradiction pass would keep charging the correct
    answer purely for having been disagreed with — the one row a verdict is
    supposed to clear.
    """
    report = report_from_path(
        busy_log,
        verdicts={"where does a task get its identity": "taskq/task.py"},
    )
    payload = report.to_dict()
    charged = report.bucket_records(BUCKET_WRONG_OWNER)
    assert {record.selected_owner_file for record in charged} == {
        "taskq/queue.py",
        "taskq/identity.py",
    }
    assert payload["totals"]["verdicts_applied"] == 2
    offender = payload["buckets"][BUCKET_WRONG_OWNER]["offenders"][0]
    assert offender["conflict_basis"] == "verdict"
    assert offender["expected_owner"] == "taskq/task.py"
    # The row that served the verdict's own answer is no longer a defect.
    vindicated = next(
        record for record in report.parsed.records if record.selected_owner_file == "taskq/task.py"
    )
    assert report.classifications[vindicated.line_number].bucket == BUCKET_HEALTHY


def test_two_contradictions_over_two_generations_are_two_offenders(tmp_path):
    """One question can contradict itself twice, and they are separate incidents.

    Merging them would report the second incident's occurrences under the
    first one's rival list.
    """
    path = _write(
        tmp_path / "q.jsonl",
        [
            _event(query="who owns retries", selected_owner_file="a.py", ts=_ts(1)),
            _event(query="who owns retries", selected_owner_file="b.py", ts=_ts(2)),
            _event(
                query="who owns retries",
                selected_owner_file="c.py",
                ts=_ts(3),
                generation=GEN_B,
            ),
            _event(
                query="who owns retries",
                selected_owner_file="d.py",
                ts=_ts(4),
                generation=GEN_B,
            ),
        ],
    )
    offenders = report_from_path(path).to_dict()["buckets"][BUCKET_WRONG_OWNER]["offenders"]
    assert len(offenders) == 2
    assert {offender["generation"] for offender in offenders} == {GEN_A, GEN_B}
    rivals = {tuple(offender["rival_owners"]) for offender in offenders}
    assert rivals == {("a.py", "b.py"), ("c.py", "d.py")}


def test_a_verdict_reaches_across_generations(busy_log):
    """A verdict is about the question, not about one build of the corpus.

    Generations turn over on every reindex, so scoping a verdict to one would
    make it apply to almost nothing.
    """
    report = report_from_path(
        busy_log, verdicts={"where does a task get its identity": "taskq/task.py"}
    )
    charged = {record.generation for record in report.bucket_records(BUCKET_WRONG_OWNER)}
    assert charged == {GEN_A, GEN_B}


def test_an_outage_is_not_a_rival_answer(tmp_path):
    """An error's absent owner means "did not look", not "looked and found nothing".

    Letting it join a conflict group would manufacture contradictions out of
    outages: every confident answer to a question that once errored would be
    reported as wrong.
    """
    path = _write(
        tmp_path / "q.jsonl",
        [
            _event(query="where is the retry policy", ts=_ts(1)),
            _event(
                query="where is the retry policy",
                confidence=CAUTION,
                result_count=0,
                top=[],
                selected_owner_file=None,
                ts=_ts(2),
                status="error",
                error_code="source_search_unavailable",
            ),
        ],
    )
    counts = _counts(report_from_path(path))
    assert counts[BUCKET_WRONG_OWNER] == 0
    assert counts[BUCKET_ERROR] == 1
    assert counts[BUCKET_HEALTHY] == 1


def test_an_error_predating_the_status_field_is_still_found(tmp_path):
    """The one signature only an unread corpus produces.

    ``_classify`` returns ``no_match`` for an empty window, so caution with no
    results, no owner and ``no_match`` false cannot be an honest answer — it is
    the error envelope or the hard-failure pin. Both mean the corpus was not
    fully read.
    """
    legacy = json.dumps(
        {
            "ts": _ts(1),
            "query": "why is the outbox empty",
            "mode": "hybrid",
            "limit": 5,
            "latency_ms": 2.2,
            "confidence": "caution",
            "result_count": 0,
            "top": [],
            "selected_owner_file": None,
            "selected_owner_evidence": None,
            "no_match": False,
        }
    )
    path = _write(tmp_path / "q.jsonl", [], extra_lines=(legacy,))
    payload = report_from_path(path).to_dict()
    assert payload["buckets"][BUCKET_ERROR]["count"] == 1
    assert payload["buckets"][BUCKET_ERROR]["offenders"][0]["error_basis"] == ["inferred"]
    # And the inference is disclosed rather than passed off as a reading.
    assert any("inferred" in note for note in payload["caveats"])


def test_a_degraded_answer_is_counted_apart_from_an_outage(busy_log):
    payload = report_from_path(busy_log).to_dict()
    assert payload["degraded"]["count"] == 1
    assert payload["degraded"]["by_leg"] == {"wiki dense": 1}
    # It served an answer, so it is not an error.
    assert payload["buckets"][BUCKET_ERROR]["count"] == 1


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_corruption_is_counted_and_located_without_disturbing_the_analysis(busy_log):
    """A bad line must never vanish, and must never move a good line's bucket."""
    clean = report_from_path(busy_log).to_dict()
    with busy_log.open("a", encoding="utf-8") as handle:
        handle.write('{"query": "torn write", "confid\n')
        handle.write("not json at all\n")
        handle.write('["a", "list"]\n')
        handle.write('{"mode": "hybrid", "confidence": "caution"}\n')
        handle.write('{"query": 7, "confidence": "caution"}\n')
    dirty = report_from_path(busy_log).to_dict()

    assert clean["malformed"]["count"] == 0
    assert dirty["malformed"]["count"] == 5
    assert dirty["totals"]["analysed"] == clean["totals"]["analysed"]
    assert {name: entry["count"] for name, entry in dirty["buckets"].items()} == {
        name: entry["count"] for name, entry in clean["buckets"].items()
    }
    assert dirty["malformed"]["by_reason"] == {
        "bad_type": 1,
        "invalid_json": 2,
        "missing_field": 1,
        "not_an_object": 1,
    }
    # Line numbers, because a report that cannot say where the damage is
    # leaves the reader grepping a log they already have.
    assert [entry["line"] for entry in dirty["malformed"]["sample"]] == [24, 25, 26, 27, 28]
    assert any("could not be parsed" in note for note in dirty["caveats"])


def test_the_malformed_tally_has_a_closed_vocabulary(tmp_path):
    """Reasons must not carry the parser's character offsets.

    Folding the message into the key would give a file of three thousand torn
    lines three thousand distinct "reasons" — a copy of the log wearing a
    histogram's name.
    """
    path = tmp_path / "q.jsonl"
    path.write_text("".join(f'{{"query": "q{i}", "conf\n' for i in range(20)), encoding="utf-8")
    payload = report_from_path(path).to_dict()
    assert payload["malformed"]["count"] == 20
    assert list(payload["malformed"]["by_reason"]) == ["invalid_json"]
    assert payload["malformed"]["sample_listed"] == 10


def test_a_blank_line_is_not_damage(tmp_path):
    """The writer ends every record with a newline; a trailing blank is normal."""
    path = _write(tmp_path / "q.jsonl", [_event()], extra_lines=("", ""))
    parsed = parse_log(path)
    assert parsed.blank_lines == 2
    assert not parsed.malformed
    assert len(parsed.records) == 1


def test_a_missing_log_is_a_fact_not_an_exception(tmp_path):
    payload = report_from_path(tmp_path / "never-ran.jsonl").to_dict()
    assert payload["source"]["read_error"] == "log_not_found"
    assert payload["totals"]["analysed"] == 0
    assert payload["buckets"][BUCKET_NO_MATCH]["rate"] == 0.0


def test_an_unusable_field_keeps_the_row_and_reports_the_warning(tmp_path):
    """A cosmetic defect must not discard a real quality signal."""
    line = json.dumps(
        {
            "ts": "not-a-timestamp",
            "query": "where is the retry policy",
            "confidence": "caution",
            "limit": "five",
            "result_count": 4,
            "selected_owner_file": "taskq/retry.py",
        }
    )
    path = _write(tmp_path / "q.jsonl", [], extra_lines=(line,))
    payload = report_from_path(path).to_dict()
    assert payload["totals"]["analysed"] == 1
    assert payload["totals"]["malformed"] == 0
    assert payload["field_warnings"] == {"bad_type:limit": 1, "bad_value:ts": 1}
    assert payload["buckets"][BUCKET_LOW_CONFIDENCE]["count"] == 1
    # Undated rows count in the totals but cannot be placed on the trend.
    assert payload["trend"]["undated_rows"] == 1
    assert payload["trend"]["points"] == []
    assert any("no usable timestamp" in note for note in payload["caveats"])


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------


def test_the_trend_splits_by_window(busy_log):
    daily = report_from_path(busy_log, window="day").to_dict()["trend"]
    hourly = report_from_path(busy_log, window="hour").to_dict()["trend"]
    weekly = report_from_path(busy_log, window="week").to_dict()["trend"]

    assert [point["window_start"] for point in daily["points"]] == [
        "2026-08-17T00:00:00+00:00",
        "2026-08-18T00:00:00+00:00",
    ]
    assert daily["points"][1][BUCKET_NO_MATCH] == 3
    assert daily["points"][1][f"{BUCKET_NO_MATCH}_rate"] == 1.0
    # Every row falls in the 09:00 hour of its own day, so the hourly view
    # keeps the hour rather than flattening to midnight.
    assert [point["window_start"] for point in hourly["points"]] == [
        "2026-08-17T09:00:00+00:00",
        "2026-08-18T09:00:00+00:00",
    ]
    # Both days fall in the week beginning Monday 2026-08-17.
    assert [point["window_start"] for point in weekly["points"]] == ["2026-08-17T00:00:00+00:00"]
    assert weekly["points"][0]["total"] == 23


def test_an_unknown_window_is_refused(busy_log):
    with pytest.raises(ValueError, match="window must be one of"):
        report_from_path(busy_log, window="fortnight")


# ---------------------------------------------------------------------------
# Offenders
# ---------------------------------------------------------------------------


def test_the_worst_offender_is_the_question_that_keeps_failing(busy_log):
    payload = report_from_path(busy_log).to_dict()
    bucket = payload["buckets"][BUCKET_NO_MATCH]
    assert bucket["distinct_queries"] == 2
    first = bucket["offenders"][0]
    assert first["query"] == "how do I configure kubernetes ingress"
    assert first["occurrences"] == 8
    assert first["served_owners"] == [None]
    assert first["confidences"] == [NO_MATCH]
    assert first["log_lines"][0] == 5


def test_offender_order_is_stable_across_runs(busy_log):
    first = report_from_path(busy_log).to_dict()
    second = report_from_path(busy_log).to_dict()
    assert first["buckets"] == second["buckets"]


def test_the_located_lines_are_capped_without_understating_the_problem(tmp_path):
    """How many groups to show and how many lines to locate are different caps."""
    path = _write(tmp_path / "q.jsonl", [_no_match("a repeated gap", i) for i in range(40)])
    offender = report_from_path(path, offender_limit=1).to_dict()["buckets"][BUCKET_NO_MATCH][
        "offenders"
    ][0]
    assert offender["occurrences"] == 40
    assert offender["log_lines_total"] == 40
    assert offender["log_lines_listed"] == MAX_OFFENDER_LINES
    assert len(offender["log_lines"]) == MAX_OFFENDER_LINES


def test_the_first_and_last_sighting_are_ordered_by_instant(tmp_path):
    """Not by the raw ISO text, which only sorts right at one UTC offset."""
    path = _write(
        tmp_path / "q.jsonl",
        [
            _event(
                query="q", ts="2026-08-17T09:00:00.000+02:00", confidence=NO_MATCH, no_match=True
            ),
            _event(
                query="q", ts="2026-08-17T08:30:00.000+00:00", confidence=NO_MATCH, no_match=True
            ),
        ],
    )
    offender = report_from_path(path).to_dict()["buckets"][BUCKET_NO_MATCH]["offenders"][0]
    # 09:00+02:00 is 07:00Z, so it precedes 08:30Z despite sorting later as text.
    assert offender["first_seen"] == "2026-08-17T09:00:00.000+02:00"
    assert offender["last_seen"] == "2026-08-17T08:30:00.000+00:00"


def test_a_long_query_is_clipped_rather_than_echoed_whole(tmp_path):
    path = _write(tmp_path / "q.jsonl", [_no_match("x" * 5000, 1)])
    payload = report_from_path(path).to_dict()
    assert len(payload["buckets"][BUCKET_NO_MATCH]["offenders"][0]["query"]) < 400


# ---------------------------------------------------------------------------
# Fixture export
# ---------------------------------------------------------------------------


def test_the_exporter_never_invents_ground_truth(busy_log):
    """A wrong-owner case is a question, not an assertion.

    The bucket exists because retrieval contradicted itself; which rival is
    right is exactly what the log cannot say. Writing the observed owner into
    ``expect`` would freeze the defect into a test that passes forever.
    """
    report = report_from_path(busy_log)
    suite = export_bucket(report, BUCKET_WRONG_OWNER)
    case = suite.cases[0]
    assert case.expect["kind"] == "todo"
    assert case.expect["candidates"] == ["taskq/task.py", "taskq/queue.py"]
    assert case.observed["served_owners"] == ["taskq/task.py", "taskq/queue.py"]


def test_goal_and_guard_point_the_same_bucket_in_opposite_directions(busy_log):
    report = report_from_path(busy_log)
    goal = export_bucket(report, BUCKET_NO_MATCH, intent="goal")
    guard = export_bucket(report, BUCKET_NO_MATCH, intent="guard")
    assert goal.cases[0].expect["kind"] == "no_match_resolves"
    assert guard.cases[0].expect["kind"] == "no_match"


def test_an_outage_has_no_guard_reading(busy_log):
    """Nobody ever wants a test asserting that a query must keep failing."""
    report = report_from_path(busy_log)
    for intent in ("goal", "guard"):
        suite = export_bucket(report, BUCKET_ERROR, intent=intent)
        assert suite.cases[0].expect["kind"] == "succeeds"


def test_a_verdict_outranks_both_intents(busy_log):
    report = report_from_path(
        busy_log, verdicts={"where does a task get its identity": "taskq/task.py"}
    )
    suite = export_bucket(report, BUCKET_WRONG_OWNER, intent="guard")
    owner_cases = [case for case in suite.cases if case.expect["kind"] == "owner"]
    assert owner_cases
    assert owner_cases[0].expect["owner"] == "taskq/task.py"


def test_repeated_queries_collapse_into_one_case_with_a_count(busy_log):
    suite = export_bucket(report_from_path(busy_log), BUCKET_NO_MATCH)
    assert len(suite.cases) == 2
    assert suite.cases[0].observed["occurrences"] == 8
    assert len(suite.cases[0].provenance["log_lines"]) == 8


def test_an_export_is_byte_identical_across_runs(busy_log, tmp_path):
    """A committed fixture must diff on its content, not on its ordering."""
    report = report_from_path(busy_log)
    first = write_suite(export_bucket(report, BUCKET_NO_MATCH), tmp_path / "a.json")
    second = write_suite(export_bucket(report, BUCKET_NO_MATCH), tmp_path / "b.json")

    def without_stamp(path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("generated_at")
        return json.dumps(payload, sort_keys=True)

    assert without_stamp(first) == without_stamp(second)
    assert first.read_text(encoding="utf-8").endswith("\n")


def test_a_suite_round_trips(busy_log, tmp_path):
    original = export_bucket(report_from_path(busy_log), BUCKET_NO_MATCH)
    path = write_suite(original, tmp_path / "suite.json")
    reloaded = load_suite(path)
    assert [case.to_dict() for case in reloaded.cases] == [
        case.to_dict() for case in original.cases
    ]
    assert reloaded.bucket == BUCKET_NO_MATCH
    assert json.loads(path.read_text(encoding="utf-8"))["kind"] == FIXTURE_KIND


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda doc: doc.update(kind="something.else"), "not a repowise"),
        (lambda doc: doc.update(schema_version=99), "schema_version"),
        (lambda doc: doc.update(cases="nope"), "no cases list"),
        (lambda doc: doc["cases"][0].pop("expect"), "expect.kind"),
        (lambda doc: doc["cases"][0].update(expect={"kind": "vibes"}), "unknown expect.kind"),
        (lambda doc: doc["cases"].insert(0, "a string"), "not an object"),
        (lambda doc: doc["cases"][0].update(query=""), "has no query"),
        (lambda doc: doc["cases"][0].update(limit="five"), "invalid literal"),
    ],
)
def test_a_document_that_is_not_a_suite_is_refused(busy_log, tmp_path, mutate, message):
    path = write_suite(
        export_bucket(report_from_path(busy_log), BUCKET_NO_MATCH), tmp_path / "s.json"
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    mutate(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_suite(path)


# ---------------------------------------------------------------------------
# Running an exported suite against a toy corpus
# ---------------------------------------------------------------------------


class _Embedder:
    dimensions = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class _SourceVectors:
    def __init__(self, hits: list[SourceChunkHit]) -> None:
        self._hits = hits

    async def search_by_vector(self, vector: Any, limit: int = 20) -> list[SourceChunkHit]:
        return self._hits[:limit]

    async def fetch_by_chunk_ids(self, chunk_ids: Any) -> dict[str, Any]:
        return {}


class _SourceFTS:
    def __init__(self, hits: list[Any]) -> None:
        self._hits = hits

    def query(self, match: str, limit: int = 20) -> list[Any]:
        return self._hits[:limit]

    def active_file_paths(self) -> list[str]:
        return sorted({hit.file_path for hit in self._hits})

    def term_file_evidence(self, terms: Any) -> dict[str, frozenset[str]]:
        matched = frozenset(hit.file_path for hit in self._hits)
        return {term: matched for term in terms}


class _Empty:
    async def search_by_vector(self, vector: Any, limit: int = 10) -> list[Any]:
        return []

    async def search(self, query: str, limit: int = 10) -> list[Any]:
        return []


def _hit(name: str, path: str, score: float) -> SourceChunkHit:
    return SourceChunkHit(
        chunk_id=f"{path}::{name}",
        file_path=path,
        name=name,
        kind="function",
        start_line=1,
        end_line=9,
        is_test=False,
        source="symbol",
        content_hash="h",
        snippet=f"def {name}(): ...",
        score=score,
    )


class _FTSHit:
    def __init__(self, chunk_id: str, file_path: str, score: float = 5.0) -> None:
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.score = score


def _toy_corpus(tmp_path, *, covers_ingress: bool):
    """A tiny repository, as the coordinator would see it.

    ``covers_ingress`` is the only difference between a corpus that answers
    the exported question and one that does not, which is what lets one
    fixture prove it can both fail and pass.
    """

    async def search(query: str, *, limit: int = 5, mode: str = "hybrid") -> dict[str, Any]:
        text = query.casefold()
        if "ingress" in text:
            hits = [_hit("configure_ingress", "deploy/ingress.py", 0.83)] if covers_ingress else []
        elif "retry" in text:
            hits = [_hit("retry_policy", "taskq/retry.py", 0.85)]
        else:
            hits = []
        coordinator = SourceSearchCoordinator(
            repo_path=tmp_path,
            embedder=_Embedder(),
            source_vectors=_SourceVectors(hits),
            source_fts=_SourceFTS([_FTSHit(hit.chunk_id, hit.file_path) for hit in hits]),
            wiki_vectors=_Empty(),
            wiki_fts=_Empty(),
            query_log=QueryLog(tmp_path / "run_log.jsonl"),
        )
        return await coordinator.search(query, limit=limit, mode=mode)

    return search


async def test_a_goal_suite_fails_on_the_corpus_that_produced_it(busy_log, tmp_path):
    """The point of a goal export: red until the gap it names is closed."""
    suite = export_bucket(report_from_path(busy_log), BUCKET_NO_MATCH, intent="goal")
    run = await run_suite(suite, _toy_corpus(tmp_path, covers_ingress=False))
    assert run.failed
    ingress = next(case for case in run.outcomes if "ingress" in case.query)
    assert ingress.outcome == "fail"
    assert ingress.message == "still a no-match"


async def test_the_same_goal_suite_passes_once_the_corpus_covers_it(busy_log, tmp_path):
    """Same fixture, same runner, corpus fixed: the assertion turns green."""
    suite = export_bucket(report_from_path(busy_log), BUCKET_NO_MATCH, intent="goal")
    run = await run_suite(suite, _toy_corpus(tmp_path, covers_ingress=True))
    ingress = next(case for case in run.outcomes if "ingress" in case.query)
    assert ingress.outcome == "pass"
    assert ingress.observed["owner"] == "deploy/ingress.py"


async def test_a_guard_suite_holds_the_floor_it_was_exported_at(busy_log, tmp_path):
    suite = export_bucket(report_from_path(busy_log), BUCKET_NO_MATCH, intent="guard")
    run = await run_suite(suite, _toy_corpus(tmp_path, covers_ingress=False))
    ingress = next(case for case in run.outcomes if "ingress" in case.query)
    assert ingress.outcome == "pass"
    # ...and turns red when a no-match it was holding becomes something else.
    changed = await run_suite(suite, _toy_corpus(tmp_path, covers_ingress=True))
    assert changed.failed


async def test_a_pending_case_is_never_a_failure(busy_log, tmp_path):
    """An unanswered question is not a defect, and must not gate a build."""
    suite = export_bucket(report_from_path(busy_log), BUCKET_WRONG_OWNER)
    run = await run_suite(suite, _toy_corpus(tmp_path, covers_ingress=True))
    assert run.counts()["pending"] == len(suite.cases)
    assert not run.failed


async def test_one_exploding_query_does_not_decide_the_others(busy_log, tmp_path):
    """A raising search is an error outcome, not a crash and not a fail."""
    working = _toy_corpus(tmp_path, covers_ingress=True)

    async def flaky(query: str, *, limit: int = 5, mode: str = "hybrid") -> dict[str, Any]:
        if "licence" in query:
            raise RuntimeError("store went away")
        return await working(query, limit=limit, mode=mode)

    suite = export_bucket(report_from_path(busy_log), BUCKET_NO_MATCH, intent="goal")
    run = await run_suite(suite, flaky)
    tally = run.counts()
    assert tally["error"] == 1
    assert tally["pass"] == 1
    exploded = next(case for case in run.outcomes if "licence" in case.query)
    assert exploded.message == "RuntimeError: store went away"


async def test_an_errored_search_is_not_reported_as_a_ranking_defect(tmp_path):
    """A dead index answers no question about which file owns what."""
    suite_path = tmp_path / "s.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": FIXTURE_KIND,
                "bucket": "wrong_owner",
                "intent": "goal",
                "cases": [
                    {
                        "id": "c-1",
                        "query": "where is the retry policy",
                        "mode": "hybrid",
                        "limit": 5,
                        "expect": {"kind": "owner", "owner": "taskq/retry.py"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    async def broken(query: str, *, limit: int = 5, mode: str = "hybrid") -> dict[str, Any]:
        return {"results": [], "confidence": CAUTION, "status": "error", "selected_owner": None}

    run = await run_suite(load_suite(suite_path), broken)
    assert run.outcomes[0].outcome == "error"
    assert "status=error" in run.outcomes[0].message
