"""The query-quality trail: what it records, and what it refuses to break."""

from __future__ import annotations

import json

from repowise.core.source_search.query_log import (
    FAILED_LEGS_LOGGED,
    QUERY_LOG_FILENAME,
    TOP_EVENTS_LOGGED,
    QueryEvent,
    QueryLog,
    TopEntry,
    default_query_log_path,
)


def _event(**overrides) -> QueryEvent:
    fields: dict = {
        "query": "how retrieval works",
        "mode": "hybrid",
        "limit": 5,
        "latency_ms": 12.5,
        "confidence": "confident",
        "result_count": 2,
        "top": [
            TopEntry(
                file="src/a.py",
                lane="source",
                dense_cosine=0.51,
                lexical_rank=3,
                exact_name=False,
                fused_score=0.0134,
            )
        ],
        "selected_owner_file": "src/a.py",
        "no_match": False,
    }
    fields.update(overrides)
    return QueryEvent(**fields)


def test_the_path_sits_beside_the_other_source_search_sidecars(tmp_path):
    path = default_query_log_path(tmp_path)
    assert path.name == QUERY_LOG_FILENAME
    assert path.parent == tmp_path / ".repowise" / "source_search"
    assert not path.exists()


def test_one_event_is_one_line(tmp_path):
    log = QueryLog(tmp_path / "nested" / "query_log.jsonl")
    assert log.append(_event()) is True
    assert log.append(_event(query="second")) is True

    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["query"] for line in lines] == [
        "how retrieval works",
        "second",
    ]


def test_the_record_carries_the_evidence_not_the_prose(tmp_path):
    log = QueryLog(tmp_path / "query_log.jsonl")
    log.append(_event())
    record = json.loads(log.path.read_text(encoding="utf-8"))

    assert set(record) == {
        "ts",
        "query",
        "mode",
        "limit",
        "latency_ms",
        "confidence",
        "result_count",
        "top",
        "selected_owner_file",
        "selected_owner_evidence",
        "no_match",
        "status",
        "error_code",
        "failed_legs",
        "generation",
    }
    assert record["top"][0] == {
        "file": "src/a.py",
        "lane": "source",
        "dense_cosine": 0.51,
        "lexical_rank": 3,
        "exact_name": False,
        "fused_score": 0.0134,
        "concept_coverage": None,
        "same_path_corroborated": None,
    }
    assert record["ts"].startswith("20")


def test_the_top_list_is_bounded(tmp_path):
    log = QueryLog(tmp_path / "query_log.jsonl")
    entries = [
        TopEntry(
            file=f"src/{i}.py",
            lane="source",
            dense_cosine=0.5,
            lexical_rank=None,
            exact_name=False,
            fused_score=0.01,
        )
        for i in range(TOP_EVENTS_LOGGED + 5)
    ]
    log.append(_event(top=entries))
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert len(record["top"]) == TOP_EVENTS_LOGGED


def test_a_no_match_records_itself_as_one(tmp_path):
    log = QueryLog(tmp_path / "query_log.jsonl")
    log.append(
        _event(
            confidence="no_match", result_count=0, top=[], selected_owner_file=None, no_match=True
        )
    )
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["no_match"] is True
    assert record["selected_owner_file"] is None
    assert record["top"] == []


def test_an_unwritable_log_reports_failure_rather_than_raising(tmp_path):
    """The only thing this must never do is take a good search down with it."""
    blocked = tmp_path / "query_log.jsonl"
    blocked.mkdir()
    assert QueryLog(blocked).append(_event()) is False


def test_a_caller_that_knows_nothing_of_the_new_fields_writes_the_old_meaning(tmp_path):
    """Additive means additive: the defaults reproduce the previous record.

    Every field added since the first version has to default to the value the
    old behaviour implied, because lines written under it stay in the file
    forever and the reporter reads them alongside today's.
    """
    log = QueryLog(tmp_path / "query_log.jsonl")
    log.append(_event())
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert record["error_code"] is None
    assert record["failed_legs"] == []
    assert record["generation"] is None


def test_the_failed_leg_list_is_bounded_like_the_top_list(tmp_path):
    log = QueryLog(tmp_path / "query_log.jsonl")
    log.append(
        _event(
            failed_legs=[
                {"leg": f"leg {i}", "error": "E", "detail": "d"}
                for i in range(FAILED_LEGS_LOGGED + 4)
            ]
        )
    )
    record = json.loads(log.path.read_text(encoding="utf-8"))
    assert len(record["failed_legs"]) == FAILED_LEGS_LOGGED


def test_an_unserialisable_event_is_swallowed_too(tmp_path):
    """Bad data in the record is still not a reason to fail the search."""

    class _Unserialisable:
        pass

    log = QueryLog(tmp_path / "query_log.jsonl")
    assert log.append(_event(query=_Unserialisable())) is False
    assert not log.path.exists()
