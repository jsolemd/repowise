"""Reading the query log back: one JSONL line to one analysable record.

The writer swallows everything so a bad log cannot fail a search
(:mod:`..query_log`). This module is the other half of that bargain: a bad
*line* cannot fail a report either, and — the part that matters — it cannot
disappear quietly. Every line the parser refuses is counted, given a reason,
and kept with its line number, because "3 of 40,000 lines were unreadable" and
"3,000 of 40,000 were" are the same silence if the parser only returns the
lines it liked.

Two failure grades, deliberately distinct:

* **Malformed** — the line cannot be interpreted as a query at all. It is
  excluded from every rate and reported on its own.
* **Field warning** — the line is a query, but one non-deciding field was
  unusable (an unparseable timestamp, a string where a number belonged). The
  record is kept, because discarding a real quality signal over a cosmetic
  defect loses more than it protects, and the warning is reported beside it.

The distinction is drawn at exactly one place: ``query`` and ``confidence``
are what every bucket decision reads, so those two are required and everything
else degrades.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "EXCERPT_CHARS",
    "MalformedLine",
    "ParsedLog",
    "QueryRecord",
    "TopRecord",
    "normalize_query",
    "parse_line",
    "parse_log",
]

#: How much of an unreadable line the report quotes back. Long enough to
#: recognise which query it was, short enough that a log full of corruption
#: cannot turn the report into a copy of the log.
EXCERPT_CHARS = 200

#: Fields whose absence or wrong type makes a line uninterpretable. Keep this
#: set minimal: every name added here turns a previously analysable line into a
#: discarded one, retroactively, for every log already on disk.
_REQUIRED = ("query", "confidence")


def normalize_query(text: str) -> str:
    """The key two occurrences of "the same question" are grouped under.

    Whitespace and case only. Nothing stems, splits, or strips punctuation:
    this key decides whether two rows are allowed to *contradict* each other,
    and a normaliser that merged two genuinely different questions would
    manufacture a contradiction out of nothing.
    """
    return " ".join(text.split()).casefold()


@dataclass(frozen=True, slots=True)
class TopRecord:
    """One ranked result as the log recorded it."""

    file: str | None
    lane: str | None
    dense_cosine: float | None
    lexical_rank: int | None
    exact_name: bool
    fused_score: float | None
    concept_coverage: float | None
    same_path_corroborated: bool | None


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """One logged query, typed, with its provenance kept.

    ``line_number`` is 1-based and is the whole point of keeping it: a report
    that says a query went wrong and cannot say *where in the file* leaves the
    reader grepping a log they already have.
    """

    line_number: int
    query: str
    confidence: str
    ts: datetime | None
    ts_raw: str | None
    mode: str | None
    limit: int | None
    latency_ms: float | None
    result_count: int
    top: tuple[TopRecord, ...]
    selected_owner_file: str | None
    no_match: bool
    status: str
    error_code: str | None
    failed_legs: tuple[dict[str, Any], ...]
    generation: str | None
    warnings: tuple[str, ...] = ()

    @property
    def normalized_query(self) -> str:
        return normalize_query(self.query)

    @property
    def best_dense_cosine(self) -> float | None:
        cosines = [entry.dense_cosine for entry in self.top if entry.dense_cosine is not None]
        return max(cosines) if cosines else None

    @property
    def top_files(self) -> tuple[str, ...]:
        return tuple(entry.file for entry in self.top if entry.file)


@dataclass(frozen=True, slots=True)
class MalformedLine:
    """A line the parser refused, and why.

    ``reason`` is a closed vocabulary and ``detail`` carries the parser's own
    message. They are separate because the report tallies by reason: folding
    the message in would key the tally on a string containing a character
    offset, and a file with three thousand torn lines would produce three
    thousand distinct "reasons" — a copy of the log wearing a histogram's name.
    """

    line_number: int
    reason: str
    excerpt: str
    detail: str | None = None


@dataclass
class ParsedLog:
    """Everything one pass over the file produced, including what it rejected."""

    path: Path | None = None
    records: list[QueryRecord] = field(default_factory=list)
    malformed: list[MalformedLine] = field(default_factory=list)
    blank_lines: int = 0
    lines_read: int = 0
    truncated: bool = False
    read_error: str | None = None

    @property
    def analysed(self) -> int:
        return len(self.records)

    @property
    def warned(self) -> int:
        return sum(1 for record in self.records if record.warnings)


def _excerpt(raw: str) -> str:
    text = raw.strip()
    return text[:EXCERPT_CHARS] + ("…" if len(text) > EXCERPT_CHARS else "")


def _as_float(value: Any, name: str, warnings: list[str]) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int; it is never a score
        warnings.append(f"bad_type:{name}")
        return None
    if isinstance(value, int | float):
        return float(value)
    warnings.append(f"bad_type:{name}")
    return None


def _as_int(value: Any, name: str, warnings: list[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        warnings.append(f"bad_type:{name}")
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    warnings.append(f"bad_type:{name}")
    return None


def _as_str(value: Any, name: str, warnings: list[str]) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    warnings.append(f"bad_type:{name}")
    return None


def _as_bool(value: Any, name: str, warnings: list[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    warnings.append(f"bad_type:{name}")
    return default


def _parse_ts(value: Any, warnings: list[str]) -> tuple[datetime | None, str | None]:
    if value is None:
        warnings.append("missing:ts")
        return None, None
    if not isinstance(value, str):
        warnings.append("bad_type:ts")
        return None, None
    try:
        return datetime.fromisoformat(value), value
    except ValueError:
        # Kept as raw text rather than dropped: the string is still evidence
        # about which producer wrote the line.
        warnings.append("bad_value:ts")
        return None, value


def _parse_top(value: Any, warnings: list[str]) -> tuple[TopRecord, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        warnings.append("bad_type:top")
        return ()
    entries: list[TopRecord] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            warnings.append(f"bad_type:top[{index}]")
            continue
        entries.append(
            TopRecord(
                file=_as_str(item.get("file"), f"top[{index}].file", warnings),
                lane=_as_str(item.get("lane"), f"top[{index}].lane", warnings),
                dense_cosine=_as_float(
                    item.get("dense_cosine"), f"top[{index}].dense_cosine", warnings
                ),
                lexical_rank=_as_int(
                    item.get("lexical_rank"), f"top[{index}].lexical_rank", warnings
                ),
                exact_name=_as_bool(item.get("exact_name"), f"top[{index}].exact_name", warnings),
                fused_score=_as_float(
                    item.get("fused_score"), f"top[{index}].fused_score", warnings
                ),
                concept_coverage=_as_float(
                    item.get("concept_coverage"), f"top[{index}].concept_coverage", warnings
                ),
                same_path_corroborated=(
                    item.get("same_path_corroborated")
                    if isinstance(item.get("same_path_corroborated"), bool)
                    else None
                ),
            )
        )
    return tuple(entries)


def _parse_failed_legs(value: Any, warnings: list[str]) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        warnings.append("bad_type:failed_legs")
        return ()
    legs: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            legs.append(item)
        else:
            warnings.append(f"bad_type:failed_legs[{index}]")
    return tuple(legs)


def parse_line(raw: str, line_number: int) -> QueryRecord | MalformedLine | None:
    """One line to a record, a refusal, or ``None`` for a blank line.

    A blank line is not corruption. The writer appends ``"\\n"`` after every
    record, so a file that ends in one is the normal case and counting it as
    damage would put a permanent false positive in every report.
    """
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        # The realistic corruption is a torn final line from a process that
        # died mid-write. The parser's complaint goes in ``detail``, where it
        # helps a human without fragmenting the tally.
        return MalformedLine(
            line_number, "invalid_json", _excerpt(raw), detail=str(exc.args[0])[:160]
        )
    if not isinstance(payload, dict):
        return MalformedLine(
            line_number, "not_an_object", _excerpt(raw), detail=type(payload).__name__
        )
    for name in _REQUIRED:
        if name not in payload:
            return MalformedLine(line_number, "missing_field", _excerpt(raw), detail=name)
        if not isinstance(payload[name], str):
            return MalformedLine(
                line_number,
                "bad_type",
                _excerpt(raw),
                detail=f"{name} is {type(payload[name]).__name__}",
            )

    warnings: list[str] = []
    ts, ts_raw = _parse_ts(payload.get("ts"), warnings)
    status = _as_str(payload.get("status"), "status", warnings)
    return QueryRecord(
        line_number=line_number,
        query=payload["query"],
        confidence=payload["confidence"],
        ts=ts,
        ts_raw=ts_raw,
        mode=_as_str(payload.get("mode"), "mode", warnings),
        limit=_as_int(payload.get("limit"), "limit", warnings),
        latency_ms=_as_float(payload.get("latency_ms"), "latency_ms", warnings),
        result_count=_as_int(payload.get("result_count"), "result_count", warnings) or 0,
        top=_parse_top(payload.get("top"), warnings),
        selected_owner_file=_as_str(
            payload.get("selected_owner_file"), "selected_owner_file", warnings
        ),
        no_match=_as_bool(payload.get("no_match"), "no_match", warnings),
        # A line written before the field existed is an ``ok`` line: that is
        # what it meant when it was written, and inventing an unknown third
        # state would put every historical row into it.
        status=status or "ok",
        error_code=_as_str(payload.get("error_code"), "error_code", warnings),
        failed_legs=_parse_failed_legs(payload.get("failed_legs"), warnings),
        generation=_as_str(payload.get("generation"), "generation", warnings),
        warnings=tuple(warnings),
    )


def iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Line-number/line pairs, skipping nothing.

    ``errors="replace"`` rather than ``"strict"``: a byte-level corruption
    must arrive at the parser as an unparseable *line*, which the report can
    count, instead of a ``UnicodeDecodeError`` that takes the whole pass down
    at whatever line it happened to reach.
    """
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield from enumerate(handle, start=1)


def parse_log(path: Path | str, *, max_lines: int | None = None) -> ParsedLog:
    """Read *path* into a :class:`ParsedLog`. Never raises.

    A missing file is an empty log with a stated reason, not an exception: the
    common case for "no query log" is a repository where source search has
    simply never run, and that is a fact to report rather than an error to
    raise at a caller who asked a reasonable question.
    """
    log_path = Path(path)
    parsed = ParsedLog(path=log_path)
    try:
        if not log_path.exists():
            parsed.read_error = "log_not_found"
            return parsed
        if not log_path.is_file():
            parsed.read_error = "log_not_a_file"
            return parsed
        for line_number, raw in iter_lines(log_path):
            if max_lines is not None and parsed.lines_read >= max_lines:
                parsed.truncated = True
                break
            parsed.lines_read = line_number
            outcome = parse_line(raw, line_number)
            if outcome is None:
                parsed.blank_lines += 1
            elif isinstance(outcome, MalformedLine):
                parsed.malformed.append(outcome)
            else:
                parsed.records.append(outcome)
    except OSError as exc:
        # Whatever was read before the failure is still true, so it is kept and
        # the truncation is declared. Returning nothing would report a healthy
        # empty log for an unreadable one.
        parsed.read_error = f"{type(exc).__name__}: {exc}"
        parsed.truncated = True
    return parsed
