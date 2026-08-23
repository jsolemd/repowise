"""The four quality buckets, and what each one is actually claiming.

Three of them read one record. The fourth cannot: "wrong owner" is not a
property a single log line can carry, because at the moment a line is written
nothing knows the answer was wrong. It has to come from a *later* signal, and
this module defines exactly two:

1. **Self-contradiction.** Within one corpus generation, two answers to the
   same question named different owners, and at least one of them was served
   as ``confident``. Confidence is an assertion — "this file is the answer" —
   so a differing answer to the same question over the same corpus means at
   most one of them was right. The generation scope is what separates a
   defect from a reindex: if the corpus moved between the two answers, the
   owner is *allowed* to move with it, and calling that a defect would charge
   retrieval for the repository's own history.

2. **Verdict.** An external file states the correct owner for a query and the
   served owner differs. This is ground truth rather than inference, and is
   reported under its own basis so the two are never added together as if
   they carried the same weight.

The confidence gate on (1) is load-bearing. Without it, two ``caution`` rows
disagreeing would count as a wrong owner — but a cautious answer that changes
is retrieval being appropriately unsure, which is precisely what the
low-confidence bucket already measures. Counting it twice would inflate the
most serious bucket with the least serious evidence.

Error rows never join a conflict group. An error's ``None`` owner means "did
not read the corpus", not "read it and found nothing", and treating a search
that never ran as a rival answer would manufacture contradictions out of
outages.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .records import QueryRecord, normalize_query

__all__ = [
    "BUCKETS",
    "BUCKET_DEFINITIONS",
    "BUCKET_ERROR",
    "BUCKET_HEALTHY",
    "BUCKET_LOW_CONFIDENCE",
    "BUCKET_NO_MATCH",
    "BUCKET_ORDER",
    "BUCKET_WRONG_OWNER",
    "CONFIDENCE_CAUTION",
    "CONFIDENCE_CONFIDENT",
    "CONFIDENCE_NO_MATCH",
    "Classification",
    "OwnerConflict",
    "classify",
    "classify_all",
    "detect_owner_conflicts",
    "error_basis",
    "iter_bucket",
]

# The coordinator's wire values for confidence, restated rather than imported.
# Importing ``..coordinator`` here would pull LanceDB and the whole retrieval
# stack into a reporting pass that reads a text file. The cost of restating
# them is drift, and ``test_query_report.py`` pays it down with an assertion
# that these three equal the coordinator's constants — so a rename there fails
# a test instead of silently reclassifying every row in every report.
CONFIDENCE_CONFIDENT = "confident"
CONFIDENCE_CAUTION = "caution"
CONFIDENCE_NO_MATCH = "no_match"

BUCKET_ERROR = "error"
BUCKET_WRONG_OWNER = "wrong_owner"
BUCKET_NO_MATCH = "no_match"
BUCKET_LOW_CONFIDENCE = "low_confidence"
BUCKET_HEALTHY = "healthy"

#: The four quality buckets, worst first. ``healthy`` is not one of them — it
#: is the remainder, and is reported so the four rates can be read against
#: something rather than floating free.
BUCKETS = (BUCKET_ERROR, BUCKET_WRONG_OWNER, BUCKET_NO_MATCH, BUCKET_LOW_CONFIDENCE)

#: Precedence for the single bucket a row is *counted* under, so the rates sum
#: to one and no row is double-counted. Every bucket a row also matched is
#: kept beside it, so precedence hides nothing.
BUCKET_ORDER = (*BUCKETS, BUCKET_HEALTHY)

BUCKET_DEFINITIONS: dict[str, str] = {
    BUCKET_ERROR: (
        "The search reached no corpus, so its empty result set describes the retrieval "
        "stack and not the repository. Declared by the record's status field; on records "
        "written before that field existed, inferred from the signature only this case "
        "produces (caution, zero results, no owner, no_match false)."
    ),
    BUCKET_WRONG_OWNER: (
        "A later signal contradicts the owner this answer served. Either the same query "
        "over the same corpus generation was answered with a different owner and at least "
        "one of those answers was confident, or an external verdict names a different owner."
    ),
    BUCKET_NO_MATCH: (
        "The search read the corpus and declared it has no answer. Counted as a quality "
        "event because a question the corpus keeps failing is a coverage gap, not an error."
    ),
    BUCKET_LOW_CONFIDENCE: (
        "An answer was served with the caution flag: the evidence supported a candidate "
        "list but not an ownership claim."
    ),
    BUCKET_HEALTHY: (
        "A confident answer, with no later signal against it. The remainder, reported so "
        "the four quality rates can be read against a denominator."
    ),
}


@dataclass(frozen=True, slots=True)
class OwnerConflict:
    """A group of rows that disagree about who owns the answer."""

    normalized_query: str
    generation: str | None
    #: Every distinct owner served for this question, ``None`` included. Order
    #: is by first appearance, so the reader sees which answer came first.
    owners: tuple[str | None, ...]
    line_numbers: tuple[int, ...]
    #: ``self_contradiction`` or ``verdict``.
    basis: str
    #: The owner an external verdict asserts. ``None`` for self-contradiction,
    #: where the whole point is that the right answer is not known.
    expected_owner: str | None = None
    #: Whether the corpus generation was recorded. When it was not, the group
    #: could not be scoped to one corpus and a reindex may explain it.
    generation_known: bool = True


@dataclass(frozen=True, slots=True)
class Classification:
    """Which bucket a row is counted under, and everything it also matched."""

    bucket: str
    matched: tuple[str, ...]
    error_basis: str | None = None
    conflict: OwnerConflict | None = None
    degraded: bool = False


def error_basis(record: QueryRecord) -> str | None:
    """Why this row is an error, or ``None`` if it is not one.

    The inferred arm is a coupling to coordinator behaviour and is named as
    one. At the commit this was written against, ``_classify`` returns
    ``no_match`` whenever the result window is empty, so the only ways to
    reach caution with an empty window are the error envelope and the pin that
    a hard leg failure applies. Both mean the corpus was not fully read, which
    is what this bucket is for. If the coordinator ever serves a plain empty
    caution for some third reason, this arm starts over-reporting — which is
    why every inferred row is labelled, counted apart from declared ones, and
    called out in the report's caveats.
    """
    if record.status == "error":
        return "declared"
    if (
        record.confidence == CONFIDENCE_CAUTION
        and record.result_count == 0
        and not record.no_match
        and record.selected_owner_file is None
        and not record.top
    ):
        return "inferred"
    return None


def _is_no_match(record: QueryRecord) -> bool:
    # Either spelling counts. The boolean is the producer's own summary and the
    # confidence string is the value the response carried; a record where they
    # disagree is still telling us a no-match happened.
    return record.no_match or record.confidence == CONFIDENCE_NO_MATCH


def detect_owner_conflicts(
    records: Sequence[QueryRecord],
    *,
    verdicts: Mapping[str, str | None] | None = None,
) -> dict[int, OwnerConflict]:
    """Map line number to the conflict that row participates in.

    Verdicts win where both apply: ground truth about the right owner is
    strictly better evidence than two answers disagreeing, and reporting a row
    under the weaker basis when the stronger one is available would understate
    what is known about it.
    """
    conflicts: dict[int, OwnerConflict] = {}
    eligible = [record for record in records if error_basis(record) is None]

    normalized_verdicts: dict[str, str | None] = {}
    if verdicts:
        # Not scoped to a generation, deliberately. A verdict is a statement
        # about the question, not about one build of the corpus, and a
        # generation changes on every reindex — scoping it would make a
        # verdict apply to almost nothing.
        normalized_verdicts = {normalize_query(key): value for key, value in verdicts.items()}
        for record in eligible:
            key = record.normalized_query
            if key not in normalized_verdicts:
                continue
            expected = normalized_verdicts[key]
            if record.selected_owner_file == expected:
                continue
            conflicts[record.line_number] = OwnerConflict(
                normalized_query=key,
                generation=record.generation,
                owners=(record.selected_owner_file,),
                line_numbers=(record.line_number,),
                basis="verdict",
                expected_owner=expected,
                generation_known=record.generation is not None,
            )

    groups: dict[tuple[str, str | None], list[QueryRecord]] = defaultdict(list)
    for record in eligible:
        # A question with a verdict is settled: the rows that match it are
        # right and the rows that do not are already charged above. Letting
        # the self-contradiction pass run over it too would charge the
        # *correct* row as well, purely for having been disagreed with.
        if record.normalized_query in normalized_verdicts:
            continue
        groups[(record.normalized_query, record.generation)].append(record)

    for (key, generation), members in groups.items():
        if len(members) < 2:
            continue
        owners: list[str | None] = []
        for record in members:
            if record.selected_owner_file not in owners:
                owners.append(record.selected_owner_file)
        if len(owners) < 2:
            continue
        if not any(record.confidence == CONFIDENCE_CONFIDENT for record in members):
            # Answers that never claimed to be sure are allowed to differ.
            continue
        conflict = OwnerConflict(
            normalized_query=key,
            generation=generation,
            owners=tuple(owners),
            line_numbers=tuple(record.line_number for record in members),
            basis="self_contradiction",
            generation_known=generation is not None,
        )
        for record in members:
            conflicts.setdefault(record.line_number, conflict)
    return conflicts


def classify(
    record: QueryRecord,
    conflict: OwnerConflict | None = None,
) -> Classification:
    """Which bucket *record* counts under, given any conflict it is part of."""
    matched: list[str] = []
    basis = error_basis(record)
    if basis is not None:
        matched.append(BUCKET_ERROR)
    if conflict is not None:
        matched.append(BUCKET_WRONG_OWNER)
    if _is_no_match(record):
        matched.append(BUCKET_NO_MATCH)
    if record.confidence == CONFIDENCE_CAUTION:
        matched.append(BUCKET_LOW_CONFIDENCE)

    bucket = next((name for name in BUCKET_ORDER if name in matched), BUCKET_HEALTHY)
    return Classification(
        bucket=bucket,
        matched=tuple(matched),
        error_basis=basis,
        conflict=conflict,
        # Legs died but an answer was still served. Not an error — the corpus
        # was read, just not all of it — and tracked separately so a partially
        # blind stack is visible without being counted as an outage.
        degraded=bool(record.failed_legs) and basis is None,
    )


def classify_all(
    records: Sequence[QueryRecord],
    *,
    verdicts: Mapping[str, str | None] | None = None,
) -> dict[int, Classification]:
    """Classify every record in one pass, conflicts resolved first."""
    conflicts = detect_owner_conflicts(records, verdicts=verdicts)
    return {
        record.line_number: classify(record, conflicts.get(record.line_number))
        for record in records
    }


def iter_bucket(
    records: Iterable[QueryRecord],
    classifications: Mapping[int, Classification],
    bucket: str,
) -> list[QueryRecord]:
    """Records counted under *bucket*, in log order."""
    return [
        record
        for record in records
        if classifications.get(record.line_number, Classification(BUCKET_HEALTHY, ())).bucket
        == bucket
    ]
