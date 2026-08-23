"""Why each member is in the slice, and in what order.

Rank is not decoration. It is the order the budget spends itself in: when a
slice does not fit, members drop from the bottom, so a wrong ranking silently
becomes a wrong slice. Two properties hold it honest.

**Tiers before scores.** Seeds always precede reached members, because a task
that names ``login`` must not have ``login`` fall off the end because some
central utility scored higher. Within a tier the order is the score.

**Every component is bounded and named.** The score is a weighted sum of five
normalised signals, each in ``[0, 1]``, so a member's ``score`` and its
``reasons`` describe the same thing and neither can drift from the other:

``proximity``
    ``1 / (1 + distance)``. The dominant term — the further a node is from
    where the task starts, the less likely the task touches it.
``references``
    Distinct persisted graph relations into the discovering frontier,
    saturating at ``_REF_SATURATION``. A node reached by six different call
    sites in the slice is load-bearing for it; a node reached once may be
    incidental.
``confidence``
    The strongest edge that brought it in. A ``same_file`` call is a
    certainty at 0.95 and a ``global_unique`` guess is 0.50
    (:mod:`repowise.core.ingestion.models` carries the whole vocabulary), and
    a slice ordered without that treats both as the same evidence.
``centrality``
    PageRank, normalised **within the slice** rather than against a global
    constant. Absolute PageRank values differ by orders of magnitude between a
    200-file repo and a 20,000-file one; a fixed multiplier would make this
    term dominate on one and vanish on the other.
``query``
    Whether the task's own words name this member. Saturates at three terms.

Test material keeps its place in the ordering but is damped, so a slice that
includes tests still reads as production code first.
"""

from __future__ import annotations

from repowise.core.slices.models import SliceMember

_W_PROXIMITY = 0.40
_W_REFERENCES = 0.20
_W_CONFIDENCE = 0.15
_W_CENTRALITY = 0.15
_W_QUERY = 0.10

_REF_SATURATION = 8
_QUERY_SATURATION = 3
_TEST_DAMPING = 0.6

#: Named so a response can state the ordering contract rather than imply it.
RANK_COMPONENTS: tuple[str, ...] = (
    "proximity",
    "references",
    "confidence",
    "centrality",
    "query",
)


def score_components(member: SliceMember, *, max_pagerank: float) -> dict[str, float]:
    """The five normalised signals behind one member's score."""
    centrality = (member.pagerank / max_pagerank) if max_pagerank > 0.0 else 0.0
    return {
        "proximity": 1.0 / (1.0 + max(0, member.distance)),
        "references": min(member.reference_count, _REF_SATURATION) / _REF_SATURATION,
        "confidence": max(0.0, min(member.max_confidence, 1.0)),
        "centrality": max(0.0, min(centrality, 1.0)),
        "query": min(member.query_hits, _QUERY_SATURATION) / _QUERY_SATURATION,
    }


def _score(member: SliceMember, *, max_pagerank: float) -> float:
    parts = score_components(member, max_pagerank=max_pagerank)
    value = (
        _W_PROXIMITY * parts["proximity"]
        + _W_REFERENCES * parts["references"]
        + _W_CONFIDENCE * parts["confidence"]
        + _W_CENTRALITY * parts["centrality"]
        + _W_QUERY * parts["query"]
    )
    if member.is_test:
        value *= _TEST_DAMPING
    return value


def rank_members(members: list[SliceMember]) -> list[SliceMember]:
    """Score, sort and number every member. Returns the sorted list.

    Sort key is ``(tier, -score, distance, node_id)``: seeds first, then score
    descending, ties broken deterministically so two calls on one slice can
    never disagree about which member the budget drops.
    """
    if not members:
        return []

    max_pagerank = max((m.pagerank for m in members), default=0.0)
    for member in members:
        member.score = (
            member.seed_score if member.is_seed else _score(member, max_pagerank=max_pagerank)
        )

    ordered = sorted(
        members,
        key=lambda m: (0 if m.is_seed else 1, -m.score, m.distance, m.node_id),
    )
    for position, member in enumerate(ordered, start=1):
        member.rank = position
    return ordered


def ranking_contract() -> dict[str, object]:
    """The ordering rule, stated on every response that carries a ranking."""
    return {
        "tiers": ["seed", "reached"],
        "components": list(RANK_COMPONENTS),
        "weights": {
            "proximity": _W_PROXIMITY,
            "references": _W_REFERENCES,
            "confidence": _W_CONFIDENCE,
            "centrality": _W_CENTRALITY,
            "query": _W_QUERY,
        },
        "tie_break": ["score", "distance", "node_id"],
        "drop_order": "lowest rank first",
    }
