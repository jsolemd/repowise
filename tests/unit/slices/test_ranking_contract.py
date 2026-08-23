"""The ranking contract, in isolation from any database.

Rank decides which members the budget drops, so these are the assertions the
"shuffle the ranking" mutation leg has to break. Each one names a distinct
property: numbering, tiering, ordering, and determinism.
"""

from __future__ import annotations

from repowise.core.slices.models import SliceMember
from repowise.core.slices.ranking import (
    RANK_COMPONENTS,
    rank_members,
    ranking_contract,
    score_components,
)


def _member(node_id: str, **kwargs) -> SliceMember:
    defaults = dict(
        node_type="file",
        layer="file",
        file_path=node_id,
        distance=1,
    )
    defaults.update(kwargs)
    return SliceMember(node_id=node_id, **defaults)


def test_ranks_are_dense_and_start_at_one() -> None:
    members = rank_members([_member(f"f{i}.py", distance=i) for i in range(1, 6)])
    assert [m.rank for m in members] == [1, 2, 3, 4, 5]


def test_seeds_always_precede_reached_members() -> None:
    """Even a seed that scores below every reached member stays first.

    A task that names ``login`` must not have ``login`` fall off the bottom of
    a budgeted response because a central utility two hops away scored higher.
    """
    seed = _member("seed.py", distance=0, is_seed=True, seed_score=0.01)
    strong = _member("hot.py", distance=1, reference_count=8, max_confidence=1.0, pagerank=1.0)
    members = rank_members([strong, seed])

    assert members[0].node_id == "seed.py"
    assert members[0].score < members[1].score


def test_reached_members_are_ordered_by_score_descending() -> None:
    near = _member("near.py", distance=1, reference_count=4, max_confidence=0.95)
    mid = _member("mid.py", distance=2, reference_count=2, max_confidence=0.9)
    far = _member("far.py", distance=4, reference_count=1, max_confidence=0.5)
    members = rank_members([far, near, mid])

    assert [m.node_id for m in members] == ["near.py", "mid.py", "far.py"]
    scores = [m.score for m in members]
    assert scores == sorted(scores, reverse=True)


def test_ties_break_deterministically_on_node_id() -> None:
    """Two calls must never disagree about which member the budget drops."""
    identical = [_member(nid, distance=2, reference_count=1) for nid in ("c.py", "a.py", "b.py")]
    first = [m.node_id for m in rank_members(list(identical))]
    second = [m.node_id for m in rank_members(list(reversed(identical)))]

    assert first == second == ["a.py", "b.py", "c.py"]


def test_closer_beats_further_all_else_equal() -> None:
    close = _member("close.py", distance=1)
    distant = _member("distant.py", distance=5)
    members = rank_members([distant, close])
    assert members[0].node_id == "close.py"


def test_more_references_beat_fewer_at_the_same_distance() -> None:
    many = _member("many.py", distance=2, reference_count=8)
    few = _member("few.py", distance=2, reference_count=1)
    members = rank_members([few, many])
    assert members[0].node_id == "many.py"


def test_stronger_edge_confidence_beats_a_guess() -> None:
    certain = _member("certain.py", distance=2, max_confidence=0.95)
    guess = _member("guess.py", distance=2, max_confidence=0.50)
    members = rank_members([guess, certain])
    assert members[0].node_id == "certain.py"


def test_test_material_is_damped_not_removed() -> None:
    production = _member("prod.py", distance=2, reference_count=2)
    testing = _member("test_prod.py", distance=2, reference_count=2, is_test=True)
    members = rank_members([testing, production])

    assert [m.node_id for m in members] == ["prod.py", "test_prod.py"]
    assert members[1].score > 0.0, "damped, not zeroed — it is still in the slice"


def test_centrality_is_normalised_inside_the_slice() -> None:
    """PageRank is compared against the slice's own maximum, not a constant.

    Absolute PageRank differs by orders of magnitude between a 200-file repo
    and a 20,000-file one; a fixed multiplier would make this component
    dominate on one and vanish on the other.
    """
    small = [
        _member("a.py", distance=1, pagerank=0.0009),
        _member("b.py", distance=1, pagerank=0.0001),
    ]
    large = [
        _member("a.py", distance=1, pagerank=0.9),
        _member("b.py", distance=1, pagerank=0.1),
    ]
    assert [m.score for m in rank_members(small)] == [m.score for m in rank_members(large)]


def test_score_components_are_all_bounded() -> None:
    member = _member(
        "x.py", distance=0, reference_count=999, max_confidence=5.0, pagerank=2.0, query_hits=99
    )
    parts = score_components(member, max_pagerank=1.0)
    assert set(parts) == set(RANK_COMPONENTS)
    assert all(0.0 <= value <= 1.0 for value in parts.values())


def test_ranking_contract_states_the_drop_order() -> None:
    contract = ranking_contract()
    assert contract["tiers"] == ["seed", "reached"]
    assert contract["drop_order"] == "lowest rank first"
    assert set(contract["weights"]) == set(RANK_COMPONENTS)


def test_empty_input_is_empty_output() -> None:
    assert rank_members([]) == []
