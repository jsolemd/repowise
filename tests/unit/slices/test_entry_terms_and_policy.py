"""Task-query tokenisation, policy clamping, and view normalisation.

All pure — no database — so the behaviour that decides *where a slice starts*
and *how far it goes* is checkable without an index.
"""

from __future__ import annotations

from repowise.core.slices.entry_points import count_query_hits, query_terms
from repowise.core.slices.models import VIEWS, WalkPolicy
from repowise.core.slices.views import normalize_view


def test_stop_words_are_dropped_and_domain_nouns_are_not() -> None:
    terms = query_terms("fix the login token so that it can refresh")
    assert "login" in terms
    assert "token" in terms
    assert "refresh" in terms
    assert "the" not in terms
    assert "fix" not in terms
    assert "can" not in terms


def test_identifiers_are_split_on_case_and_underscore() -> None:
    terms = query_terms("refreshToken and issue_session_token")
    assert "refreshtoken" in terms
    assert "refresh" in terms
    assert "issue_session_token" in terms
    assert "session" in terms


def test_short_fragments_are_ignored() -> None:
    assert "id" not in query_terms("update the id column")
    assert "column" in query_terms("update the id column")


def test_terms_are_deduplicated_and_ordered_longest_first() -> None:
    terms = query_terms("token TOKEN tokenizer")
    assert terms == sorted(set(terms), key=lambda t: (-len(t), t))
    assert terms.count("token") == 1


def test_an_empty_task_yields_no_terms() -> None:
    assert query_terms("") == []
    assert query_terms("   ") == []


def test_query_hits_counts_distinct_terms_across_fields() -> None:
    terms = ["login", "token", "missing"]
    assert count_query_hits(terms, "login", "src/token/store.py") == 2
    assert count_query_hits(terms, None, None) == 0
    assert count_query_hits([], "login") == 0


def test_policy_clamps_every_knob_into_range() -> None:
    clamped = WalkPolicy(
        downstream_depth=99,
        upstream_depth=-4,
        seed_symbol_fanout=9999,
        max_members=0,
        min_edge_confidence=7.5,
    ).clamped()

    assert clamped.downstream_depth == WalkPolicy.MAX_DEPTH
    assert clamped.upstream_depth == 0
    assert clamped.seed_symbol_fanout == 50
    assert clamped.max_members == 1
    assert clamped.min_edge_confidence == 1.0


def test_policy_round_trips_through_its_dict_form() -> None:
    policy = WalkPolicy(downstream_depth=3, upstream_depth=2, include_tests=True, max_members=99)
    assert WalkPolicy.from_dict(policy.to_dict()) == policy


def test_policy_defaults_are_asymmetric() -> None:
    """Deep in the direction the seed reaches, shallow in the one that reaches it."""
    policy = WalkPolicy()
    assert policy.downstream_depth > policy.upstream_depth


def test_unknown_views_fall_back_to_the_middle_fidelity() -> None:
    assert normalize_view(None) == "skeleton"
    assert normalize_view("") == "skeleton"
    assert normalize_view("nonsense") == "skeleton"
    for view in VIEWS:
        assert normalize_view(view.upper()) == view
