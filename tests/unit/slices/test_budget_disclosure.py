"""The budget never drops a member quietly.

Every test here is a way the budgeter could cheat: return a short list with a
clean-looking report, drop from the wrong end, forget to count the envelope,
or answer "nothing fits" with an empty success. The mutation leg for the
budget breaks the first of those, and these are what catch it.
"""

from __future__ import annotations

import pytest

from repowise.core.slices.budget import fit_members, payload_tokens
from repowise.core.slices.errors import BudgetTooSmallError
from repowise.core.slices.models import SliceMember
from repowise.core.slices.ranking import rank_members


def _priced(count: int, *, filler: int = 40) -> list[tuple[SliceMember, dict]]:
    members = rank_members(
        [
            SliceMember(
                node_id=f"pkg/file{i:02d}.py",
                node_type="file",
                layer="file",
                file_path=f"pkg/file{i:02d}.py",
                distance=1 + (i % 4),
                reference_count=(count - i),
            )
            for i in range(count)
        ]
    )
    return [(m, {**m.identity(), "why": ["x" * filler]}) for m in members]


def test_everything_fits_reports_no_truncation() -> None:
    priced = _priced(5)
    kept, report = fit_members(priced, view="card", budget_tokens=100_000, envelope_tokens=50)

    assert len(kept) == 5
    assert report.truncated is False
    assert report.dropped == []
    assert report.included_members == report.total_members == 5
    assert "All 5 member(s) fit" in report.disclosure()


def test_over_budget_drops_from_the_bottom_of_the_ranking() -> None:
    priced = _priced(20)
    per_member = payload_tokens(priced[0][1])
    budget = 100 + per_member * 5

    kept, report = fit_members(priced, view="card", budget_tokens=budget, envelope_tokens=100)

    assert report.truncated is True
    assert 0 < len(kept) < 20
    assert report.included_members == len(kept)
    assert report.total_members == 20
    assert len(report.dropped) == 20 - len(kept)

    kept_ranks = [payload["rank"] for payload in kept]
    dropped_ranks = [d.rank for d in report.dropped]
    assert kept_ranks == sorted(kept_ranks)
    assert max(kept_ranks) < min(dropped_ranks)
    assert sorted(kept_ranks + dropped_ranks) == list(range(1, 21))


def test_the_disclosure_names_the_range_and_the_recovery() -> None:
    priced = _priced(20)
    per_member = payload_tokens(priced[0][1])
    _kept, report = fit_members(
        priced, view="skeleton", budget_tokens=100 + per_member * 4, envelope_tokens=100
    )
    sentence = report.disclosure()

    assert "were dropped" in sentence
    assert "budget.dropped" in sentence
    assert "budget_tokens" in sentence
    assert str(report.total_members) in sentence
    assert f"rank {min(d.rank for d in report.dropped)}" in sentence


def test_every_dropped_member_is_individually_accounted_for() -> None:
    priced = _priced(15)
    per_member = payload_tokens(priced[0][1])
    _kept, report = fit_members(
        priced, view="card", budget_tokens=100 + per_member * 3, envelope_tokens=100
    )
    payload = report.to_dict()

    assert payload["dropped_members"] == len(payload["dropped"]) == len(report.dropped)
    assert payload["dropped_members"] > 0
    assert "dropped_list_truncated" not in payload
    for entry in payload["dropped"]:
        assert entry["node"].startswith("pkg/file")
        assert entry["tokens"] > 0
        assert entry["rank"] >= 1


def test_a_huge_drop_list_is_itself_bounded_and_the_bound_is_disclosed() -> None:
    """The honesty list must not become the thing that blows the budget.

    The count and the rank span stay exact; only the per-member itemisation is
    cut, and the cut is stated. The complete list still reaches the caller via
    the ``on_budget`` hook, which is what the MCP layer feeds to the omission
    store.
    """
    priced = _priced(300)
    per_member = payload_tokens(priced[0][1])
    _kept, report = fit_members(
        priced, view="card", budget_tokens=200 + per_member * 3, envelope_tokens=200
    )
    payload = report.to_dict()

    assert payload["dropped_members"] > 100
    assert len(payload["dropped"]) == 25
    assert payload["dropped_list_truncated"] == payload["dropped_members"] - 25
    assert str(payload["dropped_members"]) in payload["disclosure"]
    assert "highest-ranked" in payload["disclosure"]
    assert len(report.dropped) == payload["dropped_members"], (
        "the report object keeps every drop even when the response itemises 25"
    )


def test_the_report_accounts_for_the_envelope_too() -> None:
    priced = _priced(6)
    _kept, report = fit_members(priced, view="card", budget_tokens=100_000, envelope_tokens=777)
    assert report.envelope_tokens == 777
    assert report.used_tokens == 777 + report.members_tokens


def test_a_budget_that_holds_nothing_raises_rather_than_returning_empty() -> None:
    """An empty member list would read as 'this task needs no code'."""
    priced = _priced(10)
    with pytest.raises(BudgetTooSmallError) as excinfo:
        fit_members(priced, view="full", budget_tokens=120, envelope_tokens=100)

    details = excinfo.value.details()
    assert details["total_members"] == 10
    assert details["minimum_budget_tokens"] > 120
    assert details["top_member_tokens"] > 0


def test_a_genuinely_empty_slice_is_a_success_not_an_error() -> None:
    """Zero members is a legal build result and must not be confused with a failure."""
    kept, report = fit_members([], view="card", budget_tokens=5000, envelope_tokens=100)
    assert kept == []
    assert report.total_members == 0
    assert report.truncated is False


def test_at_least_one_member_survives_whenever_one_can() -> None:
    priced = _priced(200)
    per_member = payload_tokens(priced[0][1])
    kept, report = fit_members(
        priced,
        view="card",
        budget_tokens=100 + per_member + 60,
        envelope_tokens=100,
    )
    assert len(kept) >= 1
    assert report.truncated is True
    assert len(report.dropped) == 200 - len(kept)
