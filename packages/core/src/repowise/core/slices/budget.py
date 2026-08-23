"""Fitting a slice into a caller's token budget, out loud.

The whole contract is one sentence: **a slice never returns fewer members
than it has without saying so.** Everything below exists to make that
non-negotiable rather than best-effort.

* Members are priced individually, at the requested view, in rank order.
  Pricing is the same 4-chars-per-token scale every repowise subsystem
  reports on (:func:`repowise.core.distill.budget.estimate_tokens`), applied
  to the compact JSON the transport actually serialises — structural overhead
  (quotes, braces, field names) is a third of a small member's cost and
  pricing only the prose would undercount it.
* The response *envelope* — summary, seeds, ranking contract, the budget
  block itself — is priced first and reserved. A budgeter that prices only
  the list it trims will overshoot by whatever the rest of the response
  costs.
* Drops happen from the bottom of the ranking, and every dropped member is
  named in :attr:`BudgetReport.dropped` with its rank and price, alongside a
  plain sentence saying what happened and how to get the rest.
* If not even the top-ranked member fits, that is
  :class:`~repowise.core.slices.errors.BudgetTooSmallError`, not a response
  with an empty list. An agent reading ``members: []`` concludes the task
  needs nothing; that is the single most expensive lie this service could
  tell.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from repowise.core.distill.budget import estimate_tokens
from repowise.core.slices.errors import BudgetTooSmallError
from repowise.core.slices.models import SliceMember

#: Slack held back from the caller's budget for the ``budget`` block itself,
#: which cannot be priced until after the fit is decided. Small and fixed: the
#: block is a handful of integers plus one sentence plus the dropped list, and
#: the dropped list is priced explicitly below.
_DISCLOSURE_RESERVE_TOKENS = 60

#: Per-dropped-member cost of the disclosure entry. Reserved as each drop is
#: recorded, so a slice with two hundred drops cannot overflow the budget with
#: its own honesty.
_DROP_ENTRY_TOKENS = 12

#: How many drops are itemised in the response. The *count* and the *rank span*
#: are always exact; only the per-member list is cut, because a 400-member
#: slice can drop 380 and an unbounded honesty list would then be the thing
#: that blows the budget. The full list still reaches the omission store, so
#: nothing is lost — see ``tool_slices._disclose_drops``.
_DROPPED_LIST_CAP = 25


def payload_tokens(obj: Any) -> int:
    """Price an arbitrary JSON-serialisable payload on the shared scale."""
    return estimate_tokens(json.dumps(obj, separators=(",", ":"), default=str))


@dataclass(frozen=True)
class DroppedMember:
    """One member the budget could not hold."""

    node_id: str
    node_type: str
    rank: int
    score: float
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node_id,
            "node_type": self.node_type,
            "rank": self.rank,
            "score": round(self.score, 4),
            "tokens": self.tokens,
        }


@dataclass
class BudgetReport:
    """What the budget did, in enough detail to argue with."""

    view: str
    budget_tokens: int
    envelope_tokens: int
    members_tokens: int
    total_members: int
    included_members: int
    dropped: list[DroppedMember] = field(default_factory=list)
    truncated: bool = False

    @property
    def used_tokens(self) -> int:
        return self.envelope_tokens + self.members_tokens

    def disclosure(self) -> str:
        """The sentence a reader sees when members were dropped."""
        if not self.truncated:
            return (
                f"All {self.total_members} member(s) fit at view={self.view!r} "
                f"({self.used_tokens} of {self.budget_tokens} tokens)."
            )
        ranks = [d.rank for d in self.dropped]
        span = f"{min(ranks)}-{max(ranks)}" if len(ranks) > 1 else str(ranks[0])
        listed = min(len(self.dropped), _DROPPED_LIST_CAP)
        which = (
            f"the {listed} highest-ranked of them are listed under budget.dropped"
            if listed < len(self.dropped)
            else "and are listed under budget.dropped"
        )
        return (
            f"Token budget {self.budget_tokens} held {self.included_members} of "
            f"{self.total_members} member(s) at view={self.view!r}; "
            f"{len(self.dropped)} lower-ranked member(s) (rank {span}) were dropped "
            f"— {which}. Re-fetch with a larger budget_tokens, or view='card', "
            f"to see them."
        )

    def to_dict(self) -> dict[str, Any]:
        listed = self.dropped[:_DROPPED_LIST_CAP]
        payload: dict[str, Any] = {
            "view": self.view,
            "budget_tokens": self.budget_tokens,
            "envelope_tokens": self.envelope_tokens,
            "members_tokens": self.members_tokens,
            "used_tokens": self.used_tokens,
            "total_members": self.total_members,
            "included_members": self.included_members,
            "dropped_members": len(self.dropped),
            "truncated": self.truncated,
            "dropped": [d.to_dict() for d in listed],
            "disclosure": self.disclosure(),
        }
        if len(listed) < len(self.dropped):
            payload["dropped_list_truncated"] = len(self.dropped) - len(listed)
        return payload


def fit_members(
    priced: list[tuple[SliceMember, dict[str, Any]]],
    *,
    view: str,
    budget_tokens: int,
    envelope_tokens: int,
) -> tuple[list[dict[str, Any]], BudgetReport]:
    """Keep members in rank order while they fit; disclose everything else.

    ``priced`` must already be in rank order — the budget spends itself in
    that order, which is the only thing that makes "dropped by rank" true.

    Raises :class:`BudgetTooSmallError` when the envelope alone, or the
    envelope plus the highest-ranked member, exceeds the budget.
    """
    total = len(priced)
    report = BudgetReport(
        view=view,
        budget_tokens=budget_tokens,
        envelope_tokens=envelope_tokens,
        members_tokens=0,
        total_members=total,
        included_members=0,
    )
    if total == 0:
        return [], report

    ceiling = budget_tokens - envelope_tokens - _DISCLOSURE_RESERVE_TOKENS
    costs = [payload_tokens(payload) for _, payload in priced]

    if ceiling < costs[0]:
        raise BudgetTooSmallError(
            budget_tokens=budget_tokens,
            envelope_tokens=envelope_tokens + _DISCLOSURE_RESERVE_TOKENS,
            smallest_member_tokens=costs[0],
            view=view,
            total_members=total,
        )

    kept: list[dict[str, Any]] = []
    spent = 0
    cutoff = total
    for index, ((_member, payload), cost) in enumerate(zip(priced, costs, strict=True)):
        remaining = total - index - 1
        # Reserve room for the disclosure entries this cut would need. Without
        # it a slice can spend its last tokens on one more member and then have
        # nowhere to say it dropped the other ninety. Index 0 is exempt: the
        # guard above already proved it fits, and dropping every member is the
        # one outcome this module refuses to produce quietly.
        reserve = _DROP_ENTRY_TOKENS * min(remaining, 25) if remaining else 0
        if index > 0 and spent + cost + reserve > ceiling:
            cutoff = index
            break
        kept.append(payload)
        spent += cost

    if cutoff < total:
        report.truncated = True
        report.dropped = [
            DroppedMember(
                node_id=member.node_id,
                node_type=member.node_type,
                rank=member.rank,
                score=member.score,
                tokens=cost,
            )
            for (member, _), cost in zip(priced[cutoff:], costs[cutoff:], strict=True)
        ]

    report.members_tokens = spent
    report.included_members = len(kept)
    return kept, report
