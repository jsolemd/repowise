"""Pair base-side and head-side findings, and say what changed between them."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..health import HealthFindingData
from ..health.aggregation import finding_raw_deduction
from .identity import finding_key, line_distance, normalize_path, severity_rank
from .models import ChangeKind, FindingKey

if TYPE_CHECKING:
    from .sources import FileChange

#: Health-impact movement below this is rounding, not a regression.
_IMPACT_EPSILON = 0.01


@dataclass(slots=True)
class MatchedFinding:
    """One head finding and the base finding it corresponds to, if any."""

    head: HealthFindingData
    base: HealthFindingData | None
    key: FindingKey
    ordinal: int
    kind: ChangeKind

    @property
    def severity_before(self) -> str | None:
        return str(self.base.severity) if self.base is not None else None


@dataclass(slots=True)
class MatchResult:
    matched: list[MatchedFinding] = field(default_factory=list)
    resolved: list[HealthFindingData] = field(default_factory=list)

    @property
    def unchanged_total(self) -> int:
        return sum(1 for m in self.matched if m.kind == "unchanged")

    def of_kind(self, *kinds: ChangeKind) -> list[MatchedFinding]:
        return [m for m in self.matched if m.kind in kinds]


class FindingMatcher:
    """Match two finding sets by tolerant identity, then classify each pair.

    Pairing is per identity group: reserve corresponding untouched lines when
    the diff supplies them, then use proximity for the remainder. A marker
    that fires twice in one symbol keeps a stable correspondence, and a
    same-marker replacement remains one finding.
    """

    def __init__(
        self,
        rename_map: dict[str, str] | None = None,
        *,
        changes: dict[str, FileChange] | None = None,
    ) -> None:
        self.rename_map = rename_map or {}
        self.changes = changes or {}

    def match(self, base: list[HealthFindingData], head: list[HealthFindingData]) -> MatchResult:
        base_groups = self._group(base, normalize=True)
        head_groups = self._group(head, normalize=False)
        result = MatchResult()
        for key, head_items in head_groups.items():
            base_items = base_groups.pop(key, [])
            matched, unmatched = self._pair(key, base_items, head_items)
            result.matched.extend(matched)
            # Base findings this group could not account for are gone from head.
            result.resolved.extend(unmatched)
        for leftovers in base_groups.values():
            result.resolved.extend(leftovers)
        return result

    # -- internals ----------------------------------------------------------

    def _group(
        self, findings: list[HealthFindingData], *, normalize: bool
    ) -> dict[FindingKey, list[HealthFindingData]]:
        groups: dict[FindingKey, list[HealthFindingData]] = {}
        for finding in findings:
            path = (
                normalize_path(finding.file_path, self.rename_map)
                if normalize
                else finding.file_path
            )
            groups.setdefault(finding_key(finding, path=path), []).append(finding)
        for items in groups.values():
            items.sort(key=lambda f: (f.line_start or 0, f.line_end or 0))
        return groups

    def _pair(
        self, key: FindingKey, base: list[HealthFindingData], head: list[HealthFindingData]
    ) -> tuple[list[MatchedFinding], list[HealthFindingData]]:
        """Reserve unchanged lines, then pair remaining findings closest-first.

        Closest-first over all candidate pairs, not first-come-first-served:
        letting the earliest head finding claim the only base peer lets an
        unrelated new finding absorb a moved one, which both hides the new
        finding and reports the moved one as introduced.
        """
        base_lines, head_lines = self._unchanged_lines(key.path, base, head)
        candidates = sorted(
            (
                not (bi in base_lines and hi in head_lines and base_lines[bi] == head_lines[hi]),
                line_distance(b, h), hi, bi,
            )
            for hi, h in enumerate(head)
            for bi, b in enumerate(base)
        )
        peers: dict[int, HealthFindingData] = {}
        taken: set[int] = set()
        for _unmapped, _distance, head_index, base_index in candidates:
            if head_index in peers or base_index in taken:
                continue
            peers[head_index] = base[base_index]
            taken.add(base_index)
        matched = [
            MatchedFinding(item, peers.get(ordinal), key, ordinal, _classify(peers.get(ordinal), item))
            for ordinal, item in enumerate(head)
        ]
        return matched, [b for i, b in enumerate(base) if i not in taken]

    def _unchanged_lines(
        self, path: str, base: list[HealthFindingData], head: list[HealthFindingData]
    ) -> tuple[dict[int, int], dict[int, int]]:
        """Rank untouched lines after removing each side's changed lines.

        Their ranks correspond exactly across a reliable unified-zero diff.
        Reserve these pairs before proximity, so an insertion cannot steal a
        moved finding. Rewritten lines retain the existing tolerant matching.
        """
        change = self.changes.get(path)
        if change is None or not change.diff_reliable or change.diff is None:
            return {}, {}
        diff = change.diff
        added = sorted(diff.new_lines)
        base_lines = {}
        for index, finding in enumerate(base):
            line = finding.line_start
            if line is None or any(start <= line <= end for start, end in diff.old_ranges):
                continue
            base_lines[index] = line - sum(
                end - start + 1 for start, end in diff.old_ranges if end < line
            )
        head_lines = {
            index: finding.line_start - bisect_left(added, finding.line_start)
            for index, finding in enumerate(head)
            if finding.line_start is not None and finding.line_start not in diff.new_lines
        }
        return base_lines, head_lines


def _classify(base: HealthFindingData | None, head: HealthFindingData) -> ChangeKind:
    """Introduced when nothing matched; worsened only on a real regression."""
    if base is None:
        return "introduced"
    if severity_rank(head.severity) > severity_rank(base.severity):
        return "worsened"
    if severity_rank(head.severity) < severity_rank(base.severity):
        return "unchanged"
    if finding_raw_deduction(head) - finding_raw_deduction(base) > _IMPACT_EPSILON:
        return "worsened"
    return "unchanged"
