"""Durable state for incremental deleted-file prune refusals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRUNE_REFUSALS_STATE_KEY = "prune_refusals"


@dataclass(frozen=True, slots=True)
class PruneRefusal:
    """One table whose apparent mass deletion exceeded the safety floor."""

    table: str
    candidate_paths: int
    persisted_paths: int

    @property
    def reason(self) -> str:
        """Compact status reason for machine-facing trust payloads."""
        return f"prune_refused: {self.table} {self.candidate_paths}/{self.persisted_paths}"

    @property
    def message(self) -> str:
        """Concrete operator-facing account of what the guard refused."""
        return (
            f"Deleted-file prune refused for {self.table}: {self.candidate_paths} of "
            f"{self.persisted_paths} paths looked deleted, which reads as a broken "
            "run rather than a commit. Re-run with --accept-mass-deletion after "
            "confirming the deletion."
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "candidate_paths": self.candidate_paths,
            "persisted_paths": self.persisted_paths,
        }

    @classmethod
    def from_state(cls, value: Any) -> PruneRefusal | None:
        if not isinstance(value, dict):
            return None
        table = value.get("table")
        candidates = value.get("candidate_paths")
        persisted = value.get("persisted_paths")
        if not isinstance(table, str) or not table:
            return None
        if not isinstance(candidates, int) or candidates < 0:
            return None
        if not isinstance(persisted, int) or persisted < 0:
            return None
        return cls(table=table, candidate_paths=candidates, persisted_paths=persisted)


@dataclass(frozen=True, slots=True)
class DeletedFilePruneOutcome:
    """What the deleted/excluded-file prune established in one persistence run."""

    attempted: bool = False
    pruned_paths: int = 0
    refusals: tuple[PruneRefusal, ...] = ()
    tombstoned_page_ids: tuple[str, ...] = ()


def state_prune_refusals(state: dict[str, Any]) -> tuple[PruneRefusal, ...]:
    """Validated prune refusals from a repository state mapping."""
    block = state.get(PRUNE_REFUSALS_STATE_KEY)
    findings = block.get("findings") if isinstance(block, dict) else None
    if not isinstance(findings, list):
        return ()
    parsed = (PruneRefusal.from_state(item) for item in findings)
    return tuple(item for item in parsed if item is not None)


def prune_repair_base(state: dict[str, Any]) -> str | None:
    """Commit before the oldest unresolved refused prune, when recorded."""
    if not state_prune_refusals(state):
        return None
    block = state.get(PRUNE_REFUSALS_STATE_KEY)
    value = block.get("from_commit") if isinstance(block, dict) else None
    return value if isinstance(value, str) and value else None


def apply_prune_outcome(
    state: dict[str, Any],
    prior_state: dict[str, Any],
    outcome: DeletedFilePruneOutcome,
    *,
    from_commit: str | None,
) -> None:
    """Persist or clear the refusal without erasing it on a skipped attempt."""
    if not outcome.attempted:
        return
    if not outcome.refusals:
        state.pop(PRUNE_REFUSALS_STATE_KEY, None)
        return

    oldest = prune_repair_base(prior_state) or from_commit
    block: dict[str, Any] = {
        "findings": [refusal.to_state() for refusal in outcome.refusals],
    }
    if oldest:
        block["from_commit"] = oldest
    state[PRUNE_REFUSALS_STATE_KEY] = block
