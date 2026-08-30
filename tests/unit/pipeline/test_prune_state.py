"""Durable state contract for a refused deleted-file prune."""

from repowise.core.pipeline.prune_state import (
    DeletedFilePruneOutcome,
    PruneRefusal,
    apply_prune_outcome,
    prune_repair_base,
    state_prune_refusals,
)


def test_refusal_persists_the_repair_base_until_a_clean_attempt() -> None:
    refusal = PruneRefusal("graph_nodes", 448, 776)
    prior = {"last_sync_commit": "base-1"}
    refused = {**prior, "last_sync_commit": "head-2"}

    apply_prune_outcome(
        refused,
        prior,
        DeletedFilePruneOutcome(attempted=True, refusals=(refusal,)),
        from_commit="base-1",
    )

    assert prune_repair_base(refused) == "base-1"
    assert state_prune_refusals(refused) == (refusal,)

    healed = {**refused, "last_sync_commit": "head-3"}
    apply_prune_outcome(
        healed,
        refused,
        DeletedFilePruneOutcome(attempted=True, pruned_paths=448),
        from_commit="base-1",
    )

    assert state_prune_refusals(healed) == ()
    assert prune_repair_base(healed) is None


def test_skipped_prune_attempt_cannot_erase_a_refusal() -> None:
    prior = {
        "prune_refusals": {
            "from_commit": "base-1",
            "findings": [
                {
                    "table": "git_metadata",
                    "candidate_paths": 40,
                    "persisted_paths": 40,
                }
            ],
        }
    }
    unchanged = dict(prior)

    apply_prune_outcome(
        unchanged,
        prior,
        DeletedFilePruneOutcome(),
        from_commit="base-1",
    )

    assert unchanged == prior


def test_a_repair_base_without_a_valid_refusal_is_ignored() -> None:
    assert prune_repair_base({"prune_refusals": {"from_commit": "base", "findings": []}}) is None
