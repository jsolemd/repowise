"""A repo opted out of the fan-out contributes nothing cross-repo.

``WorkspaceRepoEntry.federated`` says a repo can be indexed and addressable
without belonging in workspace-wide answers. v0.49.0 added four cross-repo
answers (the contracts page, the system graph, ``cross_repo_blast_radius`` and
the cross-repo test impact on ``get_change_risk`` / ``get_blast_radius``), and
every one of them reads the contract store rather than re-enumerating the
workspace. So the store's repo gate is where the opt-out has to hold; if it
leaks there, it leaks into all four at once.
"""

from __future__ import annotations

from pathlib import Path

from repowise.core.workspace.config import RepoEntry, WorkspaceConfig
from repowise.core.workspace.contracts import contract_repo_paths
from repowise.core.workspace.system_graph import _detect_boundaries_by_repo


def _indexed(root: Path, name: str) -> None:
    (root / name / ".repowise").mkdir(parents=True, exist_ok=True)


def _config(*entries: RepoEntry) -> WorkspaceConfig:
    return WorkspaceConfig(version=1, repos=list(entries), default_repo=entries[0].alias)


def test_a_non_federated_repo_is_not_extracted_for_contracts(tmp_path: Path) -> None:
    for name in ("web", "api", "engine"):
        _indexed(tmp_path, name)
    config = _config(
        RepoEntry(path="web", alias="web"),
        RepoEntry(path="api", alias="api"),
        RepoEntry(path="engine", alias="engine", federated=False),
    )

    paths = contract_repo_paths(config, tmp_path)

    assert set(paths) == {"web", "api"}
    assert "engine" not in paths


def test_the_default_is_still_every_indexed_repo(tmp_path: Path) -> None:
    """The opt-out is opt-in. Nothing changes for a workspace nobody set it on."""
    for name in ("web", "api"):
        _indexed(tmp_path, name)
    config = _config(RepoEntry(path="web", alias="web"), RepoEntry(path="api", alias="api"))

    assert set(contract_repo_paths(config, tmp_path)) == {"web", "api"}


def test_upstreams_indexed_gate_still_applies(tmp_path: Path) -> None:
    """Federated is a second gate, not a replacement for the first."""
    _indexed(tmp_path, "web")
    (tmp_path / "api").mkdir()  # present, never indexed
    config = _config(RepoEntry(path="web", alias="web"), RepoEntry(path="api", alias="api"))

    assert set(contract_repo_paths(config, tmp_path)) == {"web"}


def test_opting_out_does_not_remove_the_repo_from_the_workspace(tmp_path: Path) -> None:
    """The opt-out narrows fan-out only — discovery and direct addressing stay."""
    from repowise.core.workspace.registry import RepoRegistry

    for name in ("web", "engine"):
        _indexed(tmp_path, name)
    config = _config(
        RepoEntry(path="web", alias="web"),
        RepoEntry(path="engine", alias="engine", federated=False),
    )
    registry = RepoRegistry(tmp_path, config)

    assert "engine" in registry.get_all_aliases()
    assert "engine" not in registry.get_federated_aliases()
    assert registry.resolve_repo_param("engine") == "engine"
    assert "engine" not in registry.resolve_repo_param("all")


def test_every_cross_repo_producer_reads_one_gate(tmp_path: Path) -> None:
    """Contracts, co-change detection and the system graph share the set.

    Measured at the v0.49.0 landing: the contract store was gated but the
    co-change detector and the boundary walk were not, so the opted-out repo
    ranked as the top impacted repo in ``get_blast_radius`` on every product
    target through ``co_change`` edges alone.
    """
    for name in ("web", "api", "engine"):
        _indexed(tmp_path, name)
    config = _config(
        RepoEntry(path="web", alias="web"),
        RepoEntry(path="api", alias="api"),
        RepoEntry(path="engine", alias="engine", federated=False),
    )

    gate = config.federated_indexed_repo_paths(tmp_path)

    assert set(gate) == {"web", "api"}
    assert contract_repo_paths(config, tmp_path) == gate
    assert set(_detect_boundaries_by_repo(config, tmp_path)) == {"web", "api"}


def test_the_gate_keeps_upstreams_indexed_rule(tmp_path: Path) -> None:
    _indexed(tmp_path, "web")
    (tmp_path / "api").mkdir()  # present, never indexed, federated by default
    config = _config(RepoEntry(path="web", alias="web"), RepoEntry(path="api", alias="api"))

    assert set(config.federated_indexed_repo_paths(tmp_path)) == {"web"}
