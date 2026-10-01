"""The workspace update's daily LanceDB version sweep.

The reconciliation timer runs ``repowise update --workspace`` every ten
minutes, so the sweep rides on it and runs once a day per repository; see
:mod:`repowise.core.lance_retention` for the rule and its reader contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from repowise.cli.helpers import console, run_async
from repowise.cli.ui.brand import format_bytes


def sweep_workspace_lance_versions(ws_root: Path, ws_config: Any, repo_filter: str | None) -> None:
    """Prune the members' old LanceDB versions when due; report what moved."""

    from repowise.core.lance_retention import sweep_repositories

    repos = [
        (ws_root / entry.path).resolve()
        for entry in ws_config.repos
        if repo_filter is None or entry.alias == repo_filter
    ]
    for outcome in run_async(sweep_repositories(ws_root, repos)):
        if outcome.status == "swept":
            for table in outcome.tables:
                if table.versions_removed:
                    console.print(
                        f"  [dim]{outcome.repo.name} {Path(table.table).stem}: "
                        f"{table.versions_removed:,} old versions removed, "
                        f"{format_bytes(table.bytes_removed)} reclaimed[/dim]"
                    )
        elif outcome.status in {"busy", "failed"}:
            console.print(
                f"  [yellow]{outcome.repo.name}: LanceDB version sweep {outcome.status}: "
                f"{outcome.reason}[/yellow]"
            )
