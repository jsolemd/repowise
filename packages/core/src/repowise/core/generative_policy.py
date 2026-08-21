"""Hard deployment policy for runtimes that must never call a generative model."""

from __future__ import annotations

import os
from pathlib import Path

NO_GENERATIVE_ENV = "REPOWISE_TOOLS_NO_GENERATIVE"
NO_GENERATIVE_INIT_COMPLETE_MARKER = "RepoWise no-generative init completed"
NO_GENERATIVE_INDEX_MARKER = f"Generative indexing disabled: {NO_GENERATIVE_ENV}=1"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def generative_calls_disabled(repo_path: Path | str | None = None) -> bool:
    """Return whether any applicable hard no-generative source is enabled.

    A truthy repository or workspace-root policy is fail-closed: an earlier
    falsy dotenv merge or process value cannot mask it. The pure repo reader
    also avoids loading unrelated credentials while a workspace is still
    choosing its primary repository.
    """

    values = [os.environ.get(NO_GENERATIVE_ENV, "")]
    if repo_path is not None:
        from repowise.core.repo_config import load_repo_env
        from repowise.core.workspace import find_workspace_root

        path = Path(repo_path)
        policy_paths = {path}
        workspace_root = find_workspace_root(path)
        if workspace_root is not None:
            policy_paths.add(workspace_root)
        values.extend(
            load_repo_env(policy_path).get(NO_GENERATIVE_ENV, "") for policy_path in policy_paths
        )
    return any(value.strip().lower() in _TRUTHY for value in values)
