"""Host-side source-index reconciliation for CLI workflows.

The core lifecycle deliberately receives an embedder instead of resolving
configuration: core is also used by the server and workspace runtime, whose
configuration ownership differs.  This module is the CLI adapter.  It keeps
provider selection in the same place as wiki-vector selection and turns an
unavailable semantic backend into an explicit pending/degraded condition.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any


class SourceSearchUnavailableError(RuntimeError):
    """The source outbox is durable, but no truthful embedder can drain it."""


def _embedder_label(identity: Any) -> str:
    return f"{identity.provider}/{identity.model}/{identity.dims}d"


async def configured_source_recipe_changes(repo_path: Path | str) -> tuple[str, ...]:
    """Describe stored/runtime source-recipe drift for workspace skip logic.

    Missing source indexes are left to their normal initialization path. Once
    a publication exists, however, its parser and embedder identities are the
    same facts exposed by ``get_index_status`` and consumed by reconciliation;
    a commit-only skip must not overrule either mismatch.
    """
    from repowise.core.source_search import source_search_enabled

    if not source_search_enabled():
        return ()

    from repowise.cli.providers.embedders import build_embedder, resolve_embedder_for_repo
    from repowise.core.ingestion.parse_cache import parser_fingerprint
    from repowise.core.source_search.manifest import (
        default_manifest_path,
        identify_embedder,
        recipe_fingerprint,
    )
    from repowise.core.source_search.status import inspect_source_index

    repo = Path(repo_path).resolve()
    if not default_manifest_path(repo).is_file():
        return ()
    status = await inspect_source_index(repo, verify_stores=False)
    if status.manifest_state != "ok":
        return ()

    resolved_name = resolve_embedder_for_repo(repo).strip().lower()
    runtime_embedder = identify_embedder(
        build_embedder(resolved_name, repo),
        provider=resolved_name,
    )
    runtime_parser = parser_fingerprint()
    changes: list[str] = []
    if status.parser_fingerprint is not None and status.parser_fingerprint != runtime_parser:
        changes.append(
            f"parser fingerprint {status.parser_fingerprint[:12]} → {runtime_parser[:12]}"
        )
    if status.embedder is not None and status.embedder != runtime_embedder:
        changes.append(
            "embedder identity "
            f"{_embedder_label(status.embedder)} → {_embedder_label(runtime_embedder)}"
        )
    indexed_recipe = getattr(status, "recipe_fingerprint", None)
    runtime_recipe = recipe_fingerprint(runtime_embedder)
    if indexed_recipe and indexed_recipe != runtime_recipe and not changes:
        changes.append(f"source recipe fingerprint {indexed_recipe[:12]} → {runtime_recipe[:12]}")
    return tuple(changes)


async def reconcile_configured_source_index(
    repo_path: Path | str,
    *,
    embedder_name: str | None = None,
    embedder: Any | None = None,
    force_full: bool = False,
    batch_size: int = 16,
    allow_keyless: bool = False,
    db_url: str | None = None,
):
    """Drain one repository's durable source queue with its pinned recipe.

    Returns ``None`` when the feature flag is off.  Once enabled, inability to
    construct a real semantic embedder raises
    :class:`SourceSearchUnavailableError`; callers may keep the primary wiki
    update successful, but must surface the source lane as degraded.  The
    queued update remains pending for a later retry.
    """

    from repowise.core.source_search import source_search_enabled

    if not source_search_enabled():
        return None

    from repowise.cli.providers.embedders import build_embedder, resolve_embedder_for_repo
    from repowise.core.providers.embedding.base import KeylessEmbedder
    from repowise.core.source_search.lifecycle import (
        reconcile_source_index,
        record_source_index_error,
    )
    from repowise.core.source_search.manifest import identify_embedder

    repo = Path(repo_path).resolve()
    resolved_name = (embedder_name or resolve_embedder_for_repo(repo)).strip().lower()
    implementation = embedder if embedder is not None else build_embedder(resolved_name, repo)
    if isinstance(implementation, KeylessEmbedder) and not allow_keyless:
        message = (
            "source search requires a real embedder; configure Ollama or another "
            "semantic backend, then rerun the update"
        )
        with suppress(Exception):
            await record_source_index_error(repo, message, db_url=db_url)
        raise SourceSearchUnavailableError(message)

    # When the caller hands us an already-built adapter, its concrete module
    # is the runtime fact; a stale config label must not enter the recipe.
    provider = None if embedder is not None else resolved_name
    # A2 uses one query vector for both the source and wiki dense legs.  An
    # asymmetric document/query prefix would require separate query vectors;
    # recording prefixes before that A3 protocol exists would fingerprint a
    # recipe the reader cannot actually execute.
    identity = identify_embedder(implementation, provider=provider)
    return await reconcile_source_index(
        repo,
        embedder=implementation,
        embedder_identity=identity,
        db_url=db_url,
        force_full=force_full,
        batch_size=batch_size,
    )


async def reconcile_configured_source_indexes(
    repo_paths: list[Path],
) -> dict[Path, Any | Exception | None]:
    """Drain several workspace members without hiding a sibling's failure."""

    outcomes: dict[Path, Any | Exception | None] = {}
    for repo_path in repo_paths:
        try:
            outcomes[repo_path] = await reconcile_configured_source_index(repo_path)
        except Exception as exc:
            outcomes[repo_path] = exc
    return outcomes
