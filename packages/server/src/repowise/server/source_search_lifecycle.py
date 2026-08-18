"""Server-owned adapter for the core source-index lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any


async def reconcile_server_source_index(
    repo_path: Path | str,
    *,
    embedder: Any,
    db_url: str | None = None,
):
    """Drain a job's source queue with the server's already-loaded embedder."""

    from repowise.core.source_search import source_search_enabled

    if not source_search_enabled():
        return None

    from repowise.core.providers.embedding.base import KeylessEmbedder
    from repowise.core.source_search.lifecycle import (
        reconcile_source_index,
        record_source_index_error,
    )
    from repowise.core.source_search.manifest import identify_embedder

    repo = Path(repo_path).resolve()
    if embedder is None or isinstance(embedder, KeylessEmbedder):
        message = "source search is pending because the server has no semantic embedder"
        await record_source_index_error(repo, message, db_url=db_url)
        raise RuntimeError(message)

    # Prefix-aware asymmetric embeddings are an A3 protocol: the shared A2
    # coordinator embeds once for both wiki and source dense legs.
    identity = identify_embedder(embedder)
    return await reconcile_source_index(
        repo,
        embedder=embedder,
        embedder_identity=identity,
        db_url=db_url,
    )
