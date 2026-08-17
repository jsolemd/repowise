"""Building the source-search coordinator from each host's own state.

:class:`~repowise.core.source_search.coordinator.SourceSearchCoordinator` takes
every store it uses as an argument, which is what lets one implementation serve
two hosts — but the two hosts keep those stores in different places. The MCP
server holds them in module globals filled in by its lifespan; the REST app
holds them on ``app.state`` and builds them from a database URL. This module is
where that difference lives, so neither the coordinator nor the two call sites
have to know about the other host.

It lives in the server package rather than beside the coordinator because it
reads server state. Core must not import the server, and a "wiring" module in
core that did would be that import wearing a different name.

Both entry points are **fail-soft**: anything missing — no source index for
this repository, a mock embedder, an unreadable sidecar — returns ``None``, and
the caller falls through to the stock search path. A half-built coordinator
answering with half a corpus would be worse than the behaviour it replaced.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: One coordinator per MCP process, and one lock so two concurrent first
#: queries do not both open the FTS sidecar.
_mcp_coordinator: Any = None
_mcp_lock = asyncio.Lock()
_mcp_attempted = False

#: Where the REST app caches its own instance.
_REST_ATTR = "source_search_coordinator"

#: How long a caller waits for the MCP server's background vector-store load.
#: Bounded and short: the coordinator can be rebuilt on the next query, and a
#: search that blocks on startup is the failure mode this guards against.
_READY_TIMEOUT = 10.0


def _build(repo_path: Path | str, wiki_vectors: Any, wiki_fts: Any) -> Any:
    """A coordinator over *repo_path*, or None when it has no source index.

    The embedder is taken from the wiki vector store rather than resolved
    again, because the two corpora are only comparable when one embedder wrote
    both — the dense fusion is arithmetic on that assumption.
    """
    from repowise.core.providers.embedding import store_has_semantic_vectors
    from repowise.core.source_search.coordinator import SourceSearchCoordinator
    from repowise.core.source_search.fts import SourceFTSIndex, default_fts_path
    from repowise.core.source_search.manifest import default_manifest_path
    from repowise.core.source_search.vector_store import SourceChunkVectorStore

    repo = Path(repo_path)
    fts_path = default_fts_path(repo)
    if not default_manifest_path(repo).exists() or not fts_path.exists():
        log.debug("source-search: no source index at %s", repo)
        return None

    embedder = getattr(wiki_vectors, "_embedder", None)
    if embedder is None or not store_has_semantic_vectors(wiki_vectors):
        # A keyless or mock embedder cannot query a corpus embedded by a real
        # one — the widths differ, and where they do not the vectors are noise.
        log.debug("source-search: no semantic embedder available, staying on the stock path")
        return None

    try:
        source_fts = SourceFTSIndex(fts_path)
    except Exception:
        log.debug("source-search: could not open the FTS sidecar", exc_info=True)
        return None

    return SourceSearchCoordinator(
        repo_path=repo,
        embedder=embedder,
        source_vectors=SourceChunkVectorStore(
            str(repo / ".repowise" / "lancedb"), embedder=embedder
        ),
        source_fts=source_fts,
        wiki_vectors=wiki_vectors,
        wiki_fts=wiki_fts,
    )


async def mcp_coordinator() -> Any:
    """The MCP server's coordinator, built once from its lifespan state.

    Waits for the background vector-store load first. Not on the context
    helper's readiness event: in single-repo mode that helper substitutes a
    *fresh, never-set* event when the state global is None, and awaiting it
    costs a caller thirty seconds for no reason. This awaits the state global
    itself, which the lifespan sets, and treats its absence as "nothing to wait
    for" rather than as "wait".
    """
    global _mcp_coordinator, _mcp_attempted

    from repowise.server.mcp_server import _state

    if _mcp_attempted:
        return _mcp_coordinator
    async with _mcp_lock:
        if _mcp_attempted:
            return _mcp_coordinator
        ready = _state._vector_store_ready
        if ready is not None:
            try:
                await asyncio.wait_for(ready.wait(), timeout=_READY_TIMEOUT)
            except TimeoutError:
                log.debug("source-search: vector stores still loading, staying on the stock path")
                return None
        repo_path = _state._repo_path
        if not repo_path:
            _mcp_attempted = True
            return None
        try:
            _mcp_coordinator = _build(repo_path, _state._vector_store, _state._fts)
        except Exception:
            log.debug("source-search: MCP coordinator construction failed", exc_info=True)
            _mcp_coordinator = None
        _mcp_attempted = True
        return _mcp_coordinator


async def rest_coordinator(app_state: Any) -> Any:
    """The REST app's coordinator, built once and cached on ``app.state``."""
    existing = getattr(app_state, _REST_ATTR, None)
    if existing is not None:
        return existing
    if getattr(app_state, f"{_REST_ATTR}_attempted", False):
        return None

    repo_path = _repo_root_from_db_url(getattr(app_state, "db_url", "") or "")
    coordinator = None
    if repo_path is not None:
        try:
            coordinator = _build(
                repo_path,
                getattr(app_state, "vector_store", None),
                getattr(app_state, "fts", None),
            )
        except Exception:
            log.debug("source-search: REST coordinator construction failed", exc_info=True)
    setattr(app_state, _REST_ATTR, coordinator)
    setattr(app_state, f"{_REST_ATTR}_attempted", True)
    return coordinator


def _repo_root_from_db_url(db_url: str) -> Path | None:
    """The repository a wiki database belongs to, from its URL.

    The REST app records where it opened ``wiki.db`` and not which repository
    that was, so the root is read back off the path: ``<repo>/.repowise/wiki.db``
    is the layout every other component already depends on. Returns None for a
    URL that is not a repo-local SQLite file — a Postgres deployment or an
    in-memory database has no worktree behind it to index.
    """
    marker = "sqlite+aiosqlite:///"
    if not db_url.startswith(marker):
        return None
    path = Path(db_url[len(marker) :])
    if path.parent.name != ".repowise":
        return None
    return path.parent.parent


def reset_for_tests() -> None:
    """Forget the MCP process cache. Only tests have a reason to."""
    global _mcp_coordinator, _mcp_attempted
    _mcp_coordinator = None
    _mcp_attempted = False
