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
_mcp_generation: tuple[Any, ...] | None = None

#: Where the REST app caches its own instance.
_REST_ATTR = "source_search_coordinator"
_REST_GENERATION_ATTR = "source_search_generation"
_rest_lock = asyncio.Lock()

#: How long a caller waits for the MCP server's background vector-store load.
#: Bounded and short: the coordinator can be rebuilt on the next query, and a
#: search that blocks on startup is the failure mode this guards against.
_READY_TIMEOUT = 10.0


class _StatusCoordinator:
    """Add live queue health without changing the ranking coordinator."""

    def __init__(self, inner: Any, repo: Path, source_vectors: Any, source_fts: Any) -> None:
        self._inner = inner
        self._repo = repo
        self._source_vectors = source_vectors
        self._source_fts = source_fts

    async def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        response = await self._inner.search(*args, **kwargs)
        try:
            from repowise.core.source_search.status import inspect_source_index

            status = await inspect_source_index(self._repo, verify_stores=False)
            meta = response.setdefault("_meta", {}).setdefault("source_search", {})
            meta.update(
                {
                    "status": status.state,
                    "generation_id": status.generation_id,
                    "generation_sequence": status.generation_sequence,
                    "pending_updates": status.pending_updates,
                    "blocked_updates": status.blocked_updates,
                    "building_updates": status.building_updates,
                    "ready_updates": status.ready_updates,
                    "stale_files": len(status.stale_files),
                }
            )
            if status.last_error:
                meta["last_error"] = status.last_error
            if status.degraded:
                meta["degraded"] = True
                meta.setdefault("degraded_reason", status.last_error or status.state)
        except Exception:
            log.debug("source-search: status enrichment failed", exc_info=True)
        return response

    async def close(self) -> None:
        await self._source_vectors.close()
        self._source_fts.close()


def _manifest_key(repo_path: Path | str) -> tuple[Any, ...] | None:
    from repowise.core.source_search.manifest import default_manifest_path, read_manifest

    manifest = read_manifest(default_manifest_path(repo_path))
    if manifest is None:
        return None
    return (
        manifest.generation_id,
        manifest.generation_sequence,
        manifest.recipe_fingerprint,
        manifest.lance_table,
        manifest.fts_path,
    )


def _build(repo_path: Path | str, wiki_vectors: Any, wiki_fts: Any) -> Any:
    """A coordinator over *repo_path*, or None when it has no source index.

    The embedder is taken from the wiki vector store rather than resolved
    again, because the two corpora are only comparable when one embedder wrote
    both — the dense fusion is arithmetic on that assumption.
    """
    from repowise.core.providers.embedding import store_has_semantic_vectors
    from repowise.core.source_search.coordinator import SourceSearchCoordinator
    from repowise.core.source_search.fts import SourceFTSIndex
    from repowise.core.source_search.generation import GenerationRef
    from repowise.core.source_search.manifest import default_manifest_path, read_manifest
    from repowise.core.source_search.vector_store import SourceChunkVectorStore

    repo = Path(repo_path)
    manifest = read_manifest(default_manifest_path(repo))
    if manifest is None:
        log.debug("source-search: no source index at %s", repo)
        return None
    fts_path = repo / manifest.fts_path
    if not fts_path.is_file():
        log.warning("source-search: active FTS generation is missing at %s", fts_path)
        return None

    embedder = getattr(wiki_vectors, "_embedder", None)
    if embedder is None or not store_has_semantic_vectors(wiki_vectors):
        # A keyless or mock embedder cannot query a corpus embedded by a real
        # one — the widths differ, and where they do not the vectors are noise.
        log.debug("source-search: no semantic embedder available, staying on the stock path")
        return None
    if int(getattr(embedder, "dimensions", 0) or 0) != manifest.embedder.dims:
        log.warning(
            "source-search: configured embedder width does not match active generation "
            "(%s != %s)",
            getattr(embedder, "dimensions", None),
            manifest.embedder.dims,
        )
        return None

    generation = GenerationRef(manifest.generation_id, manifest.generation_sequence)

    try:
        source_fts = SourceFTSIndex(fts_path, generation=generation)
    except Exception:
        log.debug("source-search: could not open the FTS sidecar", exc_info=True)
        return None

    try:
        source_vectors = SourceChunkVectorStore(
            str(repo / ".repowise" / "lancedb"),
            embedder=embedder,
            table_name=manifest.lance_table,
            generation=generation,
        )
    except Exception:
        source_fts.close()
        raise
    coordinator = SourceSearchCoordinator(
        repo_path=repo,
        embedder=embedder,
        source_vectors=source_vectors,
        source_fts=source_fts,
        wiki_vectors=wiki_vectors,
        wiki_fts=wiki_fts,
    )
    return _StatusCoordinator(coordinator, repo, source_vectors, source_fts)


async def mcp_coordinator() -> Any:
    """The MCP server's coordinator, built once from its lifespan state.

    Waits for the background vector-store load first. Not on the context
    helper's readiness event: in single-repo mode that helper substitutes a
    *fresh, never-set* event when the state global is None, and awaiting it
    costs a caller thirty seconds for no reason. This awaits the state global
    itself, which the lifespan sets, and treats its absence as "nothing to wait
    for" rather than as "wait".
    """
    global _mcp_coordinator, _mcp_generation

    from repowise.server.mcp_server import _state

    repo_path = _state._repo_path
    generation = _manifest_key(repo_path) if repo_path else None
    if _mcp_coordinator is not None and generation == _mcp_generation:
        return _mcp_coordinator
    async with _mcp_lock:
        repo_path = _state._repo_path
        generation = _manifest_key(repo_path) if repo_path else None
        if _mcp_coordinator is not None and generation == _mcp_generation:
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
            return None
        old = _mcp_coordinator
        _mcp_coordinator = None
        _mcp_generation = None
        if old is not None:
            await old.close()
        try:
            _mcp_coordinator = _build(repo_path, _state._vector_store, _state._fts)
        except Exception:
            log.debug("source-search: MCP coordinator construction failed", exc_info=True)
            _mcp_coordinator = None
        if _mcp_coordinator is not None:
            _mcp_generation = generation
        return _mcp_coordinator


async def rest_coordinator(app_state: Any) -> Any:
    """The REST coordinator, reopened when the manifest generation changes."""
    repo_path = _repo_root_from_db_url(getattr(app_state, "db_url", "") or "")
    generation = _manifest_key(repo_path) if repo_path is not None else None
    existing = getattr(app_state, _REST_ATTR, None)
    if existing is not None and generation == getattr(app_state, _REST_GENERATION_ATTR, None):
        return existing

    async with _rest_lock:
        existing = getattr(app_state, _REST_ATTR, None)
        if existing is not None and generation == getattr(
            app_state, _REST_GENERATION_ATTR, None
        ):
            return existing
        if existing is not None:
            await existing.close()
        coordinator = None
        if repo_path is None:
            setattr(app_state, _REST_ATTR, None)
            setattr(app_state, _REST_GENERATION_ATTR, None)
            return None
        try:
            coordinator = _build(
                repo_path,
                getattr(app_state, "vector_store", None),
                getattr(app_state, "fts", None),
            )
        except Exception:
            log.debug("source-search: REST coordinator construction failed", exc_info=True)
        setattr(app_state, _REST_ATTR, coordinator)
        setattr(app_state, _REST_GENERATION_ATTR, generation if coordinator is not None else None)
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


#: Workspace mode — one coordinator per member repo, keyed by resolved repo
#: path. The value remembers the manifest generation it was built against and
#: the identity of the wiki vector store it was built FROM: the registry
#: evicts and reloads contexts, and a reloaded context arrives with fresh
#: store objects while the old coordinator still holds the closed ones.
#: Identity (``is``, via ``id``), never equality — the job-lock work already
#: paid for that lesson.
_ctx_coordinators: dict[Path, tuple[Any, tuple[Any, ...], int]] = {}
_ctx_lock = asyncio.Lock()


async def context_coordinator(ctx: Any) -> Any:
    """A coordinator over one workspace repo context, or ``None`` (fail-soft).

    The single-repo entry point reads the MCP module globals; a workspace has
    no meaningful globals to read, so this one takes the ``RepoContext``
    itself. Same doctrine otherwise: anything missing — no source index for
    this member repo, stores still loading, a mock embedder — returns ``None``
    and the caller falls through to the stock path for that repo.
    """
    repo_path = Path(ctx.path)
    generation = _manifest_key(repo_path)
    if generation is None:
        return None
    wiki_vs = getattr(ctx, "vector_store", None)
    cached = _ctx_coordinators.get(repo_path)
    if cached is not None and cached[1] == generation and cached[2] == id(wiki_vs):
        return cached[0]
    async with _ctx_lock:
        # The registry loads real vector stores in the background and repoints
        # ctx.vector_store when done; build only from the settled store.
        ready = getattr(ctx, "vector_store_ready", None)
        if ready is not None:
            try:
                await asyncio.wait_for(ready.wait(), timeout=_READY_TIMEOUT)
            except TimeoutError:
                log.debug(
                    "source-search: vector stores for %s still loading, staying on "
                    "the stock path",
                    getattr(ctx, "alias", repo_path),
                )
                return None
        wiki_vs = getattr(ctx, "vector_store", None)
        cached = _ctx_coordinators.get(repo_path)
        if cached is not None and cached[1] == generation and cached[2] == id(wiki_vs):
            return cached[0]
        if cached is not None:
            _ctx_coordinators.pop(repo_path, None)
            try:
                await cached[0].close()
            except Exception:
                log.debug("source-search: closing a stale workspace coordinator failed",
                          exc_info=True)
        try:
            built = _build(repo_path, wiki_vs, getattr(ctx, "fts", None))
        except Exception:
            log.debug(
                "source-search: workspace coordinator construction failed for %s",
                getattr(ctx, "alias", repo_path),
                exc_info=True,
            )
            return None
        if built is None:
            return None
        _ctx_coordinators[repo_path] = (built, generation, id(wiki_vs))
        return built


def reset_for_tests() -> None:
    """Forget the MCP process caches. Only tests have a reason to."""
    global _mcp_coordinator, _mcp_generation
    _mcp_coordinator = None
    _mcp_generation = None
    _ctx_coordinators.clear()
