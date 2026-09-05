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
import re
import weakref
from collections import OrderedDict
from contextlib import asynccontextmanager
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

#: Maximum DEFINITION rows materialized for one served search row. The response
#: itself is capped, but a single dense file can contain far more declarations
#: than the eight symbol names the payload can carry.
_CONTAINED_SITE_FETCH_LIMIT = 200


def _served_paths(response: dict[str, Any]) -> set[str]:
    """Every repository path this particular response is standing behind."""

    paths = {
        str(row.get("file"))
        for row in (response.get("results") or [])
        if isinstance(row, dict) and row.get("file")
    }
    paths.update(
        str(row.get("path"))
        for row in (response.get("candidates") or [])
        if isinstance(row, dict) and row.get("path")
    )
    return paths


def _identifier_token(name: str) -> Any:
    """A matcher for *name* standing alone rather than inside a longer word.

    Bounded by identifier characters rather than by ``\\b``, because a name may
    legitimately begin or end with punctuation — a C++ destructor ``~Foo``, a
    Ruby predicate ``valid?`` — and ``\\b`` anchors against the punctuation
    instead of against the word, which either fails to match or matches inside
    ``invalid?``.
    """
    escaped = re.escape(name)
    return re.compile(rf"(?<![0-9A-Za-z_]){escaped}(?![0-9A-Za-z_])")


#: Hard ceiling on the text one row's corroboration read may collect. The
#: cited line range bounds the read in every ordinary case; this bounds the
#: pathological one, where a "line" is a minified bundle megabytes wide. Far
#: above any real source window, and small enough that a full page of rows
#: reading one each stays beneath the cost of the query that produced them.
_LIVE_SLICE_MAX_CHARS = 256_000


def _live_slice(repo: Path, file_path: str, start_line: int, end_line: int) -> str | None:
    """The cited lines of *file_path* as the working tree currently holds them.

    ``None`` means "this file cannot answer", and every caller treats that as
    a signal to fall back rather than as a finding — the enrichment above is
    built on never reading silence as evidence, and an unreadable file is
    silence. Four ways to get it: a path that is not repository-relative or
    escapes the root, a path with no regular file behind it, a range that is
    not a range, and a *start* line the file no longer reaches.

    Only the start line is checked against the file's length. A window row
    routinely cites a nominal end far past the last line of a short file, so
    an end beyond EOF is the ordinary case and the slice simply stops there;
    it is a start beyond EOF that says the row is describing a region which no
    longer exists, and that is the drift worth refusing on.

    Reading up to *end_line* is strictly less work than indexing this file
    already did, which is the real bound; the char ceiling only covers the
    file whose line structure makes "up to end_line" meaningless.
    """
    try:
        if start_line < 1 or end_line < start_line:
            return None
        relative = Path(file_path)
        if relative.is_absolute():
            return None
        root = repo.resolve()
        target = (root / relative).resolve()
        if root not in target.parents:
            return None
        if not target.is_file():
            return None
        collected: list[str] = []
        size = 0
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                if number < start_line:
                    continue
                if number > end_line:
                    break
                remaining = _LIVE_SLICE_MAX_CHARS - size
                if len(line) >= remaining:
                    # Cut inside the line, not after it: one minified line can
                    # be the whole budget several times over, and appending it
                    # first to discover that defeats the ceiling.
                    collected.append(line[:remaining])
                    break
                collected.append(line)
                size += len(line)
        if not collected:
            return None
        return "".join(collected)
    except Exception:
        log.debug("source-search: live slice unavailable for %s", file_path, exc_info=True)
        return None


class _StatusCoordinator:
    """Add live queue health without changing the ranking coordinator."""

    def __init__(
        self,
        inner: Any,
        repo: Path,
        source_vectors: Any,
        source_fts: Any,
        session_factory: Any = None,
    ) -> None:
        self._inner = inner
        self._repo = repo
        self._source_vectors = source_vectors
        self._source_fts = source_fts
        self._session_factory = session_factory
        #: Resolved once per coordinator, which is once per repository — the
        #: cache is safe precisely because a coordinator never spans repos.
        #: ``False`` is "not looked up yet", ``None`` is "looked up, no row".
        self._repository_id: Any = False

    async def search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        async with coordinator_lease(self):
            return await self._search(*args, **kwargs)

    async def _search(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        response = await self._inner.search(*args, **kwargs)
        divergence: Any = None
        try:
            from repowise.core.source_search.status import inspect_source_index
            from repowise.core.source_search.worktree import (
                HOT_PATH_CACHE_TTL,
                WorkingTreeDivergence,
            )

            # ``fts`` is the store this coordinator already holds open, which
            # is what lets a ``verify_stores=False`` read still answer which
            # paths the generation serves — and therefore whether the working
            # tree has moved out from under this very response.
            status = await inspect_source_index(
                self._repo,
                session_factory=self._session_factory,
                verify_stores=False,
                fts=self._source_fts,
                working_tree_max_age=HOT_PATH_CACHE_TTL,
            )
            meta = response.setdefault("_meta", {}).setdefault("source_search", {})
            reader_generation = self._source_fts.generation
            same_generation = reader_generation.generation_id == status.generation_id
            meta.update(
                {
                    "status": status.state if same_generation else "stale",
                    "generation_id": reader_generation.generation_id,
                    "generation_sequence": reader_generation.sequence,
                    "published_generation_id": status.generation_id,
                    "pending_updates": status.pending_updates,
                    "blocked_updates": status.blocked_updates,
                    "building_updates": status.building_updates,
                    "ready_updates": status.ready_updates,
                    "stale_files": len(status.stale_files),
                }
            )
            # Repo-wide counts answer "is the corpus behind the checkout".
            # ``served_*`` answers the question a reader of *this* response
            # actually has: are the rows in front of me describing files that
            # have since changed or stopped existing. A count alone cannot,
            # because a repo may diverge on files this query never touched.
            served = _served_paths(response)
            # The live inspector used the published manifest's ingest hashes.
            # They cannot verify an older reader; do not use them to certify
            # its snippets or enrich their definitions from a newer index.
            divergence = (
                status.working_tree
                if same_generation
                else WorkingTreeDivergence(
                    checked=False, unavailable_reason="reader_generation_changed"
                )
            )
            meta["working_tree"] = {
                "checked": divergence.checked,
                "modified": len(divergence.modified),
                "deleted": len(divergence.deleted),
                "served_modified": sorted(served & frozenset(divergence.modified)),
                "served_deleted": sorted(served & frozenset(divergence.deleted)),
                "unavailable_reason": divergence.unavailable_reason,
            }
            if status.last_error:
                meta["last_error"] = status.last_error
            if status.degraded:
                meta["degraded"] = True
                meta.setdefault("degraded_reason", status.last_error or status.state)
        except Exception as exc:
            log.debug("source-search: status enrichment failed", exc_info=True)
            response.setdefault("_meta", {}).setdefault("source_search", {}).update(
                {"status": "unknown", "status_error": type(exc).__name__}
            )
        try:
            await self._name_contained_definitions(response, divergence)
        except Exception:
            log.debug("source-search: definition naming failed", exc_info=True)
        return response

    async def _repo_id(self, session: Any) -> Any:
        """This repository's id, resolved once and remembered."""
        if self._repository_id is False:
            from repowise.server.mcp_server import _get_repo

            try:
                self._repository_id = (await _get_repo(session)).id
            except Exception:
                log.debug("source-search: no repository row to name definitions from")
                self._repository_id = None
        return self._repository_id

    async def _name_contained_definitions(self, response: dict[str, Any], divergence: Any) -> None:
        """Name nested definitions on served rows that still name none.

        The coordinator can only name a nested symbol that retrieval happened
        to surface as a rival for the same file. The residue is real: a file
        window names no symbol at all, and an enclosing chunk whose helper
        never entered the candidate set has nothing to be re-cited from. Both
        are exactly the rows whose span the reference-site index can describe,
        so this asks it — from the wiring, because the coordinator holds no
        database handle and is not going to start.

        **Silence is never evidence.** Reference sites are replaced repo-wide
        while chunks update incrementally, so the two can sit at different
        generations and an absent DEFINITION row means "not recorded", never
        "not there". Nothing here removes or contradicts a name the coordinator
        already established; it only adds, and a row it cannot speak for is
        left exactly as it arrived.

        Two gates decide whether it can speak for a row.

        **Freshness.** A file the working-tree check flags as modified or
        deleted has moved out from under both indexes, so no position in it can
        be asserted. An *unchecked* tree is treated the same way: the check not
        having run is not a finding of cleanliness, and enriching on it would
        be reading silence as evidence in the other direction.

        **Corroboration.** The name has to appear as a standalone identifier
        in the file's own text at the lines this row cites, read back from the
        working tree rather than from the stored snippet. Snippets are cut at
        ``STORED_SNIPPET_CHARS``, and that cut is a property of the storage
        format, not a statement about the code: a helper declared past it is
        no less real than one declared before it, so withholding it named the
        cap rather than the file. Reading the cited lines back off disk asks
        the question the row is actually making a claim about. The freshness
        gate above is what lets the two sources be exchanged at all — a file
        it has not flagged is one the working tree still matches — so this
        widens what can be proved without weakening the proof.

        What a reader gives up is being able to find every named symbol inside
        the snippet in front of them; a name can now sit past the cut. What the
        row asserts is unchanged, and it was never "this is visible above": it
        is that the definition starts inside the lines cited. When the file
        cannot be read, has nothing file-backed behind it, or no longer
        reaches those lines, the stored snippet corroborates exactly as it did
        before, so no row is named on weaker evidence than it would have been.
        """
        result_rows = [row for row in (response.get("results") or []) if isinstance(row, dict)]
        # Core's coordinator only emits names whose complete span is enclosed.
        # The refsite fallback below intentionally answers the different
        # declaration question: whether the definition starts inside the row.
        for row in result_rows:
            if row.get("contains_symbols"):
                row.setdefault("containment", "full_span_enclosure")

        if self._session_factory is None or divergence is None or not divergence.checked:
            return
        rows = [
            row
            for row in result_rows
            if not row.get("contains_symbols")
            and row.get("file")
            and isinstance(row.get("start_line"), int)
            and isinstance(row.get("end_line"), int)
            and row.get("snippet")
        ]
        diverged = frozenset(divergence.modified) | frozenset(divergence.deleted)
        rows = [row for row in rows if str(row["file"]) not in diverged]
        if not rows:
            return

        from sqlalchemy import func, select

        from repowise.core.persistence.database import get_session
        from repowise.core.refsites.schema import ReferenceSite
        from repowise.core.refsites.store import SqlReferenceSiteStore
        from repowise.core.refsites.taxonomy import ReferenceKind
        from repowise.core.source_search.coordinator import (
            MAX_CONTAINED_SYMBOLS,
            symbol_path_of,
        )

        async with get_session(self._session_factory) as session:
            repository_id = await self._repo_id(session)
            if not repository_id:
                return
            store = SqlReferenceSiteStore(session)
            for row in rows:
                file_path = str(row["file"])
                own = str(row.get("symbol_path") or "")
                sites = await store.definitions_in_range(
                    repository_id,
                    file_path,
                    int(row["start_line"]),
                    int(row["end_line"]),
                    limit=_CONTAINED_SITE_FETCH_LIMIT,
                )
                if len(sites) == _CONTAINED_SITE_FETCH_LIMIT:
                    total = int(
                        (
                            await session.execute(
                                select(func.count())
                                .select_from(ReferenceSite)
                                .where(
                                    ReferenceSite.repository_id == repository_id,
                                    ReferenceSite.file_path == file_path,
                                    ReferenceSite.kind == str(ReferenceKind.DEFINITION),
                                    ReferenceSite.start_line >= int(row["start_line"]),
                                    ReferenceSite.start_line <= int(row["end_line"]),
                                )
                            )
                        ).scalar_one()
                    )
                    if total > len(sites):
                        row["sites_truncated"] = True
                        row["sites_total"] = total
                # The row's own lines, read from the working tree, with the
                # stored snippet as the fallback when they cannot be read.
                corroborating = _live_slice(
                    self._repo,
                    file_path,
                    int(row["start_line"]),
                    int(row["end_line"]),
                ) or str(row["snippet"])
                names: list[str] = []
                for site in sites:
                    chain = symbol_path_of(site.target_symbol_id or "", file_path)
                    if not chain or chain == own:
                        continue
                    # The same nesting proof the coordinator applies: when the
                    # row names a symbol, what it contains must extend that
                    # name, or it is a neighbour rather than an inhabitant.
                    if own and not chain.startswith(f"{own}::"):
                        continue
                    # An empty name corroborates against anything: the pattern
                    # it compiles to is zero-width and matches at the first
                    # position it is tried. The column is not empty-constrained,
                    # so the check has to be made rather than assumed.
                    if not site.name or not _identifier_token(site.name).search(corroborating):
                        continue
                    if chain not in names:
                        names.append(chain)
                    if len(names) >= MAX_CONTAINED_SYMBOLS:
                        break
                if names:
                    row["contains_symbols"] = names
                    row["containment"] = "definition_start_in_span"

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


def _embedder_identity_mismatch(embedder: Any, stored: Any) -> str | None:
    """A description of how *embedder* disagrees with the manifest's, or None.

    Width alone does not make two corpora comparable. Federation ranks repos
    against each other on dense cosine, and a cosine is only a shared unit when
    one model produced both vectors — two 1536-wide models from different
    families pass the width check and then get compared as if their spaces were
    the same one. The manifest already records who wrote it, so ask.

    Disagreement has to be *proved*, not assumed from silence: a manifest
    written before these fields existed reads back as empty strings (see
    ``SourceIndexManifest.from_dict``), and refusing on an empty field would
    lock the lane out of every legacy index over a fact nobody recorded.
    Compared the way :mod:`~repowise.server.mcp_server.tool_index_status`
    already compares them, through ``identify_embedder`` — the same two sides,
    so the two answers cannot drift apart.
    """
    from repowise.core.source_search.manifest import identify_embedder

    try:
        live = identify_embedder(embedder)
    except Exception:  # pragma: no cover - identity is best-effort
        log.debug("source-search: could not identify the live embedder", exc_info=True)
        return None
    for field, live_value, stored_value in (
        ("provider", live.provider, stored.provider),
        ("model", live.model, stored.model),
    ):
        one, two = (live_value or "").strip().lower(), (stored_value or "").strip().lower()
        if one and two and one != two:
            return f"{field} {two!r} on disk, {one!r} live"
    return None


async def _wiki_tombstone_ids(repo_path: Path, session_factory: Any) -> frozenset[str]:
    """Current tombstoned wiki page ids for one coordinator's repository.

    This read runs once per federated query, before either wiki retriever sets
    its fixed candidate window. Updates can tombstone pages without changing
    the active source generation, so caching the answer with the coordinator
    would make the filter stale for exactly the interval it exists to close.
    """
    from sqlalchemy import select

    from repowise.core.persistence.crud import get_repository_by_path
    from repowise.core.persistence.database import get_session
    from repowise.core.persistence.models import Page

    async with get_session(session_factory) as session:
        repository = await get_repository_by_path(session, str(repo_path))
        if repository is None:
            # Construction succeeded from this repository's stores, but its
            # SQL identity cannot be proved. Raising disables the wiki legs;
            # an empty set would turn missing evidence into a claim of liveness.
            raise LookupError(f"repository row not found for {repo_path}")
        rows = await session.execute(
            select(Page.id).where(
                Page.repository_id == repository.id,
                Page.freshness_status == "tombstone",
            )
        )
        return frozenset(rows.scalars().all())


def _build(
    repo_path: Path | str,
    wiki_vectors: Any,
    wiki_fts: Any,
    session_factory: Any = None,
) -> Any:
    """A coordinator over *repo_path*, or None when it has no source index.

    The embedder is taken from the wiki vector store rather than resolved
    again, because the two corpora are only comparable when one embedder wrote
    both — the dense fusion is arithmetic on that assumption.

    *session_factory* is the wiki database this repository's reference sites
    and page freshness rows live in. When present, the coordinator checks the
    tombstone set before fixing each wiki leg's fetch window; it still receives
    a callback rather than a database handle, so storage wiring stays here.
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
            "source-search: configured embedder width does not match active generation (%s != %s)",
            getattr(embedder, "dimensions", None),
            manifest.embedder.dims,
        )
        return None
    if mismatch := _embedder_identity_mismatch(embedder, manifest.embedder):
        log.warning(
            "source-search: active generation at %s was embedded by a different model "
            "(%s), staying on the stock path",
            repo,
            mismatch,
        )
        return None

    generation = GenerationRef(manifest.generation_id, manifest.generation_sequence)

    try:
        source_fts = SourceFTSIndex(fts_path, generation=generation, read_only=True)
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
        manifest=manifest,
        # A wiki store without its page-status database cannot prove which
        # rows are still serveable. Keep the independent source lanes, but do
        # not admit an unverified wiki lane.
        wiki_vectors=wiki_vectors if session_factory is not None else None,
        wiki_fts=wiki_fts if session_factory is not None else None,
        wiki_tombstones=(
            (lambda: _wiki_tombstone_ids(repo, session_factory))
            if session_factory is not None
            else None
        ),
    )
    return _StatusCoordinator(coordinator, repo, source_vectors, source_fts, session_factory)


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
        # Retired rather than closed, for the reason ``_retired`` gives: this
        # one's caller is ``search_codebase`` itself, which awaits ``search()``
        # on a coordinator it took from here with nothing held in between.
        _retire(old)
        await _sweep_retired()
        generation = _manifest_key(repo_path)
        try:
            _mcp_coordinator = _build(
                repo_path, _state._vector_store, _state._fts, _state._session_factory
            )
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
        generation = _manifest_key(repo_path) if repo_path is not None else None
        existing = getattr(app_state, _REST_ATTR, None)
        if existing is not None and generation == getattr(app_state, _REST_GENERATION_ATTR, None):
            return existing
        if existing is not None:
            _retire(existing)
        await _sweep_retired()
        generation = _manifest_key(repo_path) if repo_path is not None else None
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
                getattr(app_state, "session_factory", None),
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
#: Identity, never equality — the job-lock work already paid for that lesson —
#: and held weakly rather than as an address (see ``_store_token``).
#:
#: Ordered and bounded, least-recently-used first. Every entry is an open
#: LanceDB table plus an open SQLite sidecar, so an unbounded map is one file
#: descriptor pair per repo ever queried, held for the life of the process —
#: which for a workspace server is every repo in it, forever.
_ctx_coordinators: OrderedDict[Path, tuple[Any, tuple[Any, ...], int]] = OrderedDict()

#: How many repos keep an open coordinator. Sized for a fan-out to stay warm
#: across a whole federated query (the workspaces this serves are single
#: digits) while still bounding the descriptors a long-lived server holds.
_CTX_CACHE_MAX = 8

#: One lock per repo path, not one global lock. The build has to be
#: serialised per repo — two concurrent first queries must not both open the
#: FTS sidecar — but a build for repo A has nothing to serialise against a
#: build for repo B, and a global lock turned a federated fan-out over N cold
#: repos into N sequential readiness timeouts.
#:
#: Created without an await between the lookup and the insert, so two
#: coroutines cannot mint two locks for one path. Bounded by the number of
#: distinct repo paths the process ever queries (a workspace's size); a lock
#: is a few bytes and is not a handle, so these are not swept with the
#: coordinators they guard.
_ctx_locks: dict[Path, asyncio.Lock] = {}

# Retired readers close after their final borrower releases them. Active
# requests own the memory bound; elapsed time never proves a reader is idle.
_retired: list[Any] = []
_borrowers: dict[int, int] = {}


@asynccontextmanager
async def coordinator_lease(coordinator: Any):
    """Keep a reader alive across awaits, including federation preparation."""
    key = id(coordinator)
    _borrowers[key] = _borrowers.get(key, 0) + 1
    try:
        yield coordinator
    finally:
        remaining = _borrowers[key] - 1
        if remaining:
            _borrowers[key] = remaining
        else:
            del _borrowers[key]
            await _sweep_retired()


def _store_token(store: Any) -> Any:
    """A handle on the wiki store an entry was built from, owning nothing.

    Weak, not ``id()``. The identity check exists because a reloaded context
    arrives with fresh store objects while the cached coordinator still holds
    the closed ones — but an ``int`` address only answers that question while
    the object it named is alive. Once the old context is disposed and freed,
    CPython hands the very next allocation the same address, and the check
    then reports a brand-new store as the one already built from, serving a
    coordinator over stores that were closed with the context.

    A dead referent compares equal to nothing, which is the right answer: an
    entry whose store has been collected is stale by definition. A store that
    cannot be weakly referenced falls back to the address, which is no worse
    than what this replaces.
    """
    if store is None:
        return None
    try:
        return weakref.ref(store)
    except TypeError:  # pragma: no cover - every real store supports weakref
        return id(store)


def _same_store(token: Any, store: Any) -> bool:
    """Whether a cache entry's token still names *store*."""
    if token is None or store is None:
        return token is None and store is None
    if isinstance(token, weakref.ref):
        return token() is store
    return token == id(store)


def _lock_for(repo_path: Path) -> asyncio.Lock:
    """The build lock for one repo path, created on first use."""
    lock = _ctx_locks.get(repo_path)
    if lock is None:
        lock = _ctx_locks[repo_path] = asyncio.Lock()
    return lock


def _retire(coordinator: Any) -> None:
    """Hand a coordinator over for closing later (see ``_retired``)."""
    if coordinator is not None and not any(item is coordinator for item in _retired):
        _retired.append(coordinator)


async def _sweep_retired() -> None:
    """Close retired coordinators with no remaining borrower.

    Claims its victims before the first ``await`` — the slice and the delete
    are one uninterruptible step — so two concurrent sweeps cannot close the
    same coordinator twice.
    """
    if not _retired:
        return
    doomed = [item for item in _retired if not _borrowers.get(id(item))]
    _retired[:] = [item for item in _retired if _borrowers.get(id(item))]
    for coordinator in doomed:
        try:
            await coordinator.close()
        except Exception:
            log.debug("source-search: closing a retired coordinator failed", exc_info=True)


def _evict(repo_path: Path) -> None:
    """Drop one repo's cached coordinator, retiring it for close."""
    entry = _ctx_coordinators.pop(repo_path, None)
    if entry is not None:
        _retire(entry[0])


async def context_coordinator(ctx: Any) -> Any:
    """A coordinator over one workspace repo context, or ``None`` (fail-soft).

    The single-repo entry point reads the MCP module globals; a workspace has
    no meaningful globals to read, so this one takes the ``RepoContext``
    itself. Same doctrine otherwise: anything missing — no source index for
    this member repo, stores still loading, a mock embedder — returns ``None``
    and the caller falls through to the stock path for that repo.

    Three things are deliberate about the order of operations here.

    **The readiness wait happens before the build lock.** Waiting is a
    read of the context's own event and serialising it buys nothing, while
    holding a lock across it costs every other repo in a fan-out the full
    timeout in turn.

    **The manifest key is re-read inside the lock.** The one read before it is
    a cache probe and nothing more. A coroutine that waited on the lock woke up
    into a world where the repo may have republished and another request may
    already have built for the newer generation; deciding what to evict on the
    key it read before it slept would pop and close that fresher coordinator
    and then cache its own build under a stale key.

    **A replaced coordinator is retired, not closed** — see ``_retired``.
    """
    repo_path = Path(ctx.path)
    await _sweep_retired()

    generation = _manifest_key(repo_path)
    if generation is None:
        # The source index was removed. Evicting is the point: without it the
        # entry outlives the index it reads, pinning open handles onto a
        # generation that no longer exists for as long as the process runs.
        if repo_path in _ctx_coordinators:
            async with _lock_for(repo_path):
                _evict(repo_path)
            await _sweep_retired()
        return None

    wiki_vs = getattr(ctx, "vector_store", None)
    cached = _ctx_coordinators.get(repo_path)
    if cached is not None and cached[1] == generation and _same_store(cached[2], wiki_vs):
        _ctx_coordinators.move_to_end(repo_path)
        return cached[0]

    # The registry loads real vector stores in the background and repoints
    # ctx.vector_store when done; build only from the settled store.
    ready = getattr(ctx, "vector_store_ready", None)
    if ready is not None and not ready.is_set():
        try:
            await asyncio.wait_for(ready.wait(), timeout=_READY_TIMEOUT)
        except TimeoutError:
            log.debug(
                "source-search: vector stores for %s still loading, staying on the stock path",
                getattr(ctx, "alias", repo_path),
            )
            return None

    async with _lock_for(repo_path):
        generation = _manifest_key(repo_path)
        if generation is None:
            _evict(repo_path)
            return None
        wiki_vs = getattr(ctx, "vector_store", None)
        cached = _ctx_coordinators.get(repo_path)
        if cached is not None and cached[1] == generation and _same_store(cached[2], wiki_vs):
            _ctx_coordinators.move_to_end(repo_path)
            return cached[0]
        _evict(repo_path)
        # Reclaim before publishing the replacement. An await after inserting
        # it would let another request retire/close it before this caller can
        # take its lease. Re-read publication after the asynchronous close.
        await _sweep_retired()
        generation = _manifest_key(repo_path)
        if generation is None:
            return None
        wiki_vs = getattr(ctx, "vector_store", None)
        try:
            built = _build(
                repo_path,
                wiki_vs,
                getattr(ctx, "fts", None),
                getattr(ctx, "session_factory", None),
            )
        except Exception:
            log.debug(
                "source-search: workspace coordinator construction failed for %s",
                getattr(ctx, "alias", repo_path),
                exc_info=True,
            )
            return None
        if built is None:
            return None
        _ctx_coordinators[repo_path] = (built, generation, _store_token(wiki_vs))
        _ctx_coordinators.move_to_end(repo_path)
        while len(_ctx_coordinators) > _CTX_CACHE_MAX:
            _, evicted = _ctx_coordinators.popitem(last=False)
            _retire(evicted[0])
    # No await between publication and delivery. LRU retirements close on the
    # next acquisition or when this request releases its reader lease.
    return built


def reset_for_tests() -> None:
    """Forget the MCP process caches, closing what they held.

    Best-effort on the closing, because ``close()`` is a coroutine and this is
    not: a caller already inside a loop gets the close scheduled on it rather
    than awaited. Dropping the references without closing leaks a LanceDB table
    and an SQLite sidecar per cached repo across a test session, which is how a
    suite ends up failing on file handles rather than on its assertions.

    The per-repo locks go too. An :class:`asyncio.Lock` binds to the loop that
    first contends it, and pytest gives each test its own loop.
    """
    global _mcp_coordinator, _mcp_generation
    for entry in _ctx_coordinators.values():
        _retire(entry[0])
    _ctx_coordinators.clear()
    _ctx_locks.clear()
    if _mcp_coordinator is not None:
        _retire(_mcp_coordinator)
    _mcp_coordinator = None
    _mcp_generation = None
    retired, _retired[:] = list(_retired), []
    for coordinator in retired:
        _close_detached(coordinator)


def _close_detached(coordinator: Any) -> None:
    """Close a coordinator from synchronous code, however we can."""
    coro = coordinator.close()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            asyncio.run(coro)
        except Exception:
            log.debug("source-search: detached close failed", exc_info=True)
        return
    task = loop.create_task(coro)
    task.add_done_callback(lambda done: done.cancelled() or done.exception())
