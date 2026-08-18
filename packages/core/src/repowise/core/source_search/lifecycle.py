"""Crash-safe source-index reconciliation across SQL, FTS5, and LanceDB.

``wiki.db`` owns the durable change queue.  FTS and Lance stage versioned rows
for a future generation, readers stay pinned to the manifest's old generation,
and publication is one atomic manifest replace after both stores verify.  A
retry uses the same generation id and row keys, so every interruption point is
convergent rather than compensating.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from filelock import FileLock, Timeout
from sqlalchemy import select

from repowise.core.providers.embedding.base import Embedder

from .chunks import (
    MAX_WINDOW_FILE_BYTES,
    SourceChunk,
    SymbolRecord,
    build_symbol_chunk,
    iter_file_windows,
    looks_binary,
    window_eligible,
)
from .fts import SourceFTSIndex, generation_fts_path
from .generation import LEGACY_GENERATION, GenerationRef, vector_table_for_recipe
from .manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    corpus_hash_entries,
    default_manifest_path,
    read_manifest,
    recipe_fingerprint,
    write_manifest,
)
from .outbox import (
    BLOCKED,
    BUILDING,
    PENDING,
    PUBLISHED,
    READY,
    SourceChange,
    SourceUpdateRecord,
    enqueue_full_update,
    load_unpublished_updates,
    mark_update_state,
    mark_updates_ready,
)
from .vector_store import SourceChunkVectorStore

__all__ = [
    "SourceIndexDeferredError",
    "SourceLifecycleResult",
    "reconcile_source_index",
    "record_source_index_error",
]

log = structlog.get_logger(__name__)

_BATCH_SIZE = 16
_LOCK_FILENAME = "reconcile.lock"
_FAILURE_STAGES = frozenset(
    {
        "after_claim",
        "after_chunks",
        "after_fts",
        "after_vector",
        "after_verify",
        "after_ready",
        "after_manifest",
        "after_publish",
    }
)


class SourceIndexDeferredError(RuntimeError):
    """The durable queue is intact, but current inputs are not safe to publish."""


@dataclass(frozen=True, slots=True)
class SourceLifecycleResult:
    status: str
    generation_id: str | None
    generation_sequence: int | None
    updates_consumed: int
    embedded: int
    reused: int
    chunks: int
    stale_files: int
    total_seconds: float
    load_seconds: float = 0.0
    embed_seconds: float = 0.0
    write_seconds: float = 0.0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _Plan:
    target: SourceUpdateRecord
    consumed: tuple[SourceUpdateRecord, ...]
    full: bool
    close_paths: tuple[str, ...]
    replace: tuple[SourceChange, ...]
    stale_files: dict[str, str]


def _inject(injector: Callable[[str], None] | None, stage: str) -> None:
    if stage not in _FAILURE_STAGES:
        raise ValueError(f"unknown source-index failure stage: {stage}")
    if injector is not None:
        injector(stage)


def _manifest_generation(manifest: SourceIndexManifest | None) -> GenerationRef:
    if manifest is None:
        return LEGACY_GENERATION
    return GenerationRef(manifest.generation_id or "legacy", manifest.generation_sequence)


def _relative_to_repo(repo: Path, path: Path) -> str:
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return str(path)


async def _repository_id(session: Any, repo: Path) -> str:
    from repowise.core.persistence.models import Repository

    rows = list((await session.execute(select(Repository))).scalars().all())
    for row in rows:
        try:
            if row.local_path and Path(row.local_path).resolve() == repo:
                return str(row.id)
        except OSError:
            continue
    if len(rows) == 1:
        return str(rows[0].id)
    raise RuntimeError(f"No indexed repository matching {repo} in the wiki database")


async def _load_work(repo: Path, db_url: str | None) -> tuple[str, list[SourceUpdateRecord]]:
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        get_session,
        init_db,
        resolve_db_url,
    )

    engine = create_engine(db_url or resolve_db_url(repo))
    try:
        await init_db(engine)
        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            repository_id = await _repository_id(session, repo)
            updates = await load_unpublished_updates(session, repository_id)
        return repository_id, updates
    finally:
        await engine.dispose()


async def _enqueue_manual_full(repo: Path, db_url: str | None) -> None:
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        get_session,
        init_db,
        resolve_db_url,
    )

    engine = create_engine(db_url or resolve_db_url(repo))
    try:
        await init_db(engine)
        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            repository_id = await _repository_id(session, repo)
            await enqueue_full_update(session, repository_id, repo)
    finally:
        await engine.dispose()


async def _set_state(
    repo: Path,
    db_url: str | None,
    generation_ids: list[str],
    state: str,
    *,
    error: str | None = None,
    artifact: dict[str, Any] | None = None,
) -> None:
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        get_session,
        resolve_db_url,
    )

    engine = create_engine(db_url or resolve_db_url(repo))
    try:
        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            if state == READY:
                if artifact is None:
                    raise ValueError("a ready source generation requires its artifact manifest")
                await mark_updates_ready(session, generation_ids, artifact)
            else:
                await mark_update_state(session, generation_ids, state, error=error)
    finally:
        await engine.dispose()


async def record_source_index_error(
    repo_path: Path | str,
    error: str,
    *,
    db_url: str | None = None,
) -> None:
    """Attach a host/configuration failure to durable pending work."""

    repo = Path(repo_path).resolve()
    _repository, updates = await _load_work(repo, db_url)
    pending_ids = [
        update.generation_id
        for update in updates
        if update.state in {PENDING, BUILDING}
    ]
    if pending_ids:
        await _set_state(repo, db_url, pending_ids, PENDING, error=error)


def _select_plan(
    updates: Sequence[SourceUpdateRecord],
    manifest: SourceIndexManifest | None,
    fingerprint: str,
    *,
    active_invalid: bool = False,
) -> _Plan | None:
    active_sequence = manifest.generation_sequence if manifest is not None else 0
    outstanding = [update for update in updates if update.sequence > active_sequence]
    processable = [update for update in outstanding if update.upstream_ready and update.state != BLOCKED]
    if not processable:
        return None
    target = processable[-1]
    consumed = tuple(update for update in outstanding if update.sequence <= target.sequence)
    full = bool(
        manifest is None
        or manifest.generation_sequence <= 0
        or manifest.recipe_fingerprint != fingerprint
        or active_invalid
        or any(update.mode == "full" or not update.upstream_ready for update in consumed)
    )
    if full:
        return _Plan(target, consumed, True, (), (), {})

    if manifest is None:  # Defensive narrowing: ``full`` is true in this case.
        raise AssertionError("incremental source-index plan requires an active manifest")

    close: set[str] = set()
    replacements: dict[str, SourceChange] = {}
    stale = dict(manifest.stale_files)
    for update in consumed:
        for change in update.changes:
            if change.path == "__full__":
                continue
            successful = change.parse_state in {"parsed", "window"}
            if change.status == "renamed" and change.old_path:
                if successful:
                    close.add(change.old_path)
                    stale.pop(change.old_path, None)
                else:
                    stale[change.old_path] = f"rename target {change.path} did not parse"
            if change.status == "deleted":
                close.add(change.path)
                replacements.pop(change.path, None)
                stale.pop(change.path, None)
            elif successful:
                close.add(change.path)
                replacements[change.path] = change
                stale.pop(change.path, None)
            elif change.parse_state == "failed":
                replacements.pop(change.path, None)
                stale[change.path] = "latest parse failed; previous source chunks retained"
            else:
                close.add(change.path)
                replacements.pop(change.path, None)
                stale.pop(change.path, None)
    return _Plan(
        target=target,
        consumed=consumed,
        full=False,
        close_paths=tuple(sorted(close)),
        replace=tuple(replacements[path] for path in sorted(replacements)),
        stale_files=stale,
    )


async def _load_symbols_for_paths(
    repo: Path,
    db_url: str | None,
    file_paths: Sequence[str] | None = None,
) -> list[SymbolRecord]:
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        resolve_db_url,
    )
    from repowise.core.persistence.models import WikiSymbol

    engine = create_engine(db_url or resolve_db_url(repo))
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            repository_id = await _repository_id(session, repo)
            statement = select(WikiSymbol).where(WikiSymbol.repository_id == repository_id)
            if file_paths is not None:
                if not file_paths:
                    return []
                statement = statement.where(WikiSymbol.file_path.in_(file_paths))
            rows = list(
                (
                    await session.execute(
                        statement.order_by(
                            WikiSymbol.file_path, WikiSymbol.start_line, WikiSymbol.symbol_id
                        )
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()
    return [
        SymbolRecord(
            symbol_id=row.symbol_id,
            file_path=row.file_path,
            name=row.name,
            qualified_name=row.qualified_name,
            kind=row.kind,
            signature=row.signature or "",
            docstring=row.docstring,
            start_line=row.start_line,
            end_line=row.end_line,
            language=row.language or "",
        )
        for row in rows
    ]


def _decode(data: bytes) -> str:
    from repowise.core.ingestion.source_text import decode_source

    return decode_source(data)


def _read_changed_bytes(repo: Path, change: SourceChange) -> bytes:
    path = repo / change.path
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SourceIndexDeferredError(
            f"{change.path} changed again or is unreadable: {exc}"
        ) from exc
    from repowise.core.ingestion.models import compute_content_hash

    actual = compute_content_hash(data)
    if change.content_hash and actual != change.content_hash:
        raise SourceIndexDeferredError(
            f"{change.path} changed after its SQL update; waiting for the next saved-file event"
        )
    return data


def _chunks_for_replacements(
    repo: Path,
    changes: Sequence[SourceChange],
    symbols: Sequence[SymbolRecord],
) -> list[SourceChunk]:
    by_file: dict[str, list[SymbolRecord]] = defaultdict(list)
    for symbol in symbols:
        by_file[symbol.file_path].append(symbol)

    chunks: list[SourceChunk] = []
    for change in changes:
        data = _read_changed_bytes(repo, change)
        if looks_binary(data) or len(data) > MAX_WINDOW_FILE_BYTES:
            continue
        text = _decode(data)
        lines = text.splitlines()
        file_symbols = by_file.get(change.path, [])
        symbol_chunks = [build_symbol_chunk(symbol, lines) for symbol in file_symbols]
        chunks.extend(symbol_chunks)
        if window_eligible(change.path, indexed_symbols=len(symbol_chunks)):
            chunks.extend(iter_file_windows(change.path, text))
    return chunks


async def _full_chunks(repo: Path, db_url: str | None) -> list[SourceChunk]:
    # Reuse the A1 corpus owner.  These helpers are deliberately pure after the
    # symbol read, so lifecycle does not grow a second tracked-file recipe.
    from .indexer import _build_symbol_chunks, _build_window_chunks

    symbols = await _load_symbols_for_paths(repo, db_url)
    unreadable = sorted(
        {
            symbol.file_path
            for symbol in symbols
            if not (repo / symbol.file_path).is_file()
        }
    )
    if unreadable:
        raise SourceIndexDeferredError(
            "source files referenced by persisted symbols are unreadable: "
            + ", ".join(unreadable[:5])
        )
    symbol_chunks = _build_symbol_chunks(repo, symbols)
    return [*symbol_chunks, *_build_window_chunks(repo, symbol_chunks)]


async def _embed_chunks(
    embedder: Embedder,
    identity: EmbedderIdentity,
    chunks: Sequence[SourceChunk],
    reusable: dict[str, list[float]],
    *,
    batch_size: int,
) -> tuple[list[tuple[SourceChunk, list[float]]], int, int]:
    from .indexer import _embed_with_retry

    vectors: dict[str, list[float]] = {}
    missing: dict[str, SourceChunk] = {}
    for chunk in chunks:
        reused = reusable.get(chunk.content_hash)
        if reused is not None and len(reused) == identity.dims:
            vectors[chunk.content_hash] = reused
        else:
            missing.setdefault(chunk.content_hash, chunk)

    pending = list(missing.values())
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [f"{identity.document_prefix}{chunk.text}" for chunk in batch]
        fresh = await _embed_with_retry(embedder, texts)
        for chunk, vector in zip(batch, fresh, strict=True):
            if len(vector) != identity.dims:
                raise RuntimeError(
                    f"Embedder returned {len(vector)} dimensions; recipe requires {identity.dims}"
                )
            vectors[chunk.content_hash] = [float(value) for value in vector]

    items = [(chunk, vectors[chunk.content_hash]) for chunk in chunks]
    embedded = len(pending)
    return items, embedded, len(chunks) - embedded


def _head_commit(repo: Path) -> str | None:
    from .indexer import _head_commit as read_head

    return read_head(repo)


async def _verify_candidate(
    repo: Path,
    embedder: Embedder,
    candidate: SourceIndexManifest,
) -> bool:
    generation = _manifest_generation(candidate)
    fts_path = repo / candidate.fts_path
    if not fts_path.is_file():
        return False
    try:
        with SourceFTSIndex(fts_path, generation=generation) as fts:
            if not fts.verify_generation(
                generation,
                recipe_fingerprint=candidate.recipe_fingerprint,
                expected_count=candidate.symbol_chunks + candidate.file_window_chunks,
            ):
                return False
        store = SourceChunkVectorStore(
            str(repo / ".repowise" / "lancedb"),
            embedder=embedder,
            table_name=candidate.lance_table,
            generation=generation,
        )
        try:
            return await store.verify_generation(
                generation,
                recipe_fingerprint=candidate.recipe_fingerprint,
                expected_count=candidate.symbol_chunks + candidate.file_window_chunks,
            )
        finally:
            await store.close()
    except Exception:
        log.warning(
            "source_index_active_generation_invalid",
            generation=candidate.generation_id,
            exc_info=True,
        )
        return False


async def reconcile_source_index(
    repo_path: Path | str,
    *,
    embedder: Embedder,
    embedder_identity: EmbedderIdentity,
    db_url: str | None = None,
    force_full: bool = False,
    batch_size: int = _BATCH_SIZE,
    failure_injector: Callable[[str], None] | None = None,
) -> SourceLifecycleResult:
    """Drain durable changes under one repository-wide publication lock.

    FTS and LanceDB have independent transactions, so two reconcilers must
    never stage the same generation concurrently.  A filesystem lock is both
    process-safe and crash-safe: the kernel releases it when its owner exits,
    unlike a durable ``building`` bit that needs a timeout before retry.
    """

    if batch_size < 1:
        raise ValueError("source-index embedding batch size must be positive")
    started = time.perf_counter()
    repo = Path(repo_path).resolve()
    lock_path = repo / ".repowise" / "source_search" / _LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(lock_path, timeout=0)
    try:
        lock.acquire()
    except Timeout:
        active = read_manifest(default_manifest_path(repo))
        return SourceLifecycleResult(
            status="busy",
            generation_id=active.generation_id if active else None,
            generation_sequence=active.generation_sequence if active else None,
            updates_consumed=0,
            embedded=0,
            reused=0,
            chunks=(active.symbol_chunks + active.file_window_chunks) if active else 0,
            stale_files=len(active.stale_files) if active else 0,
            total_seconds=round(time.perf_counter() - started, 3),
            error="another source-index reconciler is already running",
        )
    try:
        return await _reconcile_source_index_unlocked(
            repo,
            embedder=embedder,
            embedder_identity=embedder_identity,
            db_url=db_url,
            force_full=force_full,
            batch_size=batch_size,
            failure_injector=failure_injector,
        )
    finally:
        lock.release()


async def _reconcile_source_index_unlocked(
    repo_path: Path | str,
    *,
    embedder: Embedder,
    embedder_identity: EmbedderIdentity,
    db_url: str | None = None,
    force_full: bool = False,
    batch_size: int = _BATCH_SIZE,
    failure_injector: Callable[[str], None] | None = None,
) -> SourceLifecycleResult:
    """Drain durable changes and atomically publish one source generation."""

    started = time.perf_counter()
    repo = Path(repo_path).resolve()
    fingerprint = recipe_fingerprint(embedder_identity)
    if force_full:
        await _enqueue_manual_full(repo, db_url)

    _repository, updates = await _load_work(repo, db_url)
    active = read_manifest(default_manifest_path(repo))

    if active is not None and active.recipe_fingerprint != fingerprint and not updates:
        await _enqueue_manual_full(repo, db_url)
        _repository, updates = await _load_work(repo, db_url)

    managed_active = active is not None and active.generation_sequence > 0
    active_invalid = bool(
        managed_active and active is not None and not await _verify_candidate(repo, embedder, active)
    )
    if active_invalid and not updates:
        await _enqueue_manual_full(repo, db_url)
        _repository, updates = await _load_work(repo, db_url)

    # Recovery after a crash just after manifest publication: publication is
    # already complete; only the relational bookkeeping lagged.
    if active is not None:
        already = [
            update.generation_id
            for update in updates
            if update.sequence <= active.generation_sequence
        ]
        if already:
            await _set_state(repo, db_url, already, PUBLISHED)
            _repository, updates = await _load_work(repo, db_url)

    plan = _select_plan(updates, active, fingerprint, active_invalid=active_invalid)
    if plan is None:
        blocked = [update for update in updates if not update.upstream_ready]
        return SourceLifecycleResult(
            status="degraded" if blocked else "current",
            generation_id=active.generation_id if active else None,
            generation_sequence=active.generation_sequence if active else None,
            updates_consumed=0,
            embedded=0,
            reused=0,
            chunks=(active.symbol_chunks + active.file_window_chunks) if active else 0,
            stale_files=len(active.stale_files) if active else 0,
            total_seconds=round(time.perf_counter() - started, 3),
            error=blocked[-1].last_error if blocked else None,
        )

    consumed_ids = [update.generation_id for update in plan.consumed]
    target = plan.target.ref

    # A verified ready generation is publishable without rebuilding.  This is
    # the crash window between cross-store verification and manifest replace.
    if plan.target.state == READY and plan.target.artifact:
        candidate = SourceIndexManifest.from_dict(plan.target.artifact)
        if candidate.recipe_fingerprint == fingerprint and await _verify_candidate(
            repo, embedder, candidate
        ):
            write_manifest(default_manifest_path(repo), candidate)
            _inject(failure_injector, "after_manifest")
            await _set_state(repo, db_url, consumed_ids, PUBLISHED)
            _inject(failure_injector, "after_publish")
            return SourceLifecycleResult(
                status="published",
                generation_id=candidate.generation_id,
                generation_sequence=candidate.generation_sequence,
                updates_consumed=len(consumed_ids),
                embedded=0,
                reused=0,
                chunks=candidate.symbol_chunks + candidate.file_window_chunks,
                stale_files=len(candidate.stale_files),
                total_seconds=round(time.perf_counter() - started, 3),
            )

    await _set_state(repo, db_url, consumed_ids, BUILDING)
    _inject(failure_injector, "after_claim")

    parent = _manifest_generation(active)
    same_recipe = bool(
        managed_active
        and not active_invalid
        and active is not None
        and active.recipe_fingerprint == fingerprint
    )
    table_name = (
        active.lance_table
        if same_recipe and active is not None
        else (
            f"{vector_table_for_recipe(fingerprint)}_{target.generation_id[:8]}"
            if active_invalid
            else vector_table_for_recipe(fingerprint)
        )
    )
    lance_dir = repo / ".repowise" / "lancedb"
    lance_dir.mkdir(parents=True, exist_ok=True)
    active_store = SourceChunkVectorStore(
        str(lance_dir),
        embedder=embedder,
        table_name=table_name,
        generation=parent,
    )
    prior_store = active_store
    if active is not None and active.lance_table != table_name:
        prior_store = SourceChunkVectorStore(
            str(lance_dir),
            embedder=embedder,
            table_name=active.lance_table,
            generation=parent,
        )

    try:
        load_started = time.perf_counter()
        if plan.full:
            chunks = await _full_chunks(repo, db_url)
            stale_files: dict[str, str] = {}
        else:
            close_paths = list(plan.close_paths)
            symbols = await _load_symbols_for_paths(
                repo, db_url, [change.path for change in plan.replace]
            )
            chunks = _chunks_for_replacements(repo, plan.replace, symbols)
            stale_files = dict(plan.stale_files)
        if not chunks and (plan.full or await active_store.count() == 0):
            raise SourceIndexDeferredError("the reconciled source corpus is empty")
        _inject(failure_injector, "after_chunks")
        load_seconds = time.perf_counter() - load_started

        embed_started = time.perf_counter()
        try:
            reusable = await prior_store.vectors_by_content_hash() if same_recipe else {}
        except Exception:
            log.warning("source_index_vector_reuse_skipped", exc_info=True)
            reusable = {}
        items, embedded, reused = await _embed_chunks(
            embedder,
            embedder_identity,
            chunks,
            reusable,
            batch_size=batch_size,
        )
        embed_seconds = time.perf_counter() - embed_started
        # A later target can coalesce updates whose earlier target was staged
        # but never published.  Sequence visibility alone would make those
        # abandoned rows appear at the later sequence, so remove each
        # superseded generation before computing or staging the candidate.
        superseded = [update.ref for update in plan.consumed if update.ref != target]
        for generation in superseded:
            await active_store.rollback_generation(generation)

        if plan.full:
            try:
                close_paths = await active_store.active_file_paths()
            except Exception:
                # An incompatible pre-A2 table is rebuild-only.  `_ensure_table`
                # below recreates it after the full corpus has been embedded.
                close_paths = []
        active_count = await active_store.count()
        closing_count = await active_store.count_for_files(close_paths)
        vector_expected = active_count - closing_count + len(chunks)

        write_started = time.perf_counter()
        fts_path = generation_fts_path(repo)
        with SourceFTSIndex(fts_path, generation=parent) as fts:
            for generation in superseded:
                fts.rollback_generation(generation)
            fts_close_paths = fts.active_file_paths() if plan.full else close_paths
            fts_expected = (
                fts.count() - fts.count_for_files(fts_close_paths) + len(chunks)
            )
            fts.stage_generation(
                target,
                close_paths=fts_close_paths,
                chunks=chunks,
                recipe_fingerprint=fingerprint,
            )
            if not fts.verify_generation(
                target,
                recipe_fingerprint=fingerprint,
                expected_count=fts_expected,
            ):
                raise RuntimeError("FTS generation verification failed")
        _inject(failure_injector, "after_fts")

        await active_store.stage_generation(
            target,
            close_paths=close_paths,
            items=items,
            recipe_fingerprint=fingerprint,
            expected_count=vector_expected,
        )
        _inject(failure_injector, "after_vector")

        inventory = await active_store.inventory(sequence=target.sequence)
        if len(inventory.entries) != vector_expected:
            raise RuntimeError("Lance inventory disagrees with the staged generation count")
        if not await active_store.verify_generation(
            target,
            recipe_fingerprint=fingerprint,
            expected_count=vector_expected,
        ):
            raise RuntimeError("Lance generation verification failed")
        if fts_expected != vector_expected:
            raise RuntimeError(
                "Cross-store generation count mismatch: "
                f"FTS={fts_expected}, Lance={vector_expected}"
            )
        _inject(failure_injector, "after_verify")

        candidate = SourceIndexManifest(
            recipe_fingerprint=fingerprint,
            corpus_hash=corpus_hash_entries(inventory.entries),
            symbol_chunks=inventory.symbol_chunks,
            file_window_chunks=inventory.file_window_chunks,
            files_covered=inventory.files_covered,
            indexed_commit=_head_commit(repo),
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            embedder=embedder_identity,
            generation_id=target.generation_id,
            generation_sequence=target.sequence,
            lance_table=table_name,
            fts_path=_relative_to_repo(repo, fts_path),
            stale_files=stale_files,
        )
        await _set_state(
            repo,
            db_url,
            consumed_ids,
            READY,
            artifact=candidate.to_dict(),
        )
        _inject(failure_injector, "after_ready")

        write_manifest(default_manifest_path(repo), candidate)
        _inject(failure_injector, "after_manifest")
        await _set_state(repo, db_url, consumed_ids, PUBLISHED)
        _inject(failure_injector, "after_publish")
        write_seconds = time.perf_counter() - write_started
        return SourceLifecycleResult(
            status="published",
            generation_id=target.generation_id,
            generation_sequence=target.sequence,
            updates_consumed=len(consumed_ids),
            embedded=embedded,
            reused=reused,
            chunks=vector_expected,
            stale_files=len(stale_files),
            total_seconds=round(time.perf_counter() - started, 3),
            load_seconds=round(load_seconds, 3),
            embed_seconds=round(embed_seconds, 3),
            write_seconds=round(write_seconds, 3),
        )
    except Exception as exc:
        # The manifest still points at the prior generation unless the failure
        # happened just after its atomic replace.  In both cases retry is safe:
        # startup first recognizes an already-active generation, otherwise the
        # deterministic staged row ids are rebuilt in place.
        await _set_state(repo, db_url, consumed_ids, PENDING, error=str(exc))
        log.warning(
            "source_index_reconcile_deferred",
            generation=target.generation_id,
            error=str(exc),
        )
        raise
    finally:
        await active_store.close()
        if prior_store is not active_store:
            await prior_store.close()
