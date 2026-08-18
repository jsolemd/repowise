"""Building the source corpus: symbols, then windows, then both stores.

One pass, full rebuild. Incremental updates are a later unit, and doing them
badly is worse than not doing them: a corpus that is half one recipe and half
another ranks unpredictably and gives no signal that it is doing so. So both
tables are dropped and rewritten every run, and the manifest is written last —
which makes its presence, and only its presence, mean "this index is complete".

The one thing carried across a rebuild is *vectors*. Embedding 8,000 chunks is
the expensive half and almost none of them changed, so the stored vectors are
read into memory before the table is dropped and reused for any chunk whose
text still hashes the same, provided the previous manifest says they were
computed by the recipe in force now.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

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
from .fts import SourceFTSIndex, default_fts_path
from .manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    corpus_hash,
    default_manifest_path,
    read_manifest,
    recipe_fingerprint,
    write_manifest,
)
from .vector_store import SourceChunkVectorStore

__all__ = ["EMBED_BATCH_SIZE", "SourceIndexResult", "build_source_index"]

if TYPE_CHECKING:  # the ORM pulls SQLAlchemy; a refusing CLI must not pay for it
    from repowise.core.persistence.models import Repository

log = structlog.get_logger(__name__)

#: Texts per embedder request. Matches what the page store already sends and
#: what a local Ollama endpoint handles without queueing.
EMBED_BATCH_SIZE = 16

#: A batch is retried this many times before the run gives up. The embedder
#: providers have no retry of their own, and a local endpoint under load
#: refuses a request rather than queueing it — one transient refusal should not
#: cost an eight-thousand-chunk rebuild.
_EMBED_ATTEMPTS = 4
_EMBED_BACKOFF_SECONDS = 0.25

#: Chunks buffered before a LanceDB write, decoupled from the embed batch on
#: purpose. Every write is one Lance transaction — a fragment, a version
#: manifest and a transaction record — so writing per 16-chunk embed batch cost
#: 513 fragments and 16 MiB of manifests for an 8,200-chunk index, against
#: 36 MiB of actual data. Flushing per 512 puts that back under a megabyte.
_LANCE_WRITE_BATCH = 512

#: How often the run says where it is. Chunks, not batches, so the cadence does
#: not change if the batch size does.
_PROGRESS_EVERY = 500


@dataclass(frozen=True, slots=True)
class SourceIndexResult:
    """What one build did, for the CLI to print and a test to assert on."""

    symbol_chunks: int
    file_window_chunks: int
    files_covered: int
    embedded: int
    reused: int
    corpus_hash: str
    recipe_fingerprint: str
    indexed_commit: str | None
    manifest_path: str
    lancedb_path: str
    fts_path: str
    load_seconds: float
    embed_seconds: float
    write_seconds: float
    total_seconds: float


async def build_source_index(
    repo_path: Path | str,
    *,
    embedder: Embedder,
    embedder_identity: EmbedderIdentity,
    db_url: str | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> SourceIndexResult:
    """Rebuild *repo_path*'s source-chunk corpus from its wiki index and worktree.

    *embedder_identity* is what goes in the manifest and the recipe
    fingerprint. It is passed rather than derived because an ``Embedder`` is a
    structural protocol with no provider or model on it, and guessing from the
    class name is how a fingerprint ends up not describing the thing it
    fingerprints.
    """
    started = time.perf_counter()
    repo = Path(repo_path).resolve()

    load_started = time.perf_counter()
    symbols, indexed_commit = await _load_symbols(repo, db_url)
    symbol_chunks = _build_symbol_chunks(repo, symbols)
    window_chunks = _build_window_chunks(repo, symbol_chunks)
    load_seconds = time.perf_counter() - load_started

    chunks = [*symbol_chunks, *window_chunks]
    if not chunks:
        raise RuntimeError(
            f"No source chunks for {repo}. Index the repository first "
            "('repowise init') so wiki_symbols is populated."
        )
    files_covered = len({chunk.file_path for chunk in chunks})
    fingerprint = recipe_fingerprint(embedder_identity)
    corpus = corpus_hash(chunks)

    log.info(
        "source_index_corpus_built",
        repo=str(repo),
        symbol_chunks=len(symbol_chunks),
        file_window_chunks=len(window_chunks),
        files_covered=files_covered,
    )

    manifest_path = default_manifest_path(repo)
    fts_path = default_fts_path(repo)
    lancedb_path = repo / ".repowise" / "lancedb"
    lancedb_path.mkdir(parents=True, exist_ok=True)

    store = SourceChunkVectorStore(str(lancedb_path), embedder=embedder)
    previous = read_manifest(manifest_path)
    reusable: dict[str, list[float]] = {}
    if previous is not None and previous.recipe_fingerprint == fingerprint:
        stored = await store.stored_vectors()
        reusable = {
            entry.content_hash: entry.vector
            for entry in stored.values()
            if entry.content_hash and len(entry.vector) == embedder_identity.dims
        }

    # The manifest is the completion marker, so it is cleared before anything
    # it describes is destroyed. A run that dies from here on leaves an index
    # that announces itself as incomplete rather than one that lies.
    manifest_path.unlink(missing_ok=True)

    write_started = time.perf_counter()
    with SourceFTSIndex(fts_path) as fts:
        fts.recreate()
        fts.index_chunks(chunks)
    write_seconds = time.perf_counter() - write_started

    await store.drop()
    embed_started = time.perf_counter()
    embedded, reused = await _embed_and_store(
        store,
        embedder,
        chunks,
        reusable,
        batch_size,
        document_prefix=embedder_identity.document_prefix,
    )
    embed_seconds = time.perf_counter() - embed_started
    await store.close()

    write_manifest(
        manifest_path,
        SourceIndexManifest(
            recipe_fingerprint=fingerprint,
            corpus_hash=corpus,
            symbol_chunks=len(symbol_chunks),
            file_window_chunks=len(window_chunks),
            files_covered=files_covered,
            indexed_commit=indexed_commit,
            built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            embedder=embedder_identity,
        ),
    )

    return SourceIndexResult(
        symbol_chunks=len(symbol_chunks),
        file_window_chunks=len(window_chunks),
        files_covered=files_covered,
        embedded=embedded,
        reused=reused,
        corpus_hash=corpus,
        recipe_fingerprint=fingerprint,
        indexed_commit=indexed_commit,
        manifest_path=str(manifest_path),
        lancedb_path=str(lancedb_path),
        fts_path=str(fts_path),
        load_seconds=round(load_seconds, 3),
        embed_seconds=round(embed_seconds, 3),
        write_seconds=round(write_seconds, 3),
        total_seconds=round(time.perf_counter() - started, 3),
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


async def _load_symbols(repo: Path, db_url: str | None) -> tuple[list[SymbolRecord], str | None]:
    """Every persisted symbol for *repo*, plus the commit its index was built at.

    Scoped to one repository row rather than reading the whole table: a
    workspace-mode store holds several repositories in one database, and
    chunking another repository's symbols against this one's worktree would
    slice bodies out of files that never had them.
    """
    from sqlalchemy import select
    from sqlalchemy.exc import OperationalError

    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        resolve_db_url,
    )
    from repowise.core.persistence.models import Repository, WikiSymbol

    unindexed = RuntimeError(
        f"No indexed repository matching {repo} in the wiki database. Run 'repowise init' first."
    )
    engine = create_engine(db_url or resolve_db_url(repo))
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            try:
                repositories: list[Repository] = list(
                    (await session.execute(select(Repository))).scalars().all()
                )
            except OperationalError as exc:
                # No wiki database at all, or one with no schema in it. SQLite
                # creates an empty file on connect, so "never indexed" arrives
                # here as a missing table rather than a missing file — and it
                # is the same answer either way. Deliberately not calling
                # ``init_db``: a read must not leave a store behind.
                raise unindexed from exc
            row = _pick_repository(repositories, repo)
            if row is None:
                raise unindexed
            symbols = list(
                (
                    await session.execute(
                        select(WikiSymbol)
                        .where(WikiSymbol.repository_id == row.id)
                        .order_by(WikiSymbol.file_path, WikiSymbol.start_line, WikiSymbol.symbol_id)
                    )
                )
                .scalars()
                .all()
            )
            head = row.head_commit
    finally:
        await engine.dispose()

    return [
        SymbolRecord(
            symbol_id=symbol.symbol_id,
            file_path=symbol.file_path,
            name=symbol.name,
            qualified_name=symbol.qualified_name,
            kind=symbol.kind,
            signature=symbol.signature or "",
            docstring=symbol.docstring,
            start_line=symbol.start_line,
            end_line=symbol.end_line,
            language=symbol.language or "",
        )
        for symbol in symbols
    ], head or _head_commit(repo)


def _pick_repository(repositories: Sequence[Repository], repo: Path) -> Repository | None:
    """The repository row for *repo*: by path, or the only one there is."""
    for row in repositories:
        local = row.local_path or ""
        try:
            if local and Path(local).resolve() == repo:
                return row
        except OSError:
            continue
    return repositories[0] if len(repositories) == 1 else None


def _build_symbol_chunks(repo: Path, symbols: Sequence[SymbolRecord]) -> list[SourceChunk]:
    """One chunk per symbol, reading each containing file exactly once."""
    chunks: list[SourceChunk] = []
    lines_by_file: dict[str, list[str] | None] = {}
    for symbol in symbols:
        if symbol.file_path not in lines_by_file:
            text = _read_text(repo / symbol.file_path)
            lines_by_file[symbol.file_path] = None if text is None else text.splitlines()
        lines = lines_by_file[symbol.file_path]
        if lines is None:
            # The index knows about a file the worktree no longer has. Its
            # header alone would be a chunk with no source in it, which is
            # exactly the kind of row a retriever should never return.
            continue
        chunks.append(build_symbol_chunk(symbol, lines))
    missing = sum(1 for value in lines_by_file.values() if value is None)
    if missing:
        log.info("source_index_files_unreadable", count=missing)
    return chunks


def _build_window_chunks(repo: Path, symbol_chunks: Sequence[SourceChunk]) -> list[SourceChunk]:
    """Line windows over every tracked file the symbol lane does not cover.

    Driven by ``git ls-files`` rather than by what the wiki pipeline walked,
    deliberately: the formats this lane exists for are usually in the repo's
    ``exclude_patterns`` and so were never ingested at all.
    """
    covered: Counter[str] = Counter(chunk.file_path for chunk in symbol_chunks)
    chunks: list[SourceChunk] = []
    skipped_large = 0
    skipped_binary = 0
    for rel in _tracked_paths(repo):
        if not window_eligible(rel, indexed_symbols=covered.get(rel, 0)):
            continue
        try:
            data = (repo / rel).read_bytes()
        except OSError:
            continue
        if len(data) > MAX_WINDOW_FILE_BYTES:
            skipped_large += 1
            continue
        if looks_binary(data):
            skipped_binary += 1
            continue
        chunks.extend(iter_file_windows(rel, _decode(data)))
    if skipped_large or skipped_binary:
        log.info("source_index_windows_skipped", oversized=skipped_large, binary=skipped_binary)
    return chunks


def _tracked_paths(repo: Path) -> list[str]:
    """Every path git tracks, POSIX-relative.

    Raises on failure rather than returning an empty set the way
    ``pipeline.persist``'s tracked-path witness does. There, silence means "no
    opinion" and another witness answers; here it is the only input the window
    lane has, and an empty list would quietly produce an index missing every
    shell script and compose file — the exact thing this lane exists for.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z"],
            capture_output=True,
            timeout=120,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not list git-tracked files in {repo}: {exc}") from exc
    return [p for p in completed.stdout.decode("utf-8", "replace").split("\0") if p]


def _read_text(path: Path) -> str | None:
    try:
        return _decode(path.read_bytes())
    except OSError:
        return None


def _decode(data: bytes) -> str:
    """Decode repo bytes the way the ingestion pipeline does.

    ``ingestion.source_text.decode_source`` rather than ``bytes.decode``
    because it normalises CRLF: without it, a Windows-checkout file splits into
    lines carrying a trailing ``\\r`` and every chunk from it differs from the
    same file read in text mode.
    """
    from repowise.core.ingestion.source_text import decode_source

    return decode_source(data)


def _head_commit(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.decode("utf-8", "replace").strip() or None


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


async def _embed_and_store(
    store: SourceChunkVectorStore,
    embedder: Embedder,
    chunks: Sequence[SourceChunk],
    reusable: dict[str, list[float]],
    batch_size: int,
    *,
    document_prefix: str = "",
) -> tuple[int, int]:
    """Embed what changed, reuse what did not, write everything. Returns (embedded, reused)."""
    embedded = 0
    reused = 0
    since_report = 0
    buffered: list[tuple[SourceChunk, list[float]]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        pending = [chunk for chunk in batch if chunk.content_hash not in reusable]
        vectors: dict[str, list[float]] = {}
        if pending:
            fresh = await _embed_with_retry(
                embedder, [f"{document_prefix}{chunk.text}" for chunk in pending]
            )
            vectors = {
                chunk.content_hash: [float(v) for v in vector]
                for chunk, vector in zip(pending, fresh, strict=True)
            }
            embedded += len(pending)
        for chunk in batch:
            vector = vectors.get(chunk.content_hash)
            if vector is None:
                vector = reusable[chunk.content_hash]
                reused += 1
            buffered.append((chunk, vector))
        if len(buffered) >= _LANCE_WRITE_BATCH:
            await store.upsert(buffered)
            buffered = []

        since_report += len(batch)
        if since_report >= _PROGRESS_EVERY:
            since_report = 0
            log.info(
                "source_index_embedding",
                done=start + len(batch),
                total=len(chunks),
                embedded=embedded,
                reused=reused,
            )
    if buffered:
        await store.upsert(buffered)
    return embedded, reused


async def _embed_with_retry(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """One embedder call, retried with exponential backoff before it is fatal.

    Deliberately fatal at the end. The alternative — dropping the batch and
    carrying on, which is what the page store does — would write a manifest
    claiming a corpus that the vector table does not hold, and nothing
    downstream could tell.
    """
    last: Exception | None = None
    for attempt in range(_EMBED_ATTEMPTS):
        try:
            return await embedder.embed(texts)
        except Exception as exc:
            last = exc
            if attempt == _EMBED_ATTEMPTS - 1:
                break
            delay = _EMBED_BACKOFF_SECONDS * (2**attempt)
            log.warning(
                "source_index_embed_retry", attempt=attempt + 1, delay=delay, error=str(exc)
            )
            await asyncio.sleep(delay)
    raise RuntimeError(
        f"Embedding failed after {_EMBED_ATTEMPTS} attempts for a batch of {len(texts)} chunks"
    ) from last
