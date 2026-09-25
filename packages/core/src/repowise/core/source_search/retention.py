"""Retire source-search storage that no published generation can read.

Publication never deletes: staging closes rows by sequence, a recipe change
builds a new table and FTS file beside the old ones, and LanceDB keeps every
table version. Without retention a repository's ``.repowise`` grows with every
saved edit while readers use a small fraction of it.

Retention runs after a publish, under the reconcile lock, and keeps exactly
what the new generation and the one before it can see. A reader that picked up
the previous manifest just before the flip therefore finishes its request;
anything older has already been replaced by the manifest-keyed reopen.

* Tables and FTS files named by neither manifest are removed.
* Rows closed at or before the previous generation are removed from the live
  stores, and Lance versions older than the previous publication (less a
  margin) are cleaned up. This runs when dead rows reach a quarter of the
  visible corpus or the version history passes a limit, so a one-file edit
  does not rewrite the table.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog

from repowise.core.providers.embedding.base import MockEmbedder

from .fts import SOURCE_SEARCH_DIRNAME, SourceFTSIndex
from .generation import GenerationRef
from .manifest import SourceIndexManifest
from .vector_store import SOURCE_CHUNKS_TABLE, SourceChunkVectorStore

__all__ = ["RetentionResult", "retire_unreachable"]

log = structlog.get_logger(__name__)

#: Retire rows once dead rows reach this fraction (1/N) of the visible corpus.
DEAD_FRACTION = 4
#: ...or once a store's version history passes this many versions.
VERSION_LIMIT = 200
#: Lance versions committed this long before the previous publication survive,
#: covering the staging commits that precede ``built_at``.
VERSION_MARGIN = timedelta(minutes=10)

_TABLE_NAME = re.compile(rf"^{SOURCE_CHUNKS_TABLE}(_[0-9a-f]+)*$")
_FTS_FILE = re.compile(r"^source_fts[^/]*\.db(-wal|-shm|-journal)?$")


@dataclass(frozen=True, slots=True)
class RetentionResult:
    tables_dropped: tuple[str, ...]
    fts_files_removed: tuple[str, ...]
    rows_retired: int
    fts_rows_retired: int


def _built_at(manifest: SourceIndexManifest) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(manifest.built_at)
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


async def retire_unreachable(
    repo: Path,
    *,
    published: SourceIndexManifest,
    previous: SourceIndexManifest | None,
) -> RetentionResult:
    """Remove storage that neither *published* nor *previous* can read."""

    kept = [published] if previous is None else [published, previous]
    lance_dir = repo / ".repowise" / "lancedb"

    import lancedb  # type: ignore[import]

    db = await lancedb.connect_async(str(lance_dir))
    keep_tables = {manifest.lance_table for manifest in kept}
    dropped = tuple(
        sorted(
            name
            for name in await db.table_names()
            if _TABLE_NAME.match(name) and name not in keep_tables
        )
    )
    for name in dropped:
        await db.drop_table(name)

    fts_dir = repo / ".repowise" / SOURCE_SEARCH_DIRNAME
    keep_files = {(repo / manifest.fts_path).name for manifest in kept}
    removed: list[str] = []
    for entry in sorted(fts_dir.iterdir()) if fts_dir.is_dir() else []:
        base = re.sub(r"-(wal|shm|journal)$", "", entry.name)
        if entry.is_file() and _FTS_FILE.match(entry.name) and base not in keep_files:
            os.unlink(entry)
            removed.append(entry.name)

    rows = fts_rows = 0
    floor = previous.generation_sequence if previous is not None else 0
    previous_built = _built_at(previous) if previous is not None else None
    if floor > 0 and previous_built is not None:
        generation = GenerationRef(published.generation_id, published.generation_sequence)
        store = SourceChunkVectorStore(
            str(lance_dir),
            # Retention never embeds; the store needs an embedder only to exist.
            embedder=MockEmbedder(),
            table_name=published.lance_table,
            generation=generation,
        )
        try:
            dead, visible, versions = await store.retention_pressure(floor)
            if dead * DEAD_FRACTION >= max(visible, 1) or versions > VERSION_LIMIT:
                keep_since = datetime.now(UTC) - previous_built + VERSION_MARGIN
                rows = await store.retire_before(floor, keep_versions_since=keep_since)
        finally:
            await store.close()

        with SourceFTSIndex(repo / published.fts_path, generation=generation) as fts:
            if fts.dead_row_count(floor) * DEAD_FRACTION >= max(fts.count(), 1):
                fts_rows = fts.retire_before(floor)

    result = RetentionResult(
        tables_dropped=dropped,
        fts_files_removed=tuple(removed),
        rows_retired=rows,
        fts_rows_retired=fts_rows,
    )
    if dropped or removed or rows or fts_rows:
        log.info(
            "source_index_retention",
            repo=str(repo),
            tables_dropped=list(dropped),
            fts_files_removed=removed,
            rows_retired=rows,
            fts_rows_retired=fts_rows,
        )
    return result
