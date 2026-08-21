"""The lexical source-search leg, with generation-aware FTS5 visibility.

Identifiers are normalised before storage so camelCase, snake_case, and dotted
spellings share one token stream.  Lifecycle updates add a second invariant:
rows are versioned by generation.  A writer may stage a future generation in
this SQLite file while readers continue to query the manifest's active
generation; no half-applied cross-store update can leak into a result.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .chunks import SourceChunk
from .generation import (
    LEGACY_GENERATION,
    OPEN_ENDED_GENERATION,
    GenerationRef,
    version_row_id,
)

__all__ = [
    "FTS_DB_FILENAME",
    "FTS_SCHEMA_VERSION",
    "GENERATION_FTS_DB_FILENAME",
    "MIN_TOKEN_CHARS",
    "SOURCE_SEARCH_DIRNAME",
    "TOKENIZER_VERSION",
    "SourceFTSHit",
    "SourceFTSIndex",
    "SourceFileInventory",
    "default_fts_path",
    "generation_fts_path",
    "tokenize",
    "tokenizer_parameters",
]

TOKENIZER_VERSION = "camel-split/1"
FTS_SCHEMA_VERSION = "source-fts/2"
MIN_TOKEN_CHARS = 2

SOURCE_SEARCH_DIRNAME = "source_search"
FTS_DB_FILENAME = "source_fts.db"  # A1/legacy path; readers remain compatible.
GENERATION_FTS_DB_FILENAME = "source_fts_v2.db"

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_RUN = re.compile(r"[A-Za-z0-9]+")

_TABLE = "source_fts"
_VERSIONS = "source_fts_versions"
_GENERATIONS = "source_fts_generations"
_IN_CHUNK = 500

_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {_TABLE} USING fts5(
    row_key UNINDEXED,
    chunk_id UNINDEXED,
    file_path UNINDEXED,
    tokens,
    name
);
CREATE TABLE IF NOT EXISTS {_VERSIONS}(
    row_key TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    valid_from INTEGER NOT NULL,
    valid_to INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_source_fts_versions_visibility
    ON {_VERSIONS}(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS ix_source_fts_versions_file
    ON {_VERSIONS}(file_path);
CREATE TABLE IF NOT EXISTS {_GENERATIONS}(
    generation_id TEXT PRIMARY KEY,
    generation_sequence INTEGER NOT NULL UNIQUE,
    recipe_fingerprint TEXT NOT NULL,
    chunk_count INTEGER NOT NULL
);
"""


def tokenize(text: str) -> list[str]:
    """Return the case/identifier-normalised token stream for *text*."""

    spaced = _CAMEL_BOUNDARY.sub(" ", text)
    return [run.lower() for run in _TOKEN_RUN.findall(spaced) if len(run) >= MIN_TOKEN_CHARS]


def tokenizer_parameters() -> tuple[tuple[str, str], ...]:
    """Tokenizer and storage-schema knobs included in the recipe fingerprint."""

    return (
        ("tokenizer_version", TOKENIZER_VERSION),
        ("min_token_chars", str(MIN_TOKEN_CHARS)),
        ("fts_schema_version", FTS_SCHEMA_VERSION),
    )


def default_fts_path(repo_path: Path | str) -> Path:
    """The A1-compatible source FTS path."""

    return Path(repo_path) / ".repowise" / SOURCE_SEARCH_DIRNAME / FTS_DB_FILENAME


def generation_fts_path(repo_path: Path | str) -> Path:
    """The generation-aware FTS path used by lifecycle-managed indexes."""

    return Path(repo_path) / ".repowise" / SOURCE_SEARCH_DIRNAME / GENERATION_FTS_DB_FILENAME


@dataclass(frozen=True, slots=True)
class SourceFTSHit:
    """One BM25 match.  *score* is higher-is-better."""

    chunk_id: str
    file_path: str
    score: float


@dataclass(frozen=True, slots=True)
class SourceFileInventory:
    """Exact active-generation chunk counts for one repository-relative path."""

    total: int
    symbol: int
    file_window: int


class SourceFTSIndex:
    """Synchronous FTS5 index bound to one visible generation."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        generation: GenerationRef | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.generation = generation or LEGACY_GENERATION
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # The manifest may publish immediately after this connection commits.
        # FULL prevents that publication pointer from outliving the staged
        # lexical transaction after a host crash or power loss.
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._versioned = self._has_column(_TABLE, "row_key")

    def _has_column(self, table: str, column: str) -> bool:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(str(row[1]) == column for row in rows)

    # -- writing ---------------------------------------------------------

    def recreate(self) -> None:
        """Replace every lexical row and generation record with an empty v2 schema."""

        self._conn.executescript(
            f"DROP TABLE IF EXISTS {_TABLE};\n"
            f"DROP TABLE IF EXISTS {_VERSIONS};\n"
            f"DROP TABLE IF EXISTS {_GENERATIONS};\n"
            f"{_SCHEMA}"
        )
        self._conn.commit()
        self._versioned = True

    @staticmethod
    def _chunk_row(chunk: SourceChunk, generation: GenerationRef) -> tuple[str, ...]:
        return (
            version_row_id(generation, chunk.chunk_id),
            chunk.chunk_id,
            chunk.file_path,
            " ".join(tokenize(chunk.text)),
            " ".join(tokenize(chunk.name)),
        )

    def index_chunks(self, chunks: Iterable[SourceChunk]) -> int:
        """Insert chunks into this index's bound generation."""

        materialized = list(chunks)
        self.stage_generation(
            self.generation,
            close_paths=[],
            chunks=materialized,
            recipe_fingerprint="",
        )
        return len(materialized)

    def stage_generation(
        self,
        generation: GenerationRef,
        *,
        close_paths: Sequence[str],
        chunks: Sequence[SourceChunk],
        recipe_fingerprint: str,
    ) -> int:
        """Idempotently stage one future generation in a single SQLite transaction.

        Rows current in ``self.generation`` for *close_paths* stop being visible
        at the new sequence.  Fresh rows begin there.  Until the manifest flips,
        readers bound to the parent sequence still see exactly their old corpus.
        """

        if not self._versioned:
            raise RuntimeError("A legacy FTS database must be rebuilt before staging generations")
        if generation.sequence < self.generation.sequence or (
            generation.sequence == self.generation.sequence and close_paths
        ):
            raise ValueError("a staged generation must not precede the active generation")

        rows = [self._chunk_row(chunk, generation) for chunk in chunks]
        with self._conn:
            # Retrying this generation starts its new rows over, while the
            # parent rows' valid_to=N assignment is naturally idempotent.
            retry_keys = [
                str(row[0])
                for row in self._conn.execute(
                    f"SELECT row_key FROM {_VERSIONS} WHERE generation_id = ?",
                    (generation.generation_id,),
                ).fetchall()
            ]
            self._delete_row_keys(retry_keys)

            for start in range(0, len(close_paths), _IN_CHUNK):
                batch = list(dict.fromkeys(close_paths[start : start + _IN_CHUNK]))
                if not batch:
                    continue
                placeholders = ", ".join("?" for _ in batch)
                self._conn.execute(
                    f"UPDATE {_VERSIONS} SET valid_to = ? "
                    f"WHERE file_path IN ({placeholders}) "
                    "AND valid_from <= ? AND valid_to > ?",
                    (
                        generation.sequence,
                        *batch,
                        self.generation.sequence,
                        self.generation.sequence,
                    ),
                )

            if rows:
                self._conn.executemany(
                    f"INSERT INTO {_TABLE}(row_key, chunk_id, file_path, tokens, name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
                self._conn.executemany(
                    f"INSERT INTO {_VERSIONS}(row_key, chunk_id, generation_id, file_path, "
                    "valid_from, valid_to) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            row_key,
                            chunk_id,
                            generation.generation_id,
                            file_path,
                            generation.sequence,
                            OPEN_ENDED_GENERATION,
                        )
                        for row_key, chunk_id, file_path, _tokens, _name in rows
                    ],
                )

            visible = self._count_at(generation.sequence)
            self._conn.execute(
                f"INSERT INTO {_GENERATIONS}(generation_id, generation_sequence, "
                "recipe_fingerprint, chunk_count) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(generation_id) DO UPDATE SET "
                "generation_sequence=excluded.generation_sequence, "
                "recipe_fingerprint=excluded.recipe_fingerprint, "
                "chunk_count=excluded.chunk_count",
                (
                    generation.generation_id,
                    generation.sequence,
                    recipe_fingerprint,
                    visible,
                ),
            )
        return len(rows)

    def rollback_generation(self, generation: GenerationRef) -> None:
        """Remove an unpublished staged generation and reopen its parent rows."""

        if not self._versioned:
            return
        with self._conn:
            keys = [
                str(row[0])
                for row in self._conn.execute(
                    f"SELECT row_key FROM {_VERSIONS} WHERE generation_id = ?",
                    (generation.generation_id,),
                ).fetchall()
            ]
            self._delete_row_keys(keys)
            self._conn.execute(
                f"UPDATE {_VERSIONS} SET valid_to = ? WHERE valid_to = ?",
                (OPEN_ENDED_GENERATION, generation.sequence),
            )
            self._conn.execute(
                f"DELETE FROM {_GENERATIONS} WHERE generation_id = ?",
                (generation.generation_id,),
            )

    def _delete_row_keys(self, row_keys: Sequence[str]) -> None:
        for start in range(0, len(row_keys), _IN_CHUNK):
            batch = row_keys[start : start + _IN_CHUNK]
            if not batch:
                continue
            placeholders = ", ".join("?" for _ in batch)
            self._conn.execute(
                f"DELETE FROM {_TABLE} WHERE row_key IN ({placeholders})", tuple(batch)
            )
            self._conn.execute(
                f"DELETE FROM {_VERSIONS} WHERE row_key IN ({placeholders})", tuple(batch)
            )

    def delete_by_file(self, file_paths: Sequence[str]) -> int:
        """Physically remove every historical row for *file_paths* (maintenance API)."""

        if not file_paths:
            return 0
        if not self._versioned:
            deleted = 0
            for start in range(0, len(file_paths), _IN_CHUNK):
                batch = file_paths[start : start + _IN_CHUNK]
                placeholders = ", ".join("?" for _ in batch)
                cursor = self._conn.execute(
                    f"DELETE FROM {_TABLE} WHERE file_path IN ({placeholders})", tuple(batch)
                )
                deleted += max(cursor.rowcount, 0)
            self._conn.commit()
            return deleted

        keys: list[str] = []
        for start in range(0, len(file_paths), _IN_CHUNK):
            batch = file_paths[start : start + _IN_CHUNK]
            placeholders = ", ".join("?" for _ in batch)
            keys.extend(
                str(row[0])
                for row in self._conn.execute(
                    f"SELECT row_key FROM {_VERSIONS} WHERE file_path IN ({placeholders})",
                    tuple(batch),
                ).fetchall()
            )
        with self._conn:
            self._delete_row_keys(keys)
        return len(keys)

    # -- verification and reading ---------------------------------------

    def verify_generation(
        self,
        generation: GenerationRef,
        *,
        recipe_fingerprint: str,
        expected_count: int,
    ) -> bool:
        """Whether the staged ledger and visible row count match the build plan."""

        if not self._versioned:
            return False
        row = self._conn.execute(
            f"SELECT generation_sequence, recipe_fingerprint, chunk_count "
            f"FROM {_GENERATIONS} WHERE generation_id = ?",
            (generation.generation_id,),
        ).fetchone()
        return bool(
            row
            and int(row[0]) == generation.sequence
            and str(row[1]) == recipe_fingerprint
            and int(row[2]) == expected_count
            and self._count_at(generation.sequence) == expected_count
        )

    def query(self, match: str, limit: int = 20) -> list[SourceFTSHit]:
        """Rank active chunks against free text by BM25."""

        tokens = tokenize(match)
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        try:
            if self._versioned:
                rows = self._conn.execute(
                    f"SELECT f.chunk_id, f.file_path, bm25({_TABLE}) AS rank "
                    f"FROM {_TABLE} AS f JOIN {_VERSIONS} AS v ON v.row_key = f.row_key "
                    f"WHERE {_TABLE} MATCH ? AND v.valid_from <= ? AND v.valid_to > ? "
                    "ORDER BY rank LIMIT ?",
                    (
                        expression,
                        self.generation.sequence,
                        self.generation.sequence,
                        limit,
                    ),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    f"SELECT chunk_id, file_path, bm25({_TABLE}) AS rank "
                    f"FROM {_TABLE} WHERE {_TABLE} MATCH ? ORDER BY rank LIMIT ?",
                    (expression, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [SourceFTSHit(chunk_id=r[0], file_path=r[1], score=-float(r[2])) for r in rows]

    def _count_at(self, sequence: int) -> int:
        return int(
            self._conn.execute(
                f"SELECT count(*) FROM {_VERSIONS} WHERE valid_from <= ? AND valid_to > ?",
                (sequence, sequence),
            ).fetchone()[0]
        )

    def count(self) -> int:
        """How many chunks are visible in this index's bound generation."""

        if self._versioned:
            return self._count_at(self.generation.sequence)
        return int(self._conn.execute(f"SELECT count(*) FROM {_TABLE}").fetchone()[0])

    def active_file_paths(self) -> list[str]:
        """Distinct paths visible at this index's bound generation."""

        if not self._versioned:
            rows = self._conn.execute(f"SELECT DISTINCT file_path FROM {_TABLE}").fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT DISTINCT file_path FROM {_VERSIONS} "
                "WHERE valid_from <= ? AND valid_to > ?",
                (self.generation.sequence, self.generation.sequence),
            ).fetchall()
        return sorted(str(row[0]) for row in rows if row[0])

    def count_for_files(self, file_paths: Sequence[str]) -> int:
        """Visible row count owned by *file_paths*."""

        if not file_paths:
            return 0
        total = 0
        for start in range(0, len(file_paths), _IN_CHUNK):
            batch = list(dict.fromkeys(file_paths[start : start + _IN_CHUNK]))
            placeholders = ", ".join("?" for _ in batch)
            if self._versioned:
                row = self._conn.execute(
                    f"SELECT count(*) FROM {_VERSIONS} "
                    f"WHERE file_path IN ({placeholders}) "
                    "AND valid_from <= ? AND valid_to > ?",
                    (*batch, self.generation.sequence, self.generation.sequence),
                ).fetchone()
            else:
                row = self._conn.execute(
                    f"SELECT count(*) FROM {_TABLE} WHERE file_path IN ({placeholders})",
                    tuple(batch),
                ).fetchone()
            total += int(row[0])
        return total

    def inventory_for_file(self, file_path: str) -> SourceFileInventory:
        """Return exact visible symbol/window counts for *file_path*.

        The lexical store predates an explicit ``source`` column. File-window
        chunk ids nevertheless have one canonical constructor in
        ``iter_file_windows``: ``file:<path>:<start>-<end>``. Match that full
        shape against rows already scoped to *file_path*; every other row is a
        symbol chunk. This reads the active generation only and never consults
        the newer SQL symbol table, whose contents may be ahead of publication.
        """

        if self._versioned:
            rows = self._conn.execute(
                f"SELECT f.chunk_id FROM {_TABLE} AS f "
                f"JOIN {_VERSIONS} AS v ON v.row_key = f.row_key "
                "WHERE v.file_path = ? AND v.valid_from <= ? AND v.valid_to > ?",
                (file_path, self.generation.sequence, self.generation.sequence),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT chunk_id FROM {_TABLE} WHERE file_path = ?",
                (file_path,),
            ).fetchall()

        window_id = re.compile(rf"^file:{re.escape(file_path)}:\d+-\d+$")
        window_count = sum(bool(window_id.fullmatch(str(row[0]))) for row in rows)
        total = len(rows)
        return SourceFileInventory(
            total=total,
            symbol=total - window_count,
            file_window=window_count,
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SourceFTSIndex:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
