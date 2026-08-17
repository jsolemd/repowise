"""The lexical leg: BM25 over a camelCase-split token stream.

Identifiers are the one place a code search cannot rely on a natural-language
tokenizer. ``parseConfig``, ``parse_config`` and ``PARSE_CONFIG`` name the same
thing to a reader and three different things to FTS5's ``unicode61``
tokenizer, so a query written in one casing misses a definition written in
another. :func:`tokenize` normalises all three to ``parse config`` before
anything is stored, and the query goes through the same function — which is
the whole trick, and why the function is public.

The index is a SQLite sidecar rather than a table inside ``wiki.db``. Same
reasoning as :mod:`repowise.core.precedent.store`: a rebuild here drops and
recreates its table, and doing that inside the database the wiki pipeline is
writing to would make one feature's rebuild the other's lock contention.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .chunks import SourceChunk

__all__ = [
    "FTS_DB_FILENAME",
    "MIN_TOKEN_CHARS",
    "SOURCE_SEARCH_DIRNAME",
    "TOKENIZER_VERSION",
    "SourceFTSHit",
    "SourceFTSIndex",
    "default_fts_path",
    "tokenize",
    "tokenizer_parameters",
]

#: Bumped when :func:`tokenize` changes. Hashed into the manifest's recipe
#: fingerprint, because a stored token stream and a query tokenized by a
#: different rule silently stop matching rather than failing.
TOKENIZER_VERSION = "camel-split/1"

#: Single-character tokens are dropped: they match everywhere and rank nowhere.
MIN_TOKEN_CHARS = 2

SOURCE_SEARCH_DIRNAME = "source_search"
FTS_DB_FILENAME = "source_fts.db"

#: Insert a space at a lower→upper transition, so ``parseConfig`` becomes two
#: tokens while ``HTTPServer`` keeps its acronym intact (there is no lowercase
#: character before the ``S``, so no boundary is found — ``httpserver`` stays
#: one token, and a query for it tokenizes identically).
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: Everything else — underscores, dots, slashes, punctuation — is a separator.
_TOKEN_RUN = re.compile(r"[A-Za-z0-9]+")

_TABLE = "source_fts"

#: Bound on one ``IN`` list, so a large delete stays one statement per chunk
#: rather than one variable past whatever SQLite was compiled with.
_IN_CHUNK = 500

_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {_TABLE} USING fts5(
    chunk_id UNINDEXED,
    file_path UNINDEXED,
    tokens,
    name
);
"""


def tokenize(text: str) -> list[str]:
    """Split *text* into the token stream this index stores and queries.

    camelCase boundaries first, then alphanumeric runs, then lowercase, then
    drop anything shorter than :data:`MIN_TOKEN_CHARS`. Order matters: running
    the boundary rule after lowercasing would find no boundaries at all.
    """
    spaced = _CAMEL_BOUNDARY.sub(" ", text)
    return [run.lower() for run in _TOKEN_RUN.findall(spaced) if len(run) >= MIN_TOKEN_CHARS]


def tokenizer_parameters() -> tuple[tuple[str, str], ...]:
    """The tokenizer knobs, as ``(name, value)`` pairs for the recipe fingerprint.

    A function rather than the two constants, so
    :func:`repowise.core.source_search.manifest.recipe_fingerprint` resolves
    every input the same way — at call time, from the module that owns it.
    Importing the constants bound them at import instead, which meant a change
    to the tokenizer left the fingerprint saying the recipe was unchanged.
    """
    return (
        ("tokenizer_version", TOKENIZER_VERSION),
        ("min_token_chars", str(MIN_TOKEN_CHARS)),
    )


def default_fts_path(repo_path: Path | str) -> Path:
    """Path to *repo_path*'s source-search FTS database (not created here)."""
    return Path(repo_path) / ".repowise" / SOURCE_SEARCH_DIRNAME / FTS_DB_FILENAME


@dataclass(frozen=True, slots=True)
class SourceFTSHit:
    """One BM25 match. *score* is higher-is-better, unlike raw ``bm25()``."""

    chunk_id: str
    file_path: str
    score: float


class SourceFTSIndex:
    """Synchronous FTS5 index over source chunks.

    Synchronous like :class:`repowise.core.precedent.store.EpisodeStore`, and
    for the same reason: the callers are an indexer and a retriever, where an
    asyncio loop wrapped around SQLite buys nothing.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- writing ---------------------------------------------------------

    def recreate(self) -> None:
        """Drop and rebuild the empty table.

        A full rebuild replaces the corpus wholesale, and ``DELETE FROM`` on an
        FTS5 table leaves its internal b-trees carrying the old postings until
        an ``optimize``. Dropping is both faster and the only way the file
        stops growing across rebuilds.
        """
        self._conn.executescript(f"DROP TABLE IF EXISTS {_TABLE};\n{_SCHEMA}")
        self._conn.commit()

    def index_chunks(self, chunks: Iterable[SourceChunk]) -> int:
        """Insert *chunks*. Returns how many rows were written."""
        rows = [
            (
                chunk.chunk_id,
                chunk.file_path,
                " ".join(tokenize(chunk.text)),
                " ".join(tokenize(chunk.name)),
            )
            for chunk in chunks
        ]
        if not rows:
            return 0
        self._conn.executemany(
            f"INSERT INTO {_TABLE}(chunk_id, file_path, tokens, name) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def delete_by_file(self, file_paths: Sequence[str]) -> int:
        """Remove every row belonging to *file_paths*. Returns rows deleted."""
        if not file_paths:
            return 0
        deleted = 0
        for start in range(0, len(file_paths), _IN_CHUNK):
            batch = file_paths[start : start + _IN_CHUNK]
            placeholders = ", ".join("?" * len(batch))
            cursor = self._conn.execute(
                f"DELETE FROM {_TABLE} WHERE file_path IN ({placeholders})", tuple(batch)
            )
            deleted += cursor.rowcount if cursor.rowcount > 0 else 0
        self._conn.commit()
        return deleted

    # -- reading ---------------------------------------------------------

    def query(self, match: str, limit: int = 20) -> list[SourceFTSHit]:
        """Rank chunks against *match* by BM25.

        *match* is free text, tokenized by :func:`tokenize` and joined with
        ``OR`` so a multi-word question still ranks partial matches rather than
        returning nothing. Tokens are alphanumeric by construction, so nothing
        in the expression can be read as FTS5 syntax; they are quoted anyway,
        because a bare token that happens to spell ``or`` or ``near`` is an
        operator to the parser.

        FTS5's default BM25 is k1=1.2, b=0.75 and the columns are unweighted,
        which is the configuration this recipe was measured with. ``bm25()``
        returns a negative number that sorts ascending, so the sign is flipped
        here and every caller can treat higher as better, exactly as the vector
        leg's cosine similarity does.
        """
        tokens = tokenize(match)
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        try:
            rows = self._conn.execute(
                f"SELECT chunk_id, file_path, bm25({_TABLE}) AS rank "
                f"FROM {_TABLE} WHERE {_TABLE} MATCH ? ORDER BY rank LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # A malformed expression is a query problem, not a store problem —
            # an empty result is the honest answer and keeps a retriever's
            # other leg alive.
            return []
        return [SourceFTSHit(chunk_id=r[0], file_path=r[1], score=-float(r[2])) for r in rows]

    def count(self) -> int:
        """How many chunks are indexed."""
        return int(self._conn.execute(f"SELECT count(*) FROM {_TABLE}").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SourceFTSIndex:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
