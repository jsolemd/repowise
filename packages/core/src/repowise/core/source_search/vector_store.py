"""The dense leg: source chunks in a LanceDB table of their own.

Shares the repo's ``.repowise/lancedb/`` directory with
:class:`~repowise.core.persistence.vector_store.lancedb_store.LanceDBVectorStore`
and nothing else. That store's schema is page-shaped — ``page_id``, ``title``,
``target_path`` — and a chunk has none of those; widening it would have made
one row type mean two things and forced every page reader to ask which kind it
was holding. So this is a second table with its own schema, borrowing the
idioms that store proved: lazy table creation, drop-and-recreate on a
dimension change, ``merge_insert`` upsert, and explicit cosine distance so a
score means the same thing on both legs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from repowise.core.providers.embedding.base import Embedder

from .chunks import SourceChunk

__all__ = [
    "SOURCE_CHUNKS_TABLE",
    "STORED_SNIPPET_CHARS",
    "SourceChunkHit",
    "SourceChunkRecord",
    "SourceChunkVectorStore",
    "StoredVector",
]

#: The table name inside ``.repowise/lancedb/``.
SOURCE_CHUNKS_TABLE = "source_chunks"

#: How much of a chunk's text each row keeps. Enough for a retriever to show
#: evidence without re-reading the file, and bounded so the table stays a
#: vector index rather than a second copy of the repository.
STORED_SNIPPET_CHARS = 2000

#: Bound on one quoted ``IN`` predicate. LanceDB's ``.where()`` is a DataFusion
#: SQL string with no bind parameters, so the list is built by hand and chunked
#: to keep the statement bounded.
_IN_CHUNK = 500


@dataclass(frozen=True, slots=True)
class StoredVector:
    """A vector already in the table, keyed by what it was computed from."""

    content_hash: str
    vector: list[float]


@dataclass(frozen=True, slots=True)
class SourceChunkRecord:
    """A stored chunk's metadata, with no retrieval score attached.

    What a row says about itself, separated from what a particular query
    thinks of it. The lexical leg needs exactly this: FTS5 ranks ``chunk_id``s
    and knows nothing about kinds, line bounds or bodies, so a hit it found
    alone would otherwise reach a caller without the evidence a hit is
    supposed to carry.
    """

    chunk_id: str
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    is_test: bool
    source: str
    content_hash: str
    snippet: str


@dataclass(frozen=True, slots=True)
class SourceChunkHit:
    """One nearest-neighbour match. *score* is cosine similarity."""

    chunk_id: str
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    is_test: bool
    source: str
    content_hash: str
    snippet: str
    score: float


def _quoted_in(column: str, values: Sequence[str]) -> str:
    """An injection-safe ``column IN (...)`` filter, quote-doubling escape."""
    quoted = ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)
    return f"{column} IN ({quoted})"


class SourceChunkVectorStore:
    """Vector store for source chunks, backed by LanceDB.

    The table is created lazily on the first write, because its schema depends
    on the embedder's dimension and there is no honest default for that.
    """

    def __init__(
        self,
        db_path: str,
        embedder: Embedder,
        table_name: str = SOURCE_CHUNKS_TABLE,
    ) -> None:
        self._db_path = db_path
        self._embedder = embedder
        self._table_name = table_name
        self._db: Any = None
        self._table: Any = None

    # -- connection ------------------------------------------------------

    async def _ensure_connected(self) -> None:
        if self._db is not None:
            return
        try:
            import lancedb  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "LanceDB is not installed. Install it with: pip install repowise-core[search]"
            ) from exc

        self._db = await lancedb.connect_async(self._db_path)
        self._table = await self._open_table_if_present()

    async def _open_table_if_present(self) -> Any:
        """The table, or None when this directory has none yet.

        Asked by opening rather than by listing. ``table_names()`` is
        deprecated and warns on every call, and its replacement
        ``list_tables()`` returns a paginated response object rather than a
        list of names — reading it as a list finds nothing, which presents as
        "no table here" and then fails on ``create_table`` with a schema
        mismatch. Opening asks about the one table this store owns and cannot
        be wrong about the answer.
        """
        try:
            return await self._db.open_table(self._table_name)
        except ValueError:
            return None

    @staticmethod
    def _existing_vector_dim(schema: Any) -> int | None:
        """The table's fixed vector width, or None if it has no fixed one."""
        try:
            field = schema.field("vector")
        except KeyError:
            return None
        list_size = getattr(field.type, "list_size", None)
        # pyarrow uses ``list_size == -1`` for variable-length lists.
        if isinstance(list_size, int) and list_size > 0:
            return list_size
        return None

    async def _ensure_table(self, dim: int) -> None:
        """Create the table, dropping a stale one whose vectors are the wrong width.

        Switching embedders is the case this exists for: without the drop,
        every write fails deep inside LanceDB with an IO error that never
        mentions dimensions.
        """
        try:
            import pyarrow as pa  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for SourceChunkVectorStore. "
                "It is installed automatically with lancedb."
            ) from exc

        if self._table is not None:
            existing = self._existing_vector_dim(await self._table.schema())
            if existing is None or existing == dim:
                return
            await self._db.drop_table(self._table_name)
            self._table = None

        schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("file_path", pa.string()),
                pa.field("name", pa.string()),
                pa.field("kind", pa.string()),
                pa.field("start_line", pa.int32()),
                pa.field("end_line", pa.int32()),
                pa.field("is_test", pa.bool_()),
                pa.field("source", pa.string()),
                pa.field("content_hash", pa.string()),
                pa.field("snippet", pa.string()),
            ]
        )
        self._table = await self._db.create_table(self._table_name, schema=schema, exist_ok=True)

    # -- writing ---------------------------------------------------------

    @staticmethod
    def _row(chunk: SourceChunk, vector: Sequence[float]) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "vector": [float(v) for v in vector],
            "file_path": chunk.file_path,
            "name": chunk.name,
            "kind": chunk.kind,
            "start_line": int(chunk.start_line),
            "end_line": int(chunk.end_line),
            "is_test": bool(chunk.is_test),
            "source": chunk.source,
            "content_hash": chunk.content_hash,
            "snippet": chunk.text[:STORED_SNIPPET_CHARS],
        }

    async def upsert(self, items: Sequence[tuple[SourceChunk, Sequence[float]]]) -> int:
        """Write ``(chunk, vector)`` pairs, keyed on ``chunk_id``. Returns rows written."""
        if not items:
            return 0
        await self._ensure_connected()
        await self._ensure_table(len(items[0][1]))
        rows = [self._row(chunk, vector) for chunk, vector in items]
        await (
            self._table.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        return len(rows)

    async def drop(self) -> None:
        """Remove the table entirely, so the next write starts from an empty one."""
        await self._ensure_connected()
        if self._table is None:
            return
        await self._db.drop_table(self._table_name)
        self._table = None

    async def delete_by_file(self, file_paths: Sequence[str]) -> None:
        """Remove every chunk belonging to *file_paths*."""
        if not file_paths:
            return
        await self._ensure_connected()
        if self._table is None:
            return
        for start in range(0, len(file_paths), _IN_CHUNK):
            batch = file_paths[start : start + _IN_CHUNK]
            await self._table.delete(_quoted_in("file_path", batch))

    # -- reading ---------------------------------------------------------

    async def stored_vectors(self) -> dict[str, StoredVector]:
        """Every stored vector, keyed by ``chunk_id``.

        Read before a rebuild drops the table, so a chunk whose text has not
        changed keeps the vector it already has instead of paying for the same
        embedding twice. The caller is responsible for checking that the recipe
        behind those vectors is still the current one — this store has no way
        to know that.
        """
        await self._ensure_connected()
        if self._table is None:
            return {}
        rows = await self._table.query().select(["chunk_id", "content_hash", "vector"]).to_list()
        return {
            str(row["chunk_id"]): StoredVector(
                content_hash=str(row.get("content_hash") or ""),
                vector=[float(v) for v in row["vector"]],
            )
            for row in rows
            if row.get("vector") is not None
        }

    async def count(self) -> int:
        """How many chunks are stored."""
        await self._ensure_connected()
        if self._table is None:
            return 0
        return int(await self._table.count_rows())

    @staticmethod
    def _record(row: dict[str, Any]) -> SourceChunkRecord:
        """One stored row as a record, defaulting every field it may be missing."""
        return SourceChunkRecord(
            chunk_id=str(row["chunk_id"]),
            file_path=str(row.get("file_path") or ""),
            name=str(row.get("name") or ""),
            kind=str(row.get("kind") or ""),
            start_line=int(row.get("start_line") or 0),
            end_line=int(row.get("end_line") or 0),
            is_test=bool(row.get("is_test")),
            source=str(row.get("source") or ""),
            content_hash=str(row.get("content_hash") or ""),
            snippet=str(row.get("snippet") or ""),
        )

    async def search_by_vector(
        self, vector: Sequence[float], limit: int = 20
    ) -> list[SourceChunkHit]:
        """Nearest chunks to *vector* by cosine similarity, higher is better."""
        await self._ensure_connected()
        if self._table is None:
            return []
        builder = self._table.query().nearest_to([float(v) for v in vector])
        if hasattr(builder, "distance_type"):
            builder = builder.distance_type("cosine")
        rows = await builder.limit(limit).to_list()
        return [
            SourceChunkHit(
                **asdict(self._record(row)),
                score=1.0 - float(row.get("_distance", 1.0)),
            )
            for row in rows
        ]

    async def fetch_by_chunk_ids(self, chunk_ids: Sequence[str]) -> dict[str, SourceChunkRecord]:
        """The stored rows for *chunk_ids*, keyed by id; missing ids are absent.

        A filtered scan rather than a vector query, because the caller already
        knows which chunks it wants: the lexical leg ranks by BM25 and gets
        back ids, and a hit only that leg found still has to arrive at a reader
        with its kind, its line bounds and its body.
        """
        if not chunk_ids:
            return {}
        await self._ensure_connected()
        if self._table is None:
            return {}
        out: dict[str, SourceChunkRecord] = {}
        for start in range(0, len(chunk_ids), _IN_CHUNK):
            batch = chunk_ids[start : start + _IN_CHUNK]
            rows = await self._table.query().where(_quoted_in("chunk_id", batch)).to_list()
            for row in rows:
                record = self._record(row)
                out[record.chunk_id] = record
        return out

    async def close(self) -> None:
        self._table = None
        self._db = None
