"""Generation-aware LanceDB storage for source chunks.

Rows use visibility intervals, so staging generation N never changes what a
reader bound to generation N-1 can see.  The publication manifest is therefore
the only active-generation switch even though LanceDB and SQLite are updated in
separate transactions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from repowise.core.providers.embedding.base import Embedder

from .chunks import SourceChunk
from .generation import (
    LEGACY_GENERATION,
    OPEN_ENDED_GENERATION,
    GenerationRef,
    version_row_id,
)

__all__ = [
    "SOURCE_CHUNKS_TABLE",
    "SOURCE_GENERATIONS_TABLE",
    "STORED_SNIPPET_CHARS",
    "SourceChunkHit",
    "SourceChunkRecord",
    "SourceChunkVectorStore",
    "SourceIndexInventory",
    "StoredVector",
]

SOURCE_CHUNKS_TABLE = "source_chunks"
SOURCE_GENERATIONS_TABLE = "source_chunk_generations"
STORED_SNIPPET_CHARS = 2000
_IN_CHUNK = 500
_REQUIRED_COLUMNS = frozenset(
    {
        "row_key",
        "chunk_id",
        "generation_id",
        "valid_from",
        "valid_to",
        "vector",
        "file_path",
        "name",
        "kind",
        "start_line",
        "end_line",
        "is_test",
        "source",
        "content_hash",
        "snippet",
    }
)
_VISIBILITY_COLUMNS = frozenset({"row_key", "generation_id", "valid_from", "valid_to"})


@dataclass(frozen=True, slots=True)
class StoredVector:
    """A vector already in the active generation."""

    content_hash: str
    vector: list[float]


@dataclass(frozen=True, slots=True)
class SourceChunkRecord:
    """A visible chunk's metadata, without a query score."""

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
    """One nearest-neighbour match.  *score* is cosine similarity."""

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


@dataclass(frozen=True, slots=True)
class SourceIndexInventory:
    """Counts and corpus-hash inputs for one visible generation."""

    symbol_chunks: int
    file_window_chunks: int
    files_covered: int
    entries: tuple[tuple[str, str], ...]


def _quoted_in(column: str, values: Sequence[str]) -> str:
    quoted = ", ".join("'" + str(value).replace("'", "''") + "'" for value in values)
    return f"{column} IN ({quoted})"


def _quoted_eq(column: str, value: str) -> str:
    return f"{column} = '{value.replace(chr(39), chr(39) * 2)}'"


class SourceChunkVectorStore:
    """Vector store bound to one manifest generation."""

    def __init__(
        self,
        db_path: str,
        embedder: Embedder,
        table_name: str = SOURCE_CHUNKS_TABLE,
        *,
        generation: GenerationRef | None = None,
    ) -> None:
        self._db_path = db_path
        self._embedder = embedder
        self._table_name = table_name
        self.generation = generation or LEGACY_GENERATION
        self._db: Any = None
        self._table: Any = None
        self._generation_table: Any = None
        self._versioned: bool | None = None

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
        self._table = await self._open_table_if_present(self._table_name)
        self._generation_table = await self._open_table_if_present(SOURCE_GENERATIONS_TABLE)
        if self._table is not None:
            schema = await self._table.schema()
            self._versioned = _VISIBILITY_COLUMNS.issubset(schema.names)

    async def _open_table_if_present(self, table_name: str) -> Any:
        try:
            return await self._db.open_table(table_name)
        except ValueError:
            return None

    @staticmethod
    def _existing_vector_dim(schema: Any) -> int | None:
        try:
            field = schema.field("vector")
        except KeyError:
            return None
        list_size = getattr(field.type, "list_size", None)
        return list_size if isinstance(list_size, int) and list_size > 0 else None

    async def _ensure_table(self, dim: int, *, managed: bool) -> None:
        try:
            import pyarrow as pa  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for SourceChunkVectorStore. "
                "It is installed automatically with lancedb."
            ) from exc

        if self._table is not None:
            schema = await self._table.schema()
            existing = self._existing_vector_dim(schema)
            names = frozenset(schema.names)
            versioned = _VISIBILITY_COLUMNS.issubset(names)
            self._versioned = versioned
            if managed and not versioned:
                raise RuntimeError(
                    "Refusing to stage a managed generation into the legacy source_chunks "
                    "table; build it beside the legacy table and publish via the manifest"
                )
            required = _REQUIRED_COLUMNS if managed else (_REQUIRED_COLUMNS - _VISIBILITY_COLUMNS)
            if existing == dim and required.issubset(names):
                return
            await self._db.drop_table(self._table_name)
            self._table = None
            self._versioned = None

        fields = [
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
        if managed:
            fields = [
                pa.field("row_key", pa.string()),
                fields[0],
                pa.field("generation_id", pa.string()),
                pa.field("valid_from", pa.int64()),
                pa.field("valid_to", pa.int64()),
                *fields[1:],
            ]
        schema = pa.schema(fields)
        self._table = await self._db.create_table(self._table_name, schema=schema, exist_ok=True)
        self._versioned = managed

    async def _ensure_generation_table(self) -> None:
        if self._generation_table is not None:
            return
        try:
            import pyarrow as pa  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("pyarrow is required for source generation metadata") from exc
        schema = pa.schema(
            [
                pa.field("generation_id", pa.string()),
                pa.field("generation_sequence", pa.int64()),
                pa.field("recipe_fingerprint", pa.string()),
                pa.field("chunk_count", pa.int64()),
                pa.field("table_name", pa.string()),
            ]
        )
        self._generation_table = await self._db.create_table(
            SOURCE_GENERATIONS_TABLE, schema=schema, exist_ok=True
        )

    def _visibility(self, sequence: int | None = None) -> str | None:
        active = self.generation.sequence if sequence is None else sequence
        if not self._versioned:
            if active != LEGACY_GENERATION.sequence:
                raise RuntimeError(
                    "A non-legacy source manifest points at a legacy Lance table; "
                    "refusing an unversioned read"
                )
            return None
        return f"valid_from <= {active} AND valid_to > {active}"

    def _visible_query(self, query: Any, sequence: int | None = None) -> Any:
        visibility = self._visibility(sequence)
        return query.where(visibility) if visibility is not None else query

    # -- writing ---------------------------------------------------------

    @staticmethod
    def _row(
        chunk: SourceChunk,
        vector: Sequence[float],
        generation: GenerationRef,
    ) -> dict[str, Any]:
        return {
            "row_key": version_row_id(generation, chunk.chunk_id),
            "chunk_id": chunk.chunk_id,
            "generation_id": generation.generation_id,
            "valid_from": generation.sequence,
            "valid_to": OPEN_ENDED_GENERATION,
            "vector": [float(value) for value in vector],
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

    @staticmethod
    def _legacy_row(chunk: SourceChunk, vector: Sequence[float]) -> dict[str, Any]:
        row = SourceChunkVectorStore._row(chunk, vector, LEGACY_GENERATION)
        for column in _VISIBILITY_COLUMNS:
            row.pop(column)
        return row

    async def upsert(self, items: Sequence[tuple[SourceChunk, Sequence[float]]]) -> int:
        """Write items into this store's bound generation, idempotently."""
        if not items:
            return 0
        await self._ensure_connected()
        managed = self.generation.sequence > LEGACY_GENERATION.sequence
        await self._ensure_table(len(items[0][1]), managed=managed)
        if managed:
            rows = [self._row(chunk, vector, self.generation) for chunk, vector in items]
            key = "row_key"
        else:
            rows = [self._legacy_row(chunk, vector) for chunk, vector in items]
            key = "chunk_id"
        await (
            self._table.merge_insert(key)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        return len(rows)

    async def _upsert_for_generation(
        self,
        items: Sequence[tuple[SourceChunk, Sequence[float]]],
        generation: GenerationRef,
    ) -> int:
        if not items:
            return 0
        await self._ensure_connected()
        await self._ensure_table(len(items[0][1]), managed=True)
        rows = [self._row(chunk, vector, generation) for chunk, vector in items]
        await (
            self._table.merge_insert("row_key")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        return len(rows)

    async def stage_generation(
        self,
        generation: GenerationRef,
        *,
        close_paths: Sequence[str],
        items: Sequence[tuple[SourceChunk, Sequence[float]]],
        recipe_fingerprint: str,
        expected_count: int,
    ) -> int:
        """Idempotently stage future rows while preserving the active view."""

        if generation.sequence <= self.generation.sequence:
            raise ValueError("a staged generation must follow the active generation")
        await self._ensure_connected()
        if items:
            await self._ensure_table(len(items[0][1]), managed=True)
        if self._table is not None:
            await self.rollback_generation(generation)
            for start in range(0, len(close_paths), _IN_CHUNK):
                batch = list(dict.fromkeys(close_paths[start : start + _IN_CHUNK]))
                if not batch:
                    continue
                await self._table.update(
                    where=f"{_quoted_in('file_path', batch)} AND {self._visibility()}",
                    updates={"valid_to": generation.sequence},
                )
            await self._upsert_for_generation(items, generation)

        visible = await self.count(sequence=generation.sequence)
        if visible != expected_count:
            raise RuntimeError(
                f"Lance generation {generation.generation_id} has {visible} visible rows; "
                f"expected {expected_count}"
            )
        await self._record_generation(
            generation,
            recipe_fingerprint=recipe_fingerprint,
            chunk_count=visible,
        )
        return len(items)

    async def rollback_generation(self, generation: GenerationRef) -> None:
        """Remove an unpublished staged generation and reopen its parent rows."""

        await self._ensure_connected()
        if self._table is not None and self._versioned:
            await self._table.delete(_quoted_eq("generation_id", generation.generation_id))
            await self._table.update(
                where=f"valid_to = {generation.sequence}",
                updates={"valid_to": OPEN_ENDED_GENERATION},
            )
        if self._generation_table is not None:
            await self._generation_table.delete(
                _quoted_eq("generation_id", generation.generation_id)
            )

    async def _record_generation(
        self,
        generation: GenerationRef,
        *,
        recipe_fingerprint: str,
        chunk_count: int,
    ) -> None:
        await self._ensure_generation_table()
        row = {
            "generation_id": generation.generation_id,
            "generation_sequence": generation.sequence,
            "recipe_fingerprint": recipe_fingerprint,
            "chunk_count": chunk_count,
            "table_name": self._table_name,
        }
        await (
            self._generation_table.merge_insert("generation_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([row])
        )

    async def drop(self) -> None:
        await self._ensure_connected()
        if self._table is None:
            return
        await self._db.drop_table(self._table_name)
        self._table = None
        self._versioned = None

    async def delete_by_file(self, file_paths: Sequence[str]) -> None:
        """Physically remove all historical chunks belonging to *file_paths*."""

        if not file_paths:
            return
        await self._ensure_connected()
        if self._table is None:
            return
        for start in range(0, len(file_paths), _IN_CHUNK):
            await self._table.delete(_quoted_in("file_path", file_paths[start : start + _IN_CHUNK]))

    # -- reading and verification ---------------------------------------

    async def stored_vectors(self) -> dict[str, StoredVector]:
        """Visible vectors keyed by chunk id."""

        await self._ensure_connected()
        if self._table is None:
            return {}
        query = self._visible_query(self._table.query()).select(
            ["chunk_id", "content_hash", "vector"]
        )
        rows = await query.to_list()
        return {
            str(row["chunk_id"]): StoredVector(
                content_hash=str(row.get("content_hash") or ""),
                vector=[float(value) for value in row["vector"]],
            )
            for row in rows
            if row.get("vector") is not None
        }

    async def vectors_by_content_hash(self) -> dict[str, list[float]]:
        """One active vector per content hash, for recipe-safe reuse."""

        return {
            stored.content_hash: stored.vector
            for stored in (await self.stored_vectors()).values()
            if stored.content_hash
        }

    async def active_file_paths(self) -> list[str]:
        """Distinct file paths visible at the bound generation."""

        await self._ensure_connected()
        if self._table is None:
            return []
        rows = await self._visible_query(self._table.query()).select(["file_path"]).to_list()
        return sorted({str(row.get("file_path") or "") for row in rows if row.get("file_path")})

    async def count_for_files(self, file_paths: Sequence[str]) -> int:
        """Visible row count owned by *file_paths*."""

        if not file_paths:
            return 0
        await self._ensure_connected()
        if self._table is None:
            return 0
        total = 0
        for start in range(0, len(file_paths), _IN_CHUNK):
            batch = file_paths[start : start + _IN_CHUNK]
            visibility = self._visibility()
            where = _quoted_in("file_path", batch)
            if visibility is not None:
                where = f"{where} AND {visibility}"
            total += int(await self._table.count_rows(where))
        return total

    async def inventory(self, *, sequence: int | None = None) -> SourceIndexInventory:
        """Counts plus ``(chunk_id, content_hash)`` pairs for a visible corpus."""

        await self._ensure_connected()
        if self._table is None:
            return SourceIndexInventory(0, 0, 0, ())
        rows = await (
            self._visible_query(self._table.query(), sequence)
            .select(["chunk_id", "content_hash", "file_path", "source"])
            .to_list()
        )
        return SourceIndexInventory(
            symbol_chunks=sum(1 for row in rows if row.get("source") == "symbol"),
            file_window_chunks=sum(1 for row in rows if row.get("source") == "file_window"),
            files_covered=len({str(row.get("file_path") or "") for row in rows}),
            entries=tuple(
                (str(row.get("chunk_id") or ""), str(row.get("content_hash") or ""))
                for row in rows
            ),
        )

    async def count(self, *, sequence: int | None = None) -> int:
        await self._ensure_connected()
        if self._table is None:
            return 0
        visibility = self._visibility(sequence)
        return int(
            await self._table.count_rows(visibility)
            if visibility is not None
            else await self._table.count_rows()
        )

    async def verify_generation(
        self,
        generation: GenerationRef,
        *,
        recipe_fingerprint: str,
        expected_count: int,
    ) -> bool:
        await self._ensure_connected()
        if self._generation_table is None:
            return False
        rows = await (
            self._generation_table.query()
            .where(_quoted_eq("generation_id", generation.generation_id))
            .limit(1)
            .to_list()
        )
        if not rows:
            return False
        row: Mapping[str, Any] = rows[0]
        return bool(
            int(row.get("generation_sequence") or -1) == generation.sequence
            and str(row.get("recipe_fingerprint") or "") == recipe_fingerprint
            and str(row.get("table_name") or "") == self._table_name
            and int(row.get("chunk_count") or -1) == expected_count
            and await self.count(sequence=generation.sequence) == expected_count
        )

    @staticmethod
    def _record(row: Mapping[str, Any]) -> SourceChunkRecord:
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
        await self._ensure_connected()
        if self._table is None:
            return []
        builder = self._table.query().nearest_to([float(value) for value in vector])
        if hasattr(builder, "distance_type"):
            builder = builder.distance_type("cosine")
        visibility = self._visibility()
        if visibility is not None:
            builder = builder.where(visibility)
        rows = await builder.limit(limit).to_list()
        return [
            SourceChunkHit(
                **asdict(self._record(row)),
                score=1.0 - float(row.get("_distance", 1.0)),
            )
            for row in rows
        ]

    async def fetch_by_chunk_ids(self, chunk_ids: Sequence[str]) -> dict[str, SourceChunkRecord]:
        if not chunk_ids:
            return {}
        await self._ensure_connected()
        if self._table is None:
            return {}
        out: dict[str, SourceChunkRecord] = {}
        for start in range(0, len(chunk_ids), _IN_CHUNK):
            batch = chunk_ids[start : start + _IN_CHUNK]
            visibility = self._visibility()
            where = _quoted_in("chunk_id", batch)
            if visibility is not None:
                where = f"{where} AND {visibility}"
            rows = await self._table.query().where(where).to_list()
            for row in rows:
                record = self._record(row)
                out[record.chunk_id] = record
        return out

    async def close(self) -> None:
        self._table = None
        self._generation_table = None
        self._db = None
        self._versioned = None
