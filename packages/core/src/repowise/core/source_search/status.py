"""Observable health of the source-search publication pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from .fts import SourceFTSIndex
from .generation import GenerationRef
from .manifest import default_manifest_path, read_manifest

__all__ = ["SourceIndexStatus", "inspect_source_index"]


@dataclass(frozen=True, slots=True)
class SourceIndexStatus:
    state: str
    generation_id: str | None
    generation_sequence: int | None
    indexed_commit: str | None
    recipe_fingerprint: str | None
    pending_updates: int
    blocked_updates: int
    building_updates: int
    ready_updates: int
    stale_files: dict[str, str] = field(default_factory=dict)
    expected_chunks: int = 0
    fts_chunks: int | None = None
    vector_chunks: int | None = None
    lance_table: str | None = None
    fts_path: str | None = None
    last_error: str | None = None
    integrity_errors: tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        return self.state in {"degraded", "inconsistent"}

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["degraded"] = self.degraded
        return result


async def _outbox_rows(
    repo: Path,
    db_url: str | None,
    *,
    after_sequence: int,
) -> list[Any]:
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        resolve_db_url,
    )
    from repowise.core.persistence.models import Repository, SourceIndexUpdate

    engine = create_engine(db_url or resolve_db_url(repo))
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            repositories = list((await session.execute(select(Repository))).scalars().all())
            repository_id = None
            for row in repositories:
                try:
                    if row.local_path and Path(row.local_path).resolve() == repo:
                        repository_id = row.id
                        break
                except OSError:
                    continue
            if repository_id is None and len(repositories) == 1:
                repository_id = repositories[0].id
            if repository_id is None:
                return []
            return list(
                (
                    await session.execute(
                        select(SourceIndexUpdate)
                        .where(
                            SourceIndexUpdate.repository_id == repository_id,
                            SourceIndexUpdate.sequence > after_sequence,
                        )
                        .order_by(SourceIndexUpdate.sequence)
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()


async def inspect_source_index(
    repo_path: Path | str,
    *,
    embedder: Any | None = None,
    db_url: str | None = None,
    verify_stores: bool = True,
) -> SourceIndexStatus:
    """Inspect publication, queue and optional cross-store count parity."""

    repo = Path(repo_path).resolve()
    manifest = read_manifest(default_manifest_path(repo))
    active_sequence = manifest.generation_sequence if manifest is not None else 0
    errors: list[str] = []
    try:
        rows = await _outbox_rows(repo, db_url, after_sequence=active_sequence)
    except Exception as exc:
        rows = []
        errors.append(f"source outbox unreadable: {exc}")

    outstanding = rows
    pending = sum(row.state == "pending" for row in outstanding)
    blocked = sum(row.state == "blocked" for row in outstanding)
    building = sum(row.state == "building" for row in outstanding)
    ready = sum(row.state == "ready" for row in outstanding)
    last_error = next(
        (str(row.last_error) for row in reversed(outstanding) if row.last_error),
        None,
    )

    expected = (
        manifest.symbol_chunks + manifest.file_window_chunks if manifest is not None else 0
    )
    fts_count: int | None = None
    vector_count: int | None = None
    if manifest is not None and verify_stores:
        generation = GenerationRef(manifest.generation_id, manifest.generation_sequence)
        fts_path = repo / manifest.fts_path
        if not fts_path.is_file():
            errors.append(f"FTS store missing: {manifest.fts_path}")
        else:
            try:
                with SourceFTSIndex(fts_path, generation=generation) as fts:
                    fts_count = fts.count()
                if fts_count != expected:
                    errors.append(f"FTS count mismatch: expected {expected}, found {fts_count}")
            except Exception as exc:
                errors.append(f"FTS store unreadable: {exc}")

        if embedder is not None:
            try:
                from .vector_store import SourceChunkVectorStore

                store = SourceChunkVectorStore(
                    str(repo / ".repowise" / "lancedb"),
                    embedder=embedder,
                    table_name=manifest.lance_table,
                    generation=generation,
                )
                try:
                    vector_count = await store.count()
                finally:
                    await store.close()
                if vector_count != expected:
                    errors.append(
                        f"Lance count mismatch: expected {expected}, found {vector_count}"
                    )
            except Exception as exc:
                errors.append(f"Lance store unreadable: {exc}")

    if errors:
        state = "inconsistent"
    elif manifest is None:
        state = "degraded" if outstanding else "missing"
    elif blocked or last_error or manifest.stale_files:
        state = "degraded"
    elif pending or building or ready:
        state = "pending"
    else:
        state = "current"

    return SourceIndexStatus(
        state=state,
        generation_id=manifest.generation_id if manifest else None,
        generation_sequence=manifest.generation_sequence if manifest else None,
        indexed_commit=manifest.indexed_commit if manifest else None,
        recipe_fingerprint=manifest.recipe_fingerprint if manifest else None,
        pending_updates=pending,
        blocked_updates=blocked,
        building_updates=building,
        ready_updates=ready,
        stale_files=dict(manifest.stale_files) if manifest else {},
        expected_chunks=expected,
        fts_chunks=fts_count,
        vector_chunks=vector_count,
        lance_table=manifest.lance_table if manifest else None,
        fts_path=manifest.fts_path if manifest else None,
        last_error=last_error,
        integrity_errors=tuple(errors),
    )
