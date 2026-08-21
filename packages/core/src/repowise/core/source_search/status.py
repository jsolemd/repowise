"""Observable health of the source-search publication pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from .fts import SourceFTSIndex
from .generation import GenerationRef
from .manifest import EmbedderIdentity, default_manifest_path, inspect_manifest

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
    manifest_state: str = "missing"
    manifest_error: str | None = None
    built_at: str | None = None
    published_at: str | None = None
    embedder: EmbedderIdentity | None = None
    parser_fingerprint: str | None = None
    symbol_chunks: int = 0
    file_window_chunks: int = 0
    files_covered: int = 0
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


@dataclass(frozen=True, slots=True)
class _SourceUpdateSnapshot:
    active: Any | None
    counts: dict[str, int]
    outstanding_total: int
    last_error: str | None


async def _source_update_snapshot(
    repo: Path,
    db_url: str | None,
    *,
    active_sequence: int,
    active_generation_id: str | None,
) -> _SourceUpdateSnapshot:
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
                return _SourceUpdateSnapshot(None, {}, 0, None)

            active_candidate = None
            if active_sequence > 0:
                active_candidate = (
                    (
                        await session.execute(
                            select(SourceIndexUpdate).where(
                                SourceIndexUpdate.repository_id == repository_id,
                                SourceIndexUpdate.sequence == active_sequence,
                            )
                        )
                    )
                    .scalars()
                    .one_or_none()
                )
            outstanding_filter = (
                SourceIndexUpdate.repository_id == repository_id,
                SourceIndexUpdate.sequence > active_sequence,
            )
            count_rows = (
                await session.execute(
                    select(SourceIndexUpdate.state, func.count())
                    .where(*outstanding_filter)
                    .group_by(SourceIndexUpdate.state)
                )
            ).all()
            counts = {str(state): int(count) for state, count in count_rows}
            last_error_value = (
                await session.execute(
                    select(SourceIndexUpdate.last_error)
                    .where(
                        *outstanding_filter,
                        SourceIndexUpdate.last_error.is_not(None),
                        SourceIndexUpdate.last_error != "",
                    )
                    .order_by(SourceIndexUpdate.sequence.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            active = (
                active_candidate
                if active_candidate is not None
                and active_candidate.generation_id == active_generation_id
                and active_candidate.state in {"ready", "published"}
                else None
            )
            return _SourceUpdateSnapshot(
                active=active,
                counts=counts,
                outstanding_total=sum(counts.values()),
                last_error=str(last_error_value) if last_error_value else None,
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
    manifest_result = inspect_manifest(default_manifest_path(repo))
    manifest = manifest_result.manifest
    active_sequence = manifest.generation_sequence if manifest is not None else 0
    errors: list[str] = []
    if manifest_result.state == "unreadable":
        errors.append(f"source manifest unreadable: {manifest_result.error}")
    try:
        updates = await _source_update_snapshot(
            repo,
            db_url,
            active_sequence=active_sequence,
            active_generation_id=manifest.generation_id if manifest is not None else None,
        )
    except Exception as exc:
        updates = _SourceUpdateSnapshot(None, {}, 0, None)
        errors.append(f"source outbox unreadable: {exc}")

    active_update = updates.active
    pending = updates.counts.get("pending", 0)
    blocked = updates.counts.get("blocked", 0)
    building = updates.counts.get("building", 0)
    ready = updates.counts.get("ready", 0)
    last_error = updates.last_error

    expected = manifest.symbol_chunks + manifest.file_window_chunks if manifest is not None else 0
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
            lance_path = repo / ".repowise" / "lancedb"
            if not lance_path.is_dir():
                errors.append("Lance store missing: .repowise/lancedb")
            else:
                try:
                    from .vector_store import SourceChunkVectorStore

                    store = SourceChunkVectorStore(
                        str(lance_path),
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
        state = "degraded" if updates.outstanding_total else "missing"
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
        manifest_state=manifest_result.state,
        manifest_error=manifest_result.error,
        built_at=manifest.built_at if manifest else None,
        published_at=(
            active_update.published_at.isoformat()
            if active_update is not None and active_update.published_at is not None
            else None
        ),
        embedder=manifest.embedder if manifest else None,
        parser_fingerprint=(
            str(active_update.parser_fingerprint) if active_update is not None else None
        ),
        symbol_chunks=manifest.symbol_chunks if manifest else 0,
        file_window_chunks=manifest.file_window_chunks if manifest else 0,
        files_covered=manifest.files_covered if manifest else 0,
        stale_files=dict(manifest.stale_files) if manifest else {},
        expected_chunks=expected,
        fts_chunks=fts_count,
        vector_chunks=vector_count,
        lance_table=manifest.lance_table if manifest else None,
        fts_path=manifest.fts_path if manifest else None,
        last_error=last_error,
        integrity_errors=tuple(errors),
    )
