"""Registry bootstrap and config-sync helpers for the docs subsystem."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from repowise.docs import db
from repowise.docs.indexer import delete_by_library, get_chunk_count, relabel_library_chunks
from repowise.docs.jobs.policy import compute_next_freshness_check_at
from repowise.docs.library.models import (
    JobEnqueueDisposition,
    JobType,
    LibrariesConfig,
    LibraryConfig,
    LibrarySourceType,
    LibraryState,
    LibraryStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class LegacyMigrationResult:
    aliases_migrated: int = 0
    chunks_relabelled: int = 0
    legacy_chunks_deleted: int = 0


def find_libraries_yaml() -> Path:
    """Find the canonical libraries.yaml path."""
    candidates: list[Path] = []
    env_override = os.environ.get("DOC_SEARCH_LIBRARIES_YAML", "").strip()
    if env_override:
        candidates.append(Path(env_override))

    candidates.extend(
        [
            Path("/workspaces/SoleMD.Infra/infra/repowise/docs/libraries.yaml"),
            Path("libraries.yaml"),
            Path(__file__).parent.parent / "libraries.yaml",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"libraries.yaml not found. Checked: {[str(p) for p in candidates]}")


def load_libraries_config(yaml_path: Path | None = None) -> LibrariesConfig:
    """Load the canonical library configuration."""
    resolved = yaml_path or find_libraries_yaml()
    logger.info("Reading libraries from %s", resolved)
    with resolved.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return LibrariesConfig.model_validate(data)


async def _adopt_existing_chunks(lib_state: LibraryState) -> bool:
    chunk_count = await get_chunk_count(lib_state.library_id)
    if chunk_count <= 0:
        return False

    checked_at = datetime.now(UTC)
    await db.update_library_status(
        lib_state.library_id,
        status=LibraryStatus.READY,
        chunk_count=chunk_count,
        indexed_at=checked_at,
        freshness_checked_at=checked_at,
        next_freshness_check_at=compute_next_freshness_check_at(
            lib_state.library_id,
            lib_state.priority,
            checked_at,
        ),
        error_message="",
    )
    logger.info("Adopted %d existing chunks for %s", chunk_count, lib_state.library_id)
    return True


def _needs_metadata_reconciliation(lib_state: LibraryState, *, snapshot_ready: bool) -> bool:
    """Existing chunks are not enough; freshness and graph sync need metadata too."""
    if lib_state.source_type == LibrarySourceType.SNAPSHOT and not snapshot_ready:
        return False
    return not lib_state.current_sha or lib_state.file_count <= 0


async def _migrate_legacy_library_ids(
    lib_config: LibraryConfig,
    existing: dict[str, LibraryState],
) -> LegacyMigrationResult:
    result = LegacyMigrationResult()
    library_id = lib_config.library_id
    if not library_id:
        return result

    for legacy_id in lib_config.legacy_ids:
        if legacy_id == library_id:
            continue

        legacy_chunk_count = await get_chunk_count(legacy_id)
        new_chunk_count = await get_chunk_count(library_id)
        if legacy_chunk_count > 0:
            if new_chunk_count == 0:
                result.chunks_relabelled += await relabel_library_chunks(legacy_id, library_id)
            else:
                result.legacy_chunks_deleted += await delete_by_library(legacy_id)

        legacy_state = existing.get(legacy_id)
        if legacy_state is not None:
            migrated_state = await db.migrate_library_alias(legacy_id, lib_config)
            if migrated_state is not None:
                result.aliases_migrated += 1
                existing.pop(legacy_id, None)
                existing[library_id] = migrated_state

    return result


async def sync_library_registry(
    *,
    yaml_path: Path | None = None,
    queue_missing_jobs: bool,
) -> dict[str, int | bool]:
    """Sync the docs registry from config without causing unnecessary churn.

    Existing libraries are updated in-place from `libraries.yaml` but otherwise left
    alone. Newly discovered libraries are either adopted from existing Qdrant chunks
    or queued for initial indexing when background workers are available.
    """
    config = load_libraries_config(yaml_path)
    existing = {lib.library_id: lib for lib in await db.list_libraries()}

    upserted = 0
    adopted = 0
    queued = 0
    reconciliation_requested = 0
    new_libraries = 0
    aliases_migrated = 0
    chunks_relabelled = 0
    legacy_chunks_deleted = 0

    for lib_config in config.libraries:
        was_present = lib_config.library_id in existing
        lib_state = await db.upsert_library(lib_config)
        upserted += 1

        legacy_migration = await _migrate_legacy_library_ids(lib_config, existing)
        aliases_migrated += legacy_migration.aliases_migrated
        chunks_relabelled += legacy_migration.chunks_relabelled
        legacy_chunks_deleted += legacy_migration.legacy_chunks_deleted
        if legacy_migration.aliases_migrated:
            was_present = True

        if was_present:
            snapshot_ready = (
                lib_state.source_type != LibrarySourceType.SNAPSHOT
                or await db.get_snapshot_state(lib_state.library_id) is not None
            )
            if (
                queue_missing_jobs
                and lib_state.status == LibraryStatus.READY
                and _needs_metadata_reconciliation(
                    lib_state,
                    snapshot_ready=snapshot_ready,
                )
            ):
                reconciliation_requested += 1
                result = await db.enqueue_job(
                    lib_state.library_id,
                    JobType.INCREMENTAL,
                    priority=max(lib_state.priority, 6),
                )
                if result.disposition == JobEnqueueDisposition.QUEUED:
                    queued += 1
            continue

        new_libraries += 1
        if await _adopt_existing_chunks(lib_state):
            adopted += 1
            snapshot_ready = (
                lib_state.source_type != LibrarySourceType.SNAPSHOT
                or await db.get_snapshot_state(lib_state.library_id) is not None
            )
            if queue_missing_jobs and _needs_metadata_reconciliation(
                lib_state,
                snapshot_ready=snapshot_ready,
            ):
                reconciliation_requested += 1
                result = await db.enqueue_job(
                    lib_state.library_id,
                    JobType.INCREMENTAL,
                    priority=max(lib_state.priority, 6),
                )
                if result.disposition == JobEnqueueDisposition.QUEUED:
                    queued += 1
            continue

        if queue_missing_jobs:
            result = await db.enqueue_job(
                lib_state.library_id,
                JobType.FULL,
                priority=lib_state.priority,
            )
            if result.disposition == JobEnqueueDisposition.QUEUED:
                queued += 1

    return {
        "configured": len(config.libraries),
        "upserted": upserted,
        "new_libraries": new_libraries,
        "adopted_from_qdrant": adopted,
        "jobs_queued": queued,
        "metadata_reconciliation_requested": reconciliation_requested,
        "legacy_aliases_migrated": aliases_migrated,
        "legacy_chunks_relabelled": chunks_relabelled,
        "legacy_chunks_deleted": legacy_chunks_deleted,
        "queue_missing_jobs": queue_missing_jobs,
        "registry_previously_empty": len(existing) == 0,
    }


async def seed_libraries(
    yaml_path: Path | None = None,
    *,
    queue_missing_jobs: bool = True,
) -> dict[str, int | bool]:
    """CLI-friendly seeding entrypoint."""
    await db.init_pool()
    try:
        return await sync_library_registry(
            yaml_path=yaml_path,
            queue_missing_jobs=queue_missing_jobs,
        )
    finally:
        await db.close_pool()
