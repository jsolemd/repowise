"""Plugin records written with the index: the ``index-persistence`` seam.

Every implementation runs, in registration order. ``apply`` writes in the
index transaction and must not commit; raising rolls back the whole write.
``publish`` runs after the commit and the writer's cleanup; its failure reaches
the caller and the commit stays, so a plugin retries from what ``apply`` recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from repowise.core.registry.plugins import SeamRegistry


@dataclass(frozen=True)
class IndexWrite:
    mode: Literal["full", "incremental"]
    parsed_files: list[Any] | None
    changed_paths: tuple[str, ...] = ()
    deleted_paths: tuple[str, ...] = ()
    traversal_complete: bool = False
    failures: tuple[str, ...] = ()  # Degraded update steps; empty for full writes.


class IndexPersistence(Protocol):
    async def apply(self, session: AsyncSession, repository_id: str, write: IndexWrite) -> None:
        """Write in *session* without committing."""
        ...

    async def publish(self, repository_id: str, repo_path: Path) -> None:
        """Publish committed records; the plugin owns retries."""
        ...


index_persistence: SeamRegistry[IndexPersistence] = SeamRegistry("index-persistence")


async def apply_index_persistence(
    session: AsyncSession, repository_id: str, write: IndexWrite
) -> None:
    for _, implementation in index_persistence.all():
        await implementation.apply(session, repository_id, write)


async def publish_index_persistence(repository_id: str, repo_path: Path) -> None:
    for _, implementation in index_persistence.all():
        await implementation.publish(repository_id, repo_path)


__all__ = [
    "IndexPersistence",
    "IndexWrite",
    "apply_index_persistence",
    "index_persistence",
    "publish_index_persistence",
]
