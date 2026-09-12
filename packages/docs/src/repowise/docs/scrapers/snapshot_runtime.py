"""Runtime automation bridge for snapshot-backed documentation scrapers."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from repowise.docs import db
from repowise.docs.library.models import LibrarySourceType, LibraryState, LibraryStatus
from repowise.docs.scrapers import cosmograph, onepassword_developer, pubtator3, semantic_scholar
from repowise.docs.snapshot_publish import SnapshotPublishResult, publish_snapshot_directory

logger = logging.getLogger(__name__)


ProbeFn = Callable[[], str | Awaitable[str]]
FetchFn = Callable[..., str | Awaitable[str]]


@dataclass(frozen=True)
class SnapshotSourceHandler:
    library_id: str
    probe_source_ref: ProbeFn
    fetch_into: FetchFn


def _fetch_semantic_scholar_snapshot(
    *,
    output_dir: Path,
    persist_freshness: bool = False,
) -> str:
    source_ref = semantic_scholar.fetch_swagger_specs(
        output_dir=output_dir,
        persist_freshness=persist_freshness,
    )
    semantic_scholar.process_swagger_specs(output_dir=output_dir)
    return source_ref


SNAPSHOT_SOURCE_HANDLERS: dict[str, SnapshotSourceHandler] = {
    "/codeatlas/cosmograph": SnapshotSourceHandler(
        library_id="/codeatlas/cosmograph",
        probe_source_ref=cosmograph.probe_source_ref,
        fetch_into=cosmograph.fetch_from_site,
    ),
    "/codeatlas/1password-developer": SnapshotSourceHandler(
        library_id="/codeatlas/1password-developer",
        probe_source_ref=onepassword_developer.probe_source_ref,
        fetch_into=onepassword_developer.fetch_docs,
    ),
    "/codeatlas/pubtator3": SnapshotSourceHandler(
        library_id="/codeatlas/pubtator3",
        probe_source_ref=pubtator3.probe_source_ref,
        fetch_into=pubtator3.fetch,
    ),
    "/codeatlas/semantic-scholar-api": SnapshotSourceHandler(
        library_id="/codeatlas/semantic-scholar-api",
        probe_source_ref=semantic_scholar.probe_source_ref,
        fetch_into=_fetch_semantic_scholar_snapshot,
    ),
}


async def _resolve_maybe_async(value: str | Awaitable[str]) -> str:
    if inspect.isawaitable(value):
        return await value
    return value


def supports_snapshot_automation(library_id: str) -> bool:
    return library_id in SNAPSHOT_SOURCE_HANDLERS


async def get_snapshot_source_ref(library_id: str) -> str | None:
    handler = SNAPSHOT_SOURCE_HANDLERS.get(library_id)
    if handler is None:
        return None
    source_ref = (await _resolve_maybe_async(handler.probe_source_ref())).strip()
    return source_ref or None


def _temporary_dir_prefix(library_id: str) -> str:
    normalized = library_id.strip("/").replace("/", "-")
    return f"repowise-{normalized}-"


async def refresh_snapshot_library(library_id: str) -> SnapshotPublishResult:
    handler = SNAPSHOT_SOURCE_HANDLERS.get(library_id)
    if handler is None:
        raise ValueError(f"No snapshot automation registered for {library_id}")

    with TemporaryDirectory(prefix=_temporary_dir_prefix(library_id)) as temp_dir:
        output_dir = Path(temp_dir)
        source_ref = await _resolve_maybe_async(
            handler.fetch_into(output_dir=output_dir, persist_freshness=False)
        )
        markdown_files = list(output_dir.rglob("*.md"))
        if not markdown_files:
            raise RuntimeError(f"{library_id} fetch completed without markdown output")
        return await publish_snapshot_directory(
            library_id,
            output_dir,
            source_ref=source_ref or None,
        )


async def bootstrap_missing_snapshot_libraries(
    libraries: Sequence[LibraryState] | None = None,
) -> dict[str, object]:
    current_libraries = list(libraries) if libraries is not None else await db.list_libraries()
    snapshot_libraries = [
        library
        for library in current_libraries
        if library.source_type == LibrarySourceType.SNAPSHOT
        and supports_snapshot_automation(library.library_id)
    ]

    attempted = 0
    published_libraries: list[str] = []
    failed_libraries: dict[str, str] = {}

    for library in snapshot_libraries:
        if await db.get_snapshot_state(library.library_id) is not None:
            continue
        attempted += 1
        try:
            result = await refresh_snapshot_library(library.library_id)
        except Exception as exc:
            message = str(exc)
            failed_libraries[library.library_id] = message
            await db.update_library_status(
                library.library_id,
                status=LibraryStatus.ERROR,
                error_message=message,
                last_freshness_state="unknown",
                last_freshness_error=message,
            )
            logger.exception("Snapshot bootstrap failed for %s", library.library_id)
            continue

        published_libraries.append(library.library_id)
        logger.info(
            "Bootstrapped snapshot source %s with %d files",
            result.library_id,
            result.file_count,
        )

    return {
        "configured": len(snapshot_libraries),
        "attempted": attempted,
        "published": len(published_libraries),
        "failed": len(failed_libraries),
        "published_libraries": published_libraries,
        "failed_libraries": failed_libraries,
    }
