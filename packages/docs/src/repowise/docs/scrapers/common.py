"""Shared helpers for RepoWise snapshot-backed documentation scrapers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from repowise.docs import db
from repowise.docs.snapshot_publish import publish_snapshot_directory

FRESHNESS_PATH = Path(__file__).with_name("freshness.json")
SNAPSHOT_METADATA_NAME = ".codeatlas-snapshot.json"


def load_freshness() -> dict[str, Any]:
    if not FRESHNESS_PATH.exists():
        return {}
    return json.loads(FRESHNESS_PATH.read_text(encoding="utf-8"))


def save_freshness(data: dict[str, Any]) -> None:
    FRESHNESS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_freshness_entry(name: str, *, persist: bool = True, **fields: Any) -> None:
    if not persist:
        return
    data = load_freshness()
    entry = dict(data.get(name, {}))
    entry.update(fields)
    data[name] = entry
    save_freshness(data)


def write_snapshot_metadata(output_dir: Path, **fields: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / SNAPSHOT_METADATA_NAME
    metadata_path.write_text(
        json.dumps(fields, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_snapshot_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / SNAPSHOT_METADATA_NAME
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def publish_output_dir(
    library_id: str,
    output_dir: Path,
    *,
    source_ref: str | None = None,
    cleanup_output: bool = False,
) -> None:
    metadata = read_snapshot_metadata(output_dir)
    effective_source_ref = source_ref or metadata.get("source_ref")

    async def _run() -> None:
        await db.init_pool()
        try:
            result = await publish_snapshot_directory(
                library_id,
                output_dir,
                source_ref=effective_source_ref,
            )
        finally:
            await db.close_pool()

        disposition = (
            f", queued={result.queued_job_type}/{result.queue_disposition}"
            if result.queued_job_type
            else ""
        )
        print(
            f"Published {result.file_count} files to {library_id} "
            f"(changed={result.changed}, source_ref={result.source_ref[:16]}{disposition})"
        )

    try:
        asyncio.run(_run())
    finally:
        if cleanup_output:
            shutil.rmtree(output_dir, ignore_errors=True)


def compute_source_digest(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\n\0\n")
    return digest.hexdigest()
