"""Durable retry ledger for post-transaction search-index cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from repowise.core.fsutils import atomic_write_text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_KINDS = ("fts", "vectors")


def _path(repo_path: Path) -> Path:
    return repo_path / ".repowise" / "cleanup-debt.json"


def load_cleanup_debt(repo_path: Path) -> dict[str, set[str]]:
    path = _path(repo_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, TypeError):
        raw = {}
    return {kind: set(raw.get(kind) or []) for kind in _KINDS}


def record_cleanup_debt(repo_path: Path, kind: str, page_ids: set[str]) -> None:
    if not page_ids:
        return
    debt = load_cleanup_debt(repo_path)
    debt[kind].update(page_ids)
    path = _path(repo_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        json.dumps({key: sorted(values) for key, values in debt.items()}, indent=2),
    )


def clear_cleanup_debt(repo_path: Path, kind: str, page_ids: set[str]) -> None:
    if not page_ids:
        return
    debt = load_cleanup_debt(repo_path)
    debt[kind].difference_update(page_ids)
    path = _path(repo_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        path,
        json.dumps({key: sorted(values) for key, values in debt.items()}, indent=2),
    )


async def exclude_live_cleanup_ids(
    repo_path: Path, engine: AsyncEngine, page_ids: set[str]
) -> set[str]:
    """Keep restored pages out of post-commit index deletion.

    Page IDs survive deletion and regeneration. Retry debt therefore records an
    intent to recheck, not unconditional permission to delete. Call only after
    the page transaction commits, under the caller's update ownership. A failed
    authority read leaves the debt intact and must stop index deletion.
    """
    if not page_ids:
        return set()
    from sqlalchemy import select

    from repowise.core.persistence.models import Page

    live: set[str] = set()
    ids = sorted(page_ids)
    async with engine.connect() as connection:
        for offset in range(0, len(ids), 500):
            rows = await connection.execute(
                select(Page.id).where(
                    Page.id.in_(ids[offset : offset + 500]),
                    Page.freshness_status != "tombstone",
                )
            )
            live.update(rows.scalars())
    # A committed live page invalidates both kinds of historical deletion debt.
    debt = load_cleanup_debt(repo_path)
    for kind in _KINDS:
        clear_cleanup_debt(repo_path, kind, live & debt[kind])
    return page_ids - live
