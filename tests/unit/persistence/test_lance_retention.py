"""LanceDB version retention: what the rule removes, and that readers survive it."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, timedelta
from pathlib import Path

import pytest
from filelock import FileLock

from repowise.core import lance_retention
from repowise.core.lance_retention import (
    STAMP_FILENAME,
    VERSION_RETENTION,
    prune_versions,
    sweep_repositories,
)
from repowise.core.persistence.vector_store import LanceDBVectorStore, lancedb_store
from repowise.core.providers.embedding.base import MockEmbedder
from repowise.core.source_search.lifecycle import reconcile_lock_path
from repowise.core.update_lock import (
    release_update_lock,
    release_workspace_lock,
    try_acquire_update_lock,
    update_workspace_lock,
)

pytest.importorskip("lancedb")

# The versions on either side of the boundary are written this far apart, so
# the test can place "a retention window ago" between them. Lance cleans
# against its own clock a moment later, so the boundary keeps a margin, and the
# gap must clear it.
_GAP = lance_retention._CLOCK_MARGIN.total_seconds() + 0.3


def _page(store: LanceDBVectorStore, page_id: str, text: str):
    return store.embed_and_upsert(page_id, text, {"title": page_id, "target_path": f"{page_id}.py"})


async def _version_numbers(store: LanceDBVectorStore) -> list[int]:
    return [version["version"] for version in await store._table.list_versions()]


async def test_versions_superseded_past_the_window_are_removed(tmp_path):
    store = LanceDBVectorStore(str(tmp_path), embedder=MockEmbedder())
    for index in range(3):
        await _page(store, f"page{index}", f"first draft {index}")
    await asyncio.sleep(_GAP)
    await _page(store, "page0", "second draft")
    boundary = (await store._table.list_versions())[-1]
    await asyncio.sleep(_GAP)
    await _page(store, "page3", "written inside the window")
    before = await _version_numbers(store)

    # A window after the boundary version was written, the write after it is
    # still inside the window, so the boundary version was current then.
    now = boundary["timestamp"].astimezone(UTC) + VERSION_RETENTION + timedelta(seconds=_GAP / 2)
    result = await prune_versions(store._table, now=now)

    after = await _version_numbers(store)
    older = [version for version in before if version < boundary["version"]]
    assert older
    assert result.versions_removed == len(older)
    assert result.bytes_removed > 0
    assert set(before) - set(after) == set(older)
    assert min(after) == boundary["version"]
    assert await store.list_page_ids() == {"page0", "page1", "page2", "page3"}
    assert len(await store.search("second draft", limit=4)) == 4


async def test_a_table_with_nothing_old_enough_is_not_rewritten(tmp_path):
    # Compaction would copy the live rows into a version the window then keeps.
    store = LanceDBVectorStore(str(tmp_path), embedder=MockEmbedder())
    for index in range(3):
        await _page(store, f"page{index}", f"draft {index}")
    before = await _version_numbers(store)

    result = await prune_versions(store._table)

    assert (result.versions_removed, result.fragments_compacted) == (0, 0)
    assert await _version_numbers(store) == before
    assert await store.list_page_ids() == {"page0", "page1", "page2"}


async def test_a_long_held_reader_moves_past_removed_versions(tmp_path, monkeypatch):
    # The MCP server and dashboard hold a wiki handle for the process's life.
    # A refreshing handle reads current rows; a pinned one would read files
    # that cleanup has deleted. Interval 0 stands in for "the interval passed".
    monkeypatch.setattr(lancedb_store, "READ_CONSISTENCY_INTERVAL", timedelta(0))
    writer = LanceDBVectorStore(str(tmp_path), embedder=MockEmbedder())
    await _page(writer, "page0", "before the reader opened")
    reader = LanceDBVectorStore(str(tmp_path), embedder=MockEmbedder())
    assert await reader.list_page_ids() == {"page0"}

    await asyncio.sleep(_GAP)
    await _page(writer, "page1", "after the reader opened")
    latest = (await writer._table.list_versions())[-1]
    now = latest["timestamp"].astimezone(UTC) + VERSION_RETENTION
    assert (await prune_versions(writer._table, now=now)).versions_removed > 0

    assert await reader.list_page_ids() == {"page0", "page1"}
    assert len(await reader.search("after the reader opened", limit=2)) == 2


def _member(root: Path, name: str) -> Path:
    repo = root / name
    (repo / ".repowise" / "lancedb").mkdir(parents=True)
    return repo


async def _seed(repo: Path) -> None:
    store = LanceDBVectorStore(str(repo / ".repowise" / "lancedb"), embedder=MockEmbedder())
    await _page(store, "page0", "seed")
    await store.close()


async def test_sweep_skips_a_repository_whose_writer_holds_a_lock(tmp_path):
    held, free = _member(tmp_path, "held"), _member(tmp_path, "free")
    await _seed(held)
    await _seed(free)

    assert try_acquire_update_lock(held, None) is None
    try:
        first = await sweep_repositories(tmp_path, [held, free])
    finally:
        release_update_lock(held)
    assert [outcome.status for outcome in first] == ["busy", "swept"]
    assert not (held / ".repowise" / STAMP_FILENAME).exists()
    assert [Path(table.table).stem for table in first[1].tables] == ["wiki_pages"]

    publication = FileLock(reconcile_lock_path(held), timeout=0)
    reconcile_lock_path(held).parent.mkdir(parents=True, exist_ok=True)
    with publication:
        assert (await sweep_repositories(tmp_path, [held]))[0].status == "busy"

    assert (await sweep_repositories(tmp_path, [held]))[0].status == "swept"
    stamp = json.loads((held / ".repowise" / STAMP_FILENAME).read_text(encoding="utf-8"))
    assert "swept_at" in stamp
    assert [outcome.status for outcome in await sweep_repositories(tmp_path, [held, free])] == [
        "not_due",
        "not_due",
    ]


async def test_sweep_waits_for_a_workspace_update(tmp_path):
    repo = _member(tmp_path, "member")
    await _seed(repo)

    assert update_workspace_lock(tmp_path) is None
    try:
        assert (await sweep_repositories(tmp_path, [repo]))[0].status == "busy"
    finally:
        release_workspace_lock(tmp_path)
    assert (await sweep_repositories(tmp_path, [repo]))[0].status == "swept"
    assert not (tmp_path / ".repowise-workspace" / ".update.lock").exists()
