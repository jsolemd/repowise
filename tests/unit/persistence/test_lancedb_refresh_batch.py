"""``refresh_batch``: generation writes a page's vector only when it is stale.

Template pages are re-rendered on every run, and the generator used to embed
and upsert each one again whether or not anything about it had changed. On
2026-09-28 that wrote 4,320 identical rows into one repository's store, and
each write is a new Lance version. The store now keys every row by its
embedder and the exact text embedded, and skips a row whose key still holds.
"""

from __future__ import annotations

import pytest

from repowise.core.persistence.vector_store import embed_item
from repowise.core.persistence.vector_store.in_memory import InMemoryVectorStore
from repowise.core.persistence.vector_store.lancedb_store import LanceDBVectorStore
from repowise.core.providers.embedding.base import MockEmbedder

pytest.importorskip("lancedb")


class _CountingEmbedder(MockEmbedder):
    def __init__(self, model: str = "mock-a") -> None:
        self.model = model
        self.texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return await super().embed(texts)


def _item(page_id: str, *, content: str = "Body.", page_type: str = "file_page", **extra):
    item = embed_item(
        page_id,
        title=extra.get("title", page_id),
        page_type=page_type,
        target_path=f"{page_id}.py",
        summary=extra.get("summary", "Summary."),
        content=content,
        page_metadata=extra.get("page_metadata"),
    )
    assert item is not None
    return item


async def _version(store: LanceDBVectorStore) -> int:
    return await store._table.version()  # type: ignore[union-attr]


async def _vectors(store: LanceDBVectorStore) -> dict[str, list[float]]:
    rows = await store._table.query().select(["page_id", "vector"]).to_list()  # type: ignore[union-attr]
    return {row["page_id"]: list(row["vector"]) for row in rows}


@pytest.fixture
async def seeded(tmp_path):
    embedder = _CountingEmbedder()
    store = LanceDBVectorStore(str(tmp_path / "lance"), embedder)
    items = [_item("a"), _item("b")]
    assert await store.refresh_batch(items) == 2
    embedder.texts.clear()
    try:
        yield store, embedder, items
    finally:
        await store.close()


async def test_unchanged_pages_are_neither_embedded_nor_written(seeded) -> None:
    store, embedder, items = seeded
    before = await _version(store)

    assert await store.refresh_batch(items) == 0

    assert embedder.texts == []
    assert await _version(store) == before


async def test_a_changed_input_is_re_embedded(seeded) -> None:
    store, embedder, items = seeded

    changed = _item("a", content="A different body.")
    assert await store.refresh_batch([changed, items[1]]) == 1

    assert embedder.texts == [changed[1]]


async def test_vocabulary_is_part_of_the_input(seeded) -> None:
    """Not just the body: every field the recipe puts in the text counts."""
    store, embedder, items = seeded

    changed = _item("a", page_metadata={"file_vocabulary": "lance merge_insert"})
    assert await store.refresh_batch([changed, items[1]]) == 1

    assert embedder.texts == [changed[1]]


async def test_a_missing_row_is_populated(seeded) -> None:
    store, embedder, items = seeded
    await store.delete("b")

    assert await store.refresh_batch(items) == 1

    assert embedder.texts == [items[1][1]]
    assert await store.list_page_ids() == {"a", "b"}


async def test_a_different_model_re_embeds_everything(seeded, tmp_path) -> None:
    store, _embedder, items = seeded
    await store.close()
    other = _CountingEmbedder(model="mock-b")
    reopened = LanceDBVectorStore(str(tmp_path / "lance"), other)
    try:
        assert await reopened.refresh_batch(items) == 2
        assert len(other.texts) == 2
    finally:
        await reopened.close()


async def test_a_metadata_only_change_keeps_the_vector(seeded) -> None:
    """``page_type`` is a row column but not in the embedded text."""
    store, embedder, items = seeded
    vector_before = (await _vectors(store))["a"]

    retyped = _item("a", page_type="module_page")
    assert retyped[1] == items[0][1]
    assert await store.refresh_batch([retyped, items[1]]) == 0

    assert embedder.texts == []
    rows = await store._table.query().where("page_id = 'a'").to_list()  # type: ignore[union-attr]
    assert rows[0]["page_type"] == "module_page"
    assert list(rows[0]["vector"]) == vector_before
    # Written once, and not again on the next pass.
    before = await _version(store)
    assert await store.refresh_batch([retyped, items[1]]) == 0
    assert await _version(store) == before


async def test_a_table_from_before_the_key_is_re_embedded_once(tmp_path) -> None:
    """Rows written by an older build carry no key, so none can be trusted."""
    import lancedb
    import pyarrow as pa

    path = str(tmp_path / "lance")
    db = await lancedb.connect_async(path)
    legacy = pa.schema(
        [
            pa.field("page_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 8)),
            pa.field("title", pa.string()),
            pa.field("page_type", pa.string()),
            pa.field("target_path", pa.string()),
            pa.field("content_snippet", pa.string()),
        ]
    )
    table = await db.create_table("wiki_pages", schema=legacy)
    item = _item("a")
    await table.add(
        [
            {
                "page_id": "a",
                "vector": [0.0] * 8,
                "title": "a",
                "page_type": "file_page",
                "target_path": "a.py",
                "content_snippet": "Body.",
            }
        ]
    )

    embedder = _CountingEmbedder()
    store = LanceDBVectorStore(path, embedder)
    try:
        assert await store.refresh_batch([item]) == 1
        assert await store.refresh_batch([item]) == 0
        assert len(embedder.texts) == 1
    finally:
        await store.close()


async def test_embed_batch_still_always_writes(seeded) -> None:
    """``reindex`` and ``doctor --repair`` rely on ``embed_batch`` overwriting."""
    store, embedder, items = seeded

    await store.embed_batch(items)

    assert len(embedder.texts) == 2


async def test_an_ephemeral_store_embeds_every_page() -> None:
    embedder = _CountingEmbedder()
    store = InMemoryVectorStore(embedder)
    items = [_item("a"), _item("b")]

    assert await store.refresh_batch(items) == 2
    assert await store.refresh_batch(items) == 2
    assert len(embedder.texts) == 4
