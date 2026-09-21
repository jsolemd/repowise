"""The dense leg: schema, upsert semantics, and the dimension-change recreate."""

from __future__ import annotations

from dataclasses import replace

import pytest

from repowise.core.providers.embedding.base import MockEmbedder
from repowise.core.source_search.chunks import SymbolRecord, build_symbol_chunk
from repowise.core.source_search.generation import GenerationRef
from repowise.core.source_search.vector_store import (
    STORED_SNIPPET_CHARS,
    SourceChunkVectorStore,
)


class _WideEmbedder(MockEmbedder):
    """A second embedder with a different width, to force a table recreate."""

    dimensions = 16

    async def embed(self, texts: list[str]) -> list[list[float]]:
        narrow = await super().embed(texts)
        return [vec + vec for vec in narrow]


def _chunk(name: str, body: str = "def f():\n    pass", path: str | None = None):
    path = path or f"src/{name}.py"
    return build_symbol_chunk(
        SymbolRecord(
            symbol_id=f"{path}::{name}",
            file_path=path,
            name=name,
            qualified_name=name,
            kind="function",
            signature="",
            docstring=None,
            start_line=1,
            end_line=len(body.splitlines()),
            language="python",
        ),
        body.splitlines(),
    )


async def _store(tmp_path, embedder=None) -> SourceChunkVectorStore:
    return SourceChunkVectorStore(str(tmp_path / "lancedb"), embedder=embedder or MockEmbedder())


async def _embed(embedder, chunks):
    vectors = await embedder.embed([chunk.text for chunk in chunks])
    return list(zip(chunks, vectors, strict=True))


async def test_round_trip_returns_the_stored_metadata(tmp_path):
    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    store = await _store(tmp_path, embedder)
    chunks = [_chunk("alpha"), _chunk("beta", path="tests/test_beta.py")]
    assert await store.upsert(await _embed(embedder, chunks)) == 2
    assert await store.count() == 2

    query = (await embedder.embed([chunks[0].text]))[0]
    hits = await store.search_by_vector(query, limit=2)
    top = hits[0]
    assert top.chunk_id == "src/alpha.py::alpha"
    assert top.file_path == "src/alpha.py"
    assert top.name == "alpha"
    assert top.kind == "function"
    assert top.source == "symbol"
    assert top.is_test is False
    assert top.content_hash == chunks[0].content_hash
    assert top.snippet == chunks[0].text
    assert top.score == pytest.approx(1.0, abs=1e-5)
    await store.close()


async def test_the_test_flag_survives_the_round_trip(tmp_path):
    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    store = await _store(tmp_path, embedder)
    chunk = _chunk("beta", path="tests/test_beta.py")
    await store.upsert(await _embed(embedder, [chunk]))
    hits = await store.search_by_vector((await embedder.embed([chunk.text]))[0], limit=1)
    assert hits[0].is_test is True
    await store.close()


async def test_the_snippet_is_bounded(tmp_path):
    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    store = await _store(tmp_path, embedder)
    chunk = _chunk("big", body="\n".join("x" * 200 for _ in range(50)))
    await store.upsert(await _embed(embedder, [chunk]))
    hits = await store.search_by_vector((await embedder.embed([chunk.text]))[0], limit=1)
    assert len(hits[0].snippet) == STORED_SNIPPET_CHARS
    await store.close()


async def test_upsert_is_keyed_on_chunk_id(tmp_path):
    """Re-writing the same chunk id updates the row rather than adding one."""
    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    store = await _store(tmp_path, embedder)
    first = _chunk("alpha", body="def f():\n    return 1")
    await store.upsert(await _embed(embedder, [first]))
    second = _chunk("alpha", body="def f():\n    return 2")
    await store.upsert(await _embed(embedder, [second]))
    assert await store.count() == 1
    stored = await store.stored_vectors()
    assert stored["src/alpha.py::alpha"].content_hash == second.content_hash
    await store.close()


async def test_stored_vectors_are_readable_for_reuse(tmp_path):
    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    store = await _store(tmp_path, embedder)
    chunks = [_chunk("alpha"), _chunk("beta")]
    items = await _embed(embedder, chunks)
    await store.upsert(items)
    stored = await store.stored_vectors()
    assert set(stored) == {"src/alpha.py::alpha", "src/beta.py::beta"}
    for chunk, vector in items:
        entry = stored[chunk.chunk_id]
        assert entry.content_hash == chunk.content_hash
        assert entry.vector == pytest.approx(list(vector), abs=1e-6)
    await store.close()


async def test_reuse_reads_only_requested_hashes_across_files_and_batches(tmp_path):
    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    store = await _store(tmp_path, embedder)
    # More hashes than one predicate batch, and more duplicates than one result
    # batch: neither a query limit nor collecting the entire corpus is needed.
    chunks = [_chunk(f"symbol_{index}") for index in range(230)]
    items = await _embed(embedder, chunks)
    duplicates = [
        (replace(chunks[0], chunk_id=f"copy-{index}", file_path=f"other/{index}.py"), items[0][1])
        for index in range(300)
    ]
    await store.upsert([*duplicates, *items])
    requested = [chunk.content_hash for chunk in chunks[::2]]
    actual = await store.vectors_by_content_hash([*requested, requested[0], "missing'", ""])
    assert set(actual) == set(requested)
    for chunk, vector in items[::2]:
        assert actual[chunk.content_hash] == pytest.approx(vector, abs=1e-6)
    await store.close()


async def test_reuse_respects_bound_generation_during_staging(tmp_path):
    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    first = GenerationRef("first", 1)
    second = GenerationRef("second", 2)
    store = SourceChunkVectorStore(str(tmp_path / "lancedb"), embedder=embedder, generation=first)
    old = _chunk("alpha", body="def alpha():\n    return 1")
    new = _chunk("alpha", body="def alpha():\n    return 2")
    unrelated = _chunk("unrelated")
    await store.upsert(await _embed(embedder, [old, unrelated]))
    await store.stage_generation(
        second,
        close_paths=[old.file_path],
        items=await _embed(embedder, [new]),
        recipe_fingerprint="same-recipe",
        expected_count=2,
    )
    requested = [old.content_hash, new.content_hash]
    assert set(await store.vectors_by_content_hash(requested)) == {old.content_hash}
    await store.close()
    active = SourceChunkVectorStore(str(tmp_path / "lancedb"), embedder=embedder, generation=second)
    assert set(await active.vectors_by_content_hash(requested)) == {new.content_hash}
    await active.close()


async def test_empty_reuse_does_not_open_the_database(tmp_path, monkeypatch):
    store = await _store(tmp_path)

    async def unexpected_connection():
        pytest.fail("empty reuse must not open LanceDB")

    monkeypatch.setattr(store, "_ensure_connected", unexpected_connection)
    assert await store.vectors_by_content_hash([""]) == {}


async def test_delete_by_file_removes_only_that_file(tmp_path):
    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    store = await _store(tmp_path, embedder)
    await store.upsert(
        await _embed(embedder, [_chunk("alpha", path="src/a.py"), _chunk("beta", path="src/b.py")])
    )
    await store.delete_by_file(["src/a.py"])
    assert await store.count() == 1
    assert set(await store.stored_vectors()) == {"src/b.py::beta"}
    await store.close()


async def test_drop_leaves_an_empty_store(tmp_path):
    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    store = await _store(tmp_path, embedder)
    await store.upsert(await _embed(embedder, [_chunk("alpha")]))
    await store.drop()
    assert await store.count() == 0
    assert await store.stored_vectors() == {}
    await store.close()


async def test_a_dimension_change_recreates_the_table(tmp_path):
    """Switching embedders must drop the old vectors, not fail on write.

    Without the recreate every write lands deep inside LanceDB as an opaque IO
    error that never mentions dimensions.
    """
    pytest.importorskip("lancedb")
    narrow = MockEmbedder()
    store = SourceChunkVectorStore(str(tmp_path / "lancedb"), embedder=narrow)
    await store.upsert(await _embed(narrow, [_chunk("alpha"), _chunk("beta")]))
    assert await store.count() == 2
    await store.close()

    wide = _WideEmbedder()
    reopened = SourceChunkVectorStore(str(tmp_path / "lancedb"), embedder=wide)
    chunk = _chunk("gamma")
    await reopened.upsert(await _embed(wide, [chunk]))
    assert await reopened.count() == 1
    stored = await reopened.stored_vectors()
    assert set(stored) == {"src/gamma.py::gamma"}
    assert len(stored["src/gamma.py::gamma"].vector) == wide.dimensions
    await reopened.close()


async def test_an_empty_upsert_creates_nothing(tmp_path):
    pytest.importorskip("lancedb")
    store = await _store(tmp_path)
    assert await store.upsert([]) == 0
    assert await store.count() == 0
    assert await store.stored_vectors() == {}
    await store.close()


async def test_the_page_store_is_untouched(tmp_path):
    """Both tables share a directory; writing one must not disturb the other."""
    pytest.importorskip("lancedb")
    from repowise.core.persistence.vector_store import LanceDBVectorStore

    embedder = MockEmbedder()
    lance_dir = str(tmp_path / "lancedb")
    pages = LanceDBVectorStore(lance_dir, embedder=embedder)
    await pages.embed_and_upsert("page-1", "hello world", {"title": "Hello"})

    store = SourceChunkVectorStore(lance_dir, embedder=embedder)
    await store.upsert(await _embed(embedder, [_chunk("alpha")]))
    await store.close()

    assert await pages.list_page_ids() == {"page-1"}


async def test_nonlegacy_manifest_cannot_read_a_legacy_table(tmp_path):
    """A torn manifest/table migration is an integrity error, never 'current'."""

    pytest.importorskip("lancedb")
    embedder = MockEmbedder()
    legacy = await _store(tmp_path, embedder)
    await legacy.upsert(await _embed(embedder, [_chunk("alpha")]))
    await legacy.close()

    torn = SourceChunkVectorStore(
        str(tmp_path / "lancedb"),
        embedder=embedder,
        generation=GenerationRef("not-legacy", 1),
    )
    with pytest.raises(RuntimeError, match="non-legacy source manifest"):
        await torn.count()
    await torn.close()
