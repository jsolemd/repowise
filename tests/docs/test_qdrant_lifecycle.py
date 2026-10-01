"""Docs collection reconciliation does no unnecessary writes and remains cancellable."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, models

from repowise.docs.chunking.models import ChunkType, DocChunk
from repowise.docs.indexer import collection, mutations
from repowise.docs.indexer.search import search_bm25


@pytest.fixture
def ready_collection():
    return models.CollectionInfo(
        status="green",
        optimizer_status="ok",
        segments_count=1,
        config=models.CollectionConfig(
            params=models.CollectionParams(
                vectors={"dense": models.VectorParams(size=768, distance="Cosine", on_disk=True)},
                sparse_vectors={
                    "bm25": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=True),
                        modifier="idf",
                    )
                },
            ),
            hnsw_config=models.HnswConfig(
                m=16, ef_construct=100, full_scan_threshold=10000, on_disk=True
            ),
            optimizer_config=models.OptimizersConfig(
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=0,
                flush_interval_sec=5,
                indexing_threshold=10000,
                memmap_threshold=10000,
            ),
            wal_config=models.WalConfig(wal_capacity_mb=32, wal_segments_ahead=0),
        ),
        payload_schema={
            name: models.PayloadIndexInfo(data_type="keyword", points=0)
            for name in ("library_id", "file_path", "chunk_type")
        },
    )


@pytest.fixture
def qdrant(ready_collection):
    client = MagicMock(spec=AsyncQdrantClient)
    client.collection_exists.return_value = True
    client.get_collection.return_value = ready_collection
    return client


async def test_unchanged_collection_performs_no_writes(qdrant):
    for _ in range(2):
        await collection.ensure_collection(qdrant)
    assert qdrant.collection_exists.await_count == 2
    assert qdrant.get_collection.await_count == 2
    qdrant.create_collection.assert_not_awaited()
    qdrant.update_collection.assert_not_awaited()
    qdrant.create_payload_index.assert_not_awaited()


async def test_interrupted_creation_recovers_only_missing_indexes(qdrant, ready_collection):
    ready_collection.payload_schema.clear()
    qdrant.collection_exists.side_effect = [False, True, True]

    async def create_index(*, collection_name, field_name, field_schema, wait):
        assert wait is True
        # The first run created one index before its connection failed.
        if field_name == "file_path" and qdrant.create_payload_index.await_count == 2:
            raise httpx.ReadTimeout("interrupted")
        ready_collection.payload_schema[field_name] = models.PayloadIndexInfo(
            data_type=field_schema,
            points=0,
        )

    qdrant.create_payload_index.side_effect = create_index
    with pytest.raises(httpx.ReadTimeout):
        await collection.ensure_collection(qdrant)
    await collection.ensure_collection(qdrant)
    await collection.ensure_collection(qdrant)

    qdrant.create_collection.assert_awaited_once()
    assert [call.kwargs["field_name"] for call in qdrant.create_payload_index.await_args_list] == [
        "library_id",
        "file_path",
        "file_path",
        "chunk_type",
    ]
    qdrant.update_collection.assert_not_awaited()


@pytest.mark.parametrize("policy", ["dense", "sparse", "hnsw", "optimizer"])
async def test_storage_drift_updates_only_the_changed_policy(qdrant, ready_collection, policy):
    current = ready_collection.model_copy(deep=True)
    if policy == "dense":
        current.config.params.vectors["dense"].on_disk = False
        expected = {"vectors_config": {"dense": models.VectorParamsDiff(on_disk=True)}}
    elif policy == "sparse":
        current.config.params.sparse_vectors["bm25"].index.on_disk = False
        expected = {
            "sparse_vectors_config": {
                "bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True), modifier="idf"
                )
            }
        }
    elif policy == "hnsw":
        current.config.hnsw_config.on_disk = False
        expected = {"hnsw_config": models.HnswConfigDiff(on_disk=True)}
    else:
        current.config.optimizer_config.memmap_threshold = 0
        expected = {
            "optimizers_config": models.OptimizersConfigDiff(
                indexing_threshold=10000, memmap_threshold=10000
            )
        }
    qdrant.get_collection.side_effect = [current, ready_collection]
    await collection.ensure_collection(qdrant)
    assert qdrant.update_collection.await_args.kwargs == {
        "collection_name": "doc-search",
        **expected,
    }
    await collection.ensure_collection(qdrant)
    qdrant.update_collection.assert_awaited_once()
    qdrant.create_payload_index.assert_not_awaited()


async def test_incompatible_vectors_are_not_recreated(qdrant, ready_collection):
    ready_collection.config.params.vectors["dense"].size = 1024
    with pytest.raises(ValueError, match="768D cosine"):
        await collection.ensure_collection(qdrant)
    qdrant.create_collection.assert_not_awaited()
    qdrant.update_collection.assert_not_awaited()
    qdrant.delete_collection.assert_not_awaited()


async def test_incompatible_payload_index_is_not_replaced(qdrant, ready_collection):
    ready_collection.payload_schema["library_id"].data_type = models.PayloadSchemaType.INTEGER
    with pytest.raises(ValueError, match=r"library_id.*keyword"):
        await collection.ensure_collection(qdrant)
    qdrant.create_payload_index.assert_not_awaited()
    qdrant.delete_payload_index.assert_not_awaited()


async def test_failed_collection_read_is_not_mistaken_for_missing(qdrant):
    qdrant.get_collection.side_effect = httpx.ReadTimeout("unavailable")
    with pytest.raises(httpx.ReadTimeout):
        await collection.ensure_collection(qdrant)
    qdrant.create_collection.assert_not_awaited()
    qdrant.update_collection.assert_not_awaited()


async def test_native_qdrant_search_yields_and_propagates_cancellation():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def transport(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = AsyncQdrantClient(
        url="http://qdrant.test",
        check_compatibility=False,
        transport=httpx.MockTransport(transport),
    )
    try:
        task = asyncio.create_task(search_bm25("/test/docs", "lookup", client=client))
        await asyncio.wait_for(entered.wait(), timeout=1)
        # The event loop remains available while the real SDK awaits transport.
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
    finally:
        await client.close()


async def test_runtime_closes_shared_client_once(monkeypatch, qdrant):
    monkeypatch.setattr(collection, "_CLIENT", qdrant)
    assert collection.get_qdrant_client() is qdrant
    await collection.close_qdrant_client()
    await collection.close_qdrant_client()
    qdrant.close.assert_awaited_once()
    assert collection._CLIENT is None


async def test_chunk_upload_awaits_every_batch_once(monkeypatch, qdrant):
    chunks = [
        DocChunk(
            library_id="/test/docs",
            file_path="api.md",
            commit_sha="test",
            chunk_type=ChunkType.DOC,
            content=f"section {i}",
            breadcrumb=[],
            section_anchor=f"section-{i}",
            source_url="",
        )
        for i in range(205)
    ]
    embed = AsyncMock(return_value=[[0.1, 0.2]] * len(chunks))
    monkeypatch.setattr(mutations, "embed_documents", embed)
    assert await mutations.upsert_chunks(chunks, qdrant) == len(chunks)
    embed.assert_awaited_once()
    uploads = qdrant.upsert.await_args_list
    assert [len(call.kwargs["points"]) for call in uploads] == [100, 100, 5]
    assert [point.id for call in uploads for point in call.kwargs["points"]] == [
        chunk.id for chunk in chunks
    ]
    assert all(call.kwargs["wait"] is True for call in uploads)
