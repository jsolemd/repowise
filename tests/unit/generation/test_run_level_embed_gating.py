"""Embed gating in ``_GenerationRun.run_level``.

Pages whose content was reused verbatim from the prior run (prompt-hash /
content-hash cache hit) already have an identical vector in any store that
persists across runs; re-embedding them re-bills the embedder for every
unchanged page on every update. run_level therefore skips them, but ONLY when
the store persists across runs: the in-memory store starts empty each run and
still needs every page embedded.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from repowise.core.generation.models import GeneratedPage
from repowise.core.generation.page_generator.orchestrate import _GenerationRun


def _page(page_id: str, *, reused: bool = False) -> GeneratedPage:
    now = datetime.now(UTC).isoformat()
    page = GeneratedPage(
        page_id=page_id,
        page_type="file_page",
        title=page_id,
        content=f"# {page_id}",
        source_hash="deadbeef",
        model_name="mock-model",
        provider_name="mock",
        input_tokens=0 if reused else 1,
        output_tokens=0 if reused else 1,
        cached_tokens=0,
        generation_level=2,
        target_path=page_id,
        created_at=now,
        updated_at=now,
    )
    if reused:
        page.metadata["reused_from_prior_run"] = True
    return page


class _RecordingStore:
    def __init__(self, *, persists: bool) -> None:
        self.persists_across_runs = persists
        self.batches: list[list[tuple]] = []

    async def refresh_batch(self, items):
        self.batches.append(list(items))
        return len(items)


def _fake_run(store) -> SimpleNamespace:
    return SimpleNamespace(
        semaphore=asyncio.Semaphore(4),
        job_system=None,
        job_id=None,
        on_page_done=None,
        on_page_ready=None,
        vector_store=store,
        completed_page_summaries={},
        timings=None,
    )


def _run_level(store) -> list[str]:
    async def _go():
        async def fresh():
            return _page("fresh.py")

        async def reused():
            return _page("reused.py", reused=True)

        run = _fake_run(store)
        await _GenerationRun.run_level(run, [("p1", fresh()), ("p2", reused())], level=2)
        return [pid for batch in store.batches for (pid, *_rest) in batch]

    return asyncio.run(_go())


def test_persistent_store_skips_reused_pages() -> None:
    store = _RecordingStore(persists=True)
    embedded = _run_level(store)
    assert "fresh.py" in embedded
    assert "reused.py" not in embedded


def test_ephemeral_store_still_embeds_reused_pages() -> None:
    """In-memory stores are rebuilt every run: reuse must not starve them."""
    store = _RecordingStore(persists=False)
    embedded = _run_level(store)
    assert "fresh.py" in embedded
    assert "reused.py" in embedded


def test_a_re_rendered_page_that_did_not_change_is_not_written_again(tmp_path) -> None:
    """Template pages are never marked reused, but re-render identically.

    Each run used to embed and upsert every one of them again: 4,320 identical
    rows in one store on 2026-09-28.
    """
    import pytest

    pytest.importorskip("lancedb")
    from repowise.core.persistence.vector_store import LanceDBVectorStore
    from repowise.core.providers.embedding.base import MockEmbedder

    calls: list[int] = []

    class _Counting(MockEmbedder):
        async def embed(self, texts):
            calls.append(len(texts))
            return await super().embed(texts)

    store = LanceDBVectorStore(str(tmp_path / "lance"), _Counting())

    async def _render_twice() -> tuple[int, int]:
        async def rendered():
            return _page("template.py")

        await _GenerationRun.run_level(_fake_run(store), [("p", rendered())], level=2)
        version = await store._table.version()
        await _GenerationRun.run_level(_fake_run(store), [("p", rendered())], level=2)
        return version, await store._table.version()

    first, second = asyncio.run(_render_twice())
    assert calls == [1]
    assert second == first


class _FailingStore:
    persists_across_runs = True

    async def refresh_batch(self, items):
        raise AttributeError("module 'lancedb' has no attribute 'connect_async'")


def test_a_failed_embed_is_counted_on_the_generator() -> None:
    """Callers read the count to refuse calling semantic search healthy.

    A warning string alone left init and update exiting 0 with a semantic
    index that held none of the pages.
    """
    warnings: list[str] = []
    gen = SimpleNamespace(embed_failed_pages=0)

    async def _go():
        async def fresh():
            return _page("fresh.py")

        run = _fake_run(_FailingStore())
        run.gen = gen
        run.on_warning = warnings.append
        await _GenerationRun.run_level(run, [("p1", fresh())], level=2)

    asyncio.run(_go())

    assert gen.embed_failed_pages == 1
    assert warnings and "Embedding failed for 1 page(s)" in warnings[0]
