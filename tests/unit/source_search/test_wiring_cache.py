"""The workspace coordinator cache: what it keeps, what it closes, and when.

The stores are fakes here, deliberately, for the reason ``test_coordinator``
gives about its own: what is under test is the *lifecycle policy* — which
object is handed out, which is retired, what a second caller sees while a first
is still building — and none of that is a property of LanceDB or FTS5. Real
stores would also make the orderings unpinnable, because the invariants below
are about which coroutine observes what at which await, and a real store's
internal awaits are not ours to schedule.

The build path over a real published generation is covered where the real
generation already exists: ``test_lifecycle`` has the toy corpus and the
manifest flip, and asserts there that a rebuilt workspace reader answers from
the newer one.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from repowise.server import source_search_wiring as w

_G1 = ("g1", 1, "recipe", "tbl", "fts.db")
_G2 = ("g2", 2, "recipe", "tbl", "fts.db")


class _Store:
    """A stand-in wiki vector store. A real one is a class instance, so this is
    weak-referenceable; ``object()`` is not, and would take the fallback path."""


class _FakeCoordinator:
    """Fails its search once closed, which is what the real stores do."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.closed = False

    async def search(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError("Cannot operate on a closed database.")
        return {"results": [], "_meta": {}}

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _clean_caches():
    w.reset_for_tests()
    yield
    w.reset_for_tests()


def _ctx(alias: str, path: str, store: object, *, ready: asyncio.Event | None = None):
    if ready is None:
        ready = asyncio.Event()
        ready.set()
    return SimpleNamespace(
        alias=alias,
        path=Path(path),
        vector_store=store,
        fts=object(),
        vector_store_ready=ready,
    )


def _fixed(monkeypatch, generation=_G1):
    """Patch the manifest key to a mutable holder, and count every build."""
    state = {"gen": generation}
    built: list[_FakeCoordinator] = []

    monkeypatch.setattr(w, "_manifest_key", lambda path: state["gen"])

    def fake_build(path, wiki_vectors, wiki_fts):
        built.append(_FakeCoordinator(f"c{len(built)}"))
        return built[-1]

    monkeypatch.setattr(w, "_build", fake_build)
    return state, built


# ---------------------------------------------------------------------------
# The cache itself
# ---------------------------------------------------------------------------


async def test_a_second_query_on_one_generation_reuses_the_open_coordinator(monkeypatch):
    _, built = _fixed(monkeypatch)
    ctx = _ctx("alpha", "/ws/alpha", _Store())

    first = await w.context_coordinator(ctx)
    second = await w.context_coordinator(ctx)

    assert first is second
    assert len(built) == 1


async def test_a_published_generation_is_rebuilt_from(monkeypatch):
    state, built = _fixed(monkeypatch)
    ctx = _ctx("alpha", "/ws/alpha", _Store())

    first = await w.context_coordinator(ctx)
    state["gen"] = _G2
    second = await w.context_coordinator(ctx)

    assert second is not first
    assert len(built) == 2
    assert w._ctx_coordinators[Path("/ws/alpha")][1] == _G2


async def test_a_reloaded_context_rebuilds_even_on_one_generation(monkeypatch):
    """The registry hands back fresh store objects; the old ones are closed."""
    _, built = _fixed(monkeypatch)

    first = await w.context_coordinator(_ctx("alpha", "/ws/alpha", _Store()))
    second = await w.context_coordinator(_ctx("alpha", "/ws/alpha", _Store()))

    assert second is not first
    assert len(built) == 2


async def test_concurrent_first_queries_build_one_coordinator(monkeypatch):
    _, built = _fixed(monkeypatch)
    ctx = _ctx("alpha", "/ws/alpha", _Store())

    out = await asyncio.gather(*(w.context_coordinator(ctx) for _ in range(4)))

    assert len(built) == 1
    assert all(item is built[0] for item in out)


async def test_stores_still_loading_stay_on_the_stock_path(monkeypatch):
    _, built = _fixed(monkeypatch)
    monkeypatch.setattr(w, "_READY_TIMEOUT", 0.05)
    ctx = _ctx("alpha", "/ws/alpha", _Store(), ready=asyncio.Event())

    assert await w.context_coordinator(ctx) is None
    assert built == []


# ---------------------------------------------------------------------------
# M3 — a replaced coordinator may still be inside somebody's search()
# ---------------------------------------------------------------------------


async def test_a_replaced_coordinator_is_not_closed_under_its_caller(monkeypatch):
    """The invariant: replacement retires, it does not close.

    A caller takes the coordinator from the cache and then awaits ``search()``
    on it holding nothing, so a rebuild landing in between used to close its
    stores mid-query — surfacing as "Cannot operate on a closed database",
    raised out of ``search_codebase`` on the scoped lane, which has no guard
    around the search itself.
    """
    state, _ = _fixed(monkeypatch)
    ctx = _ctx("alpha", "/ws/alpha", _Store())

    borrowed = await w.context_coordinator(ctx)

    state["gen"] = _G2
    replacement = await w.context_coordinator(ctx)
    assert replacement is not borrowed

    assert borrowed.closed is False
    assert await borrowed.search("q", limit=5, mode="concept", base_meta={}) is not None


async def test_a_retired_coordinator_is_closed_once_its_grace_has_passed(monkeypatch):
    """Deferred, not leaked: the close still happens, on a later pass."""
    state, _ = _fixed(monkeypatch)
    monkeypatch.setattr(w, "_RETIRE_GRACE", 0.0)
    ctx = _ctx("alpha", "/ws/alpha", _Store())

    borrowed = await w.context_coordinator(ctx)
    state["gen"] = _G2
    await w.context_coordinator(ctx)

    # The sweep runs at the head of the next call, so the retirement outlives
    # the cycle that made it however short the grace.
    await w.context_coordinator(ctx)

    assert borrowed.closed is True
    assert w._retired == []


async def test_retirements_cannot_pile_up_without_bound(monkeypatch):
    """Past the hard cap the oldest are closed early — memory beats the grace."""
    state, built = _fixed(monkeypatch)
    monkeypatch.setattr(w, "_RETIRE_MAX", 2)
    ctx = _ctx("alpha", "/ws/alpha", _Store())

    for sequence in range(6):
        state["gen"] = ("g", sequence, "recipe", "tbl", "fts.db")
        await w.context_coordinator(ctx)

    assert len(w._retired) <= 2
    assert sum(1 for coordinator in built if coordinator.closed) >= 3


# ---------------------------------------------------------------------------
# M4 — the cache has to learn about removal, and has to have a ceiling
# ---------------------------------------------------------------------------


async def test_a_removed_source_index_evicts_and_retires_its_coordinator(monkeypatch):
    """A vanished manifest used to early-return before any cache maintenance,
    pinning an open reader onto a generation that no longer exists."""
    state, _ = _fixed(monkeypatch)
    monkeypatch.setattr(w, "_RETIRE_GRACE", 0.0)
    ctx = _ctx("alpha", "/ws/alpha", _Store())

    coordinator = await w.context_coordinator(ctx)
    assert coordinator is not None

    state["gen"] = None
    assert await w.context_coordinator(ctx) is None
    assert Path("/ws/alpha") not in w._ctx_coordinators

    assert await w.context_coordinator(ctx) is None
    assert coordinator.closed is True


async def test_the_cache_is_bounded_and_evicts_least_recently_used(monkeypatch):
    _fixed(monkeypatch)
    monkeypatch.setattr(w, "_CTX_CACHE_MAX", 3)

    contexts = [_ctx(f"r{i}", f"/ws/r{i}", _Store()) for i in range(5)]
    for ctx in contexts:
        await w.context_coordinator(ctx)

    assert len(w._ctx_coordinators) == 3
    assert set(w._ctx_coordinators) == {Path("/ws/r2"), Path("/ws/r3"), Path("/ws/r4")}


async def test_reuse_keeps_a_repo_out_of_the_eviction_queue(monkeypatch):
    _fixed(monkeypatch)
    monkeypatch.setattr(w, "_CTX_CACHE_MAX", 2)

    alpha = _ctx("alpha", "/ws/alpha", _Store())
    beta = _ctx("beta", "/ws/beta", _Store())
    gamma = _ctx("gamma", "/ws/gamma", _Store())

    await w.context_coordinator(alpha)
    await w.context_coordinator(beta)
    await w.context_coordinator(alpha)  # alpha is now the recently used one
    await w.context_coordinator(gamma)

    assert set(w._ctx_coordinators) == {Path("/ws/alpha"), Path("/ws/gamma")}


async def test_reset_for_tests_closes_what_it_drops(monkeypatch):
    """Dropping the references without closing leaks a table and a sidecar per
    repo across a suite, which fails on handles rather than on assertions."""
    _, built = _fixed(monkeypatch)
    await w.context_coordinator(_ctx("alpha", "/ws/alpha", _Store()))

    w.reset_for_tests()
    await asyncio.sleep(0)  # the close is scheduled; reset_for_tests is sync

    assert built[0].closed is True
    assert w._ctx_coordinators == {}


# ---------------------------------------------------------------------------
# M6 — decide on the generation you find, not the one you captured
# ---------------------------------------------------------------------------


async def test_a_waiter_decides_on_the_generation_it_finds_not_the_one_it_captured(
    monkeypatch,
):
    """A coroutine that slept between reading the manifest key and acting on it
    wakes into a world that moved: another request may already have built for a
    newer generation. Deciding on the captured key pops and closes that fresher
    coordinator — which a live request holds — and then caches its own build
    under a key that is already stale.
    """
    state, built = _fixed(monkeypatch)
    monkeypatch.setattr(w, "_READY_TIMEOUT", 5.0)
    store = _Store()

    # The waiter blocks on readiness, which is where it loses its footing.
    waiting = asyncio.Event()
    waiter_ctx = _ctx("alpha", "/ws/alpha", store, ready=waiting)
    settled_ctx = _ctx("alpha", "/ws/alpha", store)

    waiter = asyncio.create_task(w.context_coordinator(waiter_ctx))
    await asyncio.sleep(0)  # let it read _G1 and park on the event

    state["gen"] = _G2
    fresher = await w.context_coordinator(settled_ctx)
    assert fresher is not None

    waiting.set()
    woken = await waiter

    assert woken is fresher, "the waiter rebuilt over a coordinator it should have reused"
    assert len(built) == 1
    assert fresher.closed is False


# ---------------------------------------------------------------------------
# M7 — a readiness wait is a wait, not a queue
# ---------------------------------------------------------------------------


async def test_cold_repos_wait_concurrently_not_one_after_another(monkeypatch):
    """A global build lock held across the readiness wait cost a federated
    fan-out one full timeout per cold repo, in series."""
    _fixed(monkeypatch)
    monkeypatch.setattr(w, "_READY_TIMEOUT", 0.15)

    contexts = [
        _ctx(alias, f"/ws/{alias}", _Store(), ready=asyncio.Event())
        for alias in ("a", "b", "c")
    ]

    loop = asyncio.get_running_loop()
    started = loop.time()
    out = await asyncio.gather(*(w.context_coordinator(ctx) for ctx in contexts))
    elapsed = loop.time() - started

    assert out == [None, None, None]
    assert elapsed < 0.40, f"three cold repos took {elapsed:.2f}s; serialised would be ~0.45s"


async def test_two_callers_for_one_cold_repo_also_wait_concurrently(monkeypatch):
    """Which is why the wait happens before the build lock rather than under a
    per-repo one: two queries against the same cold repo are still one wait."""
    _fixed(monkeypatch)
    monkeypatch.setattr(w, "_READY_TIMEOUT", 0.15)

    ctx = _ctx("alpha", "/ws/alpha", _Store(), ready=asyncio.Event())

    loop = asyncio.get_running_loop()
    started = loop.time()
    out = await asyncio.gather(*(w.context_coordinator(ctx) for _ in range(3)))
    elapsed = loop.time() - started

    assert out == [None, None, None]
    assert elapsed < 0.30, f"three callers took {elapsed:.2f}s; serialised would be ~0.45s"


# ---------------------------------------------------------------------------
# L7 — a cosine is only a shared unit under one model
# ---------------------------------------------------------------------------


class _Embedder:
    dimensions = 8

    def __init__(self, model: str) -> None:
        self._model = model


def _identity(provider: str, model: str, dims: int = 8):
    from repowise.core.source_search.manifest import EmbedderIdentity

    return EmbedderIdentity(provider=provider, model=model, dims=dims)


def test_a_different_model_at_the_same_width_is_a_mismatch():
    stored = _identity("test_wiring_cache", "text-embed-a")
    assert w._embedder_identity_mismatch(_Embedder("text-embed-b"), stored)


def test_the_same_model_is_not_a_mismatch():
    stored = _identity("test_wiring_cache", "text-embed-a")
    assert w._embedder_identity_mismatch(_Embedder("text-embed-a"), stored) is None


def test_a_manifest_that_recorded_no_identity_is_not_a_mismatch():
    """Legacy manifests read back as empty strings. Refusing on a fact nobody
    wrote down would lock the lane out of every index built before the field."""
    assert w._embedder_identity_mismatch(_Embedder("text-embed-a"), _identity("", "")) is None


# ---------------------------------------------------------------------------
# The store-identity token
# ---------------------------------------------------------------------------


def test_a_token_whose_store_was_collected_matches_nothing():
    """Which is why the token is weak rather than an address.

    The registry disposes a context and its stores are freed; the next
    allocation can land on the same address, and an ``id()`` comparison then
    reports a brand-new store as the one already built from — handing back a
    coordinator over stores that were closed with the old context.
    """
    import gc

    store = _Store()
    token = w._store_token(store)
    assert w._same_store(token, store)

    del store
    gc.collect()

    assert token() is None
    assert w._same_store(token, _Store()) is False
