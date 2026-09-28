"""Unit tests for cooperative cancellation (issue #341, Ctrl-C half).

The signal-handler installation is interactive and not exercised here; these
cover the token mechanics, the BaseException contract that keeps broad
``except Exception`` guards from swallowing a cancellation, scope restoration,
and the duplication detector's polling of :func:`check_cancelled`.
"""

from __future__ import annotations

import asyncio
import signal
import threading
from types import SimpleNamespace

import pytest

from repowise.core.cancellation import (
    CancellationToken,
    PipelineCancelled,
    cancellation_scope,
    check_cancelled,
    get_active_token,
    set_active_token,
)


@pytest.fixture(autouse=True)
def _reset_token():
    """Never let a test leak its token into the next one."""
    set_active_token(None)
    yield
    set_active_token(None)


def test_check_cancelled_noop_without_token():
    assert get_active_token() is None
    check_cancelled()  # must not raise


def test_check_cancelled_noop_when_not_cancelled():
    token = CancellationToken()
    set_active_token(token)
    check_cancelled()  # armed but not flipped → no-op
    assert token.cancelled is False


def test_check_cancelled_raises_when_cancelled():
    token = CancellationToken()
    token.cancel()
    set_active_token(token)
    with pytest.raises(PipelineCancelled):
        check_cancelled()


def test_pipeline_cancelled_is_base_exception_not_exception():
    """Broad ``except Exception`` must not swallow a cancellation."""
    assert issubclass(PipelineCancelled, BaseException)
    assert not issubclass(PipelineCancelled, Exception)

    token = CancellationToken()
    token.cancel()
    set_active_token(token)
    caught_by_exception = False
    try:
        try:
            check_cancelled()
        except Exception:
            caught_by_exception = True
    except PipelineCancelled:
        pass
    assert caught_by_exception is False


def test_cancellation_scope_arms_and_restores():
    assert get_active_token() is None
    with cancellation_scope() as token:
        assert get_active_token() is token
        assert token.cancelled is False
    # Restored to the prior (absent) token on exit.
    assert get_active_token() is None


def test_cancellation_scope_restores_even_on_error():
    with pytest.raises(ValueError), cancellation_scope():
        raise ValueError("boom")
    assert get_active_token() is None


def test_detect_clones_bails_when_cancelled():
    from repowise.core.analysis.health.duplication.detector import detect_clones

    token = CancellationToken()
    token.cancel()
    set_active_token(token)

    files = [SimpleNamespace(file_info=SimpleNamespace(path="a.py", abs_path="a.py", language="python"), symbols=[])]
    with pytest.raises(PipelineCancelled):
        detect_clones(files)


async def test_concurrent_jobs_keep_their_own_worker_cancellation_token():
    loop = asyncio.get_running_loop()
    ready = [asyncio.Event(), asyncio.Event()]
    release = threading.Event()
    tokens = [CancellationToken(), CancellationToken()]

    def work(index):
        loop.call_soon_threadsafe(ready[index].set)
        assert release.wait(5), "test did not release its worker"
        check_cancelled()
        return get_active_token()

    async def job(index):
        previous = get_active_token()
        set_active_token(tokens[index])
        try:
            return await asyncio.to_thread(work, index)
        finally:
            set_active_token(previous)

    jobs = [asyncio.create_task(job(index)) for index in range(2)]
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in ready)), 5)
        tokens[0].cancel()
    finally:
        release.set()
        results = await asyncio.gather(*jobs, return_exceptions=True)
    assert isinstance(results[0], PipelineCancelled)
    assert results[1] is tokens[1]
    assert get_active_token() is None


async def test_cancelled_awaiter_cannot_disarm_its_running_worker():
    loop = asyncio.get_running_loop()
    ready = asyncio.Event()
    finished = loop.create_future()
    release = threading.Event()
    token = CancellationToken()

    def work():
        loop.call_soon_threadsafe(ready.set)
        assert release.wait(5), "test did not release its worker"
        try:
            check_cancelled()
        except PipelineCancelled:
            cancelled = True
        else:
            cancelled = False
        loop.call_soon_threadsafe(finished.set_result, cancelled)

    async def job():
        set_active_token(token)
        try:
            await asyncio.to_thread(work)
        finally:
            set_active_token(None)

    task = asyncio.create_task(job())
    try:
        await asyncio.wait_for(ready.wait(), 5)
        token.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert get_active_token() is None
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert await asyncio.wait_for(finished, 5)


def test_nested_cli_cancellation_scope_restores_its_parent():
    with cancellation_scope() as outer:
        with cancellation_scope() as inner:
            assert get_active_token() is inner
            inner.cancel()
            with pytest.raises(PipelineCancelled):
                check_cancelled()
        assert get_active_token() is outer
        check_cancelled()
    assert get_active_token() is None


async def test_cli_interrupt_handler_cancels_the_inherited_worker_token():
    with cancellation_scope() as token:
        assert await asyncio.to_thread(get_active_token) is token
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        with pytest.raises(PipelineCancelled):
            await asyncio.to_thread(check_cancelled)
    assert get_active_token() is None
