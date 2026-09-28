"""Async health owns its worker threads through failure and cancellation."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from repowise.core.analysis.health.duplication import DuplicationReport
from repowise.core.analysis.health.engine import FileComplexity, HealthAnalyzer
from repowise.core.cancellation import (
    CancellationToken,
    PipelineCancelled,
    check_cancelled,
    get_active_token,
    set_active_token,
)


def _analyzer(paths=("first.py",)):
    return HealthAnalyzer(None, parsed_files=[
        SimpleNamespace(file_info=SimpleNamespace(path=path, language="python"))
        for path in paths
    ])


@pytest.mark.parametrize("failure", ["progress", "cancel", "parent"])
async def test_failure_drains_duplicate_worker_without_cancelling_caller(monkeypatch, failure):
    loop = asyncio.get_running_loop()
    started, finished = asyncio.Event(), threading.Event()
    release = threading.Event()
    walked = asyncio.Event()
    caller = CancellationToken()
    restored = []
    analyzer = _analyzer()

    def duplicate(*_args, **_kwargs):
        loop.call_soon_threadsafe(started.set)
        try:
            while not release.wait(0.01):
                check_cancelled()
            return DuplicationReport()
        finally:
            finished.set()

    def step(_path):
        walked.set()
        if failure == "progress":
            raise RuntimeError("progress failed")

    async def analyze():
        previous = get_active_token()
        set_active_token(caller)
        try:
            return await analyzer.analyze_async(on_step=step)
        finally:
            restored.append(get_active_token())
            set_active_token(previous)

    monkeypatch.setattr("repowise.core.analysis.health.engine.detect_clones", duplicate)
    monkeypatch.setattr(analyzer, "_walk", lambda *_: FileComplexity(functions=[], classes=[]))
    task = asyncio.create_task(analyze())
    try:
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.wait_for(walked.wait(), 5)
        if failure == "cancel":
            task.cancel()
        elif failure == "parent":
            caller.cancel()
        expected = {"progress": RuntimeError, "cancel": asyncio.CancelledError,
                    "parent": PipelineCancelled}[failure]
        with pytest.raises(expected):
            await asyncio.wait_for(task, 5)
        worker_finished_before_return = finished.is_set()
    finally:
        release.set()
        await asyncio.to_thread(finished.wait, 5)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert worker_finished_before_return
    assert caller.cancelled is (failure == "parent")
    assert restored == [caller]


async def test_progress_failure_drains_running_walk_and_cancels_queued_walks(monkeypatch):
    second_started, release = threading.Event(), threading.Event()
    second_finished = threading.Event()
    progress_failed = asyncio.Event()
    analyzer = _analyzer(("first.py", "second.py", *(f"queued{i}.py" for i in range(20))))
    walked = []

    def walk(pf, _vocab):
        walked.append(pf.file_info.path)
        if pf.file_info.path == "first.py":
            assert second_started.wait(5)
        elif pf.file_info.path == "second.py":
            second_started.set()
            try:
                assert release.wait(5)
            finally:
                second_finished.set()
        return FileComplexity(functions=[], classes=[])

    def step(_path):
        progress_failed.set()
        raise RuntimeError("progress failed")

    monkeypatch.setattr(analyzer, "_walk", walk)
    task = asyncio.create_task(analyzer.analyze_async(
        {"disabled_biomarkers": ["dry_violation"]}, on_step=step, max_workers=2,
    ))
    try:
        await asyncio.wait_for(progress_failed.wait(), 5)
        # Flush the failure notification and cancellation callbacks without
        # imposing a timing budget on the worker.
        for _ in range(5):
            await asyncio.sleep(0)
        returned_before_worker = task.done()
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="progress failed"):
            await asyncio.wait_for(task, 5)
        await asyncio.to_thread(second_finished.wait, 5)
    assert not returned_before_worker
    assert second_finished.is_set()
    assert "queued19.py" not in walked


@pytest.mark.parametrize("failure", ["progress", "cancel"])
async def test_repeated_cancellation_preserves_worker_drain(monkeypatch, failure):
    loop = asyncio.get_running_loop()
    started = threading.Event()
    cleanup = asyncio.Event()
    release, finished = threading.Event(), threading.Event()
    analyzer = _analyzer()

    def duplicate(*_args, **_kwargs):
        started.set()
        try:
            while not release.wait(0.01):
                check_cancelled()
            return DuplicationReport()
        finally:
            loop.call_soon_threadsafe(cleanup.set)
            release.wait(5)
            finished.set()

    def walk(*_args):
        assert started.wait(5)
        return FileComplexity(functions=[], classes=[])

    def step(_path):
        if failure == "progress":
            raise RuntimeError("progress failed")

    monkeypatch.setattr("repowise.core.analysis.health.engine.detect_clones", duplicate)
    monkeypatch.setattr(analyzer, "_walk", walk)
    task = asyncio.create_task(analyzer.analyze_async(on_step=step))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        if failure == "cancel":
            task.cancel()
        await asyncio.wait_for(cleanup.wait(), 5)
        task.cancel()
        await asyncio.sleep(0)
        returned_during_cleanup = task.done()
    finally:
        release.set()
        expected = RuntimeError if failure == "progress" else asyncio.CancelledError
        with pytest.raises(expected):
            await asyncio.wait_for(task, 5)
        await asyncio.to_thread(finished.wait, 5)
    assert not returned_during_cleanup
    assert finished.is_set()


@pytest.mark.parametrize("setup", ["empty", "cache_failure"])
async def test_no_duplicate_worker_before_prewalk_is_ready(monkeypatch, setup):
    analyzer = _analyzer()
    duplicate = Mock(return_value=DuplicationReport())
    monkeypatch.setattr("repowise.core.analysis.health.engine.detect_clones", duplicate)
    if setup == "empty":
        report = await analyzer.analyze_async(changed_files=[])
        assert report.metrics == []
    else:
        analyzer._walk_cache = SimpleNamespace(load=Mock(side_effect=RuntimeError("cache failed")))
        with pytest.raises(RuntimeError, match="cache failed"):
            await analyzer.analyze_async()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    duplicate.assert_not_called()


async def test_duplicate_failure_preserves_health_fallback(tmp_path, monkeypatch):
    from repowise.core.ingestion import ASTParser, FileTraverser, GraphBuilder

    source = b"def increment(value):\n    return value + 1\n"
    (tmp_path / "app.py").write_bytes(source)
    info = next(iter(FileTraverser(tmp_path).traverse()))
    parsed = ASTParser().parse_file(info, source)
    builder = GraphBuilder(repo_path=tmp_path)
    builder.add_file(parsed)
    graph = builder.build()
    expected = HealthAnalyzer(graph, parsed_files=[parsed], repo_root=tmp_path).analyze()
    monkeypatch.setattr(
        "repowise.core.analysis.health.engine.detect_clones",
        Mock(side_effect=RuntimeError("detector unavailable")),
    )
    caller = CancellationToken()
    previous = get_active_token()
    set_active_token(caller)
    try:
        actual = await HealthAnalyzer(
            graph, parsed_files=[parsed], repo_root=tmp_path,
        ).analyze_async()
        assert get_active_token() is caller
        assert not caller.cancelled
    finally:
        set_active_token(previous)
    assert actual.metrics == expected.metrics
    assert actual.findings == expected.findings
    assert actual.kpis == expected.kpis
