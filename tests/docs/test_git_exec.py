"""Git timeouts and cancellation must release the process before returning."""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from repowise.docs.library import git_exec


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancelled"])
async def test_interrupted_git_is_reaped(monkeypatch, cancel):
    real_spawn = asyncio.create_subprocess_exec
    started = asyncio.Event()
    children = []

    async def spawn(*args, **kwargs):
        child = await real_spawn(
            sys.executable, "-c", "import time; time.sleep(60)", **kwargs
        )
        children.append(child)
        started.set()
        return child

    monkeypatch.setattr(git_exec, "get_settings", lambda: SimpleNamespace(github_token=""))
    monkeypatch.setattr(git_exec.asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(git_exec.run_git("status", timeout=0.05 if not cancel else 60))
    await started.wait()
    child = children[0]
    if cancel:
        task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError if cancel else git_exec.GitError):
            await task
        assert child.returncode is not None, "run_git returned before reaping its process"
        if os.name == "posix":
            with pytest.raises(ProcessLookupError):
                os.kill(child.pid, 0)
    finally:
        if child.returncode is None:
            child.kill()
        await child.communicate()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
@pytest.mark.parametrize("already_exited", [False, True])
async def test_timeout_kills_git_helpers_and_drains_pipes(monkeypatch, already_exited):
    child = SimpleNamespace(
        pid=123456, returncode=None,
        communicate=AsyncMock(side_effect=[TimeoutError, (b"", b"")]),
        kill=Mock(),
    )
    spawn = AsyncMock(return_value=child)
    kill_group = Mock(side_effect=ProcessLookupError if already_exited else None)
    monkeypatch.setattr(git_exec, "get_settings", lambda: SimpleNamespace(github_token=""))
    monkeypatch.setattr(git_exec.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(git_exec.os, "killpg", kill_group)

    with pytest.raises(git_exec.GitError, match="timed out"):
        await git_exec.run_git("fetch", timeout=0.01)

    assert spawn.call_args.kwargs["start_new_session"] is True
    kill_group.assert_called_once()
    assert kill_group.call_args.args[0] == child.pid
    assert child.communicate.await_count == 2, "cleanup must drain pipes and reap Git"
