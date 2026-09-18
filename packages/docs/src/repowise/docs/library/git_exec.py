"""Low-level git subprocess helpers for doc-search."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path

from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)

_TRANSIENT_PATTERNS: list[str] = [
    "could not resolve host",
    "couldn't connect to server",
    "connection refused",
    "connection timed out",
    "connection reset",
    "failed to connect",
    "network is unreachable",
    "name or service not known",
    "no route to host",
    "ssl",
    "tls",
    "rate limit",
    "service unavailable",
    "internal server error",
    "unexpected disconnect",
    "the remote end hung up",
    "early eof",
    "timed out",
]

_PERMANENT_PATTERNS: list[str] = [
    "couldn't find remote ref",
    "remote ref .* not found",
    "repository not found",
    "could not read from remote",
    "authentication failed",
    "permission denied",
    "invalid username or password",
]

_PERMANENT_RE = re.compile("|".join(_PERMANENT_PATTERNS), re.IGNORECASE)

MAX_CONCURRENT_GIT_OPS = 2
_git_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]
_git_semaphores = weakref.WeakKeyDictionary()
_askpass_script: Path | None = None


def _is_transient_error(stderr: str) -> bool:
    """Check if a git error is transient and worth retrying."""
    stderr_lower = stderr.lower()
    return any(pattern in stderr_lower for pattern in _TRANSIENT_PATTERNS)


def _is_permanent_error(stderr: str) -> bool:
    """Check if a git error is permanent and should not be retried."""
    return bool(_PERMANENT_RE.search(stderr))


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the git operations semaphore."""
    loop = asyncio.get_running_loop()
    sem = _git_semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(MAX_CONCURRENT_GIT_OPS)
        _git_semaphores[loop] = sem
    return sem


def _get_askpass_script() -> Path:
    """Get or create an askpass script for non-interactive git authentication."""
    global _askpass_script
    if _askpass_script is not None:
        return _askpass_script

    script_dir = Path(tempfile.gettempdir()) / "doc_search"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "git_askpass.sh"

    if not script_path.exists():
        script_path.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            '  *Username*) echo "${DOC_SEARCH_GIT_USERNAME:-x-access-token}" ;;\n'
            '  *Password*) echo "${DOC_SEARCH_GIT_PASSWORD:-}" ;;\n'
            '  *) echo "" ;;\n'
            "esac\n",
            encoding="utf-8",
        )

    script_path.chmod(0o700)
    _askpass_script = script_path
    return script_path


@dataclass
class GitError(Exception):
    """Git operation error."""

    command: str
    return_code: int
    stderr: str

    def __str__(self) -> str:
        return f"Git command '{self.command}' failed (code {self.return_code}): {self.stderr}"


async def run_git(
    *args: str,
    cwd: Path | None = None,
    timeout: float = 300,
    retries: int = 3,
    backoff_base: float = 2.0,
) -> str:
    """Run a git command with rate limiting and exponential backoff."""
    settings = get_settings()
    semaphore = _get_semaphore()
    cmd = ["git", *args]

    for attempt in range(retries):
        async with semaphore:
            try:
                env = os.environ.copy()
                if settings.github_token:
                    env["DOC_SEARCH_GIT_USERNAME"] = "x-access-token"
                    env["DOC_SEARCH_GIT_PASSWORD"] = settings.github_token
                    env["GIT_ASKPASS"] = str(_get_askpass_script())
                    env["GIT_ASKPASS_REQUIRE"] = "force"
                    env["GIT_TERMINAL_PROMPT"] = "0"

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    start_new_session=os.name == "posix",
                )

                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except (TimeoutError, asyncio.CancelledError) as exc:
                    # Git may have helpers holding its pipes open. Stop the
                    # whole group, then drain and reap before releasing the
                    # concurrency slot or propagating cancellation.
                    try:
                        if os.name == "posix":
                            os.killpg(proc.pid, signal.SIGKILL)
                        else:
                            proc.kill()
                    except ProcessLookupError:
                        pass
                    await proc.communicate()
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    raise GitError(
                        command=" ".join(cmd),
                        return_code=-1,
                        stderr=f"Command timed out after {timeout}s",
                    ) from None

                stdout_str = stdout.decode("utf-8", errors="replace")
                stderr_str = stderr.decode("utf-8", errors="replace")

                if proc.returncode != 0:
                    if _is_permanent_error(stderr_str):
                        logger.warning(
                            "Git permanent error (attempt %s/%s): %s",
                            attempt + 1,
                            retries,
                            stderr_str.strip()[:120],
                        )
                        raise GitError(
                            command=" ".join(cmd),
                            return_code=proc.returncode or -1,
                            stderr=stderr_str,
                        )

                    if _is_transient_error(stderr_str) and attempt < retries - 1:
                        wait_time = backoff_base ** (attempt + 1)
                        logger.warning(
                            "Git transient error, waiting %ss before retry (attempt %s/%s): %s",
                            wait_time,
                            attempt + 1,
                            retries,
                            stderr_str.strip()[:120],
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    raise GitError(
                        command=" ".join(cmd),
                        return_code=proc.returncode or -1,
                        stderr=stderr_str,
                    )

                return stdout_str

            except GitError:
                raise
            except Exception as exc:
                raise GitError(command=" ".join(cmd), return_code=-1, stderr=str(exc)) from exc

    raise GitError(command=" ".join(cmd), return_code=-1, stderr=f"Failed after {retries} retries")
