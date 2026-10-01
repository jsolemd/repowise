"""What happens when the embedding server cannot embed, and who hears about it.

A query whose embedding fails is still answered from the keyword lanes: an error
would send an agent to grep, which is worse than full-text results. But a
keyword-only answer must never pass for a whole one, because an agent reports a
lexical miss to a person as an absence. So the failure is noted on the request
that hit it (:func:`note_keyword_only`), and the MCP wire layer puts a top-level
``warning`` first in that response.

Recovery belongs to the host, not to RepoWise. ``REPOWISE_EMBEDDER_ENSURE_COMMAND``
names a command that brings the server back or says why it is off. It prints
JSON ``{"state", "recovered", "detail", "remedy", "gaming", "seconds"}`` and
exits 0 when the server is serving (the caller retries), 3 when it is off by
design (gaming mode), and 1 on a fault it could not repair. RepoWise runs it
after an embed fails and relays what it says; with the variable unset, the
failure is reported without a recovery attempt.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

__all__ = [
    "ENSURE_COMMAND_ENV",
    "EmbedderUnavailableError",
    "EmbedderVerdict",
    "ensure_embedder",
    "keyword_only_notices",
    "keyword_only_warning",
    "note_keyword_only",
]

ENSURE_COMMAND_ENV = "REPOWISE_EMBEDDER_ENSURE_COMMAND"

#: Exit status of the ensure command, by contract.
_STATES = {0: "serving", 3: "off_by_design", 1: "fault"}
#: A restart waits for the model to load; past this the command is abandoned.
_ENSURE_TIMEOUT_S = 120.0
#: A verdict that the server is down is reused for this long, so a burst of
#: failing queries (a federated search fans out per repository) runs the
#: command once. Every query still tries the server first, so a resumed server
#: answers on the next query regardless.
_VERDICT_TTL_S = 30.0

_ensure_lock = threading.Lock()
_last_verdict: tuple[float, EmbedderVerdict] | None = None

_notices: ContextVar[list[str] | None] = ContextVar("repowise_keyword_only", default=None)


@dataclass(frozen=True, slots=True)
class EmbedderVerdict:
    """What the ensure command said. ``state`` is ``unknown`` when it said nothing usable."""

    state: str
    detail: str = ""
    remedy: str | None = None
    gaming: bool = False
    recovered: bool = False


class EmbedderUnavailableError(RuntimeError):
    """The embedding server could not embed, after a retry and any recovery."""

    def __init__(self, cause: str, verdict: EmbedderVerdict | None) -> None:
        self.cause = cause
        self.verdict = verdict
        self.warning = keyword_only_warning(cause, verdict)
        super().__init__(self.warning)


def keyword_only_warning(cause: str, verdict: EmbedderVerdict | None) -> str:
    """The sentence an agent reads first in a keyword-only response."""

    detail = (verdict.detail if verdict is not None else "").rstrip(".")
    if verdict is None or verdict.state == "unknown":
        reason = f"the embedding server failed ({cause})"
        if detail:
            reason += f" and recovery gave no verdict ({detail})"
    elif verdict.state == "off_by_design":
        reason = (
            "the embedding server is stopped for gaming mode"
            if verdict.gaming
            else f"the embedding server is off by design ({detail or cause})"
        )
    elif verdict.state == "serving":
        reason = f"the embedding server reports serving but still failed ({cause})"
    else:
        reason = f"the embedding server is down and could not be restarted ({detail or cause})"
    text = f"Keyword-only results: semantic search did not run because {reason}."
    if verdict is not None and verdict.remedy:
        # The remedy is the host's own sentence, e.g. "`solemd gaming resume` restores it."
        remedy = verdict.remedy.strip()
        text += f" {remedy}" if remedy.endswith(".") else f" {remedy}."
    return text + " A miss here is not evidence of absence."


def ensure_command() -> list[str] | None:
    raw = os.environ.get(ENSURE_COMMAND_ENV, "").strip()
    return shlex.split(raw) if raw else None


def _parse(returncode: int, stdout: str, stderr: str) -> EmbedderVerdict:
    try:
        reply = json.loads(stdout)
        if not isinstance(reply, dict):
            raise ValueError("not an object")
    except ValueError:
        tail = (stderr or stdout).strip().splitlines()[-1:] or [f"exit {returncode}"]
        return EmbedderVerdict("unknown", detail=tail[0][:200])
    return EmbedderVerdict(
        # The exit status is the contract's control signal; the JSON explains it.
        state=_STATES.get(returncode, "unknown"),
        detail=str(reply.get("detail") or "")[:300],
        remedy=str(reply["remedy"])[:300] if reply.get("remedy") else None,
        gaming=bool(reply.get("gaming")),
        recovered=bool(reply.get("recovered")),
    )


def _run_ensure(argv: list[str]) -> EmbedderVerdict:
    global _last_verdict
    with _ensure_lock:
        if _last_verdict is not None and time.monotonic() - _last_verdict[0] < _VERDICT_TTL_S:
            return _last_verdict[1]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=_ENSURE_TIMEOUT_S, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            verdict = EmbedderVerdict("unknown", detail=f"{argv[0]}: {exc}"[:200])
        else:
            verdict = _parse(proc.returncode, proc.stdout, proc.stderr)
        # A serving verdict is never reused: it means "retry now", once.
        _last_verdict = None if verdict.state == "serving" else (time.monotonic(), verdict)
        return verdict


async def ensure_embedder() -> EmbedderVerdict | None:
    """Ask the host to bring the embedding server back; ``None`` when no command is set."""

    argv = ensure_command()
    if argv is None:
        return None
    # A thread, so concurrent callers on any event loop share one run through
    # the lock, and a cancelled request cannot orphan a half-read subprocess.
    return await asyncio.to_thread(_run_ensure, argv)


@contextmanager
def keyword_only_notices() -> Iterator[list[str]]:
    """Collect this request's keyword-only warnings, across the tasks it spawns."""

    notices: list[str] = []
    token = _notices.set(notices)
    try:
        yield notices
    finally:
        _notices.reset(token)


def note_keyword_only(warning: str) -> None:
    """Record *warning* on the current request; outside one, nobody is listening."""

    notices = _notices.get()
    if notices is not None and warning not in notices:
        notices.append(warning)
