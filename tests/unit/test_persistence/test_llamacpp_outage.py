"""What LlamaCppEmbedder does when llama-server cannot embed.

An httpx MockTransport stands in for the server, and a small script stands in
for the host's ensure command, so neither a server nor the SoleMD CLI is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from repowise.core.providers.embedding import llamacpp, outage
from repowise.core.providers.embedding.llamacpp import LlamaCppEmbedder
from repowise.core.providers.embedding.outage import (
    ENSURE_COMMAND_ENV,
    EmbedderUnavailableError,
    keyword_only_notices,
)

_REAL_CLIENT = httpx.AsyncClient


class _Server:
    """llama-server answering /props and /v1/embeddings; down until *up* exists."""

    def __init__(self, up: Path, statuses: list[int] | None = None) -> None:
        self.up = up
        self.statuses = list(statuses or [])
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        if not self.up.exists():
            raise httpx.ConnectError("Connection refused", request=request)
        if self.statuses:
            return httpx.Response(self.statuses.pop(0), request=request)
        if request.url.path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 64}})
        inputs = json.loads(request.content)["input"]
        rows = [{"index": i, "embedding": [1.0, 0.0, 0.0, 0.0]} for i in range(len(inputs))]
        return httpx.Response(200, json={"data": rows})


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Server:
    fake = _Server(tmp_path / "up")
    monkeypatch.setattr(
        "repowise.core.providers.embedding.llamacpp.httpx.AsyncClient",
        lambda timeout: _REAL_CLIENT(timeout=timeout, transport=httpx.MockTransport(fake)),
    )
    monkeypatch.setattr(llamacpp, "_RETRY_BACKOFF_S", 0.0)
    monkeypatch.setattr(outage, "_last_verdict", None)
    monkeypatch.delenv(ENSURE_COMMAND_ENV, raising=False)
    return fake


def _ensure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exit_code: int,
    reply: dict,
    starts_server: bool = False,
) -> Path:
    """Install a stand-in ensure command; returns the file that counts its runs."""
    runs = tmp_path / "ensure-runs"
    script = tmp_path / "ensure.py"
    script.write_text(
        "import pathlib, sys\n"
        f"runs = pathlib.Path({str(runs)!r})\n"
        "runs.write_text((runs.read_text() if runs.exists() else '') + 'x')\n"
        + (f"pathlib.Path({str(tmp_path / 'up')!r}).touch()\n" if starts_server else "")
        + f"print({json.dumps(reply)!r})\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(ENSURE_COMMAND_ENV, f"{sys.executable} {script} embedder ensure --json")
    return runs


def _embedder() -> LlamaCppEmbedder:
    return LlamaCppEmbedder(base_url="http://llama.test", dimensions=4)


async def test_a_503_while_the_model_loads_is_retried_without_recovery(
    server, tmp_path, monkeypatch
) -> None:
    server.up.touch()
    server.statuses = [503]
    runs = _ensure(tmp_path, monkeypatch, exit_code=0, reply={"state": "serving"})

    assert len(await _embedder().embed(["query"])) == 1
    assert not runs.exists()


async def test_a_down_server_is_brought_back_by_ensure_and_the_embed_retried(
    server, tmp_path, monkeypatch
) -> None:
    runs = _ensure(
        tmp_path,
        monkeypatch,
        exit_code=0,
        reply={"state": "serving", "recovered": True, "detail": "restarted", "seconds": 3},
        starts_server=True,
    )

    with keyword_only_notices() as notices:
        assert len(await _embedder().embed(["query"])) == 1
    assert runs.read_text() == "x"
    assert notices == []


async def test_gaming_mode_answers_keyword_only_and_names_the_way_back(
    server, tmp_path, monkeypatch
) -> None:
    # The reply `solemd embedder ensure --json` gives while a gaming session holds the GPU.
    _ensure(
        tmp_path,
        monkeypatch,
        exit_code=3,
        reply={
            "state": "off_by_design",
            "recovered": False,
            "detail": "Gaming mode (full, active) stopped llama-server.service.",
            "remedy": "`solemd gaming resume` restores it.",
            "gaming": {"mode": "full", "phase": "active"},
            "seconds": 0.01,
        },
    )

    with keyword_only_notices() as notices, pytest.raises(EmbedderUnavailableError) as raised:
        await _embedder().embed(["query"])
    assert raised.value.verdict is not None and raised.value.verdict.state == "off_by_design"
    assert notices == [raised.value.warning]
    assert notices[0] == (
        "Keyword-only results: semantic search did not run because the embedding server is "
        "stopped for gaming mode. `solemd gaming resume` restores it. A miss here is not "
        "evidence of absence."
    )


async def test_a_fault_ensure_cannot_repair_carries_its_detail_and_remedy(
    server, tmp_path, monkeypatch
) -> None:
    _ensure(
        tmp_path,
        monkeypatch,
        exit_code=1,
        reply={
            "state": "fault",
            "recovered": False,
            "detail": "llama-server.service was failed and did not come back after "
            "`systemctl --user start`: model file missing.",
            "remedy": "Check `systemctl --user status llama-server`; the /infra skill owns it.",
            "gaming": None,
        },
    )

    with pytest.raises(EmbedderUnavailableError) as raised:
        await _embedder().embed(["query"])
    assert "could not be restarted" in raised.value.warning
    assert "model file missing" in raised.value.warning
    assert "systemctl --user status llama-server" in raised.value.warning


async def test_without_an_ensure_command_the_failure_is_still_reported(server) -> None:
    with keyword_only_notices() as notices, pytest.raises(EmbedderUnavailableError) as raised:
        await _embedder().embed(["query"])
    assert raised.value.verdict is None
    assert "ConnectError" in notices[0]


async def test_a_burst_of_failures_asks_for_recovery_once(server, tmp_path, monkeypatch) -> None:
    runs = _ensure(
        tmp_path, monkeypatch, exit_code=3, reply={"state": "off_by_design", "gaming": True}
    )
    for _ in range(3):
        with pytest.raises(EmbedderUnavailableError):
            await _embedder().embed(["query"])
    assert runs.read_text() == "x"


async def test_an_ensure_command_that_says_nothing_usable_is_reported_as_such(
    server, tmp_path, monkeypatch
) -> None:
    script = tmp_path / "broken.py"
    script.write_text(
        "import sys\nprint('usage: solemd embedder ...', file=sys.stderr)\nsys.exit(2)\n"
    )
    monkeypatch.setenv(ENSURE_COMMAND_ENV, f"{sys.executable} {script}")

    with pytest.raises(EmbedderUnavailableError) as raised:
        await _embedder().embed(["query"])
    assert raised.value.verdict is not None and raised.value.verdict.state == "unknown"
    assert "usage: solemd embedder" in raised.value.warning


async def test_a_refused_request_is_not_an_outage(server, tmp_path, monkeypatch) -> None:
    server.up.touch()
    server.statuses = [400]
    runs = _ensure(tmp_path, monkeypatch, exit_code=0, reply={"state": "serving"})

    with keyword_only_notices() as notices, pytest.raises(httpx.HTTPStatusError):
        await _embedder().embed(["query"])
    assert notices == []
    assert not runs.exists()
