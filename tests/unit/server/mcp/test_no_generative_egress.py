"""Retrieval and indexing must be incapable of reaching an LLM.

Excluding the two generative tools from the MCP surface is only half the claim
a no-LLM deployment makes; the other half is that the tools which remain, and
the indexing that feeds them, never generate either. That is asserted here by
poisoning the provider layer rather than by reading the code: every alias of
``get_provider`` raises and records, so any path that tries to construct a
provider fails loudly and leaves a fingerprint even where the caller swallows
exceptions (``_resolve_provider_for_answer`` does exactly that).

The provider env is deliberately *populated* in these tests. A run that proves
nothing resolved while no API key was configured proves only that the key was
missing.
"""

from __future__ import annotations

import importlib
import ipaddress
import os
import socket
import subprocess
from pathlib import Path

import pytest

# Every module that re-exports the factory. ``_resolve_provider_for_answer``
# imports from the first, ``build_enrichment_provider`` and the CLI resolver
# from the third; all three bind the name at import time, so patching one alias
# would leave the others live.
_PROVIDER_FACTORY_MODULES = (
    "repowise.core.providers.llm.registry",
    "repowise.core.providers.llm",
    "repowise.core.providers",
)

_LOOPBACK_HOSTNAMES = frozenset({"localhost", "localhost.localdomain"})


@pytest.fixture
def provider_calls(monkeypatch) -> list[str]:
    """Poison every ``get_provider`` alias; return the list of attempted names.

    Raising is the loud half. Recording is the half that survives a caller
    that catches ``Exception`` and degrades quietly, which is precisely what a
    silent LLM call would look like from the outside.
    """
    calls: list[str] = []

    def _poisoned(name: str, *args, **kwargs):
        calls.append(name)
        raise AssertionError(f"generative provider invoked: {name!r}")

    for module_name in _PROVIDER_FACTORY_MODULES:
        monkeypatch.setattr(importlib.import_module(module_name), "get_provider", _poisoned)
    return calls


@pytest.fixture
def loopback_only(monkeypatch) -> list[str]:
    """Block outbound TCP to anything but loopback; return the hosts attempted.

    Loopback stays open because a local embedding server (Ollama, TEI) is a
    legitimate part of retrieval and is not generation. Non-IP sockets are left
    alone: AF_UNIX is how multiprocessing talks to its own workers.
    """
    attempted: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _check(sock: socket.socket, address) -> None:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return
        host = str(address[0] if isinstance(address, tuple) else address)
        attempted.append(host)
        try:
            allowed = ipaddress.ip_address(host.split("%")[0]).is_loopback
        except ValueError:
            allowed = host.lower() in _LOOPBACK_HOSTNAMES
        if not allowed:
            raise AssertionError(f"outbound network connection attempted to {host!r}")

    def _connect(sock, address, *args, **kwargs):
        _check(sock, address)
        return real_connect(sock, address, *args, **kwargs)

    def _connect_ex(sock, address, *args, **kwargs):
        _check(sock, address)
        return real_connect_ex(sock, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _connect_ex)
    return attempted


@pytest.fixture
def configured_provider_env(monkeypatch) -> None:
    """A provider that *would* resolve, so an attempt cannot fail for lack of one."""
    monkeypatch.setenv("REPOWISE_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-never-used")


# --- the doubles themselves have to work ----------------------------------


def test_poison_fires_on_the_answer_resolution_seam(provider_calls, configured_provider_env):
    """Positive control: the double is wired where get_answer actually reaches.

    ``_resolve_provider_for_answer`` catches everything and returns None, so
    without this control a test asserting "no provider was resolved" would pass
    just as happily against a typo in the patch target.
    """
    from repowise.server.mcp_server.tool_answer.synthesis import _resolve_provider_for_answer

    assert _resolve_provider_for_answer(None) is None
    assert provider_calls, "the poisoned factory was never reached"


def test_socket_guard_blocks_non_loopback(loopback_only):
    sock = socket.socket()
    try:
        with pytest.raises(AssertionError, match="outbound network connection"):
            sock.connect(("93.184.216.34", 80))
    finally:
        sock.close()


def test_socket_guard_allows_loopback(loopback_only):
    sock = socket.socket()
    sock.settimeout(0.2)
    try:
        sock.connect(("127.0.0.1", 1))
    except AssertionError:  # pragma: no cover - the failure this test exists for
        pytest.fail("the guard blocked a loopback connection")
    except OSError:
        pass  # refused/timed out: the syscall ran, which is all that is claimed
    finally:
        sock.close()


# --- search ---------------------------------------------------------------


@pytest.fixture
def ready_vector_store():
    """Signal vector-store readiness, which the real server does at startup.

    Without it every semantic search burns the full 30s readiness timeout on an
    event nobody will ever set.
    """
    import asyncio

    import repowise.server.mcp_server as mcp_mod

    event = asyncio.Event()
    event.set()
    mcp_mod._vector_store_ready = event
    yield
    mcp_mod._vector_store_ready = None


async def _seed_vectors() -> None:
    import repowise.server.mcp_server as mcp_mod

    await mcp_mod._vector_store.embed_and_upsert(
        "file_page:src/auth/service.py",
        "Auth Service — Main authentication service class",
        {"title": "Auth Service", "page_type": "file_page", "target_path": "src/auth/service.py"},
    )
    await mcp_mod._vector_store.embed_and_upsert(
        "file_page:src/db/models.py",
        "DB Models — SQLAlchemy ORM models",
        {"title": "DB Models", "page_type": "file_page", "target_path": "src/db/models.py"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "query"),
    [
        ("concept", "authentication service"),
        ("symbol", "AuthService"),
        ("path", "src/auth/service.py"),
        ("hybrid", "how does AuthService work"),
    ],
)
async def test_search_never_resolves_a_provider(
    setup_mcp,
    ready_vector_store,
    provider_calls,
    loopback_only,
    configured_provider_env,
    mode,
    query,
):
    from repowise.server.mcp_server import search_codebase

    await _seed_vectors()
    result = await search_codebase(query, mode=mode)

    assert result["results"], f"{mode} search returned nothing — the test proved nothing"
    assert provider_calls == []


@pytest.mark.asyncio
async def test_search_auto_routing_never_resolves_a_provider(
    setup_mcp, ready_vector_store, provider_calls, loopback_only, configured_provider_env
):
    """mode="auto" picks the route itself; none of its branches may generate."""
    from repowise.server.mcp_server import search_codebase

    await _seed_vectors()
    for query in ("AuthService", "src/auth/service.py", "authentication service"):
        assert (await search_codebase(query))["results"], query

    assert provider_calls == []


# --- indexing -------------------------------------------------------------


def _git_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "class AuthService:\n"
        '    """Authenticate users."""\n\n'
        "    def login(self, user: str) -> bool:\n"
        "        return bool(user)\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "models.py").write_text(
        "class User:\n    name: str\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# sample\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.dev", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


@pytest.mark.asyncio
async def test_index_only_pipeline_never_resolves_a_provider(
    tmp_path, provider_calls, configured_provider_env
):
    """A full index-only run: parse, graph, git, health, dead code, decisions.

    FAST mode is the index-only configuration — it forces ``generate_docs`` off
    whatever the caller passed, so this also covers an operator who left the
    flag on. No provider is injected, and none may be resolved from the env.
    """
    from repowise.core.pipeline.modes import OrchestratorMode
    from repowise.core.pipeline.orchestrator import run_pipeline

    result = await run_pipeline(
        _git_repo(tmp_path),
        generate_docs=True,  # FAST must override this
        llm_client=None,
        mode=OrchestratorMode.FAST,
    )

    assert result.generated_pages is None
    assert result.parsed_files, "nothing was indexed — the test proved nothing"
    assert provider_calls == []


@pytest.mark.asyncio
async def test_hard_policy_nulls_an_injected_pipeline_provider(tmp_path, monkeypatch):
    """The shared pipeline boundary must defeat callers that inject a provider."""
    from repowise.core.generative_policy import NO_GENERATIVE_ENV
    from repowise.core.pipeline.modes import OrchestratorMode
    from repowise.core.pipeline.orchestrator import run_pipeline

    calls: list[tuple[tuple, dict]] = []

    class PoisonProvider:
        async def generate(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("hard no-generative policy reached an injected provider")

    repo = _git_repo(tmp_path)
    env_dir = repo / ".repowise"
    env_dir.mkdir()
    (env_dir / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=1\n",
        encoding="utf-8",
    )
    # A falsy process value cannot mask the repository's hard policy.
    monkeypatch.setenv(NO_GENERATIVE_ENV, "0")
    result = await run_pipeline(
        repo,
        generate_docs=True,
        llm_client=PoisonProvider(),
        mode=OrchestratorMode.STANDARD,
    )

    assert result.parsed_files, "nothing was indexed — the test proved nothing"
    assert result.generated_pages is None
    assert calls == []


def test_no_prose_hard_policy_never_resolves_a_cli_provider(
    tmp_path,
    monkeypatch,
    provider_calls,
    configured_provider_env,
):
    """Configured local/remote providers cannot bypass the init policy."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import (
        NO_GENERATIVE_ENV,
        NO_GENERATIVE_INDEX_MARKER,
        NO_GENERATIVE_INIT_COMPLETE_MARKER,
    )

    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    result = CliRunner().invoke(
        cli,
        [
            "init",
            str(_git_repo(tmp_path)),
            "--no-prose",
            "--mode",
            "standard",
            "--no-editor-setup",
            "--no-onboarding",
            "--no-agents",
            "--no-claude-md",
            "--no-codex",
            "--no-workspace",
            "--max-file-pages",
            "0",
            "--progress",
            "json",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    nonempty_lines = [line.strip() for line in result.output.splitlines() if line.strip()]
    assert nonempty_lines[-2:] == [
        NO_GENERATIVE_INIT_COMPLETE_MARKER,
        NO_GENERATIVE_INDEX_MARKER,
    ]
    assert nonempty_lines.count(NO_GENERATIVE_INIT_COMPLETE_MARKER) == 1
    assert nonempty_lines.count(NO_GENERATIVE_INDEX_MARKER) == 1
    assert provider_calls == []


def test_hard_policy_rejects_prose_before_provider_resolution(
    tmp_path,
    monkeypatch,
    provider_calls,
    configured_provider_env,
):
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import NO_GENERATIVE_ENV

    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    result = CliRunner().invoke(
        cli,
        ["init", str(tmp_path), "--prose", "--yes"],
    )

    assert result.exit_code == 2
    assert f"{NO_GENERATIVE_ENV}=1 requires --no-prose" in result.output
    assert provider_calls == []


@pytest.mark.parametrize("truthy", ["1", "true", "TRUE", "yes", "on", " On "])
def test_repo_dotenv_hard_policy_rejects_prose_before_provider_resolution(
    tmp_path,
    monkeypatch,
    provider_calls,
    truthy,
):
    """Repo-local policy must be read after dotenv and before provider setup."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import NO_GENERATIVE_ENV

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in (NO_GENERATIVE_ENV, "REPOWISE_PROVIDER", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    os.environ[NO_GENERATIVE_ENV] = "0"
    repo = _git_repo(tmp_path)
    env_dir = repo / ".repowise"
    env_dir.mkdir()
    (env_dir / ".env").write_text(
        f"{NO_GENERATIVE_ENV}={truthy}\n"
        "REPOWISE_PROVIDER=anthropic\n"
        "ANTHROPIC_API_KEY=sk-ant-test-never-used\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["init", str(repo), "--prose", "--provider", "anthropic", "--yes", "--no-workspace"],
    )

    assert result.exit_code == 2
    assert f"{NO_GENERATIVE_ENV}=1 requires --no-prose" in result.output
    assert provider_calls == []


def test_repo_dotenv_hard_policy_rejects_seed_delegation_before_provider_resolution(
    tmp_path,
    monkeypatch,
    provider_calls,
):
    """An explicit seed cannot bypass the repo-local init policy."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import NO_GENERATIVE_ENV

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in (NO_GENERATIVE_ENV, "REPOWISE_PROVIDER", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    target = tmp_path / "target"
    seed = tmp_path / "seed"
    target.mkdir()
    seed.mkdir()
    _git_repo(target)
    _git_repo(seed)
    env_dir = target / ".repowise"
    env_dir.mkdir()
    (env_dir / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=1\n"
        "REPOWISE_PROVIDER=anthropic\n"
        "ANTHROPIC_API_KEY=sk-ant-test-never-used\n",
        encoding="utf-8",
    )
    seed_calls: list[dict] = []

    def _seed_should_not_run(**kwargs):
        seed_calls.append(kwargs)
        raise AssertionError("seed delegation crossed the hard no-generative boundary")

    monkeypatch.setattr(
        "repowise.cli.worktree.seed_index_from_base",
        _seed_should_not_run,
    )

    result = CliRunner().invoke(
        cli,
        [
            "init",
            str(target),
            "--seed-from",
            str(seed),
            "--prose",
            "--provider",
            "anthropic",
            "--yes",
            "--no-workspace",
        ],
    )

    assert result.exit_code == 2
    assert f"{NO_GENERATIVE_ENV}=1 requires --no-prose" in result.output
    assert seed_calls == []
    assert provider_calls == []


def test_successful_hard_policy_seed_delegation_emits_terminal_witnesses(
    tmp_path,
    monkeypatch,
    provider_calls,
):
    """A seeded init has the same auditable success boundary as a cold init."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import (
        NO_GENERATIVE_ENV,
        NO_GENERATIVE_INDEX_MARKER,
        NO_GENERATIVE_INIT_COMPLETE_MARKER,
    )

    target = tmp_path / "target"
    seed = tmp_path / "seed"
    target.mkdir()
    seed.mkdir()
    _git_repo(target)
    _git_repo(seed)
    seed_calls: list[dict] = []
    update_calls: list[dict] = []

    def _seed_succeeds(**kwargs):
        seed_calls.append(kwargs)
        return True

    def _update_succeeds(**kwargs):
        update_calls.append(kwargs)

    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    monkeypatch.setattr("repowise.cli.worktree.seed_index_from_base", _seed_succeeds)
    monkeypatch.setattr(
        "repowise.cli.commands.update_cmd.command.run_update",
        _update_succeeds,
    )

    result = CliRunner().invoke(
        cli,
        [
            "init",
            str(target),
            "--seed-from",
            str(seed),
            "--no-prose",
            "--yes",
            "--no-workspace",
        ],
    )

    assert result.exit_code == 0, result.output
    nonempty_lines = [line.strip() for line in result.output.splitlines() if line.strip()]
    assert nonempty_lines[-2:] == [
        NO_GENERATIVE_INIT_COMPLETE_MARKER,
        NO_GENERATIVE_INDEX_MARKER,
    ]
    assert nonempty_lines.count(NO_GENERATIVE_INIT_COMPLETE_MARKER) == 1
    assert nonempty_lines.count(NO_GENERATIVE_INDEX_MARKER) == 1
    assert len(seed_calls) == 1
    assert len(update_calls) == 1
    assert update_calls[0]["index_only"] is True
    assert provider_calls == []


@pytest.mark.parametrize(
    ("mode_flag", "expected_message"),
    [
        ("--docs", "requires --index-only or --no-docs"),
        ("--full", "forbids --full"),
    ],
)
def test_repo_dotenv_hard_policy_rejects_generative_update_before_provider_resolution(
    tmp_path,
    monkeypatch,
    provider_calls,
    mode_flag,
    expected_message,
):
    """Direct update honors the same repo-local hard policy as init."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import NO_GENERATIVE_ENV

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in (NO_GENERATIVE_ENV, "REPOWISE_PROVIDER", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    os.environ[NO_GENERATIVE_ENV] = "0"
    repo = _git_repo(tmp_path)
    env_dir = repo / ".repowise"
    env_dir.mkdir()
    (env_dir / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=true\n"
        "REPOWISE_PROVIDER=anthropic\n"
        "ANTHROPIC_API_KEY=sk-ant-test-never-used\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["update", str(repo), mode_flag, "--provider", "anthropic", "--no-workspace"],
    )

    assert result.exit_code == 2
    assert expected_message in result.output
    assert provider_calls == []


def test_workspace_repo_dotenv_hard_policy_rejects_provider_resolution(
    tmp_path,
    monkeypatch,
    provider_calls,
):
    """The workspace primary repo dotenv is the same hard-policy boundary."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import NO_GENERATIVE_ENV

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in (NO_GENERATIVE_ENV, "REPOWISE_PROVIDER", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    first = tmp_path / "a-primary"
    second = tmp_path / "b-secondary"
    first.mkdir()
    second.mkdir()
    _git_repo(first)
    _git_repo(second)
    root_env_dir = tmp_path / ".repowise"
    root_env_dir.mkdir()
    (root_env_dir / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=0\n",
        encoding="utf-8",
    )
    env_dir = first / ".repowise"
    env_dir.mkdir()
    (env_dir / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=true\n"
        "REPOWISE_PROVIDER=anthropic\n"
        "ANTHROPIC_API_KEY=sk-ant-test-never-used\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["init", str(tmp_path), "--prose", "--provider", "anthropic", "--yes"],
    )

    assert result.exit_code == 2
    assert f"{NO_GENERATIVE_ENV}=1 requires --no-prose" in result.output
    assert provider_calls == []


def test_workspace_root_policy_emits_terminal_witnesses(
    tmp_path,
    monkeypatch,
    provider_calls,
):
    """The outer aggregate policy survives the workspace-owner handoff."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import (
        NO_GENERATIVE_ENV,
        NO_GENERATIVE_INDEX_MARKER,
        NO_GENERATIVE_INIT_COMPLETE_MARKER,
    )

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in (NO_GENERATIVE_ENV, "REPOWISE_PROVIDER", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    os.environ[NO_GENERATIVE_ENV] = "0"

    first = tmp_path / "a-primary"
    second = tmp_path / "b-secondary"
    first.mkdir()
    second.mkdir()
    _git_repo(first)
    _git_repo(second)
    root_env = tmp_path / ".repowise"
    root_env.mkdir()
    (root_env / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=true\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "init",
            str(tmp_path),
            "--no-prose",
            "--mode",
            "standard",
            "--no-editor-setup",
            "--no-onboarding",
            "--no-agents",
            "--no-claude-md",
            "--no-codex",
            "--max-file-pages",
            "0",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    nonempty_lines = [line.strip() for line in result.output.splitlines() if line.strip()]
    assert nonempty_lines[-2:] == [
        NO_GENERATIVE_INIT_COMPLETE_MARKER,
        NO_GENERATIVE_INDEX_MARKER,
    ]
    assert nonempty_lines.count(NO_GENERATIVE_INIT_COMPLETE_MARKER) == 1
    assert nonempty_lines.count(NO_GENERATIVE_INDEX_MARKER) == 1
    assert provider_calls == []


@pytest.mark.parametrize("policy_owner", ["workspace", "repo"])
def test_workspace_update_hard_policy_rejects_docs_before_provider_resolution(
    tmp_path,
    monkeypatch,
    provider_calls,
    policy_owner,
):
    """Workspace update honors both root and selected-repo hard policy."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import NO_GENERATIVE_ENV
    from repowise.core.workspace.config import RepoEntry, WorkspaceConfig

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in (NO_GENERATIVE_ENV, "REPOWISE_PROVIDER", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    os.environ[NO_GENERATIVE_ENV] = "0"

    root = tmp_path / "workspace"
    repo = root / "repo"
    repo.mkdir(parents=True)
    _git_repo(repo)
    WorkspaceConfig(
        version=1,
        repos=[RepoEntry(alias="repo", path="repo", is_primary=True)],
    ).save(root)
    policy_path = root if policy_owner == "workspace" else repo
    env_dir = policy_path / ".repowise"
    env_dir.mkdir(exist_ok=True)
    (env_dir / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=true\n"
        "REPOWISE_PROVIDER=anthropic\n"
        "ANTHROPIC_API_KEY=sk-ant-test-never-used\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        [
            "update",
            str(root),
            "--workspace",
            "--docs",
            "--provider",
            "anthropic",
        ],
    )

    assert result.exit_code == 2
    assert f"{NO_GENERATIVE_ENV}=1 requires --index-only or --no-docs" in result.output
    assert provider_calls == []


def test_nested_repo_dotenv_hard_policy_outranks_container_falsy_value(
    tmp_path,
    monkeypatch,
    provider_calls,
):
    """A container dotenv cannot shadow its one discovered repository."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import NO_GENERATIVE_ENV

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in (NO_GENERATIVE_ENV, "REPOWISE_PROVIDER", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    repo = tmp_path / "nested"
    repo.mkdir()
    _git_repo(repo)
    root_env_dir = tmp_path / ".repowise"
    root_env_dir.mkdir()
    (root_env_dir / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=0\n",
        encoding="utf-8",
    )
    repo_env_dir = repo / ".repowise"
    repo_env_dir.mkdir()
    (repo_env_dir / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=true\n"
        "REPOWISE_PROVIDER=anthropic\n"
        "ANTHROPIC_API_KEY=sk-ant-test-never-used\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["init", str(tmp_path), "--prose", "--provider", "anthropic", "--yes"],
    )

    assert result.exit_code == 2
    assert f"{NO_GENERATIVE_ENV}=1 requires --no-prose" in result.output
    assert provider_calls == []


def test_flag_off_preserves_index_only_decision_provider_resolution(
    tmp_path,
    monkeypatch,
    provider_calls,
    configured_provider_env,
):
    """The guard is opt-in; upstream index-only decision behavior remains."""
    from click.testing import CliRunner

    from repowise.cli.main import cli
    from repowise.core.generative_policy import NO_GENERATIVE_ENV

    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)
    result = CliRunner().invoke(
        cli,
        [
            "init",
            str(_git_repo(tmp_path)),
            "--no-prose",
            "--mode",
            "standard",
            "--no-editor-setup",
            "--no-onboarding",
            "--no-agents",
            "--no-claude-md",
            "--no-codex",
            "--no-workspace",
            "--max-file-pages",
            "0",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert provider_calls, "flag-off control never attempted provider resolution"


# --- the exclusion list has to stay complete -------------------------------


_MCP_SERVER_SRC = (
    Path(__file__).resolve().parents[4] / "packages/server/src/repowise/server/mcp_server"
)

# Reaching any of these is reaching the LLM: the factory, the interface, and the
# two named resolvers built on them.
_PROVIDER_MARKERS = (
    "get_provider",
    "BaseProvider",
    "build_enrichment_provider",
    "_resolve_provider_for_answer",
)


def test_only_the_known_generative_tools_touch_the_provider_layer():
    """A third tool learning to generate must break this, not widen the hole.

    ``GENERATIVE_TOOL_NAMES`` is a hand-written list, and a hand-written list of
    which code paths reach a model is worth exactly as much as its last review.
    This is that review, run every time.
    """
    reaching = {
        py.relative_to(_MCP_SERVER_SRC).as_posix()
        for py in _MCP_SERVER_SRC.rglob("*.py")
        if any(m in py.read_text(encoding="utf-8", errors="ignore") for m in _PROVIDER_MARKERS)
    }
    assert reaching == {
        # get_answer
        "tool_answer/answer.py",
        "tool_answer/synthesis.py",
        # generate_refactoring_code
        "tool_refactoring.py",
    }, reaching
