"""REPOWISE_TOOLS_NO_GENERATIVE removes the LLM-calling tools from the surface.

The claim under test is stronger than "they are off by default": with the flag
set, ``tools/list`` must not name them under *any* selection an operator can
write, and the flag being unset must leave the upstream surface byte-identical.
Both halves are asserted against the real registration path — the singleton
FastMCP server, the real registry apply, the real selection layer — because the
thing being proven is what a client sees, not what a helper returns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from repowise.core.registry import mcp_tool_registry
from repowise.server.mcp_server import _tool_selection, ensure_full_surface
from repowise.server.mcp_server._tool_selection import (
    GENERATIVE_TOOL_NAMES,
    NO_GENERATIVE_ENV,
    apply_tool_selection,
    describe_tool_surface,
    no_generative_tools_enabled,
    resolve_enabled_tools,
    set_tool_override,
    snapshot_full_surface,
)


@pytest.fixture
def rebind():
    """Re-run the bind-time step of ``ensure_full_surface`` on demand.

    ``snapshot_full_surface`` is where the exclusion bites, and it runs once per
    process — long before a test can set the env var. The returned callable
    replays it against a pristine copy of the full tool set, which is exactly
    what a server booting with the flag already set would do. Both globals are
    restored afterwards so the rest of the suite still sees every tool.
    """
    mcp = ensure_full_surface()
    saved = _tool_selection._full_surface
    saved_selected = _tool_selection._selected_surface
    full = dict(saved) if saved is not None else dict(mcp._tool_manager._tools)

    def _rebind():
        _tool_selection._full_surface = None
        mcp._tool_manager._tools = dict(full)
        snapshot_full_surface(mcp)
        return mcp

    yield _rebind

    _tool_selection._full_surface = saved
    _tool_selection._selected_surface = saved_selected
    mcp._tool_manager._tools = dict(full)


@pytest.fixture
def repo(tmp_path):
    """A non-workspace repo path with no ``mcp.tools`` block of its own."""
    (tmp_path / ".repowise").mkdir()
    return str(tmp_path)


async def _served(mcp) -> set[str]:
    return {t.name for t in await mcp.list_tools()}


# --- flag parsing ----------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_flag_on_for_truthy_values(monkeypatch, value):
    monkeypatch.setenv(NO_GENERATIVE_ENV, value)
    assert no_generative_tools_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_flag_off_otherwise(monkeypatch, value):
    monkeypatch.setenv(NO_GENERATIVE_ENV, value)
    assert no_generative_tools_enabled() is False


def test_flag_absent_is_off(monkeypatch):
    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)
    assert no_generative_tools_enabled() is False


def test_excluded_names_are_real_tools():
    """A rename upstream must break this test, not silently void the exclusion."""
    ensure_full_surface()
    registered = {e.name for e in mcp_tool_registry.entries()}
    assert registered >= GENERATIVE_TOOL_NAMES


# --- flag off: upstream behaviour is untouched -----------------------------


@pytest.mark.asyncio
async def test_flag_off_serves_get_answer_by_default(monkeypatch, rebind, repo):
    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)
    mcp = rebind()

    apply_tool_selection(mcp, repo_path=repo, override=None)
    served = await _served(mcp)
    assert "get_answer" in served
    # Still opt-in, exactly as upstream: default surface, so not present.
    assert "generate_refactoring_code" not in served


@pytest.mark.asyncio
async def test_flag_off_serves_refactoring_when_opted_in(monkeypatch, rebind, repo):
    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)
    mcp = rebind()

    apply_tool_selection(mcp, repo_path=repo, override="all")
    assert await _served(mcp) >= GENERATIVE_TOOL_NAMES

    apply_tool_selection(mcp, repo_path=repo, override="+generate_refactoring_code")
    assert "generate_refactoring_code" in await _served(mcp)


# --- flag on: absent from the served surface under every selection ---------


@pytest.mark.asyncio
async def test_bind_time_removal_needs_no_selection(monkeypatch, rebind):
    """A consumer that reads ``mcp_server.mcp`` directly never runs selection."""
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    mcp = rebind()

    assert not (GENERATIVE_TOOL_NAMES & await _served(mcp))
    assert not (GENERATIVE_TOOL_NAMES & set(_tool_selection._full_surface))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        None,
        "all",
        "lean",
        "+get_answer",
        "+get_answer,+generate_refactoring_code",
        "get_answer,search_codebase",
        ["get_answer", "search_codebase"],
    ],
    ids=["default", "all", "lean", "delta", "delta-both", "allowlist", "allowlist-list"],
)
async def test_flag_on_defeats_every_override(monkeypatch, rebind, repo, override):
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    mcp = rebind()

    enabled = apply_tool_selection(mcp, repo_path=repo, override=override)
    assert not (GENERATIVE_TOOL_NAMES & enabled)
    assert not (GENERATIVE_TOOL_NAMES & await _served(mcp))
    # The rest of the surface is untouched — this is a two-tool exclusion, not
    # a lockdown mode.
    assert "search_codebase" in await _served(mcp)


@pytest.mark.asyncio
async def test_served_tool_guidance_never_requires_the_suppressed_answer_tool(
    monkeypatch, rebind, repo
):
    """The descriptions a client receives are part of the deployed contract."""
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    mcp = rebind()
    apply_tool_selection(mcp, repo_path=repo, override="all")

    listed = {tool.name: tool.description or "" for tool in await mcp.list_tools()}
    assert "get_answer" not in listed
    for tool in ("search_codebase", "get_symbol"):
        assert "get_answer" not in listed[tool]


@pytest.mark.asyncio
async def test_repo_policy_outranks_falsy_process_value(monkeypatch, rebind, repo):
    """A prior falsy dotenv merge cannot mask the repo's hard policy."""
    monkeypatch.setenv(NO_GENERATIVE_ENV, "0")
    (Path(repo) / ".repowise" / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=true\n",
        encoding="utf-8",
    )
    mcp = rebind()

    enabled = apply_tool_selection(mcp, repo_path=repo, override="all")
    assert not (GENERATIVE_TOOL_NAMES & enabled)
    assert not (GENERATIVE_TOOL_NAMES & await _served(mcp))
    listed = {row["name"] for row in describe_tool_surface(repo)["tools"]}
    assert not (GENERATIVE_TOOL_NAMES & listed)


@pytest.mark.asyncio
async def test_workspace_member_policy_outranks_falsy_root(monkeypatch, rebind, tmp_path):
    """A workspace tool surface is bounded by every repo it can target."""
    from repowise.core.workspace.config import RepoEntry, WorkspaceConfig

    root = tmp_path / "workspace"
    member = root / "member"
    (root / ".repowise").mkdir(parents=True)
    (member / ".repowise").mkdir(parents=True)
    WorkspaceConfig(
        version=1,
        repos=[RepoEntry(alias="member", path="member", is_primary=True)],
    ).save(root)
    (root / ".repowise" / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=0\n",
        encoding="utf-8",
    )
    (member / ".repowise" / ".env").write_text(
        f"{NO_GENERATIVE_ENV}=true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(NO_GENERATIVE_ENV, "0")
    mcp = rebind()

    enabled = apply_tool_selection(mcp, repo_path=str(root), override="all")
    assert not (GENERATIVE_TOOL_NAMES & enabled)
    assert not (GENERATIVE_TOOL_NAMES & await _served(mcp))
    listed = {row["name"] for row in describe_tool_surface(str(root))["tools"]}
    assert not (GENERATIVE_TOOL_NAMES & listed)


@pytest.mark.asyncio
async def test_unreadable_workspace_policy_scope_fails_closed(monkeypatch, rebind, tmp_path):
    """A malformed workspace cannot silently omit member hard policies."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".repowise-workspace.yaml").write_text("repos: [\n", encoding="utf-8")
    monkeypatch.setenv(NO_GENERATIVE_ENV, "0")
    mcp = rebind()

    enabled = apply_tool_selection(mcp, repo_path=str(root), override="all")
    assert not (GENERATIVE_TOOL_NAMES & enabled)
    assert not (GENERATIVE_TOOL_NAMES & await _served(mcp))


@pytest.mark.asyncio
async def test_repeated_selection_cannot_restore(monkeypatch, rebind, repo):
    """Selection is idempotent and rebuilds from the snapshot; so must this be."""
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    mcp = rebind()

    apply_tool_selection(mcp, repo_path=repo, override="lean")
    apply_tool_selection(mcp, repo_path=repo, override="all")
    apply_tool_selection(mcp, repo_path=repo, override="+get_answer")
    assert not (GENERATIVE_TOOL_NAMES & await _served(mcp))


def test_config_attempt_is_logged_once(monkeypatch, caplog):
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    ensure_full_surface()
    entries = mcp_tool_registry.entries()

    with caplog.at_level(logging.WARNING, logger="repowise.mcp"):
        resolve_enabled_tools(entries, is_workspace=False, override="+get_answer")

    suppressed = [r for r in caplog.records if "get_answer" in r.getMessage()]
    assert len(suppressed) == 1
    assert NO_GENERATIVE_ENV in suppressed[0].getMessage()


def test_default_surface_does_not_warn(monkeypatch, caplog):
    """Only a selection that *asked* for the tool is a conflict worth logging."""
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    ensure_full_surface()
    entries = mcp_tool_registry.entries()

    with caplog.at_level(logging.WARNING, logger="repowise.mcp"):
        enabled = resolve_enabled_tools(entries, is_workspace=False, override=None)

    assert "get_answer" not in enabled
    assert not [r for r in caplog.records if "get_answer" in r.getMessage()]


def test_settings_ui_omits_excluded_tools(monkeypatch, repo):
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")
    listed = {t["name"] for t in describe_tool_surface(repo)["tools"]}
    assert not (GENERATIVE_TOOL_NAMES & listed)
    assert "search_codebase" in listed


def test_settings_ui_lists_them_when_flag_off(monkeypatch, repo):
    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)
    listed = {t["name"] for t in describe_tool_surface(repo)["tools"]}
    assert listed >= GENERATIVE_TOOL_NAMES


def test_config_request_is_reported_as_suppressed_not_unknown(monkeypatch, caplog, repo):
    """The row is hidden, but the tool is still a tool — say why it went away."""
    set_tool_override(repo, "+get_answer")
    monkeypatch.setenv(NO_GENERATIVE_ENV, "1")

    with caplog.at_level(logging.WARNING, logger="repowise.mcp"):
        listed = {t["name"] for t in describe_tool_surface(repo)["tools"]}

    assert "get_answer" not in listed
    messages = [r.getMessage() for r in caplog.records if "get_answer" in r.getMessage()]
    assert messages, "a config naming a suppressed tool should say so"
    assert all("unknown" not in m.lower() for m in messages), messages
