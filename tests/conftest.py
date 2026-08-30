"""Root test configuration — fixtures available to all test modules."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Drop the deployment no-generative policy before anything is collected.

    This is the same reset as :func:`_no_ambient_generative_policy`, one phase
    earlier, and both are needed. A fixture cannot cover this case: test
    modules do work at *import* time, and collection runs before any fixture
    exists. ``tests/unit/cli/test_agent_matrix.py`` binds ``COUNTS =
    GEN.tool_counts()`` at module scope, and ``tool_counts`` calls
    ``ensure_full_surface`` — so with ``REPOWISE_TOOLS_NO_GENERATIVE`` inherited
    from the environment, the generative tools are stripped from the
    process-wide FastMCP singleton and from ``_tool_selection._full_surface``
    while pytest is still collecting.

    That snapshot is taken once and kept ("first non-empty snapshot wins"), and
    every later ``apply_tool_selection`` rebuilds the advertised surface from
    it, so a single import decides what the server can serve for the rest of
    the session. ``pytest_configure`` runs before the first test module is
    imported, which is the last moment where clearing the variable still
    reaches that snapshot.
    """
    from repowise.core.generative_policy import NO_GENERATIVE_ENV

    os.environ.pop(NO_GENERATIVE_ENV, None)


@pytest.fixture(scope="session", autouse=True)
def _no_telemetry_network(tmp_path_factory: pytest.TempPathFactory):
    """Guarantee no test emits telemetry over the network or into real state.

    The MCP instrument seam emits an ``mcp_tool_call`` event via the core
    emitter's ``_post``; a test that drives the real wrapper with consent
    enabled would otherwise POST to the production ingest endpoint. Patch that
    sink to a no-op. Tests that assert emit behaviour re-patch it at function
    scope and still never touch the network.

    The CLI's ``command_run`` path is not patched here — its tests patch
    ``PlatformClient.post`` at the class level, so patching the
    ``default_client`` instance would shadow those patches. Its delivery is
    already off under pytest (``emitter._under_test``); what this fixture adds
    is redirecting the event spool, so a test that records an event cannot
    leave it queued in the real ``~/.repowise`` for a later real invocation to
    deliver.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    try:
        from repowise.core.platform import telemetry as _core_telemetry

        mp.setattr(_core_telemetry, "_post", lambda envelope: None, raising=False)
    except Exception:
        pass
    try:
        from repowise.cli.platform.telemetry import spool as _cli_spool

        spool_path = tmp_path_factory.mktemp("telemetry") / _cli_spool.SPOOL_FILENAME
        mp.setattr(_cli_spool, "_path", lambda: spool_path)
    except Exception:
        pass
    yield
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def _no_real_editor_setup():
    """Guarantee no test repoints the developer's real global editor config.

    ``repowise init`` (and ``doctor``'s self-heal path) defaults to
    ``--editor-setup`` on, which rewrites the *global*, machine-wide MCP
    server registration (e.g. ``~/.claude/settings.json``) to point at
    whatever repo path was just indexed. A test that drives the real CLI
    against a ``tmp_path`` fixture repo — e.g.
    ``tests/integration/test_cli.py``'s lock/watcher tests, which call
    ``init`` through ``CliRunner`` in-process rather than mocking it out —
    was doing exactly that: it repointed the developer's actual Claude Code
    MCP entry at a pytest temp directory that gets wiped on reboot, breaking
    every other project's ``repowise`` MCP tools until the *next*
    ``doctor``/``update`` run happened to self-heal it back (and even then,
    an already-running Claude Code session keeps using the stale spawn
    command it cached at connect time, since it has no reason to re-read
    the file mid-session).

    ``REPOWISE_SKIP_EDITOR_SETUP`` is the exact env var editor_setup.py and
    doctor's self-heal migrations already gate on — set once, for the whole
    session, so no test (present or future) can hit this by omission the
    way the lock/watcher tests did.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    mp.setenv("REPOWISE_SKIP_EDITOR_SETUP", "1")
    yield
    mp.undo()


@pytest.fixture(scope="session", autouse=True)
def _no_real_home_config_writes(tmp_path_factory: pytest.TempPathFactory):
    """Guarantee no test writes the developer's real ``~/.codex`` or ``~/.claude``.

    ``_no_real_editor_setup`` above covers ``init``/``doctor`` through the
    ``REPOWISE_SKIP_EDITOR_SETUP`` gate, but the explicit installers do not read
    that gate — ``repowise hook rewrite install`` is a user asking for the hook,
    so it installs. Driven through ``CliRunner`` on a machine that has Codex
    (``~/.codex`` exists, ``codex --version`` passes the version gate),
    ``tests/unit/distill/test_allow_rule.py`` was writing the distill rewrite
    hook into the developer's real ``~/.codex/hooks.json`` on every suite run
    (2026-08-29): that file's ``settings_path`` fixture isolates only the Claude
    half. Codex then hash-checks the file and asks the user to re-trust a hook
    they had removed on purpose.

    The three user-level writer paths are wrapped, not replaced: each still
    resolves through ``Path.home()``, so a test that moves ``HOME`` (the
    ``agents``/``doctor`` tests do) keeps seeing its own temp home, and a test
    that patches the same name at function scope still wins and restores to
    this wrapper. Only when ``Path.home()`` is still the *real* home — i.e. the
    test isolated nothing — does the path land in a session temp dir instead.
    ``HOME`` itself is left alone: git identity, ``uv``, and the telemetry spool
    guard above all resolve through it.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    real_home = Path.home()
    guard_home = tmp_path_factory.mktemp("home")

    def _guarded(*parts: str):
        def resolve() -> Path:
            home = Path.home()
            if home == real_home:
                home = guard_home
            return home.joinpath(*parts)

        return resolve

    try:
        from repowise.cli.editor_integrations import codex_config

        mp.setattr(codex_config, "_codex_hooks_path", _guarded(".codex", "hooks.json"))
    except Exception:
        pass
    try:
        from repowise.cli.editor_integrations import claude_config

        mp.setattr(
            claude_config, "_claude_code_settings_path", _guarded(".claude", "settings.json")
        )
    except Exception:
        pass
    try:
        from repowise.cli.agent_targets.targets import codex as codex_target

        mp.setattr(codex_target, "user_prompts_dir", _guarded(".codex", "prompts"))
    except Exception:
        pass
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _no_ambient_generative_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with the hard no-generative policy *off*.

    ``REPOWISE_TOOLS_NO_GENERATIVE`` is a deployment switch, not a test knob:
    ``generative_calls_disabled`` reads it straight off the process
    environment, and a truthy value is fail-closed everywhere it is consulted.
    The CLI refuses any model-written ``init``/``update`` outright, the server
    refuses chat, provider validation and generative jobs, and
    ``snapshot_full_surface`` deletes the generative tools from the singleton
    FastMCP server the first time a test builds the surface.

    No test sets it — a *deployment* does, and hands it to pytest through the
    inherited environment (SoleMD pins it in the RepoWise service unit and
    carries the same pin in its test recipe). Every docs-generating test then
    fails while the ``--index-only`` test beside it passes, and the same run
    from a plain shell is green. That reads as flakiness and has been
    misfiled as test-ordering pollution; it is neither, because it does not
    depend on order at all.

    Clearing it per test makes the policy something a test opts *into*, which
    is what the tests that exercise it already do (``monkeypatch.setenv``);
    the defensive ``monkeypatch.delenv`` several of them carry becomes
    redundant rather than wrong. Being function-scoped is the point: a test
    that writes the variable through raw ``os.environ`` rather than
    ``monkeypatch`` cannot leak it into the next one.

    :func:`pytest_configure` above clears the same variable before collection,
    which is what protects the import-time readers this fixture is too late
    for. Keep both.

    Repo- and workspace-local ``.repowise/.env`` policy is deliberately left
    alone: those files are written by the tests that assert on them.
    """
    from repowise.core.generative_policy import NO_GENERATIVE_ENV

    monkeypatch.delenv(NO_GENERATIVE_ENV, raising=False)


@pytest.fixture(autouse=True)
def _isolate_db_url_env():
    """Stop one test's database URL from redirecting every later test.

    ``resolve_db_url`` checks ``REPOWISE_DB_URL`` / ``REPOWISE_DATABASE_URL``
    *before* the explicit ``repo_path`` it is handed, so whatever holds those
    variables decides where every index write in the process lands.

    ``serve_cmd`` assigns ``REPOWISE_DB_URL`` directly on ``os.environ`` when it
    finds a ``.repowise/`` beside the cwd, which is right for a real ``serve``
    process and wrong inside pytest: a test that drives the real ``serve``
    command in-process leaves the *developer's own* database pinned for the
    rest of the session, and every later test that indexes a ``tmp_path``
    fixture repo writes its rows there instead of into its own store. That is
    how a suite run leaves hundreds of ``repo`` rows, pointing at temp
    directories, in the checkout you are working in.

    Snapshot and restore rather than pin a value: a test that sets its own URL
    must keep working, and one that sets none must still fall through to
    ``<repo>/.repowise/wiki.db``. Only the leak across the test boundary is
    closed.
    """
    saved = {k: os.environ.get(k) for k in ("REPOWISE_DB_URL", "REPOWISE_DATABASE_URL")}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _isolate_structlog_config():
    """Restore structlog's global configuration after every test.

    ``configure_cli_logging`` / ``silence_logs_for_machine_output`` install a
    filtering bound logger at ERROR process-wide and never undo it, so the
    first test to exercise a CLI command silences every ``info`` and
    ``warning`` for the rest of the session.

    That is invisible until a test asserts on a log record.
    ``structlog.testing.capture_logs`` swaps the *processor chain*, not the
    *wrapper class*, so a filtering logger drops the event before ``LogCapture``
    ever runs and the test reads an empty list. Tests collected after
    ``tests/unit/cli`` therefore pass alone and fail in a full run.

    Snapshot and restore rather than reset to defaults: a test that configures
    structlog on purpose keeps working, and the next test still starts clean.
    """
    import structlog

    saved = structlog.get_config()
    try:
        yield
    finally:
        structlog.configure(**saved)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_repo_path(repo_root: Path) -> Path:
    """Path to the multi-language sample repository used in integration tests."""
    path = repo_root / "tests" / "fixtures" / "sample_repo"
    assert path.exists(), (
        f"Sample repo not found at {path}. Run 'make install' to ensure test fixtures are in place."
    )
    return path


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    """Path to the tests/fixtures/ directory."""
    return repo_root / "tests" / "fixtures"
