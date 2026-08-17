"""The feature flag: what it accepts, and what importing the package costs."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from repowise.core.source_search import SOURCE_SEARCH_ENV, source_search_enabled


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "On", " true "])
def test_truthy_values_enable_it(monkeypatch, value):
    monkeypatch.setenv(SOURCE_SEARCH_ENV, value)
    assert source_search_enabled()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe", "2"])
def test_everything_else_leaves_it_off(monkeypatch, value):
    monkeypatch.setenv(SOURCE_SEARCH_ENV, value)
    assert not source_search_enabled()


def test_unset_is_off(monkeypatch):
    """Off by default: an unset variable must leave the product as it was."""
    monkeypatch.delenv(SOURCE_SEARCH_ENV, raising=False)
    assert not source_search_enabled()


def test_the_value_is_read_at_call_time(monkeypatch):
    """A test that sets the variable must be seen by an already-imported module."""
    monkeypatch.delenv(SOURCE_SEARCH_ENV, raising=False)
    assert not source_search_enabled()
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    assert source_search_enabled()


def test_asking_the_flag_imports_nothing_heavy():
    """Reading the flag must not drag LanceDB or SQLAlchemy onto the path.

    The package init is the one module a caller touches to find out whether
    this feature is on. If asking that question costs the imports the feature
    itself needs, "off" stops being free.
    """
    probe = (
        "import sys\n"
        "import repowise.core.source_search as s\n"
        "assert s.source_search_enabled() is False\n"
        "leaked = [m for m in ('lancedb', 'pyarrow', 'sqlalchemy', 'structlog')\n"
        "          if m in sys.modules]\n"
        "assert not leaked, leaked\n"
        "submodules = [m for m in sys.modules if m.startswith('repowise.core.source_search.')]\n"
        "assert not submodules, submodules\n"
    )
    env = {**os.environ, "PYTHONPATH": _pythonpath()}
    env.pop(SOURCE_SEARCH_ENV, None)
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def _pythonpath() -> str:
    """The core src root, so the probe subprocess imports this working tree."""
    import pathlib

    import repowise.core

    # .../packages/core/src/repowise/core/__init__.py -> .../packages/core/src
    return str(pathlib.Path(repowise.core.__file__).resolve().parents[2])
