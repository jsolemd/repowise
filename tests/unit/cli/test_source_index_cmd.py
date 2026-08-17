"""``repowise source-index``: the flag gate and the JSON it prints."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from repowise.cli.commands.source_index_cmd import source_index_command
from repowise.core.source_search import SOURCE_SEARCH_ENV


@pytest.fixture
def runner():
    """Streams separated: the JSON summary is stdout, the structlog trace is stderr.

    click < 8.2 mixes them unless asked not to; 8.2 removed the parameter and
    separates them by default. Mirrors ``test_update_progress_json``.
    """
    try:
        return CliRunner(mix_stderr=False)  # type: ignore[call-arg]
    except TypeError:
        return CliRunner()


def test_it_refuses_when_the_flag_is_unset(runner, monkeypatch, tmp_path):
    monkeypatch.delenv(SOURCE_SEARCH_ENV, raising=False)
    result = runner.invoke(source_index_command, ["--repo", str(tmp_path)])
    assert result.exit_code != 0
    assert SOURCE_SEARCH_ENV in result.stderr


@pytest.mark.parametrize("value", ["0", "false", "off", ""])
def test_it_refuses_on_a_falsy_flag(runner, monkeypatch, tmp_path, value):
    monkeypatch.setenv(SOURCE_SEARCH_ENV, value)
    result = runner.invoke(source_index_command, ["--repo", str(tmp_path)])
    assert result.exit_code != 0
    assert "disabled" in result.stderr


def test_refusing_writes_nothing_into_the_repo(runner, monkeypatch, tmp_path):
    """Off must be free: no .repowise directory, no database, no index."""
    monkeypatch.delenv(SOURCE_SEARCH_ENV, raising=False)
    runner.invoke(source_index_command, ["--repo", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_it_refuses_a_keyless_embedder(runner, monkeypatch, tmp_path):
    """An 8-wide keyless vector carries no signal; an index on it is a lie."""
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    monkeypatch.setenv("REPOWISE_EMBEDDER", "nonesuch")
    result = runner.invoke(source_index_command, ["--repo", str(tmp_path)])
    assert result.exit_code != 0
    assert "No real embedder" in result.stderr


def test_it_prints_a_json_summary(runner, monkeypatch, tmp_path):
    pytest.importorskip("lancedb")
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    repo = _seeded_repo(tmp_path)

    result = runner.invoke(source_index_command, ["--repo", str(repo), "--embedder", "mock"])
    assert result.exit_code == 0, result.stderr

    payload = json.loads(result.stdout)
    assert payload["chunks"]["symbol"] == 1
    assert payload["chunks"]["file_window"] == 1
    assert payload["chunks"]["total"] == 2
    assert payload["embedding"]["provider"] == "mock"
    assert payload["embedding"]["dims"] == 8
    assert payload["embedding"]["embedded"] == 2
    assert payload["paths"]["manifest"].endswith("source_index.json")
    assert set(payload["timings_seconds"]) == {"load_and_build", "embed", "write_fts", "total"}


def test_the_summary_is_the_only_thing_on_stdout(runner, monkeypatch, tmp_path):
    """A machine reading this command must not have to strip prose off it."""
    pytest.importorskip("lancedb")
    monkeypatch.setenv(SOURCE_SEARCH_ENV, "1")
    repo = _seeded_repo(tmp_path)
    result = runner.invoke(source_index_command, ["--repo", str(repo), "--embedder", "mock"])
    assert json.loads(result.stdout)  # parses whole, with nothing appended


def _seeded_repo(tmp_path):
    """A git repo with one indexed symbol and one windowed text file."""
    import asyncio
    import subprocess

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("def run():\n    return 1\n")
    (root / "compose.yaml").write_text("services:\n  db:\n    image: postgres\n")
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.com"],
        ["config", "user.name", "T"],
        ["add", "-A"],
        ["-c", "commit.gpgsign=false", "commit", "-qm", "seed"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    asyncio.run(_seed(root))
    return root


async def _seed(root):
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        init_db,
        resolve_db_url,
    )
    from repowise.core.persistence.models import Repository, WikiSymbol

    engine = create_engine(resolve_db_url(root))
    await init_db(engine)
    async with create_session_factory(engine)() as session:
        session.add(Repository(id="r1", name="repo", local_path=str(root), head_commit="abc"))
        session.add(
            WikiSymbol(
                repository_id="r1",
                file_path="src/app.py",
                symbol_id="src/app.py::run",
                name="run",
                qualified_name="app.run",
                kind="function",
                signature="def run()",
                start_line=1,
                end_line=2,
                language="python",
            )
        )
        await session.commit()
    await engine.dispose()
