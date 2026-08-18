"""CLI write-through and visible-disable behavior in decision journal mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from repowise.cli.main import cli
from repowise.core.analysis.decisions.journal import (
    DECISIONS_JOURNAL_ENV,
    DecisionJournal,
)


@pytest.fixture
def journal_cli_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".repowise").mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, ".codeatlas/decisions.jsonl")
    return repo


def test_interactive_add_writes_journal_then_list_reads_projection(
    journal_cli_repo: Path,
) -> None:
    answers = (
        "Use the journal\n"
        "\n"
        "Write canonical JSONL first\n"
        "It makes SQLite disposable\n"
        "\n"
        "\n"
        "src/service.py\n"
        "\n"
    )
    added = CliRunner().invoke(
        cli,
        ["decision", "add", str(journal_cli_repo)],
        input=answers,
    )
    assert added.exit_code == 0, added.output

    canonical = DecisionJournal(journal_cli_repo).list()
    assert len(canonical) == 1
    assert canonical[0].title == "Use the journal"
    assert canonical[0].anchors[0].file_sha is not None

    listed = CliRunner().invoke(
        cli,
        ["decision", "list", str(journal_cli_repo), "--format", "json"],
    )
    assert listed.exit_code == 0, listed.output
    rows = json.loads(listed.output)["decisions"]
    assert [(row["id"], row["source"]) for row in rows] == [(canonical[0].id, "journal")]


def test_proposal_dismiss_and_deprecate_are_visibly_disabled(
    journal_cli_repo: Path,
) -> None:
    proposed = CliRunner().invoke(
        cli,
        [
            "decision",
            "add",
            "--title",
            "Unreviewed proposal",
            "--decision",
            "Do not write this",
            str(journal_cli_repo),
        ],
    )
    dismissed = CliRunner().invoke(
        cli,
        ["decision", "dismiss", "dec-00000001", str(journal_cli_repo)],
    )
    deprecated = CliRunner().invoke(
        cli,
        ["decision", "deprecate", "dec-00000001", str(journal_cli_repo)],
    )

    assert proposed.exit_code != 0
    assert "disabled in curated journal mode" in proposed.output
    assert dismissed.exit_code != 0
    assert "disabled in decision journal mode" in dismissed.output
    assert deprecated.exit_code != 0
    assert "disabled in decision journal mode" in deprecated.output
    assert not (journal_cli_repo / ".codeatlas" / "decisions.jsonl").exists()
