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


def test_dismiss_and_deprecate_are_visibly_disabled(journal_cli_repo: Path) -> None:
    dismissed = CliRunner().invoke(
        cli,
        ["decision", "dismiss", "dec-00000001", str(journal_cli_repo)],
    )
    deprecated = CliRunner().invoke(
        cli,
        ["decision", "deprecate", "dec-00000001", str(journal_cli_repo)],
    )

    assert dismissed.exit_code != 0
    assert "disabled in decision journal mode" in dismissed.output
    assert deprecated.exit_code != 0
    assert "disabled in decision journal mode" in deprecated.output
    assert not (journal_cli_repo / ".codeatlas" / "decisions.jsonl").exists()


def test_a_flag_driven_add_lands_a_proposal_rather_than_being_refused(
    journal_cli_repo: Path,
) -> None:
    """The B2 ruling, which the CLI was the last surface to disagree with.

    A machine may propose into the journal; only ``confirm`` promotes, and the
    git diff is where a human reviews it. Refusing here left an agent with no
    way to record anything at all, while the MCP tool wrote proposals to the
    same file — so the rule that was supposed to gate review instead just
    picked which caller got to write.
    """
    added = CliRunner().invoke(
        cli,
        [
            "decision",
            "add",
            "--title",
            "Cap embedder inputs in the provider",
            "--decision",
            "Truncate each text before the request",
            "--rationale",
            "OpenAI rejects the whole batch, it does not truncate",
            "--affects",
            "src/service.py",
            "--format",
            "json",
            str(journal_cli_repo),
        ],
    )

    assert added.exit_code == 0, added.output
    assert json.loads(added.output)["decision"]["status"] == "proposed"

    canonical = DecisionJournal(journal_cli_repo).list()
    assert len(canonical) == 1
    assert canonical[0].confirmed_at is None


def test_confirm_is_the_only_promotion(journal_cli_repo: Path) -> None:
    added = CliRunner().invoke(
        cli,
        [
            "decision",
            "add",
            "--title",
            "Token-aware smart cap",
            "--decision",
            "Count tokens when a tokenizer is cheap",
            "--rationale",
            "characters mismeasure dense text by 4x",
            "--affects",
            "src/service.py",
            "--format",
            "json",
            str(journal_cli_repo),
        ],
    )
    assert added.exit_code == 0, added.output
    decision_id = json.loads(added.output)["decision"]["id"]

    confirmed = CliRunner().invoke(cli, ["decision", "confirm", decision_id, str(journal_cli_repo)])
    assert confirmed.exit_code == 0, confirmed.output

    canonical = DecisionJournal(journal_cli_repo).list()
    assert canonical[0].confirmed_at is not None


def test_a_flag_driven_add_still_needs_a_why_and_an_anchor(journal_cli_repo: Path) -> None:
    """Proposing is allowed; proposing nothing checkable is not.

    A journal row with no rationale cannot be reviewed and one with no anchor
    cannot be scored for staleness or found from a file, so both requirements
    outlive the refusal that used to sit in front of them.
    """
    result = CliRunner().invoke(
        cli,
        [
            "decision",
            "add",
            "--title",
            "Unanchored proposal",
            "--decision",
            "Do not write this",
            str(journal_cli_repo),
        ],
    )

    assert result.exit_code != 0
    assert "requires a rationale" in result.output
    assert not (journal_cli_repo / ".codeatlas" / "decisions.jsonl").exists()


def test_the_interactive_path_still_records_confirmed(journal_cli_repo: Path) -> None:
    answers = (
        "Ratified at authoring\n"
        "\n"
        "Answer eight prompts and it is reviewed\n"
        "a person read this one\n"
        "\n"
        "\n"
        "src/service.py\n"
        "\n"
    )
    result = CliRunner().invoke(cli, ["decision", "add", str(journal_cli_repo)], input=answers)

    assert result.exit_code == 0, result.output
    assert DecisionJournal(journal_cli_repo).list()[0].confirmed_at is not None


# ---------------------------------------------------------------------------
# Workspace targeting (F48)
# ---------------------------------------------------------------------------
#
# A workspace holds one journal per repo, and until now the CLI could reach
# exactly one of them — the primary. Every other repo's decisions were
# unreachable from the command line whatever the caller typed, which is why
# ``CommandTarget.resolve_repo_alias`` existed with no caller anywhere.


@pytest.fixture
def journal_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    for alias in ("a", "b"):
        repo = root / alias
        (repo / "src").mkdir(parents=True)
        (repo / "src" / f"{alias}.py").write_text(f"VALUE = {alias!r}\n", encoding="utf-8")
    (root / ".repowise-workspace.yaml").write_text(
        "version: 1\n"
        "default_repo: a\n"
        "repos:\n"
        "- path: a\n"
        "  alias: a\n"
        "  is_primary: true\n"
        "- path: b\n"
        "  alias: b\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(DECISIONS_JOURNAL_ENV, ".codeatlas/decisions.jsonl")
    monkeypatch.chdir(root)
    return root


def _payload(result) -> dict:
    """The JSON document out of a stream that also carries the notice.

    ``--format json`` sends notices to stderr, and ``CliRunner`` mixes the two
    by default — the same reason ``test_doctor_json`` slices from the first
    brace.
    """
    return json.loads(result.output[result.output.index("{") :])


def _add_to(alias: str, extra: list[str] | None = None):
    return CliRunner().invoke(
        cli,
        [
            "decision",
            "add",
            "--repo",
            alias,
            "--title",
            f"Decision for {alias}",
            "--decision",
            f"Do the {alias} thing",
            "--rationale",
            "because the alias has to reach this repo's journal",
            "--affects",
            f"src/{alias}.py",
            "--format",
            "json",
            *(extra or []),
        ],
    )


def test_an_alias_writes_to_that_repos_journal_as_a_proposal(
    journal_workspace: Path,
) -> None:
    result = _add_to("b")

    assert result.exit_code == 0, result.output
    # The transparency notice names the repo the command actually ran on, not
    # just the workspace it found.
    assert "decision on b within ws" in result.output
    payload = _payload(result)
    assert payload["decision"]["status"] == "proposed"
    assert payload["repo"] == str(journal_workspace / "b")

    assert (journal_workspace / "b" / ".codeatlas" / "decisions.jsonl").exists()
    assert not (journal_workspace / "a" / ".codeatlas" / "decisions.jsonl").exists()
    written = DecisionJournal(journal_workspace / "b").list()
    assert [(d.title, d.confirmed_at) for d in written] == [("Decision for b", None)]


def test_confirm_promotes_in_the_repo_the_alias_names(journal_workspace: Path) -> None:
    added = _add_to("b")
    assert added.exit_code == 0, added.output
    decision_id = _payload(added)["decision"]["id"]

    confirmed = CliRunner().invoke(cli, ["decision", "confirm", decision_id, "--repo", "b"])

    assert confirmed.exit_code == 0, confirmed.output
    assert DecisionJournal(journal_workspace / "b").list()[0].confirmed_at is not None


def test_without_an_alias_the_primary_is_still_the_target(journal_workspace: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "decision",
            "add",
            "--title",
            "Primary by default",
            "--decision",
            "Keep targeting the primary when nothing is named",
            "--rationale",
            "the alias is opt-in, not a new requirement",
            "--affects",
            "src/a.py",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _payload(result)["repo"] == str(journal_workspace / "a")


def test_an_unknown_alias_names_the_ones_that_exist(journal_workspace: Path) -> None:
    result = _add_to("nope")

    assert result.exit_code != 0
    assert "Unknown repo 'nope'" in result.output
    assert "['a', 'b']" in result.output


def test_an_alias_and_a_path_together_are_an_error(journal_workspace: Path) -> None:
    """Two ways of naming one repo, so there is nothing sensible to combine."""
    result = _add_to("b", extra=[str(journal_workspace / "a")])

    assert result.exit_code != 0
    assert "not both" in result.output
    assert not (journal_workspace / "b" / ".codeatlas").exists()


def test_list_and_show_reach_the_same_repo_by_alias(journal_workspace: Path) -> None:
    added = _add_to("b")
    assert added.exit_code == 0, added.output
    decision_id = _payload(added)["decision"]["id"]

    listed = CliRunner().invoke(cli, ["decision", "list", "--repo", "b", "--format", "json"])
    assert listed.exit_code == 0, listed.output
    assert [d["id"] for d in _payload(listed)["decisions"]] == [decision_id]

    from_primary = CliRunner().invoke(cli, ["decision", "list", "--repo", "a", "--format", "json"])
    assert from_primary.exit_code == 0, from_primary.output
    assert _payload(from_primary)["decisions"] == []

    shown = CliRunner().invoke(
        cli, ["decision", "show", decision_id, "--repo", "b", "--format", "json"]
    )
    assert shown.exit_code == 0, shown.output
    assert _payload(shown)["decision"]["title"] == "Decision for b"
