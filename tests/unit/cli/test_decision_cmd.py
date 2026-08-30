"""CLI coverage for ``repowise decision`` subcommands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from repowise.cli.main import cli
from repowise.core.persistence.database import init_db
from repowise.core.persistence.models import DecisionRecord, Repository

_REPO_ID = "decision-cli-repo"


def _seed_wiki_db(repo_root: Path, decisions: list[dict]) -> None:
    """Create ``.repowise/wiki.db`` with a repository row and decision records."""

    async def _build() -> None:
        db_path = repo_root / ".repowise" / "wiki.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
        await init_db(engine)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            session.add(Repository(id=_REPO_ID, name=repo_root.name, local_path=str(repo_root)))
            for spec in decisions:
                session.add(
                    DecisionRecord(
                        id=spec["id"],
                        repository_id=_REPO_ID,
                        title=spec["title"],
                        decision=spec.get("decision", "use X"),
                        rationale=spec.get("rationale", "because Y"),
                        context=spec.get("context", "forced by Z"),
                        status=spec.get("status", "active"),
                        source=spec.get("source", "cli"),
                        confidence=spec.get("confidence", 0.9),
                        staleness_score=spec.get("staleness", 0.0),
                        evidence_file=spec["id"],
                    )
                )
            await session.commit()
        await engine.dispose()

    asyncio.run(_build())


def _payload(result) -> dict:
    """The JSON document out of a stream that also carries notices.

    ``--format json`` sends every human-facing aside to stderr and ``CliRunner``
    mixes the two, so a payload is read from the first brace on — the same
    slice ``test_doctor_json`` takes.
    """
    return json.loads(result.output[result.output.index("{") :])


@pytest.fixture
def indexed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".repowise").mkdir()
    return repo


def test_decision_help_lists_subcommands() -> None:
    result = CliRunner().invoke(cli, ["decision", "--help"])

    assert result.exit_code == 0, result.output
    for name in ("add", "list", "show", "confirm", "dismiss", "deprecate", "health"):
        assert name in result.output


def test_decision_add_records_without_prompting(indexed_repo: Path) -> None:
    """With --title and --decision, nothing is asked and the id comes back.

    ``add`` was eight blocking prompts and no flags, so with no stdin it died
    on the first one — the whole command was unreachable to anything scripted.
    """
    result = CliRunner().invoke(
        cli,
        [
            "decision", "add",
            "--title", "Escape LIKE patterns",
            "--decision", "Escape % and _ before interpolating",
            "--rationale", "an unescaped pattern scans the table",
            "--alternative", "match in Python",
            "--consequence", "one more helper on the query path",
            "--affects", "src/db/models.py",
            "--tag", "database",
            "--format", "json",
            str(indexed_repo),
        ],
        input="",
    )

    assert result.exit_code == 0, result.output
    payload = _payload(result)
    assert payload["decision"]["title"] == "Escape LIKE patterns"
    assert len(payload["decision"]["id"]) > 8, "the full id, not the table's prefix"

    shown = CliRunner().invoke(
        cli, ["decision", "show", payload["decision"]["id"], str(indexed_repo), "--format", "json"]
    )
    assert shown.exit_code == 0, shown.output
    record = _payload(shown)["decision"]
    assert record["rationale"] == "an unescaped pattern scans the table"
    assert record["alternatives"] == ["match in Python"]
    assert record["consequences"] == ["one more helper on the query path"]
    assert record["affected_files"] == ["src/db/models.py"]
    assert record["tags"] == ["database"]


def test_a_flag_driven_decision_lands_proposed(indexed_repo: Path) -> None:
    """A caller that inferred a decision has not reviewed it, and the store says so.

    The prompts still record ``active``: a person answering eight questions is
    the reviewed case. Both paths writing one status is what makes the
    difference invisible.
    """
    result = CliRunner().invoke(
        cli,
        [
            "decision", "add",
            "--title", "Prefer ruff check",
            "--decision", "Run ruff check, never ruff format",
            "--format", "json",
            str(indexed_repo),
        ],
        input="",
    )

    assert result.exit_code == 0, result.output
    assert _payload(result)["decision"]["status"] == "proposed"


def test_decision_add_prompts_record_active(indexed_repo: Path) -> None:
    """The interactive path is unchanged, including the status it writes."""
    answers = "Interactive title\ncontext\nthe decision\nwhy\n\n\n\n\n"
    result = CliRunner().invoke(
        cli, ["decision", "add", str(indexed_repo)], input=answers
    )

    assert result.exit_code == 0, result.output
    listed = CliRunner().invoke(
        cli, ["decision", "list", str(indexed_repo), "--format", "json"]
    )
    records = _payload(listed)["decisions"]
    assert [d["status"] for d in records if d["title"] == "Interactive title"] == ["active"]


def test_half_a_command_line_is_an_error_not_a_prompt(indexed_repo: Path) -> None:
    """--title alone must not fall through to the prompts.

    A caller with no stdin would hang there, or abort on a message naming
    neither flag. The exit code carries it too: printing the reason and
    returning 0 is what a script reads as success.
    """
    result = CliRunner().invoke(
        cli,
        ["decision", "add", "--title", "Half a decision", str(indexed_repo), "--format", "json"],
        input="",
    )

    assert result.exit_code == 1, result.output
    assert _payload(result)["error"].startswith("--title and --decision")


def test_decision_add_help_lists_a_flag_per_field() -> None:
    result = CliRunner().invoke(cli, ["decision", "add", "--help"])

    assert result.exit_code == 0, result.output
    for flag in (
        "--title",
        "--context",
        "--decision",
        "--rationale",
        "--alternative",
        "--consequence",
        "--affects",
        "--tag",
        "--format",
    ):
        assert flag in result.output


def test_decision_list_help_lists_filters() -> None:
    result = CliRunner().invoke(cli, ["decision", "list", "--help"])

    assert result.exit_code == 0, result.output
    assert "--status" in result.output
    assert "--source" in result.output
    assert "--proposed" in result.output
    assert "--stale-only" in result.output


def test_decision_list_empty_repo_prints_none_found(indexed_repo: Path) -> None:
    result = CliRunner().invoke(cli, ["decision", "list", str(indexed_repo)])

    assert result.exit_code == 0, result.output
    assert "No decisions found." in result.output


def test_decision_list_and_show_seeded_records(indexed_repo: Path) -> None:
    _seed_wiki_db(
        indexed_repo,
        [
            {
                "id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "title": "Prefer SQLite locally",
                "status": "active",
                "source": "cli",
            },
            {
                "id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "title": "Propose Redis sessions",
                "status": "proposed",
                "source": "git_archaeology",
            },
        ],
    )

    listed = CliRunner().invoke(cli, ["decision", "list", str(indexed_repo)])
    assert listed.exit_code == 0, listed.output
    assert "aaaaaaaa" in listed.output
    assert "bbbbbbbb" in listed.output
    assert "Prefer" in listed.output and "SQLite" in listed.output
    assert "Propose" in listed.output and "Redis" in listed.output
    assert "active" in listed.output
    assert "proposed" in listed.output

    proposed = CliRunner().invoke(cli, ["decision", "list", "--proposed", str(indexed_repo)])
    assert proposed.exit_code == 0, proposed.output
    assert "bbbbbbbb" in proposed.output
    assert "Propose" in proposed.output
    assert "aaaaaaaa" not in proposed.output
    assert "Prefer" not in proposed.output

    shown = CliRunner().invoke(cli, ["decision", "show", "aaaaaaaa", str(indexed_repo)])
    assert shown.exit_code == 0, shown.output
    assert "Prefer SQLite locally" in shown.output
    assert "Status: active" in shown.output
    assert "use X" in shown.output


def test_decision_confirm_promotes_proposed(indexed_repo: Path) -> None:
    _seed_wiki_db(
        indexed_repo,
        [
            {
                "id": "cccccccccccccccccccccccccccccccc",
                "title": "Needs confirmation",
                "status": "proposed",
            }
        ],
    )

    result = CliRunner().invoke(cli, ["decision", "confirm", "cccccccc", str(indexed_repo)])
    assert result.exit_code == 0, result.output
    assert "confirmed (active)" in result.output

    shown = CliRunner().invoke(cli, ["decision", "show", "cccccccc", str(indexed_repo)])
    assert shown.exit_code == 0, shown.output
    assert "Status: active" in shown.output


def test_decision_dismiss_and_deprecate(indexed_repo: Path) -> None:
    _seed_wiki_db(
        indexed_repo,
        [
            {
                "id": "dddddddddddddddddddddddddddddddd",
                "title": "Dismiss me",
                "status": "proposed",
            },
            {
                "id": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                "title": "Deprecate me",
                "status": "active",
            },
        ],
    )

    dismissed = CliRunner().invoke(
        cli, ["decision", "dismiss", "dddddddd", str(indexed_repo)], input="y\n"
    )
    assert dismissed.exit_code == 0, dismissed.output
    assert "dismissed" in dismissed.output

    deprecated = CliRunner().invoke(cli, ["decision", "deprecate", "eeeeeeee", str(indexed_repo)])
    assert deprecated.exit_code == 0, deprecated.output
    assert "deprecated" in deprecated.output


def test_decision_ambiguous_prefix_errors(indexed_repo: Path) -> None:
    _seed_wiki_db(
        indexed_repo,
        [
            {"id": "ffff1111111111111111111111111111", "title": "One"},
            {"id": "ffff2222222222222222222222222222", "title": "Two"},
        ],
    )

    result = CliRunner().invoke(cli, ["decision", "show", "ffff", str(indexed_repo)])
    assert result.exit_code != 0
    assert "ambiguous" in result.output.lower()


def test_decision_health_prints_summary(indexed_repo: Path) -> None:
    _seed_wiki_db(
        indexed_repo,
        [
            {
                "id": "gggggggggggggggggggggggggggggggg",
                "title": "Active one",
                "status": "active",
            },
            {
                "id": "hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh",
                "title": "Proposed one",
                "status": "proposed",
            },
            {
                "id": "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii",
                "title": "Stale one",
                "status": "active",
                "staleness": 0.8,
            },
        ],
    )

    result = CliRunner().invoke(cli, ["decision", "health", str(indexed_repo)])
    assert result.exit_code == 0, result.output
    assert "Decision Health" in result.output
    assert "Active decisions" in result.output
    assert "Proposed (needs review)" in result.output


# ---------------------------------------------------------------------------
# --repo, and what the derived store is (F48)
# ---------------------------------------------------------------------------


def test_the_derived_store_says_it_is_machine_local(indexed_repo: Path) -> None:
    """Without a journal, a decision reaches nobody but this machine.

    ``.repowise`` is derived, gitignored and rebuilt by any reindex, so a row
    recorded there is gone on the next full index and was never visible to a
    teammate. The command looks identical in both modes, and the difference
    only ever showed up as a decision that had quietly disappeared. The MCP
    tool refuses to write in this state; the CLI has always written, so it says
    what it wrote instead of changing under callers that depend on it.
    """
    result = CliRunner().invoke(
        cli,
        [
            "decision",
            "add",
            "--title",
            "Machine-local for now",
            "--decision",
            "Write to the derived store",
            str(indexed_repo),
        ],
        input="",
    )

    assert result.exit_code == 0, result.output
    assert "Machine-local" in result.output
    assert "REPOWISE_DECISIONS_JOURNAL" in result.output


def test_a_journal_backed_add_says_nothing_about_being_machine_local(
    indexed_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPOWISE_DECISIONS_JOURNAL", ".codeatlas/decisions.jsonl")
    (indexed_repo / "src").mkdir(exist_ok=True)
    (indexed_repo / "src" / "thing.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "decision",
            "add",
            "--title",
            "Journal backed",
            "--decision",
            "Write to the journal",
            "--rationale",
            "git is the review surface",
            "--affects",
            "src/thing.py",
            str(indexed_repo),
        ],
        input="",
    )

    assert result.exit_code == 0, result.output
    assert "Machine-local" not in result.output


def test_repo_alias_is_refused_outside_a_workspace(
    indexed_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One repo has no aliases, so the flag has nothing to resolve against.

    Refused with the reason rather than silently ignored: an alias that does
    nothing would look like it had targeted something.
    """
    monkeypatch.chdir(indexed_repo)
    result = CliRunner().invoke(
        cli,
        ["decision", "list", "--repo", "graph"],
    )

    assert result.exit_code != 0
    assert "--repo graph needs a workspace" in result.output


def test_repo_alias_and_path_cannot_be_combined(indexed_repo: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["decision", "list", "--repo", "graph", str(indexed_repo)],
    )

    assert result.exit_code != 0
    assert "not both" in result.output


@pytest.mark.parametrize(
    "subcommand",
    ["add", "list", "show", "confirm", "dismiss", "deprecate", "health"],
)
def test_every_decision_subcommand_offers_the_alias(subcommand: str) -> None:
    """One surface, not five of seven.

    A flag present on some subcommands and absent on the rest is worse than
    one that is missing everywhere, because the shape of the gap has to be
    memorised.
    """
    result = CliRunner().invoke(cli, ["decision", subcommand, "--help"])

    assert result.exit_code == 0, result.output
    assert "--repo ALIAS" in result.output
