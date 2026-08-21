"""Saved-file capture for the watcher source fast lane."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from sqlalchemy import select

from repowise.core.ingestion import ASTParser, FileTraverser
from repowise.core.ingestion.models import compute_content_hash
from repowise.core.persistence.crud import reconcile_symbols_for_files
from repowise.core.persistence.database import (
    create_engine,
    create_session_factory,
    get_session,
    init_db,
    resolve_db_url,
)
from repowise.core.persistence.models import Repository, SourceIndexUpdate, WikiSymbol
from repowise.core.source_search.fast_update import capture_source_changes


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


async def _seed(repo: Path) -> None:
    engine = create_engine(resolve_db_url(repo))
    try:
        await init_db(engine)
        factory = create_session_factory(engine)
        info = FileTraverser(repo).file_info_for_path("src/app.py")
        assert info is not None
        parsed = ASTParser().parse_file(info, (repo / "src" / "app.py").read_bytes())
        for symbol in parsed.symbols:
            if not getattr(symbol, "file_path", None):
                symbol.file_path = parsed.file_info.path
        async with get_session(factory) as session:
            repository = Repository(id="r1", name="repo", local_path=str(repo))
            session.add(repository)
            await session.flush()
            await reconcile_symbols_for_files(
                session,
                repository.id,
                ["src/app.py"],
                parsed.symbols,
            )
    finally:
        await engine.dispose()


async def _stored(repo: Path) -> tuple[list[WikiSymbol], list[SourceIndexUpdate]]:
    engine = create_engine(resolve_db_url(repo))
    try:
        factory = create_session_factory(engine)
        async with factory() as session:
            symbols = list((await session.execute(select(WikiSymbol))).scalars().all())
            updates = list(
                (
                    await session.execute(
                        select(SourceIndexUpdate).order_by(SourceIndexUpdate.sequence)
                    )
                )
                .scalars()
                .all()
            )
            return symbols, updates
    finally:
        await engine.dispose()


@pytest.fixture
async def fast_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("def old_name():\n    return 1\n")
    await _seed(repo)
    return repo


@pytest.fixture
async def excluded_repo(fast_repo: Path) -> Path:
    """A git corpus whose parser exclusions must not erase source windows."""

    repo = fast_repo
    (repo / "infra").mkdir()
    (repo / "infra" / "up.sh").write_text("#!/bin/sh\necho oldshell\n")
    (repo / "infra" / "service.yaml").write_text("service: oldyaml\n")
    (repo / ".gitignore").write_text("infra/service.yaml\n")
    (repo / ".repowise" / "config.yaml").write_text(
        'exclude_patterns:\n  - "**/*.py"\n  - "**/*.sh"\n  - "**/*.yaml"\n'
    )

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "add", "-f", "src/app.py", "infra/up.sh", "infra/service.yaml")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "seed")
    return repo


async def test_saved_file_symbols_and_outbox_commit_together(fast_repo: Path) -> None:
    (fast_repo / "src" / "app.py").write_text("def new_name():\n    return 2\n")

    result = await capture_source_changes(fast_repo, {"src/app.py"})

    symbols, updates = await _stored(fast_repo)
    assert result.parsed == 1
    assert [symbol.name for symbol in symbols] == ["new_name"]
    assert len(updates) == 1
    change = json.loads(updates[0].change_set_json)[0]
    assert change["path"] == "src/app.py"
    assert change["parse_state"] == "parsed"
    assert change["content_hash"]


async def test_deleted_file_prunes_symbols_and_records_a_tombstone(fast_repo: Path) -> None:
    (fast_repo / "src" / "app.py").unlink()

    result = await capture_source_changes(fast_repo, ["src/app.py"])

    symbols, updates = await _stored(fast_repo)
    assert result.removed == 1
    assert symbols == []
    change = json.loads(updates[0].change_set_json)[0]
    assert change["status"] == "deleted"
    assert change["parse_state"] == "deleted"


async def test_parser_failure_retains_last_good_symbols(fast_repo: Path, monkeypatch) -> None:
    (fast_repo / "src" / "app.py").write_text("def replacement():\n    return 3\n")

    def _fail_parse(*_args, **_kwargs):
        raise RuntimeError("synthetic parser crash")

    monkeypatch.setattr(ASTParser, "parse_file", _fail_parse)
    result = await capture_source_changes(fast_repo, ["src/app.py"])

    symbols, updates = await _stored(fast_repo)
    assert result.retained_stale == 1
    assert [symbol.name for symbol in symbols] == ["old_name"]
    assert json.loads(updates[0].change_set_json)[0]["parse_state"] == "failed"


async def test_outbox_failure_rolls_back_symbol_replacement(fast_repo: Path, monkeypatch) -> None:
    from repowise.core.source_search import outbox

    (fast_repo / "src" / "app.py").write_text("def uncommitted_name():\n    return 4\n")

    async def _fail_enqueue(*_args, **_kwargs):
        raise RuntimeError("outbox rejected")

    monkeypatch.setattr(outbox, "enqueue_incremental_update", _fail_enqueue)
    with pytest.raises(RuntimeError, match="outbox rejected"):
        await capture_source_changes(fast_repo, ["src/app.py"])

    symbols, updates = await _stored(fast_repo)
    assert [symbol.name for symbol in symbols] == ["old_name"]
    assert updates == []


@pytest.mark.parametrize(
    ("path", "contents"),
    [
        ("infra/up.sh", "#!/bin/sh\necho newshell\n"),
        ("infra/service.yaml", "service: newyaml\n"),
    ],
)
async def test_tracked_excluded_operational_files_capture_fresh_windows(
    excluded_repo: Path,
    path: str,
    contents: str,
) -> None:
    absolute = excluded_repo / path
    absolute.write_text(contents)

    result = await capture_source_changes(excluded_repo, [path])

    symbols, updates = await _stored(excluded_repo)
    change = json.loads(updates[0].change_set_json)[0]
    assert result.parsed == 0
    assert result.removed == 1
    assert [symbol.name for symbol in symbols] == ["old_name"]
    assert change["parse_state"] == "window"
    assert change["content_hash"] == compute_content_hash(absolute.read_bytes())


async def test_tracked_parser_code_excluded_by_config_becomes_a_fresh_window(
    excluded_repo: Path,
) -> None:
    path = "src/app.py"
    absolute = excluded_repo / path
    absolute.write_text("def replacement():\n    return 2\n")

    result = await capture_source_changes(excluded_repo, [path])

    symbols, updates = await _stored(excluded_repo)
    change = json.loads(updates[0].change_set_json)[0]
    assert result.parsed == 0
    assert result.removed == 1
    assert symbols == []
    assert change["parse_state"] == "window"
    assert change["content_hash"] == compute_content_hash(absolute.read_bytes())


@pytest.mark.parametrize("gitignored", [False, True])
async def test_untracked_excluded_windows_remain_unindexed(
    excluded_repo: Path,
    gitignored: bool,
) -> None:
    name = "ignored.sh" if gitignored else "untracked.sh"
    path = f"infra/{name}"
    if gitignored:
        with (excluded_repo / ".gitignore").open("a", encoding="utf-8") as handle:
            handle.write(f"{path}\n")
    (excluded_repo / path).write_text("#!/bin/sh\necho scratch\n")

    await capture_source_changes(excluded_repo, [path])

    symbols, updates = await _stored(excluded_repo)
    change = json.loads(updates[0].change_set_json)[0]
    assert [symbol.name for symbol in symbols] == ["old_name"]
    assert change["parse_state"] == "unindexed"


async def test_untracked_parser_file_still_uses_the_normal_fast_path(fast_repo: Path) -> None:
    path = "src/new_file.py"
    (fast_repo / path).write_text("def untracked_name():\n    return 2\n")

    result = await capture_source_changes(fast_repo, [path])

    symbols, updates = await _stored(fast_repo)
    assert result.parsed == 1
    assert sorted(symbol.name for symbol in symbols) == ["old_name", "untracked_name"]
    assert json.loads(updates[0].change_set_json)[0]["parse_state"] == "parsed"


async def test_tracked_window_git_discovery_failure_is_fail_loud(
    excluded_repo: Path,
    monkeypatch,
) -> None:
    from repowise.core.source_search import indexer

    def _fail_tracked_paths(_repo: Path) -> list[str]:
        raise RuntimeError("git tracked-file discovery failed")

    monkeypatch.setattr(indexer, "_tracked_paths", _fail_tracked_paths)
    path = "infra/up.sh"
    (excluded_repo / path).write_text("#!/bin/sh\necho replacement\n")

    with pytest.raises(RuntimeError, match="tracked-file discovery failed"):
        await capture_source_changes(excluded_repo, [path])

    symbols, updates = await _stored(excluded_repo)
    assert [symbol.name for symbol in symbols] == ["old_name"]
    assert updates == []


@pytest.mark.parametrize(
    ("staged", "destination_state"),
    [(True, "window"), (False, "unindexed")],
)
async def test_excluded_window_rename_tombstones_the_old_path_and_classifies_the_new_one(
    excluded_repo: Path,
    staged: bool,
    destination_state: str,
) -> None:
    old_path = "infra/up.sh"
    new_path = "infra/renamed.sh"
    if staged:
        _git(excluded_repo, "mv", old_path, new_path)
    else:
        (excluded_repo / old_path).rename(excluded_repo / new_path)

    await capture_source_changes(excluded_repo, [old_path, new_path])

    _symbols, updates = await _stored(excluded_repo)
    changes = {change["path"]: change for change in json.loads(updates[0].change_set_json)}
    assert changes[old_path]["parse_state"] == "deleted"
    assert changes[new_path]["parse_state"] == destination_state
    if destination_state == "window":
        assert changes[new_path]["content_hash"] == compute_content_hash(
            (excluded_repo / new_path).read_bytes()
        )


async def test_unreadable_tracked_window_retains_last_good_symbols(
    excluded_repo: Path,
    monkeypatch,
) -> None:
    from repowise.core.source_search import fast_update

    path = "src/app.py"
    (excluded_repo / path).write_text("def replacement():\n    return 2\n")
    monkeypatch.setattr(fast_update, "_safe_hash", lambda _path: None)

    result = await capture_source_changes(excluded_repo, [path])

    symbols, updates = await _stored(excluded_repo)
    change = json.loads(updates[0].change_set_json)[0]
    assert result.removed == 0
    assert result.retained_stale == 1
    assert [symbol.name for symbol in symbols] == ["old_name"]
    assert change["parse_state"] == "failed"
