"""Unit tests for git-worktree detection and seed preconditions."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from repowise.cli.worktree import (
    base_is_seedable,
    detect_worktree_base,
    seed_index_from_base,
    sweep_stale_seed_backups,
)


def _git(args: list[str], cwd: Path) -> None:
    subprocess.check_call(
        ["git", *args],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


@pytest.fixture()
def base_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "base"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.email", "t@t.t"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "init"], repo)
    return repo


def test_plain_repo_is_not_a_worktree(base_repo: Path) -> None:
    assert detect_worktree_base(base_repo) is None


def test_non_git_dir_is_not_a_worktree(tmp_path: Path) -> None:
    d = tmp_path / "plain"
    d.mkdir()
    assert detect_worktree_base(d) is None


def test_linked_worktree_resolves_base(base_repo: Path, tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    _git(["worktree", "add", "-b", "feature", str(wt)], base_repo)
    detected = detect_worktree_base(wt)
    assert detected is not None
    assert detected.resolve() == base_repo.resolve()


def test_submodule_git_file_is_not_a_worktree(base_repo: Path, tmp_path: Path) -> None:
    # Fake a submodule layout: .git file whose gitdir points at the parent's
    # modules dir, which does not end in a bare ".git" component.
    sub = tmp_path / "sub"
    sub.mkdir()
    modules = base_repo / ".git" / "modules" / "sub"
    modules.mkdir(parents=True)
    (sub / ".git").write_text(f"gitdir: {modules}\n", encoding="utf-8")
    assert detect_worktree_base(sub) is None


def test_base_is_seedable_requires_state_and_db(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    (repo / ".repowise").mkdir(parents=True)
    assert not base_is_seedable(repo)
    (repo / ".repowise" / "state.json").write_text("{}", encoding="utf-8")
    assert not base_is_seedable(repo)
    (repo / ".repowise" / "wiki.db").write_text("", encoding="utf-8")
    assert base_is_seedable(repo)


@pytest.fixture
def seed_pair(base_repo: Path, tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    monkeypatch.delenv("REPOWISE_DECISIONS_JOURNAL", raising=False)
    state = base_repo / ".repowise"
    state.mkdir()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base_repo, text=True).strip()
    (state / "state.json").write_text(json.dumps({"last_sync_commit": head}), encoding="utf-8")
    with sqlite3.connect(state / "wiki.db") as connection:
        connection.execute("CREATE TABLE repositories (id TEXT, name TEXT, local_path TEXT)")
    target = tmp_path / "worktree"
    _git(["worktree", "add", "-b", "seed-target", str(target)], base_repo)
    return base_repo, target


@pytest.mark.parametrize("owner", ["source", "target", "both"])
@pytest.mark.parametrize(
    "filename", ["decisions.jsonl", "decisions.yaml", "custom/decisions.jsonl"]
)
def test_seed_preserves_each_checkouts_authored_records(seed_pair, monkeypatch, owner, filename):
    source, target = seed_pair
    if filename.startswith("custom/"):
        monkeypatch.setenv("REPOWISE_DECISIONS_JOURNAL", f".repowise/{filename}")
    originals = {}
    for name, repo in (("source", source), ("target", target)):
        if owner in (name, "both"):
            path = repo / ".repowise" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            originals[path] = f"{name}'s authored records\n".encode()
            path.write_bytes(originals[path])

    assert not seed_index_from_base(root=target, repo_paths=[target], seed_base=source)

    assert all(path.read_bytes() == content for path, content in originals.items())
    if owner == "source":
        assert not (target / ".repowise" / filename).exists()
    assert not list(target.glob(".repowise-seed-*"))
    assert not list(target.glob(".repowise.bak.*"))


def test_generated_index_still_seeds_normally(seed_pair):
    source, target = seed_pair
    assert seed_index_from_base(root=target, repo_paths=[target], seed_base=source)
    assert json.loads((target / ".repowise/state.json").read_text()) == json.loads(
        (source / ".repowise/state.json").read_text()
    )
    assert (target / ".repowise/wiki.db").is_file()


@pytest.mark.parametrize("authored", [False, True])
def test_seed_failure_preserves_new_authored_records_but_cleans_generated_state(
    seed_pair, monkeypatch, authored
):
    source, target = seed_pair

    def fail_staging(directory, **kwargs):
        if authored:
            (directory / "decisions.jsonl").write_bytes(b"records created during staging\n")
        raise RuntimeError("staging failed")

    monkeypatch.setattr("repowise.cli.worktree._adopt_repository_identity", fail_staging)
    assert not seed_index_from_base(root=target, repo_paths=[target], seed_base=source)
    leftovers = list(target.glob(".repowise-seed-*"))
    assert len(leftovers) == int(authored)
    if authored:
        assert (
            leftovers[0] / "decisions.jsonl"
        ).read_bytes() == b"records created during staging\n"
    assert not (target / ".repowise").exists()


def test_seed_rechecks_target_before_replacing_state(seed_pair, monkeypatch):
    source, target = seed_pair
    journal = target / ".repowise/decisions.jsonl"

    def record_during_staging(directory, **kwargs):
        journal.parent.mkdir()
        journal.write_bytes(b"new target record\n")

    monkeypatch.setattr("repowise.cli.worktree._adopt_repository_identity", record_during_staging)
    assert not seed_index_from_base(root=target, repo_paths=[target], seed_base=source)
    assert journal.read_bytes() == b"new target record\n"
    assert not list(target.glob(".repowise-seed-*"))
    assert not list(target.glob(".repowise.bak.*"))


@pytest.mark.parametrize("prefix", [".repowise.bak.", ".repowise-seed-"])
def test_backup_sweep_preserves_authored_records(tmp_path, monkeypatch, prefix):
    monkeypatch.delenv("REPOWISE_DECISIONS_JOURNAL", raising=False)
    authored = tmp_path / f"{prefix}authored"
    generated = tmp_path / f"{prefix}generated"
    authored.mkdir()
    generated.mkdir()
    journal = authored / "decisions.jsonl"
    journal.write_bytes(b"the only copy of a decision\n")
    (generated / "wiki.db").write_bytes(b"rebuildable")
    for path in (authored, generated):
        os.utime(path, (0, 0))

    sweep_stale_seed_backups([tmp_path])

    assert journal.read_bytes() == b"the only copy of a decision\n"
    assert not generated.exists()


def test_seed_dry_run_does_not_sweep_backups(seed_pair):
    source, target = seed_pair
    backup = target / ".repowise.bak.old"
    backup.mkdir()
    (backup / "wiki.db").write_bytes(b"old index")
    assert seed_index_from_base(root=target, repo_paths=[target], seed_base=source, dry_run=True)
    assert (backup / "wiki.db").read_bytes() == b"old index"


def test_adopting_identity_rekeys_the_repo_level_pages(tmp_path: Path) -> None:
    """The overview id carries the repo name; the seeded copy must be findable and regenerable."""
    import sqlite3
    from contextlib import closing

    from repowise.cli.worktree import _adopt_repository_identity

    src, dest = tmp_path / "base", tmp_path / "feature-tree"
    db = tmp_path / "wiki.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.executescript(
            "CREATE TABLE repositories (id TEXT, name TEXT, local_path TEXT);"
            "CREATE TABLE wiki_pages (id TEXT, repository_id TEXT, page_type TEXT,"
            " target_path TEXT, parent_page_id TEXT);"
            "CREATE TABLE wiki_page_versions (id TEXT, page_id TEXT);"
            "CREATE VIRTUAL TABLE page_fts USING fts5(page_id UNINDEXED, title, target_path);"
        )
        conn.execute("INSERT INTO repositories VALUES ('r1', 'base', ?)", (str(src),))
        conn.execute(
            "INSERT INTO wiki_pages VALUES ('repo_overview:base', 'r1', 'repo_overview', 'base', NULL)"
        )
        conn.execute(
            "INSERT INTO wiki_pages VALUES ('module_page:src', 'r1', 'module_page', 'src',"
            " 'repo_overview:base')"
        )
        conn.execute("INSERT INTO wiki_page_versions VALUES ('v1', 'repo_overview:base')")
        conn.execute("INSERT INTO page_fts VALUES ('repo_overview:base', 'Overview', 'base')")
        conn.commit()

    _adopt_repository_identity(tmp_path, src_repo=src, dest_repo=dest)

    with closing(sqlite3.connect(db)) as conn:
        pages = conn.execute("SELECT id, target_path, parent_page_id FROM wiki_pages").fetchall()
        versions = conn.execute("SELECT page_id FROM wiki_page_versions").fetchall()
        fts = conn.execute("SELECT page_id, target_path FROM page_fts").fetchall()
    assert ("repo_overview:feature-tree", "feature-tree", None) in pages
    assert ("module_page:src", "src", "repo_overview:feature-tree") in pages
    assert versions == [("repo_overview:feature-tree",)]
    assert fts == [("repo_overview:feature-tree", "feature-tree")]


def test_as_commit_id_accepts_only_hex_commit_ids() -> None:
    from repowise.cli.helpers import as_commit_id

    assert as_commit_id("a" * 40) == "a" * 40
    assert as_commit_id("DeadBee") == "DeadBee"
    for bad in ("--output=x", "-abcdef0", "abc123", "a" * 41, "HEAD", "", None, 7):
        assert as_commit_id(bad) is None
