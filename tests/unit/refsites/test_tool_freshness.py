"""Served positions are as fresh as the last index, and the response says so.

``mechanically_safe`` is a positive safety assertion — "a rewriter could patch
this site unattended" — and it holds only while the served line/column
positions are current. These tests edit a real checkout after indexing and ask
both tools whether they still assert what they can no longer back.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from repowise.core.persistence.crud import upsert_repository
from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.store import SqlReferenceSiteStore
from repowise.core.source_search.worktree import reset_cache_for_tests
from repowise.server.mcp_server.tool_refsites import (
    get_reference_sites,
    preview_symbol_rename,
)
from tests.unit.refsites.conftest import TARGET_SYMBOL_ID, TOY_REPO


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin",
            "HOME": str(repo),
        },
    )


@pytest.fixture
async def committed_indexed(async_engine, tmp_path):
    """A real committed checkout, indexed, with the repo row pointing at it."""
    reset_cache_for_tests()
    repo = tmp_path / "checkout"
    repo.mkdir()
    for name, body in TOY_REPO.items():
        (repo / name).write_text(body)
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "seed")

    factory = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        row = await upsert_repository(
            session,
            name="refsites-freshness",
            local_path=str(repo),
            url="https://example.invalid/refsites-freshness",
        )
        store = SqlReferenceSiteStore(session)
        await store.replace_repository(row.id, extract_repository(repo))
        await session.commit()
        repo_row_id = row.id

    import repowise.server.mcp_server as mcp_mod

    mcp_mod._session_factory = factory
    mcp_mod._repo_path = str(repo)
    yield repo
    mcp_mod._session_factory = None
    mcp_mod._repo_path = None
    mcp_mod._registry = None
    reset_cache_for_tests()
    del repo_row_id


async def test_a_current_checkout_carries_meta_and_full_safety(committed_indexed):
    sites = await get_reference_sites(TARGET_SYMBOL_ID)
    preview = await preview_symbol_rename(TARGET_SYMBOL_ID)

    assert "_meta" in sites and "_meta" in preview
    assert "working_tree" not in sites["_meta"]
    assert "working_tree" not in preview["_meta"]
    assert preview["summary"]["mechanically_safe"] == 10
    assert not any("changed after the index" in caveat for caveat in preview["caveats"])


async def test_an_uncommitted_edit_demotes_safety_and_says_so(committed_indexed):
    """The blocker scenario: prepend lines after indexing, never re-index.

    Every widget.ts position in the store is now wrong. The preview must stop
    asserting those sites are unattended-rewrite safe, say why first, and
    disclose the divergence in ``_meta`` where envelope readers look.
    """
    reset_cache_for_tests()
    widget = committed_indexed / "widget.ts"
    widget.write_text("// shifted\n" * 40 + widget.read_text())

    preview = await preview_symbol_rename(TARGET_SYMBOL_ID)

    assert preview["_meta"]["working_tree"]["served_modified"] == ["widget.ts"]
    assert "changed after the index" in preview["caveats"][0]
    assert "widget.ts" in preview["caveats"][0]
    for site in preview["sites"]:
        if site["file"] == "widget.ts":
            assert site["mechanically_safe"] is False, site
    demoted = sum(1 for s in preview["sites"] if s["file"] == "widget.ts")
    assert demoted > 0
    assert preview["summary"]["mechanically_safe"] < 10
    assert (
        preview["summary"]["mechanically_safe"] + preview["summary"]["needs_review"]
        == preview["summary"]["total"]
    )

    sites = await get_reference_sites(TARGET_SYMBOL_ID)
    assert sites["_meta"]["working_tree"]["served_modified"] == ["widget.ts"]


async def test_a_deleted_served_file_is_disclosed_as_deleted(committed_indexed):
    reset_cache_for_tests()
    (committed_indexed / "widget.ts").unlink()

    preview = await preview_symbol_rename(TARGET_SYMBOL_ID)

    assert preview["_meta"]["working_tree"]["served_deleted"] == ["widget.ts"]
    for site in preview["sites"]:
        if site["file"] == "widget.ts":
            assert site["mechanically_safe"] is False, site


async def test_an_empty_answer_still_carries_meta(committed_indexed):
    result = await get_reference_sites("definitely_not_a_symbol_anywhere")

    assert result["status"] == "not_found"
    assert "_meta" in result
