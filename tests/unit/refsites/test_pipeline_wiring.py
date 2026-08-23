"""The index pipeline keeps the reference-site store current.

``persist_reference_sites`` is the seam both the init and update persisters
call. What has to stay true: it consumes the pipeline's own ``ParsedFile``
list (never re-parsing), it produces the same store a direct
``extract_repository`` would, it can recover the repo root on its own, and it
is a full replace — a second run leaves counts identical, not doubled.
"""

from __future__ import annotations

from sqlalchemy import func, select

from repowise.core.ingestion import ASTParser, FileTraverser
from repowise.core.pipeline.persist import persist_reference_sites
from repowise.core.refsites.pipeline import extract_repository
from repowise.core.refsites.schema import ReferenceSite, ReferenceSiteCoverage
from repowise.core.refsites.store import SqlReferenceSiteStore


def _pipeline_parse(root):
    """Parse the way the pipeline does: traverse + ASTParser over raw bytes."""
    parser = ASTParser()
    parsed = []
    for file_info in FileTraverser(root).traverse():
        raw = (root / file_info.path).read_bytes()
        parsed.append(parser.parse_file(file_info, raw))
    return parsed


async def test_wiring_writes_the_same_store_a_direct_extraction_would(
    async_session, repo_id, toy_repo
):
    parsed = _pipeline_parse(toy_repo)

    written = await persist_reference_sites(async_session, repo_id, parsed, repo_path=toy_repo)
    await async_session.commit()

    direct = extract_repository(toy_repo)
    assert written == len(direct.sites)
    store = SqlReferenceSiteStore(async_session)
    assert await store.count(repo_id) == len(direct.sites)

    coverage_rows = await async_session.execute(
        select(func.count())
        .select_from(ReferenceSiteCoverage)
        .where(ReferenceSiteCoverage.repository_id == repo_id)
    )
    assert coverage_rows.scalar_one() == len(direct.coverage)


async def test_wiring_derives_the_repo_root_when_not_told(async_session, repo_id, toy_repo):
    parsed = _pipeline_parse(toy_repo)

    written = await persist_reference_sites(async_session, repo_id, parsed)
    await async_session.commit()

    assert written == len(extract_repository(toy_repo).sites)


async def test_a_second_refresh_replaces_rather_than_accumulates(async_session, repo_id, toy_repo):
    parsed = _pipeline_parse(toy_repo)

    first = await persist_reference_sites(async_session, repo_id, parsed, repo_path=toy_repo)
    await async_session.commit()
    second = await persist_reference_sites(async_session, repo_id, parsed, repo_path=toy_repo)
    await async_session.commit()

    assert first == second
    rows = await async_session.execute(
        select(func.count())
        .select_from(ReferenceSite)
        .where(ReferenceSite.repository_id == repo_id)
    )
    assert rows.scalar_one() == first


async def test_a_partial_list_wipes_which_is_why_callers_must_pass_the_whole_tree(
    async_session, repo_id, toy_repo
):
    """Documentation of the hazard, pinned.

    The seam is a full replace: rows for files absent from *parsed_files* are
    deleted. That is correct only because both production callers pass the
    whole tree's parse (init's PipelineResult, the update path's full-tree
    rebuild). If a future refactor routes a changed-file slice here, this is
    the data loss it causes — and this test is the tripwire that names the
    precondition it broke.
    """
    parsed = _pipeline_parse(toy_repo)
    full = await persist_reference_sites(async_session, repo_id, parsed, repo_path=toy_repo)
    await async_session.commit()

    partial = [pf for pf in parsed if pf.file_info.path == "widget.ts"]
    after = await persist_reference_sites(async_session, repo_id, partial, repo_path=toy_repo)
    await async_session.commit()

    assert after < full
    store = SqlReferenceSiteStore(async_session)
    assert await store.count(repo_id) == after
