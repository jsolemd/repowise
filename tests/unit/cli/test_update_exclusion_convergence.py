"""A config exclusion converges every update-time persistence store."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select

from repowise.cli._repo_session import open_repo_db
from repowise.cli.commands.update_cmd.incremental import _build_repo_graph
from repowise.cli.commands.update_cmd.persistence import _persist_full_update_async
from repowise.core.persistence import FullTextSearch, get_session
from repowise.core.persistence.models import (
    GitMetadata,
    GraphNode,
    Page,
    SourceIndexUpdate,
    WikiSymbol,
)
from repowise.core.refsites.schema import ReferenceSite


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


async def test_update_sweeps_newly_excluded_file_from_all_derived_stores(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    for path in ("src/keep.py", "generated/drop.py"):
        target = repo_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"VALUE = {path!r}\n", encoding="utf-8")
    _git(repo_path, "init", "-q")
    _git(repo_path, "config", "user.name", "Test")
    _git(repo_path, "config", "user.email", "test@example.com")
    _git(repo_path, "add", ".")
    _git(repo_path, "commit", "-q", "-m", "seed")

    engine, sf, repo_id = await open_repo_db(repo_path, repo_name="repo")
    now = datetime.now(UTC)
    async with get_session(sf) as session:
        for path in ("src/keep.py", "generated/drop.py"):
            session.add(GraphNode(repository_id=repo_id, node_id=path, node_type="file"))
            session.add(GitMetadata(repository_id=repo_id, file_path=path))
            session.add(
                WikiSymbol(
                    repository_id=repo_id,
                    file_path=path,
                    symbol_id=f"{path}::VALUE",
                    name="VALUE",
                    qualified_name="VALUE",
                    kind="variable",
                )
            )
            session.add(
                ReferenceSite(
                    repository_id=repo_id,
                    file_path=path,
                    language="python",
                    name="VALUE",
                    kind="read",
                    start_line=1,
                    end_line=1,
                    start_col=0,
                    end_col=5,
                    range_exact=True,
                    resolution_origin="unresolved",
                    confidence=0.0,
                    tier="D",
                    extractor_version="test",
                )
            )
            session.add(
                Page(
                    id=f"file_page:{path}",
                    repository_id=repo_id,
                    page_type="file_page",
                    title=path,
                    content=f"{path} handles xylophone tuning.",
                    target_path=path,
                    source_hash="x" * 64,
                    model_name="mock",
                    provider_name="mock",
                    created_at=now,
                    updated_at=now,
                )
            )

    fts = FullTextSearch(engine)
    await fts.ensure_index()
    for path in ("src/keep.py", "generated/drop.py"):
        await fts.index(
            f"file_page:{path}",
            path,
            f"{path} handles xylophone tuning.",
        )
    await engine.dispose()

    config = repo_path / ".repowise" / "config.yaml"
    config.write_text("exclude_patterns:\n  - generated/**\n", encoding="utf-8")
    parsed_files, _sources, graph_builder, _structure, _count = _build_repo_graph(
        repo_path,
        ["generated/**"],
    )
    assert {pf.file_info.path for pf in parsed_files} == {"src/keep.py"}

    class _VectorStore:
        _embedder = None

        def __init__(self) -> None:
            self.deleted: list[str] = []

        async def delete_many(self, page_ids: list[str]) -> None:
            self.deleted.extend(page_ids)

    vectors = _VectorStore()

    async def _reconcile(*_args, **_kwargs):
        return None

    from repowise.cli import source_search_runtime

    monkeypatch.setattr(source_search_runtime, "reconcile_configured_source_index", _reconcile)
    result = await _persist_full_update_async(
        repo_path=repo_path,
        repo_name="repo",
        generated_pages=[],
        file_diffs=[
            SimpleNamespace(
                path=".repowise/config.yaml",
                old_path=None,
                status="modified",
            )
        ],
        git_meta_map={},
        new_decision_markers=[],
        decision_vector_store=vectors,
        provider=None,
        partial_health_report=None,
        dead_code_report=None,
        graph_builder=graph_builder,
        knowledge_graph_result=None,
        degraded=[],
        parsed_files=parsed_files,
    )

    assert result.prune_outcome.pruned_paths == 1
    excluded_page = "file_page:generated/drop.py"
    assert vectors.deleted == [excluded_page]

    engine, sf, _ = await open_repo_db(repo_path, repo_name="repo")
    async with get_session(sf) as session:
        symbol_paths = set(
            (
                await session.execute(
                    select(WikiSymbol.file_path).where(WikiSymbol.repository_id == repo_id)
                )
            )
            .scalars()
            .all()
        )
        refsite_paths = set(
            (
                await session.execute(
                    select(ReferenceSite.file_path).where(ReferenceSite.repository_id == repo_id)
                )
            )
            .scalars()
            .all()
        )
        page = await session.get(Page, excluded_page)
        full_updates = (
            (
                await session.execute(
                    select(SourceIndexUpdate).where(
                        SourceIndexUpdate.repository_id == repo_id,
                        SourceIndexUpdate.mode == "full",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert symbol_paths == {"src/keep.py"}
    assert "generated/drop.py" not in refsite_paths
    assert page is not None and page.freshness_status == "tombstone"
    assert len(full_updates) == 1

    fts = FullTextSearch(engine)
    await fts.ensure_index()
    hits = {hit.page_id for hit in await fts.search("xylophone", limit=10)}
    await engine.dispose()
    assert excluded_page not in hits
    assert "file_page:src/keep.py" in hits
