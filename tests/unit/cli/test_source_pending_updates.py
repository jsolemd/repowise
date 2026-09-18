"""A current recipe must not hide retryable source publication work."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from repowise.cli.source_search_runtime import configured_source_pending_updates
from repowise.core.persistence.database import (
    create_engine,
    create_session_factory,
    get_session,
    init_db,
    resolve_db_url,
)
from repowise.core.persistence.models import Repository, SourceIndexUpdate
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    default_manifest_path,
    recipe_fingerprint,
    write_manifest,
)
from repowise.core.source_search.outbox import enqueue_full_update


@pytest.fixture(autouse=True)
def isolated_source_configuration(monkeypatch):
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")
    monkeypatch.delenv("REPOWISE_DB_URL", raising=False)
    monkeypatch.delenv("REPOWISE_DATABASE_URL", raising=False)

    from repowise.cli.providers import embedders

    def unexpected_embedder(*_args, **_kwargs):
        raise AssertionError("queue inspection must not construct an embedder")

    monkeypatch.setattr(embedders, "build_embedder", unexpected_embedder)


async def _seed_update(repo: Path, state: str, *, publish: bool = False) -> int:
    (repo / ".repowise").mkdir(parents=True, exist_ok=True)
    engine = create_engine(resolve_db_url(repo))
    try:
        await init_db(engine)
        async with get_session(create_session_factory(engine)) as session:
            repository = (await session.execute(select(Repository))).scalar_one_or_none()
            if repository is None:
                repository = Repository(id="repo", name="repo", local_path=str(repo))
                session.add(repository)
                await session.flush()
            update = await enqueue_full_update(session, repository.id, repo)
            update.state = state
            generation_id, sequence = update.generation_id, update.sequence
        if publish:
            embedder = EmbedderIdentity(provider="mock", model="test", dims=8)
            write_manifest(
                default_manifest_path(repo),
                SourceIndexManifest(
                    recipe_fingerprint=recipe_fingerprint(embedder),
                    corpus_hash="empty",
                    symbol_chunks=0,
                    file_window_chunks=0,
                    files_covered=0,
                    indexed_commit=None,
                    built_at="2026-09-18T00:00:00Z",
                    embedder=embedder,
                    generation_id=generation_id,
                    generation_sequence=sequence,
                ),
            )
        return sequence
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("pending", True),
        ("building", True),
        ("ready", True),
        ("blocked", False),
        ("published", False),
    ],
)
async def test_existing_queue_is_inspected_without_a_manifest(tmp_path, state, expected):
    await _seed_update(tmp_path, state)

    assert await configured_source_pending_updates(tmp_path) is expected

    assert not default_manifest_path(tmp_path).exists()
    assert not (tmp_path / ".repowise" / "lancedb").exists()
    assert not (tmp_path / ".repowise" / "source_search").exists()
    engine = create_engine(resolve_db_url(tmp_path))
    try:
        async with create_session_factory(engine)() as session:
            rows = list((await session.execute(select(SourceIndexUpdate))).scalars())
            assert [row.state for row in rows] == [state]
    finally:
        await engine.dispose()


async def test_current_manifest_does_not_hide_a_later_pending_update(tmp_path):
    published = await _seed_update(tmp_path, "published", publish=True)
    assert await configured_source_pending_updates(tmp_path) is False

    pending = await _seed_update(tmp_path, "pending")

    assert pending > published
    assert await configured_source_pending_updates(tmp_path) is True


async def test_unreadable_manifest_does_not_hide_the_queue_or_get_rewritten(tmp_path):
    await _seed_update(tmp_path, "pending")
    manifest = default_manifest_path(tmp_path)
    manifest.write_text("{interrupted publication", encoding="utf-8")

    assert await configured_source_pending_updates(tmp_path) is True
    assert manifest.read_text(encoding="utf-8") == "{interrupted publication"


async def test_configured_database_location_is_respected(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    external_db = tmp_path / "elsewhere.db"
    monkeypatch.setenv("REPOWISE_DB_URL", f"sqlite+aiosqlite:///{external_db}")
    await _seed_update(repo, "pending")

    assert await configured_source_pending_updates(repo) is True
    assert not (repo / ".repowise" / "wiki.db").exists()


async def test_missing_database_is_not_initialized(tmp_path, monkeypatch):
    from repowise.core.source_search import status

    inspect = AsyncMock(side_effect=AssertionError("missing database must not be opened"))
    monkeypatch.setattr(status, "inspect_source_index", inspect)

    assert await configured_source_pending_updates(str(tmp_path)) is False
    inspect.assert_not_awaited()
    assert not (tmp_path / ".repowise").exists()


async def test_feature_disabled_does_not_inspect_existing_work(tmp_path, monkeypatch):
    from repowise.core.source_search import status

    await _seed_update(tmp_path, "pending")
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "0")
    inspect = AsyncMock(side_effect=AssertionError("disabled lane must not be inspected"))
    monkeypatch.setattr(status, "inspect_source_index", inspect)

    assert await configured_source_pending_updates(tmp_path) is False
    inspect.assert_not_awaited()
