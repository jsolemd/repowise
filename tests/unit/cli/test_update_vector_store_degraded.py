"""Tests for the update path's decision vector store (issue #1370).

``_build_update_vector_store`` swallows failures and returns None so the
decision upsert still works without a store — but the failure must land in
the run's ``degraded`` list, or the panel says nothing while semantic dedup
is silently off.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from repowise.cli.commands.update_cmd.incremental import (
    _build_update_vector_store,
    cleanup_retired_page_vectors,
)
from repowise.core.pipeline.cleanup_debt import load_cleanup_debt, record_cleanup_debt


@pytest.fixture
def page_catalog(tmp_path):
    from repowise.core.persistence.database import create_engine, init_db, resolve_db_url

    (tmp_path / ".repowise").mkdir()

    async def initialize():
        engine = create_engine(resolve_db_url(tmp_path))
        try:
            await init_db(engine)
        finally:
            await engine.dispose()

    asyncio.run(initialize())
    return tmp_path


def test_failure_is_recorded_in_degraded() -> None:
    """A store build failure must surface in the degraded list."""
    degraded: list[str] = []
    with patch(
        "repowise.cli.providers.build_embedder",
        side_effect=RuntimeError("boom"),
    ):
        assert _build_update_vector_store("/tmp/repo", {"embedder": "ollama"}, degraded) is None
    assert any("Decision vector store" in d and "boom" in d for d in degraded)


def test_failure_without_degraded_stays_silent() -> None:
    """Callers without a degraded list keep the old silent contract."""
    with patch(
        "repowise.cli.providers.build_embedder",
        side_effect=RuntimeError("boom"),
    ):
        assert _build_update_vector_store("/tmp/repo", {"embedder": "ollama"}) is None


def test_success_returns_the_store() -> None:
    """A healthy build returns the store and records nothing."""
    degraded: list[str] = []
    store = object()
    with (
        patch(
            "repowise.cli.providers.build_embedder",
            return_value=object(),
        ),
        patch(
            "repowise.cli.providers.build_vector_store",
            return_value=store,
        ),
    ):
        assert _build_update_vector_store("/tmp/repo", {"embedder": "ollama"}, degraded) is store
    assert degraded == []


def test_failed_vector_cleanup_retries_without_new_retirements(page_catalog) -> None:
    """A swept SQL row cannot rediscover its vector id on a later update."""
    tmp_path = page_catalog
    page_id = "module_page:removed"
    store = SimpleNamespace(delete_many=AsyncMock(side_effect=OSError("store unavailable")))
    with pytest.raises(OSError, match="store unavailable"):
        cleanup_retired_page_vectors(tmp_path, [page_id], vector_store=store)
    assert load_cleanup_debt(tmp_path)["vectors"] == {page_id}

    store.delete_many = AsyncMock()
    cleanup_retired_page_vectors(tmp_path, [], vector_store=store)

    store.delete_many.assert_awaited_once_with([page_id])
    assert load_cleanup_debt(tmp_path)["vectors"] == set()


def test_vector_cleanup_store_open_failure_keeps_debt_for_retry(page_catalog) -> None:
    """Adapter construction belongs to the host and cannot discard cleanup debt."""
    tmp_path = page_catalog
    page_id = "file_page:removed.py"
    (tmp_path / ".repowise" / "lancedb").mkdir(parents=True)
    store = SimpleNamespace(delete_many=AsyncMock(), close=AsyncMock())
    with (
        patch("repowise.cli.providers.build_embedder", return_value=object()),
        patch(
            "repowise.cli.providers.build_vector_store",
            side_effect=[RuntimeError("lance unavailable"), store],
        ),
    ):
        with pytest.raises(RuntimeError, match="lance unavailable"):
            cleanup_retired_page_vectors(tmp_path, [page_id])
        assert load_cleanup_debt(tmp_path)["vectors"] == {page_id}

        cleanup_retired_page_vectors(tmp_path, [])

    store.delete_many.assert_awaited_once_with([page_id])
    store.close.assert_awaited_once()
    assert load_cleanup_debt(tmp_path)["vectors"] == set()


def test_restored_page_survives_old_vector_deletion_debt(page_catalog) -> None:
    from repowise.core.persistence.crud import upsert_page, upsert_repository
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        resolve_db_url,
    )
    from tests.unit.persistence.helpers import make_page_kwargs

    page_id = "file_page:restored.py"

    async def write(status):
        engine = create_engine(resolve_db_url(page_catalog))
        try:
            async with create_session_factory(engine)() as session:
                repo = await upsert_repository(session, name="test", local_path=str(page_catalog))
                await upsert_page(
                    session,
                    **make_page_kwargs(
                        repo.id,
                        page_id=page_id,
                        target_path="restored.py",
                        freshness_status=status,
                    ),
                )
                await session.commit()
        finally:
            await engine.dispose()

    asyncio.run(write("tombstone"))
    store = SimpleNamespace(delete_many=AsyncMock(side_effect=OSError("offline")))
    with pytest.raises(OSError, match="offline"):
        cleanup_retired_page_vectors(page_catalog, [page_id], vector_store=store)
    record_cleanup_debt(page_catalog, "fts", {page_id})
    asyncio.run(write("fresh"))
    store.delete_many.reset_mock(side_effect=True)

    cleanup_retired_page_vectors(page_catalog, [], vector_store=store)

    store.delete_many.assert_not_awaited()
    assert load_cleanup_debt(page_catalog) == {"fts": set(), "vectors": set()}


@pytest.mark.parametrize("broken_store", [False, True])
def test_cleanup_without_page_authority_preserves_debt(tmp_path, broken_store) -> None:
    from sqlalchemy.exc import SQLAlchemyError

    if broken_store:
        (tmp_path / ".repowise").mkdir()
        (tmp_path / ".repowise/wiki.db").write_bytes(b"not a database")
    store = SimpleNamespace(delete_many=AsyncMock())
    with pytest.raises(SQLAlchemyError if broken_store else RuntimeError):
        cleanup_retired_page_vectors(tmp_path, ["retired"], vector_store=store)
    store.delete_many.assert_not_awaited()
    assert load_cleanup_debt(tmp_path)["vectors"] == {"retired"}
    if not broken_store:
        assert not (tmp_path / ".repowise/wiki.db").exists()


def test_no_debt_opens_no_database_or_vector_store(tmp_path) -> None:
    with (
        patch("repowise.core.persistence.database.create_engine") as engine,
        patch("repowise.cli.commands.update_cmd.incremental._build_update_vector_store") as store,
    ):
        cleanup_retired_page_vectors(tmp_path, [])
    engine.assert_not_called()
    store.assert_not_called()
