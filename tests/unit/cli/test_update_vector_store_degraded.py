"""Tests for the update path's decision vector store (issue #1370).

``_build_update_vector_store`` swallows failures and returns None so the
decision upsert still works without a store — but the failure must land in
the run's ``degraded`` list, or the panel says nothing while semantic dedup
is silently off.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from repowise.cli.commands.update_cmd.incremental import (
    _build_update_vector_store,
    cleanup_retired_page_vectors,
)
from repowise.core.pipeline.cleanup_debt import load_cleanup_debt


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
    with patch(
        "repowise.cli.providers.build_embedder",
        return_value=object(),
    ), patch(
        "repowise.cli.providers.build_vector_store",
        return_value=store,
    ):
        assert _build_update_vector_store("/tmp/repo", {"embedder": "ollama"}, degraded) is store
    assert degraded == []


def test_failed_vector_cleanup_retries_without_new_retirements(tmp_path) -> None:
    """A swept SQL row cannot rediscover its vector id on a later update."""
    page_id = "module_page:removed"
    store = SimpleNamespace(delete_many=AsyncMock(side_effect=OSError("store unavailable")))
    with pytest.raises(OSError, match="store unavailable"):
        cleanup_retired_page_vectors(tmp_path, [page_id], vector_store=store)
    assert load_cleanup_debt(tmp_path)["vectors"] == {page_id}

    store.delete_many = AsyncMock()
    cleanup_retired_page_vectors(tmp_path, [], vector_store=store)

    store.delete_many.assert_awaited_once_with([page_id])
    assert load_cleanup_debt(tmp_path)["vectors"] == set()
