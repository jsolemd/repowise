"""Host recovery is bounded and only recaptures a known saved-file race."""

from unittest.mock import AsyncMock

import pytest

from repowise.cli.source_search_runtime import reconcile_configured_source_index
from repowise.core.providers.embedding.base import MockEmbedder
from repowise.core.source_search import fast_update, lifecycle


@pytest.mark.parametrize(
    ("error", "expected_calls"),
    [
        (lifecycle.SourceFileChangedError("src/app.py"), 2),
        (lifecycle.SourceIndexDeferredError("unreadable input"), 1),
        (RuntimeError("embedding backend unavailable"), 1),
    ],
)
async def test_only_file_changes_receive_one_recapture(
    tmp_path, monkeypatch, error, expected_calls
):
    monkeypatch.setenv("REPOWISE_SOURCE_SEARCH", "1")
    reconcile = AsyncMock(side_effect=error)
    capture = AsyncMock()
    monkeypatch.setattr(lifecycle, "reconcile_source_index", reconcile)
    monkeypatch.setattr(fast_update, "capture_source_changes", capture)
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'external.db'}"

    with pytest.raises(type(error), match=str(error)):
        await reconcile_configured_source_index(
            tmp_path,
            embedder=MockEmbedder(),
            embedder_name="mock",
            allow_keyless=True,
            db_url=db_url,
        )

    assert reconcile.await_count == expected_calls
    assert all(call.kwargs["db_url"] == db_url for call in reconcile.await_args_list)
    if expected_calls == 2:
        capture.assert_awaited_once_with(tmp_path, {"src/app.py"}, db_url=db_url)
    else:
        capture.assert_not_awaited()
