"""Runtime/source-manifest identity drift used by update skip decisions."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from repowise.cli.source_search_runtime import configured_source_recipe_changes
from repowise.core.source_search.manifest import EmbedderIdentity


class _RuntimeEmbedder:
    dimensions = 768
    _model = "embeddinggemma"


def test_configured_recipe_changes_match_status_identities(tmp_path: Path, monkeypatch) -> None:
    """Both parser and embedder differences are named from the status facts."""
    manifest = tmp_path / ".repowise" / "source_index.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    status = SimpleNamespace(
        manifest_state="ok",
        parser_fingerprint="parser-v1",
        embedder=EmbedderIdentity(provider="ollama", model="old-model", dims=384),
    )

    import repowise.core.ingestion.parse_cache as parse_cache
    import repowise.core.source_search as source_search
    import repowise.core.source_search.status as source_status
    from repowise.cli.providers import embedders

    monkeypatch.setattr(source_search, "source_search_enabled", lambda: True)
    monkeypatch.setattr(
        source_status,
        "inspect_source_index",
        AsyncMock(return_value=status),
    )
    monkeypatch.setattr(parse_cache, "parser_fingerprint", lambda: "parser-v2")
    monkeypatch.setattr(embedders, "resolve_embedder_for_repo", lambda _repo: "ollama")
    monkeypatch.setattr(embedders, "build_embedder", lambda _name, _repo: _RuntimeEmbedder())

    changes = asyncio.run(configured_source_recipe_changes(tmp_path))

    assert changes == (
        "parser fingerprint parser-v1 → parser-v2",
        "embedder identity ollama/old-model/384d → ollama/embeddinggemma/768d",
    )
