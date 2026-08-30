"""Both ``persist_result`` routes share one complete post-index hook seam."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from repowise.cli.commands.init_cmd.persistence import persist_result


class _Graph:
    def number_of_nodes(self) -> int:
        return 0


class _GraphBuilder:
    def graph(self) -> _Graph:
        return _Graph()


def _result(index_done: bool) -> SimpleNamespace:
    parsed = SimpleNamespace(file_info=SimpleNamespace(path="src/app.py"))
    return SimpleNamespace(
        repo_name="repo",
        index_persisted_incrementally=index_done,
        parsed_files=[parsed],
        generated_pages=[],
        tech_stack=None,
        vector_store=None,
        dead_code_report=None,
        health_report=None,
        decision_report=None,
        git_metadata_list=[],
        knowledge_graph_result=None,
        authoritative_page_types=set(),
        preserved_page_ids=set(),
        graph_builder=_GraphBuilder(),
    )


@pytest.mark.parametrize("index_done", [False, True])
async def test_both_routes_fire_every_post_index_hook(tmp_path, monkeypatch, index_done):
    """A hook added to the shared seam is reached from either branch."""
    from repowise.cli import source_search_runtime
    from repowise.core import pipeline, source_search
    from repowise.core.pipeline import page_tree_sync
    from repowise.core.pipeline import persist as persist_module
    from repowise.core.source_search import outbox

    calls: list[str] = []

    async def _record(name: str, value=None):
        calls.append(name)
        return value

    monkeypatch.setattr(
        persist_module, "_prune_stale_file_rows", lambda *_a, **_k: _record("index_prune")
    )
    monkeypatch.setattr(
        persist_module, "persist_ingestion", lambda *_a, **_k: _record("ingestion", 0)
    )
    monkeypatch.setattr(persist_module, "persist_git", lambda *_a, **_k: _record("git"))
    monkeypatch.setattr(persist_module, "persist_analysis", lambda *_a, **_k: _record("analysis"))
    monkeypatch.setattr(
        persist_module, "persist_generation", lambda *_a, **_k: _record("generation")
    )
    monkeypatch.setattr(
        persist_module, "persist_reference_sites", lambda *_a, **_k: _record("refsites", 1)
    )
    monkeypatch.setattr(
        persist_module,
        "_sweep_stale_generated_pages",
        lambda *_a, **_k: _record("structural_sweep", ["structural"]),
    )
    monkeypatch.setattr(
        persist_module,
        "sweep_retired_pages",
        lambda *_a, **_k: _record("retired_sweep", ["retired"]),
    )
    monkeypatch.setattr(page_tree_sync, "rebuild_page_tree", lambda *_a, **_k: _record("page_tree"))
    monkeypatch.setattr(source_search, "source_search_enabled", lambda: True)
    monkeypatch.setattr(outbox, "enqueue_full_update", lambda *_a, **_k: _record("source_enqueue"))
    monkeypatch.setattr(
        pipeline,
        "tombstone_absent_file_pages",
        lambda *_a, **_k: _record("absent_tombstones", []),
    )
    monkeypatch.setattr(
        source_search_runtime,
        "reconcile_configured_source_index",
        lambda *_a, **_k: _record("source_reconcile"),
    )

    repo_path = tmp_path / ("incremental" if index_done else "full")
    repo_path.mkdir()
    await persist_result(_result(index_done), repo_path)

    post_index = [
        "analysis",
        "retired_sweep",
        "generation",
        "refsites",
        "structural_sweep",
        "page_tree",
        "source_enqueue",
    ]
    assert [name for name in calls if name in post_index] == post_index
    assert "absent_tombstones" in calls
    assert "source_reconcile" in calls
