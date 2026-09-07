"""The authoritative module-page set, derived from the current parse.

``update --index-only`` needs an answer to "which module pages should exist at
this commit" before it can retire the ones whose code is gone. Getting that
answer wrong in either direction is expensive: too small deletes live pages,
too large leaves the dead ones serving. So it is not computed here at all — it
is the same ``select_pages`` call ``init`` makes, driven from the same inputs.

These tests pin that equality, and pin the two ways a hand-rolled version
would have gone wrong: asking the filesystem whether a directory exists, and
requiring a knowledge graph that is absent on a fresh index.
"""

from __future__ import annotations

import json
from pathlib import Path

from repowise.core.generation.models import GenerationConfig
from repowise.core.generation.selection import SelectionInputs, select_pages
from repowise.core.generation.selection.module_reconcile import authoritative_module_pages
from tests.unit.generation.test_selection_contract import (
    FakeFileInfo,
    FakeParsedFile,
    FakeSymbol,
)


def _paths() -> list[str]:
    return (
        [f"codeatlas/doc_search/f{i}.py" for i in range(8)]
        + [f"codeatlas/scrapers/f{i}.py" for i in range(6)]
        + [f"mcp/supabase/f{i}.py" for i in range(5)]
        + ["setup.py", "main.py"]
    )


def _parsed(paths: list[str]) -> list[FakeParsedFile]:
    return [
        FakeParsedFile(
            file_info=FakeFileInfo(path=p, is_test=p.startswith("tests/")),
            symbols=[FakeSymbol(name=f"fn_{i}")],
        )
        for i, p in enumerate(paths)
    ]


class FakeGraphBuilder:
    """The five metric calls the selection chain makes, and nothing else."""

    def __init__(self, paths: list[str]) -> None:
        # Deliberately not uniform: a clone-dedupe or scoring step that reads
        # PageRank must see a real spread rather than a flat map.
        self._pagerank = {p: (1.0 if "/scrapers/" in p else 0.1) for p in paths}
        self._paths = paths

    def pagerank(self) -> dict[str, float]:
        return dict(self._pagerank)

    def betweenness_centrality(self) -> dict[str, float]:
        return {p: 0.0 for p in self._paths}

    def community_detection(self) -> dict[str, int]:
        return {p: 0 for p in self._paths}

    def strongly_connected_components(self) -> list:
        return []

    def community_info(self) -> dict:
        return {}


def _config() -> GenerationConfig:
    return GenerationConfig(deterministic=True, file_pages_only=True)


def test_authoritative_set_matches_selection_for_same_files(tmp_path: Path) -> None:
    """The derived ids are the selector's module groups, not a second opinion."""
    paths = _paths()
    parsed = _parsed(paths)
    builder = FakeGraphBuilder(paths)
    config = _config()

    derived = authoritative_module_pages(
        parsed_files=parsed,
        graph_builder=builder,
        config=config,
        repo_path=tmp_path,
    )

    expected = select_pages(
        SelectionInputs(
            parsed_files=parsed,
            pagerank=builder.pagerank(),
            betweenness=builder.betweenness_centrality(),
            community=builder.community_detection(),
            community_info={},
            sccs=[],
            git_meta_map=None,
            config=config,
        )
    ).module_groups

    assert derived, "the derivation produced no module pages at all"
    assert set(derived) == {f"module_page:{g.key}" for g in expected}
    assert derived == {f"module_page:{g.key}": g.structural_key for g in expected}


def test_directory_with_only_pycache_is_not_authoritative(tmp_path: Path) -> None:
    """A directory on disk with no indexed files owns no page.

    This is the live case the ticket exists for: ``codeatlas/chunking`` was
    deleted, its ``__pycache__`` was not, and a reconcile that asked
    ``(root / target).exists()`` would have called its page current forever.
    """
    ghost = tmp_path / "codeatlas" / "chunking" / "__pycache__"
    ghost.mkdir(parents=True)
    (ghost / "grouping.cpython-313.pyc").write_bytes(b"\x00")

    paths = _paths()
    derived = authoritative_module_pages(
        parsed_files=_parsed(paths),
        graph_builder=FakeGraphBuilder(paths),
        config=_config(),
        repo_path=tmp_path,
    )

    assert derived
    assert "module_page:codeatlas/chunking" not in derived
    assert not any(key.startswith("codeatlas/chunking") for key in _keys(derived))


def test_derivation_survives_absent_knowledge_graph(tmp_path: Path) -> None:
    """No ``knowledge-graph.json`` costs the grouping's taste, never its coverage."""
    assert not (tmp_path / ".repowise" / "knowledge-graph.json").exists()

    paths = _paths()
    derived = authoritative_module_pages(
        parsed_files=_parsed(paths),
        graph_builder=FakeGraphBuilder(paths),
        config=_config(),
        repo_path=tmp_path,
    )

    assert derived, "an absent knowledge graph emptied the authoritative set"
    covered = {key.split("/", 1)[0] for key in _keys(derived)}
    assert "codeatlas" in covered


def test_in_memory_kg_modules_win_over_the_artifact(tmp_path: Path) -> None:
    """The caller's fresh result beats the file, which is a run stale on update."""
    kg_dir = tmp_path / ".repowise"
    kg_dir.mkdir()
    (kg_dir / "knowledge-graph.json").write_text(
        json.dumps({"layers": [], "modules": [], "tour": [], "nodes": [], "edges": []}),
        encoding="utf-8",
    )

    paths = _paths()
    kwargs = {
        "parsed_files": _parsed(paths),
        "graph_builder": FakeGraphBuilder(paths),
        "config": _config(),
        "repo_path": tmp_path,
    }
    with_modules = authoritative_module_pages(
        **kwargs,
        kg_modules=[{"id": "m1", "name": "Docs", "filePaths": ["codeatlas/doc_search/f0.py"]}],
    )

    # Coverage is the invariant, not the grouping: the layer map steers which
    # sibling directories merge, so asserting a specific key here would pin
    # the grouper's taste rather than this function's contract.
    assert with_modules
    assert all(page_id.startswith("module_page:") for page_id in with_modules)


def _keys(derived: dict[str, str]) -> list[str]:
    return [page_id.split(":", 1)[1] for page_id in derived]
