"""``get_index_status`` against a checkout that moved after the build.

The tool could already see a new commit, because ``HEAD`` moves in the
repository rather than in anything the build wrote down. It could not see an
edit or a deletion, because the only working-tree fact it had was the list the
build itself recorded — which cannot grow afterwards. These tests drive the
tool over a real repository, mutate it, and ask.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from repowise.core.persistence.models import Repository
from repowise.core.source_search.chunks import SymbolRecord, build_symbol_chunk
from repowise.core.source_search.fts import SourceFTSIndex
from repowise.core.source_search.generation import GenerationRef
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    default_manifest_path,
    write_manifest,
)
from repowise.core.source_search.status import SourceIndexStatus

_GENERATION = GenerationRef("generation-1", 1)
_FTS_REL = ".repowise/source_search/source_fts_v2.db"
_EMBEDDER = EmbedderIdentity(provider="mock", model="mock-embedder", dims=8)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=Dev", "-c", "user.email=dev@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _chunk(path: str):
    name = Path(path).stem
    return build_symbol_chunk(
        SymbolRecord(
            symbol_id=f"{path}::{name}",
            file_path=path,
            name=name,
            qualified_name=name,
            kind="function",
            signature="",
            docstring=None,
            start_line=1,
            end_line=2,
            language="python",
        ),
        [f"def {name}():", "    return 1"],
    )


def _build_repo(root: Path, paths: list[str]) -> tuple[Path, str]:
    """A committed checkout with a published corpus over exactly *paths*.

    Returns the root and the commit the corpus was built at, so a test that
    moves ``HEAD`` afterwards can keep the manifest pointing where it was.
    """

    (root / "src").mkdir(parents=True)
    for path in paths:
        (root / path).write_text(f"def {Path(path).stem}():\n    return 1\n", encoding="utf-8")
    (root / "notes.md").write_text("# not indexed\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "seed")
    indexed_commit = _git(root, "rev-parse", "HEAD")
    with SourceFTSIndex(root / _FTS_REL, generation=_GENERATION) as fts:
        fts.index_chunks([_chunk(path) for path in paths])
    write_manifest(
        default_manifest_path(root),
        SourceIndexManifest(
            recipe_fingerprint="recipe-1",
            corpus_hash="corpus-1",
            symbol_chunks=len(paths),
            file_window_chunks=0,
            files_covered=len(paths),
            indexed_commit=indexed_commit,
            built_at="2026-08-21T16:00:00+00:00",
            embedder=_EMBEDDER,
            generation_id=_GENERATION.generation_id,
            generation_sequence=_GENERATION.sequence,
            fts_path=_FTS_REL,
        ),
    )
    return root, indexed_commit


def _ingest_dirty(root: Path, path: str, body: str) -> str:
    """Write uncommitted *body* to *path* and return the hash a build would record."""

    from repowise.core.ingestion.models import compute_content_hash

    (root / path).write_text(body, encoding="utf-8")
    return compute_content_hash((root / path).read_bytes())


def _write_state(root: Path, state: dict) -> None:
    """Write ``.repowise/state.json``, the record only the update path keeps."""

    (root / ".repowise").mkdir(parents=True, exist_ok=True)
    (root / ".repowise" / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _healthy(indexed_commit: str, chunks: int) -> SourceIndexStatus:
    """Every freshness input except the working tree, pinned to healthy.

    A real ``inspect_source_index`` over a temporary repository cannot reach
    ``trustworthy``: there is no dense store to count and no update row to read
    a parser fingerprint off, and both of those honestly degrade the verdict.
    Holding them constant is what makes the assertions below statements about
    the working tree rather than about the fixture.
    """

    return SourceIndexStatus(
        state="current",
        generation_id=_GENERATION.generation_id,
        generation_sequence=_GENERATION.sequence,
        indexed_commit=indexed_commit,
        recipe_fingerprint="recipe-1",
        pending_updates=0,
        blocked_updates=0,
        building_updates=0,
        ready_updates=0,
        manifest_state="ok",
        built_at="2026-08-21T16:00:00+00:00",
        published_at="2026-08-21T16:01:00+00:00",
        embedder=_EMBEDDER,
        parser_fingerprint="parser-1",
        symbol_chunks=chunks,
        file_window_chunks=0,
        files_covered=chunks,
        expected_chunks=chunks,
        fts_chunks=chunks,
        vector_chunks=chunks,
        fts_path=_FTS_REL,
    )


@pytest.fixture
async def status_tool(tmp_path, monkeypatch, session, factory, repo_id, vector_store, fts):
    """``get_index_status`` bound to a real two-file repository.

    The live working-tree read is the real one; everything else the verdict
    depends on is pinned by :func:`_healthy`.
    """

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path, indexed_commit = _build_repo(tmp_path / "repo", ["src/alpha.py", "src/beta.py"])
    repository = await session.get(Repository, repo_id)
    repository.local_path = str(repo_path)
    repository.name = repo_path.name
    await session.commit()
    context = SimpleNamespace(
        alias="default",
        path=repo_path,
        session_factory=factory,
        vector_store=vector_store,
        fts=fts,
    )

    async def resolve(_repo: str | None) -> SimpleNamespace:
        return context

    real_inspect = module.inspect_source_index

    async def inspect(path, **kwargs):
        live = await real_inspect(path, **kwargs)
        return replace(_healthy(indexed_commit, 2), working_tree=live.working_tree)

    monkeypatch.setattr(module, "_resolve_repo_context", resolve)
    monkeypatch.setattr(module, "inspect_source_index", inspect)
    monkeypatch.setattr(module, "_runtime_identities", lambda _ctx: (_EMBEDDER, "parser-1", None))
    return module, repo_path


async def test_a_clean_checkout_is_trustworthy_and_says_it_looked(status_tool):
    module, _repo = status_tool

    result = await module.get_index_status()

    assert result["trust"] == {"search_results": "trustworthy", "reasons": []}
    working_tree = result["generation"]["working_tree"]
    assert working_tree["checked"] is True
    assert working_tree["divergent_indexed_path_count"] == 0
    assert working_tree["modified"] == []
    assert working_tree["deleted"] == []


async def test_a_deleted_indexed_file_is_stale_and_named(status_tool):
    """The silent failure this whole surface exists to prevent.

    Chunks for a file the working tree no longer has kept serving at full
    confidence with nothing anywhere saying so, because the only working-tree
    list the payload carried was written by the build.
    """

    module, repo = status_tool
    _git(repo, "rm", "-q", "src/alpha.py")

    result = await module.get_index_status()

    assert result["trust"]["search_results"] == "stale"
    assert "indexed_files_deleted" in result["trust"]["reasons"]
    working_tree = result["generation"]["working_tree"]
    assert working_tree["deleted"] == ["src/alpha.py"]
    assert working_tree["deleted_count"] == 1
    assert working_tree["divergent_indexed_path_count"] == 1
    # The build's own record is untouched and still says nothing, which is why
    # it could never have carried this.
    assert result["generation"]["uncommitted_indexed_path_count"] == 0


async def test_an_uncommitted_edit_to_an_indexed_file_is_stale_and_named(status_tool):
    module, repo = status_tool
    (repo / "src/beta.py").write_text("def beta():\n    return 99\n", encoding="utf-8")

    result = await module.get_index_status()

    assert result["trust"]["search_results"] == "stale"
    assert "indexed_files_modified" in result["trust"]["reasons"]
    working_tree = result["generation"]["working_tree"]
    assert working_tree["modified"] == ["src/beta.py"]
    assert working_tree["modified_count"] == 1
    assert working_tree["deleted"] == []
    assert result["generation"]["uncommitted_indexed_path_count"] == 0


async def test_an_edit_outside_the_corpus_leaves_the_verdict_alone(status_tool):
    """Blanket caution is not disclosure.

    If every dirty checkout came back stale the reason would carry no
    information, and the acceptance probe would pass for the wrong reason.
    """

    module, repo = status_tool
    (repo / "notes.md").write_text("# edited\n", encoding="utf-8")
    (repo / "src/gamma.py").write_text("def gamma():\n    return 1\n", encoding="utf-8")

    result = await module.get_index_status()

    assert result["trust"] == {"search_results": "trustworthy", "reasons": []}
    assert result["generation"]["working_tree"]["divergent_indexed_path_count"] == 0


async def test_a_new_commit_still_reports_only_index_behind_head(status_tool):
    """The signal that already worked has to keep working, and keep working alone.

    Committing the change is what makes ``HEAD`` the thing that moved; the
    working tree is clean again, and saying otherwise would double-report one
    divergence under two names.
    """

    module, repo = status_tool
    (repo / "src/alpha.py").write_text("def alpha():\n    return 2\n", encoding="utf-8")
    _git(repo, "add", "src/alpha.py")
    _git(repo, "commit", "-qm", "advance")

    result = await module.get_index_status()

    assert result["trust"] == {
        "search_results": "stale",
        "reasons": ["index_behind_head"],
    }
    assert result["generation"]["commit_matches"] is False
    assert result["generation"]["working_tree"]["divergent_indexed_path_count"] == 0
    assert result["generation"]["working_tree"]["checked"] is True


async def test_a_branch_sized_divergence_is_capped_with_the_exact_total(
    tmp_path, monkeypatch, session, factory, repo_id, vector_store, fts
):
    """Switching branches can retire thousands of indexed paths at once.

    The list is bounded like every other array in this payload, and the cap is
    disclosed rather than silent — a truncated list beside a truncated count
    would understate exactly the condition this block exists to report.
    """

    from repowise.core.distill.store import OmissionStore

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    paths = [f"src/mod_{index:04d}.py" for index in range(120)]
    repo_path, indexed_commit = _build_repo(tmp_path / "repo", paths)
    repository = await session.get(Repository, repo_id)
    repository.local_path = str(repo_path)
    await session.commit()
    context = SimpleNamespace(
        alias="default",
        path=repo_path,
        session_factory=factory,
        vector_store=vector_store,
        fts=fts,
    )
    real_inspect = module.inspect_source_index

    async def inspect(path, **kwargs):
        live = await real_inspect(path, **kwargs)
        return replace(_healthy(indexed_commit, 120), working_tree=live.working_tree)

    async def resolve(_repo: str | None) -> SimpleNamespace:
        return context

    monkeypatch.setattr(module, "_resolve_repo_context", resolve)
    monkeypatch.setattr(module, "inspect_source_index", inspect)
    monkeypatch.setattr(module, "_runtime_identities", lambda _ctx: (_EMBEDDER, "parser-1", None))
    _git(repo_path, "rm", "-q", *paths)

    result = await module.get_index_status()

    working_tree = result["generation"]["working_tree"]
    assert working_tree["deleted_count"] == 120
    assert working_tree["deleted_listed"] == 50
    assert len(working_tree["deleted"]) == 50
    omitted = result["_meta"]["omitted"]
    with OmissionStore.open_default(repo_path) as store:
        restored = "\n".join(store.get(ref) or "" for ref in omitted["refs"])
    assert "src/mod_0119.py" in restored
    assert "src/mod_0000.py" not in restored



# --- The build's own record of what it read out of the working tree ---------
#
# ``uncommitted_indexed_paths`` answers a different question from the
# ``working_tree`` block above it. That block is live: what has the checkout
# done to the corpus *since* the build. This one is historical: which paths did
# the build read uncommitted bytes for at all. A reader uses it to decide
# whether indexed answers describe committed code or someone's unsaved work,
# so it has to name every such path — and a full source-index rebuild records
# them in the manifest, never in ``state.json``.


def _rebuilt_dirty(root: Path, paths: list[str], dirty: list[str]) -> tuple[str, dict[str, str]]:
    """A corpus over *paths* whose build read uncommitted bytes for *dirty*.

    Exactly what ``repowise source-index`` leaves behind: the manifest carries
    the ingest record for every path it read dirty, and nothing writes
    ``state.json`` — that is the update path's file, and no update ran.
    """

    _build_repo(root, paths)
    record = {
        path: _ingest_dirty(root, path, f"def {Path(path).stem}():\n    return 999\n")
        for path in dirty
    }
    manifest_path = default_manifest_path(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["working_tree_ingest"] = record
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(manifest["indexed_commit"]), record


async def _bind(module, monkeypatch, session, factory, repo_id, vector_store, fts, repo_path,
                indexed_commit, chunks):
    """Bind ``get_index_status`` to *repo_path* with only the working tree live."""

    repository = await session.get(Repository, repo_id)
    repository.local_path = str(repo_path)
    repository.name = repo_path.name
    await session.commit()
    context = SimpleNamespace(
        alias="default",
        path=repo_path,
        session_factory=factory,
        vector_store=vector_store,
        fts=fts,
    )
    real_inspect = module.inspect_source_index

    async def inspect(path, **kwargs):
        live = await real_inspect(path, **kwargs)
        return replace(
            _healthy(indexed_commit, chunks),
            working_tree=live.working_tree,
            working_tree_ingest=live.working_tree_ingest,
        )

    async def resolve(_repo: str | None) -> SimpleNamespace:
        return context

    monkeypatch.setattr(module, "_resolve_repo_context", resolve)
    monkeypatch.setattr(module, "inspect_source_index", inspect)
    monkeypatch.setattr(module, "_runtime_identities", lambda _ctx: (_EMBEDDER, "parser-1", None))


@pytest.fixture
async def rebuilt_tool(tmp_path, monkeypatch, session, factory, repo_id, vector_store, fts):
    """``get_index_status`` over a repo rebuilt whole from one dirty file."""

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    indexed_commit, _record = _rebuilt_dirty(
        repo_path, ["src/alpha.py", "src/beta.py"], ["src/beta.py"]
    )
    await _bind(module, monkeypatch, session, factory, repo_id, vector_store, fts,
                repo_path, indexed_commit, 2)
    return module, repo_path


async def test_a_full_rebuild_discloses_the_dirty_bytes_it_ingested(rebuilt_tool):
    """The freshness surface must not deny content it is actively serving.

    A full rebuild writes its working-tree record to the manifest and leaves
    ``state.json`` alone. A payload sourced from ``state.json`` therefore
    reports an empty list on exactly the build that read the most uncommitted
    work — and reports it as a fact, not as an absence of evidence.
    """

    module, _repo = rebuilt_tool

    result = await module.get_index_status()

    generation = result["generation"]
    assert generation["uncommitted_indexed_paths"] == ["src/beta.py"]
    assert generation["uncommitted_indexed_path_count"] == 1
    assert generation["uncommitted_indexed_paths_listed"] == 1


async def test_the_disclosed_basis_names_the_manifest_it_read(rebuilt_tool):
    """A basis line that names the wrong store is its own defect.

    The payload publishes where its answer came from so a reader can judge it.
    Naming ``state.json`` while serving the manifest's record leaves the field
    unfalsifiable by the person it exists to inform.
    """

    module, _repo = rebuilt_tool

    result = await module.get_index_status()

    basis = result["generation"]["uncommitted_indexed_paths_basis"]
    assert "working_tree_ingest" in basis
    assert "manifest" in basis


async def test_path_mode_admits_the_file_was_indexed_from_the_working_tree(rebuilt_tool):
    """``working_tree_indexed`` is asked about one path and must not answer false.

    Path mode is where a reader takes the fine-grained answer, and it read the
    same empty ``state.json`` list the status block did.
    """

    module, _repo = rebuilt_tool

    result = await module.get_index_status(mode="path", path="src/beta.py")

    assert result["path"]["working_tree_indexed"] is True
    assert result["path"]["indexed"] is True


async def test_ingested_bytes_still_on_disk_are_not_also_called_stale(rebuilt_tool):
    """Disclosure and staleness are separate claims about the same file.

    The corpus serves exactly the bytes on disk, so the live block is right to
    stay quiet. Reporting the path as modified to compensate for the missing
    disclosure would trade one wrong answer for another.
    """

    module, _repo = rebuilt_tool

    result = await module.get_index_status()

    working_tree = result["generation"]["working_tree"]
    assert working_tree["checked"] is True
    assert working_tree["modified"] == []
    assert "indexed_files_modified" not in result["trust"]["reasons"]


async def test_a_dirty_checkout_rebuilt_whole_discloses_every_ingested_file(
    tmp_path, monkeypatch, session, factory, repo_id, vector_store, fts
):
    """The B4 scenario: several uncommitted edits, then one full rebuild.

    ``state.json`` is present and syntactically fine — it simply carries no
    ``working_tree_paths`` key, because the build that ingested this work was a
    source-index rebuild rather than an update. Every ingested path has to be
    named, not only the one that sorts first.
    """

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    paths = ["src/alpha.py", "src/beta.py", "src/gamma.py"]
    indexed_commit, _record = _rebuilt_dirty(repo_path, paths, paths)
    _write_state(repo_path, {"last_sync_commit": indexed_commit})
    await _bind(module, monkeypatch, session, factory, repo_id, vector_store, fts,
                repo_path, indexed_commit, 3)

    result = await module.get_index_status()

    generation = result["generation"]
    assert generation["uncommitted_indexed_paths"] == paths
    assert generation["uncommitted_indexed_path_count"] == 3
    assert generation["working_tree"]["modified"] == []


async def test_an_untracked_path_recorded_only_in_state_json_survives(
    tmp_path, monkeypatch, session, factory, repo_id, vector_store, fts
):
    """The manifest cannot hold every uncommitted path, so it cannot be the only source.

    ``working_tree_ingest`` is built from ``git diff HEAD``, which never lists
    an untracked file; the update path records those in ``state.json``. Serving
    the manifest *instead of* that list would fix the rebuild case by dropping
    every untracked file the watcher indexed — the same defect pointed the
    other way.
    """

    module = importlib.import_module("repowise.server.mcp_server.tool_index_status")
    repo_path = tmp_path / "repo"
    indexed_commit, _record = _rebuilt_dirty(
        repo_path, ["src/alpha.py", "src/beta.py"], ["src/beta.py"]
    )
    (repo_path / "src/scratch.py").write_text("def scratch():\n    return 1\n", encoding="utf-8")
    _write_state(repo_path, {"working_tree_paths": ["src/scratch.py"]})
    await _bind(module, monkeypatch, session, factory, repo_id, vector_store, fts,
                repo_path, indexed_commit, 2)

    result = await module.get_index_status()

    generation = result["generation"]
    assert generation["uncommitted_indexed_paths"] == ["src/beta.py", "src/scratch.py"]
    assert generation["uncommitted_indexed_path_count"] == 2
    path_result = await module.get_index_status(mode="path", path="src/scratch.py")
    assert path_result["path"]["working_tree_indexed"] is True
