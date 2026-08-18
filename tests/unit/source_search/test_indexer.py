"""End to end on a small repo: both lanes, the manifest, and vector reuse."""

from __future__ import annotations

import subprocess

import pytest

from repowise.core.providers.embedding.base import MockEmbedder
from repowise.core.source_search.fts import SourceFTSIndex, default_fts_path
from repowise.core.source_search.indexer import build_source_index
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    default_manifest_path,
    read_manifest,
)
from repowise.core.source_search.vector_store import SourceChunkVectorStore

pytest.importorskip("lancedb")

_IDENTITY = EmbedderIdentity(provider="mock", model="MockEmbedder", dims=8)

_APP = '''import os


def parse_config(path):
    """Read it."""
    return os.path.exists(path)


class Runner:
    def run(self):
        return parse_config("x")
'''

_COMPOSE = "services:\n  db:\n    image: postgres:16\n    ports:\n      - 5432:5432\n"
_SCRIPT = "#!/usr/bin/env bash\nset -euo pipefail\ndocker compose up -d\n"
_README = "# Notes\n\nThis file has no grammar and no listed suffix.\n"


class _CountingEmbedder(MockEmbedder):
    """MockEmbedder that records how many texts it was asked to embed."""

    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        return await super().embed(texts)


class _FlakyEmbedder(MockEmbedder):
    """Fails *failures* times, then succeeds — the transient-endpoint case."""

    def __init__(self, failures: int) -> None:
        self.remaining = failures
        self.attempts = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise RuntimeError("connection refused")
        return await super().embed(texts)


@pytest.fixture
async def repo(tmp_path):
    """A git repo with two indexed symbols and three unindexed text files."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "infra").mkdir()
    (root / "src" / "app.py").write_text(_APP)
    (root / "infra" / "compose.yaml").write_text(_COMPOSE)
    (root / "infra" / "up.sh").write_text(_SCRIPT)
    (root / "README.md").write_text(_README)

    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.com"],
        ["config", "user.name", "T"],
        ["add", "-A"],
        ["-c", "commit.gpgsign=false", "commit", "-qm", "seed"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)

    await _seed_symbols(root)
    return root


async def _seed_symbols(root):
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        init_db,
        resolve_db_url,
    )
    from repowise.core.persistence.models import Repository, WikiSymbol

    engine = create_engine(resolve_db_url(root))
    await init_db(engine)
    factory = create_session_factory(engine)
    async with factory() as session:
        repository = Repository(id="r1", name="repo", local_path=str(root), head_commit="deadbeef")
        session.add(repository)
        session.add_all(
            [
                WikiSymbol(
                    repository_id="r1",
                    file_path="src/app.py",
                    symbol_id="src/app.py::parse_config",
                    name="parse_config",
                    qualified_name="app.parse_config",
                    kind="function",
                    signature="def parse_config(path)",
                    start_line=4,
                    end_line=6,
                    docstring="Read it.",
                    language="python",
                ),
                WikiSymbol(
                    repository_id="r1",
                    file_path="src/app.py",
                    symbol_id="src/app.py::Runner",
                    name="Runner",
                    qualified_name="app.Runner",
                    kind="class",
                    signature="class Runner",
                    start_line=9,
                    end_line=11,
                    language="python",
                ),
            ]
        )
        await session.commit()
    await engine.dispose()


async def _build(root, embedder=None, identity=_IDENTITY):
    return await build_source_index(
        root, embedder=embedder or MockEmbedder(), embedder_identity=identity
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


async def test_both_lanes_are_indexed(repo):
    result = await _build(repo)
    assert result.symbol_chunks == 2
    assert result.file_window_chunks == 2  # compose.yaml + up.sh, one window each
    assert result.files_covered == 3
    assert result.indexed_commit == "deadbeef"


async def test_a_file_with_no_grammar_and_no_listed_suffix_is_left_out(repo):
    await _build(repo)
    with SourceFTSIndex(default_fts_path(repo)) as fts:
        assert not [hit for hit in fts.query("grammar") if hit.file_path == "README.md"]


async def test_the_excluded_text_files_are_reachable_by_search(repo):
    """The point of the window lane: a compose file the wiki never ingested."""
    await _build(repo)
    with SourceFTSIndex(default_fts_path(repo)) as fts:
        hits = fts.query("postgres")
        assert hits
        assert hits[0].file_path == "infra/compose.yaml"


async def test_a_symbol_is_reachable_by_its_camel_spelling(repo):
    await _build(repo)
    with SourceFTSIndex(default_fts_path(repo)) as fts:
        hits = fts.query("parseConfig")
        assert hits[0].chunk_id == "src/app.py::parse_config"


async def test_both_stores_hold_the_same_number_of_chunks(repo):
    result = await _build(repo)
    total = result.symbol_chunks + result.file_window_chunks
    store = SourceChunkVectorStore(str(repo / ".repowise" / "lancedb"), embedder=MockEmbedder())
    assert await store.count() == total
    await store.close()
    with SourceFTSIndex(default_fts_path(repo)) as fts:
        assert fts.count() == total


# ---------------------------------------------------------------------------
# Manifest and rebuilds
# ---------------------------------------------------------------------------


async def test_the_manifest_records_the_build(repo):
    result = await _build(repo)
    manifest = read_manifest(default_manifest_path(repo))
    assert manifest is not None
    assert manifest.corpus_hash == result.corpus_hash
    assert manifest.recipe_fingerprint == result.recipe_fingerprint
    assert manifest.symbol_chunks == 2
    assert manifest.file_window_chunks == 2
    assert manifest.embedder == _IDENTITY


async def test_an_unchanged_corpus_rebuilds_to_the_same_hash(repo):
    first = await _build(repo)
    second = await _build(repo)
    assert second.corpus_hash == first.corpus_hash
    assert second.recipe_fingerprint == first.recipe_fingerprint


async def test_a_rebuild_reuses_every_unchanged_vector(repo):
    await _build(repo)
    counter = _CountingEmbedder()
    second = await _build(repo, embedder=counter)
    assert second.reused == second.symbol_chunks + second.file_window_chunks
    assert second.embedded == 0
    assert counter.texts == 0


async def test_edited_source_is_re_embedded_and_the_rest_is_not(repo):
    await _build(repo)
    (repo / "infra" / "up.sh").write_text(_SCRIPT + "echo done\n")
    counter = _CountingEmbedder()
    second = await _build(repo, embedder=counter)
    assert second.embedded == 1
    assert second.reused == 3
    assert counter.texts == 1


async def test_same_symbol_id_with_changed_body_is_re_embedded(repo):
    """Reuse follows text, not the stable symbol id."""

    await _build(repo)
    app = repo / "src" / "app.py"
    app.write_text(_APP.replace("return os.path.exists(path)", "return not os.path.exists(path)"))
    counter = _CountingEmbedder()
    second = await _build(repo, embedder=counter)
    assert second.embedded == 1
    assert second.reused == 3
    assert counter.texts == 1


async def test_a_changed_recipe_forfeits_reuse(repo):
    """Vectors from another embedder must not be carried into this index."""
    await _build(repo)
    counter = _CountingEmbedder()
    second = await _build(
        repo,
        embedder=counter,
        identity=EmbedderIdentity(provider="mock", model="OtherModel", dims=8),
    )
    assert second.reused == 0
    assert second.embedded == 4


async def test_a_rebuild_does_not_accumulate_rows(repo):
    """Full rebuild means replace, not append."""
    first = await _build(repo)
    await _build(repo)
    store = SourceChunkVectorStore(str(repo / ".repowise" / "lancedb"), embedder=MockEmbedder())
    assert await store.count() == first.symbol_chunks + first.file_window_chunks
    await store.close()


async def test_a_deleted_file_leaves_the_index(repo):
    await _build(repo)
    (repo / "infra" / "compose.yaml").unlink()
    subprocess.run(
        ["git", "-C", str(repo), "rm", "-q", "--cached", "infra/compose.yaml"],
        check=True,
        capture_output=True,
    )
    second = await _build(repo)
    assert second.file_window_chunks == 1
    with SourceFTSIndex(default_fts_path(repo)) as fts:
        assert fts.query("postgres") == []


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


async def test_a_transient_embedder_failure_is_retried(repo):
    flaky = _FlakyEmbedder(failures=2)
    result = await _build(repo, embedder=flaky)
    assert result.embedded == 4
    assert flaky.attempts == 3


async def test_a_persistent_embedder_failure_leaves_no_manifest(repo):
    """No manifest means the next run rebuilds, rather than trusting a partial index."""
    await _build(repo)
    assert default_manifest_path(repo).exists()
    with pytest.raises(RuntimeError, match="Embedding failed after 4 attempts"):
        await _build(
            repo,
            embedder=_FlakyEmbedder(failures=99),
            identity=EmbedderIdentity(provider="mock", model="Other", dims=8),
        )
    assert not default_manifest_path(repo).exists()


async def test_an_unindexed_repo_says_so(tmp_path):
    root = tmp_path / "bare"
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True)
    with pytest.raises(RuntimeError, match="No indexed repository"):
        await _build(root)
