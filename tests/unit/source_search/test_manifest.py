"""The manifest: what stays the same across a rebuild, and what must not."""

from __future__ import annotations

import json

import pytest

from repowise.core.source_search import chunks as chunks_mod
from repowise.core.source_search import fts as fts_mod
from repowise.core.source_search.chunks import SymbolRecord, build_symbol_chunk, iter_file_windows
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    corpus_hash,
    default_manifest_path,
    inspect_manifest,
    read_manifest,
    recipe_fingerprint,
    write_manifest,
)

_OLLAMA = EmbedderIdentity(provider="ollama", model="embeddinggemma", dims=768)


def _corpus():
    symbol = build_symbol_chunk(
        SymbolRecord(
            symbol_id="src/app.py::run",
            file_path="src/app.py",
            name="run",
            qualified_name="app.run",
            kind="function",
            signature="def run()",
            docstring=None,
            start_line=1,
            end_line=2,
            language="python",
        ),
        ["def run():", "    return 1"],
    )
    windows = list(iter_file_windows("infra/up.sh", "set -e\necho hi\n"))
    return [symbol, *windows]


# ---------------------------------------------------------------------------
# Corpus hash
# ---------------------------------------------------------------------------


def test_the_same_corpus_hashes_the_same():
    assert corpus_hash(_corpus()) == corpus_hash(_corpus())


def test_enumeration_order_does_not_change_the_hash():
    corpus = _corpus()
    assert corpus_hash(corpus) == corpus_hash(list(reversed(corpus)))


def test_changed_text_changes_the_hash():
    corpus = _corpus()
    changed = [*corpus[:-1], *iter_file_windows("infra/up.sh", "set -e\necho bye\n")]
    assert corpus_hash(changed) != corpus_hash(corpus)


def test_a_removed_chunk_changes_the_hash():
    corpus = _corpus()
    assert corpus_hash(corpus[:-1]) != corpus_hash(corpus)


def test_an_empty_corpus_still_hashes():
    assert corpus_hash([]) != ""


# ---------------------------------------------------------------------------
# Recipe fingerprint
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_stable_for_one_recipe():
    assert recipe_fingerprint(_OLLAMA) == recipe_fingerprint(_OLLAMA)


@pytest.mark.parametrize(
    "identity",
    [
        EmbedderIdentity(provider="openai", model="embeddinggemma", dims=768),
        EmbedderIdentity(provider="ollama", model="nomic-embed-text", dims=768),
        EmbedderIdentity(provider="ollama", model="embeddinggemma", dims=1024),
    ],
)
def test_a_different_embedder_fingerprints_differently(identity):
    assert recipe_fingerprint(identity) != recipe_fingerprint(_OLLAMA)


@pytest.mark.parametrize(
    ("module", "attribute", "value"),
    [
        (chunks_mod, "SMART_CAP_CHARS", 4000),
        (chunks_mod, "WINDOW_LINES", 200),
        (chunks_mod, "WINDOW_STRIDE", 100),
        (chunks_mod, "RECIPE_VERSION", "source-chunk/2"),
    ],
)
def test_a_changed_recipe_parameter_changes_the_fingerprint(monkeypatch, module, attribute, value):
    """Every knob that changes chunk text must invalidate stored vectors."""
    before = recipe_fingerprint(_OLLAMA)
    monkeypatch.setattr(module, attribute, value)
    assert recipe_fingerprint(_OLLAMA) != before


def test_a_changed_tokenizer_version_changes_the_fingerprint(monkeypatch):
    before = recipe_fingerprint(_OLLAMA)
    monkeypatch.setattr(fts_mod, "TOKENIZER_VERSION", "camel-split/2")
    assert recipe_fingerprint(_OLLAMA) != before


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------


def _manifest(**overrides) -> SourceIndexManifest:
    base = {
        "recipe_fingerprint": recipe_fingerprint(_OLLAMA),
        "corpus_hash": corpus_hash(_corpus()),
        "symbol_chunks": 7721,
        "file_window_chunks": 479,
        "files_covered": 940,
        "indexed_commit": "8d1e42e9",
        "built_at": "2026-08-17T17:22:56+00:00",
        "embedder": _OLLAMA,
    }
    base.update(overrides)
    return SourceIndexManifest(**base)


def test_default_path_sits_beside_the_index(tmp_path):
    assert default_manifest_path(tmp_path) == tmp_path / ".repowise" / "source_index.json"


def test_write_then_read_round_trips(tmp_path):
    path = tmp_path / "source_index.json"
    write_manifest(path, _manifest())
    assert read_manifest(path) == _manifest()


def test_the_ingest_record_round_trips(tmp_path):
    path = tmp_path / "source_index.json"
    manifest = _manifest(working_tree_ingest={"src/alpha.py": "abc123"})
    write_manifest(path, manifest)
    loaded = read_manifest(path)
    assert loaded is not None
    assert loaded.working_tree_ingest == {"src/alpha.py": "abc123"}


def test_a_manifest_without_the_ingest_record_reads_as_empty(tmp_path):
    """Every pre-F25 manifest on disk lacks the field and must stay readable."""
    import dataclasses

    path = tmp_path / "source_index.json"
    write_manifest(path, _manifest())
    raw = json.loads(path.read_text())
    raw.pop("working_tree_ingest")
    path.write_text(json.dumps(raw))
    loaded = read_manifest(path)
    assert loaded is not None
    assert loaded.working_tree_ingest == {}
    assert dataclasses.replace(loaded, working_tree_ingest={}) == loaded


def test_the_written_document_carries_every_field(tmp_path):
    path = tmp_path / "source_index.json"
    write_manifest(path, _manifest())
    raw = json.loads(path.read_text())
    assert set(raw) == {
        "built_at",
        "corpus_hash",
        "embedder",
        "file_window_chunks",
        "files_covered",
        "fts_path",
        "generation_id",
        "generation_sequence",
        "indexed_commit",
        "lance_table",
        "recipe_fingerprint",
        "stale_files",
        "symbol_chunks",
        "working_tree_ingest",
    }
    assert raw["embedder"] == {
        "provider": "ollama",
        "model": "embeddinggemma",
        "dims": 768,
        "document_prefix": "",
        "query_prefix": "",
    }


def test_writing_leaves_no_temporary_file_behind(tmp_path):
    path = tmp_path / "source_index.json"
    write_manifest(path, _manifest())
    write_manifest(path, _manifest(symbol_chunks=1))
    assert [p.name for p in tmp_path.iterdir()] == ["source_index.json"]


def test_a_missing_manifest_reads_as_none(tmp_path):
    assert read_manifest(tmp_path / "nope.json") is None


def test_observer_manifest_read_distinguishes_missing(tmp_path):
    result = inspect_manifest(tmp_path / "nope.json")

    assert result.state == "missing"
    assert result.manifest is None
    assert result.error is None


@pytest.mark.parametrize("body", ["not json", "[]", "null", '"a string"'])
def test_an_unusable_manifest_reads_as_none_rather_than_raising(tmp_path, body):
    """A rebuild deciding whether to reuse vectors must degrade, not crash."""
    path = tmp_path / "source_index.json"
    path.write_text(body)
    assert read_manifest(path) is None


@pytest.mark.parametrize("body", ["not json", "[]", "null", '"a string"'])
def test_observer_manifest_read_reports_unreadable(tmp_path, body):
    path = tmp_path / "source_index.json"
    path.write_text(body)

    result = inspect_manifest(path)

    assert result.state == "unreadable"
    assert result.manifest is None
    assert result.error


def test_observer_manifest_read_reports_ok(tmp_path):
    path = tmp_path / "source_index.json"
    write_manifest(path, _manifest())

    result = inspect_manifest(path)

    assert result.state == "ok"
    assert result.manifest == _manifest()
    assert result.error is None


def test_a_manifest_missing_fields_degrades_to_defaults(tmp_path):
    path = tmp_path / "source_index.json"
    path.write_text(json.dumps({"corpus_hash": "abc"}))
    manifest = read_manifest(path)
    assert manifest is not None
    assert manifest.corpus_hash == "abc"
    assert manifest.recipe_fingerprint == ""
    assert manifest.symbol_chunks == 0
    assert manifest.indexed_commit is None
