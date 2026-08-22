"""What ``inspect_source_index`` publishes about its own integrity.

The findings here are a contract with one consumer in another package
(``mcp_server/tool_index_status.py``), which decides whether search results
may be called trustworthy.  These tests pin the *structured* half of that
contract — component and code — so a reworded message can never silently
change a verdict, and a new fault kind can never slip past the consumer
wearing a name it already knows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repowise.core.source_search import status as status_module
from repowise.core.source_search.chunks import SymbolRecord, build_symbol_chunk
from repowise.core.source_search.fts import SourceFTSIndex
from repowise.core.source_search.generation import GenerationRef
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    default_manifest_path,
    write_manifest,
)
from repowise.core.source_search.status import (
    CODE_COUNT_MISMATCH,
    CODE_MISSING,
    CODE_UNREADABLE,
    COMPONENT_LEXICAL,
    EVIDENCE_PRESERVING_CODES,
    INTEGRITY_CODES,
    INTEGRITY_COMPONENTS,
    inspect_source_index,
)

_GENERATION = GenerationRef("generation-1", 1)
_FTS_REL = ".repowise/source_search/source_fts_v2.db"


def _chunk(name: str) -> object:
    path = f"src/{name}.py"
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


@pytest.fixture(autouse=True)
def _quiet_outbox(monkeypatch):
    """Answer the update-queue question with a clean snapshot.

    These tests are about the store-verification findings; letting the real
    query run would add an outbox finding whose content depends on whatever
    database the environment happens to resolve.
    """

    async def snapshot(*_args, **_kwargs):
        return status_module._SourceUpdateSnapshot(None, {}, 0, None)

    monkeypatch.setattr(status_module, "_source_update_snapshot", snapshot)


@pytest.fixture
def published_repo(tmp_path: Path) -> Path:
    """A repo whose manifest matches a real two-chunk lexical store."""

    repo = tmp_path / "repo"
    repo.mkdir()
    with SourceFTSIndex(repo / _FTS_REL, generation=_GENERATION) as fts:
        fts.index_chunks([_chunk("alpha"), _chunk("beta")])
    write_manifest(
        default_manifest_path(repo),
        SourceIndexManifest(
            recipe_fingerprint="recipe-1",
            corpus_hash="corpus-1",
            symbol_chunks=2,
            file_window_chunks=0,
            files_covered=2,
            indexed_commit="commit-1",
            built_at="2026-08-21T16:00:00+00:00",
            embedder=EmbedderIdentity(provider="mock", model="mock-embedder", dims=8),
            generation_id=_GENERATION.generation_id,
            generation_sequence=_GENERATION.sequence,
            fts_path=_FTS_REL,
        ),
    )
    return repo


async def test_a_healthy_publication_reports_no_integrity_findings(published_repo):
    status = await inspect_source_index(published_repo)

    assert status.integrity_findings == ()
    assert status.integrity_errors == ()
    assert status.state == "current"
    assert status.fts_chunks == 2


async def test_a_missing_lexical_store_is_named_by_component_and_code(published_repo):
    store = published_repo / _FTS_REL
    store.unlink()
    store.parent.rmdir()

    status = await inspect_source_index(published_repo)

    assert [(f.component, f.code) for f in status.integrity_findings] == [
        (COMPONENT_LEXICAL, CODE_MISSING)
    ]
    # The consumer must not be able to reach "stale" from this: the evidence
    # it would need to judge the publication is the store that is gone.
    assert CODE_MISSING not in EVIDENCE_PRESERVING_CODES
    assert not store.parent.exists()


async def test_an_unreadable_lexical_store_is_neither_created_nor_migrated(published_repo):
    """A status read on a corrupt store must report it, not repair it.

    The default ``SourceFTSIndex`` constructor would apply the schema script
    to this file and then report a count mismatch against an index it had just
    fabricated. Read-only opens turn that into an honest "unreadable".
    """

    store = published_repo / _FTS_REL
    store.write_bytes(b"")

    status = await inspect_source_index(published_repo)

    assert [(f.component, f.code) for f in status.integrity_findings] == [
        (COMPONENT_LEXICAL, CODE_UNREADABLE)
    ]
    assert status.fts_chunks is None
    assert store.read_bytes() == b""


async def test_a_count_mismatch_is_the_one_fault_that_leaves_the_index_judgeable(published_repo):
    write_manifest(
        default_manifest_path(published_repo),
        SourceIndexManifest(
            recipe_fingerprint="recipe-1",
            corpus_hash="corpus-1",
            symbol_chunks=9,
            file_window_chunks=0,
            files_covered=2,
            indexed_commit="commit-1",
            built_at="2026-08-21T16:00:00+00:00",
            embedder=EmbedderIdentity(provider="mock", model="mock-embedder", dims=8),
            generation_id=_GENERATION.generation_id,
            generation_sequence=_GENERATION.sequence,
            fts_path=_FTS_REL,
        ),
    )

    status = await inspect_source_index(published_repo)

    assert [(f.component, f.code) for f in status.integrity_findings] == [
        (COMPONENT_LEXICAL, CODE_COUNT_MISMATCH)
    ]
    assert status.fts_chunks == 2
    assert set(EVIDENCE_PRESERVING_CODES) == {CODE_COUNT_MISMATCH}


async def test_an_unreadable_manifest_is_reported_without_reading_it_as_absent(published_repo):
    default_manifest_path(published_repo).write_text("not json", encoding="utf-8")

    status = await inspect_source_index(published_repo)

    assert status.manifest_state == "unreadable"
    assert [f.code for f in status.integrity_findings] == [CODE_UNREADABLE]
    assert status.integrity_errors == (f"source manifest unreadable: {status.manifest_error}",)


async def test_every_finding_uses_a_declared_component_and_code(published_repo):
    """The vocabulary is closed: nothing reaches a consumer unannounced.

    ``INTEGRITY_COMPONENTS`` and ``INTEGRITY_CODES`` are what the consumer
    switches on. A producer that invents a value outside them would be
    classified by the consumer's fallback rather than by its own intent.
    """

    (published_repo / _FTS_REL).write_bytes(b"")
    dirty = await inspect_source_index(published_repo)
    default_manifest_path(published_repo).write_text("not json", encoding="utf-8")
    unreadable = await inspect_source_index(published_repo)

    findings = [*dirty.integrity_findings, *unreadable.integrity_findings]
    assert findings
    for finding in findings:
        assert finding.component in INTEGRITY_COMPONENTS
        assert finding.code in INTEGRITY_CODES
        assert finding.detail


async def test_integrity_errors_is_derived_from_the_findings(published_repo):
    """Display surfaces read the prose; nobody may classify on it.

    Deriving rather than storing it is what guarantees the two can never
    disagree, which is the failure this whole shape exists to remove.
    """

    (published_repo / _FTS_REL).unlink()

    status = await inspect_source_index(published_repo)

    assert status.integrity_errors == tuple(f.detail for f in status.integrity_findings)
    assert status.to_dict()["integrity_errors"] == list(status.integrity_errors)
    assert status.to_dict()["integrity_findings"] == tuple(
        f.to_dict() for f in status.integrity_findings
    )
