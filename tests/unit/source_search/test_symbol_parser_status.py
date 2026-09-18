"""SQL provenance cannot be inferred from a current source publication."""

from pathlib import Path

import pytest

from repowise.core.ingestion.parse_cache import parser_fingerprint
from repowise.core.persistence.database import (
    create_engine,
    create_session_factory,
    get_session,
    init_db,
)
from repowise.core.persistence.models import Repository
from repowise.core.source_search.manifest import (
    EmbedderIdentity,
    SourceIndexManifest,
    default_manifest_path,
    write_manifest,
)
from repowise.core.source_search.outbox import enqueue_full_update
from repowise.core.source_search.status import inspect_source_index


@pytest.mark.parametrize("sql_parser", [None, "older-parser", "current"])
async def test_inspection_keeps_sql_and_published_parser_identities_separate(
    tmp_path: Path, sql_parser: str | None
):
    current = parser_fingerprint()
    witness = current if sql_parser == "current" else sql_parser
    repo = tmp_path / "repo"
    repo.mkdir()
    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'wiki.db'}")
    try:
        await init_db(engine)
        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            session.add(
                Repository(
                    id="repo",
                    name="repo",
                    local_path=str(repo),
                    symbols_parser_fingerprint=witness,
                )
            )
            await session.flush()
            receipt = await enqueue_full_update(session, "repo", repo)
            receipt.state = "published"
            generation_id, sequence = receipt.generation_id, receipt.sequence
        write_manifest(
            default_manifest_path(repo),
            SourceIndexManifest(
                recipe_fingerprint="current-recipe",
                corpus_hash="empty",
                symbol_chunks=0,
                file_window_chunks=0,
                files_covered=0,
                indexed_commit="current-head",
                built_at="2026-09-18T00:00:00Z",
                embedder=EmbedderIdentity(provider="mock", model="test", dims=8),
                generation_id=generation_id,
                generation_sequence=sequence,
            ),
        )

        status = await inspect_source_index(repo, session_factory=factory, verify_stores=False)

        assert status.state == "current"
        assert status.parser_fingerprint == current
        assert status.symbols_parser_fingerprint == witness
        assert status.to_dict()["symbols_parser_fingerprint"] == witness
        # Inspection neither fills a legacy witness nor repairs a stale one.
        async with factory() as session:
            repository = await session.get(Repository, "repo")
            assert repository.symbols_parser_fingerprint == witness
    finally:
        await engine.dispose()
