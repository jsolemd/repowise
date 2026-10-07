"""The PostgreSQL inventory and queries share committed live-page authority."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from repowise.core.persistence.search import FullTextSearch


@pytest.mark.parametrize("repo_id", [None, "scoped-repository"])
async def test_pg_queries_and_frequencies_exclude_tombstones(repo_id, monkeypatch):
    engine = MagicMock()
    engine.dialect.name = "postgresql"
    conn = AsyncMock()
    result = MagicMock()
    result.fetchall.return_value = []
    result.scalar.return_value = 3
    conn.execute.return_value = result
    engine.connect.return_value.__aenter__.return_value = conn
    fts = FullTextSearch(engine)
    monkeypatch.setattr(fts, "_build_ts_query", AsyncMock(return_value="owner"))
    await fts._search_postgresql("owner", 7, repo_id)
    await fts._pg_document_frequency(conn, "owner")
    await fts._pg_document_frequency(conn, "")
    await fts.list_indexed_ids()
    statements = [str(call.args[0]) for call in conn.execute.call_args_list]
    assert len(statements) == 4
    assert all("freshness_status != 'tombstone'" in sql for sql in statements)
    assert "LIMIT :lim" in statements[0]
    params = conn.execute.call_args_list[0].args[1]
    assert params["lim"] == 7
    assert params.get("repo_id") == repo_id
    assert ("repository_id = :repo_id" in statements[0]) is (repo_id is not None)
