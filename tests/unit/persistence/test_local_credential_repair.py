"""Local SQLite stores receive 0075 and 0077 without stamping Alembic ancestry."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError

from repowise.core.persistence.database import create_engine, init_db
from repowise.core.persistence.models import Base, Repository


async def _legacy_store(tmp_path):
    engine = create_engine("sqlite+aiosqlite:///" + str(tmp_path / "legacy.db"))
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Tests use synthetic strings only; model defaults create the parent.
        await conn.execute(insert(Repository).values(id="r", name="repo", local_path=""))
        for n, (kind, commit, snippet) in enumerate(
            [
                ("hardcoded_password", "", "synthetic-password"),
                ("hardcoded_secret", "history", "synthetic-key"),
                ("hardcoded_secret", "empty", None),
                ("sql_injection", "", "non-credential evidence"),
                ("sql_injection", "history", "historical non-credential evidence"),
                *[(kind, "history", "synthetic-old-value") for kind in (
                    "public_env_secret", "aws_access_key", "github_token", "slack_token",
                    "google_api_key", "stripe_key", "private_key_pem",
                )],
            ]
        ):
            await conn.execute(
                text(
                    "INSERT INTO security_findings (id,repository_id,file_path,kind,severity,snippet,line_number,commit_sha,detected_at) VALUES (:id,'r','fixture.py',:kind,'high',:snippet,:line,:commit,CURRENT_TIMESTAMP)"
                ),
                {"id": n + 1, "kind": kind, "snippet": snippet, "line": n + 1, "commit": commit},
            )
    return engine


async def _rows(engine):
    async with engine.connect() as conn:
        return [
            tuple(row)
            for row in (
                await conn.execute(text("SELECT * FROM security_findings ORDER BY id"))
            ).all()
        ]


@pytest.mark.asyncio
async def test_local_upgrade_preserves_every_field_except_credential_snippets(tmp_path):
    engine = await _legacy_store(tmp_path)
    try:
        before = await _rows(engine)
        await init_db(engine)
        after = await _rows(engine)
        # snippet is column 5; all other cells, and non-credential rows, survive.
        expected = [
            (*row[:5], "", *row[6:])
            if row[3] != "sql_injection"
            else row
            for row in before
        ]
        assert after == expected
        async with engine.connect() as conn:
            assert (
                await conn.execute(text("SELECT count(*) FROM repowise_local_data_repairs"))
            ).scalar_one() == 2
            assert (
                await conn.execute(
                    text("SELECT count(*) FROM sqlite_master WHERE name='alembic_version'")
                )
            ).scalar_one() == 0
        # New scans write masked values. Later startup must retain them.
        async with engine.begin() as conn:
            await conn.execute(text("UPDATE security_findings SET snippet='synt****' WHERE id=1"))
        refreshed = await _rows(engine)
        await init_db(engine)
        assert await _rows(engine) == refreshed
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_failed_cleanup_rolls_back_marker_and_retries(tmp_path):
    engine = await _legacy_store(tmp_path)
    try:
        before = await _rows(engine)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE TRIGGER refuse_cleanup BEFORE UPDATE ON security_findings BEGIN SELECT RAISE(ABORT, 'repair failure probe'); END"
                )
            )
        with pytest.raises(IntegrityError, match="repair failure probe"):
            await init_db(engine)
        assert await _rows(engine) == before
        async with engine.begin() as conn:
            assert (
                await conn.execute(text("SELECT count(*) FROM repowise_local_data_repairs"))
            ).scalar_one() == 0
            await conn.execute(text("DROP TRIGGER refuse_cleanup"))
        await init_db(engine)
        assert (await _rows(engine))[0][5] == ""
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_initializers_share_one_completed_repair(tmp_path):
    engine = await _legacy_store(tmp_path)
    try:
        await asyncio.gather(init_db(engine), init_db(engine))
        async with engine.begin() as conn:
            assert (
                await conn.execute(text("SELECT count(*) FROM repowise_local_data_repairs"))
            ).scalar_one() == 2
            await conn.execute(text("UPDATE security_findings SET snippet='synt****' WHERE id=2"))
        refreshed = await _rows(engine)
        await asyncio.gather(init_db(engine), init_db(engine))
        assert await _rows(engine) == refreshed
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_completed_repair_does_not_wait_for_active_writer(tmp_path):
    engine = await _legacy_store(tmp_path)
    reader = create_engine("sqlite+aiosqlite:///" + str(tmp_path / "legacy.db"), busy_timeout_ms=50)
    try:
        await init_db(engine)
        async with engine.begin() as writer:
            await writer.execute(
                text("UPDATE security_findings SET snippet='ordinary evidence' WHERE id=4")
            )
            await init_db(reader)
    finally:
        await reader.dispose()
        await engine.dispose()


@pytest.mark.asyncio
async def test_new_repair_runs_after_old_marker_and_retries_without_reapplying_old(tmp_path):
    engine = await _legacy_store(tmp_path)
    try:
        await init_db(engine)
        async with engine.begin() as conn:
            await conn.execute(text(
                "DELETE FROM repowise_local_data_repairs WHERE repair_id = "
                "'0077_clear_unmasked_credential_snippets_v1'"
            ))
            await conn.execute(text(
                "UPDATE security_findings SET snippet='synthetic-pre-0077' "
                "WHERE kind='public_env_secret'"
            ))
            await conn.execute(text(
                "CREATE TRIGGER refuse_0077 BEFORE UPDATE ON security_findings "
                "WHEN OLD.kind='public_env_secret' "
                "BEGIN SELECT RAISE(ABORT, 'new repair failure'); END"
            ))
        before = await _rows(engine)
        with pytest.raises(IntegrityError, match="new repair failure"):
            await init_db(engine)
        assert await _rows(engine) == before
        async with engine.begin() as conn:
            assert (await conn.execute(text(
                "SELECT repair_id FROM repowise_local_data_repairs"
            ))).scalars().all() == ["0075_clear_legacy_credential_snippets_v1"]
            await conn.execute(text("DROP TRIGGER refuse_0077"))
        await init_db(engine)
        assert all(row[5] == "" for row in await _rows(engine) if row[3] != "sql_injection")
        # Once both markers exist, new masked credentials survive startup.
        async with engine.begin() as conn:
            await conn.execute(text(
                "UPDATE security_findings SET snippet='****' WHERE kind='public_env_secret'"
            ))
        refreshed = await _rows(engine)
        await init_db(engine)
        assert await _rows(engine) == refreshed
    finally:
        await engine.dispose()
