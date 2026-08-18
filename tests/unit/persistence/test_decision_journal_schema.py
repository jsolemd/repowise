"""Local SQLite reconciliation for decision-journal projection columns."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from repowise.core.persistence.database import create_engine, init_db
from repowise.core.persistence.models import DecisionRecord, Repository


async def test_init_db_adds_journal_columns_to_a_populated_legacy_table(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "wiki.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_engine(db_url)
    await init_db(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(Repository(id="repo", name="repo", local_path=str(tmp_path)))
        await session.flush()
        session.add(
            DecisionRecord(
                id="legacy-decision",
                repository_id="repo",
                title="Legacy row",
                decision="It predates the journal projection columns.",
            )
        )
        await session.commit()
    await engine.dispose()

    with sqlite3.connect(db_path) as connection:
        for column in ("anchors_json", "supersedes", "confirmed_at"):
            connection.execute(f'ALTER TABLE "decision_records" DROP COLUMN "{column}"')
        connection.commit()

    reconciled = create_engine(db_url)
    await init_db(reconciled)
    await reconciled.dispose()

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1] for row in connection.execute('PRAGMA table_info("decision_records")').fetchall()
        }
        row = connection.execute(
            "SELECT anchors_json, supersedes, confirmed_at "
            "FROM decision_records WHERE id = 'legacy-decision'"
        ).fetchone()

    assert {"anchors_json", "supersedes", "confirmed_at"}.issubset(columns)
    assert row == ("[]", None, None)
