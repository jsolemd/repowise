"""Alembic environment for repowise async migrations.

Uses the run_sync pattern: context.run_migrations() is synchronous and must
be called inside an ``await connection.run_sync(...)`` block.

The database URL is resolved from (in order):
1. The DATABASE_URL environment variable.
2. The ``sqlalchemy.url`` value in alembic.ini.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

# Import the Base so Alembic can detect schema changes for autogenerate.
from repowise.core.persistence.models import Base

# Alembic Config object (access to alembic.ini values)
config = context.config

# Set up logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    url = os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url", "")
    if not url:
        raise RuntimeError(
            "No database URL configured. "
            "Set the DATABASE_URL environment variable or sqlalchemy.url in alembic.ini."
        )
    # Normalise to async driver prefix
    if url.startswith("sqlite://") and "aiosqlite" not in url:
        url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    elif (
        url.startswith("postgresql://") or url.startswith("postgres://")
    ) and "asyncpg" not in url:
        url = url.replace("://", "+asyncpg://", 1)
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (no live DB connection required)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:  # type: ignore[no-untyped-def]
    with connection.begin():
        # v0.52 reused the fork's intermediate numeric revisions. The shipped
        # solemd_0001 head remains an ancestor, but an older intermediate stamp
        # must never be mistaken for upstream DDL that has not actually run.
        tables = set(inspect(connection).get_table_names())
        if "alembic_version" in tables and "source_index_updates" in tables:
            revisions = set(
                connection.execute(text("SELECT version_num FROM alembic_version")).scalars()
            )
            if revisions & {"0065", "0066", "0067", "0068"} and "doc_drift_findings" not in tables:
                raise RuntimeError(
                    "Ambiguous pre-v0.52 fork migration stamp: verify the stored schema "
                    "and map fork 0065/0066/0067/0068 to solemd_0002/0003/0004/0005 "
                    "before upgrading. No revision was changed."
                )
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


async def run_async_migrations() -> None:
    url = _get_url()
    connectable = create_async_engine(url)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (requires a live DB connection)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
