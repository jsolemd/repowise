"""Provenance of committed symbol parses, independent of derived indexes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, select, update

from .models import Repository, WikiSymbol


async def stored_symbol_parser(session: Any, repo_id: str) -> str | None:
    return (
        await session.execute(
            select(Repository.symbols_parser_fingerprint).where(Repository.id == repo_id)
        )
    ).scalar_one_or_none()


async def symbol_parser_refresh_required(
    repo_path: Path | str, *, db_url: str | None = None
) -> bool:
    """Inspect SQL without bootstrapping a missing store or migrating a legacy one.

    A source manifest and the parse cache describe different stores. Neither
    proves that the unchanged files' parsed symbols reached this transaction.
    A missing SQL witness therefore requests one native full-tree refresh.
    """
    from sqlalchemy.engine import make_url

    from ..ingestion.parse_cache import parser_fingerprint
    from .database import create_engine, get_configured_db_url, resolve_db_url

    root = Path(repo_path).resolve()
    if (
        db_url is None
        and get_configured_db_url() is None
        and not (root / ".repowise" / "wiki.db").is_file()
    ):
        return False
    resolved_url = db_url or resolve_db_url(root)
    url = make_url(resolved_url)
    if (
        url.get_backend_name() == "sqlite"
        and url.database not in {None, ":memory:"}
        and not Path(url.database).is_file()
    ):
        return False
    engine = create_engine(resolved_url)
    try:
        async with engine.connect() as conn:

            def columns(sync_conn: Any) -> set[str]:
                inspector = inspect(sync_conn)
                if not inspector.has_table("repositories"):
                    return set()
                return {column["name"] for column in inspector.get_columns("repositories")}

            if "symbols_parser_fingerprint" not in await conn.run_sync(columns):
                return True
            rows = (
                await conn.execute(
                    select(Repository.local_path, Repository.symbols_parser_fingerprint)
                )
            ).all()
            for path, fingerprint in rows:
                try:
                    if path and Path(path).resolve() == root:
                        return fingerprint != parser_fingerprint()
                except OSError:
                    continue
            # Match the source loader's established single-repository fallback
            # when a checkout (and its local wiki.db) has moved.
            if len(rows) == 1:
                return rows[0].symbols_parser_fingerprint != parser_fingerprint()
            return False
    finally:
        await engine.dispose()


@dataclass(frozen=True)
class SymbolParserRefresh:
    """Files actually reconciled in the current SQL transaction."""

    parser_fingerprint: str
    reconciled_paths: frozenset[str]


class SymbolParserRefreshError(RuntimeError):
    """The transaction cannot attest a complete parser refresh."""


async def complete_symbol_parser_refresh(
    session: Any,
    repo_id: str,
    refresh: SymbolParserRefresh,
    *,
    required_paths: set[str],
    prune_ready: bool = True,
) -> None:
    """Certify a full refresh only after pruning and complete parse coverage.

    Retained rows are authoritative here: a missing parse cannot erase its
    old symbols or certify them as current. Empty successful parses still
    count, and authoritatively deleted/excluded rows no longer require a parse.
    The caller enqueues its full source receipt in this same transaction.
    """
    retained = set(
        (
            await session.execute(
                select(WikiSymbol.file_path).where(WikiSymbol.repository_id == repo_id).distinct()
            )
        ).scalars()
    )
    missing = (retained | required_paths) - refresh.reconciled_paths
    if not prune_ready or missing:
        detail = ", ".join(sorted(missing)[:5]) or "file cleanup was incomplete"
        raise SymbolParserRefreshError(
            f"Symbol parser refresh incomplete ({detail}); the previous SQL parser "
            "witness was retained so the next update retries."
        )
    from repowise.core.pipeline.resume.ledger import invalidate_parser_completions

    # Old ANALYSIS/GENERATION completions must not become skippable again
    # when INDEX writes a current witness, including after a process restart.
    await invalidate_parser_completions(session, repo_id)
    await session.execute(
        update(Repository)
        .where(Repository.id == repo_id)
        .values(symbols_parser_fingerprint=refresh.parser_fingerprint)
    )
