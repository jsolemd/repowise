"""Saved-file source capture ahead of the heavier incremental pipeline.

The watcher knows exactly which paths caused its debounce timer to fire.  This
module parses only those paths, commits their ``wiki_symbols`` rows and source
outbox entry in one SQL transaction, and leaves graph/wiki/health refreshes to
the normal coalesced update.  The source lifecycle can therefore publish a
saved uncommitted edit without waiting for a repository-wide graph rebuild.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from repowise.core.ingestion.models import ParsedFile, compute_content_hash


@dataclass(frozen=True, slots=True)
class FastSourceCaptureResult:
    """What one atomic saved-file capture recorded."""

    paths: tuple[str, ...]
    parsed: int
    removed: int
    retained_stale: int


def _normalise_paths(repo: Path, changed_paths: set[str] | list[str]) -> tuple[str, ...]:
    paths: set[str] = set()
    for raw in changed_paths:
        if not raw:
            continue
        candidate = (repo / str(raw)).resolve()
        try:
            relative = candidate.relative_to(repo)
        except ValueError:
            continue
        path = relative.as_posix()
        if path and path != ".":
            paths.add(path)
    return tuple(sorted(paths))


def _state_flags(repo: Path) -> tuple[bool, bool]:
    try:
        raw = json.loads((repo / ".repowise" / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return bool(raw.get("include_submodules", False)), bool(
        raw.get("include_nested_repos", False)
    )


def _safe_hash(path: Path) -> str | None:
    try:
        return compute_content_hash(path.read_bytes())
    except OSError:
        return None


async def _repository_id(session: Any, repo: Path) -> str:
    from sqlalchemy import select

    from repowise.core.persistence.models import Repository

    rows = list((await session.execute(select(Repository))).scalars().all())
    for row in rows:
        try:
            if row.local_path and Path(row.local_path).resolve() == repo:
                return str(row.id)
        except OSError:
            continue
    if len(rows) == 1:
        return str(rows[0].id)
    raise RuntimeError(f"No indexed repository matching {repo} in the wiki database")


async def capture_source_changes(
    repo_path: Path | str,
    changed_paths: set[str] | list[str],
    *,
    db_url: str | None = None,
) -> FastSourceCaptureResult:
    """Atomically persist fresh symbols and an outbox row for known paths.

    A parser exception retains the previous symbol rows and records a failed
    parse, allowing the lifecycle to keep the last known-good chunks marked
    stale.  A real deletion or a path that is no longer indexable removes its
    symbol rows and closes its source chunks on publication.
    """

    from repowise.core.ingestion import ASTParser, FileTraverser
    from repowise.core.persistence.crud import reconcile_symbols_for_files
    from repowise.core.persistence.database import (
        create_engine,
        create_session_factory,
        get_session,
        init_db,
        resolve_db_url,
    )
    from repowise.core.repo_config import load_repo_config
    from repowise.core.source_search.chunks import window_eligible
    from repowise.core.source_search.outbox import enqueue_incremental_update

    repo = Path(repo_path).resolve()
    paths = _normalise_paths(repo, changed_paths)
    if not paths:
        return FastSourceCaptureResult((), 0, 0, 0)

    config = load_repo_config(repo)
    include_submodules, include_nested_repos = _state_flags(repo)
    traverser = FileTraverser(
        repo,
        extra_exclude_patterns=list(config.get("exclude_patterns") or []) or None,
        include_submodules=include_submodules,
        include_nested_repos=include_nested_repos,
    )
    parser = ASTParser()
    parsed_files: list[ParsedFile] = []
    diffs: list[Any] = []
    clear_paths: list[str] = []
    retained_stale = 0
    tracked_paths: set[str] | None = None

    for path in paths:
        absolute = repo / path
        if not absolute.is_file():
            clear_paths.append(path)
            diffs.append(
                SimpleNamespace(
                    path=path,
                    status="deleted",
                    old_path=path,
                    new_parsed=None,
                    parse_state="deleted",
                    content_hash=None,
                )
            )
            continue

        info = traverser.file_info_for_path(path, resolve_entry_point=False)
        if info is None:
            is_tracked_window = False
            if window_eligible(path, indexed_symbols=0):
                if tracked_paths is None:
                    # The full source corpus deliberately owns operational
                    # files through git rather than FileTraverser: shell/YAML
                    # and config-excluded code commonly never reach the parser
                    # lane. Reuse that one fail-loud corpus witness, and pay for
                    # it only when a saved path actually needs the fallback.
                    from repowise.core.source_search.indexer import _tracked_paths

                    tracked_paths = set(_tracked_paths(repo))
                is_tracked_window = path in tracked_paths

            if is_tracked_window:
                content_hash = _safe_hash(absolute)
                if content_hash is None:
                    # A window is authoritative from raw bytes. Without the
                    # captured hash lifecycle could publish different bytes
                    # than this SQL change recorded, so retain last-known-good
                    # rows and disclose staleness just like a parser failure.
                    retained_stale += 1
                    diffs.append(
                        SimpleNamespace(
                            path=path,
                            status="modified",
                            old_path=None,
                            new_parsed=None,
                            parse_state="failed",
                            content_hash=None,
                        )
                    )
                    continue

                clear_paths.append(path)
                diffs.append(
                    SimpleNamespace(
                        path=path,
                        status="modified",
                        old_path=None,
                        new_parsed=None,
                        parse_state="window",
                        content_hash=content_hash,
                    )
                )
                continue

            clear_paths.append(path)
            diffs.append(
                SimpleNamespace(
                    path=path,
                    status="modified",
                    old_path=None,
                    new_parsed=None,
                    parse_state="unindexed",
                    content_hash=_safe_hash(absolute),
                )
            )
            continue

        try:
            source = absolute.read_bytes()
            parsed = parser.parse_file(info, source)
        except Exception:
            retained_stale += 1
            diffs.append(
                SimpleNamespace(
                    path=path,
                    status="modified",
                    old_path=None,
                    new_parsed=None,
                    parse_state="failed",
                    content_hash=_safe_hash(absolute),
                )
            )
            continue

        parsed_files.append(parsed)
        diffs.append(
            SimpleNamespace(
                path=path,
                status="modified",
                old_path=None,
                new_parsed=parsed,
                parse_state="parsed",
                content_hash=parsed.content_hash,
            )
        )

    engine = create_engine(db_url or resolve_db_url(repo))
    try:
        await init_db(engine)
        factory = create_session_factory(engine)
        async with get_session(factory) as session:
            repository_id = await _repository_id(session, repo)
            if parsed_files:
                symbols: list[Any] = []
                parsed_paths: list[str] = []
                for parsed in parsed_files:
                    parsed_paths.append(parsed.file_info.path)
                    for symbol in parsed.symbols:
                        if not getattr(symbol, "file_path", None):
                            symbol.file_path = parsed.file_info.path
                        symbols.append(symbol)
                await reconcile_symbols_for_files(
                    session,
                    repository_id,
                    parsed_paths,
                    symbols,
                )
            if clear_paths:
                await reconcile_symbols_for_files(session, repository_id, clear_paths, [])
            await enqueue_incremental_update(
                session,
                repository_id,
                repo,
                file_diffs=diffs,
                parsed_files=parsed_files,
            )
    finally:
        await engine.dispose()

    return FastSourceCaptureResult(
        paths=paths,
        parsed=len(parsed_files),
        removed=len(clear_paths),
        retained_stale=retained_stale,
    )
