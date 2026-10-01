"""How long a LanceDB table keeps its old versions, and the pass that holds it there.

Every Lance write commits a new table version, and a version keeps each data and
deletion file it referenced until a cleanup removes it. Lance cleans up on its
own: a table carries ``lance.auto_cleanup`` in its manifest and, every 20
commits, drops versions older than 14 days. It never compacts, and 14 days is
long at the wiki store's write rate. A template-wiki refresh commits several
``merge_insert`` calls, so on 2026-10-01 SoleMD.Make's ``wiki_pages`` held 690
rows, 4.9 MB of live data, in 826 MB of disk across about 7,200 versions, all
from the previous 14 days. The source-search store already cleans its own
versions after a publish (``source_search/retention.py``). This module holds
the rule both stores follow and the daily sweep that holds every table to it.

The rule: compaction folds the live rows into fresh fragments, and a version is
removed once it has been superseded for longer than the retention window. The
window is measured from supersession rather than from commit so that a table
idle for a month stays readable through its next compaction: the version a
reader holds was committed long ago, but it stopped being current only when
compaction replaced it.

Readers decide whether removal is safe. A LanceDB handle stays on the version it
opened unless its connection sets ``read_consistency_interval``, and the MCP
server and dashboard API hold their wiki handles for the life of the process.
Before this, those handles served the vectors of their own startup, and once a
cleanup removed that version every later read would fail on a missing file.
Every RepoWise connection now refreshes within
:data:`READ_CONSISTENCY_INTERVAL`, so no reader holds a superseded version for
longer than that plus one query, far inside any retention window.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

__all__ = [
    "READ_CONSISTENCY_INTERVAL",
    "SWEEP_INTERVAL",
    "VERSION_RETENTION",
    "RepoSweep",
    "VersionPrune",
    "prune_versions",
    "sweep_repositories",
]

log = structlog.get_logger(__name__)

#: A superseded version survives this long. Jon asked on 2026-10-01 that old
#: versions be pruned "for sure, or at least after like 14 days", to stop the
#: bloat; 14 days is what Lance's auto-cleanup already keeps, and it held
#: 1.7 GB across the workspace where one day holds 0.6 GB. These tables are an
#: index RepoWise regenerates from source, and their readers refresh within
#: seconds (below), so a day is ample margin for inspection or a rollback. A
#: store whose readers provably need less may clean sooner through
#: :func:`prune_versions`; none keeps more, because the sweep holds every table
#: to this.
VERSION_RETENTION = timedelta(days=1)

#: How stale a LanceDB read may be before the handle checks for a newer
#: version. Template wiki pages already trail edits by up to ten minutes, so
#: seconds cost no visible freshness, and the check stays off back-to-back
#: queries (a strong-consistency check added 2.5 ms a query on a 7,000-version
#: table).
READ_CONSISTENCY_INTERVAL = timedelta(seconds=5)

#: The sweep compacts and cleans each repository at most this often. The
#: reconciliation timer calls it every pass, and an idle pass costs one stat.
SWEEP_INTERVAL = timedelta(days=1)

STAMP_FILENAME = "lance_retention.json"

# Lance computes the cleanup cutoff from its own clock as the cleanup starts, a
# moment after this module computes it, so the cutoff moves this much earlier
# to keep the version that was current at the boundary.
_CLOCK_MARGIN = timedelta(seconds=1)
# Older than any version: ``optimize`` with this window compacts and removes
# nothing. LanceDB reads a missing window as seven days, never as "none".
_NO_CLEANUP = timedelta(days=36500)


@dataclass(frozen=True, slots=True)
class VersionPrune:
    table: str
    versions_before: int
    versions_removed: int
    bytes_removed: int
    fragments_compacted: int


@dataclass(frozen=True, slots=True)
class RepoSweep:
    repo: Path
    #: ``swept``, ``not_due``, ``busy`` (a writer holds a lock), ``no_store``
    #: or ``failed``. Only ``swept`` stamps the repository; the rest retry on
    #: the next pass.
    status: str
    tables: tuple[VersionPrune, ...] = ()
    reason: str | None = None


def _committed_at(version: Mapping[str, Any]) -> datetime:
    # ``list_versions`` converts Lance's UTC nanoseconds with
    # ``datetime.fromtimestamp``, which yields naive local time.
    return version["timestamp"].astimezone(UTC)


async def prune_versions(
    table: Any,
    *,
    retention: timedelta = VERSION_RETENTION,
    now: datetime | None = None,
) -> VersionPrune:
    """Compact *table*, then remove the versions superseded more than *retention* ago.

    The version that was current at ``now - retention`` is the oldest one kept:
    every version before it had been replaced by then. When no version is that
    old, the table is left alone. Compaction merges every fragment below
    Lance's million-row target, so on a RepoWise table it rewrites all the live
    rows, and the version it replaces holds the old copy for the whole window.
    Compacting daily with nothing to remove would keep a copy of the table for
    every day retained; the RepoWise alias's 250 MB source table doubled on a
    trial sweep. *now* exists for tests; Lance cleans against the real clock.
    """

    now = now or datetime.now(UTC)
    versions = sorted(await table.list_versions(), key=lambda version: version["version"])
    cutoff = now - retention
    current_at_cutoff = None
    for version in versions:
        if _committed_at(version) > cutoff:
            break
        current_at_cutoff = version
    uri = await table.uri()
    if current_at_cutoff is None or current_at_cutoff is versions[0]:
        return VersionPrune(uri, len(versions), 0, 0, 0)

    # Compaction first, on its own, so its run time cannot move the cleanup
    # boundary past the version a reader may still hold.
    compaction = (await table.optimize(cleanup_older_than=_NO_CLEANUP)).compaction
    keep_from = _committed_at(current_at_cutoff) - _CLOCK_MARGIN
    prune = (await table.optimize(cleanup_older_than=datetime.now(UTC) - keep_from)).prune

    result = VersionPrune(
        table=uri,
        versions_before=len(versions),
        versions_removed=prune.old_versions_removed,
        bytes_removed=prune.bytes_removed,
        fragments_compacted=compaction.fragments_removed,
    )
    log.info(
        "lance_versions_pruned",
        table=result.table,
        retention_days=retention / timedelta(days=1),
        versions_before=result.versions_before,
        versions_removed=result.versions_removed,
        bytes_removed=result.bytes_removed,
        fragments_compacted=result.fragments_compacted,
    )
    return result


async def _table_names(db: Any) -> list[str]:
    names: list[str] = []
    token = None
    while True:
        page = await db.list_tables(page_token=token)
        names.extend(page.tables)
        token = page.page_token
        if not token:
            return names


async def _prune_store(lance_dir: Path, now: datetime) -> tuple[VersionPrune, ...]:
    import lancedb  # type: ignore[import]

    db = await lancedb.connect_async(str(lance_dir))
    pruned = []
    for name in sorted(await _table_names(db)):
        pruned.append(await prune_versions(await db.open_table(name), now=now))
    return tuple(pruned)


def _stamp_path(repo: Path) -> Path:
    return repo / ".repowise" / STAMP_FILENAME


def _due(repo: Path, now: datetime) -> bool:
    try:
        stamp = json.loads(_stamp_path(repo).read_text(encoding="utf-8"))
        swept_at = datetime.fromisoformat(stamp["swept_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return True
    return now - swept_at >= SWEEP_INTERVAL


async def _sweep_repo(repo: Path, now: datetime) -> RepoSweep:
    from filelock import FileLock, Timeout

    from repowise.core.fsutils import atomic_write_text
    from repowise.core.source_search.lifecycle import reconcile_lock_path
    from repowise.core.update_lock import UpdateLockUnavailableError, strict_update_lock

    lance_dir = repo / ".repowise" / "lancedb"
    if not lance_dir.is_dir():
        return RepoSweep(repo, "no_store")
    reconcile_lock_path(repo).parent.mkdir(parents=True, exist_ok=True)
    publication = FileLock(reconcile_lock_path(repo), timeout=0)
    try:
        with strict_update_lock(repo), publication:
            tables = await _prune_store(lance_dir, now)
    except (UpdateLockUnavailableError, Timeout) as exc:
        return RepoSweep(repo, "busy", reason=str(exc))
    except Exception as exc:
        log.warning("lance_version_sweep_failed", repo=str(repo), exc_info=True)
        return RepoSweep(repo, "failed", reason=str(exc))
    atomic_write_text(_stamp_path(repo), json.dumps({"swept_at": now.isoformat()}) + "\n")
    return RepoSweep(repo, "swept", tables)


async def sweep_repositories(
    workspace_root: Path,
    repos: Sequence[Path],
    *,
    now: datetime | None = None,
) -> list[RepoSweep]:
    """Hold every LanceDB table of *repos* to :data:`VERSION_RETENTION`.

    Runs at most once per :data:`SWEEP_INTERVAL` per repository, and never
    beside an index writer: it takes the workspace update lock, then each
    repository's update lock and source-publication lock, and a repository
    whose lock is held is left for the next pass.
    """

    from repowise.core.update_lock import release_workspace_lock, update_workspace_lock

    now = now or datetime.now(UTC)
    results = {repo: RepoSweep(repo, "not_due") for repo in repos if not _due(repo, now)}
    due = [repo for repo in repos if repo not in results]
    if due:
        owner = update_workspace_lock(workspace_root)
        if owner is not None:
            reason = f"workspace update running (PID {owner.get('pid')})"
            results.update({repo: RepoSweep(repo, "busy", reason=reason) for repo in due})
        else:
            try:
                for repo in due:
                    results[repo] = await _sweep_repo(repo, now)
            finally:
                release_workspace_lock(workspace_root)
    return [results[repo] for repo in repos]
