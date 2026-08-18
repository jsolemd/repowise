"""``repowise watch`` — watch for file changes and auto-update."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import click

from repowise.cli._setup import configure_cli_logging
from repowise.cli.helpers import (
    console,
    ensure_repowise_dir,
    resolve_command_target,
    run_async,
)
from repowise.core.ingestion import is_candidate_source_path

# ---------------------------------------------------------------------------
# Which filesystem events are worth an update
# ---------------------------------------------------------------------------

#: Repo-root files repowise itself rewrites at the end of every update. An
#: update that re-stamps CLAUDE.md would otherwise wake the watcher that
#: started it, and — now that the watcher indexes uncommitted work — the two
#: would drive each other in a loop for as long as `watch` is left running.
_SELF_WRITTEN_FILES: frozenset[str] = frozenset({"CLAUDE.md", "AGENTS.md", ".mcp.json"})

#: Same, for the per-agent config directories the editor setup manages.
#: ``.repowise``, ``.git`` and ``.vscode`` need no entry — they are already
#: blocked by the traversal blocklist :func:`is_candidate_source_path` reads.
_SELF_WRITTEN_DIRS: frozenset[str] = frozenset({".claude", ".codex", ".cursor", ".gemini"})

# The source lane publishes after the ordinary debounce.  Repository-wide
# graph/wiki/health work waits for a longer quiet period and coalesces all
# intervening saves, keeping it out of the agent's save-to-searchable path.
_SOURCE_SLOW_QUIET_SECONDS = 8.0


def _event_paths(event: object, repo_path: Path) -> set[str]:
    """The watchable repo-relative paths a filesystem event touched.

    Both ends of a move are considered. Editors that save atomically (the
    JetBrains IDEs, Vim with ``backupcopy=no``, plenty of Windows tools) write
    a temp file and rename it over the target, which arrives as one move event
    whose ``src_path`` is the temp name — looking only there means the save
    that matters is filtered out as junk and never wakes an update.

    Paths outside the watched root are dropped rather than raising: an
    unguarded ``relative_to`` here used to take watchdog's dispatcher thread
    down with it, leaving ``watch`` apparently running but permanently deaf.
    """
    # watchdog also emits opened/closed-without-write events. Indexing reads
    # the saved file, so treating those as edits creates a feedback loop where
    # each fast update schedules the next one. Synthetic test events and older
    # backends may omit ``event_type``; preserve their historical path.
    event_type = getattr(event, "event_type", None)
    if event_type is not None and event_type not in {
        "closed",
        "created",
        "deleted",
        "modified",
        "moved",
    }:
        return set()

    found: set[str] = set()
    for attr in ("src_path", "dest_path"):
        raw = getattr(event, attr, None)
        if not raw:
            continue
        try:
            rel = str(Path(raw).relative_to(repo_path))
        except ValueError:
            continue
        if is_watchable_path(rel):
            found.add(rel)
    return found


def _release_own_update_lock(repo_path: Path) -> None:
    """Drop the repo's single-flight update lock if this process owns it.

    ``run_update`` takes that lock and only gives it back through ``atexit``,
    which is correct for the one-shot CLI runs every other caller makes and
    wrong for a watcher: the process outlives the update, so the *next* save
    finds a lock whose PID is alive (its own), reports "another update is
    already running", and defers. Every save after the first would be a no-op
    for the rest of the session.

    Ownership is checked rather than assumed, so a lock a genuinely concurrent
    ``repowise update`` holds is never deleted out from under it.
    """
    from repowise.core.update_lock import read_update_lock, release_update_lock

    payload = read_update_lock(repo_path)
    if payload is not None and payload.get("pid") == os.getpid():
        release_update_lock(repo_path)


def is_watchable_path(rel_path: str) -> bool:
    """Whether a change to *rel_path* (repo-relative) should trigger an update.

    Two filters. The first is the traversal blocklist, so a write inside
    ``node_modules/``, ``.git/`` or ``dist/`` — or to a lockfile, a compiled
    artifact, anything the index would not read — never wakes an update; a
    single ``npm install`` otherwise means thousands of triggers. The second is
    the set of files repowise writes itself, which would make the watcher
    self-perpetuating.
    """
    posix = rel_path.replace("\\", "/")
    if posix in _SELF_WRITTEN_FILES:
        return False
    if posix.split("/", 1)[0] in _SELF_WRITTEN_DIRS:
        return False
    return is_candidate_source_path(posix)


# ---------------------------------------------------------------------------
# Single-repo watch (existing behavior)
# ---------------------------------------------------------------------------


def _watch_single_repo(
    repo_path: Path,
    provider_name: str | None,
    model: str | None,
    debounce_ms: int,
    verbose: bool,
    index_only: bool = False,
) -> None:
    """Watch a single repo for changes and trigger updates."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    from repowise.cli.commands.update_cmd.command import run_update
    from repowise.core.source_search import source_search_enabled

    ensure_repowise_dir(repo_path)

    changed_paths: set[str] = set()
    lock = threading.Lock()
    # Serialises the updates themselves. The debounce timer only guarantees a
    # quiet gap before a run starts, not that the previous run has finished —
    # an update that outlives the next quiet gap would otherwise have a second
    # one started on top of it.
    update_lock = threading.Lock()
    timer: threading.Timer | None = None
    source_fast_enabled = source_search_enabled()
    slow_paths: set[str] = set()
    fast_captured_paths: set[str] = set()
    slow_timer: threading.Timer | None = None
    slow_epoch = 0

    def _run_update(paths: set[str], *, source_captured: set[str] | None = None) -> None:
        if not paths:
            return

        console.print(f"[cyan]Detected {len(paths)} changed file(s), updating...[/cyan]")
        try:
            if source_captured:
                from repowise.core.source_search.outbox import suppress_incremental_paths

                context = suppress_incremental_paths(source_captured)
            else:
                from contextlib import nullcontext

                context = nullcontext()
            with context:
                run_update(
                    path=str(repo_path),
                    provider_name=provider_name,
                    model=model,
                    since=None,
                    reasoning=None,
                    cascade_budget=None,
                    dry_run=False,
                    workspace=False,
                    no_workspace=True,
                    repo_alias=None,
                    index_only=index_only,
                    verbose=verbose,
                    # The whole point of a file watcher: the edits it saw are
                    # in the working tree, and nothing has been committed.
                    include_working_tree=True,
                )
        except Exception as e:
            console.print(f"[red]Update failed: {e}[/red]")
        finally:
            _release_own_update_lock(repo_path)

    def _run_source_fast_path(paths: set[str]) -> bool:
        """Capture *paths* durably, then make their generation searchable."""

        from repowise.core.source_search.fast_update import capture_source_changes

        capture = run_async(capture_source_changes(repo_path, paths))
        try:
            from repowise.cli.source_search_runtime import reconcile_configured_source_index

            outcome = run_async(reconcile_configured_source_index(repo_path))
        except Exception as exc:
            # Capture already committed.  The queue/status surface is the
            # durable retry signal, so the heavy lane must not duplicate it.
            console.print(f"[yellow]Source search reconcile deferred: {exc}[/yellow]")
        else:
            if outcome is not None and outcome.status == "busy":
                console.print("[yellow]Source search update already running[/yellow]")
            elif outcome is not None:
                console.print(
                    f"[green]Source searchable:[/green] {len(capture.paths)} file(s) "
                    f"in {outcome.total_seconds:.3f}s"
                )
        return True

    def _on_slow_trigger(epoch: int) -> None:
        nonlocal slow_timer
        # Do not wait behind an active heavy update only to run a timer that a
        # newer save has already superseded.
        with lock:
            if epoch != slow_epoch:
                return
        with update_lock:
            with lock:
                if epoch != slow_epoch:
                    return
                paths = set(slow_paths)
                captured = fast_captured_paths & paths
                slow_paths.difference_update(paths)
                fast_captured_paths.difference_update(paths)
                slow_timer = None
            _run_update(paths, source_captured=captured)

    def _schedule_slow_update() -> None:
        nonlocal slow_timer, slow_epoch
        with lock:
            slow_epoch += 1
            epoch = slow_epoch
            if slow_timer is not None:
                slow_timer.cancel()
            slow_timer = threading.Timer(
                _SOURCE_SLOW_QUIET_SECONDS,
                _on_slow_trigger,
                args=(epoch,),
            )
            slow_timer.daemon = True
            slow_timer.start()

    def _on_trigger() -> None:
        nonlocal timer
        with lock:
            paths = set(changed_paths)
            changed_paths.clear()
            timer = None

        if not paths:
            return

        if not source_fast_enabled:
            # Feature-off behaviour stays byte-for-byte in spirit: one
            # debounced, serial update and no extra parsing or timers.
            with update_lock:
                _run_update(paths)
            return

        console.print(f"[cyan]Detected {len(paths)} saved file(s), indexing source...[/cyan]")
        captured = False
        try:
            captured = _run_source_fast_path(paths)
        except Exception as e:
            console.print(f"[red]Source fast update failed: {e}[/red]")

        with lock:
            slow_paths.update(paths)
            if captured:
                fast_captured_paths.update(paths)
            else:
                fast_captured_paths.difference_update(paths)
        _schedule_slow_update()

    class RepowiseHandler(FileSystemEventHandler):
        def on_any_event(self, event):
            nonlocal slow_epoch, slow_timer, timer
            if event.is_directory:
                return
            watched = _event_paths(event, repo_path)
            if not watched:
                return

            with lock:
                changed_paths.update(watched)
                if timer is not None:
                    timer.cancel()
                if source_fast_enabled and slow_timer is not None:
                    slow_timer.cancel()
                    slow_timer = None
                    slow_epoch += 1
                timer = threading.Timer(debounce_ms / 1000.0, _on_trigger)
                timer.daemon = True
                timer.start()

    observer = Observer()
    observer.schedule(RepowiseHandler(), str(repo_path), recursive=True)
    observer.start()

    console.print(f"[bold]Watching {repo_path}... Ctrl+C to stop[/bold]")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[yellow]Stopped watching.[/yellow]")
    observer.join()


# ---------------------------------------------------------------------------
# Workspace watch — per-repo handlers with independent debounce
# ---------------------------------------------------------------------------


def _watch_workspace(
    ws_root: Path,
    debounce_ms: int,
) -> None:
    """Watch all repos in a workspace with per-repo debounce timers."""
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    from repowise.core.workspace import WorkspaceConfig, update_workspace

    ws_config = WorkspaceConfig.load(ws_root)

    if not ws_config.repos:
        console.print("[yellow]No repos in workspace config.[/yellow]")
        return

    # Per-repo state: each repo gets its own change set and debounce timer
    repo_locks: dict[str, threading.Lock] = {}
    repo_changed: dict[str, set[str]] = {}
    repo_timers: dict[str, threading.Timer | None] = {}
    # Serialises a repo's own updates; see the single-repo watcher.
    repo_update_locks: dict[str, threading.Lock] = {}

    for entry in ws_config.repos:
        repo_locks[entry.alias] = threading.Lock()
        repo_changed[entry.alias] = set()
        repo_timers[entry.alias] = None
        repo_update_locks[entry.alias] = threading.Lock()

    def _make_trigger(alias: str) -> Callable[[], None]:
        """Create a trigger function for a specific repo."""

        def _on_trigger() -> None:
            with repo_update_locks[alias]:
                with repo_locks[alias]:
                    paths = set(repo_changed[alias])
                    repo_changed[alias].clear()
                    repo_timers[alias] = None

                if not paths:
                    return

                console.print(f"[cyan]{alias}: {len(paths)} changed file(s), updating...[/cyan]")
                try:
                    # Reload config in case it was updated
                    current_config = WorkspaceConfig.load(ws_root)
                    results = run_async(
                        update_workspace(
                            ws_root,
                            current_config,
                            repo_filter=alias,
                            # Same reason as the single-repo watcher: what the
                            # watcher saw is uncommitted, and the staleness
                            # check is otherwise commit-to-commit only.
                            include_working_tree=True,
                        )
                    )
                    from repowise.cli.source_search_runtime import (
                        reconcile_configured_source_index,
                    )

                    source_entry = current_config.get_repo(alias)
                    if source_entry is not None:
                        try:
                            source_outcome = run_async(
                                reconcile_configured_source_index(
                                    (ws_root / source_entry.path).resolve()
                                )
                            )
                        except Exception as exc:
                            console.print(
                                f"  [yellow]{alias}: source search reconcile deferred: "
                                f"{exc}[/yellow]"
                            )
                        else:
                            if source_outcome is not None and source_outcome.status == "busy":
                                console.print(
                                    f"  [yellow]{alias}: source search update already running[/yellow]"
                                )
                    for r in results:
                        if r.error:
                            console.print(f"  [red]{alias}: {r.error}[/red]")
                        elif r.updated:
                            console.print(
                                f"  [green]\u2713[/green] {alias}: "
                                f"{r.file_count} files, {r.symbol_count:,} symbols"
                            )
                        elif r.skipped_reason == "up_to_date":
                            console.print(f"  [dim]{alias}: already up to date[/dim]")
                except Exception as e:
                    console.print(f"  [red]{alias} update failed: {e}[/red]")

        return _on_trigger

    class WorkspaceRepoHandler(FileSystemEventHandler):
        """Handler for a single repo directory within the workspace."""

        def __init__(self, alias: str, repo_path: Path):
            super().__init__()
            self._alias = alias
            self._repo_path = repo_path

        def on_any_event(self, event):
            if event.is_directory:
                return
            watched = _event_paths(event, self._repo_path)
            if not watched:
                return

            alias = self._alias
            with repo_locks[alias]:
                repo_changed[alias].update(watched)
                old_timer = repo_timers[alias]
                if old_timer is not None:
                    old_timer.cancel()
                new_timer = threading.Timer(
                    debounce_ms / 1000.0,
                    _make_trigger(alias),
                )
                new_timer.daemon = True
                new_timer.start()
                repo_timers[alias] = new_timer

    observer = Observer()
    scheduled = 0
    for entry in ws_config.repos:
        abs_path = (ws_root / entry.path).resolve()
        if not abs_path.is_dir():
            console.print(f"  [yellow]Skipping {entry.alias}: directory not found[/yellow]")
            continue
        handler = WorkspaceRepoHandler(entry.alias, abs_path)
        observer.schedule(handler, str(abs_path), recursive=True)
        scheduled += 1

    if scheduled == 0:
        console.print("[yellow]No repo directories found to watch.[/yellow]")
        return

    observer.start()

    repo_list = ", ".join(e.alias for e in ws_config.repos)
    console.print(
        f"[bold]Watching workspace ({scheduled} repos: {repo_list})... Ctrl+C to stop[/bold]"
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[yellow]Stopped watching.[/yellow]")
    observer.join()


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@click.command("watch")
@click.argument("path", required=False, default=None)
@click.option("--provider", "provider_name", default=None, help="LLM provider name.")
@click.option("--model", default=None, help="Model identifier override.")
@click.option("--debounce", "debounce_ms", type=int, default=2000, help="Debounce delay in ms.")
@click.option(
    "--workspace",
    "-w",
    is_flag=True,
    default=False,
    help="Force workspace mode (watch all repos in the workspace).",
)
@click.option(
    "--no-workspace",
    is_flag=True,
    default=False,
    help="Force single-repo mode even when invoked from a workspace.",
)
@click.option(
    "--index-only",
    is_flag=True,
    default=False,
    help=(
        "Skip LLM page regeneration on every trigger. A watched repo fires an "
        "update per save, so on a repo indexed with docs that is a model call "
        "per save; this keeps the index, graph and health current for free. "
        "Workspace mode is index-only either way."
    ),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show debug logs from the pipeline.",
)
def watch_command(
    path: str | None,
    provider_name: str | None,
    model: str | None,
    debounce_ms: int,
    workspace: bool,
    no_workspace: bool,
    index_only: bool,
    verbose: bool,
) -> None:
    """Watch for file changes and auto-update wiki pages.

    Picks up uncommitted work: staged, unstaged and untracked files all reach
    the index, which is the difference between this and the post-commit hook.

    Auto-detects workspace mode when invoked from a workspace root.
    """
    configure_cli_logging(verbose=verbose)

    target = resolve_command_target(
        path=path,
        workspace_flag=workspace,
        no_workspace_flag=no_workspace,
    )
    target.notice(console, command="watch")

    if target.is_workspace:
        assert target.ws_root is not None
        _watch_workspace(target.ws_root, debounce_ms)
    else:
        assert target.repo_path is not None
        _watch_single_repo(
            target.repo_path,
            provider_name,
            model,
            debounce_ms,
            verbose,
            index_only,
        )
