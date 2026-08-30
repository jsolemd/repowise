"""``repowise decision`` — manage architectural decision records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from repowise.cli.commands import _tool_adapters as _ta
from repowise.cli.helpers import (
    console,
    ensure_repowise_dir,
    get_db_url_for_repo,
    resolve_command_target,
    run_async,
)
from repowise.cli.output import emit_json, format_option, notice_console
from repowise.core.analysis.decisions.provenance import LISTABLE_SOURCES
from repowise.core.precedent.currency import describe_decision_currency

#: The ladder's real sources plus the no-filter sentinel. Derived, because the
#: hand-written copy had drifted: it offered ``readme_mining`` (since retired)
#: while omitting ``session``, the source carrying a user's own words and the
#: one you would most want to filter for.
_SOURCE_CHOICES: tuple[str, ...] = (*LISTABLE_SOURCES, "all")


def _journal_mode_enabled() -> bool:
    from repowise.core.analysis.decisions.journal import (
        DecisionJournalError,
        decisions_journal_path,
    )

    try:
        return decisions_journal_path() is not None
    except DecisionJournalError as exc:
        raise click.ClickException(str(exc)) from exc


def repo_option() -> Any:
    """The ``--repo <alias>`` every decision subcommand takes.

    One decorator rather than seven copies of the same option, so the help
    text and the destination name cannot drift between subcommands the way a
    per-command spelling would.
    """
    return click.option(
        "--repo",
        "repo_alias",
        default=None,
        metavar="ALIAS",
        help="Workspace repo to target, by alias. Alternative to the path argument.",
    )


def _resolve_decision_repo(
    path: str | None, fmt: str = "table", repo_alias: str | None = None
) -> Path:
    """Resolve the repo path for decision subcommands.

    Honors workspace auto-detection: in workspace mode without an explicit
    path or alias, targets the primary repo and prints a transparency notice.

    ``--repo <alias>`` and the positional path are two ways of naming one
    repo, so passing both is an error rather than a silent precedence rule.
    Every decision a workspace holds lives in its own repo's journal, and
    until now the only reachable one from the CLI was the primary — the
    workspace's other repos had no spelling at all, which is why the alias
    resolver on ``CommandTarget`` existed with no caller.
    """
    if repo_alias is not None and path is not None:
        raise click.UsageError(
            "Pass --repo <alias> or a path, not both — the alias already names a repo."
        )

    target = resolve_command_target(path=path)

    resolved: Path | None = None
    if repo_alias is not None:
        if not target.is_workspace:
            raise click.ClickException(
                f"--repo {repo_alias} needs a workspace; no .repowise-workspace.yaml "
                "was found at or above here, so there are no aliases to resolve."
            )
        resolved = target.resolve_repo_alias(repo_alias)
        if resolved is None:
            available = target.ws_config.repo_aliases() if target.ws_config else []
            # Same sentence the MCP tool raises, so an alias typo reads the
            # same whichever surface the caller reached for.
            raise click.ClickException(f"Unknown repo '{repo_alias}'. Available: {available}")
        # Set before the notice so the transparency line names the repo the
        # command actually ran on, not the workspace as a whole.
        target.repo_filter = repo_alias

    target.notice(notice_console(fmt), command="decision")

    if resolved is not None:
        return resolved
    if target.is_workspace:
        primary = target.primary_path()
        if primary is None:
            raise click.ClickException("Workspace has no primary repo configured.")
        return primary
    assert target.repo_path is not None
    return target.repo_path


@click.group("decision")
def decision_group() -> None:
    """Manage architectural decision records."""


async def _resolve_decision_id(session, decision_id: str) -> str | None:
    """Expand a (possibly truncated) decision id to the full stored id.

    ``decision list`` prints 8-char prefixes, so every id-taking subcommand
    accepts a unique prefix. Returns None when nothing matches; raises on an
    ambiguous prefix.
    """
    from sqlalchemy import select

    from repowise.core.persistence.models import DecisionRecord
    from repowise.core.persistence.sql import LIKE_ESCAPE, escape_like

    result = await session.execute(
        select(DecisionRecord.id)
        .where(DecisionRecord.id.like(f"{escape_like(decision_id)}%", escape=LIKE_ESCAPE))
        .limit(2)
    )
    ids = [row[0] for row in result.all()]
    if len(ids) > 1:
        raise click.ClickException(
            f"Decision id prefix {decision_id!r} is ambiguous; use more characters."
        )
    return ids[0] if ids else None


# ---------------------------------------------------------------------------
# decision add
# ---------------------------------------------------------------------------


@decision_group.command("add")
@click.argument("path", required=False, default=None)
@click.option("--title", default=None, help="Decision title (short).")
@click.option("--context", default=None, help="What forced this decision?")
@click.option("--decision", "decision_text", default=None, help="What was chosen?")
@click.option("--rationale", default=None, help="Why it was chosen.")
@click.option(
    "--alternative", "alternatives", multiple=True, help="A rejected alternative. Repeatable."
)
@click.option(
    "--consequence", "consequences", multiple=True, help="A tradeoff accepted. Repeatable."
)
@click.option(
    "--affects", "affected", multiple=True, help="A file or module this governs. Repeatable."
)
@click.option("--tag", "tags", multiple=True, help="A tag. Repeatable.")
@repo_option()
@format_option()
def decision_add(
    path: str | None,
    title: str | None,
    context: str | None,
    decision_text: str | None,
    rationale: str | None,
    alternatives: tuple[str, ...],
    consequences: tuple[str, ...],
    affected: tuple[str, ...],
    tags: tuple[str, ...],
    repo_alias: str | None,
    fmt: str,
) -> None:
    """Add an architectural decision, interactively or from flags.

    With both --title and --decision, records without prompting and prints the
    new id, so a script or an agent can call it. Everything else is optional.

    A flag-driven record lands as `proposed`, where the prompts record `active`.
    A person answering eight questions has reviewed the decision; a caller
    inferring one from a diff has not, and the store should be able to tell
    them apart. Promote with `repowise decision confirm <id>`.

    That holds in journal mode too: a proposal enters the JSONL with
    `confirmed_at` null, and the git diff is the review. Journal mode still
    requires a rationale and at least one `--affects` anchor either way.
    """
    # Flags and prompts are the two paths, and a half-filled command line is
    # neither: falling through to the prompts would hang a caller that has no
    # stdin, which is the failure this command exists to stop having.
    non_interactive = bool(title and decision_text)
    if not non_interactive:
        flagged = any((title, context, decision_text, rationale)) or any(
            (alternatives, consequences, affected, tags)
        )
        if flagged or fmt == "json":
            _ta.emit_error(
                {
                    "error": "--title and --decision are both required to add a "
                    "decision without prompting.",
                    "guidance": "Run `repowise decision add` with no flags to be "
                    "prompted for each field instead.",
                },
                fmt,
            )

    repo_path = _resolve_decision_repo(path, fmt, repo_alias)
    ensure_repowise_dir(repo_path)

    journal_mode = _journal_mode_enabled()
    status = "proposed" if non_interactive else "active"
    alternatives_list = list(alternatives)
    consequences_list = list(consequences)
    affected_files = list(affected)
    tags_list = list(tags)

    if not non_interactive:
        console.print("[bold]Add Architectural Decision[/bold]\n")

        title = click.prompt("Decision title (short)")
        context = click.prompt("Context (what forced this decision?)", default="")
        decision_text = click.prompt("Decision (what was chosen?)")
        rationale = click.prompt("Rationale (why?)", default="")

        alternatives_raw = click.prompt(
            "Rejected alternatives (comma-separated, optional)", default=""
        )
        alternatives_list = [a.strip() for a in alternatives_raw.split(",") if a.strip()]

        consequences_raw = click.prompt(
            "Tradeoffs/consequences (comma-separated, optional)", default=""
        )
        consequences_list = [c.strip() for c in consequences_raw.split(",") if c.strip()]

        affected_raw = click.prompt(
            "Affected files/modules (comma-separated, optional)", default=""
        )
        affected_files = [f.strip() for f in affected_raw.split(",") if f.strip()]

        tags_raw = click.prompt(
            "Tags (comma-separated: auth, database, api, performance, security, infra, testing)",
            default="",
        )
        tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]

    if journal_mode:
        unsupported: list[str] = []
        if context:
            unsupported.append("context")
        if alternatives_list:
            unsupported.append("alternatives")
        if consequences_list:
            unsupported.append("consequences")
        if tags_list:
            unsupported.append("tags")
        if unsupported:
            raise click.ClickException(
                "Decision journal mode cannot losslessly store fields: " + ", ".join(unsupported)
            )
        if not (rationale or "").strip() or not affected_files:
            raise click.ClickException(
                "Decision journal mode requires a rationale (the why) and at least "
                "one affected file anchor"
            )

    async def _persist() -> tuple[str, str]:
        from repowise.core.persistence import (
            create_engine,
            create_session_factory,
            get_session,
            init_db,
            upsert_decision,
            upsert_repository,
        )

        url = get_db_url_for_repo(repo_path)
        engine = create_engine(url)
        await init_db(engine)
        sf = create_session_factory(engine)

        async with get_session(sf) as session:
            repo = await upsert_repository(session, name=repo_path.name, local_path=str(repo_path))
            if journal_mode:
                from repowise.core.analysis.decisions.journal_projection import (
                    record_journal_decision,
                )

                # A flag-driven record is a machine's reading of a diff, so it
                # enters the journal unconfirmed and the git diff is what a
                # human reviews. Refusing it instead — which this did — left
                # an agent with no way to propose at all, while the MCP tool
                # proposed freely against the same file. `confirm` remains the
                # only proposed→active transition either way.
                rec = await record_journal_decision(
                    session,
                    repo.id,
                    title=title,
                    decision=decision_text,
                    why=rationale or "",
                    anchors=[{"file": file, "symbol": None} for file in affected_files],
                    confirmed=not non_interactive,
                )
            else:
                rec = await upsert_decision(
                    session,
                    repository_id=repo.id,
                    title=title,
                    status=status,
                    context=context or "",
                    decision=decision_text,
                    rationale=rationale or "",
                    alternatives=alternatives_list,
                    consequences=consequences_list,
                    affected_files=affected_files,
                    affected_modules=[],
                    tags=tags_list,
                    source="cli",
                    confidence=1.0,
                )
            # The stored status, not the one asked for: in journal mode it is
            # the projection that decides, from whether ``confirmed_at`` is set.
            decision_id, stored_status = rec.id, rec.status

        await engine.dispose()
        return decision_id, stored_status

    decision_id, status = run_async(_persist())

    if not journal_mode:
        # The derived store is rebuilt from scratch by any reindex and is not
        # tracked by git, so a row written here reaches nobody else. Said out
        # loud because the command looks identical in both modes and the
        # difference only shows up as a decision that quietly went missing.
        notice_console(fmt).print(
            "[yellow]Machine-local:[/yellow] this row went to .repowise (derived, "
            "gitignored, rebuilt on reindex). Set REPOWISE_DECISIONS_JOURNAL to "
            "record decisions in a git-tracked journal instead."
        )

    if fmt == "json":
        # The full id, not the table's 8-char prefix — a caller that parses
        # this is about to pass it back to `confirm` or `show`.
        emit_json(
            {
                "repo": str(repo_path),
                "decision": {"id": decision_id, "title": title, "status": status},
            }
        )
        return
    console.print(
        f"\n[green]Decision recorded[/green] [dim]({status})[/dim] — "
        f"ID: [bold]{decision_id[:8]}[/bold]"
    )


# ---------------------------------------------------------------------------
# decision list
# ---------------------------------------------------------------------------


@decision_group.command("list")
@click.argument("path", required=False, default=None)
@click.option(
    "--status",
    type=click.Choice(["proposed", "active", "deprecated", "superseded", "dismissed", "all"]),
    default="all",
)
@click.option(
    "--source",
    type=click.Choice(_SOURCE_CHOICES),
    default="all",
)
@click.option("--proposed", is_flag=True, default=False, help="Show only proposed decisions.")
@click.option("--stale-only", is_flag=True, default=False, help="Show only stale decisions.")
@repo_option()
@format_option()
def decision_list(
    path: str | None,
    status: str,
    source: str,
    proposed: bool,
    stale_only: bool,
    repo_alias: str | None,
    fmt: str,
) -> None:
    """List architectural decision records."""
    repo_path = _resolve_decision_repo(path, fmt, repo_alias)

    async def _query() -> list:
        from repowise.core.persistence import (
            create_engine,
            create_session_factory,
            get_session,
            init_db,
            list_decisions,
            upsert_repository,
        )

        url = get_db_url_for_repo(repo_path)
        engine = create_engine(url)
        await init_db(engine)
        sf = create_session_factory(engine)

        async with get_session(sf) as session:
            repo = await upsert_repository(session, name=repo_path.name, local_path=str(repo_path))
            if _journal_mode_enabled():
                from repowise.core.analysis.decisions.journal_projection import (
                    refresh_decision_journal,
                )

                await refresh_decision_journal(session, repo.id, repo_root=repo_path)
            decisions = await list_decisions(
                session,
                repo.id,
                status=status if status != "all" else ("proposed" if proposed else None),
                source=source if source != "all" else None,
                include_proposed=True,
                limit=100,
            )

        await engine.dispose()
        return decisions

    decisions = run_async(_query())

    if proposed:
        decisions = [d for d in decisions if d.status == "proposed"]
    if stale_only:
        decisions = [d for d in decisions if d.staleness_score >= 0.5]

    if fmt == "json":
        emit_json(
            {
                "repo": str(repo_path),
                "decisions": [
                    {
                        # Full id, not the table's 8-char prefix: the prefix
                        # exists to fit a column, and every id-taking
                        # subcommand accepts either.
                        "id": d.id,
                        "title": d.title,
                        "status": d.status,
                        "source": d.source,
                        "confidence": d.confidence,
                        "staleness_score": d.staleness_score,
                        "created_at": d.created_at.isoformat() if d.created_at else None,
                    }
                    for d in decisions
                ],
            }
        )
        return

    if not decisions:
        console.print("[dim]No decisions found.[/dim]")
        return

    table = Table(title="Architectural Decisions")
    table.add_column("ID", style="dim", width=8)
    table.add_column("Title", max_width=40)
    table.add_column("Status")
    table.add_column("Source", style="dim")
    table.add_column("Conf.", justify="right")
    table.add_column("Stale", justify="right")
    table.add_column("Created", style="dim")

    status_colors = {
        "active": "green",
        "proposed": "yellow",
        "deprecated": "red",
        "superseded": "dim",
        "dismissed": "dim",
    }

    for d in decisions:
        color = status_colors.get(d.status, "white")
        stale_str = f"{d.staleness_score:.1f}" if d.staleness_score > 0 else "-"
        created = d.created_at.strftime("%Y-%m-%d") if d.created_at else ""
        table.add_row(
            d.id[:8],
            d.title[:40],
            f"[{color}]{d.status}[/{color}]",
            d.source,
            f"{d.confidence:.0%}",
            stale_str,
            created,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# decision show
# ---------------------------------------------------------------------------


@decision_group.command("show")
@click.argument("decision_id")
@click.argument("path", required=False, default=None)
@repo_option()
@format_option()
def decision_show(decision_id: str, path: str | None, repo_alias: str | None, fmt: str) -> None:
    """Show full details of a decision record."""
    repo_path = _resolve_decision_repo(path, fmt, repo_alias)
    journal_mode = _journal_mode_enabled()

    async def _query():
        from repowise.core.persistence import (
            create_engine,
            create_session_factory,
            get_decision,
            get_session,
            init_db,
        )

        url = get_db_url_for_repo(repo_path)
        engine = create_engine(url)
        await init_db(engine)
        sf = create_session_factory(engine)

        async with get_session(sf) as session:
            if journal_mode:
                from repowise.core.analysis.decisions.journal_projection import (
                    refresh_decision_journal,
                )
                from repowise.core.persistence import upsert_repository

                repo = await upsert_repository(
                    session, name=repo_path.name, local_path=str(repo_path)
                )
                await refresh_decision_journal(session, repo.id, repo_root=repo_path)
            full_id = await _resolve_decision_id(session, decision_id)
            rec = await get_decision(session, full_id) if full_id else None

        await engine.dispose()
        return rec

    rec = run_async(_query())
    if rec is None:
        notice_console(fmt).print(f"[red]Decision not found: {decision_id}[/red]")
        if fmt == "json":
            emit_json({"query": decision_id, "decision": None})
        return

    if fmt == "json":
        emit_json(
            {
                "query": decision_id,
                "decision": {
                    "id": rec.id,
                    "title": rec.title,
                    "status": rec.status,
                    "source": rec.source,
                    "confidence": rec.confidence,
                    "staleness_score": rec.staleness_score,
                    "created_at": rec.created_at.isoformat() if rec.created_at else None,
                    "currency": describe_decision_currency(
                        repo_path,
                        created_at=rec.created_at,
                        nodes=json.loads(rec.affected_files_json or "[]"),
                    ),
                    "context": rec.context,
                    "decision": rec.decision,
                    "rationale": rec.rationale,
                    "alternatives": json.loads(rec.alternatives_json),
                    "consequences": json.loads(rec.consequences_json),
                    # Not clipped to 10 the way the panel clips it: the panel
                    # clips to stay readable, and a caller asking for json is
                    # asking for the record, not a summary of it.
                    "affected_files": json.loads(rec.affected_files_json),
                    "tags": json.loads(rec.tags_json),
                    "evidence_file": rec.evidence_file,
                    "evidence_line": rec.evidence_line,
                },
            }
        )
        return

    lines = [
        f"[bold]{rec.title}[/bold]",
        f"Status: {rec.status}  |  Source: {rec.source}  |  Confidence: {rec.confidence:.0%}",
        f"Staleness: {rec.staleness_score:.2f}",
    ]
    # The stored score is a proportion; this is the fact behind it, asked of
    # git at read time. `show` is one record on demand, which is exactly where
    # a subprocess is affordable — nothing on the hook or update path may do
    # this. None means git could not decide, and then we say nothing.
    currency = describe_decision_currency(
        repo_path,
        created_at=rec.created_at,
        nodes=json.loads(rec.affected_files_json or "[]"),
    )
    if currency:
        lines.append(f"[dim]{currency}[/dim]")
    lines.append("")
    if rec.context:
        lines.append(f"[cyan]Context:[/cyan] {rec.context}")
    if rec.decision:
        lines.append(f"[cyan]Decision:[/cyan] {rec.decision}")
    if rec.rationale:
        lines.append(f"[cyan]Rationale:[/cyan] {rec.rationale}")

    alternatives = json.loads(rec.alternatives_json)
    if alternatives:
        lines.append("[cyan]Alternatives rejected:[/cyan]")
        for a in alternatives:
            lines.append(f"  - {a}")

    consequences = json.loads(rec.consequences_json)
    if consequences:
        lines.append("[cyan]Consequences:[/cyan]")
        for c in consequences:
            lines.append(f"  - {c}")

    affected = json.loads(rec.affected_files_json)
    if affected:
        lines.append(f"[cyan]Affected files:[/cyan] {', '.join(affected[:10])}")

    tags = json.loads(rec.tags_json)
    if tags:
        lines.append(f"[cyan]Tags:[/cyan] {', '.join(tags)}")

    if rec.evidence_file:
        loc = rec.evidence_file
        if rec.evidence_line:
            loc += f":{rec.evidence_line}"
        lines.append(f"[cyan]Evidence:[/cyan] {loc}")

    console.print(Panel("\n".join(lines), title=f"Decision {rec.id[:8]}"))


# ---------------------------------------------------------------------------
# decision confirm
# ---------------------------------------------------------------------------


@decision_group.command("confirm")
@click.argument("decision_id")
@click.argument("path", required=False, default=None)
@repo_option()
def decision_confirm(decision_id: str, path: str | None, repo_alias: str | None) -> None:
    """Confirm a proposed decision (set status to active)."""
    repo_path = _resolve_decision_repo(path, repo_alias=repo_alias)
    journal_mode = _journal_mode_enabled()

    async def _update():
        from repowise.core.persistence import (
            create_engine,
            create_session_factory,
            get_session,
            init_db,
            update_decision_status,
        )

        url = get_db_url_for_repo(repo_path)
        engine = create_engine(url)
        await init_db(engine)
        sf = create_session_factory(engine)

        async with get_session(sf) as session:
            journal_repo_id: str | None = None
            if journal_mode:
                from repowise.core.analysis.decisions.journal_projection import (
                    confirm_journal_decision,
                    refresh_decision_journal,
                )
                from repowise.core.persistence import upsert_repository

                repo = await upsert_repository(
                    session, name=repo_path.name, local_path=str(repo_path)
                )
                journal_repo_id = repo.id
                await refresh_decision_journal(session, repo.id, repo_root=repo_path)
            full_id = await _resolve_decision_id(session, decision_id)
            if full_id and journal_mode:
                assert journal_repo_id is not None
                rec = await confirm_journal_decision(session, journal_repo_id, full_id)
            else:
                rec = await update_decision_status(session, full_id, "active") if full_id else None

        await engine.dispose()
        return rec

    rec = run_async(_update())
    if rec is None:
        console.print(f"[red]Decision not found: {decision_id}[/red]")
    else:
        console.print(f"[green]Decision {rec.id[:8]} confirmed (active)[/green]")


# ---------------------------------------------------------------------------
# decision dismiss
# ---------------------------------------------------------------------------


@decision_group.command("dismiss")
@click.argument("decision_id")
@click.argument("path", required=False, default=None)
@repo_option()
def decision_dismiss(decision_id: str, path: str | None, repo_alias: str | None) -> None:
    """Dismiss a proposed decision (kept as a tombstone; never re-proposed)."""
    repo_path = _resolve_decision_repo(path, repo_alias=repo_alias)

    if _journal_mode_enabled():
        raise click.ClickException(
            "Dismiss is disabled in decision journal mode because the canonical "
            "format has no dismissed status"
        )

    if not click.confirm(f"Dismiss decision {decision_id[:8]}?"):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    async def _dismiss():
        from repowise.core.persistence import (
            create_engine,
            create_session_factory,
            get_session,
            init_db,
            update_decision_status,
        )

        url = get_db_url_for_repo(repo_path)
        engine = create_engine(url)
        await init_db(engine)
        sf = create_session_factory(engine)

        async with get_session(sf) as session:
            full_id = await _resolve_decision_id(session, decision_id)
            rec = await update_decision_status(session, full_id, "dismissed") if full_id else None

        await engine.dispose()
        return rec

    rec = run_async(_dismiss())
    if rec is not None:
        console.print(
            f"[green]Decision {rec.id[:8]} dismissed[/green] "
            "[dim](kept as a tombstone; reindexing will not re-propose it)[/dim]"
        )
    else:
        console.print(f"[red]Decision not found: {decision_id}[/red]")


# ---------------------------------------------------------------------------
# decision deprecate
# ---------------------------------------------------------------------------


@decision_group.command("deprecate")
@click.argument("decision_id")
@click.argument("path", required=False, default=None)
@click.option("--superseded-by", default=None, help="ID of the decision that replaces this one.")
@repo_option()
def decision_deprecate(
    decision_id: str, path: str | None, superseded_by: str | None, repo_alias: str | None
) -> None:
    """Deprecate an active decision."""
    repo_path = _resolve_decision_repo(path, repo_alias=repo_alias)

    if _journal_mode_enabled():
        raise click.ClickException(
            "Deprecate is disabled in decision journal mode; record or select a "
            "successor and use canonical supersession instead"
        )

    async def _update():
        from repowise.core.persistence import (
            create_engine,
            create_session_factory,
            get_session,
            init_db,
            update_decision_status,
        )

        url = get_db_url_for_repo(repo_path)
        engine = create_engine(url)
        await init_db(engine)
        sf = create_session_factory(engine)

        async with get_session(sf) as session:
            full_id = await _resolve_decision_id(session, decision_id)
            rec = (
                await update_decision_status(
                    session, full_id, "deprecated", superseded_by=superseded_by
                )
                if full_id
                else None
            )

        await engine.dispose()
        return rec

    rec = run_async(_update())
    if rec is None:
        console.print(f"[red]Decision not found: {decision_id}[/red]")
    else:
        console.print(f"[yellow]Decision {rec.id[:8]} deprecated.[/yellow]")


# ---------------------------------------------------------------------------
# decision health
# ---------------------------------------------------------------------------


@decision_group.command("health")
@click.argument("path", required=False, default=None)
@repo_option()
@format_option()
def decision_health(path: str | None, repo_alias: str | None, fmt: str) -> None:
    """Show decision health: stale decisions, proposed, ungoverned hotspots."""
    repo_path = _resolve_decision_repo(path, fmt, repo_alias)

    async def _query():
        from repowise.core.persistence import (
            create_engine,
            create_session_factory,
            get_decision_health_summary,
            get_session,
            init_db,
            upsert_repository,
        )

        url = get_db_url_for_repo(repo_path)
        engine = create_engine(url)
        await init_db(engine)
        sf = create_session_factory(engine)

        async with get_session(sf) as session:
            repo = await upsert_repository(session, name=repo_path.name, local_path=str(repo_path))
            journal_health = None
            if _journal_mode_enabled():
                from repowise.core.analysis.decisions.journal_projection import (
                    refresh_decision_journal,
                )

                journal_health = await refresh_decision_journal(
                    session, repo.id, repo_root=repo_path
                )
            health = await get_decision_health_summary(session, repo.id)

        await engine.dispose()
        return health, journal_health

    health, journal_health = run_async(_query())
    summary = health["summary"]

    if fmt == "json":
        # The table caps each list (5 stale, 10 hotspots, 5 proposed) to keep
        # the report short; json carries them whole.
        emit_json(
            {
                "repo": str(repo_path),
                "summary": summary,
                "stale_decisions": [
                    {"id": d.id, "title": d.title, "staleness_score": d.staleness_score}
                    for d in health["stale_decisions"]
                ],
                "ungoverned_hotspots": list(health["ungoverned_hotspots"]),
                "proposed_awaiting_review": [
                    {"id": d.id, "title": d.title, "source": d.source}
                    for d in health["proposed_awaiting_review"]
                ],
                "journal": journal_health.to_dict() if journal_health is not None else None,
            }
        )
        return

    console.print("[bold]Decision Health[/bold]\n")

    if journal_health is not None:
        console.print(
            "[dim]Journal:[/dim] "
            f"{journal_health.path}  "
            f"{journal_health.projected_count} projected  "
            f"sha256:{journal_health.content_hash[:12]}  "
            f"lock={'available' if journal_health.lock_acquirable else 'busy'}\n"
        )

    # Summary stats
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Active decisions", str(summary.get("active", 0)))
    stats_table.add_row("Proposed (needs review)", f"[yellow]{summary.get('proposed', 0)}[/yellow]")
    stats_table.add_row("Stale decisions", f"[red]{summary.get('stale', 0)}[/red]")
    unscoped = summary.get("unscoped", 0)
    if unscoped:
        # Not folded into "stale": these were never checked, which is a
        # different thing from checked and found to have drifted.
        stats_table.add_row("Unscoped (cannot be checked)", f"[yellow]{unscoped}[/yellow]")
    stats_table.add_row("Deprecated", str(summary.get("deprecated", 0)))
    console.print(stats_table)

    # Stale decisions
    stale = health["stale_decisions"]
    if stale:
        console.print(f"\n[red]Stale decisions ({len(stale)}):[/red]")
        for d in stale[:5]:
            console.print(f"  {d.id[:8]}  {d.title[:50]}  (staleness: {d.staleness_score:.2f})")

    # Ungoverned hotspots
    ungoverned = health["ungoverned_hotspots"]
    if ungoverned:
        console.print(f"\n[yellow]Ungoverned hotspots ({len(ungoverned)}):[/yellow]")
        for fp in ungoverned[:10]:
            console.print(f"  {fp}")

    # Proposed
    proposed = health["proposed_awaiting_review"]
    if proposed:
        console.print(f"\n[yellow]Proposed decisions ({len(proposed)}):[/yellow]")
        for d in proposed[:5]:
            console.print(f"  {d.id[:8]}  {d.title[:50]}  (source: {d.source})")
