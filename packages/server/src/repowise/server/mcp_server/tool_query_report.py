"""MCP tool for retrieval quality: what search got wrong, and a suite to prove it.

Reads ``.repowise/source_search/query_log.jsonl`` — a text file — and nothing
else. It opens no vector store, embeds nothing and runs no query, which is why
it is safe to point at a live repository while indexing is underway.

Three modes. ``report`` says how retrieval is doing; ``export`` turns one bucket
of it into a fixture suite; ``run`` executes a suite written earlier and says
which cases now pass. Report and export touch no index at all, which is what
makes them usable on a repository whose index is broken — exactly when the
error bucket is worth reading. Only ``run`` needs a working index, because
running a query is the whole point of it.

Export can write a suite to disk, and the write is confined to one directory
under the repository. An agent-facing tool that accepted an arbitrary path
would be a file-write primitive wearing a reporting tool's name.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from repowise.core.persistence.database import get_session
from repowise.core.registry import mcp_tool_registry as mcp
from repowise.core.source_search.query_log import default_query_log_path
from repowise.core.source_search.query_report import (
    BUCKETS,
    INTENTS,
    TIME_WINDOWS,
    export_bucket,
    load_suite,
    report_from_path,
    run_suite,
    write_suite,
)
from repowise.server.mcp_server._helpers import (
    _get_repo,
    _resolve_repo_context,
    _unsupported_repo_all,
    attach_ignored_arguments,
    resolve_enum_argument,
)
from repowise.server.mcp_server._meta import build_meta as _build_meta

_MODES = frozenset({"report", "export", "run"})

#: Cases returned inline. A whole bucket can be thousands of queries, and the
#: exact total sits beside the slice so this never hides the size of a problem.
_MAX_INLINE_CASES = 25

#: Lines read from the log in one call. A repository that has been answering
#: queries for months must not be able to turn one tool call into a minutes-long
#: read; truncation is declared in the payload when it happens.
_MAX_LINES = 200_000


def _export_root(repo_path: Path) -> Path:
    """The one directory an exported suite may be written to.

    Not a caller-chosen path: this tool is reachable by an agent, and "write
    JSON wherever you like" is not a capability a reporting tool should hand
    out.
    """
    return repo_path / ".repowise" / "source_search" / "eval"


def _confine(repo_path: Path, requested: str, *, verb: str) -> tuple[Path | None, str | None]:
    """Confine *requested* to the export directory, or refuse it.

    ``resolve()`` before comparing, so ``../`` and a symlink pointing out of
    the tree are both rejected by the same check rather than by two that can
    disagree. The same confinement applies to reads: a tool that would load
    and echo any JSON file on the host is an exfiltration primitive, whatever
    the docstring says it is for.
    """
    raw = requested.strip()
    if not raw or "\x00" in raw:
        return None, "invalid_path"
    root = _export_root(repo_path)
    candidate = Path(raw)
    target = candidate if candidate.is_absolute() else root / candidate
    try:
        resolved = target.resolve()
        resolved.relative_to(root.resolve())
    except (OSError, ValueError):
        return None, f"outside_export_dir: {verb}s are confined to {root.as_posix()}"
    if resolved.suffix != ".json":
        return None, "must_end_in_json"
    return resolved, None


@mcp.tool(default=False)
async def get_query_quality(
    mode: str = "report",
    bucket: str | None = None,
    window: str = "day",
    intent: str = "goal",
    limit: int | None = None,
    write_to: str | None = None,
    suite: str | None = None,
    verdicts: str | None = None,
    repo: str | None = None,
) -> dict[str, Any]:
    """Report what retrieval got wrong, export a bucket as an eval suite, or run one.

    ``report`` buckets the source-search query log into no-match,
    low-confidence, wrong-owner and error, with counts, rates, a trend and the
    worst offenders in each. Rows it could not use are declared in ``caveats``,
    never dropped. ``export`` turns one bucket into fixture cases; where the log
    does not establish the right answer the case is ``expect.kind = "todo"`` and
    reports as pending. ``run`` executes a written suite; pending never fails.

    Args:
        mode: ``report`` (default), ``export`` or ``run``.
        bucket: Which bucket to export. Required for export mode.
        window: Trend granularity — ``hour``, ``day`` (default) or ``week``.
        intent: ``goal`` (default) asserts the behaviour after the fix, so the
            suite is red until retrieval improves. ``guard`` asserts today's
            behaviour as a floor: green now, red on regression.
        limit: Offenders per bucket in report mode; cases in export mode.
        write_to: Filename for the exported suite. Writes are confined to the
            repository's ``.repowise/source_search/eval/`` directory.
        suite: Filename of the suite to execute. Required for run mode, read
            from that same directory.
        verdicts: Filename of a JSON object mapping query to correct owner,
            read from that same directory. Turns wrong-owner from an inference
            into a measurement.
        repo: Repository path, name, ID, or workspace alias.
    """
    started = perf_counter()
    if repo == "all":
        return _unsupported_repo_all("get_query_quality")

    ignored: list[dict[str, Any]] = []
    resolved_mode = (
        resolve_enum_argument(mode, _MODES, argument="mode", ignored=ignored) or "report"
    )
    resolved_window = (
        resolve_enum_argument(window, set(TIME_WINDOWS), argument="window", ignored=ignored)
        or "day"
    )
    resolved_intent = (
        resolve_enum_argument(intent, set(INTENTS), argument="intent", ignored=ignored) or "goal"
    )

    ctx = await _resolve_repo_context(repo)
    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)
    repo_path = Path(ctx.path).resolve()

    payload: dict[str, Any] = {
        "mode": resolved_mode,
        "repo": {
            "alias": getattr(ctx, "alias", "default"),
            "id": repository.id,
            "name": repository.name,
            "path": repo_path.as_posix(),
        },
    }

    if resolved_mode == "run":
        await _run_mode(payload, ctx, repo_path, suite)
        attach_ignored_arguments(payload, ignored)
        payload["_meta"] = _build_meta(
            timing_ms=(perf_counter() - started) * 1000.0,
            repository=repository,
        )
        return payload

    loaded_verdicts: dict[str, str | None] | None = None
    if verdicts is not None:
        loaded_verdicts, verdict_error = _load_verdicts(repo_path, verdicts)
        if verdict_error is not None:
            # Refused, not partially applied. A verdict overrides the
            # self-contradiction detector entirely, so a malformed one must
            # change nothing rather than change some rows.
            payload["verdicts"] = {"applied": False, "reason": verdict_error}

    report = report_from_path(
        default_query_log_path(repo_path),
        window=resolved_window,
        offender_limit=limit if (limit and resolved_mode == "report") else 10,
        max_lines=_MAX_LINES,
        verdicts=loaded_verdicts,
    )

    if resolved_mode == "report":
        payload["report"] = report.to_dict()
        if report.parsed.read_error == "log_not_found":
            payload["next_action"] = (
                "No query log exists for this repository yet. It is written by source "
                "search as it answers queries; run some searches and ask again."
            )
    else:
        if bucket is None:
            payload["error"] = "bucket is required when mode='export'"
            payload["valid_buckets"] = list(BUCKETS)
        elif bucket not in BUCKETS:
            payload["error"] = f"unknown bucket {bucket!r}"
            payload["valid_buckets"] = list(BUCKETS)
        else:
            exported = export_bucket(report, bucket, intent=resolved_intent, limit=limit)
            document = exported.to_dict()
            cases = document.pop("cases")
            payload["suite"] = document
            payload["case_count"] = len(cases)
            payload["cases_listed"] = min(len(cases), _MAX_INLINE_CASES)
            payload["cases"] = cases[:_MAX_INLINE_CASES]
            payload["pending_cases"] = sum(1 for case in cases if case["expect"]["kind"] == "todo")
            if write_to is not None:
                target, path_error = _confine(repo_path, write_to, verb="write")
                if path_error is not None:
                    payload["write"] = {"written": False, "reason": path_error}
                else:
                    assert target is not None
                    try:
                        written = write_suite(exported, target)
                    except OSError as exc:
                        payload["write"] = {
                            "written": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    else:
                        payload["write"] = {
                            "written": True,
                            "path": written.as_posix(),
                            "cases": len(cases),
                        }

    attach_ignored_arguments(payload, ignored)
    payload["_meta"] = _build_meta(
        timing_ms=(perf_counter() - started) * 1000.0,
        repository=repository,
    )
    return payload


def _load_verdicts(
    repo_path: Path, requested: str
) -> tuple[dict[str, str | None] | None, str | None]:
    """Read a verdicts file from the export directory, or say why not.

    Every entry is type-checked before any of them is used. A verdict is the
    strongest evidence this tool accepts, so a file with one bad row is
    refused whole rather than applied in part.
    """
    target, path_error = _confine(repo_path, requested, verb="read")
    if path_error is not None:
        return None, path_error
    assert target is not None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "verdicts file must be a JSON object of query -> owner"
    for key, value in payload.items():
        if not isinstance(key, str) or not (value is None or isinstance(value, str)):
            return None, f"verdicts entry {key!r} must map a query to an owner path or null"
    return payload, None


async def _run_mode(
    payload: dict[str, Any],
    ctx: Any,
    repo_path: Path,
    suite: str | None,
) -> None:
    """Execute a written suite against the live source index.

    Fail-soft on a missing coordinator rather than raising: no source index is
    a configuration, and the caller needs to be told *that* rather than handed
    an exception it cannot distinguish from a broken suite.
    """
    if suite is None:
        payload["error"] = "suite is required when mode='run'"
        return
    target, path_error = _confine(repo_path, suite, verb="read")
    if path_error is not None:
        payload["error"] = path_error
        return
    assert target is not None
    try:
        loaded = load_suite(target)
    except (OSError, ValueError) as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return

    from repowise.server.source_search_wiring import context_coordinator, coordinator_lease

    coordinator = await context_coordinator(ctx)
    if coordinator is None:
        payload["error"] = "no source index is available for this repository"
        payload["next_action"] = "Build one with 'repowise source-index', then run again."
        return
    payload["suite_path"] = target.as_posix()
    async with coordinator_lease(coordinator):
        payload["run"] = (await run_suite(loaded, coordinator.search)).to_dict()


__all__ = ["get_query_quality"]
