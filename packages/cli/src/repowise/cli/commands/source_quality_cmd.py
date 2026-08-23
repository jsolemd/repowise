"""``repowise source-quality`` — the CLI adapter over the ``get_query_quality`` tool.

Three verbs over one file, ``.repowise/source_search/query_log.jsonl``:

* ``report`` aggregates it into the four quality buckets.
* ``export`` turns one bucket into a fixture suite.
* ``run`` executes a suite and exits non-zero if any case failed.

Adapters, not reimplementations: each one calls the MCP tool through
``tool_bridge`` the way ``repowise search`` calls ``search_codebase``, so a CLI
answer and an MCP answer are the same answer. The alternative — reaching past
the tool into ``core.source_search.query_report`` — would give the report two
owners, and the one that drifts is always the one nobody is looking at.

``run`` is the only verb that needs a working index. ``report`` and ``export``
read a text file, which is what makes them usable on a repository whose index
is broken — exactly when the error bucket is worth reading.
"""

from __future__ import annotations

import click

from repowise.cli.commands import _tool_adapters as _ta
from repowise.cli.output import emit_json
from repowise.core.source_search.query_report import BUCKETS, INTENTS, TIME_WINDOWS


@click.group("source-quality")
def source_quality_group() -> None:
    """Read the source-search query log: quality buckets and eval fixtures."""


def _repo_option(fn):
    return click.option(
        "--path",
        "path",
        default=None,
        type=click.Path(exists=True, file_okay=False),
        help="Repo root. Defaults to the current directory.",
    )(fn)


def _call(path: str | None, **kwargs) -> dict:
    from repowise.cli.helpers import resolve_repo_path

    repo_path = resolve_repo_path(path)

    def _factory():
        from repowise.server.mcp_server.tool_query_report import get_query_quality

        return get_query_quality(**kwargs)

    return _ta.run(repo_path, _factory, "get_query_quality")


@source_quality_group.command("report")
@_repo_option
@click.option("--window", type=click.Choice(TIME_WINDOWS), default="day", help="Trend granularity.")
@click.option(
    "--offenders",
    type=click.IntRange(1, 200),
    default=10,
    help="Worst offenders listed per bucket.",
)
@click.option(
    "--verdicts",
    "verdicts",
    default=None,
    metavar="NAME.json",
    help=(
        "JSON object mapping query text to the correct owner file, read from "
        ".repowise/source_search/eval/. Turns wrong-owner from an inference "
        "into a measurement."
    ),
)
def report_command(path: str | None, window: str, offenders: int, verdicts: str | None) -> None:
    """Aggregate the query log into the four quality buckets.

    Counts, rates, a trend, and the worst offenders per bucket. Everything the
    pass could not account for is reported under "caveats" rather than dropped:
    a quality report that quietly excluded the rows it could not parse would be
    measuring the wrong denominator in exactly the situation it exists to
    detect.
    """
    emit_json(_call(path, mode="report", window=window, limit=offenders, verdicts=verdicts))


@source_quality_group.command("export")
@_repo_option
@click.argument("bucket", type=click.Choice(BUCKETS))
@click.option(
    "--intent",
    type=click.Choice(INTENTS),
    default="goal",
    help=(
        "'goal' asserts the behaviour after the fix (red until retrieval "
        "improves); 'guard' asserts today's behaviour as a floor (green now, "
        "red on regression)."
    ),
)
@click.option(
    "--limit",
    type=click.IntRange(1, 5000),
    default=None,
    help="Cases to export. Most-repeated queries first.",
)
@click.option(
    "--out",
    "out",
    default=None,
    metavar="NAME.json",
    help=(
        "Write the suite under .repowise/source_search/eval/ instead of to "
        "stdout. Writes are confined to that directory."
    ),
)
def export_command(
    path: str | None,
    bucket: str,
    intent: str,
    limit: int | None,
    out: str | None,
) -> None:
    """Turn one bucket into a runnable fixture suite.

    The exporter never invents ground truth. A case whose right answer the log
    does not establish is emitted with expect.kind "todo" and reports as
    pending, neither a pass nor a failure, until a human answers it. Writing the
    observed owner into the expectation would freeze the defect into a test that
    passes forever because it asserts the bug.
    """
    emit_json(_call(path, mode="export", bucket=bucket, intent=intent, limit=limit, write_to=out))


@source_quality_group.command("run")
@_repo_option
@click.argument("suite", metavar="NAME.json")
def run_command(path: str | None, suite: str) -> None:
    """Execute a fixture suite against this repository's source index.

    Exits 1 if any case failed or errored. Pending cases never fail the run: an
    unanswered question is not a defect and must not gate a build.
    """
    payload = _call(path, mode="run", suite=suite)
    emit_json(payload)
    run = payload.get("run")
    if payload.get("error") or (isinstance(run, dict) and run.get("failed")):
        raise SystemExit(1)
