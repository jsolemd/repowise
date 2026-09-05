"""Savings ledger — SQL for recording and summarizing distillation events.

Operates on the omissions sidecar connection (see ``store.py``); kept as free
functions so future surfaces (hook script, MCP budgeter) can record savings
without instantiating a full :class:`~repowise.core.distill.store.OmissionStore`.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

#: MCP usage is operational telemetry, not an activity ledger. One row per
#: tool/day is retained for this many inclusive UTC calendar days.
MCP_USAGE_RETENTION_DAYS = 30


def _usage_table_exists(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mcp_usage_daily'"
        ).fetchone()
        is not None
    )


def _usage_cutoff_day(now: float | None = None) -> str:
    current = datetime.fromtimestamp(now if now is not None else time.time(), UTC).date()
    return (current - timedelta(days=MCP_USAGE_RETENTION_DAYS - 1)).isoformat()


def migrate_legacy_mcp_savings(conn: sqlite3.Connection) -> None:
    """Collapse old per-call ``mcp:*`` savings rows, then delete them.

    The insert and delete share one transaction, so concurrent server workers
    can neither duplicate nor lose a legacy row. Reopening the store is
    idempotent: after the first successful migration there is nothing to move.
    """
    if not _usage_table_exists(conn):
        return
    cutoff = _usage_cutoff_day()
    has_legacy = conn.execute("SELECT 1 FROM savings WHERE source LIKE 'mcp:%' LIMIT 1").fetchone()
    has_expired = conn.execute(
        "SELECT 1 FROM mcp_usage_daily WHERE day < ? LIMIT 1", (cutoff,)
    ).fetchone()
    if has_legacy is None and has_expired is None:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            """
            SELECT created_at, source, raw_tokens, distilled_tokens
            FROM savings WHERE source LIKE 'mcp:%'
            ORDER BY id
            """
        ).fetchall()
        grouped: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * 6)
        for created_at, source, raw_tokens, distilled_tokens in rows:
            tool = str(source).removeprefix("mcp:").removesuffix(":dead_end")
            if not tool:
                continue
            day = datetime.fromtimestamp(float(created_at), UTC).date().isoformat()
            saved = int(raw_tokens) - int(distilled_tokens)
            stats = grouped[(day, tool)]
            stats[0] += 1  # calls represented by the legacy savings event
            stats[1] += int(str(source).endswith(":dead_end"))
            stats[2] += 1  # saving_calls
            stats[3] += int(saved > 0)  # positive_saving_calls
            stats[4] += int(raw_tokens)
            stats[5] += int(distilled_tokens)

        for (day, tool), stats in grouped.items():
            conn.execute(
                """
                INSERT INTO mcp_usage_daily
                    (day, tool, calls, error_calls, saving_calls,
                     positive_saving_calls, replaced_tokens, delivered_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(day, tool) DO UPDATE SET
                    calls = calls + excluded.calls,
                    error_calls = error_calls + excluded.error_calls,
                    saving_calls = saving_calls + excluded.saving_calls,
                    positive_saving_calls = positive_saving_calls
                        + excluded.positive_saving_calls,
                    replaced_tokens = replaced_tokens + excluded.replaced_tokens,
                    delivered_tokens = delivered_tokens + excluded.delivered_tokens
                """,
                (day, tool, stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]),
            )
        conn.execute("DELETE FROM savings WHERE source LIKE 'mcp:%'")
        conn.execute("DELETE FROM mcp_usage_daily WHERE day < ?", (cutoff,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def record_mcp_usage(
    conn: sqlite3.Connection,
    *,
    tool: str,
    duration_ms: int,
    error: bool,
    no_match: bool,
    degraded: bool,
    replaced_tokens: int,
    delivered_tokens: int,
    occurred_at: float | None = None,
) -> None:
    """Fold one MCP call into a fixed-cardinality daily aggregate.

    No query, target, repository path, session id, or individual event is
    stored. The table can contain at most roughly ``days × tool names`` rows.
    Calls without a defensible counterfactual still contribute usage/outcome
    counts, while their token-saving fields remain zero.
    """
    stamped = occurred_at if occurred_at is not None else time.time()
    day = datetime.fromtimestamp(stamped, UTC).date().isoformat()
    duration = max(0, int(duration_ms))
    replaced = max(0, int(replaced_tokens))
    delivered = max(0, int(delivered_tokens))
    saving = replaced > 0 or (error and delivered > 0)
    positive = saving and replaced > delivered
    counted_replaced = replaced if saving else 0
    counted_delivered = delivered if saving else 0
    with conn:
        conn.execute(
            """
            INSERT INTO mcp_usage_daily
                (day, tool, calls, error_calls, no_match_calls, degraded_calls,
                 duration_ms_total, saving_calls, positive_saving_calls,
                 replaced_tokens, delivered_tokens)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(day, tool) DO UPDATE SET
                calls = calls + 1,
                error_calls = error_calls + excluded.error_calls,
                no_match_calls = no_match_calls + excluded.no_match_calls,
                degraded_calls = degraded_calls + excluded.degraded_calls,
                duration_ms_total = duration_ms_total + excluded.duration_ms_total,
                saving_calls = saving_calls + excluded.saving_calls,
                positive_saving_calls = positive_saving_calls
                    + excluded.positive_saving_calls,
                replaced_tokens = replaced_tokens + excluded.replaced_tokens,
                delivered_tokens = delivered_tokens + excluded.delivered_tokens
            """,
            (
                day,
                tool,
                int(error),
                int(no_match),
                int(degraded),
                duration,
                int(saving),
                int(positive),
                counted_replaced,
                counted_delivered,
            ),
        )
        conn.execute("DELETE FROM mcp_usage_daily WHERE day < ?", (_usage_cutoff_day(stamped),))


def mcp_usage_summary(conn: sqlite3.Connection, *, since: float | None = None) -> dict[str, Any]:
    """Return retained local MCP usage totals and a per-tool breakdown."""
    empty = {
        "calls": 0,
        "error_calls": 0,
        "no_match_calls": 0,
        "degraded_calls": 0,
        "duration_ms_total": 0,
        "avg_duration_ms": 0.0,
        "window_days": MCP_USAGE_RETENTION_DAYS,
        "per_tool": [],
    }
    if not _usage_table_exists(conn):
        return empty
    where = ""
    params: tuple[float, ...] = ()
    if since is not None:
        where = " WHERE day >= date(?, 'unixepoch')"
        params = (since,)
    rows = conn.execute(
        """
        SELECT tool, SUM(calls), SUM(error_calls), SUM(no_match_calls),
               SUM(degraded_calls), SUM(duration_ms_total),
               SUM(saving_calls), SUM(positive_saving_calls),
               SUM(replaced_tokens - delivered_tokens)
        FROM mcp_usage_daily
        """
        + where
        + " GROUP BY tool ORDER BY SUM(calls) DESC, tool ASC",
        params,
    ).fetchall()
    per_tool = [
        {
            "tool": row[0],
            "calls": row[1],
            "error_calls": row[2],
            "no_match_calls": row[3],
            "degraded_calls": row[4],
            "duration_ms_total": row[5],
            "avg_duration_ms": round(row[5] / row[1], 1) if row[1] else 0.0,
            "saving_calls": row[6],
            "positive_saving_calls": row[7],
            "saved_tokens": row[8],
        }
        for row in rows
    ]
    calls = sum(row["calls"] for row in per_tool)
    duration = sum(row["duration_ms_total"] for row in per_tool)
    return {
        "calls": calls,
        "error_calls": sum(row["error_calls"] for row in per_tool),
        "no_match_calls": sum(row["no_match_calls"] for row in per_tool),
        "degraded_calls": sum(row["degraded_calls"] for row in per_tool),
        "duration_ms_total": duration,
        "avg_duration_ms": round(duration / calls, 1) if calls else 0.0,
        "window_days": MCP_USAGE_RETENTION_DAYS,
        "per_tool": per_tool,
    }


def record_saving(
    conn: sqlite3.Connection,
    *,
    filter_name: str,
    source: str,
    command: str | None,
    raw_tokens: int,
    distilled_tokens: int,
) -> None:
    """Append one distillation event to the savings ledger."""
    conn.execute(
        """
        INSERT INTO savings
            (created_at, filter, source, command, raw_tokens, distilled_tokens)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (time.time(), filter_name, source, command, raw_tokens, distilled_tokens),
    )
    conn.commit()


def savings_summary(conn: sqlite3.Connection, *, since: float | None = None) -> dict[str, Any]:
    """Aggregate ledger totals, overall and per filter.

    *since* is a Unix timestamp; only events at or after it are counted.
    """
    rows = savings_rollup(conn, by="filter", since=since)
    per_filter = {
        row["group"]: {
            key: row[key] for key in ("events", "raw_tokens", "distilled_tokens", "saved_tokens")
        }
        for row in rows
    }
    events = sum(row["events"] for row in rows)
    total_raw = sum(row["raw_tokens"] for row in rows)
    total_distilled = sum(row["distilled_tokens"] for row in rows)
    return {
        "events": events,
        "raw_tokens": total_raw,
        "distilled_tokens": total_distilled,
        "saved_tokens": total_raw - total_distilled,
        "per_filter": per_filter,
    }


def distill_summary(conn: sqlite3.Connection, *, since: float | None = None) -> dict[str, Any]:
    """Ledger totals for the **distill** surface only (excludes ``mcp:*``).

    Identical in shape to :func:`savings_summary` but scoped to non-MCP
    sources, so the hero card can report a clean ``repowise distill`` figure now
    that Phase 2 also writes counterfactual ``mcp:<tool>`` rows into the same
    ``savings`` ledger. Per-filter buckets likewise drop MCP tool rows.

    *since* is a Unix timestamp lower bound on ``created_at``.
    """
    where = "source NOT LIKE 'mcp:%'"
    params: tuple[float, ...] = ()
    if since is not None:
        where += " AND created_at >= ?"
        params = (since,)
    total_raw, total_distilled, events = conn.execute(
        "SELECT COALESCE(SUM(raw_tokens),0), COALESCE(SUM(distilled_tokens),0),"
        f" COUNT(*) FROM savings WHERE {where}",
        params,
    ).fetchone()
    per_filter = {
        row[0]: {
            "events": row[1],
            "raw_tokens": row[2],
            "distilled_tokens": row[3],
            "saved_tokens": row[2] - row[3],
        }
        for row in conn.execute(
            "SELECT filter, COUNT(*), SUM(raw_tokens), SUM(distilled_tokens)"
            f" FROM savings WHERE {where} GROUP BY filter ORDER BY SUM(raw_tokens) DESC",
            params,
        )
    }
    return {
        "events": events,
        "raw_tokens": total_raw,
        "distilled_tokens": total_distilled,
        "saved_tokens": total_raw - total_distilled,
        "per_filter": per_filter,
    }


def mcp_savings_summary(conn: sqlite3.Connection, *, since: float | None = None) -> dict[str, Any]:
    """Unified MCP savings view — counterfactual ledger, truncation as fallback.

    Two MCP signals live in the sidecar:

    * **Counterfactual** rows in the ``savings`` ledger (``source='mcp:<tool>'``)
      written by the Phase 2 instrumentation: ``saved = replaced - delivered``.
      Because ``delivered`` is measured *after* response-budget truncation, the
      truncation saving is already folded into this delta.
    * **Truncation drops** in the ``omissions`` table (also ``source='mcp:<tool>'``)
      from :func:`mcp_drops_summary` — the only signal for tools that have no
      counterfactual estimator yet.

    Merging per tool with **counterfactual precedence** avoids double counting:
    a tool with counterfactual rows reports its ledger ``saved_tokens`` (which
    subsumes truncation); a tool with only drops reports its dropped tokens.
    Each ``per_tool`` row is tagged ``kind`` = ``"counterfactual"`` | ``"truncation"``.

    Returns ``{events, tokens, queries, per_tool}`` where ``tokens`` is total
    saved, ``queries`` counts counterfactual tool calls (the "N MCP queries
    answered" headline), and ``events`` counts every contributing event.
    """
    ledger: dict[str, dict[str, int]] = {}
    for row in savings_rollup(conn, by="source", since=since):
        if not row["group"].startswith("mcp:"):
            continue
        tool = _strip_mcp_prefix(row["group"])
        bucket = ledger.setdefault(tool, {"events": 0, "saved_tokens": 0})
        bucket["events"] += row["events"]
        bucket["saved_tokens"] += row["saved_tokens"]
    drops = mcp_drops_summary(conn, since=since)["per_tool"]

    per_tool: list[dict[str, Any]] = []
    queries = 0
    for tool, row in ledger.items():
        per_tool.append(
            {
                "tool": tool,
                "events": row["events"],
                "tokens": row["saved_tokens"],
                "kind": "counterfactual",
            }
        )
        queries += row["events"]
    for tool, stats in drops.items():
        if tool in ledger:
            continue  # counterfactual already subsumes this tool's truncation
        per_tool.append(
            {
                "tool": tool,
                "events": stats["events"],
                "tokens": stats["tokens"],
                "kind": "truncation",
            }
        )

    per_tool.sort(key=lambda r: r["tokens"], reverse=True)
    return {
        "events": sum(r["events"] for r in per_tool),
        "tokens": sum(r["tokens"] for r in per_tool),
        "queries": queries,
        "per_tool": per_tool,
    }


def mcp_drops_summary(conn: sqlite3.Connection, *, since: float | None = None) -> dict[str, Any]:
    """Truncation savings the MCP server already wrote to the omissions store.

    MCP tools drop content past their response budget into the ``omissions``
    table under ``source='mcp:<tool>'`` but never call
    :func:`record_saving`, so these savings are invisible to
    :func:`savings_summary`. This reads them straight from ``omissions``:
    total dropped tokens plus a per-tool rollup (with the ``mcp:`` prefix
    stripped). It is the *truncation-only* view — Phase 2 additionally records
    counterfactual ``mcp:*`` rows into the ``savings`` ledger.

    *since* is a Unix timestamp lower bound on ``created_at``.
    """
    where = "source LIKE 'mcp:%'"
    params: tuple[float, ...] = ()
    if since is not None:
        where += " AND created_at >= ?"
        params = (since,)
    events, tokens = conn.execute(
        f"SELECT COUNT(*), COALESCE(SUM(original_tokens), 0) FROM omissions WHERE {where}",
        params,
    ).fetchone()
    per_tool = {
        _strip_mcp_prefix(row[0]): {"events": row[1], "tokens": row[2]}
        for row in conn.execute(
            "SELECT source, COUNT(*), COALESCE(SUM(original_tokens), 0)"
            f" FROM omissions WHERE {where} GROUP BY source ORDER BY SUM(original_tokens) DESC",
            params,
        )
    }
    return {"events": events, "tokens": tokens, "per_tool": per_tool}


def _strip_mcp_prefix(source: str) -> str:
    """``mcp:get_risk`` → ``get_risk`` (passthrough for anything else)."""
    prefix = "mcp:"
    stripped = source[len(prefix) :] if source.startswith(prefix) else source
    return stripped.removesuffix(":dead_end")


#: Grouping dimensions accepted by :func:`savings_rollup`. ``day`` buckets by
#: the event's local calendar date; ``filter``/``source`` group on the raw
#: ledger columns.
ROLLUP_DIMENSIONS: tuple[str, ...] = ("filter", "day", "source")

_ROLLUP_COLUMNS = {
    "filter": "filter",
    "source": "source",
    "day": "date(created_at, 'unixepoch', 'localtime')",
}


def savings_rollup(
    conn: sqlite3.Connection,
    *,
    by: str = "filter",
    since: float | None = None,
) -> list[dict[str, Any]]:
    """Grouped ledger totals — one row per *by* bucket.

    *by* is one of :data:`ROLLUP_DIMENSIONS`. Rows carry ``group``,
    ``events``, ``raw_tokens``, ``distilled_tokens``, ``saved_tokens``.
    ``day`` rollups are ordered chronologically; the rest by tokens saved,
    descending. *since* is a Unix timestamp lower bound.
    """
    if by not in _ROLLUP_COLUMNS:
        raise ValueError(f"Unknown rollup dimension {by!r}; expected one of {ROLLUP_DIMENSIONS}")
    group_col = _ROLLUP_COLUMNS[by]
    where = " WHERE created_at >= ?" if since is not None else ""
    params: tuple[float, ...] = (since,) if since is not None else ()
    order = "1 ASC" if by == "day" else "SUM(raw_tokens - distilled_tokens) DESC"
    rows = conn.execute(
        f"SELECT {group_col}, COUNT(*), SUM(raw_tokens), SUM(distilled_tokens)"
        f" FROM savings{where} GROUP BY 1 ORDER BY {order}",
        params,
    ).fetchall()
    combined: dict[str, dict[str, Any]] = {}
    for row in rows:
        combined[row[0]] = {
            "group": row[0],
            "events": row[1],
            "raw_tokens": row[2],
            "distilled_tokens": row[3],
            "saved_tokens": row[2] - row[3],
        }

    # MCP calls use bounded daily aggregates. Project their counterfactual
    # columns back into the historical rollup shape so CLI/API consumers do
    # not lose savings when legacy per-call rows are migrated away.
    if _usage_table_exists(conn):
        usage_where = " WHERE saving_calls > 0"
        usage_params: tuple[float, ...] = ()
        if since is not None:
            usage_where += " AND day >= date(?, 'unixepoch')"
            usage_params = (since,)
        usage_rows = conn.execute(
            """
            SELECT day, tool, SUM(saving_calls), SUM(replaced_tokens),
                   SUM(delivered_tokens)
            FROM mcp_usage_daily
            """
            + usage_where
            + " GROUP BY day, tool",
            usage_params,
        ).fetchall()
        for day, tool, events, raw, distilled in usage_rows:
            group = day if by == "day" else (f"mcp:{tool}" if by == "source" else tool)
            bucket = combined.setdefault(
                group,
                {
                    "group": group,
                    "events": 0,
                    "raw_tokens": 0,
                    "distilled_tokens": 0,
                    "saved_tokens": 0,
                },
            )
            bucket["events"] += events
            bucket["raw_tokens"] += raw
            bucket["distilled_tokens"] += distilled
            bucket["saved_tokens"] += raw - distilled

    result = list(combined.values())
    result.sort(
        key=(
            (lambda item: item["group"])
            if by == "day"
            else (lambda item: (-item["saved_tokens"], item["group"]))
        )
    )
    return result
