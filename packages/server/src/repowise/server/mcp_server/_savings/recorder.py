"""Fold MCP usage and counterfactual savings into bounded daily aggregates.

A thin, best-effort bridge from a measured tool call to one ``tool × UTC day``
row in the repo-local omission sidecar. The row contains counts, outcome flags,
latency totals, and token totals only — never query text, targets, paths,
sessions, or individual events. Thirty-day pruning happens in the same write.

The SQLite handle is opened per call and closed immediately, so a long-running
MCP server never holds a WAL handle that would contend with hook-side writers.

Failure posture matches the rest of distill: a failed write degrades to a
silent no-op. Recording savings must never perturb a tool response.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from repowise.core.distill.store import (
    OMISSIONS_DB_FILENAME,
    OMISSIONS_DIRNAME,
    OmissionStore,
)

logger = logging.getLogger(__name__)


def record_mcp_call(
    repo_root: str | Path | None,
    tool: str,
    *,
    duration_ms: int,
    error: bool,
    no_match: bool,
    degraded: bool,
    replaced_tokens: int,
    delivered_tokens: int,
) -> bool:
    """Aggregate one call in its selected repository. Never raises."""
    if not repo_root or not tool:
        return False
    metadata_dir = Path(repo_root) / ".repowise"
    if not metadata_dir.is_dir():
        # An index is the opt-in boundary. Never create RepoWise state in an
        # arbitrary path merely because a caller supplied it as ``repo``.
        return False
    db_path = metadata_dir / OMISSIONS_DIRNAME / OMISSIONS_DB_FILENAME
    try:
        store = OmissionStore(db_path)
    except Exception:
        logger.debug("mcp usage store open failed", exc_info=True)
        return False
    try:
        store.record_mcp_usage(
            tool=tool,
            duration_ms=duration_ms,
            error=error,
            no_match=no_match,
            degraded=degraded,
            replaced_tokens=replaced_tokens,
            delivered_tokens=delivered_tokens,
        )
        return True
    except Exception:
        logger.debug("mcp usage write failed; dropping silently", exc_info=True)
        return False
    finally:
        with contextlib.suppress(Exception):
            store.close()


def record_mcp_saving(
    repo_root: str | Path | None,
    tool: str,
    replaced_tokens: int,
    delivered_tokens: int,
) -> bool:
    """Fold one positive counterfactual into the daily aggregate.

    ``raw_tokens`` is the counterfactual raw-exploration cost the answer
    replaced; ``distilled_tokens`` is what the agent actually received (measured
    after response-budget truncation, so the truncation saving is folded into
    the delta). Calls with no net saving (``replaced <= delivered``) are skipped
    by this compatibility seam; the main instrumentation uses
    :func:`record_mcp_call` and counts every invocation.

    The store is resolved **repo-locally** (``<repo>/.repowise/omissions``) — the
    exact sidecar the Costs endpoint reads. An indexed repository materialises
    it lazily on first use; an arbitrary path without ``.repowise`` does not.
    We deliberately do *not* fall back to the ``~/.repowise`` home store: a row
    landing there would be invisible to the dashboard and would pollute a global
    store across unrelated repos. Never raises.
    """
    if replaced_tokens <= delivered_tokens or not repo_root:
        return False
    return record_mcp_call(
        repo_root,
        tool,
        duration_ms=0,
        error=False,
        no_match=False,
        degraded=False,
        replaced_tokens=replaced_tokens,
        delivered_tokens=delivered_tokens,
    )


def record_mcp_dead_end(
    repo_root: str | Path | None,
    tool: str,
    delivered_tokens: int,
) -> bool:
    """Debit a dead-end call: tokens delivered, nothing replaced.

    An error response costs the agent its full delivered size and saves
    nothing. Aggregating raw=0 / distilled=delivered makes it a negative
    contribution (saved = raw - distilled) — without these debits savings only
    ever credit and overstate net value (the E11
    sign-flip: +138.7k claimed while the session net-spent).
    """
    if delivered_tokens <= 0 or not repo_root:
        return False
    return record_mcp_call(
        repo_root,
        tool,
        duration_ms=0,
        error=True,
        no_match=False,
        degraded=False,
        replaced_tokens=0,
        delivered_tokens=delivered_tokens,
    )
