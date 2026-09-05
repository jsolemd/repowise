"""Instrumentation middleware — bounded local usage and MCP savings.

:func:`instrument` wraps a registered MCP tool so that every call, after the
tool produces its (already budget-trimmed) response, measures the delivered
token count, derives the counterfactual raw-exploration cost it replaced, and
folds the call into one repo-local ``tool × UTC day`` aggregate. It optionally
stamps ``_meta.tokens_saved`` / ``_meta.replaced_tokens`` onto the response for
transparency. No query, target, path, session, or per-call record is persisted.

Two non-negotiables:

* **Byte-identical output.** The wrapped tool returns exactly what the tool
  returned, save for the optional additive ``_meta`` savings fields. The whole
  savings path is wrapped in a ``try`` that degrades to returning the untouched
  result on any failure.
* **Signature-preserving.** FastMCP introspects each tool's signature to build
  its schemas, so the wrapper copies ``functools.wraps`` metadata *and* pins an
  evaluated ``__signature__`` (:mod:`.._signature`) — a bare ``*args,
  **kwargs`` wrapper would erase the tool's parameters from the MCP schema, and
  an unevaluated one costs it the output schema.

The counterfactual comes from one of two places, in order of trust:
  1. a value the tool *declared* via :func:`declare_replaced` (it held the exact
     artifact, e.g. ``get_symbol`` with the whole source file);
  2. otherwise the conservative estimator in :mod:`.counterfactual`.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

from repowise.core.distill.budget import estimate_tokens
from repowise.server.mcp_server._signature import preserve

from . import counterfactual
from .recorder import record_mcp_call

logger = logging.getLogger(__name__)

#: Coarse, non-identifying result fields worth reporting per tool call. All are
#: enums or booleans (confidence tier, retrieval quality, staleness) — never
#: query text, paths, or repo/symbol names. See the telemetry privacy contract.
#:
#: ``degraded`` names WHY a get_answer reply carries no synthesised prose (no
#: provider, versus a provider that failed), which is the difference between an
#: install working as designed and one that broke.
_META_FLAGS = ("index_behind", "embedder_degraded")
_RESULT_ENUMS = ("confidence", "retrieval_quality", "grounding", "degraded")


def _semantic_search_state() -> bool | None:
    """The install's vector-leg state, or ``None`` when it was never evaluated."""
    from repowise.server.mcp_server._meta import semantic_search_state

    return semantic_search_state()


def _results_count_bucket(result: Any) -> str | None:
    """Bucket a search-shaped result count (never the results themselves)."""
    if not isinstance(result, dict):
        return None
    items = result.get("results")
    if not isinstance(items, list):
        return None
    n = len(items)
    if n == 0:
        return "0"
    if n <= 3:
        return "1-3"
    if n <= 10:
        return "4-10"
    return "10+"


def _telemetry_properties(tool: str, result: Any, duration_ms: int) -> dict[str, Any]:
    """Build the anonymous ``mcp_tool_call`` properties for *result*.

    Only coarse enums / booleans / bucketed counts — no user-identifying data.
    """
    is_error, _, _ = _result_signals(result)
    props: dict[str, Any] = {
        "tool": tool,
        "status": "error" if is_error else "ok",
        "duration_ms": duration_ms,
    }
    if isinstance(result, dict):
        for key in _RESULT_ENUMS:
            value = result.get(key)
            if isinstance(value, str) and value:
                props[key] = value
        meta = result.get("_meta")
        if isinstance(meta, dict):
            for key in _META_FLAGS:
                if isinstance(meta.get(key), bool):
                    props[key] = meta[key]
        bucket = _results_count_bucket(result)
        if bucket is not None:
            props["results_bucket"] = bucket
    # Read from server state rather than from the response. `embedder_degraded`
    # is False on a keyless install by design, so it only ever catches
    # misconfiguration and the larger keyless population - retrieval genuinely
    # full-text-only - was invisible. Taking it here keeps the caller's response
    # exactly as it was: this is a fact about the install, and the agent already
    # has everything it needs to see it.
    semantic_search = _semantic_search_state()
    if semantic_search is not None:
        props["semantic_search"] = semantic_search
    return props


def _emit_telemetry(tool: str, result: Any, duration_ms: int) -> None:
    """Emit one anonymous ``mcp_tool_call`` event. Best-effort, never raises.

    This is the field-visibility counterpart to the local bounded aggregate: it
    tells us which tools agents actually reach for, at what confidence, and how
    often results come back stale/degraded — the adoption signal the local
    ledger can't aggregate across installs.
    """
    from repowise.core.platform import telemetry

    telemetry.record_event("mcp_tool_call", _telemetry_properties(tool, result, duration_ms))


#: ``_meta`` key a tool sets to declare its own counterfactual (see
#: :func:`declare_replaced`). The wrapper reads and then leaves it in place as a
#: transparency annotation.
_DECLARED_KEY = "replaced_tokens"


def declare_replaced(result: dict[str, Any], tokens: int) -> None:
    """Let a tool declare an exact counterfactual the estimator can't compute.

    Writes ``result["_meta"]["replaced_tokens"]``; the wrapper prefers this over
    the generic estimator. Used by tools that already hold the replaced artifact
    in memory (e.g. ``get_symbol`` knows the full file it sliced one symbol out
    of). Best-effort and additive — never raises, only mutates ``_meta``.
    """
    if not isinstance(result, dict) or not isinstance(tokens, int) or tokens <= 0:
        return
    meta = result.setdefault("_meta", {})
    if isinstance(meta, dict):
        meta[_DECLARED_KEY] = tokens


def _declared_tokens(result: Any) -> int | None:
    """Return a tool-declared counterfactual from ``_meta``, if present."""
    if not isinstance(result, dict):
        return None
    meta = result.get("_meta")
    if not isinstance(meta, dict):
        return None
    value = meta.get(_DECLARED_KEY)
    return value if isinstance(value, int) and value > 0 else None


def _delivered_tokens(result: Any) -> int:
    """Estimate tokens the agent actually received for *result*."""
    try:
        text = json.dumps(result, default=str)
    except Exception:
        return 0
    return estimate_tokens(text)


def _result_signals(result: Any) -> tuple[bool, bool, bool]:
    """Return coarse ``(error, no_match, degraded)`` outcome flags."""
    if not isinstance(result, dict):
        return False, False, False
    status = result.get("status")
    confidence = result.get("confidence")
    error = bool(result.get("error")) or status in {"error", "failed"}
    no_match = (
        result.get("no_match") is True
        or confidence == "no_match"
        or status in {"no_match", "not_found"}
    )
    meta = result.get("_meta")
    meta = meta if isinstance(meta, dict) else {}
    degraded = any(
        value is True
        for value in (
            result.get("retrieval_degraded"),
            meta.get("retrieval_degraded"),
            meta.get("index_behind"),
            meta.get("embedder_degraded"),
        )
    )
    return error, no_match, degraded


async def _record(
    tool: str,
    result: Any,
    duration_ms: int,
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Measure and aggregate one call in the repository selected by ``repo=``."""
    declared = _declared_tokens(result)
    replaced = (
        declared if declared is not None else counterfactual.replaced_tokens_for(tool, result)
    )
    delivered = _delivered_tokens(result)
    error, no_match, degraded = _result_signals(result)

    # Resolve the actual workspace alias selected by this invocation. Using
    # the server's default repo here silently attributed cross-repo calls to
    # whichever repository happened to boot the process.
    from repowise.server.mcp_server._budget import resolve_response_budget_repo_root

    repo_root = await resolve_response_budget_repo_root(
        signature, args, kwargs, fallback_to_default=False
    )
    written = record_mcp_call(
        repo_root,
        tool,
        duration_ms=duration_ms,
        error=error,
        no_match=no_match,
        degraded=degraded,
        replaced_tokens=max(0, replaced),
        delivered_tokens=delivered,
    )
    if written and replaced > delivered and isinstance(result, dict):
        meta = result.setdefault("_meta", {})
        if isinstance(meta, dict):
            meta["replaced_tokens"] = replaced
            meta["tokens_saved"] = replaced - delivered


def instrument(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async MCP tool *fn* to record its savings. Signature-preserving.

    Non-coroutine callables are returned unchanged — every OSS MCP tool is
    ``async``, and a sync tool has no measured-response hook here.
    """
    if not inspect.iscoroutinefunction(fn):
        return fn

    tool = getattr(fn, "__name__", "tool")
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    async def _wrapped(*args: Any, **kwargs: Any) -> Any:
        import time

        _t0 = time.perf_counter()
        result = await fn(*args, **kwargs)
        duration_ms = int((time.perf_counter() - _t0) * 1000)
        try:
            await _record(tool, result, duration_ms, signature, args, kwargs)
        except Exception:  # pragma: no cover - defensive; savings never break a tool
            logger.debug("mcp savings instrumentation failed for %s", tool, exc_info=True)
        try:
            _emit_telemetry(tool, result, duration_ms)
        except Exception:  # pragma: no cover - defensive; telemetry never breaks a tool
            logger.debug("mcp telemetry emit failed for %s", tool, exc_info=True)
        return result

    return preserve(_wrapped, fn)
