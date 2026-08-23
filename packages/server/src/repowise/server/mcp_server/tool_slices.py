"""MCP Tools: build_task_slice / get_task_slice / extend_task_slice.

A *task slice* is the part of the codebase one task needs, held under an id
that survives the call. The three tools are the three things an agent does
with one: cut it, read it again (possibly at a different fidelity or budget),
and grow it when the first cut turned out to be too narrow.

Why three tools and not one with a mode flag: they fail differently. Building
can fail because nothing matched the task; fetching can fail because the id is
wrong; extending can fail because the slice is already fully expanded. Folding
them together would force one response shape to carry three unrelated recovery
paths, which is how "no members" ends up meaning four different things.

Every failure here is shaped, and **no failure response carries a member
list**. A tool that answers a bad slice id with ``{"members": []}`` has told
the agent the task needs no code — the single most expensive thing this
surface could get wrong. Failures carry ``status`` plus ``error`` plus
whatever evidence recovers them, and nothing that could be mistaken for a
result.

Budget drops are disclosed *and* recoverable: the dropped members go to the
shared omission store, so the ``[repowise#<ref>]`` marker on the response
resolves through ``repowise expand`` or ``get_symbol`` exactly like every
other truncation this server emits.
"""

from __future__ import annotations

import time
from typing import Any

from repowise.core.persistence.database import get_session
from repowise.core.registry import mcp_tool_registry as mcp
from repowise.core.slices import (
    BudgetReport,
    SliceError,
    SliceStore,
    WalkPolicy,
    build_slice,
    extend_slice,
    load_slice,
    render_slice,
)
from repowise.server.mcp_server._budget import OmissionCollector, effective_char_budget
from repowise.server.mcp_server._helpers import (
    _get_exclude_spec,
    _get_repo,
    _resolve_repo_context,
    _unsupported_repo_all,
)
from repowise.server.mcp_server._meta import build_meta as _build_meta

#: Default response budget for a slice render, in tokens. Below the shared
#: ``TOKEN_BUDGET`` because a slice is usually one step in a longer session
#: and should leave the agent room to act on it.
DEFAULT_BUDGET_TOKENS = 6000

#: Hard floor. Under this the envelope alone eats the budget and every call
#: would raise ``budget_too_small``, which is a worse answer than clamping.
MIN_BUDGET_TOKENS = 400


def _clamp_budget(requested: int) -> int:
    """Clamp a caller's budget under the live MCP host cap.

    ``effective_char_budget`` already lowers our ceiling when the host's
    ``MAX_MCP_OUTPUT_TOKENS`` is narrower than our own default; dividing back
    to tokens keeps a slice from being the one response that trips the host's
    reject-with-isError path.
    """
    from repowise.server.mcp_server._budget import CHARS_PER_TOKEN

    ceiling = effective_char_budget(max(requested, MIN_BUDGET_TOKENS) * CHARS_PER_TOKEN)
    return max(MIN_BUDGET_TOKENS, ceiling // CHARS_PER_TOKEN)


def _failure(exc: SliceError, **extra: Any) -> dict[str, Any]:
    """Shape a typed slice failure. Deliberately carries no member list."""
    payload: dict[str, Any] = {
        "status": exc.status,
        "error": str(exc),
        **exc.details(),
        **extra,
    }
    payload["_meta"] = _build_meta()
    return payload


class _DropRecorder:
    """Captures the full budget report so the drops can be made recoverable.

    The response itemises only the highest-ranked drops (an unbounded honesty
    list is the one thing that could blow the budget it is reporting on). The
    omission store gets all of them, so ``repowise expand <ref>`` returns the
    complete set rather than the same truncated view.
    """

    def __init__(self) -> None:
        self.report: BudgetReport | None = None

    def __call__(self, report: BudgetReport) -> None:
        self.report = report

    def attach(self, response: dict[str, Any], tool: str, repo_root: str) -> None:
        if self.report is None or not self.report.dropped:
            return
        collector = OmissionCollector(tool, repo_root=repo_root)
        collector.add(
            f"slice {response.get('slice_id')} — {len(self.report.dropped)} member(s) "
            f"dropped to fit budget_tokens={self.report.budget_tokens} "
            f"at view={self.report.view!r}",
            [drop.to_dict() for drop in self.report.dropped],
        )
        collector.attach(response)


def _policy_from_args(
    downstream_depth: int,
    upstream_depth: int,
    include_tests: bool,
    max_members: int,
) -> WalkPolicy:
    return WalkPolicy(
        downstream_depth=downstream_depth,
        upstream_depth=upstream_depth,
        include_tests=include_tests,
        max_members=max_members,
    ).clamped()


@mcp.tool(default=False)
async def build_task_slice(
    task: str,
    entry_points: list[str] | None = None,
    repo: str | None = None,
    view: str = "skeleton",
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    downstream_depth: int = 2,
    upstream_depth: int = 1,
    include_tests: bool = False,
    max_members: int = 400,
    include_edges: bool = False,
) -> dict[str, Any]:
    """Cut a persistent, extendable slice of the codebase for one task.

    Resolves where the task starts (from its own words, or from entry points
    you name), walks the dependency graph outward — deep in the direction the
    entry point *reaches*, shallow in the direction that *reaches it* — ranks
    every member by why it is there, and stores the result under a slice id
    you can re-fetch and grow.

    Call this instead of a series of get_context / get_dependents probes when
    a task spans more than one file and you want the working set once, ranked,
    inside a budget you choose.

    Args:
        task: The task in your own words. Used to nominate entry points and
            to weight members whose names the task mentions.
        entry_points: Explicit starting nodes — file paths, ``path::Symbol``
            ids, or unambiguous symbol names. Composes with ``task``.
        repo: Repository path, name, or ID.
        view: Fidelity per member — "card" (identity + why), "skeleton"
            (+ signatures and defined symbols), "full" (+ source).
        budget_tokens: Response ceiling. Over-budget members drop from the
            bottom of the ranking and every drop is listed and recoverable.
        downstream_depth: How far to follow what the entry points use (0-6).
        upstream_depth: How far to follow what uses the entry points (0-6).
        include_tests: Include test files and traverse through them.
        max_members: Ceiling on slice size; reaching it is reported.
        include_edges: Return the traversed edges as well as the members.
    """
    if repo == "all":
        return _unsupported_repo_all("build_task_slice")

    started = time.perf_counter()
    ctx = await _resolve_repo_context(repo)
    exclude_spec = _get_exclude_spec(ctx.path)
    drops = _DropRecorder()
    policy = _policy_from_args(downstream_depth, upstream_depth, include_tests, max_members)

    try:
        store = SliceStore.open_default(ctx.path)
    except SliceError as exc:
        return _failure(exc, task=task)

    try:
        async with get_session(ctx.session_factory) as session:
            repository = await _get_repo(session)
            record = await build_slice(
                session,
                repo_id=repository.id,
                repo_path=str(ctx.path),
                task=task,
                store=store,
                entry_points=entry_points,
                policy=policy,
                exclude_spec=exclude_spec,
            )
            response = await render_slice(
                session,
                record,
                view=view,
                budget_tokens=_clamp_budget(budget_tokens),
                repo_root=ctx.path,
                include_edges=include_edges,
                on_budget=drops,
            )
    except SliceError as exc:
        return _failure(exc, task=task)
    finally:
        store.close()

    response["_meta"] = _build_meta(
        timing_ms=(time.perf_counter() - started) * 1000,
        repository=repository,
        hint=response.get("next"),
    )
    # After ``_meta`` is built, never before: the collector writes its refs
    # into that same dict, and stamping a fresh envelope over the top would
    # drop the very pointer that makes the drop recoverable.
    drops.attach(response, "build_task_slice", str(ctx.path))
    return response


@mcp.tool(default=False)
async def get_task_slice(
    slice_id: str,
    repo: str | None = None,
    view: str = "skeleton",
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    max_source_lines: int = 200,
    include_edges: bool = False,
) -> dict[str, Any]:
    """Re-read a stored task slice, at any fidelity and any budget.

    The slice is not re-walked: this reads the stored members and re-renders
    them, so switching from "card" to "full" or raising the budget costs a
    serialisation, not a graph traversal. An unknown slice id is an error with
    the recent ids attached, never an empty slice.

    Args:
        slice_id: The id build_task_slice returned (``sl_<12 hex>``).
        repo: Repository path, name, or ID.
        view: "card", "skeleton", or "full".
        budget_tokens: Response ceiling; drops are listed and recoverable.
        max_source_lines: Per-member source cap for view="full".
        include_edges: Return the traversed edges as well as the members.
    """
    if repo == "all":
        return _unsupported_repo_all("get_task_slice")

    started = time.perf_counter()
    ctx = await _resolve_repo_context(repo)
    drops = _DropRecorder()

    try:
        store = SliceStore.open_default(ctx.path)
    except SliceError as exc:
        return _failure(exc, slice_id=slice_id)

    try:
        record = load_slice(store, slice_id)
        async with get_session(ctx.session_factory) as session:
            repository = await _get_repo(session)
            response = await render_slice(
                session,
                record,
                view=view,
                budget_tokens=_clamp_budget(budget_tokens),
                max_source_lines=max(1, max_source_lines),
                repo_root=ctx.path,
                include_edges=include_edges,
                on_budget=drops,
            )
    except SliceError as exc:
        return _failure(exc, slice_id=slice_id)
    finally:
        store.close()

    response["_meta"] = _build_meta(
        timing_ms=(time.perf_counter() - started) * 1000,
        repository=repository,
        hint=response.get("next"),
    )
    # After ``_meta`` is built, never before: the collector writes its refs
    # into that same dict, and stamping a fresh envelope over the top would
    # drop the very pointer that makes the drop recoverable.
    drops.attach(response, "get_task_slice", str(ctx.path))
    return response


@mcp.tool(default=False)
async def extend_task_slice(
    slice_id: str,
    repo: str | None = None,
    extra_downstream: int = 1,
    extra_upstream: int = 0,
    entry_points: list[str] | None = None,
    task_addendum: str | None = None,
    view: str = "card",
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    include_edges: bool = False,
) -> dict[str, Any]:
    """Grow a stored slice from where its walk stopped.

    Resumes from the slice's frontier — the members whose own edges were
    never expanded — rather than re-walking from the entry points. Members
    already in the slice keep their distance and are not rediscovered. The
    ``extension`` block reports how many frontier nodes were resumed from and
    how many queries it cost, so the claim is checkable rather than asserted.

    Args:
        slice_id: The id to grow.
        repo: Repository path, name, or ID.
        extra_downstream: Additional depth in the direction the slice reaches.
        extra_upstream: Additional depth in the direction that reaches it.
        entry_points: New starting nodes to fold into the same slice.
        task_addendum: Extra task wording to nominate new entry points from.
        view: Fidelity for the response render (defaults to the cheap one —
            an extension is usually read for what it added).
        budget_tokens: Response ceiling; drops are listed and recoverable.
        include_edges: Return the traversed edges as well as the members.
    """
    if repo == "all":
        return _unsupported_repo_all("extend_task_slice")

    started = time.perf_counter()
    ctx = await _resolve_repo_context(repo)
    exclude_spec = _get_exclude_spec(ctx.path)
    drops = _DropRecorder()

    try:
        store = SliceStore.open_default(ctx.path)
    except SliceError as exc:
        return _failure(exc, slice_id=slice_id)

    try:
        async with get_session(ctx.session_factory) as session:
            repository = await _get_repo(session)
            record, extension = await extend_slice(
                session,
                store=store,
                slice_id=slice_id,
                extra_downstream=extra_downstream,
                extra_upstream=extra_upstream,
                entry_points=entry_points,
                task_addendum=task_addendum,
                exclude_spec=exclude_spec,
            )
            response = await render_slice(
                session,
                record,
                view=view,
                budget_tokens=_clamp_budget(budget_tokens),
                repo_root=ctx.path,
                include_edges=include_edges,
                extension=extension,
                on_budget=drops,
            )
    except SliceError as exc:
        return _failure(exc, slice_id=slice_id)
    finally:
        store.close()

    response["_meta"] = _build_meta(
        timing_ms=(time.perf_counter() - started) * 1000,
        repository=repository,
        hint=response.get("next"),
    )
    # After ``_meta`` is built, never before: the collector writes its refs
    # into that same dict, and stamping a fresh envelope over the top would
    # drop the very pointer that makes the drop recoverable.
    drops.attach(response, "extend_task_slice", str(ctx.path))
    return response
