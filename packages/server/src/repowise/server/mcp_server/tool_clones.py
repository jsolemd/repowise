"""MCP tool: find_clones — duplicated regions, exact and optionally near.

The tool is a thin adapter. Everything that decides *what a clone is*
lives behind :class:`~repowise.core.analysis.clone_service.CloneService`,
including the detector's on-disk caches, which nothing here can see and
nothing here reports. This module's job is to name the files in scope,
call the boundary, and hand the normalized result to the agent with its
degradations intact.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import select

from repowise.core.analysis.clone_service import (
    DEFAULT_NEAR_THRESHOLD,
    CloneService,
    SourceIndexNearClones,
)
from repowise.core.analysis.health.duplication import DEFAULT_MIN_LINES
from repowise.core.persistence.database import get_session
from repowise.core.persistence.models import GitMetadata, GraphNode
from repowise.core.registry import mcp_tool_registry as mcp
from repowise.server.mcp_server._helpers import (
    _get_exclude_spec,
    _get_repo,
    _resolve_repo_context,
    _unsupported_repo_all,
    is_excluded,
)
from repowise.server.mcp_server._meta import build_meta as _build_meta

#: Findings per response. The transport budget is the binding constraint;
#: a clone finding serializes to ~250 bytes, and past ~40 the answer stops
#: being something an agent acts on and starts being a report.
_MAX_FINDINGS = 40

#: Bounds on the near-clone threshold. Below 0.70 dense similarity over
#: same-repository code stops discriminating at all, and 1.0 can only ever
#: match a byte-identical chunk, which is the exact leg's job.
_NEAR_THRESHOLD_FLOOR = 0.70
_NEAR_THRESHOLD_CEILING = 0.999

_DEFINITION = {
    "exact": (
        "A region of at least min_lines whose normalized token stream — identifiers "
        "collapsed to a placeholder, literals collapsed, comments and whitespace "
        "dropped — appears verbatim in two places. Renaming a variable does not break a "
        "match; restructuring control flow does."
    ),
    "near": (
        "A pair of indexed symbols whose embeddings sit at or above near_threshold "
        "cosine, excluding pairs the exact leg already reported. Near findings are "
        "similarity evidence, not proof of duplication: read both sites before acting."
    ),
}


async def _scan_inputs(
    ctx: Any, *, include_tests: bool
) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]], Any]:
    """The file set and co-change map for one repository, from the index."""
    exclude_spec = _get_exclude_spec(ctx.path)
    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)

        rows = await session.execute(
            select(GraphNode.node_id, GraphNode.language, GraphNode.is_test).where(
                GraphNode.repository_id == repository.id,
                GraphNode.node_type == "file",
            )
        )
        # Test files duplicate each other constantly and on purpose — shared
        # setup, table-driven cases — so they are out of scope unless asked
        # for, in both legs rather than only in the near one.
        entries = [
            (path, language or "")
            for path, language, is_test in rows.all()
            if path and (include_tests or not is_test) and not is_excluded(path, exclude_spec)
        ]

        # Co-change history lets the detector rank actively-maintained
        # duplication above dormant duplication. Missing history is not an
        # error; every pair simply carries a zero co-change count.
        meta_rows = await session.execute(
            select(GitMetadata.file_path, GitMetadata.co_change_partners_json).where(
                GitMetadata.repository_id == repository.id
            )
        )
        git_meta_map = {
            path: {"co_change_partners_json": partners}
            for path, partners in meta_rows.all()
            if path
        }

    return sorted(entries), git_meta_map, repository


def _clamp_threshold(value: float) -> tuple[float, str | None]:
    clamped = min(max(float(value), _NEAR_THRESHOLD_FLOOR), _NEAR_THRESHOLD_CEILING)
    if clamped == float(value):
        return clamped, None
    return clamped, (
        f"near_threshold={value} is outside the useful range and was clamped to "
        f"{clamped}. Below {_NEAR_THRESHOLD_FLOOR} dense similarity over code from one "
        f"repository stops discriminating; at 1.0 only byte-identical chunks match, "
        "which the exact leg already covers."
    )


@mcp.tool(default=False)
async def find_clones(
    repo: str | None = None,
    path: str | None = None,
    min_lines: int = DEFAULT_MIN_LINES,
    limit: int = 20,
    include_near: bool = False,
    near_threshold: float = DEFAULT_NEAR_THRESHOLD,
    cross_directory_only: bool = False,
    include_intra_file: bool = True,
    include_tests: bool = False,
) -> dict:
    """Duplicated code regions, with the definition of "duplicate" attached.

    Run before extracting a helper, to see whether the thing you are about
    to write already exists twice. Findings are pairs of sites; each one
    carries a stable id, both line ranges, and whether the two sites sit in
    different directories.

    Args:
        repo: usually omitted.
        path: restrict the scan to a directory or file prefix.
        min_lines: smallest region to report (default 6).
        limit: max findings returned (clamped to 40).
        include_near: also run dense similarity over the source index for
            paraphrased duplication the token detector cannot see. Off by
            default; requires a built source index. Near findings are tagged
            kind="near" and are never merged into the exact set.
        near_threshold: cosine floor for near findings (default 0.86,
            clamped to [0.70, 0.999]). Ignored unless include_near is set.
        cross_directory_only: only report pairs whose sites are in
            different directories.
        include_intra_file: report regions duplicated within one file
            (default true).
        include_tests: scan test files too. Off by default in both legs —
            tests duplicate each other by design.
    """
    started = time.perf_counter()
    if repo == "all":
        return _unsupported_repo_all("find_clones")

    requested_limit = limit
    limit = min(max(limit, 1), _MAX_FINDINGS)
    min_lines = max(int(min_lines), 1)
    ignored: list[dict[str, Any]] = []

    ctx = await _resolve_repo_context(repo)
    entries, git_meta_map, repository = await _scan_inputs(ctx, include_tests=include_tests)

    if not entries:
        # An index with no file nodes cannot be scanned. Saying "no clones"
        # here would be a claim about the repository; it is a claim about
        # the index.
        return {
            "status": "failed",
            "findings": [],
            "definition": _DEFINITION,
            "no_results_reason": (
                "This repository has no indexed files, so no duplication scan ran. This "
                "is not a finding of zero duplication — run an index first."
            ),
            "degradations": [
                {
                    "code": "index_empty",
                    "scope": "scan",
                    "impact": "results_unavailable",
                    "detail": "No file nodes are present in the index for this repository.",
                }
            ],
            "_meta": _build_meta(
                timing_ms=(time.perf_counter() - started) * 1000, repository=repository
            ),
        }

    service = CloneService.from_paths(
        ctx.path, entries, cache_dir=ctx.path / ".repowise", git_meta_map=git_meta_map
    )

    if include_near:
        threshold, clamp_note = _clamp_threshold(near_threshold)
        if clamp_note:
            ignored.append({"argument": "near_threshold", "reason": clamp_note})
        index = SourceIndexNearClones.for_repo(ctx.path)
        report = await service.scan_with_near_clones(
            index,
            min_lines=min_lines,
            path_prefix=path,
            include_intra_file=include_intra_file,
            cross_directory_only=cross_directory_only,
            threshold=threshold,
            include_tests=include_tests,
        )
    else:
        report = service.scan(
            min_lines=min_lines,
            path_prefix=path,
            include_intra_file=include_intra_file,
            cross_directory_only=cross_directory_only,
        )

    payload = report.as_dict(limit=limit)
    payload["definition"] = _DEFINITION if include_near else {"exact": _DEFINITION["exact"]}
    # The filters that produced this answer travel with it, the same way a
    # pattern's parameters do. A caller comparing two responses has to be
    # able to see which scope each one ran under.
    payload["scope"] = {
        "path": path,
        "min_lines": min_lines,
        "include_tests": include_tests,
        "include_intra_file": include_intra_file,
        "cross_directory_only": cross_directory_only,
    }
    if requested_limit > _MAX_FINDINGS:
        payload["limit_note"] = (
            f"Requested limit={requested_limit} exceeded the transport budget and was "
            f"clamped to {_MAX_FINDINGS}. Narrow with path or cross_directory_only."
        )
    if ignored:
        payload["ignored_arguments"] = ignored
    payload["_meta"] = _build_meta(
        timing_ms=(time.perf_counter() - started) * 1000,
        repository=repository,
        targets=sorted({site["file"] for f in payload["findings"] for site in f["sites"]}),
        hint=_hint_for(payload, include_near=include_near),
    )
    return payload


def _hint_for(payload: dict[str, Any], *, include_near: bool) -> str | None:
    if payload["status"] == "failed":
        return "The scan did not run. Read degradations before treating this as a result."
    if not payload["findings"] and not include_near:
        return (
            "No exact duplication at this size. Paraphrased duplication is invisible to "
            "the token detector — retry with include_near=true if the source index is built."
        )
    if payload["summary"]["truncated"]:
        return "More findings than the limit; narrow with path or raise min_lines."
    return None
