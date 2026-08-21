"""MCP tools for source-index trust checks and bounded reindex queueing."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select

from repowise.core.ingestion import FileTraverser, is_candidate_source_path
from repowise.core.persistence.database import get_session
from repowise.core.persistence.models import GenerationJob
from repowise.core.registry import mcp_tool_registry as mcp
from repowise.core.source_search.chunks import parser_eligible, window_eligible
from repowise.core.source_search.coordinator import LEG_SOURCE_DENSE, LEG_SOURCE_LEXICAL
from repowise.core.source_search.fts import SourceFTSIndex
from repowise.core.source_search.generation import GenerationRef
from repowise.core.source_search.manifest import identify_embedder
from repowise.core.source_search.status import SourceIndexStatus, inspect_source_index
from repowise.core.workspace.update import get_head_commit, read_repo_state
from repowise.server.job_executor import execute_job
from repowise.server.mcp_server import _state
from repowise.server.mcp_server._helpers import (
    _get_repo,
    _resolve_repo_context,
    _unsupported_repo_all,
    attach_ignored_arguments,
    resolve_enum_argument,
)
from repowise.server.mcp_server._meta import build_meta as _build_meta
from repowise.server.services.job_queue import queue_index_only_job

_READ_MODES = frozenset({"status", "path"})
_SOURCE_MANIFEST_COMPONENT = "source manifest"
_SOURCE_QUEUE_COMPONENT = "source queue"
_SOURCE_PUBLICATION_COMPONENT = "source publication"

# Standalone MCP has no FastAPI ``app.state``. Keep the process-owned
# registries stable across calls, but snapshot repo-specific attributes in a
# fresh namespace for every launch. A single mutable namespace could otherwise
# swap one running repo's FTS/embedder state when another repo queued work.
_MCP_BACKGROUND_TASKS: set[Any] = set()
_MCP_JOB_TASKS: dict[str, Any] = {}
_MCP_JOB_EVENTS: dict[str, Any] = {}
_MCP_JOB_CANCEL_TOKENS: dict[str, Any] = {}
_MCP_WORKSPACE_VECTOR_STORES: dict[str, Any] = {}


def _repo_payload(ctx: Any, repository: Any) -> dict[str, Any]:
    return {
        "alias": getattr(ctx, "alias", "default"),
        "id": repository.id,
        "name": repository.name,
        "path": Path(ctx.path).as_posix(),
    }


def _runtime_identities(ctx: Any) -> tuple[Any | None, str | None, str | None]:
    embedder = getattr(ctx.vector_store, "_embedder", None)
    runtime_embedder = None
    embedder_error = None
    if embedder is not None:
        try:
            runtime_embedder = identify_embedder(embedder)
        except Exception as exc:
            embedder_error = f"{type(exc).__name__}: {exc}"

    runtime_parser = None
    parser_error = None
    try:
        from repowise.core.ingestion.parse_cache import parser_fingerprint

        runtime_parser = parser_fingerprint()
    except Exception as exc:
        parser_error = f"{type(exc).__name__}: {exc}"
    identity_errors = [error for error in (embedder_error, parser_error) if error]
    return runtime_embedder, runtime_parser, "; ".join(identity_errors) or None


def _trust_state(
    status: SourceIndexStatus,
    *,
    head_commit: str | None,
    runtime_embedder: Any | None,
    runtime_parser: str | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    # These failures remove the evidence needed to judge the publication at
    # all. Do not let a known queue row turn a broken manifest/store into the
    # stronger claim "stale".
    hard_unknown = status.manifest_state != "ok" or any(
        marker in error.lower()
        for error in status.integrity_errors
        for marker in ("unreadable", "missing")
    )
    if hard_unknown:
        if status.manifest_state != "ok":
            reasons.append(f"manifest_{status.manifest_state}")
        for error in status.integrity_errors:
            lowered = error.lower()
            if "outbox" in lowered:
                reasons.append("source_queue_unverified")
            elif "manifest" in lowered:
                reasons.append("manifest_unverified")
        if status.fts_chunks is None:
            reasons.append("fts_unverified")
        if status.vector_chunks is None:
            reasons.append("vector_unverified")
        if not reasons:
            reasons.append("source_publication_unverified")
        return "unknown", list(dict.fromkeys(reasons))

    known_stale: list[str] = []
    if (
        head_commit is not None
        and status.indexed_commit is not None
        and head_commit != status.indexed_commit
    ):
        known_stale.append("index_behind_head")
    if status.pending_updates or status.building_updates or status.ready_updates:
        known_stale.append("updates_outstanding")
    if status.blocked_updates:
        known_stale.append("updates_blocked")
    if status.stale_files:
        known_stale.append("stale_files")
    if status.fts_chunks is not None and status.fts_chunks != status.expected_chunks:
        known_stale.append("fts_count_mismatch")
    if status.vector_chunks is not None and status.vector_chunks != status.expected_chunks:
        known_stale.append("vector_count_mismatch")
    if (
        status.parser_fingerprint is not None
        and runtime_parser is not None
        and status.parser_fingerprint != runtime_parser
    ):
        known_stale.append("parser_fingerprint_mismatch")
    if (
        status.embedder is not None
        and runtime_embedder is not None
        and status.embedder != runtime_embedder
    ):
        known_stale.append("embedder_identity_mismatch")
    if known_stale:
        return "stale", known_stale

    if head_commit is None:
        reasons.append("head_unavailable")
    if status.indexed_commit is None:
        reasons.append("indexed_commit_unavailable")
    if status.fts_chunks is None:
        reasons.append("fts_unverified")
    if status.vector_chunks is None:
        reasons.append("vector_unverified")
    if status.parser_fingerprint is None or runtime_parser is None:
        reasons.append("parser_identity_unavailable")
    if status.embedder is None or runtime_embedder is None:
        reasons.append("embedder_identity_unavailable")
    if status.state != "current":
        reasons.append(f"index_state_{status.state}")
    if reasons:
        return "unknown", reasons
    return "trustworthy", []


def _degradation_findings(status: SourceIndexStatus) -> list[dict[str, str]]:
    """Structured index-health findings, distinct from retrieval failures.

    ``failed_legs`` is reserved for runtime retrieval failures whose ``error``
    value is an exception class.  Publication health has durable status codes,
    so it uses a separate field and shape instead of overloading that contract.
    """
    findings: list[dict[str, str]] = []
    for detail in status.integrity_errors:
        lowered = detail.lower()
        if "fts" in lowered:
            component = LEG_SOURCE_LEXICAL
        elif "lance" in lowered:
            component = LEG_SOURCE_DENSE
        elif "manifest" in lowered:
            component = _SOURCE_MANIFEST_COMPONENT
        elif "outbox" in lowered:
            component = _SOURCE_QUEUE_COMPONENT
        else:
            component = _SOURCE_PUBLICATION_COMPONENT
        code = "count_mismatch" if "count mismatch" in lowered else "unavailable"
        findings.append({"component": component, "code": code, "detail": detail})
    if status.last_error:
        findings.append(
            {
                "component": _SOURCE_QUEUE_COMPONENT,
                "code": "update_failed",
                "detail": status.last_error,
            }
        )
    if status.stale_files:
        findings.append(
            {
                "component": _SOURCE_PUBLICATION_COMPONENT,
                "code": "stale_files",
                "detail": f"{len(status.stale_files)} source files are stale",
            }
        )
    if status.blocked_updates and not status.last_error:
        findings.append(
            {
                "component": _SOURCE_QUEUE_COMPONENT,
                "code": "updates_blocked",
                "detail": f"{status.blocked_updates} source-index update rows are blocked",
            }
        )
    return findings


async def _status_payload(ctx: Any, repository: Any, *, started: float) -> tuple[dict, Any, dict]:
    repo_path = Path(ctx.path).resolve()
    embedder = getattr(ctx.vector_store, "_embedder", None)
    status = await inspect_source_index(
        repo_path,
        embedder=embedder,
        verify_stores=True,
    )
    head_commit = get_head_commit(repo_path)
    repo_state = read_repo_state(repo_path)
    working_tree_paths = sorted(
        str(path) for path in (repo_state.get("working_tree_paths") or []) if isinstance(path, str)
    )
    runtime_embedder, runtime_parser, identity_error = _runtime_identities(ctx)
    trust, trust_reasons = _trust_state(
        status,
        head_commit=head_commit,
        runtime_embedder=runtime_embedder,
        runtime_parser=runtime_parser,
    )

    fts_parity = (
        status.fts_chunks == status.expected_chunks if status.fts_chunks is not None else None
    )
    vector_parity = (
        status.vector_chunks == status.expected_chunks if status.vector_chunks is not None else None
    )
    parser_match = (
        status.parser_fingerprint == runtime_parser
        if status.parser_fingerprint is not None and runtime_parser is not None
        else None
    )
    embedder_match = (
        status.embedder == runtime_embedder
        if status.embedder is not None and runtime_embedder is not None
        else None
    )
    commit_match = (
        status.indexed_commit == head_commit
        if status.indexed_commit is not None and head_commit is not None
        else None
    )

    payload: dict[str, Any] = {
        "status": status.state,
        "mode": "status",
        "repo": _repo_payload(ctx, repository),
        "trust": {
            "search_results": trust,
            "reasons": trust_reasons,
        },
        "generation": {
            "id": status.generation_id,
            "sequence": status.generation_sequence,
            "indexed_commit": status.indexed_commit,
            "head_commit": head_commit,
            "commit_matches": commit_match,
            "built_at": status.built_at,
            "published_at": status.published_at,
            "uncommitted_indexed_paths": working_tree_paths,
            "stale_files": dict(status.stale_files),
            "stale_file_count": len(status.stale_files),
        },
        "queue": {
            "pending": status.pending_updates,
            "building": status.building_updates,
            "ready": status.ready_updates,
            "blocked": status.blocked_updates,
            "total": (
                status.pending_updates
                + status.building_updates
                + status.ready_updates
                + status.blocked_updates
            ),
            "unit": "source_index_update_rows_after_active_generation",
        },
        "stores": {
            "manifest": {
                "state": status.manifest_state,
                "error": status.manifest_error,
                "chunks": {
                    "symbol": status.symbol_chunks,
                    "file_window": status.file_window_chunks,
                    "total": status.expected_chunks,
                },
                "files_covered": status.files_covered,
            },
            "fts_chunks": status.fts_chunks,
            "vector_chunks": status.vector_chunks,
            "parity": {"fts": fts_parity, "vector": vector_parity},
            "unit": "active_source_chunks",
        },
        "recipe": {
            "fingerprint": status.recipe_fingerprint,
            "indexed_parser_fingerprint": status.parser_fingerprint,
            "runtime_parser_fingerprint": runtime_parser,
            "parser_matches": parser_match,
            "indexed_embedder": asdict(status.embedder) if status.embedder is not None else None,
            "runtime_embedder": (
                asdict(runtime_embedder) if runtime_embedder is not None else None
            ),
            "embedder_matches": embedder_match,
            "identity_error": identity_error,
        },
        "last_update": status.published_at or status.built_at,
        "degraded": status.degraded,
        "_meta": _build_meta(
            timing_ms=(perf_counter() - started) * 1000.0,
            repository=repository,
        ),
    }
    if status.degraded:
        findings = _degradation_findings(status)
        payload["degraded_reason"] = (
            status.last_error
            or "; ".join(status.integrity_errors)
            or (findings[0]["detail"] if findings else status.state)
        )
        payload["degradation_findings"] = findings
    return payload, status, repo_state


def _normalize_path(repo_path: Path, requested: str) -> tuple[str | None, str | None]:
    raw = requested.strip()
    if not raw or "\x00" in raw:
        return None, "invalid_path"
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_path / raw
    try:
        resolved = candidate.resolve()
        relative = resolved.relative_to(repo_path.resolve())
    except (OSError, ValueError):
        return None, "outside_repo"
    normalized = relative.as_posix()
    return (normalized, None) if normalized not in {"", "."} else (None, "invalid_path")


def _tracked_state(repo_path: Path, rel_path: str) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel_path],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None


def _file_inventory(
    repo_path: Path,
    status: SourceIndexStatus,
    rel_path: str,
) -> tuple[dict[str, int] | None, str | None]:
    if status.manifest_state != "ok":
        return None, f"manifest_{status.manifest_state}"
    if status.generation_id is None or status.generation_sequence is None or not status.fts_path:
        return None, "active_generation_unavailable"
    fts_path = repo_path / status.fts_path
    if not fts_path.is_file():
        return None, "fts_store_missing"
    try:
        generation = GenerationRef(status.generation_id, status.generation_sequence)
        with SourceFTSIndex(fts_path, generation=generation) as fts:
            return asdict(fts.inventory_for_file(rel_path)), None
    except Exception as exc:
        return None, f"FTS inventory unavailable: {type(exc).__name__}: {exc}"


def _wiki_eligibility(repo_path: Path, rel_path: str) -> tuple[bool | None, str | None]:
    try:
        from repowise.core.repo_config import load_repo_config

        excludes = list(load_repo_config(repo_path).get("exclude_patterns") or [])
        traverser = FileTraverser(
            repo_path,
            extra_exclude_patterns=excludes,
        )
        if traverser.dir_chain_skipped(Path(rel_path).parent):
            return False, "directory policy rejected the path; deciding rule is unavailable"
        info = traverser.file_info_for_path(rel_path, resolve_entry_point=False)
        if info is None:
            return False, "file policy rejected the path; deciding rule is unavailable"
        return True, None
    except Exception as exc:
        return None, f"traverser unavailable: {type(exc).__name__}: {exc}"


def _path_mode_payload(
    base: dict[str, Any],
    status: SourceIndexStatus,
    repo_state: dict[str, Any],
    *,
    requested: str,
) -> dict[str, Any]:
    repo_path = Path(base["repo"]["path"]).resolve()
    normalized, path_error = _normalize_path(repo_path, requested)
    base["mode"] = "path"
    if path_error is not None:
        base["path"] = {
            "requested": requested,
            "normalized": normalized,
            "reason": path_error,
            "eligible": False,
            "indexed": False,
        }
        return base

    assert normalized is not None
    absolute = repo_path / normalized
    exists = absolute.exists()
    is_file = absolute.is_file()
    if not exists or not is_file:
        base["path"] = {
            "requested": requested,
            "normalized": normalized,
            "exists": exists,
            "is_file": is_file,
            "reason": "missing" if not exists else "not_file",
            "eligible": False,
            "indexed": False,
        }
        return base

    inventory, inventory_error = _file_inventory(repo_path, status, normalized)
    indexed = inventory["total"] > 0 if inventory is not None else None
    tracked = _tracked_state(repo_path, normalized)
    parser_policy = parser_eligible(normalized)
    wiki_eligible, wiki_unknown = _wiki_eligibility(repo_path, normalized)
    shape_candidate = is_candidate_source_path(normalized)

    if inventory is not None:
        window_policy: bool | None = window_eligible(
            normalized,
            indexed_symbols=inventory["symbol"],
        )
    elif parser_policy:
        # Code files are windowed only when their symbol count is zero. Without
        # active inventory, choosing either value would be a guess.
        window_policy = None
    else:
        window_policy = window_eligible(normalized, indexed_symbols=0)

    working_tree_paths = {
        str(path) for path in (repo_state.get("working_tree_paths") or []) if isinstance(path, str)
    }
    working_tree_indexed = normalized in working_tree_paths
    parser_lane = parser_policy and wiki_eligible is True
    if window_policy is True:
        if indexed is True or working_tree_indexed or tracked is True:
            window_lane: bool | None = True
        elif tracked is False:
            window_lane = False
        else:
            window_lane = None
    elif window_policy is False:
        window_lane = False
    else:
        window_lane = None

    if indexed is True or parser_lane or window_lane is True:
        eligible: bool | None = True
    elif indexed is None or wiki_eligible is None or window_lane is None:
        eligible = None
    else:
        eligible = False

    stale_reason = status.stale_files.get(normalized)
    unknown_reason = None
    if indexed is None:
        reason = "unknown"
        unknown_reason = inventory_error or "active inventory unavailable"
    elif indexed:
        reason = "parser_failed_stale" if stale_reason else "indexed"
    elif eligible is True:
        reason = "eligible_not_indexed"
    elif tracked is False and window_policy is True and not parser_lane:
        reason = "untracked_window_only"
    elif eligible is False and wiki_unknown is None:
        reason = "not_source_eligible"
    else:
        reason = "unknown"
        unknown_reason = wiki_unknown or "one or more eligibility facts are unavailable"

    lanes = []
    if inventory is not None and inventory["symbol"]:
        lanes.append("symbol")
    if inventory is not None and inventory["file_window"]:
        lanes.append("file_window")
    base["path"] = {
        "requested": requested,
        "normalized": normalized,
        "exists": True,
        "is_file": True,
        "tracked": tracked,
        "untracked": (not tracked) if tracked is not None else None,
        "working_tree_indexed": working_tree_indexed,
        "path_shape_candidate": shape_candidate,
        "eligibility": {
            "eligible": eligible,
            "parser": {
                "policy_eligible": parser_policy,
                "wiki_eligible": wiki_eligible,
                "lane_eligible": parser_lane,
                "unknown_reason": wiki_unknown,
            },
            "file_window": {
                "policy_eligible": window_policy,
                "lane_eligible": window_lane,
                "requires_tracked_path": True,
            },
        },
        "indexed": indexed,
        "generation": (
            {"id": status.generation_id, "sequence": status.generation_sequence}
            if indexed
            else None
        ),
        "stale": bool(stale_reason),
        "stale_reason": stale_reason,
        "lanes": lanes,
        "chunks": {
            **(inventory or {"total": None, "symbol": None, "file_window": None}),
            "unit": "active_source_chunks_for_path",
            "unknown_reason": inventory_error,
        },
        "reason": reason,
        "unknown_reason": unknown_reason,
    }
    return base


async def _active_jobs(session_factory: Any, repository_id: str) -> list[dict[str, Any]]:
    async with get_session(session_factory) as session:
        rows = list(
            (
                await session.execute(
                    select(GenerationJob)
                    .where(
                        GenerationJob.repository_id == repository_id,
                        GenerationJob.status.in_(["pending", "running"]),
                    )
                    .order_by(GenerationJob.created_at)
                )
            )
            .scalars()
            .all()
        )
    jobs = []
    for row in rows:
        config_error = None
        try:
            config = json.loads(row.config_json) if row.config_json else {}
        except (TypeError, ValueError):
            config = {}
            config_error = "invalid_config_json"
        job = {
            "id": row.id,
            "state": row.status,
            "mode": None if config_error else config.get("mode") or "sync",
            "bypass_current_noop": config.get("bypass_current_noop", config.get("force")),
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        if config_error:
            job["config_error"] = config_error
        jobs.append(job)
    return jobs


def _job_runtime(ctx: Any) -> Any:
    return SimpleNamespace(
        session_factory=ctx.session_factory,
        fts=ctx.fts,
        vector_store=ctx.vector_store,
        cross_repo_enricher=_state._cross_repo_enricher,
        background_tasks=_MCP_BACKGROUND_TASKS,
        job_tasks=_MCP_JOB_TASKS,
        job_events=_MCP_JOB_EVENTS,
        job_cancel_tokens=_MCP_JOB_CANCEL_TOKENS,
        workspace_vector_stores=_MCP_WORKSPACE_VECTOR_STORES,
    )


@mcp.tool()
async def get_index_status(
    mode: str = "status",
    path: str | None = None,
    repo: str | None = None,
) -> dict[str, Any]:
    """Check whether source-search results are trustworthy, stale, or unknown.

    Use ``mode="status"`` before relying on indexed results. Use
    ``mode="path"`` with one repository-relative path to inspect its exact
    active-generation inventory, parser/window eligibility,
    tracked state, and any fact the public index APIs cannot determine.

    Args:
        mode: ``status`` (default) or ``path``.
        path: One repository-relative path; required for path mode.
        repo: Repository path, name, ID, or workspace alias.
    """
    started = perf_counter()
    if repo == "all":
        return _unsupported_repo_all("get_index_status")

    ignored: list[dict[str, Any]] = []
    resolved_mode = (
        resolve_enum_argument(mode, _READ_MODES, argument="mode", ignored=ignored) or "status"
    )
    ctx = await _resolve_repo_context(repo)
    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)
    payload, status, repo_state = await _status_payload(ctx, repository, started=started)
    if resolved_mode == "path":
        if path is None:
            payload.update(
                {
                    "mode": "path",
                    "error": "path is required when mode='path'",
                }
            )
        else:
            payload = _path_mode_payload(
                payload,
                status,
                repo_state,
                requested=path,
            )
    elif path is not None:
        ignored.append(
            {
                "argument": "path",
                "values": [path],
                "valid": ["mode='path'"],
            }
        )
    attach_ignored_arguments(payload, ignored)
    payload["_meta"]["timing_ms"] = round((perf_counter() - started) * 1000.0, 2)
    return payload


@mcp.tool(default=False)
async def reindex_repository(
    repo: str | None = None,
    confirm: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Preview or queue a non-generative repository reindex.

    Without ``confirm=true`` this is read-only and states the known cost.
    Confirmed work uses the existing ``index_only`` job, never a source-store
    shortcut. ``force=true`` queues work even when the verified index is
    already current. This tool is off in the default MCP profile.

    Args:
        repo: Repository path, name, ID, or workspace alias.
        confirm: Explicitly authorize queueing after reviewing the preview.
        force: Rebuild even when status is already trustworthy/current.
    """
    started = perf_counter()
    if repo == "all":
        return _unsupported_repo_all("reindex_repository")

    ctx = await _resolve_repo_context(repo)
    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)
    index, status, _repo_state = await _status_payload(ctx, repository, started=started)
    active_jobs = await _active_jobs(ctx.session_factory, repository.id)
    trust = index["trust"]["search_results"]
    current_noop = trust == "trustworthy" and not force
    estimate = {
        "basis": "active_generation",
        "files": status.files_covered if status.manifest_state == "ok" else None,
        "chunks": status.expected_chunks if status.manifest_state == "ok" else None,
        "maximum_embeddings_for_active_generation": (
            status.expected_chunks if status.manifest_state == "ok" else None
        ),
        "checkout_changes_may_change_totals": True,
    }
    reindex: dict[str, Any] = {
        "force": force,
        "will_run": False,
        "confirmation_required": False,
        "active_jobs": active_jobs,
        "active_job_count": len(active_jobs),
        "cost": {
            "generative_calls": 0,
            "scope": "repository index_only pipeline and derived-index publication",
            "force_effect": "queue_when_current_only",
            "estimate": estimate,
        },
    }

    if active_jobs:
        operation_status = "already_running"
        reindex["job"] = active_jobs[-1]
    elif current_noop:
        operation_status = "current"
    elif not confirm:
        operation_status = "confirmation_required"
        reindex["confirmation_required"] = True
        reindex["next_action"] = "Retry with confirm=true after reviewing cost."
    else:
        queued = await queue_index_only_job(
            app_state=_job_runtime(ctx),
            session_factory=ctx.session_factory,
            repository_id=repository.id,
            force=force,
            executor=execute_job,
        )
        operation_status = queued.status
        reindex["will_run"] = queued.status == "accepted"
        reindex["job"] = {
            "id": queued.job_id,
            "state": queued.job_state,
            "mode": "index_only",
            "force": queued.force,
            "existing": queued.existing,
        }

    return {
        "status": operation_status,
        "repo": _repo_payload(ctx, repository),
        "index_status": index,
        "reindex": reindex,
        "_meta": _build_meta(
            timing_ms=(perf_counter() - started) * 1000.0,
            repository=repository,
        ),
    }


__all__ = ["get_index_status", "reindex_repository"]
