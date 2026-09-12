"""Stable MCP response envelope owned by the documentation service."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any

from mcp.types import TextContent

from repowise.docs.server.payload_compact import DOC_TOOL_COMPACTORS

OUTPUT_JSON = "json"
OUTPUT_MARKDOWN = "markdown"
_DROPPED_FACET_KEYS = frozenset({"search_hints", "asset_refs"})
_NOISY_TRUE_FACET_PREFIXES = ("uses_", "has_use_")
_SMALL_COMPLEXITY_KEYS = frozenset({"cyclomatic", "cognitive", "lines"})
_GENERIC_LIST_LIMITS: dict[str, int] = {
    "best_signals": 8,
    "bridge_files_checked": 8,
    "bridge_files_used": 6,
    "call_lines": 8,
    "css_at_rules": 6,
    "css_selector_targets": 10,
    "css_selectors": 8,
    "css_var_defs": 10,
    "css_var_refs": 10,
    "domain_facets": 8,
    "evidence_policy": 6,
    "feature_entrypoints": 8,
    "findings": 4,
    "import_packages": 6,
    "library_tags": 6,
    "most_imported_modules": 10,
    "next_tool_chain": 4,
    "pending_files": 10,
    "ranking_features": 6,
    "recent_indexed_files": 10,
    "route_paths": 8,
    "shared_surfaces": 6,
    "sql_tables": 8,
    "sync_history": 4,
    "tailwind_classes": 8,
    "top_functions": 8,
    "ui_primitives": 8,
    "verification_policy": 6,
    "why_selected": 6,
}


def output_mode(
    tool_schema_by_name: dict[str, dict],
    tool_name: str,
    arguments: dict | None,
) -> str:
    """Resolve a call's output mode from the canonical documentation schema."""
    schema = tool_schema_by_name.get(tool_name) or {}
    properties = schema.get("properties") if isinstance(schema, dict) else {}
    output_property = properties.get("output") if isinstance(properties, dict) else {}
    default_mode = OUTPUT_JSON
    allowed_modes = {OUTPUT_MARKDOWN, OUTPUT_JSON}
    if isinstance(output_property, dict):
        configured = str(output_property.get("default", OUTPUT_JSON)).strip().lower()
        enum_values = output_property.get("enum")
        if isinstance(enum_values, list):
            allowed_modes = {
                str(value).strip().lower()
                for value in enum_values
                if str(value).strip().lower() in {OUTPUT_MARKDOWN, OUTPUT_JSON}
            } or allowed_modes
        if configured in allowed_modes:
            default_mode = configured
    if not isinstance(arguments, dict):
        return default_mode
    requested = str(arguments.get("output", default_mode)).strip().lower()
    return requested if requested in allowed_modes else default_mode


def _to_jsonable(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(child) for child in value]
    return value


def _normalize_path(path: str) -> str:
    normalized = str(path or "").replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    prefix = ".claude/worktrees/"
    if normalized.lower().startswith(prefix):
        parts = normalized.split("/", 3)
        if len(parts) == 4:
            return parts[3]
    return normalized


def _is_test_path(path: str) -> bool:
    normalized = _normalize_path(path).lower()
    if normalized.endswith("_test.py"):
        return True
    if any(segment in normalized for segment in ("/__tests__/", "/__test__/", "/test/", "/tests/")):
        return True
    if normalized.startswith(("test/", "tests/")):
        return True
    filename = normalized.rsplit("/", 1)[-1]
    return any(part in {"test", "spec"} for part in filename.split(".")[:-1])


def _is_non_source_path(path: str) -> bool:
    normalized = _normalize_path(path).lower()
    return normalized.endswith((".md", ".mdx", ".txt", ".rst")) or any(
        segment in normalized for segment in ("/docs/", "/doc/", "/research/", "/artifacts/")
    )


def _classify_role(path: str, row: dict[str, Any]) -> tuple[str, list[str]]:
    normalized = _normalize_path(path)
    lowered = normalized.lower()
    if not normalized:
        return "unknown", ["missing_path"]
    if _is_test_path(normalized):
        role, reasons = "test_evidence", ["test_path"]
    elif any(
        segment in f"/{lowered.strip('/')}/"
        for segment in ("/generated/", "/dist/", "/build/", "/.next/", "/node_modules/")
    ):
        role, reasons = "generated_or_cache", ["generated_path"]
    elif any(
        segment in f"/{lowered.strip('/')}/"
        for segment in ("/logs/", "/artifacts/", "/tmp/", "/cache/")
    ):
        role, reasons = "runtime_evidence", ["runtime_path"]
    elif lowered.startswith("content/") and lowered.endswith((".md", ".mdx", ".markdown", ".txt")):
        role, reasons = "content_owner", ["content_source_path"]
    elif _is_non_source_path(normalized):
        role, reasons = "docs_context", ["non_source_path"]
    else:
        filename = lowered.rsplit("/", 1)[-1]
        config_names = {
            "dockerfile",
            "compose.yaml",
            "compose.yml",
            "docker-compose.yaml",
            "docker-compose.yml",
            "pyproject.toml",
            "package.json",
            "tsconfig.json",
        }
        if filename in config_names or filename.endswith(
            (".env", ".ini", ".json", ".toml", ".yaml", ".yml")
        ):
            role, reasons = "config_owner", ["config_path"]
        else:
            role, reasons = "implementation_owner", ["source_path"]
    facets = row.get("file_facets") if isinstance(row.get("file_facets"), dict) else row
    if facets.get("route_paths"):
        reasons.append("route_surface")
    if facets.get("sql_tables"):
        reasons.append("query_surface")
    if facets.get("backend_role") or facets.get("frontend_role"):
        reasons.append("role_facets")
    return role, sorted(set(reasons))


def _results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("results")
    if isinstance(rows, list):
        typed = [row for row in rows if isinstance(row, dict)]
        if typed:
            return typed
    return []


def _recommended(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("recommended_start")
    return value if isinstance(value, dict) else None


def _select_result(
    recommended: dict[str, Any] | None,
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not results:
        return recommended
    if not recommended:
        return results[0]
    target_file = str(recommended.get("file") or recommended.get("file_path") or "").strip()
    target_name = str(recommended.get("symbol") or recommended.get("name") or "").strip()
    for row in results:
        row_file = str(row.get("file") or row.get("file_path") or "").strip()
        row_name = str(row.get("name") or row.get("symbol") or "").strip()
        if target_file and row_file == target_file and (not target_name or row_name == target_name):
            return row
    return results[0]


def _explicit_target(payload: dict[str, Any]) -> dict[str, Any] | None:
    source = str(payload.get("source") or "").strip()
    source_match = re.fullmatch(r"(?P<file>[^\n]+\.[A-Za-z0-9]+)(?::\d+(?:-\d+)?)?", source)
    file_path = str(
        payload.get("file")
        or payload.get("file_path")
        or payload.get("target_file")
        or payload.get("source_file")
        or (source_match.group("file") if source_match else "")
        or ""
    ).strip()
    if not file_path:
        return None
    return {"file": file_path, "retrieval_rank_features": ["explicit_tool_target"]}


def _attach_evidence_card(
    *, tool_name: str, payload: dict[str, Any], confidence: str, next_action: str
) -> None:
    _ = next_action
    if isinstance(payload.get("evidence_card"), dict):
        return
    results = _results(payload)
    recommended = _recommended(payload)
    selected = _select_result(recommended, results)
    selected_path = str(
        (selected or {}).get("file") or (selected or {}).get("file_path") or ""
    ).strip()
    explicit = None
    if not selected_path:
        explicit = _explicit_target(payload)
        if explicit:
            selected = explicit
            selected_path = str(explicit["file"])
    role, role_reasons = _classify_role(selected_path, selected or {})
    selected_owner: dict[str, object] = {"role": role, "role_reasons": role_reasons}
    if selected_path:
        selected_owner = {"file": selected_path, **selected_owner}
    selected_name = str(
        (selected or {}).get("name") or (selected or {}).get("symbol") or ""
    ).strip()
    if selected_name:
        selected_owner["symbol"] = selected_name

    findings: list[dict[str, str]] = []
    if str(confidence).strip().lower() == "low":
        findings.append(
            {
                "severity": "warn",
                "code": "low_confidence_decision",
                "message": "Tool confidence is low; verify before editing.",
            }
        )
    verifier: dict[str, object] = {"status": "warn" if findings else "pass"}
    if findings:
        verifier["findings"] = findings

    card: dict[str, object] = {
        "tool": tool_name,
        "selected_owner": selected_owner,
    }
    why: list[str] = []
    if isinstance(recommended, dict):
        recommended_why = str(recommended.get("why") or "").strip()
        if recommended_why:
            why.append(recommended_why)
        why_not_top = str(recommended.get("why_not_top_ranked") or "").strip()
        if why_not_top:
            why.append(f"why_not_top_ranked: {why_not_top}")
    why.extend(
        str(value).strip()
        for value in (selected or {}).get("retrieval_rank_features", []) or []
        if str(value).strip()
    )
    rank_explain = (selected or {}).get("rank_explain")
    if isinstance(rank_explain, dict):
        why.extend(
            f"matched_on:{str(value).strip()}"
            for value in rank_explain.get("matched_on", []) or []
            if str(value).strip()
        )
    if why:
        card["why_selected"] = list(dict.fromkeys(why))[:8]
    card["verifier"] = verifier
    payload["evidence_card"] = card


def _truncate_list_field(container: dict[str, object], key: str, limit: int) -> None:
    value = container.get(key)
    if not isinstance(value, list) or len(value) <= limit:
        return
    container[key] = list(value[:limit])
    container[f"{key}_more"] = len(value) - limit


def _nonzero_numeric(value: object) -> bool:
    try:
        return float(value or 0.0) != 0.0
    except (TypeError, ValueError):
        return bool(value)


def _compact_facets(facets: dict[str, object]) -> dict[str, object]:
    compacted: dict[str, object] = {}
    for key, raw in facets.items():
        if key in _DROPPED_FACET_KEYS:
            continue
        if isinstance(raw, bool):
            if not raw:
                continue
            if key.startswith(_NOISY_TRUE_FACET_PREFIXES) or key in {
                "has_use_client",
                "has_use_server",
                "has_server_only",
                "has_metadata_export",
                "has_viewport_export",
            }:
                continue
        child = _compact_node(raw)
        if child is None or (isinstance(child, str) and not child.strip()):
            continue
        compacted[key] = child
    for key, limit in _GENERIC_LIST_LIMITS.items():
        _truncate_list_field(compacted, key, limit)
    return compacted


def _compact_rank_explain(rank_explain: dict[str, object]) -> dict[str, object]:
    compacted: dict[str, object] = {}
    matched_on = rank_explain.get("matched_on")
    if matched_on:
        compacted["matched_on"] = matched_on
    if rank_explain.get("path_only"):
        compacted["path_only"] = True
    source_bucket = str(rank_explain.get("source_bucket", "") or "").strip()
    if source_bucket and source_bucket != "source":
        compacted["source_bucket"] = source_bucket
    if rank_explain.get("file_scope_mode"):
        compacted["file_scope_mode"] = rank_explain["file_scope_mode"]
    if rank_explain.get("file_scope_rank") is not None:
        compacted["file_scope_rank"] = rank_explain["file_scope_rank"]
    try:
        gds_bonus = float(rank_explain.get("gds_bonus") or 0.0)
    except (TypeError, ValueError):
        gds_bonus = 0.0
    if gds_bonus > 0:
        compacted["gds_bonus"] = gds_bonus
    try:
        quality_score = float(rank_explain.get("quality_score") or 0.0)
    except (TypeError, ValueError):
        quality_score = 0.0
    if quality_score not in {0.0, 1.0}:
        compacted["quality_score"] = quality_score
    file_signals = rank_explain.get("file_signals")
    if isinstance(file_signals, list) and file_signals:
        compacted["signal_preview"] = list(file_signals[:6])
        if len(file_signals) > 6:
            compacted["signal_preview_more"] = len(file_signals) - 6
    return compacted


def _compact_dict(value: dict[str, object]) -> dict[str, object]:
    if value and set(value).issubset(_SMALL_COMPLEXITY_KEYS):
        compact_small = {
            key: raw for key, raw in value.items() if _nonzero_numeric(raw) or key == "cyclomatic"
        }
        if compact_small == {"cyclomatic": 0}:
            return {}
        return compact_small
    if value.get("simple_name") == value.get("name"):
        value.pop("simple_name", None)
    if value.get("response_confidence") == value.get("confidence"):
        value.pop("response_confidence", None)
    if isinstance(value.get("file_facets"), dict):
        facets = _compact_facets(dict(value["file_facets"]))
        value["file_facets"] = facets
        for key in list(value):
            if key != "file_facets" and key in facets:
                value.pop(key, None)
    if isinstance(value.get("file_metadata"), dict):
        value["file_metadata"] = _compact_facets(dict(value["file_metadata"]))
    if isinstance(value.get("rank_explain"), dict):
        rank_explain = _compact_rank_explain(dict(value["rank_explain"]))
        if rank_explain:
            value["rank_explain"] = rank_explain
        else:
            value.pop("rank_explain", None)
    if isinstance(value.get("heuristics"), dict) and value.get("probable_call_sites") == value[
        "heuristics"
    ].get("probable_call_sites"):
        value.pop("probable_call_sites", None)
    if isinstance(value.get("heuristics"), dict) and value.get("backstop_error") == value[
        "heuristics"
    ].get("backstop_error"):
        value.pop("backstop_error", None)
    for key, limit in _GENERIC_LIST_LIMITS.items():
        _truncate_list_field(value, key, limit)
    return value


def _compact_node(value: object) -> object:
    if isinstance(value, dict):
        compacted: dict[str, object] = {}
        for key, raw in value.items():
            child = _compact_node(raw)
            if child is None or (isinstance(child, str) and not child.strip()):
                continue
            if isinstance(child, dict) and not child:
                continue
            compacted[str(key)] = child
        return _compact_dict(compacted)
    if isinstance(value, (list, tuple)):
        return [_compact_node(child) for child in value if child is not None]
    return value


def compact_payload(tool_name: str, payload: object | None) -> object | None:
    if not isinstance(payload, dict):
        return payload
    compacted = _compact_node(payload)
    handler = DOC_TOOL_COMPACTORS.get(tool_name)
    if handler is not None and isinstance(compacted, dict):
        return handler(compacted)
    return compacted


def render_response(
    *,
    tool_name: str,
    status: str,
    confidence: str,
    next_action: str,
    body: str,
    output: str,
    payload: object | None = None,
) -> str:
    if output == OUTPUT_JSON:
        jsonable = _to_jsonable(payload) if payload is not None else None
        if isinstance(jsonable, dict):
            _attach_evidence_card(
                tool_name=tool_name,
                payload=jsonable,
                confidence=confidence,
                next_action=next_action,
            )
        return json.dumps(
            {
                "status": status,
                "tool": tool_name,
                "confidence": confidence,
                "next_action": next_action,
                "payload": compact_payload(tool_name, jsonable),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return "\n".join(
        [
            "# Tool Response",
            "",
            f"- status: `{status}`",
            f"- tool: `{tool_name}`",
            f"- confidence: `{confidence}`",
            f"- next_action: {next_action}",
            "",
            body.strip(),
        ]
    ).strip()


def respond(
    ctx: Any,
    *,
    tool_name: str,
    status: str,
    confidence: str,
    next_action: str,
    body: str,
    output: str,
    payload: object | None = None,
    include_scope: bool = True,
) -> list[TextContent]:
    payload_with_scope = payload
    if include_scope and isinstance(payload, dict):
        payload_with_scope = dict(payload)
        project = str(getattr(getattr(ctx, "settings", None), "project", "solemd.infra"))
        payload_with_scope.setdefault("scope", {"project": project})
    return [
        TextContent(
            type="text",
            text=render_response(
                tool_name=tool_name,
                status=status,
                confidence=confidence,
                next_action=next_action,
                body=body,
                output=output,
                payload=payload_with_scope,
            ),
        )
    ]


def error_payload(
    *,
    code: str,
    message: str,
    details: dict | None = None,
    context: dict | None = None,
) -> dict:
    payload: dict[str, object] = {"error": message, "error_code": code}
    if details:
        payload["error_details"] = details
    if context:
        payload.update(context)
    return payload


def respond_error(
    ctx: Any,
    *,
    tool_name: str,
    confidence: str,
    next_action: str,
    output: str,
    code: str,
    message: str,
    details: dict | None = None,
    context: dict | None = None,
    include_scope: bool = True,
) -> list[TextContent]:
    body = [f"Error (`{code}`): {message}"]
    if details:
        body.extend(("", json.dumps(_to_jsonable(details), indent=2)))
    return respond(
        ctx,
        tool_name=tool_name,
        status="error",
        confidence=confidence,
        next_action=next_action,
        body="\n".join(body),
        output=output,
        payload=error_payload(code=code, message=message, details=details, context=context),
        include_scope=include_scope,
    )


__all__ = ["output_mode", "render_response", "respond", "respond_error"]
