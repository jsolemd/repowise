"""Docs-owned payload compaction shared by both MCP transports."""

from __future__ import annotations

from collections.abc import Callable


def _truncate_list_field(container: dict[str, object], key: str, limit: int) -> None:
    """Clamp a list field while preserving the exact omitted count."""
    value = container.get(key)
    if not isinstance(value, list) or len(value) <= limit:
        return
    container[key] = list(value[:limit])
    container[f"{key}_more"] = len(value) - limit


def _compact_doc_hit(hit: dict[str, object]) -> dict[str, object]:
    if "chunk_id" not in hit and "id" in hit:
        hit["chunk_id"] = hit.pop("id")
    if hit.get("chunk_anchor") == hit.get("anchor"):
        hit.pop("chunk_anchor", None)
    if hit.get("library_name") == hit.get("library_id"):
        hit.pop("library_name", None)
    if hit.get("breadcrumb_text") == hit.get("title"):
        hit.pop("breadcrumb_text", None)
    hit.pop("source_url", None)
    _truncate_list_field(hit, "code_blocks", 1)
    code_blocks = hit.get("code_blocks")
    if isinstance(code_blocks, list):
        for block in code_blocks:
            if not isinstance(block, dict):
                continue
            if not block.get("title"):
                block.pop("title", None)
    return hit


def _compact_search_docs_payload(payload: dict[str, object]) -> dict[str, object]:
    payload.pop("read_doc_suggestion", None)
    _truncate_list_field(payload, "results", 6)
    results = payload.get("results")
    top_library_id = str(payload.get("library_id") or "").strip()
    if isinstance(results, list):
        for row in results:
            if isinstance(row, dict):
                if top_library_id and row.get("library_id") == top_library_id:
                    row.pop("library_id", None)
                _compact_doc_hit(row)
        if results and isinstance(results[0], dict):
            payload["recommended_start"] = {
                "library_id": top_library_id or results[0].get("library_id"),
                "file_path": results[0].get("file_path"),
                "anchor": results[0].get("anchor"),
                "title": results[0].get("title"),
                "chunk_type": results[0].get("chunk_type"),
                "score": results[0].get("score"),
            }

    related_sections = payload.get("related_sections")
    if isinstance(related_sections, list):
        compact_related: list[dict[str, object]] = []
        for idx, group in enumerate(related_sections):
            if idx >= 2:
                payload["related_sections_more"] = len(related_sections) - idx
                break
            if not isinstance(group, dict):
                continue
            compact_group: dict[str, object] = {}
            file_path = group.get("file_path")
            if file_path:
                compact_group["file_path"] = file_path
            sections = group.get("sections")
            if isinstance(sections, list):
                compact_group["sections"] = sections[:4]
                if len(sections) > 4:
                    compact_group["sections_more"] = len(sections) - 4
            if compact_group:
                compact_related.append(compact_group)
        payload["related_sections"] = compact_related

    _truncate_list_field(payload, "warnings", 4)
    _truncate_list_field(payload, "skipped_libraries", 4)
    _truncate_list_field(payload, "libraries", 8)
    libraries = payload.get("libraries")
    if isinstance(libraries, list):
        for row in libraries:
            if isinstance(row, dict) and row.get("library_name") == row.get("library_id"):
                row.pop("library_name", None)
    return payload


def _compact_expand_doc_chunk_payload(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("chunk_anchor") == payload.get("anchor"):
        payload.pop("chunk_anchor", None)
    payload.pop("source_url", None)
    _truncate_list_field(payload, "breadcrumb", 6)
    return payload


def _compact_list_doc_libraries_payload(payload: dict[str, object]) -> dict[str, object]:
    # An explicit `limit` means the caller sized their own page — honor it.
    # Only the unbounded listing gets clamped, and the clamp must keep the
    # pagination contract truthful: `returned`/`next_offset` must describe
    # what was actually emitted (agents page onward from these), not the
    # pre-clamp page ("returned: 83" alongside 12 rows).
    if payload.get("limit") is not None:
        return payload
    libraries = payload.get("libraries")
    before = len(libraries) if isinstance(libraries, list) else 0
    _truncate_list_field(payload, "libraries", 12)
    clamped = payload.get("libraries")
    after = len(clamped) if isinstance(clamped, list) else 0
    if after < before:
        payload["returned"] = after
        try:
            offset = int(payload.get("offset") or 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            offset = 0
        payload["next_offset"] = offset + after
    return payload


def _compact_read_doc_payload(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("truncated") is False:
        payload.pop("truncated", None)
    return payload


def _compact_update_doc_library_payload(payload: dict[str, object]) -> dict[str, object]:
    disposition = payload.pop("status", None) or payload.get("disposition")
    job_id = payload.pop("job_id", None)
    job_type = payload.pop("job_type", None)
    payload.pop("message", None)
    if disposition is not None:
        payload["disposition"] = disposition
    if job_id is not None:
        payload["job"] = {"id": job_id}
    if job_type is not None:
        payload.setdefault("job", {})
        payload["job"]["type"] = job_type
    return payload


def _compact_add_doc_library_payload(payload: dict[str, object]) -> dict[str, object]:
    disposition = payload.pop("status", None) or payload.get("disposition")
    job_id = payload.pop("job_id", None)
    job_status = payload.pop("job_status", None)
    payload.pop("message", None)
    if disposition is not None:
        payload["disposition"] = disposition
    if job_id is not None or job_status is not None:
        payload.setdefault("job", {})
        if job_id is not None:
            payload["job"]["id"] = job_id
        if job_status is not None:
            payload["job"]["disposition"] = job_status
    return payload


def _compact_delete_doc_library_payload(payload: dict[str, object]) -> dict[str, object]:
    disposition = payload.pop("status", None) or payload.get("disposition")
    payload.pop("message", None)
    if disposition is not None:
        payload["disposition"] = disposition
    return payload


def _compact_bundle_payload(payload: dict[str, object]) -> dict[str, object]:
    disposition = payload.pop("status", None) or payload.get("disposition")
    payload.pop("message", None)
    if disposition is not None:
        payload["disposition"] = disposition
    return payload


DOC_TOOL_COMPACTORS: dict[str, Callable[[dict[str, object]], dict[str, object]]] = {
    "add_doc_library": _compact_add_doc_library_payload,
    "delete_doc_library": _compact_delete_doc_library_payload,
    "expand_doc_chunk": _compact_expand_doc_chunk_payload,
    "export_doc_bundle": _compact_bundle_payload,
    "import_doc_bundle": _compact_bundle_payload,
    "list_doc_libraries": _compact_list_doc_libraries_payload,
    "read_doc": _compact_read_doc_payload,
    "search_docs": _compact_search_docs_payload,
    "search_docs_multi": _compact_search_docs_payload,
    "update_doc_library": _compact_update_doc_library_payload,
}
