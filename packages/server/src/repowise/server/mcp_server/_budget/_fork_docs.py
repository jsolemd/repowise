"""Keep documentation content and pagination honest after response shedding."""

from typing import Any

from repowise.server.mcp_server._budget.collector import OmissionCollector


def trim_document_content(result: dict[str, Any], collector: OmissionCollector, limit: int) -> None:
    payload = result.get("payload", result)
    content = payload.get("content")
    if isinstance(content, str) and len(content) > limit // 2:
        keep = max(0, limit // 2)
        collector.add("Remaining documentation content", content[keep:])
        payload["content"] = content[:keep]
        payload["truncated"] = True
        if "estimated_tokens" in payload:
            payload["estimated_tokens"] = len(payload["content"]) // 4


def sync_document_page(result: dict[str, Any]) -> None:
    payload = result.get("payload", {})
    files, page = payload.get("files"), payload.get("pagination")
    if not isinstance(files, list) or not isinstance(page, dict):
        return
    shown = len(files)
    offset, total = page["offset"], page["total"]
    page.update(returned=shown, has_more=offset + shown < total)
    page["next_offset"] = offset + shown if shown and page["has_more"] else None
    if not shown and page["has_more"]:
        result["next_action"] = (
            "The response budget omitted this page; retry list_doc_files with a smaller limit."
        )
