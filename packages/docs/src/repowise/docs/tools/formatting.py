"""LLM-facing output formatting helpers for doc-search tools."""

from __future__ import annotations

import re
from pathlib import Path

API_REFERENCE_PATTERNS = [
    re.compile(r"/(reference|api-reference)/", re.IGNORECASE),
    re.compile(r"/sdk/", re.IGNORECASE),
    re.compile(r"/functions/", re.IGNORECASE),
    re.compile(r"/methods/", re.IGNORECASE),
    re.compile(r"/hooks/", re.IGNORECASE),
    re.compile(r"/components/", re.IGNORECASE),
]

API_QUERY_PATTERN = re.compile(
    r"^(?:"
    r"[A-Z][a-z]+[A-Z]|"
    r"[a-z]+_[a-z]+|"
    r"(?:use|get|set|create|delete|update|fetch|post|put|handle|on|is|has)[A-Z]"
    r")"
)

_DEFAULT_CODE_PREVIEW_CHARS = 480


def _extract_summary(content: str, max_chars: int = 200) -> str:
    """Extract the first substantive sentence from markdown content."""
    if not content:
        return ""

    paragraphs = re.split(r"\n\n+", content.strip())
    for para in paragraphs:
        para = para.strip()
        if len(para) > 20 and not para.startswith("#") and not para.startswith("```"):
            sentences = re.split(r"(?<=[.!?])\s+", para)
            first_sentence = sentences[0] if sentences else para
            if len(first_sentence) < 15:
                continue
            if len(first_sentence) > max_chars:
                return first_sentence[: max_chars - 3] + "..."
            return first_sentence
    return ""


def _build_preview(content: str, max_chars: int = 240) -> str:
    """Build a compact preview for search results without dumping whole chunks."""
    normalized = re.sub(r"\s+", " ", content.strip())
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    clipped = normalized[: max_chars - 3].rsplit(" ", 1)[0].rstrip()
    if not clipped:
        clipped = normalized[: max_chars - 3]
    return clipped + "..."


def _compact_code_block(
    code_block: dict,
    *,
    max_chars: int = _DEFAULT_CODE_PREVIEW_CHARS,
) -> dict:
    """Return a bounded code-block payload for search responses."""
    content = (code_block.get("content") or "").strip()
    preview = _build_preview(content, max_chars=max_chars)
    result = {
        "language": code_block.get("language", ""),
        "content": preview,
        "line_count": content.count("\n") + 1 if content else 0,
        "truncated": bool(content and preview != content),
    }
    title = code_block.get("title")
    if title:
        result["title"] = title
    return result


def _summary_repeats_content(summary: str, content: str) -> bool:
    """Return True when summary is just the opening of the content."""
    normalized_summary = re.sub(r"\s+", " ", summary.strip())
    normalized_content = re.sub(r"\s+", " ", content.strip())
    if not normalized_summary or not normalized_content:
        return False
    return normalized_content.startswith(normalized_summary)


def _is_api_like_query(query: str) -> bool:
    """Detect if a query looks like an API/function name lookup."""
    query = query.strip()
    words = query.split()
    if len(words) == 1 and len(query) >= 3:
        if API_QUERY_PATTERN.match(query):
            return True
        if "_" in query or (query[0].islower() and any(c.isupper() for c in query[1:])):
            return True
    return bool(len(words) >= 1 and API_QUERY_PATTERN.match(words[0]))


def _maybe_suggest_read_doc(
    query: str, results: list[dict], exact_match: bool = False
) -> dict | None:
    """Generate a read-doc suggestion when the search looks API-like."""
    if not results:
        return None

    if not (_is_api_like_query(query) or exact_match):
        return None

    top_result = results[0]
    score = top_result.get("score")
    file_path = top_result.get("file_path", "")
    title = top_result.get("title", "")

    if score is not None and score < 0.5:
        return None

    is_reference_doc = any(p.search(file_path) for p in API_REFERENCE_PATTERNS)
    if not is_reference_doc:
        return None

    api_name = query.split()[0] if query.split() else query
    return {
        "file_path": file_path,
        "reason": f"Full API reference available for '{api_name}'",
        "title": title,
        "action": "Use read_doc with this path for complete documentation",
    }


def _derive_title(breadcrumb: list[str], file_path: str) -> str:
    """Derive a human-readable title from breadcrumb or file path."""
    if breadcrumb:
        parts = [item.strip() for item in breadcrumb if item and item.strip()]
        if parts:
            return " > ".join(parts)

    if file_path:
        return Path(file_path).stem

    return "Untitled"


def _format_as_markdown(
    results: list[dict],
    query: str,
    library_id: str,
    warning: str | None = None,
    related_sections: list[dict] | None = None,
) -> str:
    """Format search results as LLM-friendly markdown."""
    lines = [f"## Search Results: {query}"]
    lines.append(f"Library: `{library_id}` | Found: {len(results)} results")

    if warning:
        lines.append(f"\n> **Warning:** {warning}")

    lines.append("")

    for i, r in enumerate(results, 1):
        title = r.get("title") or _derive_title(r.get("breadcrumb", []), r.get("file_path", ""))
        score = r.get("score")
        category = r.get("doc_category", "other")
        chunk_type = r.get("chunk_type", "doc")
        source = r.get("file_path", "unknown")
        library_name = r.get("library_name")
        preview = (r.get("preview") or _build_preview(r.get("content", ""))).strip()

        meta_parts = [f"Source: `{source}`"]
        if library_name:
            meta_parts.insert(0, f"Library: `{library_name}`")
        if category != "other":
            meta_parts.append(f"Category: {category}")
        meta_parts.append(f"Type: {chunk_type}")
        line_start = r.get("line_start")
        line_end = r.get("line_end")
        if line_start is not None and line_end is not None:
            meta_parts.append(f"Lines: {line_start}-{line_end}")
        if score is not None:
            meta_parts.append(f"Score: {score:.2f}")

        lines.append(f"### {i}. {title}")

        summary = r.get("summary", "")
        preview_content = r.get("content", "") or preview
        if summary and not _summary_repeats_content(summary, preview_content):
            lines.append(f"*{summary}*")
            lines.append("")

        lines.append(" | ".join(meta_parts))
        lines.append("")

        if preview:
            lines.append(preview)

        code_blocks = r.get("code_blocks") or r.get("codeBlocks", [])
        for cb in code_blocks:
            lang = cb.get("language", "")
            code_content = cb.get("content", "")
            if code_content:
                lines.append("")
                lines.append(f"```{lang}")
                lines.append(code_content)
                lines.append("```")
                if cb.get("truncated"):
                    lines.append(
                        "_Code preview truncated. Use `read_doc` or `expand_doc_chunk` for full context._"
                    )

        lines.append("")
        if i < len(results):
            lines.append("---")
            lines.append("")

    if related_sections:
        lines.append("### Related Sections")
        lines.append("These sections appear in the same files as the results above:")
        lines.append("")
        for group in related_sections[:3]:
            file_path = str(group.get("file_path") or "")
            sections = group.get("sections") or []
            if not file_path:
                continue
            lines.append(f"**{file_path}:**")
            for sec in sections[:5]:
                sec_title = sec.get("breadcrumb_text", "") or sec.get("section_anchor", "")
                if sec_title:
                    lines.append(f"  - {sec_title}")
            lines.append("")

    return "\n".join(lines)
