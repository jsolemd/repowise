"""Chunk text-like documents (txt/html/xml) via markdown normalization."""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

from repowise.docs.chunking.extractors.markdown import chunk_markdown
from repowise.docs.chunking.models import DocChunk
from repowise.docs.formats import HTML_DOC_EXTENSIONS, XML_DOC_EXTENSIONS

MARKDOWN_HEADER_PATTERN = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
SCRIPT_STYLE_PATTERN = re.compile(
    r"<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL
)
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING_PATTERN = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
PRE_CODE_PATTERN = re.compile(
    r"<pre\b[^>]*>\s*<code\b([^>]*)>(.*?)</code>\s*</pre>", re.IGNORECASE | re.DOTALL
)
PRE_PATTERN = re.compile(r"<pre\b[^>]*>(.*?)</pre>", re.IGNORECASE | re.DOTALL)
BREAK_PATTERN = re.compile(r"<\s*br\s*/?>", re.IGNORECASE)
BLOCK_CLOSE_PATTERN = re.compile(
    r"</\s*(p|div|section|article|li|ul|ol|tr|table|blockquote)\s*>", re.IGNORECASE
)
TAG_PATTERN = re.compile(r"<[^>]+>")
LANG_CLASS_PATTERN = re.compile(r"(?:lang|language)-([A-Za-z0-9_+-]+)", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\n{3,}")


def _extract_language(attrs: str) -> str:
    match = LANG_CLASS_PATTERN.search(attrs)
    return match.group(1).lower() if match else ""


def _normalize_html_like(content: str) -> str:
    normalized = HTML_COMMENT_PATTERN.sub("", content)
    normalized = SCRIPT_STYLE_PATTERN.sub("", normalized)

    def replace_heading(match: re.Match[str]) -> str:
        level = int(match.group(1))
        text = TAG_PATTERN.sub("", match.group(2))
        text = " ".join(text.split())
        if not text:
            return ""
        return f"\n{'#' * level} {unescape(text)}\n"

    normalized = HEADING_PATTERN.sub(replace_heading, normalized)

    def replace_pre_code(match: re.Match[str]) -> str:
        attrs = match.group(1) or ""
        code = unescape(match.group(2).strip("\n"))
        language = _extract_language(attrs)
        fence = f"```{language}" if language else "```"
        return f"\n{fence}\n{code}\n```\n"

    normalized = PRE_CODE_PATTERN.sub(replace_pre_code, normalized)

    def replace_pre(match: re.Match[str]) -> str:
        code = unescape(TAG_PATTERN.sub("", match.group(1)).strip("\n"))
        return f"\n```\n{code}\n```\n"

    normalized = PRE_PATTERN.sub(replace_pre, normalized)
    normalized = BREAK_PATTERN.sub("\n", normalized)
    normalized = BLOCK_CLOSE_PATTERN.sub("\n", normalized)
    normalized = TAG_PATTERN.sub("", normalized)
    normalized = unescape(normalized)
    normalized = WHITESPACE_PATTERN.sub("\n\n", normalized)

    return normalized.strip()


def _inject_title_if_missing(content: str, file_path: str) -> str:
    if MARKDOWN_HEADER_PATTERN.search(content):
        return content

    stem = Path(file_path).stem.replace("-", " ").replace("_", " ").strip()
    title = stem if stem else "Document"
    return f"# {title}\n\n{content.strip()}"


def chunk_text_document(
    content: str,
    file_path: str,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
) -> list[DocChunk]:
    """Chunk text-like docs and preserve structure where possible."""
    extension = Path(file_path).suffix.lower()
    normalized = content

    if extension in HTML_DOC_EXTENSIONS or extension in XML_DOC_EXTENSIONS:
        normalized = _normalize_html_like(content)

    normalized = _inject_title_if_missing(normalized, file_path)

    return chunk_markdown(
        normalized,
        file_path,
        library_id,
        commit_sha,
        repo,
        branch,
        source_path_prefix,
    )
