"""MDX document chunking with JSX stripping."""

from __future__ import annotations

import logging
import re
from re import Pattern

from repowise.docs.chunking.extractors.markdown import (
    chunk_markdown,
    extract_code_blocks,
    restore_code_blocks,
)
from repowise.docs.chunking.models import DocChunk

logger = logging.getLogger(__name__)


# ============================================================================
# JSX stripping
# ============================================================================

# Import statements: import { X } from "path" or import X from "path"
IMPORT_PATTERN: Pattern[str] = re.compile(
    r"^import\s+.*?\s+from\s+['\"].*?['\"];?\s*$",
    re.MULTILINE,
)

# Export statements: export { X } or export default X
EXPORT_PATTERN: Pattern[str] = re.compile(
    r"^export\s+(?:default\s+)?(?:\{[^}]*\}|[a-zA-Z_]\w*);?\s*$",
    re.MULTILINE,
)

_EXACT_RESIDUE_LINES = {"} />", "/>", "</>", "<>"}
_RESIDUE_TOKENS = ("/>", "{", "}")


def strip_jsx(content: str) -> str:
    """
    Strip JSX syntax from MDX content, preserving markdown and text.

    Requirements:
    - Protect fenced code blocks (don't strip JSX inside fences)
    - Use a stateful tag scanner so '>' inside { ... } or quotes doesn't break parsing
    - Remove residue lines like `} />` that can remain after stripping
    """
    original_len = len(content)

    # Protect fenced code blocks first so JSX examples survive.
    content, code_blocks = extract_code_blocks(content)

    # Remove import/export statements (outside fences, since fences are placeholders now).
    content = IMPORT_PATTERN.sub("", content)
    content = EXPORT_PATTERN.sub("", content)

    # Strip JSX component tags (stateful scanner).
    content = _strip_component_tags_stateful(content)

    # Remove common MDX residue lines left behind by partial JSX stripping.
    content = _cleanup_residue_lines(content)

    # Clean up formatting (still safe because code blocks are placeholders).
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"^\s+$", "", content, flags=re.MULTILINE)

    # Restore fenced code blocks.
    content = restore_code_blocks(content, code_blocks)

    stripped = original_len - len(content)
    if stripped > 0:
        logger.debug(f"Stripped {stripped} characters of JSX from MDX content")

    return content.strip()


def _is_escaped(text: str, idx: int) -> bool:
    backslashes = 0
    i = idx - 1
    while i >= 0 and text[i] == "\\":
        backslashes += 1
        i -= 1
    return backslashes % 2 == 1


def _looks_like_component_tag_start(text: str, start: int) -> bool:
    """Return True if text[start:] begins with a JSX component tag (<Component ...>)."""
    if start >= len(text) or text[start] != "<":
        return False

    i = start + 1
    while i < len(text) and text[i].isspace():
        i += 1

    if i < len(text) and text[i] == "/":
        i += 1
        while i < len(text) and text[i].isspace():
            i += 1

    if i >= len(text) or not text[i].isupper():
        return False

    # First segment
    i += 1
    while i < len(text) and (text[i].isalnum() or text[i] == "_"):
        i += 1

    # Optional dot-notation segments (e.g., Cards.Card)
    while i + 1 < len(text) and text[i] == "." and text[i + 1].isupper():
        i += 2
        while i < len(text) and (text[i].isalnum() or text[i] == "_"):
            i += 1

    return True


def _find_tag_end(text: str, start: int) -> int | None:
    """Find the closing '>' for a tag starting at start, respecting braces and quotes."""
    brace_depth = 0
    quote_char: str | None = None

    i = start + 1
    while i < len(text):
        ch = text[i]

        if quote_char is not None:
            if ch == quote_char and not _is_escaped(text, i):
                quote_char = None
            i += 1
            continue

        if ch in ("'", '"'):
            quote_char = ch
            i += 1
            continue

        if ch == "{":
            brace_depth += 1
            i += 1
            continue

        if ch == "}":
            if brace_depth > 0:
                brace_depth -= 1
            i += 1
            continue

        if ch == ">" and brace_depth == 0:
            return i

        i += 1

    return None


def _strip_component_tags_stateful(content: str) -> str:
    """Remove JSX component tags while preserving their children."""
    out: list[str] = []
    i = 0

    while i < len(content):
        if content[i] == "<" and _looks_like_component_tag_start(content, i):
            end = _find_tag_end(content, i)
            if end is None:
                out.append(content[i])
                i += 1
                continue

            # Drop tag entirely.
            i = end + 1
            continue

        out.append(content[i])
        i += 1

    return "".join(out)


def _should_drop_residue_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    if stripped in _EXACT_RESIDUE_LINES:
        return True

    if any(tok in stripped for tok in _RESIDUE_TOKENS):
        alnum = sum(ch.isalnum() for ch in stripped)
        density = alnum / len(stripped) if stripped else 0.0
        if density < 0.2:
            return True

    return False


def _cleanup_residue_lines(content: str) -> str:
    lines = content.splitlines()
    kept: list[str] = []
    for line in lines:
        if _should_drop_residue_line(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def chunk_mdx(
    content: str,
    file_path: str,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
) -> list[DocChunk]:
    """
    Chunk an MDX document by stripping JSX and parsing as markdown.

    Strategy:
    1. Strip JSX syntax (imports, components, etc.)
    2. Pass cleaned content to markdown chunker

    Args:
        content: Raw MDX content
        file_path: Relative file path within repo
        library_id: Library identifier (e.g., "/langfuse/langfuse")
        commit_sha: Git commit SHA
        repo: Optional repo path for URL generation
        branch: Git branch name

    Returns:
        List of DocChunk objects
    """
    # Strip JSX
    stripped = strip_jsx(content)

    # Use markdown chunker on cleaned content
    return chunk_markdown(
        stripped,
        file_path,
        library_id,
        commit_sha,
        repo,
        branch,
        source_path_prefix,
    )
