"""AsciiDoc chunking by normalizing structure to markdown-compatible syntax."""

from __future__ import annotations

import re

from repowise.docs.chunking.extractors.markdown import chunk_markdown
from repowise.docs.chunking.models import DocChunk

# AsciiDoc section titles: = title, == section, === subsection, ...
SECTION_TITLE_PATTERN = re.compile(r"^(={1,6})\s+(.+?)\s*$")
# Attribute line before a block, e.g. [source,cypher,role=noplay]
ATTRIBUTE_LINE_PATTERN = re.compile(r"^\[(?P<attrs>[^\]]+)\]\s*$")
# Block delimiters used for listing/source blocks.
BLOCK_DELIMITER_PATTERN = re.compile(r"^\s*(?P<delim>-{4,}|\.{4,})\s*$")
# Explicit anchors, e.g. [[algorithms-page-rank]]
ANCHOR_PATTERN = re.compile(r"^\s*\[\[[^\]]+\]\]\s*$")


def _extract_source_language(attrs: str | None) -> str:
    """Extract language hint from an AsciiDoc block attribute list."""
    if not attrs:
        return ""

    parts = [part.strip() for part in attrs.split(",")]
    if not parts:
        return ""

    # [source,java], [source], [,java], [source,cypher,role=noplay]
    if parts[0] == "source":
        if len(parts) > 1 and parts[1] and "=" not in parts[1]:
            return parts[1]
        return ""

    if parts[0] == "" and len(parts) > 1 and parts[1] and "=" not in parts[1]:
        return parts[1]

    return ""


def _normalize_asciidoc(content: str) -> str:
    """
    Convert key AsciiDoc constructs to markdown-like syntax for chunking.

    We intentionally normalize only the pieces that drive chunk quality:
    section titles and source/listing blocks. The markdown chunker then handles
    hierarchy, breadcrumbs, and code chunk extraction.
    """
    lines = content.splitlines()
    out: list[str] = []

    in_block = False
    block_delim_char = ""
    pending_attrs: str | None = None

    for line in lines:
        if in_block:
            delimiter_match = BLOCK_DELIMITER_PATTERN.match(line)
            if delimiter_match and delimiter_match.group("delim")[0] == block_delim_char:
                out.append("```")
                in_block = False
                block_delim_char = ""
            else:
                out.append(line)
            continue

        if ANCHOR_PATTERN.match(line):
            continue

        attribute_match = ATTRIBUTE_LINE_PATTERN.match(line.strip())
        if attribute_match:
            pending_attrs = attribute_match.group("attrs")
            continue

        delimiter_match = BLOCK_DELIMITER_PATTERN.match(line)
        if delimiter_match:
            block_delim_char = delimiter_match.group("delim")[0]
            language = _extract_source_language(pending_attrs)
            out.append(f"```{language}" if language else "```")
            in_block = True
            pending_attrs = None
            continue

        if pending_attrs and line.strip():
            # Attribute line wasn't attached to a block delimiter; drop it.
            pending_attrs = None

        section_match = SECTION_TITLE_PATTERN.match(line)
        if section_match:
            level = min(len(section_match.group(1)), 6)
            title = section_match.group(2).strip()
            out.append(f"{'#' * level} {title}")
            continue

        # AsciiDoc single-line comments are typically authoring metadata noise.
        if line.lstrip().startswith("//"):
            continue

        out.append(line)

    if in_block:
        out.append("```")

    return "\n".join(out)


def chunk_asciidoc(
    content: str,
    file_path: str,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
) -> list[DocChunk]:
    """Chunk an AsciiDoc document via structure-preserving normalization."""
    normalized = _normalize_asciidoc(content)
    return chunk_markdown(
        normalized,
        file_path,
        library_id,
        commit_sha,
        repo,
        branch,
        source_path_prefix,
    )
