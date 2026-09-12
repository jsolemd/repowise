"""Markdown document chunking with header-based splitting."""

import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from repowise.docs.chunking.extractors.text_split import (
    estimate_tokens,
    split_large_section,
    split_large_section_with_line_spans,
)
from repowise.docs.chunking.models import (
    ChunkType,
    DocChunk,
    append_anchor,
    build_source_url,
    slugify,
)
from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)

# Regex patterns
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# Code block pattern: handles language, title, and highlight markers like /pattern/
# Examples: ```python, ```python title="example.py", ```python /highlight/
CODE_BLOCK_PATTERN = re.compile(
    r"```(\w+)?(?:\s+title=[\"']([^\"']+)[\"'])?[^\n]*\n(.*?)```",
    re.DOTALL,
)
GARBAGE_CHUNK_PATTERN = re.compile(r"^[\s{}/<>]+$")
ALNUM_PATTERN = re.compile(r"[A-Za-z0-9]")


@dataclass
class Section:
    """A markdown section defined by a header."""

    level: int  # Header level (1-6)
    title: str
    anchor: str
    content: str = ""
    line_start: int = 0
    code_blocks: list["CodeBlock"] = field(default_factory=list)


@dataclass
class CodeBlock:
    """A fenced code block within a section."""

    language: str
    title: str
    content: str
    start_pos: int  # Position in original content


@dataclass
class ParsedMarkdown:
    """Result of parsing a markdown document."""

    frontmatter: dict[str, str]
    sections: list[Section]
    title: str  # Extracted from frontmatter or first H1


def _stringify_frontmatter_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _normalize_frontmatter(parsed: Any) -> dict[str, str] | None:
    if not isinstance(parsed, dict):
        return None
    return {str(key): _stringify_frontmatter_value(value) for key, value in parsed.items()}


def _parse_frontmatter_block(raw_frontmatter: str) -> dict[str, str]:
    try:
        parsed_yaml = yaml.safe_load(raw_frontmatter) or {}
    except yaml.YAMLError as error:
        logger.debug("Failed to parse frontmatter as YAML; trying TOML: %s", error)
    else:
        normalized = _normalize_frontmatter(parsed_yaml)
        if normalized is not None:
            return normalized

    try:
        parsed_toml = tomllib.loads(raw_frontmatter)
    except tomllib.TOMLDecodeError as error:
        logger.warning("Failed to parse frontmatter as YAML mapping or TOML: %s", error)
        return {}

    normalized = _normalize_frontmatter(parsed_toml)
    if normalized is None:
        logger.warning("Ignored frontmatter that did not parse to a mapping")
        return {}
    return normalized


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """
    Extract YAML frontmatter from markdown content.

    Args:
        content: Raw markdown content

    Returns:
        Tuple of (frontmatter dict, content without frontmatter)
    """
    if not content.startswith("---"):
        return {}, content

    # Safety: avoid expensive regex scanning on very large documents.
    if len(content) > 100_000:
        first_chunk = content[:10_000]
        if first_chunk.count("---") < 2:
            return {}, content
        content_to_match = first_chunk
    else:
        content_to_match = content

    match = FRONTMATTER_PATTERN.match(content_to_match)
    if not match:
        return {}, content

    frontmatter = _parse_frontmatter_block(match.group(1))

    # Return content after frontmatter
    remaining = content[match.end() :].lstrip()
    return frontmatter, remaining


def extract_code_blocks(content: str) -> tuple[str, list[CodeBlock]]:
    """
    Extract code blocks from markdown content.

    Replaces code blocks with placeholders to simplify section splitting.

    Args:
        content: Markdown content with code blocks

    Returns:
        Tuple of (content with placeholders, list of code blocks)
    """
    code_blocks: list[CodeBlock] = []

    def replace_block(match: re.Match) -> str:
        language = match.group(1) or ""
        title = match.group(2) or ""
        code_content = match.group(3).rstrip()

        code_blocks.append(
            CodeBlock(
                language=language,
                title=title,
                content=code_content,
                start_pos=match.start(),
            )
        )

        # Replace with placeholder that won't interfere with header parsing
        return f"\n[CODE_BLOCK_{len(code_blocks) - 1}]\n"

    content_without_code = CODE_BLOCK_PATTERN.sub(replace_block, content)
    return content_without_code, code_blocks


def parse_sections(content: str, code_blocks: list[CodeBlock]) -> list[Section]:
    """
    Parse markdown content into sections based on headers.

    Args:
        content: Markdown content (with code block placeholders)
        code_blocks: List of extracted code blocks

    Returns:
        List of sections with their content
    """
    sections: list[Section] = []

    # Track header positions
    header_positions: list[tuple[int, int, str, int]] = []  # (start, end, title, level)

    for match in HEADER_PATTERN.finditer(content):
        level = len(match.group(1))
        title = match.group(2).strip()
        header_positions.append((match.start(), match.end(), title, level))

    # If no headers, treat entire content as one section
    if not header_positions:
        section = Section(
            level=0,
            title="",
            anchor="",
            content=content.strip(),
            line_start=0,
        )
        _attach_code_blocks(section, content, 0, len(content), code_blocks)
        return [section]

    # Extract content between headers
    for i, (start, end, title, level) in enumerate(header_positions):
        # Content ends at next header or end of document
        content_end = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(content)

        section_content = content[end:content_end].strip()

        section = Section(
            level=level,
            title=title,
            anchor=slugify(title),
            content=section_content,
            line_start=content[:start].count("\n"),
        )

        # Attach code blocks that fall within this section's range
        _attach_code_blocks(section, content, start, content_end, code_blocks)
        sections.append(section)

    return sections


def _attach_code_blocks(
    section: Section,
    full_content: str,
    section_start: int,
    section_end: int,
    code_blocks: list[CodeBlock],
) -> None:
    """Attach code blocks that fall within a section's range."""
    for i, block in enumerate(code_blocks):
        placeholder = f"[CODE_BLOCK_{i}]"
        if placeholder in section.content:
            section.code_blocks.append(block)


def restore_code_blocks(content: str, code_blocks: list[CodeBlock]) -> str:
    """
    Restore code blocks from placeholders.

    Args:
        content: Content with placeholders
        code_blocks: Original code blocks

    Returns:
        Content with code blocks restored
    """
    for i, block in enumerate(code_blocks):
        placeholder = f"[CODE_BLOCK_{i}]"
        fence_meta = block.language
        if block.title:
            fence_meta += f' title="{block.title}"'
        restored = f"```{fence_meta}\n{block.content}\n```"
        content = content.replace(placeholder, restored)
    return content


def build_breadcrumb(sections: list[Section], current_idx: int) -> list[str]:
    """
    Build breadcrumb hierarchy for a section.

    Args:
        sections: All sections in order
        current_idx: Index of current section

    Returns:
        List of titles forming the breadcrumb path
    """
    if current_idx < 0 or current_idx >= len(sections):
        return []

    current = sections[current_idx]
    breadcrumb = [current.title] if current.title else []

    # Walk backwards to find parent headers
    for i in range(current_idx - 1, -1, -1):
        ancestor = sections[i]
        # A section is a parent if it has a lower level number (higher in hierarchy)
        if ancestor.level < current.level and ancestor.level > 0:
            breadcrumb.insert(0, ancestor.title)
            current = ancestor

    return breadcrumb


def _is_garbage_chunk_content(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True

    # MDX/JSX stripping can leave behind fragments like `} />`.
    if GARBAGE_CHUNK_PATTERN.fullmatch(stripped):
        return True

    # Drop tiny non-text fragments that survived stripping.
    return bool(len(stripped) < 20 and not ALNUM_PATTERN.search(stripped))


def chunk_markdown(
    content: str,
    file_path: str,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
) -> list[DocChunk]:
    """
    Chunk a markdown document into searchable units.

    Strategy:
    1. Extract frontmatter
    2. Extract code blocks (replaced with placeholders)
    3. Split on headers to create sections
    4. Build breadcrumb hierarchy
    5. Create DOC chunks for prose, CODE chunks for code blocks
    6. Split large sections to respect token limit

    Args:
        content: Raw markdown content
        file_path: Relative file path within repo
        library_id: Library identifier (e.g., "/langfuse/langfuse")
        commit_sha: Git commit SHA
        repo: Optional repo path for URL generation (extracted from library_id if not provided)
        branch: Git branch name
        source_path_prefix: Optional repo-relative prefix prepended before file_path for source URLs

    Returns:
        List of DocChunk objects
    """
    settings = get_settings()
    max_tokens = settings.chunk_max_tokens

    # Extract repo from library_id if not provided
    if repo is None:
        repo = library_id.lstrip("/")

    chunks: list[DocChunk] = []

    # Parse frontmatter
    frontmatter, content_without_fm = parse_frontmatter(content)
    frontmatter_source_url = (
        frontmatter.get("source_url") or frontmatter.get("source") or ""
    ).strip()
    frontmatter_offset = 0
    if content_without_fm:
        content_start = content.find(content_without_fm)
        if content_start >= 0:
            frontmatter_offset = content[:content_start].count("\n")

    # Extract code blocks
    content_with_placeholders, code_blocks = extract_code_blocks(content_without_fm)

    # Parse sections
    sections = parse_sections(content_with_placeholders, code_blocks)

    # Track unique anchors (GitHub appends -1, -2, etc. for duplicates)
    anchor_counts: dict[str, int] = {}

    for i, section in enumerate(sections):
        breadcrumb = build_breadcrumb(sections, i)

        # Fallback breadcrumb for documents/sections without headers.
        if not breadcrumb:
            if frontmatter.get("title"):
                breadcrumb = [frontmatter["title"]]
            else:
                breadcrumb = [Path(file_path).stem]

        # Handle duplicate anchors
        base_anchor = section.anchor
        if base_anchor in anchor_counts:
            anchor_counts[base_anchor] += 1
            section.anchor = f"{base_anchor}-{anchor_counts[base_anchor]}"
        else:
            anchor_counts[base_anchor] = 0

        # Remove code block placeholders from prose content for DOC chunk
        prose_content = section.content
        for j in range(len(code_blocks)):
            prose_content = prose_content.replace(f"[CODE_BLOCK_{j}]", "").strip()

        # Skip empty sections (header-only)
        if not prose_content and not section.code_blocks:
            continue

        # Create DOC chunks for prose content
        section_parent_id: str | None = None
        if prose_content:
            for chunk_idx, (chunk_content, line_start, line_end) in enumerate(
                split_large_section_with_line_spans(
                    prose_content,
                    max_tokens,
                    start_line=frontmatter_offset + section.line_start + 1,
                )
            ):
                if _is_garbage_chunk_content(chunk_content):
                    continue

                chunk_anchor = section.anchor
                if chunk_idx > 0:
                    chunk_anchor = f"{section.anchor}-part-{chunk_idx + 1}"

                doc_chunk = DocChunk(
                    library_id=library_id,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    chunk_type=ChunkType.DOC,
                    content=chunk_content,
                    breadcrumb=breadcrumb,
                    section_anchor=chunk_anchor,
                    source_url=_build_chunk_source_url(
                        repo,
                        commit_sha,
                        file_path,
                        section.anchor,
                        frontmatter_source_url,
                        source_path_prefix,
                    ),
                    metadata={
                        **(
                            {"title": frontmatter.get("title", "")}
                            if frontmatter.get("title")
                            else {}
                        ),
                    },
                    heading_level=section.level if section.level > 0 else None,
                    token_count=estimate_tokens(chunk_content),
                    canonical_anchor=section.anchor,
                    line_start=line_start,
                    line_end=line_end,
                )
                chunks.append(doc_chunk)
                if section_parent_id is None:
                    section_parent_id = doc_chunk.id

        # Create CODE chunks for code blocks
        for code_block in section.code_blocks:
            code_metadata = {"language": code_block.language}
            if code_block.title:
                code_metadata["title"] = code_block.title

            code_line_start = (
                frontmatter_offset + content_without_fm[: code_block.start_pos].count("\n") + 1
            )
            for chunk_idx, (chunk_content, line_start, line_end) in enumerate(
                split_large_section_with_line_spans(
                    code_block.content,
                    max_tokens,
                    start_line=code_line_start,
                )
            ):
                if _is_garbage_chunk_content(chunk_content):
                    continue

                code_anchor = section.anchor
                if chunk_idx > 0:
                    code_anchor = f"{section.anchor}-code-part-{chunk_idx + 1}"

                code_chunk = DocChunk(
                    library_id=library_id,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    chunk_type=ChunkType.CODE,
                    content=chunk_content,
                    breadcrumb=breadcrumb,
                    section_anchor=code_anchor,
                    source_url=_build_chunk_source_url(
                        repo,
                        commit_sha,
                        file_path,
                        section.anchor,
                        frontmatter_source_url,
                        source_path_prefix,
                    ),
                    parent_chunk_id=section_parent_id,
                    metadata=code_metadata,
                    heading_level=section.level if section.level > 0 else None,
                    token_count=estimate_tokens(chunk_content),
                    canonical_anchor=section.anchor,
                    line_start=line_start,
                    line_end=line_end,
                )
                chunks.append(code_chunk)

    # Handle document with no sections (just content)
    if not chunks and content_without_fm.strip():
        # Create single chunk for entire document
        for chunk_idx, chunk_content in enumerate(
            split_large_section(content_without_fm, max_tokens)
        ):
            if _is_garbage_chunk_content(chunk_content):
                continue

            anchor = f"content-{chunk_idx}" if chunk_idx > 0 else ""
            chunks.append(
                DocChunk(
                    library_id=library_id,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    chunk_type=ChunkType.DOC,
                    content=chunk_content,
                    breadcrumb=[],
                    section_anchor=anchor,
                    source_url=_build_chunk_source_url(
                        repo,
                        commit_sha,
                        file_path,
                        "",
                        frontmatter_source_url,
                        source_path_prefix,
                    ),
                    metadata={
                        **(
                            {"title": frontmatter.get("title", "")}
                            if frontmatter.get("title")
                            else {}
                        ),
                    },
                    heading_level=None,  # No header for this content
                    token_count=estimate_tokens(chunk_content),
                    canonical_anchor="",
                )
            )

    logger.debug(f"Chunked {file_path}: {len(chunks)} chunks")
    return chunks


def _build_chunk_source_url(
    repo: str,
    commit_sha: str,
    file_path: str,
    section_anchor: str,
    frontmatter_source_url: str,
    source_path_prefix: str,
) -> str:
    if frontmatter_source_url:
        return append_anchor(frontmatter_source_url, section_anchor)
    return build_source_url(
        repo,
        commit_sha,
        file_path,
        section_anchor,
        path_prefix=source_path_prefix,
    )
