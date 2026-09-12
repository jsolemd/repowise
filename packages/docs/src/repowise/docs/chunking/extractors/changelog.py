"""Changelog document chunking with version-based splitting."""

import logging
import re
from dataclasses import dataclass

from repowise.docs.chunking.extractors.text_split import split_large_section_with_line_spans
from repowise.docs.chunking.models import (
    ChunkType,
    DocChunk,
    slugify,
)
from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)

# Patterns for version headers in changelogs
# Examples:
#   ## [1.2.3] - 2024-01-15
#   ## v1.2.3
#   ## 1.2.3 (2024-01-15)
#   # Version 1.2.3
#   ## [Unreleased]
VERSION_HEADER_PATTERNS = [
    # ## [1.2.3] - 2024-01-15 (Keep a Changelog format)
    re.compile(
        r"^(#{1,3})\s+\[([^\]]+)\](?:\s*[-–—]\s*(\d{4}-\d{2}-\d{2}))?",
        re.MULTILINE,
    ),
    # ## v1.2.3 or ## 1.2.3
    re.compile(
        r"^(#{1,3})\s+(v?\d+\.\d+(?:\.\d+)?(?:[-.]\w+)?)\s*(?:\((\d{4}-\d{2}-\d{2})\))?",
        re.MULTILINE,
    ),
    # # Version 1.2.3
    re.compile(
        r"^(#{1,3})\s+[Vv]ersion\s+(\d+\.\d+(?:\.\d+)?(?:[-.]\w+)?)",
        re.MULTILINE,
    ),
]

# Pattern for release date extraction if not in header
DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


@dataclass
class ChangelogEntry:
    """A single version entry in a changelog."""

    level: int  # Header level (1-3)
    version: str  # Version string (e.g., "1.2.3", "Unreleased")
    release_date: str | None  # ISO date if found
    content: str  # Full content including sub-headers
    line_start: int  # Line number where entry starts
    anchor: str  # GitHub-style anchor


def parse_changelog_entries(content: str) -> list[ChangelogEntry]:
    """
    Parse changelog content into version entries.

    Detects version headers and extracts content between them.

    Args:
        content: Raw changelog content

    Returns:
        List of ChangelogEntry objects
    """
    entries: list[ChangelogEntry] = []

    # Collect all version header matches with their positions
    all_matches: list[tuple[int, int, int, str, str | None]] = []

    for pattern in VERSION_HEADER_PATTERNS:
        for match in pattern.finditer(content):
            level = len(match.group(1))
            version = match.group(2).strip()
            date = match.group(3) if match.lastindex >= 3 and match.group(3) else None

            all_matches.append((match.start(), match.end(), level, version, date))

    # Sort by position
    all_matches.sort(key=lambda x: x[0])

    # Deduplicate overlapping matches (prefer longest match)
    deduped: list[tuple[int, int, int, str, str | None]] = []
    for m in all_matches:
        if not deduped or m[0] >= deduped[-1][1]:
            deduped.append(m)

    # Extract entries
    for i, (start, end, level, version, date) in enumerate(deduped):
        # Content goes until next entry or end of file
        content_end = deduped[i + 1][0] if i + 1 < len(deduped) else len(content)

        entry_content = content[end:content_end].strip()

        # Try to extract date from content if not in header
        if not date:
            date_match = DATE_PATTERN.search(entry_content[:200])  # Check first 200 chars
            if date_match:
                date = date_match.group(1)

        entries.append(
            ChangelogEntry(
                level=level,
                version=version,
                release_date=date,
                content=entry_content,
                line_start=content[:start].count("\n"),
                anchor=slugify(version),
            )
        )

    return entries


def chunk_changelog(
    content: str,
    file_path: str,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
) -> list[DocChunk]:
    """
    Chunk a changelog file into version-based chunks.

    Each version entry becomes one or more chunks (if very large).
    Chunks are marked with chunk_type=CHANGELOG and include version metadata.

    Args:
        content: Raw changelog content
        file_path: Relative file path within repo
        library_id: Library identifier (e.g., "/langfuse/langfuse")
        commit_sha: Git commit SHA
        repo: Optional repo path for URL generation
        branch: Git branch name

    Returns:
        List of DocChunk objects
    """
    from repowise.docs.chunking.extractors.markdown import (
        _build_chunk_source_url,
        parse_frontmatter,
    )

    settings = get_settings()
    max_tokens = settings.chunk_max_tokens

    # Extract repo from library_id if not provided
    if repo is None:
        repo = library_id.lstrip("/")

    chunks: list[DocChunk] = []

    # Strip frontmatter if present
    frontmatter, content_without_fm = parse_frontmatter(content)
    frontmatter_source_url = (
        frontmatter.get("source_url") or frontmatter.get("source") or ""
    ).strip()

    # Parse version entries
    entries = parse_changelog_entries(content_without_fm)

    if not entries:
        # No version headers found - fall back to standard markdown chunking
        from repowise.docs.chunking.extractors.markdown import chunk_markdown

        logger.debug(f"No version headers found in {file_path}, using markdown chunker")
        md_chunks = chunk_markdown(
            content,
            file_path,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
        )
        # Mark them as changelog type
        for chunk in md_chunks:
            chunk.chunk_type = ChunkType.CHANGELOG
        return md_chunks

    # Track unique anchors
    anchor_counts: dict[str, int] = {}

    for entry in entries:
        # Handle duplicate version anchors (shouldn't happen often)
        base_anchor = entry.anchor
        if base_anchor in anchor_counts:
            anchor_counts[base_anchor] += 1
            entry.anchor = f"{base_anchor}-{anchor_counts[base_anchor]}"
        else:
            anchor_counts[base_anchor] = 0

        # Build metadata
        metadata: dict[str, str] = {
            "version": entry.version,
        }
        if entry.release_date:
            metadata["release_date"] = entry.release_date

        # Build breadcrumb
        breadcrumb = ["Changelog", entry.version]

        # Create full entry content including version header
        full_content = f"## {entry.version}"
        if entry.release_date:
            full_content += f" ({entry.release_date})"
        full_content += f"\n\n{entry.content}"

        # Split if too large
        for chunk_idx, (chunk_content, line_start, line_end) in enumerate(
            split_large_section_with_line_spans(
                full_content,
                max_tokens,
                start_line=entry.line_start + 1,
            )
        ):
            if not chunk_content.strip():
                continue

            chunk_anchor = entry.anchor
            if chunk_idx > 0:
                chunk_anchor = f"{entry.anchor}-part-{chunk_idx + 1}"

            chunks.append(
                DocChunk(
                    library_id=library_id,
                    file_path=file_path,
                    commit_sha=commit_sha,
                    chunk_type=ChunkType.CHANGELOG,
                    content=chunk_content,
                    breadcrumb=breadcrumb,
                    section_anchor=chunk_anchor,
                    source_url=_build_chunk_source_url(
                        repo,
                        commit_sha,
                        file_path,
                        entry.anchor,
                        frontmatter_source_url,
                        source_path_prefix,
                    ),
                    metadata=metadata,
                    canonical_anchor=entry.anchor,
                    line_start=line_start,
                    line_end=line_end,
                )
            )

    logger.debug(f"Chunked changelog {file_path}: {len(chunks)} version entries")
    return chunks
