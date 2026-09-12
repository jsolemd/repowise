"""Chunk-expansion doc-search tool handlers."""

from __future__ import annotations

import logging
from typing import Any

from repowise.docs.db import get_library
from repowise.docs.indexer import get_chunk_by_id
from repowise.docs.library.source_runtime import get_library_file_content

from .common import ToolError, clamp_int, get_arg

logger = logging.getLogger(__name__)


async def handle_expand_chunk(
    arguments: dict[str, Any],
    *,
    get_chunk_by_id_fn=get_chunk_by_id,
    get_library_fn=get_library,
    get_library_file_content_fn=get_library_file_content,
) -> dict[str, Any]:
    """Get expanded context around a search result chunk."""
    chunk_id = str(get_arg(arguments, "chunk_id", aliases=("chunkId",), default="")).strip()
    lines_before = clamp_int(
        get_arg(arguments, "lines_before", aliases=("linesBefore",), default=30), 0, 200, 30
    )
    lines_after = clamp_int(
        get_arg(arguments, "lines_after", aliases=("linesAfter",), default=30), 0, 200, 30
    )

    if not chunk_id:
        raise ToolError("chunk_id is required")

    chunk_record = await get_chunk_by_id_fn(chunk_id)
    if not chunk_record:
        raise ToolError(f"Chunk not found: {chunk_id}")

    payload = chunk_record.payload or {}
    library_id = payload.get("library_id")
    file_path = payload.get("file_path")
    chunk_content = payload.get("content", "")
    stored_line_start = payload.get("line_start")
    stored_line_end = payload.get("line_end")

    if not library_id or not file_path:
        raise ToolError("Chunk is missing library_id or file_path")

    library = await get_library_fn(library_id)
    if not library:
        raise ToolError(f"Library not found: {library_id}")

    file_content = await get_library_file_content_fn(library, file_path)
    lines = file_content.splitlines()
    total_lines = len(lines)

    if isinstance(stored_line_start, int) and isinstance(stored_line_end, int):
        chunk_start = max(1, stored_line_start)
        chunk_end = min(total_lines, max(chunk_start, stored_line_end))
        start_line = max(1, chunk_start - lines_before)
        end_line = min(total_lines, chunk_end + lines_after)
        expanded_lines = lines[start_line - 1 : end_line]
    else:
        chunk_lines = chunk_content.splitlines()
        if not chunk_lines:
            start_line = 1
            end_line = min(total_lines, lines_before + lines_after + 1)
            expanded_lines = lines[start_line - 1 : end_line]
        else:
            first_line = chunk_lines[0].strip()
            match_line = None
            for i, line in enumerate(lines):
                if first_line and first_line in line:
                    match = True
                    for j, chunk_line in enumerate(chunk_lines[:5]):
                        if i + j >= len(lines):
                            match = False
                            break
                        if chunk_line.strip() and chunk_line.strip() not in lines[i + j]:
                            match = False
                            break
                    if match:
                        match_line = i
                        break

            if match_line is None:
                return {
                    "content": chunk_content,
                    "file_path": file_path,
                    "library_id": library_id,
                    "anchor": payload.get("canonical_anchor", payload.get("section_anchor", "")),
                    "chunk_anchor": payload.get("section_anchor", ""),
                    "warning": "Could not locate chunk in file. Returning original chunk content.",
                }

            chunk_end = match_line + len(chunk_lines)
            start_line = max(1, match_line + 1 - lines_before)
            end_line = min(total_lines, chunk_end + lines_after)
            expanded_lines = lines[start_line - 1 : end_line]

    numbered_lines = []
    for i, line in enumerate(expanded_lines):
        line_num = start_line + i
        numbered_lines.append(f"{line_num:4d} | {line}")

    expanded_content = "\n".join(numbered_lines)
    return {
        "content": expanded_content,
        "file_path": file_path,
        "library_id": library_id,
        "line_start": start_line,
        "line_end": end_line,
        "total_lines": total_lines,
        "breadcrumb": payload.get("breadcrumb", []),
        "anchor": payload.get("canonical_anchor", payload.get("section_anchor", "")),
        "chunk_anchor": payload.get("section_anchor", ""),
        "source_url": payload.get("source_url", ""),
    }
