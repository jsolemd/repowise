"""reStructuredText chunking via lightweight normalization to markdown."""

from __future__ import annotations

import re

from repowise.docs.chunking.extractors.markdown import chunk_markdown
from repowise.docs.chunking.models import DocChunk

HEADING_UNDERLINE_PATTERN = re.compile(r"^([=\-~^\"`:#*+])\1{2,}\s*$")
CODE_BLOCK_PATTERN = re.compile(
    r"^(?P<indent>\s*)\.\.\s+code-block::\s*(?P<lang>[A-Za-z0-9_+-]*)\s*$"
)
LITERAL_BLOCK_PATTERN = re.compile(r"^(?P<indent>\s*).+::\s*$")

HEADING_LEVELS: dict[str, int] = {
    "=": 1,
    "-": 2,
    "~": 3,
    "^": 4,
    '"': 5,
    "`": 6,
    ":": 6,
    "#": 6,
    "*": 6,
    "+": 6,
}


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _strip_code_indent(line: str, block_indent: int) -> str:
    if not line:
        return ""

    minimum = block_indent + 1
    width = _indent_width(line)
    cut = min(width, minimum)
    return line[cut:]


def _consume_indented_block(
    lines: list[str], start_idx: int, base_indent: int
) -> tuple[list[str], int]:
    block: list[str] = []
    idx = start_idx

    while idx < len(lines):
        line = lines[idx]

        if not line.strip():
            block.append("")
            idx += 1
            continue

        if _indent_width(line) <= base_indent:
            break

        block.append(_strip_code_indent(line, base_indent))
        idx += 1

    while block and not block[-1].strip():
        block.pop()

    return block, idx


def _normalize_rst(content: str) -> str:
    lines = content.splitlines()
    out: list[str] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]

        # Section headers (title + underline marker)
        if idx + 1 < len(lines):
            title = line.strip()
            underline = lines[idx + 1].strip()
            marker_match = HEADING_UNDERLINE_PATTERN.match(underline)
            if marker_match and title and len(underline) >= len(title):
                level = HEADING_LEVELS.get(marker_match.group(1), 6)
                out.append(f"{'#' * level} {title}")
                idx += 2
                continue

        # Explicit code-block directive
        code_match = CODE_BLOCK_PATTERN.match(line)
        if code_match:
            language = code_match.group("lang") or ""
            base_indent = len(code_match.group("indent"))
            idx += 1

            # Skip directive options and leading blank lines.
            while idx < len(lines):
                option_line = lines[idx]
                if not option_line.strip():
                    idx += 1
                    continue
                if option_line.lstrip().startswith(":"):
                    idx += 1
                    continue
                break

            block, idx = _consume_indented_block(lines, idx, base_indent)
            out.append(f"```{language}" if language else "```")
            out.extend(block)
            out.append("```")
            continue

        # Literal block introduced by trailing ::
        literal_match = LITERAL_BLOCK_PATTERN.match(line)
        if literal_match and not line.lstrip().startswith(".."):
            base_indent = len(literal_match.group("indent"))
            stripped = line.rstrip()
            prefix = stripped[:-2].rstrip()
            if prefix:
                out.append(f"{prefix}:")

            idx += 1
            while idx < len(lines) and not lines[idx].strip():
                idx += 1

            block, idx = _consume_indented_block(lines, idx, base_indent)
            if block:
                out.append("```")
                out.extend(block)
                out.append("```")
            continue

        out.append(line)
        idx += 1

    return "\n".join(out)


def chunk_rst(
    content: str,
    file_path: str,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
) -> list[DocChunk]:
    """Chunk an RST document using markdown-normalized structure."""
    normalized = _normalize_rst(content)
    return chunk_markdown(
        normalized,
        file_path,
        library_id,
        commit_sha,
        repo,
        branch,
        source_path_prefix,
    )
