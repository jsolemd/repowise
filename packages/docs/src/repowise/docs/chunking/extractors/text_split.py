"""Shared text splitting helpers for doc-search chunkers."""

from __future__ import annotations

import re
from collections.abc import Iterator


def estimate_tokens(text: str) -> int:
    """Estimate token count for text using a coarse 4-chars-per-token heuristic."""
    return len(text) // 4


def split_large_section(content: str, max_tokens: int) -> Iterator[str]:
    """Split oversized text into chunks while preserving structure when possible."""
    if estimate_tokens(content) <= max_tokens:
        yield content
        return

    max_chars = max_tokens * 4
    if max_chars <= 0:
        yield content
        return

    def split_hard(text: str) -> Iterator[str]:
        for i in range(0, len(text), max_chars):
            yield text[i : i + max_chars]

    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")

    def split_by_lines(text: str) -> Iterator[str]:
        lines = text.splitlines(keepends=True)
        current_lines: list[str] = []
        current_chars = 0

        for line in lines:
            if len(line) > max_chars:
                if current_lines:
                    yield "".join(current_lines).rstrip("\n")
                    current_lines = []
                    current_chars = 0

                for part in split_hard(line):
                    yield part.rstrip("\n")
                continue

            # Keep markdown table rows together: don't break between
            # consecutive |..| lines even if we exceed the soft limit.
            in_table = _is_table_line(line) and current_lines and _is_table_line(current_lines[-1])

            if current_chars + len(line) > max_chars and current_lines and not in_table:
                yield "".join(current_lines).rstrip("\n")
                current_lines = [line]
                current_chars = len(line)
            else:
                current_lines.append(line)
                current_chars += len(line)

        if current_lines:
            yield "".join(current_lines).rstrip("\n")

    def split_by_sentences(text: str) -> Iterator[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current_sentences: list[str] = []
        current_chars = 0
        separator = " "

        for sentence in sentences:
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current_sentences:
                    yield separator.join(current_sentences)
                    current_sentences = []
                    current_chars = 0

                yield from split_hard(sentence)
                continue

            prospective = (
                len(sentence)
                if not current_sentences
                else current_chars + len(separator) + len(sentence)
            )

            if prospective > max_chars and current_sentences:
                yield separator.join(current_sentences)
                current_sentences = [sentence]
                current_chars = len(sentence)
            else:
                current_sentences.append(sentence)
                current_chars = prospective

        if current_sentences:
            yield separator.join(current_sentences)

    paragraphs = re.split(r"\n\n+", content)
    current_chunk: list[str] = []
    current_chars = 0
    paragraph_separator = "\n\n"

    for para in paragraphs:
        if len(para) > max_chars:
            if current_chunk:
                yield paragraph_separator.join(current_chunk)
                current_chunk = []
                current_chars = 0

            if "\n" in para:
                yield from split_by_lines(para)
            else:
                yield from split_by_sentences(para)
            continue

        prospective = (
            len(para) if not current_chunk else current_chars + len(paragraph_separator) + len(para)
        )

        if prospective > max_chars and current_chunk:
            yield paragraph_separator.join(current_chunk)
            current_chunk = [para]
            current_chars = len(para)
        else:
            current_chunk.append(para)
            current_chars = prospective

    if current_chunk:
        yield paragraph_separator.join(current_chunk)


def split_large_section_with_line_spans(
    content: str,
    max_tokens: int,
    *,
    start_line: int = 1,
) -> Iterator[tuple[str, int, int]]:
    """Split oversized text and preserve approximate 1-based line spans.

    The splitter is deterministic and yields exact substrings of the original text
    where possible. We use a forward-only substring search so later handlers can
    navigate or expand chunks without rescanning the full file heuristically.
    """
    cursor = 0
    current_line = max(1, start_line)

    for part in split_large_section(content, max_tokens):
        if not part:
            continue

        match_index = content.find(part, cursor)
        if match_index < 0:
            match_index = cursor

        prefix = content[cursor:match_index]
        current_line += prefix.count("\n")
        line_start = current_line
        line_end = line_start + part.count("\n")

        yield part, line_start, line_end

        cursor = match_index + len(part)
        current_line = line_end
