"""Lightweight HTML -> markdown conversion for scraped documentation pages."""

from __future__ import annotations

import re
from collections.abc import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag


def _clean_text(text: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _inline_text(node: Tag) -> str:
    return " ".join(node.stripped_strings).strip()


def _render_list(items: Iterable[Tag], *, ordered: bool) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        text = _inline_text(item)
        if not text:
            continue
        prefix = f"{index}. " if ordered else "- "
        lines.append(prefix + text)
    return lines


def _render_table(table: Tag) -> list[str]:
    rows = []
    for tr in table.find_all("tr", recursive=False) or table.find_all("tr"):
        cells = tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"])
        row = [_inline_text(cell) for cell in cells]
        if any(row):
            rows.append(row)
    if not rows:
        return []
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    separator = ["---"] * width
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _render_node(node: Tag) -> list[str]:
    name = node.name.lower()
    if name in {"script", "style", "noscript", "svg"}:
        return []
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        text = _inline_text(node)
        return [("#" * level) + f" {text}", ""] if text else []
    if name == "p":
        text = _inline_text(node)
        return [text, ""] if text else []
    if name in {"ul", "ol"}:
        items = [child for child in node.find_all("li", recursive=False)]
        return [*_render_list(items, ordered=name == "ol"), ""]
    if name == "pre":
        text = node.get_text("\n", strip=False).strip("\n")
        return [f"```text\n{text}\n```", ""] if text else []
    if name == "blockquote":
        text = _inline_text(node)
        return [f"> {text}", ""] if text else []
    if name == "table":
        lines = _render_table(node)
        return [*lines, ""] if lines else []
    if name in {"div", "section", "article", "main", "header", "footer"}:
        lines: list[str] = []
        for child in node.children:
            if isinstance(child, NavigableString):
                continue
            if isinstance(child, Tag):
                lines.extend(_render_node(child))
        return lines

    text = _inline_text(node)
    return [text, ""] if text else []


def html_to_markdown(
    html: str,
    *,
    selectors: list[str] | None = None,
    prune_selectors: list[str] | None = None,
) -> str:
    """Convert a focused HTML subtree into rough markdown for indexing."""
    soup = BeautifulSoup(html, "html.parser")

    for selector in prune_selectors or []:
        for node in soup.select(selector):
            node.decompose()

    root: Tag | None = None
    for selector in selectors or []:
        root = soup.select_one(selector)
        if root is not None:
            break
    if root is None:
        root = soup.body or soup

    lines = _render_node(root)
    return _clean_text("\n".join(lines))
