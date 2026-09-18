"""Block-comment delimiters are syntax; Markdown and paths are content."""

from __future__ import annotations

import pytest

from repowise.core.ingestion.extractors.helpers import clean_jsdoc


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("/** One-line documentation. */", "One-line documentation."),
        ("/* Ordinary block documentation. */", "Ordinary block documentation."),
        ("/**\n * First line.\n * Second line.\n */", "First line.\nSecond line."),
        ("/** First line.\n * Closing beside prose. */", "First line.\nClosing beside prose."),
        ("/** /api/widgets/ */", "/api/widgets/"),
        ("/**\n * /api/widgets/\n * https://example.test/api/*\n */",
         "/api/widgets/\nhttps://example.test/api/*"),
        ("/** **Bold** and *emphasis*.\n * - /api/**\n *   - nested\n *\n * [link](/api/read)\n */",
         "**Bold** and *emphasis*.\n- /api/**\n  - nested\n\n[link](/api/read)"),
        ("/**\n **Bold**\n *emphasis*\n /api/**\n */", "**Bold**\n*emphasis*\n/api/**"),
        ("/** * Markdown bullet on the opening line. */", "* Markdown bullet on the opening line."),
        ("/**\n * * Markdown bullet after decoration.\n */", "* Markdown bullet after decoration."),
        ("/**\n * ```ts\n *   fetch('/api/read');\n * ```\n */", "```ts\n  fetch('/api/read');\n```"),
        ("/**\n * A hard break.  \n * Next line.\n */", "A hard break.  \nNext line."),
        ("/**/", ""),
        ("/** */", ""),
        ("/**\n *\n */", ""),
        ("/* */", ""),
        ("/api/widgets/", "/api/widgets/"),
        ("**Literal Markdown**", "**Literal Markdown**"),
    ],
)
def test_clean_jsdoc_removes_delimiters_without_consuming_content(comment: str, expected: str) -> None:
    assert clean_jsdoc(comment) == expected
