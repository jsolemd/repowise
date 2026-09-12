"""Regression coverage for the 1Password developer docs snapshot scraper."""

from __future__ import annotations

from repowise.docs.scrapers.onepassword_developer import _extract_markdown


def test_extract_markdown_strips_nul_bytes() -> None:
    html = """
    <html>
      <body>
        <main>
          <h1>Heading</h1>
          <p>Before \x00 after.</p>
          <p>Keep line breaks.</p>
        </main>
      </body>
    </html>
    """

    markdown = _extract_markdown(html)

    assert "\x00" not in markdown
    assert "Heading" in markdown
    assert "Before  after." in markdown
