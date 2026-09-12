#!/usr/bin/env python3
"""
1Password Developer docs scraper for CodeAtlas snapshot publishing.

Scrapes the official 1Password developer docs from:
  https://developer.1password.com/docs/*

This uses the public sitemap, fetches every docs page, converts the main
documentation content to markdown, and publishes the latest current-state
snapshot into CodeAtlas.

Library ID: /codeatlas/1password-developer

Usage:
    python onepassword_developer.py --fetch
    python onepassword_developer.py --publish
    python onepassword_developer.py --all
    python onepassword_developer.py --stats
"""

from __future__ import annotations

import asyncio
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

try:
    from .common import (
        compute_source_digest,
        publish_output_dir,
        update_freshness_entry,
        write_snapshot_metadata,
    )
    from .html_markdown import html_to_markdown
except ImportError:
    from common import (
        compute_source_digest,
        publish_output_dir,
        update_freshness_entry,
        write_snapshot_metadata,
    )
    from html_markdown import html_to_markdown

OUTPUT_DIR = Path("/tmp/1password-developer-docs")
LIBRARY_ID = "/codeatlas/1password-developer"
SITEMAP_URL = "https://developer.1password.com/sitemap.xml"
DOCS_PREFIX = "https://developer.1password.com/docs"
CONCURRENCY = 8

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
    )
}

PRUNE_SELECTORS = [
    "nav",
    "aside",
    "footer",
    "header",
    "button",
    "[class*=breadcrumbs]",
    "[class*=paginationNav]",
    "[class*=pagination-nav]",
    "[class*=tableOfContents]",
    "[class*=toc]",
    "[class*=DocSearch]",
]

FRESHNESS_SOURCES = [
    SITEMAP_URL,
    "https://developer.1password.com/docs/cli/reference/commands/run",
    "https://developer.1password.com/docs/environments/read-environment-variables/",
    "https://developer.1password.com/docs/ssh/agent/",
]

_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text(text: str) -> str:
    return _CONTROL_CHAR_PATTERN.sub("", text)


def _page_title(html: str) -> str:
    soup = BeautifulSoup(_sanitize_text(html), "html.parser")
    if soup.title is not None:
        title = soup.title.get_text(" ", strip=True)
        title = title.replace("| 1Password Developer", "").strip()
        if title:
            return title
    heading = soup.select_one("main h1")
    if heading is not None:
        title = heading.get_text(" ", strip=True)
        if title:
            return title
    return "1Password Developer Docs"


def _url_to_output_path(url: str) -> Path:
    path = urlparse(url).path.strip("/")
    if not path:
        return OUTPUT_DIR / "index.md"
    parts = path.split("/")
    if len(parts) == 1:
        return OUTPUT_DIR / f"{parts[0]}.md"
    return OUTPUT_DIR.joinpath(*parts[:-1], f"{parts[-1]}.md")


def _frontmatter(*, title: str, source: str) -> str:
    return (
        "---\n"
        f'title: "{title.replace(chr(34), chr(39))}"\n'
        f'source: "{source}"\n'
        "scraped: true\n"
        "---\n\n"
    )


def _build_library_readme(urls: list[str], *, output_dir: Path = OUTPUT_DIR) -> None:
    lines = [
        "---",
        'title: "1Password Developer Docs"',
        f'source: "{DOCS_PREFIX}/"',
        "scraped: true",
        "---",
        "",
        "# 1Password Developer Docs",
        "",
        "Snapshot of the official 1Password developer documentation indexed by CodeAtlas.",
        "",
        "## Included sections",
        "",
        "- CLI",
        "- Environments",
        "- SSH & Git",
        "- SDKs",
        "- Service Accounts",
        "- Connect Server",
        "- Shell Plugins",
        "- Integrations and CI/CD guides",
        "",
        f"Total docs pages captured: **{len(urls)}**",
        "",
        "## Canonical sources",
        "",
    ]
    for source in FRESHNESS_SOURCES[1:]:
        lines.append(f"- {source}")
    lines.append("")
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def _extract_markdown(html: str) -> str:
    markdown = html_to_markdown(
        _sanitize_text(html),
        selectors=["main"],
        prune_selectors=PRUNE_SELECTORS,
    )
    return _sanitize_text(markdown)


async def _load_docs_urls(client: httpx.AsyncClient) -> tuple[list[str], str]:
    response = await client.get(SITEMAP_URL)
    response.raise_for_status()
    sitemap_xml = _sanitize_text(response.text)

    root = ET.fromstring(sitemap_xml)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = []
    for loc in root.findall("sm:url/sm:loc", ns):
        value = (loc.text or "").strip()
        if value.startswith(DOCS_PREFIX):
            urls.append(value)
    normalized = sorted(set(urls))
    return normalized, sitemap_xml


async def _fetch_page(
    client: httpx.AsyncClient,
    url: str,
    *,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    async with semaphore:
        response = await client.get(url)
        response.raise_for_status()
        return url, _sanitize_text(response.text)


async def probe_source_ref() -> str:
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=HEADERS,
    ) as client:
        freshness_parts: list[str] = []
        for source in FRESHNESS_SOURCES:
            response = await client.get(source)
            response.raise_for_status()
            freshness_parts.append(_sanitize_text(response.text))
    return compute_source_digest(freshness_parts)


async def fetch_docs(
    output_dir: Path | None = None,
    *,
    persist_freshness: bool = True,
) -> str:
    output_dir = output_dir or OUTPUT_DIR
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timeout = httpx.Timeout(30.0)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers=HEADERS,
    ) as client:
        urls, sitemap_xml = await _load_docs_urls(client)
        tasks = [_fetch_page(client, url, semaphore=semaphore) for url in urls]
        pages = await asyncio.gather(*tasks)

        for url, html in pages:
            title = _page_title(html)
            markdown = _extract_markdown(html)
            output_path = _url_to_output_path(url)
            output_path = output_dir / output_path.relative_to(OUTPUT_DIR)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                _frontmatter(title=title, source=url) + markdown + "\n",
                encoding="utf-8",
            )

        freshness_parts = [sitemap_xml]
        for source in FRESHNESS_SOURCES[1:]:
            response = await client.get(source)
            response.raise_for_status()
            freshness_parts.append(_sanitize_text(response.text))

    _build_library_readme(urls, output_dir=output_dir)
    total_size_kb = round(
        sum(path.stat().st_size for path in output_dir.rglob("*.md")) / 1024,
        1,
    )
    source_digest = compute_source_digest(freshness_parts)
    write_snapshot_metadata(
        output_dir,
        source_ref=source_digest,
        pages_scraped=len(urls),
        total_size_kb=total_size_kb,
    )
    update_freshness_entry(
        "onepassword_developer",
        strategy="source_digest",
        site_url=f"{DOCS_PREFIX}/",
        doc_search_id=LIBRARY_ID,
        sources=FRESHNESS_SOURCES,
        source_digest=source_digest,
        last_scraped=datetime.now(UTC).date().isoformat(),
        pages_scraped=len(urls),
        total_size_kb=total_size_kb,
        persist=persist_freshness,
    )
    print(f"Fetched {len(urls)} docs pages into {output_dir}")
    return source_digest


def publish_to_repowise(
    output_dir: Path | None = None,
    *,
    cleanup_output: bool = False,
) -> None:
    publish_output_dir(
        LIBRARY_ID,
        output_dir or OUTPUT_DIR,
        cleanup_output=cleanup_output,
    )


def show_stats(output_dir: Path | None = None) -> None:
    target_dir = output_dir or OUTPUT_DIR
    if not target_dir.exists():
        print("No output directory found.")
        return

    md_files = sorted(target_dir.rglob("*.md"))
    total_size = sum(path.stat().st_size for path in md_files)
    print(f"Summary: {len(md_files)} markdown files, {total_size / 1024:.1f} KB total")
    for path in md_files[:20]:
        print(f"  {path.relative_to(target_dir)}")
    if len(md_files) > 20:
        print(f"  ... and {len(md_files) - 20} more")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    if command == "--fetch":
        asyncio.run(fetch_docs())
    elif command == "--publish":
        publish_to_repowise()
    elif command == "--stats":
        show_stats()
    elif command == "--all":
        asyncio.run(fetch_docs())
        publish_to_repowise(cleanup_output=True)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
