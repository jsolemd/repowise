#!/usr/bin/env python3
"""
Cosmograph docs scraper for CodeAtlas snapshot publishing.

Scrapes https://cosmograph.app documentation (docs-lib, docs-general,
docs-widget, docs-app) into current-state markdown snapshots.

Primary path:
    - Fetch server-rendered docs HTML directly
    - Discover canonical docs URLs from the section roots
    - Convert the article content to markdown
    - Update freshness metadata from Next.js build ID

Legacy path:
    - Accept a Chrome DevTools MCP browser export and convert it to markdown

Usage:
    python cosmograph.py --fetch
    python cosmograph.py --process <exported-json-file>
    python cosmograph.py --publish
    python cosmograph.py --all

Library ID: /codeatlas/cosmograph
"""

import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

try:
    from .common import publish_output_dir, update_freshness_entry, write_snapshot_metadata
    from .html_markdown import html_to_markdown
except ImportError:
    from common import publish_output_dir, update_freshness_entry, write_snapshot_metadata
    from html_markdown import html_to_markdown

OUTPUT_DIR = Path("/tmp/cosmograph-docs")
BASE_URL = "https://cosmograph.app"
LIBRARY_ID = "/codeatlas/cosmograph"
SECTION_ROOTS = ["/docs-general/", "/docs-widget/", "/docs-app/", "/docs-lib/"]
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml",
}

# --- JavaScript snippets for Chrome DevTools MCP ---

DISCOVER_URLS_JS = """
async () => {
  const sectionsToDiscover = ['/docs-general/', '/docs-widget/', '/docs-app/', '/docs-lib/'];
  const allUrls = new Set();

  for (const section of sectionsToDiscover) {
    try {
      await window.next.router.push(section);
      await new Promise(r => setTimeout(r, 2000));
      const links = document.querySelectorAll('aside a[href*="/docs-"]');
      links.forEach(a => {
        const href = a.getAttribute('href');
        if (href && href.startsWith('/docs-')) allUrls.add(href);
      });
    } catch(e) {}
  }

  return { totalUrls: allUrls.size, urls: Array.from(allUrls).sort() };
}
"""

SETUP_JS = """
async () => {
  window._allUrls = %URLS%;  // Replace with discovered URLs
  window._scraped = {};

  window._extractPage = () => {
    const article = document.querySelector('article');
    if (!article) return null;
    const content = article.querySelector('.nextra-content') || article;
    const h1 = article.querySelector('h1');
    return {
      title: h1 ? h1.innerText : '',
      text: content.innerText,
      textLen: content.innerText.length
    };
  };

  return { ready: true, totalPages: window._allUrls.length };
}
"""

BATCH_SCRAPE_JS = """
async () => {
  const BATCH_SIZE = 12;
  const start = %START%;  // Replace with batch start index
  const end = Math.min(start + BATCH_SIZE, window._allUrls.length);
  const results = {};

  for (let i = start; i < end; i++) {
    const url = window._allUrls[i];
    try {
      await window.next.router.push(url);
      await new Promise(r => setTimeout(r, 2500));
      const data = window._extractPage();
      if (data && data.textLen > 50) {
        results[url] = { title: data.title, text: data.text };
      } else {
        results[url] = { error: 'no content', textLen: data ? data.textLen : 0 };
      }
    } catch(e) { results[url] = { error: e.message }; }
  }

  Object.assign(window._scraped, results);
  const allKeys = Object.keys(window._scraped);
  const errors = allKeys.filter(k => window._scraped[k].error);

  return {
    batch: start + '-' + (end-1),
    scraped: Object.keys(results).length,
    total: allKeys.length,
    errors: errors.length,
    nextStart: end,
    done: end >= window._allUrls.length
  };
}
"""

EXPORT_JS = """
() => { return JSON.stringify(window._scraped); }
"""

# --- Breadcrumb patterns to strip from page tops ---

BREADCRUMBS = {
    "JavaScript & React library",
    "General",
    "Widget in Jupyter Notebook",
    "Web application",
    "Data requirements",
    "Features",
    "Components",
    "Legends",
    "API",
    "Classes",
    "Interfaces",
    "Enumerations",
    "Type Aliases",
    "Functions",
}


def text_to_markdown(text: str, url: str, title: str) -> str:
    """Convert extracted text content to clean markdown with frontmatter."""
    lines = text.split("\n")
    cleaned_lines = []
    found_title = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if found_title:
                cleaned_lines.append("")
            continue
        if not found_title and len(stripped) < 40 and stripped in BREADCRUMBS:
            continue
        found_title = True
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\nLast updated on.*$", "", text, flags=re.MULTILINE)

    return f"""---
title: "{title}"
source: "{BASE_URL}{url}"
scraped: true
---

{text}
"""


def url_to_filepath(url: str) -> Path:
    clean = url.strip("/")
    if not clean:
        clean = "index"
    return OUTPUT_DIR / f"{clean}.md"


def normalize_docs_path(url_or_path: str) -> str | None:
    parsed = urlparse(url_or_path)
    path = parsed.path
    if not path.startswith("/docs-"):
        return None
    return path if path.endswith("/") else f"{path}/"


def cleanup_html_markdown(markdown: str) -> str:
    lines = [line.rstrip() for line in markdown.splitlines()]
    start = 0
    for index, line in enumerate(lines):
        if line.startswith("# "):
            start = index
            break
    cleaned = "\n".join(lines[start:]).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return re.sub(r"\nLast updated on.*$", "", cleaned, flags=re.MULTILINE)


def discover_site_urls() -> tuple[list[str], str]:
    urls = set(SECTION_ROOTS)
    build_id = ""

    with httpx.Client(
        headers=REQUEST_HEADERS,
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        for root in SECTION_ROOTS:
            response = client.get(urljoin(BASE_URL, root))
            response.raise_for_status()
            html = response.text
            if not build_id:
                match = re.search(r'"buildId":"([^"]+)"', html)
                if match:
                    build_id = match.group(1)
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                normalized = normalize_docs_path(link.get("href", ""))
                if normalized is not None:
                    urls.add(normalized)

    return sorted(urls), build_id


def probe_source_ref() -> str:
    _, build_id = discover_site_urls()
    return build_id


def fetch_from_site(
    output_dir: Path | None = None,
    *,
    persist_freshness: bool = True,
) -> str:
    output_dir = output_dir or OUTPUT_DIR
    urls, build_id = discover_site_urls()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with httpx.Client(
        headers=REQUEST_HEADERS,
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        for path in urls:
            url = urljoin(BASE_URL, path)
            response = client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("article")
            if article is None:
                print(f"  SKIP (no article): {path}")
                continue

            title_tag = article.find("h1") or soup.find("title")
            title = (
                title_tag.get_text(" ", strip=True)
                if title_tag
                else path.rstrip("/").split("/")[-1]
            )
            markdown = html_to_markdown(
                str(article),
                selectors=["article"],
                prune_selectors=["script", "style", "nav", "header", "footer"],
            )
            markdown = cleanup_html_markdown(markdown)

            output = url_to_filepath(path)
            output = output_dir / output.relative_to(OUTPUT_DIR)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                f"""---
title: "{title.replace('"', "'")}"
source: "{url}"
scraped: true
---

{markdown}
""",
                encoding="utf-8",
            )

    total_size = sum(file.stat().st_size for file in output_dir.rglob("*.md"))
    write_snapshot_metadata(
        output_dir,
        source_ref=build_id,
        pages_scraped=sum(1 for _ in output_dir.rglob("*.md")),
        total_size_kb=round(total_size / 1024, 1),
    )
    update_freshness_entry(
        "cosmograph",
        build_id=build_id,
        site_url=BASE_URL,
        last_scraped=datetime.now(UTC).date().isoformat(),
        doc_search_id=LIBRARY_ID,
        pages_scraped=sum(1 for _ in output_dir.rglob("*.md")),
        total_size_kb=round(total_size / 1024, 1),
        persist=persist_freshness,
    )
    print(f"Fetched {len(urls)} pages -> {output_dir} ({total_size / 1024:.1f} KB)")
    return build_id


def process_export(input_file: str) -> None:
    """Process exported JSON from Chrome DevTools into markdown files."""
    with open(input_file) as f:
        raw = json.load(f)

    # Handle MCP tool result wrapper: [{type, text}]
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
        text = raw[0].get("text", "")
        json_match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text
    else:
        json_str = json.dumps(raw)

    parsed = json.loads(json_str)
    data = json.loads(parsed) if isinstance(parsed, str) else parsed

    print(f"Processing {len(data)} pages...")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    success = 0
    for url, page_data in sorted(data.items()):
        if isinstance(page_data, dict) and "error" in page_data:
            print(f"  SKIP (error): {url} - {page_data['error']}")
            continue

        title = page_data.get("title", url.split("/")[-2])
        text = page_data.get("text", "")

        if len(text) < 50:
            print(f"  SKIP (short): {url} ({len(text)} chars)")
            continue

        md = text_to_markdown(text, url, title)
        filepath = url_to_filepath(url)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(md)
        success += 1

    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.md"))
    print(f"\nDone: {success} pages -> {OUTPUT_DIR} ({total_size / 1024:.1f} KB)")


def publish_to_repowise(
    output_dir: Path | None = None,
    *,
    cleanup_output: bool = False,
) -> None:
    """Publish updated docs into CodeAtlas current-state storage."""
    target_dir = output_dir or OUTPUT_DIR
    if not target_dir.exists():
        raise SystemExit("Run --fetch or --process first so there is scraped output to publish.")
    publish_output_dir(
        LIBRARY_ID,
        target_dir,
        cleanup_output=cleanup_output,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python cosmograph.py --fetch")
        print("  python cosmograph.py --process <exported-json-file>")
        print("  python cosmograph.py --publish")
        print("  python cosmograph.py --all")
        sys.exit(0)

    if sys.argv[1] == "--fetch":
        fetch_from_site()

    elif sys.argv[1] == "--process":
        if len(sys.argv) < 3:
            print("Error: provide the exported JSON file path")
            sys.exit(1)
        process_export(sys.argv[2])

    elif sys.argv[1] == "--publish":
        publish_to_repowise()

    elif sys.argv[1] == "--all":
        fetch_from_site()
        publish_to_repowise(cleanup_output=True)

    else:
        print(f"Unknown command: {sys.argv[1]}")
        sys.exit(1)
