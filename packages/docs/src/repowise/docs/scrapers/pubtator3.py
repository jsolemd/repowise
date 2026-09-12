#!/usr/bin/env python3
"""
PubTator3 docs scraper for CodeAtlas snapshot publishing.

This scraper builds a current-state docs bundle from the directly accessible
PubTator3 sources rather than mirroring a synthetic GitHub repo.

Library ID: /codeatlas/pubtator3

Usage:
    python pubtator3.py --fetch
    python pubtator3.py --publish
    python pubtator3.py --all
"""

from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

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

OUTPUT_DIR = Path("/tmp/pubtator3-docs")
LIBRARY_ID = "/codeatlas/pubtator3"
README_URL = "https://raw.githubusercontent.com/ncbi-nlp/pubtator-gpt/main/README.md"
USAGE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/usage/"
API_HOME_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3/api"
FTP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _frontmatter(title: str, source: str) -> str:
    return f'---\ntitle: "{title}"\nsource: "{source}"\nscraped: true\n---\n\n'


def _fetch_text(url: str) -> str:
    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    return response.text


def _build_overview_markdown() -> str:
    return _frontmatter("PubTator3 API Overview", API_HOME_URL) + "\n".join(
        [
            "# PubTator3 API Overview",
            "",
            "## Canonical sources",
            "",
            f"- API landing page: {API_HOME_URL}",
            f"- REST usage guide: {USAGE_URL}",
            f"- GPT/OpenAPI guidance: {README_URL}",
            f"- Bulk FTP distribution: {FTP_URL}",
            "",
            "## Core endpoints",
            "",
            "- `GET /research/pubtator3-api/entity/autocomplete/`",
            "  Query parameter: `query`",
            "- `GET /research/pubtator3-api/relations`",
            "  Query parameters: `e1`, `e2`, `type`, optional `limit`",
            "- `GET /research/pubtator3-api/search/`",
            "  Query parameter: `text`, optional `page`",
            "",
            "## Notes",
            "",
            "- The endpoint-level parameter and schema details are captured in `gpt-integration.md` from the official PubTator GPT guidance repository.",
            "- The REST usage page documents authentication-free HTTP usage patterns and URL construction.",
            "- The FTP distribution is useful for bulk/offline workflows rather than interactive API queries.",
        ]
    )


def probe_source_ref() -> str:
    readme = _fetch_text(README_URL)
    usage_html = _fetch_text(USAGE_URL)
    overview = _build_overview_markdown()
    return compute_source_digest([readme, usage_html, overview])


def fetch(
    output_dir: Path | None = None,
    *,
    persist_freshness: bool = True,
) -> str:
    output_dir = output_dir or OUTPUT_DIR
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    readme = _fetch_text(README_URL)
    _write(
        output_dir / "gpt-integration.md",
        _frontmatter("PubTator3 GPT Integration", README_URL) + readme,
    )

    usage_html = _fetch_text(USAGE_URL)
    usage_markdown = html_to_markdown(
        usage_html,
        selectors=["main", "#main-content", ".usa-layout-docs__main", "article", "body"],
        prune_selectors=["script", "style", "nav", "header", "footer"],
    )
    _write(
        output_dir / "rest-usage.md",
        _frontmatter("PubTator3 REST Usage", USAGE_URL) + usage_markdown,
    )

    overview = _build_overview_markdown()
    _write(output_dir / "overview.md", overview)

    source_digest = compute_source_digest([readme, usage_html, overview])
    total_size_kb = round(sum(path.stat().st_size for path in output_dir.rglob("*.md")) / 1024, 1)
    write_snapshot_metadata(
        output_dir,
        source_ref=source_digest,
        pages_scraped=3,
        total_size_kb=total_size_kb,
    )

    update_freshness_entry(
        "pubtator3",
        site_url=API_HOME_URL,
        sources=[USAGE_URL, README_URL, FTP_URL],
        source_digest=source_digest,
        last_scraped=datetime.now(UTC).date().isoformat(),
        doc_search_id=LIBRARY_ID,
        pages_scraped=3,
        total_size_kb=total_size_kb,
        persist=persist_freshness,
    )
    return source_digest


def publish(
    output_dir: Path | None = None,
    *,
    cleanup_output: bool = False,
) -> None:
    target_dir = output_dir or OUTPUT_DIR
    if not target_dir.exists():
        raise SystemExit("Run --fetch first so there is scraped output to publish.")
    publish_output_dir(
        LIBRARY_ID,
        target_dir,
        cleanup_output=cleanup_output,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python pubtator3.py --fetch")
        print("  python pubtator3.py --publish")
        print("  python pubtator3.py --all")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "--fetch":
        fetch()
    elif cmd == "--publish":
        publish()
    elif cmd == "--all":
        fetch()
        publish(cleanup_output=True)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
