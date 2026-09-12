#!/usr/bin/env bash
# Check if scraped documentation sites have been updated.
# Supports build-ID probes for SPA docs and content-digest probes for directly
# fetched text/API sources.
#
# Usage:
#   ./check_freshness.sh          # Check all sites
#   ./check_freshness.sh --quiet  # Exit code only (0=fresh, 1=stale)
#
# Returns exit code 1 if any site is stale (needs re-scraping).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRESHNESS_FILE="$SCRIPT_DIR/freshness.json"
QUIET="${1:-}"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

if [ ! -f "$FRESHNESS_FILE" ]; then
    echo "ERROR: freshness.json not found at $FRESHNESS_FILE"
    exit 2
fi

python3 - <<'PY' "$FRESHNESS_FILE" "$QUIET" "$UA"
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

freshness_path = Path(sys.argv[1])
quiet = bool(sys.argv[2])
user_agent = sys.argv[3]

data = json.loads(freshness_path.read_text())
stale = 0

def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=10) as response:  # noqa: S310 - controlled urls from repo config
        return response.read().decode("utf-8", errors="replace")

for site, config in data.items():
    strategy = config.get("strategy", "manual")
    last_scraped = config.get("last_scraped", "unknown")

    if strategy == "next_build":
        site_url = config["site_url"]
        stored = config.get("build_id", "")
        body = fetch_text(site_url.rstrip("/") + "/")
        match = re.search(r'"buildId":"([^"]+)"', body)
        current = match.group(1) if match else "UNKNOWN"
        if current == "UNKNOWN":
            if not quiet:
                print(f"⚠  {site}: Could not fetch build ID from {site_url}")
            continue
        if current == stored:
            if not quiet:
                print(f"✓  {site}: Fresh (build {current}, scraped {last_scraped})")
            continue
        stale = 1
        if not quiet:
            print(f"✗  {site}: STALE — site rebuilt")
            print(f"   Stored:  {stored} (scraped {last_scraped})")
            print(f"   Current: {current}")
            print(f"   → Re-run scraper: python {freshness_path.parent / (site + '.py')}")
        continue

    if strategy == "source_digest":
        urls = config.get("swagger_urls") or config.get("sources") or []
        stored = config.get("source_digest", "")
        digest = hashlib.sha256()
        for url in urls:
            digest.update(fetch_text(url).encode("utf-8"))
            digest.update(b"\n\0\n")
        current = digest.hexdigest()
        if current == stored:
            if not quiet:
                print(f"✓  {site}: Fresh (digest {current[:12]}, scraped {last_scraped})")
            continue
        stale = 1
        if not quiet:
            print(f"✗  {site}: STALE — source digest changed")
            print(f"   Stored:  {stored[:12] if stored else 'NONE'} (scraped {last_scraped})")
            print(f"   Current: {current[:12]}")
            print(f"   → Re-run scraper: python {freshness_path.parent / (site + '.py')}")
        continue

    if not quiet:
        print(f"⚠  {site}: No freshness strategy configured")

sys.exit(stale)
PY
