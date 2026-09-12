#!/usr/bin/env python3
"""
Semantic Scholar API docs scraper for RepoWise snapshot publishing.

Scrapes:
  1. https://www.semanticscholar.org/product/api (overview, tutorial, gallery)
  2. https://api.semanticscholar.org (swagger specs for Graph, Recommendations, Datasets)

Unlike the Cosmograph scraper (which needs Chrome DevTools for SPA rendering),
the S2 API docs are mostly server-rendered HTML + Swagger JSON specs that can be
fetched directly.

Library ID: /codeatlas/semantic-scholar-api

Usage:
    # Fetch swagger specs and rebuild markdown
    python semantic_scholar.py --fetch

    # Post-process already-fetched JSON swagger specs
    python semantic_scholar.py --process

    # Publish updated docs into RepoWise
    python semantic_scholar.py --publish

    # Full pipeline: fetch + process + publish
    python semantic_scholar.py --all
"""

import json
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
except ImportError:
    from common import (
        compute_source_digest,
        publish_output_dir,
        update_freshness_entry,
        write_snapshot_metadata,
    )

OUTPUT_DIR = Path("/tmp/s2-api-docs")
SWAGGER_DIR = OUTPUT_DIR / "swagger-json"
LIBRARY_ID = "/codeatlas/semantic-scholar-api"

# URLs for the three swagger specs
SWAGGER_URLS = {
    "graph": "https://api.semanticscholar.org/graph/v1/swagger.json",
    "recommendations": "https://api.semanticscholar.org/recommendations/v1/swagger.json",
    "datasets": "https://api.semanticscholar.org/datasets/v1/swagger.json",
}

# Tutorial and overview pages (server-rendered, fetchable via requests)
PAGE_URLS = {
    "overview": "https://www.semanticscholar.org/product/api",
    "tutorial": "https://www.semanticscholar.org/product/api/tutorial",
    "gallery": "https://www.semanticscholar.org/product/api/gallery",
}


def probe_source_ref() -> str:
    fetched_payloads: list[str] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for url in SWAGGER_URLS.values():
            response = client.get(url)
            response.raise_for_status()
            fetched_payloads.append(json.dumps(response.json(), indent=2, sort_keys=True))
    return compute_source_digest(fetched_payloads)


def fetch_swagger_specs(
    output_dir: Path | None = None,
    *,
    persist_freshness: bool = True,
) -> str:
    """Download the three Swagger/OpenAPI JSON specs."""
    output_dir = output_dir or OUTPUT_DIR
    swagger_dir = output_dir / "swagger-json"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    swagger_dir.mkdir(parents=True, exist_ok=True)
    fetched_payloads: list[str] = []

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for name, url in SWAGGER_URLS.items():
            print(f"Fetching {name} swagger spec...")
            response = client.get(url)
            response.raise_for_status()
            payload = json.dumps(response.json(), indent=2, sort_keys=True)
            outfile = swagger_dir / f"{name}.json"
            outfile.write_text(payload)
            fetched_payloads.append(payload)
            print(f"  -> {outfile} ({outfile.stat().st_size / 1024:.1f} KB)")

    source_digest = compute_source_digest(fetched_payloads)
    write_snapshot_metadata(
        output_dir,
        source_ref=source_digest,
    )
    update_freshness_entry(
        "semantic_scholar",
        site_url="https://www.semanticscholar.org/product/api",
        swagger_urls=list(SWAGGER_URLS.values()),
        source_digest=source_digest,
        last_scraped=datetime.now(UTC).date().isoformat(),
        doc_search_id=LIBRARY_ID,
        persist=persist_freshness,
    )
    return source_digest


def swagger_to_markdown(spec: dict, api_name: str, source_url: str) -> str:
    """Convert a Swagger 2.0 spec to clean markdown documentation."""
    lines = [
        "---",
        f'title: "{spec.get("info", {}).get("title", api_name)} API Reference"',
        f'source: "{source_url}"',
        "scraped: true",
        "---",
        "",
        f"# {spec.get('info', {}).get('title', api_name)}",
        "",
        f"**Base Path:** `{spec.get('basePath', '/')}`",
        "",
    ]

    # Endpoints
    paths = spec.get("paths", {})
    if paths:
        lines.append("## Endpoints")
        lines.append("")

        for path, methods in sorted(paths.items()):
            for method, details in methods.items():
                if method in ("get", "post", "put", "delete", "patch"):
                    summary = details.get("summary", "")
                    lines.append(f"### {method.upper()} {path}")
                    if summary:
                        lines.append(f"\n{summary}")
                    lines.append("")

                    # Parameters
                    params = details.get("parameters", [])
                    if params:
                        lines.append("**Parameters:**")
                        lines.append("")
                        lines.append("| Name | In | Type | Required | Description |")
                        lines.append("|------|-----|------|----------|-------------|")
                        for p in params:
                            name = p.get("name", "")
                            loc = p.get("in", "")
                            ptype = p.get("type", p.get("schema", {}).get("type", "object"))
                            req = "Yes" if p.get("required") else "No"
                            desc = p.get("description", "").replace("\n", " ")[:100]
                            lines.append(f"| `{name}` | {loc} | {ptype} | {req} | {desc} |")
                        lines.append("")

                    # Responses
                    responses = details.get("responses", {})
                    if responses:
                        lines.append("**Responses:**")
                        lines.append("")
                        for code, resp in sorted(responses.items()):
                            desc = resp.get("description", "")
                            schema_ref = ""
                            if "schema" in resp:
                                ref = resp["schema"].get("$ref", "")
                                if ref:
                                    schema_ref = f" → `{ref.split('/')[-1]}`"
                            lines.append(f"- **{code}**: {desc}{schema_ref}")
                        lines.append("")

                    lines.append("---")
                    lines.append("")

    # Definitions / Schemas
    definitions = spec.get("definitions", {})
    if definitions:
        lines.append("## Schema Definitions")
        lines.append("")

        for name, schema in sorted(definitions.items()):
            lines.append(f"### {name}")
            desc = schema.get("description", "")
            if desc:
                lines.append(f"\n{desc}")
            lines.append("")

            props = schema.get("properties", {})
            if props:
                lines.append("| Field | Type | Description |")
                lines.append("|-------|------|-------------|")
                for field_name, field_def in props.items():
                    ftype = field_def.get("type", "")
                    if not ftype and "$ref" in field_def:
                        ftype = field_def["$ref"].split("/")[-1]
                    if field_def.get("items"):
                        item_type = field_def["items"].get("type", "")
                        if not item_type and "$ref" in field_def.get("items", {}):
                            item_type = field_def["items"]["$ref"].split("/")[-1]
                        ftype = f"array[{item_type}]"
                    fdesc = field_def.get("description", "").replace("\n", " ")[:120]
                    lines.append(f"| `{field_name}` | {ftype} | {fdesc} |")
                lines.append("")

    return "\n".join(lines)


def process_swagger_specs(output_dir: Path | None = None) -> None:
    """Convert downloaded swagger specs to markdown files."""
    output_dir = output_dir or OUTPUT_DIR
    swagger_dir = output_dir / "swagger-json"
    api_ref_dir = output_dir / "api-reference"
    api_ref_dir.mkdir(parents=True, exist_ok=True)

    for name, url in SWAGGER_URLS.items():
        json_file = swagger_dir / f"{name}.json"
        if not json_file.exists():
            print(f"  SKIP: {json_file} not found (run --fetch first)")
            continue

        spec = json.loads(json_file.read_text())
        md = swagger_to_markdown(spec, name.replace("-", " ").title(), url)
        out = api_ref_dir / f"{name}-api-generated.md"
        out.write_text(md)
        print(f"  {name} -> {out} ({out.stat().st_size / 1024:.1f} KB)")


def publish_to_repowise(
    output_dir: Path | None = None,
    *,
    cleanup_output: bool = False,
) -> None:
    """Publish updated docs into RepoWise current-state storage."""
    target_dir = output_dir or OUTPUT_DIR
    # Remove swagger JSON dir before pushing (only push markdown)
    swagger_dir = target_dir / "swagger-json"
    if swagger_dir.exists():
        shutil.rmtree(swagger_dir)
    publish_output_dir(
        LIBRARY_ID,
        target_dir,
        cleanup_output=cleanup_output,
    )


def show_stats(output_dir: Path | None = None) -> None:
    """Print summary of generated docs."""
    target_dir = output_dir or OUTPUT_DIR
    if not target_dir.exists():
        print("No output directory found.")
        return

    md_files = list(target_dir.rglob("*.md"))
    total_size = sum(f.stat().st_size for f in md_files)
    print(f"\nSummary: {len(md_files)} markdown files, {total_size / 1024:.1f} KB total")
    for f in sorted(md_files):
        rel = f.relative_to(target_dir)
        print(f"  {rel} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUsage:")
        print("  python semantic_scholar.py --fetch    # Download swagger JSON specs")
        print("  python semantic_scholar.py --process  # Convert JSON -> markdown")
        print("  python semantic_scholar.py --publish  # Publish to RepoWise")
        print("  python semantic_scholar.py --all      # Full pipeline")
        print("  python semantic_scholar.py --stats    # Show output summary")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "--fetch":
        fetch_swagger_specs()
    elif cmd == "--process":
        process_swagger_specs()
    elif cmd == "--publish":
        publish_to_repowise()
    elif cmd == "--stats":
        show_stats()
    elif cmd == "--all":
        fetch_swagger_specs()
        process_swagger_specs()
        publish_to_repowise(cleanup_output=True)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
