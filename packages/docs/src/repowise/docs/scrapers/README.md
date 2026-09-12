# RepoWise documentation scrapers

Reusable scrapers for documentation sources that do not fit the normal
git-clone indexing path. These scripts produce markdown files in `/tmp`, then
publish the latest current-state snapshot directly into RepoWise's Postgres
docs store. No GitHub mirror repos are involved.

## Architecture

```
Closed-source or irregular docs source
        │
        ▼
  scraper script (Chrome DevTools MCP, curl, or requests)
        │
        ▼
  /tmp/<library>-docs/  (generated markdown files)
        │
        ▼
  publish current-state snapshot into RepoWise Postgres
        │
        ▼
  queue incremental/full reindex in RepoWise
```

Current model:
- Real upstream docs repos stay on the git-backed indexing path.
- Scraped/generated docs use `source_type: snapshot` and publish only the latest
  current-state file set.
- Scratch outputs belong in `/tmp`. Runtime git caches belong under
  `/var/cache/code-search/doc-search/repos`. They do not belong inside the
  `packages/docs/` source tree.

## Available Scrapers

| Script | Target | Output library | Method |
|--------|--------|----------------|--------|
| `cosmograph.py` | `cosmograph.app` docs | `/codeatlas/cosmograph` | HTTP fetch + HTML to markdown |
| `onepassword_developer.py` | `developer.1password.com/docs/*` | `/codeatlas/1password-developer` | Official sitemap crawl + HTML to markdown |
| `semantic_scholar.py` | Semantic Scholar API docs | `/codeatlas/semantic-scholar-api` | HTTP fetch + markdown normalization |
| `pubtator3.py` | PubTator3 docs and API references | `/codeatlas/pubtator3` | HTTP fetch + curated overview |

## Usage

### Running a scraper

Run the full flow inside the configured RepoWise worker. Fetching and publishing
then share scratch paths and the worker's database configuration:

```bash
docker exec repowise-docs python -m repowise.docs.scrapers.cosmograph --all
docker exec repowise-docs python -m repowise.docs.scrapers.semantic_scholar --all
```

`--all` fetches, normalizes, publishes, and removes its scratch directory after
publication. `--publish` alone uses previously generated files, updates snapshot
and freshness rows, and queues indexing. The source package is installed in the
worker image; rebuild through Infra after changing scraper code.

For fetch-only debugging from the RepoWise checkout, use
`uv run --frozen --extra docs python -m repowise.docs.scrapers.cosmograph --fetch`.
Its output remains in the host's `/tmp/cosmograph-docs`; copy it into the worker's
matching path before publishing from there.

## Writing a New Scraper

Use this pattern:
1. Fetch or render the upstream docs into `/tmp/<library>-docs`.
2. Normalize to markdown files with stable relative paths.
3. Expose a deterministic `probe_source_ref()` for the scheduler/runtime path.
4. Update `freshness.json` only for host-side operational visibility.
5. Publish through `publish_output_dir()` into the configured snapshot library.

Good defaults:
- Put canonical source URLs in frontmatter or adjacent metadata.
- Keep file paths stable so incremental indexing only touches changed files.
- Prefer deterministic text normalization over HTML dumps.

## Freshness Detection

Scraped docs do not have an upstream git commit to compare. Runtime automation
uses scraper-specific `probe_source_ref()` functions to derive a live source
signal, then republishes directly into the snapshot store when that signal
changes.

Supported strategies:
- `next_build`: Next.js build ID from `__NEXT_DATA__`
- `source_digest`: content digest derived from one or more canonical pages

`freshness.json` is still kept for host-side operational inspection and the
manual `check_freshness.sh` script. It is not the runtime source of truth.

`check_freshness.sh` compares the stored signal to the live site and exits:
- `0` when all configured snapshot sources are fresh
- `1` when any configured snapshot source is stale

Cleanup contract:
- Manual `--all` flows may use stable `/tmp/<library>-docs` directories and
  delete them after publish.
- Runtime bootstrap and scheduler refreshes use temporary directories that are
  removed automatically once the snapshot has been embedded into Postgres.

Historical `/codeatlas/...` library IDs and `.codeatlas-snapshot.json` metadata
filenames remain stable compatibility identifiers. RepoWise owns their runtime.
