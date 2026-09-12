# Library Documentation Search

RepoWise owns documentation ingestion, storage access, retrieval, and management.
Code and docs tools share the RepoWise MCP at `:7350/mcp`; the dashboard's
`/docs-libraries` page lists libraries and opens a searchable document inventory.
The optional `repowise-docs` worker runs the indexer, freshness scheduler, and
recovery loop. Its `:8101` API is internal to the MCP and dashboard.

The engine and scrapers live in this module. SoleMD.Infra owns deployment and
`infra/repowise/docs/libraries.yaml`. Both code and external references use RepoWise;
use code tools for repository evidence and docs tools for library references.

The local library documentation index is the primary source for external
library references in SoleMD work. Use Context7 or web search only when its
docs are missing, stale, unavailable, or clearly low-quality for the question,
and record that fallback reason in the answer or working note. If the gap should persistently affect
future work, fix the library registration or run `update_doc_library` instead
of relying on the fallback.

## Documentation reads

- `list_doc_files`
- `resolve_library_id`
- `search_docs`
- `expand_doc_chunk`
- `list_doc_libraries`
- `read_doc`

## Documentation management

- `update_doc_library`
- `add_doc_library`
- `delete_doc_library`
- `export_doc_bundle`
- `import_doc_bundle`

The worker retains the internal `search_docs_multi` compatibility alias. On RepoWise, use `search_docs(library_ids=[...])` for cross-library search.

## Runtime Shape

```
MCP client → RepoWise :7350 ─┐
                            ├→ Docs worker :8101 → PostgreSQL / Qdrant / TEI
Dashboard → RepoWise API ────┘
```

The internal worker's `GET /readyz` reports the worker, scheduler, recovery loop,
and dependency state. It publishes no MCP endpoint. Docs-to-Neo4j metadata sync
remains retired; health reports it explicitly as disabled.

### Single writer

Exactly one process may run the background runtime. That is enforced by a
PostgreSQL session-level advisory lock (`repowise/docs/jobs/single_writer.py`), not
by configuration: a second replica logs the refusal, keeps serving reads, and
starts no worker, scheduler, recovery loop, or registry sync. Its `/readyz`
reports `background_declined_reason` and degrades to 503, so it is never
mistaken for the writer.

Operational docs-admin endpoints on `http://localhost:8101`:

- `GET /docs/health`
- `GET /docs/stats`
- `POST /docs/action/index?library=/owner/repo`
- `GET /docs/action/jobs`
- `GET /docs/action/jobs/{job_id}`

`POST /docs/action/index` changes server state and is gated, fail-closed:

| Configuration | Result |
|---|---|
| `DOC_SEARCH_WEBHOOK_TOKEN` set | require a matching `X-Doc-Search-Token` header |
| no token, `DOC_SEARCH_ALLOW_UNAUTHENTICATED=1` | allow (trusted network only) |
| neither | refuse with `401 unauthenticated_mutation_refused` |

`X-Code-Search-Token` is still accepted as a fallback header for one release,
for callers written against the retired proxy. The `GET` job routes are reads
and are not behind the gate.

The management CLI remains:

```bash
solemd compose exec repowise-docs repowise-docs seed
```

## Default Workflow

```python
resolve_library_id(library_name="mantine")
search_docs(
    library_id="/mantinedev/mantine",
    query="Combobox selected option styles",
    output="markdown",
)
read_doc(
    library_id="/mantinedev/mantine",
    path="apps/mantine.dev/src/pages/core/combobox.mdx",
)
```

Rules:

- Try the local library documentation index first: `resolve_library_id` → `search_docs` → `read_doc` /
  `expand_doc_chunk`. Pass `library_ids` for a cross-library query.
- Use Context7 only as a fallback when the local library is absent, stale,
  unavailable, or returns low-quality results for the task.
- Use `exact_match=true` for API/function/class names.
- Use `search_docs(library_ids=[...])` only for genuine cross-library comparison.
- Prefer `search_docs` previews for discovery and escalate to `read_doc` or
  `expand_doc_chunk` for full context.

## Freshness Model

- Libraries are stored in PostgreSQL with durable job state.
- The background worker, freshness scheduler, and orphan recovery loop run
  inside the `repowise-docs` lifecycle, held by one process at a time through
  the advisory lock described above.
- Snapshot-backed libraries with registered scrapers are bootstrapped directly
  into Postgres snapshot state during runtime startup before background work
  begins.
- The freshness scheduler probes live upstream source refs for those snapshot
  libraries and republishes them directly when they drift.
- Docs readiness and scheduler health are exposed through `/docs/health` and
  `/docs/stats`.
- `update_doc_library` triggers manual refresh when docs appear stale.

## Adding a Library

Do not guess repo layout.

1. Identify the correct upstream repo and branch.
2. Verify the docs layout.
3. Register with explicit include/exclude patterns only when needed.

```python
add_doc_library(
    repo="owner/repo",
    name="Library Name",
    branch="main",
    include_patterns=["**/*.md", "**/*.mdx"],
)
```

## Scraped Docs

Scrapers live in `repowise.docs.scrapers`. They do not publish to mirror repos.
They build a temporary markdown tree, publish the latest current-state snapshot
directly into PostgreSQL, and let the normal index pipeline consume that
snapshot.

Operational rules:
- Git-backed libraries still use `/var/cache/code-search/doc-search/repos/<owner>/<repo>`.
- Snapshot-backed libraries publish only the latest logical file set into the
  docs database.
- Manual scraper runs may use stable scratch paths under `/tmp`.
- Runtime automation uses temporary directories and removes them immediately
  after snapshot publication.
