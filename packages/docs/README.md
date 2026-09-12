# RepoWise documentation backend

This module owns external library registration, Git and snapshot ingestion,
chunking, hybrid retrieval, document browsing, freshness, and indexing jobs.
RepoWise MCP and API call its optional worker through `repowise.docs.client`.
The worker publishes an internal HTTP tool interface, not a separate MCP server.

Build with `docker build -t repowise-docs packages/docs`. Deployment and library
configuration live in SoleMD.Infra's `infra/repowise/docs/`. The worker's isolated
lock keeps database, scraper, and embedding dependencies out of the code navigation
process. At the monorepo root, use `uv sync --extra docs` and
`uv run --extra docs pytest tests/docs -m 'not integration and not eval'`.

Persistent SQL schemas, collections, cache paths, and library IDs are retained
from the previous deployment. Changing product ownership does not require an
index rebuild. The public RepoWise tools preserve the existing documentation
arguments and add `list_doc_files` for bounded, paginated file inventory.
