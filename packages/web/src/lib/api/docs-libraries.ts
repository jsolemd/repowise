/**
 * The documentation service's library inventory, read through this server's
 * `/api/docs-libraries` proxy.
 *
 * Typed here rather than in `@repowise-dev/api-client` because the payload is
 * not this server's: it belongs to the docs service, and only the dashboard
 * reads it. Keeping the shape next to its one consumer means the shared client
 * package does not acquire a dependency on a service it never talks to.
 *
 * Every string that the docs service can leave unset is `| null` rather than
 * optional. That distinction is the whole point of the page: a library whose
 * freshness was never established reads as "unknown", and an optional field
 * would let that state disappear into `undefined` and render as blank.
 */

import { apiGet } from "@repowise-dev/api-client";
import "./client";

/** What the indexer last did with a library. */
export type DocsLibraryStatus = "ready" | "pending" | "indexing" | "error" | (string & {});

/** What the freshness checker last concluded. `null` means it never ran. */
export type DocsFreshnessState = "fresh" | "stale" | "unknown" | (string & {});

export interface DocsLibrary {
  library_id: string;
  name: string;
  repo: string | null;
  source_type: string | null;
  status: DocsLibraryStatus;
  branch: string | null;
  docs_path: string | null;
  source_subpath: string | null;
  file_count: number;
  chunk_count: number;
  /** ISO-8601. */
  indexed_at: string | null;
  graph_synced_at: string | null;
  last_freshness_state: DocsFreshnessState | null;
  graph_sync_error: string | null;
  error_message: string | null;
  priority: number | null;
  /** The commit this library's chunks were built from. */
  current_sha: string | null;
  freshness_checked_at: string | null;
  next_freshness_check_at: string | null;
  /** The commit the remote was on at the last check. Ahead of `current_sha`
   *  means the indexed copy is behind the source. */
  last_remote_sha: string | null;
  last_freshness_error: string | null;
}

export interface DocsLibrariesSummary {
  status: string;
  total_libraries: number;
  ready_libraries: number;
  pending_libraries: number;
  indexing_libraries: number;
  error_libraries: number;
  metadata_gap_libraries: number;
  graph_gap_libraries: number;
}

export interface DocsInventory {
  total_files: number;
  total_chunks: number;
  git_libraries: number;
  snapshot_libraries: number;
}

export interface DocsSchedulerState {
  running: boolean;
  last_started_at: string | null;
  last_completed_at: string | null;
  summary: Record<string, unknown>;
}

export interface DocsWorkerState {
  running: boolean;
  id: string | null;
}

export interface DocsJob {
  id: string;
  library_id: string;
  job_type: string;
  priority: number | null;
  status: string;
  worker_id: string | null;
  files_processed: number | null;
  files_total: number | null;
  error_message: string | null;
  created_at: string | null;
  claimed_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  heartbeat_at: string | null;
}

export interface DocsJobs {
  pending: DocsJob[];
  running: DocsJob[];
  recent_failed: DocsJob[];
}

export interface DocsLibrariesResponse {
  summary: DocsLibrariesSummary;
  inventory: DocsInventory;
  libraries: DocsLibrary[];
  scheduler: DocsSchedulerState;
  /** Library id → consecutive failures reaching its remote. */
  unreachable_libraries: Record<string, number>;
  worker: DocsWorkerState;
  /** Subsystem → status word, from the docs service's own readiness block. */
  runtime: { worker?: string; scheduler?: string; recovery?: string };
  dependencies: { qdrant?: string; tei?: string; database?: string };
  health_status: string;
  jobs: DocsJobs;
  generated_at: string;
}

export const getDocsLibraries = () =>
  apiGet<DocsLibrariesResponse>("/api/docs-libraries");
