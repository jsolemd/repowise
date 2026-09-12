/** RepoWise documentation inventory, browsing, search, and management. */

import { apiGet, apiPost } from "@repowise-dev/api-client";
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

export interface DocFilesPage {
  library_id: string;
  library: {
    name: string;
    repo: string | null;
    branch: string | null;
    source_type: string;
    status: string;
    indexed_ref: string | null;
    indexed_at: string | null;
  };
  files: { file_path: string; chunk_count: number; indexed_at: string | null }[];
  pagination: { total: number; offset: number; limit: number; returned: number; has_more: boolean; next_offset: number | null };
}

export interface DocContent {
  content: string;
  path: string;
  library_id: string;
  truncated: boolean;
}

export interface DocSearchResults {
  results: { chunk_id: string; file_path: string; title: string; preview: string; source_url: string }[];
  warnings?: string[];
}

export async function callDocsTool<T>(name: string, args: Record<string, unknown>): Promise<T> {
  const result = await apiPost<{ status: string; payload: T & { error?: string } }>(
    `/api/docs-libraries/tools/${encodeURIComponent(name)}`, { ...args, output: "json" },
  );
  if (result.status === "error") throw new Error(result.payload.error || "Documentation request failed.");
  return result.payload;
}

export function documentsHref(library: string, path?: string): string {
  const params = new URLSearchParams({ library });
  if (path) params.set("path", path);
  return `/docs-libraries/documents?${params}`;
}
