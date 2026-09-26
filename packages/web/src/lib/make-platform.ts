/** Server-only, read-only Make reporting transport over its private Unix socket. */
import { request } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";

export interface MakeOperation {
  id: string;
  kind: string;
  status: string;
  created_at: string;
  updated_at: string;
  attempt_no: number | null;
  provider: string | null;
  error_code: string | null;
  error_type: string | null;
}

export interface MakeOperations {
  schema_version: 1;
  catalog_revision: number;
  items: MakeOperation[];
  next_cursor: string | null;
}

export interface MakeSummary {
  schema_version: 1;
  catalog_revision: number;
  mode: string;
  as_of: string;
  media: { selected: number; pending: number; accepted: number; rejected: number;
    revised: number; stale: number; orphaned: number; legacy_approvals: number;
    open_requests: number; ambiguous_dispatches: number; interrupted_dispatches: number;
    unresolved_verdicts: number; };
  evidence: Record<string, number>;
  meetings: { recordings: number; summarized: number; awaiting_summary: number };
  operations: Record<string, number>;
  costs_24h: { currency: string; amount_micros: number }[];
  objects: { count: number; bytes: number; unavailable: number };
  releases: number;
  backup_health: {
    stream: "database";
    healthy: boolean;
    state: "ok" | "missing" | "stale" | "unavailable";
    last_success_at: string | null;
    bytes_total: number | null;
  }[];
}

export class MakeReportError extends Error {
  constructor(public readonly status: number) {
    super(status === 409 ? "The catalog changed. Refresh this list." : "Make reporting is unavailable.");
  }
}

const MAX_BODY_BYTES = 256 * 1024;

function readReport<T extends { schema_version: 1 }>(path: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const socketPath = process.env.SOLEMD_MAKE_API_SOCKET || join(homedir(), ".local/share/solemd-make/run/api.sock");
    const req = request({ socketPath, path, method: "GET", headers: { accept: "application/json" } });
    const deadline = setTimeout(() => req.destroy(new MakeReportError(503)), 2500);
    req.on("error", () => { clearTimeout(deadline); reject(new MakeReportError(503)); });
    req.on("response", (res) => {
      let bytes = 0;
      const chunks: Buffer[] = [];
      res.on("data", (chunk: Buffer) => {
        bytes += chunk.length;
        if (bytes > MAX_BODY_BYTES) req.destroy(new MakeReportError(503));
        else chunks.push(chunk);
      });
      res.on("error", () => { clearTimeout(deadline); reject(new MakeReportError(503)); });
      res.on("end", () => {
        clearTimeout(deadline);
        if (res.statusCode !== 200) { reject(new MakeReportError(res.statusCode ?? 503)); return; }
        try {
          const value = JSON.parse(Buffer.concat(chunks).toString("utf8")) as T;
          if (value?.schema_version !== 1) throw new Error("unsupported report");
          resolve(value);
        } catch { reject(new MakeReportError(503)); }
      });
    });
    req.end();
  });
}

export function getMakeSummary() {
  return readReport<MakeSummary>("/v1/report/summary");
}

export function getMakeOperations(options: { cursor?: string; status?: string } = {}) {
  const params = new URLSearchParams({ limit: "50" });
  if (options.cursor) params.set("cursor", options.cursor);
  if (options.status) params.set("status", options.status);
  return readReport<MakeOperations>(`/v1/report/operations?${params}`);
}
