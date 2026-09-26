/**
 * Server-only, read-only transport to Make's reporting API over its private
 * Unix socket, and the contract each report must satisfy.
 *
 * Make owns the contract: make-api reads its versioned `*_v1` views and serves
 * these reports (SoleMD.Make `catalog/reporting.py`). Every response is checked
 * against the shapes below before the page sees it, so a report whose shape
 * drifted fails here, loudly, instead of rendering "undefined" in a sentence.
 * The shapes and the TypeScript types are one declaration.
 */
import { request, type IncomingMessage } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";

// --- Shape checking ---------------------------------------------------------

type Check<T> = (value: unknown, path: string) => T;
export type Shape<C> = C extends Check<infer T> ? T : never;

class ContractError extends Error {}

const fail = (path: string, expected: string): never => {
  throw new ContractError(`${path}: expected ${expected}`);
};

const num: Check<number> = (v, p) =>
  typeof v === "number" && Number.isFinite(v) ? v : fail(p, "a number");
const str: Check<string> = (v, p) => (typeof v === "string" ? v : fail(p, "a string"));
const bool: Check<boolean> = (v, p) => (typeof v === "boolean" ? v : fail(p, "a boolean"));
const nullable =
  <T>(inner: Check<T>): Check<T | null> =>
  (v, p) =>
    v === null ? null : inner(v, p);
const list =
  <T>(inner: Check<T>): Check<T[]> =>
  (v, p) =>
    Array.isArray(v) ? v.map((item, i) => inner(item, `${p}[${i}]`)) : fail(p, "a list");
const counts: Check<Record<string, number>> = (v, p) => {
  if (typeof v !== "object" || v === null || Array.isArray(v)) return fail(p, "an object");
  return Object.fromEntries(Object.entries(v).map(([k, n]) => [k, num(n, `${p}.${k}`)]));
};
const obj =
  <S extends Record<string, Check<unknown>>>(fields: S): Check<{ [K in keyof S]: Shape<S[K]> }> =>
  (v, p) => {
    if (typeof v !== "object" || v === null || Array.isArray(v)) return fail(p, "an object");
    const record = v as Record<string, unknown>;
    return Object.fromEntries(
      Object.entries(fields).map(([k, check]) => [k, check(record[k], `${p}.${k}`)]),
    ) as { [K in keyof S]: Shape<S[K]> };
  };
const report = <S extends Record<string, Check<unknown>>>(fields: S) =>
  obj({ schema_version: num, catalog_revision: num, ...fields });

// --- The reports ------------------------------------------------------------

const backup = obj({
  stream: str,
  healthy: bool,
  state: str,
  last_success_at: nullable(str),
  bytes_total: nullable(num),
});

const summaryShape = report({
  as_of: str,
  media: counts,
  evidence: counts,
  meetings: obj({ recordings: num, summarized: num, awaiting_summary: num }),
  operations: counts,
  spend_24h_micros: num,
  objects: obj({ count: num, bytes: num, unavailable: num }),
  backup_health: list(backup),
});

const operationShape = obj({
  id: str,
  kind: str,
  status: str,
  created_at: str,
  updated_at: str,
  attempt_no: nullable(num),
  provider: nullable(str),
  error_code: nullable(str),
  error_type: nullable(str),
});
const operationsShape = report({ items: list(operationShape), next_cursor: nullable(str) });

const spendShape = report({
  days: num,
  time_zone: str,
  total_micros: num,
  daily: list(obj({ day: str, micros: num, attempts: num })),
  by_kind: list(obj({ kind: str, micros: num, attempts: num })),
  by_provider: list(obj({ provider: str, micros: num, attempts: num })),
});

const mixShape = report({
  days: num,
  by_kind: list(obj({ kind: str, total: num, statuses: counts })),
  failures: list(obj({ kind: str, error_code: str, count: num, last_at: nullable(str) })),
  durations: list(
    obj({ kind: str, count: num, median_seconds: num, p90_seconds: num, max_seconds: num }),
  ),
});

const buildShape = obj({
  document_id: str,
  locator: str,
  project: nullable(str),
  project_status: nullable(str),
  topic: nullable(str),
  topic_title: nullable(str),
  engine: str,
  status: str,
  built_at: str,
  finished_at: nullable(str),
  error_code: nullable(str),
  builds: num,
  outputs: list(obj({ name: str, bytes: num })),
});
const buildsShape = report({
  items: list(buildShape),
  next_cursor: nullable(str),
  topics: list(
    obj({
      topic: nullable(str),
      title: nullable(str),
      documents: num,
      failed: num,
      latest_built_at: str,
    }),
  ),
});

const runShape = obj({
  run_slug: str,
  question: str,
  created_at: str,
  updated_at: str,
  stage: str,
  searches: num,
  references: num,
  abstracts_screened: num,
  abstracts_included: num,
  abstracts_uncertain: num,
  fulltexts_screened: num,
  fulltexts_included: num,
  fulltexts_uncertain: num,
});
const evidenceShape = report({
  runs: obj({ total: num, recent: list(runShape) }),
  claims: obj({
    total: num,
    by_status: counts,
    quotes: num,
    quotes_reviewed: num,
    quotes_independent: num,
    review_queue: obj({
      claims: num,
      oldest: list(
        obj({ slug: str, status: str, quotes: num, quotes_independent: num, committed_at: str }),
      ),
    }),
  }),
  quotas: list(obj({ provider: str, day: str, daily_limit: num, spent: num, reserved: num })),
});

const byDay = list(obj({ day: str, files: num, bytes: num }));
const retentionShape = obj({
  grace_days: num,
  expiring: obj({ files: num, bytes: num, by_day: byDay }),
  reclaimed: obj({ days: num, files: num, bytes: num, by_day: byDay }),
});
const mediaShape = report({
  cards: counts,
  queue: obj({
    pending: num,
    under_a_day: num,
    under_a_week: num,
    older: num,
    oldest_seconds: nullable(num),
  }),
  owners: list(
    obj({
      owner: str,
      owner_kind: str,
      owner_title: str,
      project: nullable(str),
      cards: num,
      pending: num,
      rejected: num,
      blocked: num,
      oldest_pending_at: nullable(str),
    }),
  ),
  retention: retentionShape,
});

const unitShape = obj({
  unit: str,
  active_state: str,
  result: str,
  failed: bool,
  last_started_at: nullable(str),
  last_finished_at: nullable(str),
});
const meetingsShape = report({
  recordings: obj({
    total: num,
    summarized: num,
    awaiting_summary: num,
    in_pipeline: num,
    released: num,
  }),
  last_admitted_at: nullable(str),
  stages: list(obj({ stage: str, statuses: counts })),
  stuck: list(obj({ recording: str, stage: str, status: str, updated_at: str })),
  stuck_after_seconds: num,
  poll: nullable(unitShape),
});

const storageShape = report({
  tiers: list(obj({ tier: str, files: num, bytes: num })),
  classes: list(
    obj({
      tier: str,
      file_class: str,
      owner: nullable(str),
      files: num,
      bytes: num,
      kept: num,
      kept_bytes: num,
      expiring: num,
      expiring_bytes: num,
      gone: num,
    }),
  ),
  ledger: (v, p) => {
    if (typeof v !== "object" || v === null) return fail(p, "an object");
    return Object.fromEntries(
      Object.entries(v).map(([k, note]) => [k, nullable(str)(note, `${p}.${k}`)]),
    ) as Record<string, string | null>;
  },
  retention: retentionShape,
  last_retention: nullable(
    obj({
      status: str,
      started_at: str,
      finished_at: nullable(str),
      deleted: nullable(num),
      deleted_bytes: nullable(num),
    }),
  ),
  unregistered: nullable(obj({ files: num, examples: list(str) })),
});

const severity: Check<"ok" | "warn" | "fail"> = (v, p) =>
  v === "ok" || v === "warn" || v === "fail" ? v : fail(p, "ok, warn or fail");
const doctorShape = report({
  findings: list(
    obj({ area: str, check: str, severity, detail: str, action: nullable(str) }),
  ),
  counts: obj({ ok: num, warn: num, fail: num }),
});

export type MakeSummary = Shape<typeof summaryShape>;
export type MakeOperation = Shape<typeof operationShape>;
export type MakeOperations = Shape<typeof operationsShape>;
export type MakeSpend = Shape<typeof spendShape>;
export type MakeOperationMix = Shape<typeof mixShape>;
export type MakeBuild = Shape<typeof buildShape>;
export type MakeBuilds = Shape<typeof buildsShape>;
export type MakeEvidence = Shape<typeof evidenceShape>;
export type MakeMedia = Shape<typeof mediaShape>;
export type MakeRetention = Shape<typeof retentionShape>;
export type MakeMeetings = Shape<typeof meetingsShape>;
export type MakeStorage = Shape<typeof storageShape>;
export type MakeDoctor = Shape<typeof doctorShape>;
export type MakeFinding = MakeDoctor["findings"][number];

/** Every report's shape by path, for the contract test against captured and live responses. */
export const MAKE_REPORT_SHAPES: Record<string, Check<unknown>> = {
  "/v1/report/summary": summaryShape,
  "/v1/report/operations": operationsShape,
  "/v1/report/spend": spendShape,
  "/v1/report/operations/mix": mixShape,
  "/v1/report/builds": buildsShape,
  "/v1/report/evidence": evidenceShape,
  "/v1/report/media": mediaShape,
  "/v1/report/meetings": meetingsShape,
  "/v1/report/storage": storageShape,
  "/v1/report/doctor": doctorShape,
};

// --- Transport ----------------------------------------------------------------

export class MakeReportError extends Error {
  constructor(public readonly status: number) {
    super(status === 409 ? "The catalog changed. Refresh this list." : "Make reporting is unavailable.");
  }
}

const MAX_BODY_BYTES = 256 * 1024;
const DEADLINE_MS = 2500;

export function makeApiSocket(): string {
  return process.env.SOLEMD_MAKE_API_SOCKET || join(homedir(), ".local/share/solemd-make/run/api.sock");
}

function readReport<T>(path: string, shape: Check<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const req = request({ socketPath: makeApiSocket(), path, method: "GET", headers: { accept: "application/json" } });
    const deadline = setTimeout(() => req.destroy(new MakeReportError(503)), DEADLINE_MS);
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
          const value = JSON.parse(Buffer.concat(chunks).toString("utf8")) as { schema_version?: unknown };
          if (value?.schema_version !== 1) throw new ContractError("unsupported schema_version");
          resolve(shape(value, path));
        } catch (error) {
          // A drifted contract is a Make/RepoWise mismatch to fix, not an outage
          // to hide: say which field broke in the server log.
          if (error instanceof ContractError) console.error(`Make report ${path} broke its contract: ${error.message}`);
          reject(new MakeReportError(503));
        }
      });
    });
    req.end();
  });
}

export const OPERATION_STATUSES = [
  "queued", "running", "dispatch_started", "ambiguous", "completed", "failed", "cancelled",
] as const;

export const getMakeSummary = () => readReport("/v1/report/summary", summaryShape);
export const getMakeEvidence = () => readReport("/v1/report/evidence", evidenceShape);
export const getMakeMedia = () => readReport("/v1/report/media", mediaShape);
export const getMakeMeetings = () => readReport("/v1/report/meetings", meetingsShape);
export const getMakeStorage = () => readReport("/v1/report/storage", storageShape);
export const getMakeDoctor = () => readReport("/v1/report/doctor", doctorShape);

export function getMakeSpend(days = 30, timeZone = "America/Los_Angeles") {
  const params = new URLSearchParams({ days: String(days), tz: timeZone });
  return readReport(`/v1/report/spend?${params}`, spendShape);
}

export function getMakeOperationMix(days = 7) {
  return readReport(`/v1/report/operations/mix?days=${days}`, mixShape);
}

export function getMakeBuilds(options: { cursor?: string; limit?: number } = {}) {
  const params = new URLSearchParams({ limit: String(options.limit ?? 50) });
  if (options.cursor) params.set("cursor", options.cursor);
  return readReport(`/v1/report/builds?${params}`, buildsShape);
}

export function getMakeOperations(options: { cursor?: string; status?: string } = {}) {
  const params = new URLSearchParams({ limit: "50" });
  if (options.cursor) params.set("cursor", options.cursor);
  if (options.status) params.set("status", options.status);
  return readReport(`/v1/report/operations?${params}`, operationsShape);
}

/** Make's catalog revision stream (server-sent events), for the page's live refresh. */
export function openMakeEvents(signal: AbortSignal): Promise<IncomingMessage> {
  return new Promise((resolve, reject) => {
    const req = request({ socketPath: makeApiSocket(), path: "/v1/events", method: "GET", headers: { accept: "text/event-stream" }, signal });
    req.on("error", () => reject(new MakeReportError(503)));
    req.on("response", (res) => {
      if (res.statusCode !== 200) { res.resume(); reject(new MakeReportError(res.statusCode ?? 503)); return; }
      resolve(res);
    });
    req.end();
  });
}

// --- Presentation, in one place -------------------------------------------------

const OPERATION_LABELS: Record<string, string> = {
  "build.slides": "Slide deck build",
  "build.typst": "PDF build",
  "build.pandoc": "Word build",
  "build.html": "Page build",
  "build.adopt": "Build adoption",
  "media.generate": "Image generation",
  "media.edit": "Image edit",
  "media.video": "Video generation",
  "media.relight": "Night relight",
  "media.generation.batch": "Generation batch",
  "media.retention": "Retention pass",
  "media.authoring": "Image authoring",
  "meeting.validate": "Recording check",
  "meeting.normalize": "Audio normalization",
  "meeting.transcribe": "Transcription",
  "meeting.diarize": "Speaker labels",
  "meeting.finalize": "Transcript assembly",
};

/** What a person calls an operation kind. Make sends the machine name. */
export function operationLabel(kind: string): string {
  const known = OPERATION_LABELS[kind];
  if (known) return known;
  const words = kind.split(".").slice(1).join(" ").replace(/[-_]/g, " ").trim() || kind;
  const area = kind.split(".")[0];
  const text = `${words.charAt(0).toUpperCase()}${words.slice(1)}`;
  return area === "media" || area === "build" || area === "meeting" ? text : `${text} (${area})`;
}

export const microsToUsd = (micros: number) => micros / 1_000_000;
