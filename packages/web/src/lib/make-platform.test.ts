import { afterEach, describe, expect, it, vi } from "vitest";
import { createServer, type Server } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getMakeOperations, getMakeSummary, MakeReportError } from "./make-platform";

let server: Server | undefined;
let directory: string | undefined;

async function serve(status: number, body: unknown, observed?: (method?: string, url?: string) => void) {
  directory = await mkdtemp(join(tmpdir(), "make-report-"));
  const socket = join(directory, "api.sock");
  vi.stubEnv("SOLEMD_MAKE_API_SOCKET", socket);
  server = createServer((req, res) => {
    observed?.(req.method, req.url);
    res.writeHead(status, { "content-type": "application/json" });
    res.end(JSON.stringify(body));
  });
  await new Promise<void>((resolve) => server!.listen(socket, resolve));
}

afterEach(async () => {
  if (server) await new Promise<void>((resolve) => server!.close(() => resolve()));
  if (directory) await rm(directory, { recursive: true, force: true });
  server = undefined;
  directory = undefined;
  vi.unstubAllEnvs();
});

describe("Make read-only reporting boundary", () => {
  it("makes a bounded GET against the operation report", async () => {
    const seen: [string | undefined, string | undefined][] = [];
    await serve(200, { schema_version: 1, catalog_revision: 7, items: [], next_cursor: null }, (method, url) => seen.push([method, url]));
    const result = await getMakeOperations({ status: "ambiguous", cursor: "opaque+cursor" });
    expect(result.items).toEqual([]);
    expect(seen).toEqual([["GET", "/v1/report/operations?limit=50&cursor=opaque%2Bcursor&status=ambiguous"]]);
  });

  it("retains a catalog conflict so the UI asks for a refresh", async () => {
    await serve(409, { error: "refresh" });
    await expect(getMakeOperations()).rejects.toMatchObject({ status: 409 });
  });

  it("does not turn an unavailable service into empty counts", async () => {
    vi.stubEnv("SOLEMD_MAKE_API_SOCKET", "/no/such/make-report.sock");
    await expect(getMakeSummary()).rejects.toBeInstanceOf(MakeReportError);
  });

  it("refuses incompatible and oversized reports", async () => {
    await serve(200, { schema_version: 2, padding: "x".repeat(300_000) });
    await expect(getMakeSummary()).rejects.toMatchObject({ status: 503 });
  });
});
