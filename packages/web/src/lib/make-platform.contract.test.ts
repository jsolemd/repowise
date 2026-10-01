import { afterEach, describe, expect, it, vi } from "vitest";
import { createServer, request, type Server } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import captured from "./make-platform.contract.json";
import {
  getMakeBuilds,
  getMakeDoctor,
  getMakeEvidence,
  getMakeMedia,
  getMakeMeetings,
  getMakeOperationMix,
  getMakeSpend,
  getMakeStorage,
  operationLabel,
} from "./make-platform";

/**
 * The dashboard's side of Make's report contract. The captured responses came
 * from make-api over a copy of the production catalog; set MAKE_CONTRACT_SOCKET
 * to a running make-api's socket to check the live service the same way.
 */
const reports = captured.reports as Record<string, { schema_version: number }>;

/**
 * Every report through the getter the dashboard calls, so a check covers the
 * schema version each report declares as well as its shape.
 */
const GETTERS: Record<string, () => Promise<unknown>> = {
  "/v1/report/spend": () => getMakeSpend(),
  "/v1/report/operations/mix": () => getMakeOperationMix(),
  "/v1/report/builds": () => getMakeBuilds(),
  "/v1/report/evidence": getMakeEvidence,
  "/v1/report/media": getMakeMedia,
  "/v1/report/meetings": getMakeMeetings,
  "/v1/report/storage": getMakeStorage,
  "/v1/report/doctor": getMakeDoctor,
};
const PATHS = Object.keys(GETTERS);
const LIVE_SOCKET = process.env.MAKE_CONTRACT_SOCKET;

let server: Server | undefined;
let directory: string | undefined;

async function serve(routes: Record<string, unknown>) {
  directory = await mkdtemp(join(tmpdir(), "make-contract-"));
  const socket = join(directory, "api.sock");
  vi.stubEnv("SOLEMD_MAKE_API_SOCKET", socket);
  server = createServer((req, res) => {
    const body = routes[(req.url ?? "").split("?")[0]!];
    res.writeHead(body === undefined ? 404 : 200, { "content-type": "application/json" });
    res.end(JSON.stringify(body ?? { error: "missing" }));
  });
  await new Promise<void>((resolve) => server!.listen(socket, resolve));
}

function liveBody(path: string): Promise<{ schema_version: number }> {
  return new Promise((resolve, reject) => {
    const req = request({ socketPath: LIVE_SOCKET, path, method: "GET" }, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (c: Buffer) => chunks.push(c));
      res.on("end", () => (res.statusCode === 200 ? resolve(JSON.parse(Buffer.concat(chunks).toString())) : reject(new Error(`${res.statusCode}`))));
    });
    req.on("error", reject);
    req.end();
  });
}

/** The same report under each schema version next to the one it carries. */
const otherVersions = (body: { schema_version: number }) =>
  [body.schema_version - 1, body.schema_version + 1].map((v) => ({ ...body, schema_version: v }));

afterEach(async () => {
  if (server) await new Promise<void>((resolve) => server!.close(() => resolve()));
  if (directory) await rm(directory, { recursive: true, force: true });
  server = undefined;
  directory = undefined;
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("Make report contract", () => {
  it("has a captured response for every report the dashboard reads", () => {
    expect(Object.keys(reports).sort()).toEqual([...PATHS].sort());
  });

  it.each(PATHS)("accepts the captured %s through its getter", async (path) => {
    await serve({ [path]: reports[path] });
    await expect(GETTERS[path]!()).resolves.toMatchObject({ schema_version: reports[path]!.schema_version });
  });

  it.each(PATHS)("refuses the captured %s under any other schema version", async (path) => {
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const routes: Record<string, unknown> = {};
    await serve(routes);
    for (const body of otherVersions(reports[path]!)) {
      routes[path] = body;
      await expect(GETTERS[path]!()).rejects.toMatchObject({ status: 503 });
    }
    expect(logged.mock.calls.map((c) => String(c[0]))).toEqual([
      expect.stringContaining("schema_version"),
      expect.stringContaining("schema_version"),
    ]);
  });

  it("refuses a count that arrives as a string instead of rendering it", async () => {
    const drifted = structuredClone(reports["/v1/report/media"]) as unknown as { queue: { pending: unknown } };
    drifted.queue.pending = "4";
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    await serve({ "/v1/report/media": drifted });
    await expect(getMakeMedia()).rejects.toMatchObject({ status: 503 });
    expect(logged.mock.calls[0]![0]).toContain("queue.pending: expected a number");
  });

  it("refuses a retired field's absence being read as a value", async () => {
    const drifted = structuredClone(reports["/v1/report/doctor"]) as unknown as { findings: { severity: string }[] };
    drifted.findings[0]!.severity = "critical";
    vi.spyOn(console, "error").mockImplementation(() => {});
    await serve({ "/v1/report/doctor": drifted });
    await expect(getMakeDoctor()).rejects.toMatchObject({ status: 503 });
  });

  it.each(["invalid", "2026-02-30"])("refuses a day that is not on the calendar (%s)", async (day) => {
    const drifted = structuredClone(reports["/v1/report/storage"]) as unknown as {
      retention: { expiring: { by_day: { day: string }[] } };
    };
    drifted.retention.expiring.by_day[0]!.day = day;
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    await serve({ "/v1/report/storage": drifted });
    await expect(getMakeStorage()).rejects.toMatchObject({ status: 503 });
    expect(logged.mock.calls[0]![0]).toContain("retention.expiring.by_day[0].day: expected a calendar date");
  });

  it.runIf(Boolean(LIVE_SOCKET)).each(PATHS)(
    "the live make-api honours %s through its getter",
    async (path) => {
      vi.stubEnv("SOLEMD_MAKE_API_SOCKET", LIVE_SOCKET!);
      await expect(GETTERS[path]!()).resolves.toBeTruthy();
    },
    15_000,
  );

  it.runIf(Boolean(LIVE_SOCKET)).each(PATHS)(
    "the live %s is refused under any other schema version",
    async (path) => {
      const body = await liveBody(path);
      vi.spyOn(console, "error").mockImplementation(() => {});
      const routes: Record<string, unknown> = {};
      await serve(routes);
      for (const other of otherVersions(body)) {
        routes[path] = other;
        await expect(GETTERS[path]!()).rejects.toMatchObject({ status: 503 });
      }
    },
    15_000,
  );
});

describe("operation labels", () => {
  it("names known kinds and reads unknown ones", () => {
    expect(operationLabel("build.slides")).toBe("Slide deck build");
    expect(operationLabel("media.reference.crop")).toBe("Reference crop");
    expect(operationLabel("evidence.identity_repair")).toBe("Identity repair (evidence)");
  });
});
