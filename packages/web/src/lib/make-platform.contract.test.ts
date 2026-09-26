import { afterEach, describe, expect, it, vi } from "vitest";
import { createServer, request, type Server } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import captured from "./make-platform.contract.json";
import { getMakeDoctor, getMakeSummary, MAKE_REPORT_SHAPES, operationLabel } from "./make-platform";

/**
 * The page's side of Make's report contract. The captured responses came from
 * make-api over a copy of the production catalog; set MAKE_CONTRACT_SOCKET to a
 * running make-api's socket to check the live service the same way.
 */
const reports = captured.reports as Record<string, unknown>;

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

afterEach(async () => {
  if (server) await new Promise<void>((resolve) => server!.close(() => resolve()));
  if (directory) await rm(directory, { recursive: true, force: true });
  server = undefined;
  directory = undefined;
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("Make report contract", () => {
  it("has a captured response for every report the page reads", () => {
    expect(Object.keys(reports).sort()).toEqual(Object.keys(MAKE_REPORT_SHAPES).sort());
  });

  it.each(Object.keys(MAKE_REPORT_SHAPES))("accepts the captured %s", (path) => {
    expect(() => MAKE_REPORT_SHAPES[path]!(reports[path], path)).not.toThrow();
  });

  it("refuses a count that arrives as a string instead of rendering it", async () => {
    const drifted = structuredClone(reports["/v1/report/summary"]) as { objects: { bytes: unknown } };
    drifted.objects.bytes = "25329733882";
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    await serve({ "/v1/report/summary": drifted });
    await expect(getMakeSummary()).rejects.toMatchObject({ status: 503 });
    expect(logged.mock.calls[0]![0]).toContain("objects.bytes: expected a number");
  });

  it("refuses a retired field's absence being read as a value", async () => {
    const drifted = structuredClone(reports["/v1/report/doctor"]) as { findings: { severity: string }[] };
    drifted.findings[0]!.severity = "critical";
    vi.spyOn(console, "error").mockImplementation(() => {});
    await serve({ "/v1/report/doctor": drifted });
    await expect(getMakeDoctor()).rejects.toMatchObject({ status: 503 });
  });

  it.runIf(Boolean(process.env.MAKE_CONTRACT_SOCKET)).each(Object.keys(MAKE_REPORT_SHAPES))(
    "the live make-api honours %s",
    async (path) => {
      const body = await new Promise<string>((resolve, reject) => {
        const req = request({ socketPath: process.env.MAKE_CONTRACT_SOCKET, path, method: "GET" }, (res) => {
          const chunks: Buffer[] = [];
          res.on("data", (c: Buffer) => chunks.push(c));
          res.on("end", () => (res.statusCode === 200 ? resolve(Buffer.concat(chunks).toString()) : reject(new Error(`${res.statusCode}`))));
        });
        req.on("error", reject);
        req.end();
      });
      expect(() => MAKE_REPORT_SHAPES[path]!(JSON.parse(body), path)).not.toThrow();
    },
  );
});

describe("operation labels", () => {
  it("names known kinds and reads unknown ones", () => {
    expect(operationLabel("build.slides")).toBe("Slide deck build");
    expect(operationLabel("media.reference.crop")).toBe("Reference crop");
    expect(operationLabel("evidence.identity_repair")).toBe("Identity repair (evidence)");
  });
});
