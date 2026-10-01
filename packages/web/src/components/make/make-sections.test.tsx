// @vitest-environment jsdom

import { render, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import captured from "@/lib/make-platform.contract.json";
import type { MakeBuilds, MakeDoctor, MakeEvidence, MakeMedia, MakeMeetings, MakeSpend, MakeStorage } from "@/lib/make-platform";
import {
  BuildsPanel,
  DoctorStrip,
  EvidencePanel,
  ImagesPanel,
  MeetingsPanel,
  NeedsYou,
  SpendPanel,
  StoragePanel,
  compactAge,
  stageTrouble,
} from "./make-sections";

// Queries are scoped to each render's container: there is no global cleanup.

const reports = captured.reports as Record<string, unknown>;
const capture = <T,>(path: string) => structuredClone(reports[path]) as T;

const doctor = (findings: MakeDoctor["findings"]): MakeDoctor => ({
  schema_version: 1,
  catalog_revision: 1,
  findings,
  counts: {
    ok: findings.filter((f) => f.severity === "ok").length,
    warn: findings.filter((f) => f.severity === "warn").length,
    fail: findings.filter((f) => f.severity === "fail").length,
  },
});

describe("Doctor", () => {
  const findings: MakeDoctor["findings"] = [
    { area: "catalog", check: "Make catalog", severity: "ok", detail: "reachable", action: null },
    { area: "media", check: "Stale renders", severity: "warn", detail: "2 renders derive from replaced bytes", action: "reselect them" },
    { area: "storage", check: "Lab bundles", severity: "fail", detail: "1 file is missing", action: "run `make catalog restore`" },
  ];

  it("draws one cell per check, failing first, then warning, then passing", () => {
    const { container } = render(<DoctorStrip doctor={doctor(findings)} />);
    const cells = [...container.querySelectorAll("[data-severity]")].map((c) => c.getAttribute("data-severity"));
    expect(cells).toEqual(["fail", "warn", "ok"]);
  });

  it("opens every open finding's check, detail and action from the strip, without a hover", () => {
    const { container } = render(<DoctorStrip doctor={doctor(findings)} />);
    const details = container.querySelector("details")!;
    expect(details.querySelector("summary")!.textContent).toMatch(/Doctor.*fail\s*1.*warn\s*1.*ok\s*1/);
    const listed = within(details).getAllByRole("listitem").map((li) => li.textContent);
    expect(listed).toEqual([
      expect.stringMatching(/Lab bundles.*1 file is missing.*Run make catalog restore\./),
      expect.stringMatching(/Stale renders.*2 renders derive from replaced bytes.*Reselect them\./),
    ]);
  });

  it("is a plain strip when every check passes", () => {
    const { container } = render(<DoctorStrip doctor={doctor([findings[0]!])} />);
    expect(container.querySelector("details")).toBeNull();
    expect(container.querySelectorAll("[data-severity]")).toHaveLength(1);
  });

  it("says when the doctor did not answer", () => {
    const { container } = render(<DoctorStrip doctor={null} />);
    expect(within(container).getByRole("status").textContent).toBe("Doctor · no answer");
  });
});

describe("Needs you", () => {
  const media = capture<MakeMedia>("/v1/report/media");
  const meetings = capture<MakeMeetings>("/v1/report/meetings");
  const evidence = capture<MakeEvidence>("/v1/report/evidence");

  it("puts each count waiting on Jon in its own tile, the images tile opening the review library", () => {
    const { container } = render(<NeedsYou media={media} meetings={meetings} evidence={evidence} libraryUrl="https://library.test/make" />);
    const link = within(container).getByRole("link");
    expect(link.getAttribute("href")).toBe("https://library.test/make");
    expect(link.textContent).toMatch(/Images to review\s*4/);
    expect(container.textContent).toMatch(/Notes to file\s*3/);
    expect(container.textContent).toMatch(/Copies to fetch\s*96\s*of 142/);
    expect(container.textContent).toMatch(/Stale concepts\s*0/);
    expect(container.textContent).toMatch(/Docs to rebuild\s*0/);
  });

  it("shows how long the oldest image has waited only while images wait", () => {
    const waiting = render(<NeedsYou media={media} meetings={meetings} evidence={evidence} libraryUrl="#" />);
    expect(within(waiting.container).getByRole("link").textContent).toContain("oldest waiting 23h");
    const idle = structuredClone(media);
    idle.queue = { ...idle.queue, pending: 0, under_a_day: 0 };
    const empty = render(<NeedsYou media={idle} meetings={meetings} evidence={evidence} libraryUrl="#" />);
    expect(within(empty.container).getByRole("link").textContent).not.toContain("oldest");
  });

  it("marks a report that did not answer, rather than drawing a zero", () => {
    const { container } = render(<NeedsYou media={media} meetings={null} evidence={evidence} libraryUrl="#" />);
    expect(container.textContent).toMatch(/Notes to file\s*—\s*no answer/);
  });
});

describe("Evidence", () => {
  it("counts only the open questions as open", () => {
    const evidence = capture<MakeEvidence>("/v1/report/evidence");
    evidence.questions = { total: 3, by_state: { open: 1, answered: 1, withdrawn: 1 } };
    const { container } = render(<EvidencePanel evidence={evidence} />);
    expect(container.textContent).toMatch(/1\s*open question(?!s)/);
    expect(container.textContent).not.toMatch(/3\s*open/);
  });
});

describe("Builds", () => {
  it("prints a group's failed count", () => {
    const builds = capture<MakeBuilds>("/v1/report/builds");
    builds.groups = [{ group: "lectures", documents: 5, failed: 2, latest_built_at: "2026-10-01T00:00:00Z" }];
    const { container } = render(<BuildsPanel builds={builds} />);
    expect(container.textContent).toContain("2 failed");
    expect(within(container).getByRole("img", { name: "other documents 3, failed 2" })).toBeTruthy();
  });

  it("never calls a document built when Make counts only its failures", () => {
    // A queued, running or cancelled latest build is among the documents that did not fail.
    const builds = capture<MakeBuilds>("/v1/report/builds");
    builds.items = [{ ...builds.items[0]!, status: "running" }];
    builds.groups = [{ group: "test", documents: 1, failed: 0, latest_built_at: "2026-10-01T00:00:00Z" }];
    const { container } = render(<BuildsPanel builds={builds} />);
    expect(container.innerHTML).not.toMatch(/\bbuilt\b/);
  });
});

describe("Images", () => {
  it("draws every verdict Make reports, revised ones included, and names a state it does not know", () => {
    const media = capture<MakeMedia>("/v1/report/media");
    media.cards = { accept: 4, revise: 2, pending: 1, retired: 3 };
    const { container } = render(<ImagesPanel media={media} />);
    expect(within(container).getByRole("img", { name: "accepted 4, revised 2, rejected 0, pending 1, other 3" })).toBeTruthy();
    expect(container.textContent).toMatch(/10\s*renders/);
  });
});

describe("Meetings", () => {
  const meetings = () => {
    const m = capture<MakeMeetings>("/v1/report/meetings");
    m.recordings = { total: 4, summarized: 1, awaiting_summary: 1, in_pipeline: 2, released: 1 };
    return m;
  };

  it("keeps the report's three recording shares and counts stuck stage runs apart", () => {
    // Two stuck stages of one recording: stage runs cannot say how many recordings they hold back.
    const m = meetings();
    m.stages = [];
    m.stuck = ["transcribe", "diarize"].map((stage) => ({ recording: "same-id", stage, status: "running", updated_at: "2026-10-01T00:00:00Z" }));
    const { container } = render(<MeetingsPanel meetings={m} />);
    expect(within(container).getByRole("img", { name: "filed 1, to file 1, in pipeline 2" })).toBeTruthy();
    expect(container.querySelector("[data-stage-trouble]")!.textContent).toMatch(/stage runs\s*2 stuck$/);
  });

  it("counts failed stage runs, retries included, and shows nothing when none stuck or failed", () => {
    const m = meetings();
    m.stuck = [];
    m.stages = [{ stage: "transcribe", statuses: { completed: 2, failed: 3 } }];
    expect(stageTrouble(m)).toEqual({ stuck: 0, failed: 3, where: ["transcribe failed ×3"] });
    m.stages = [{ stage: "transcribe", statuses: { completed: 2 } }];
    const { container } = render(<MeetingsPanel meetings={m} />);
    expect(container.querySelector("[data-stage-trouble]")).toBeNull();
  });
});

describe("Empty charts", () => {
  it("leave a bar with nothing drawn out of the accessibility tree instead of naming it with nothing", () => {
    const storage = capture<MakeStorage>("/v1/report/storage");
    storage.classes = [];
    const spend = capture<MakeSpend>("/v1/report/spend");
    spend.by_provider = [];
    const { container } = render(
      <>
        <StoragePanel storage={storage} />
        <SpendPanel spend={spend} />
      </>,
    );
    expect(container.querySelectorAll('[role="img"]:not([aria-label]), [role="img"][aria-label=""]')).toHaveLength(0);
  });
});

describe("Spend", () => {
  it("prints the largest day on its column, so the peak reads without a hover", () => {
    const spend = capture<MakeSpend>("/v1/report/spend");
    spend.daily = [
      { day: "2026-09-29", micros: 1_000_000, attempts: 1 },
      { day: "2026-09-30", micros: 8_280_000, attempts: 3 },
      { day: "2026-10-01", micros: 0, attempts: 0 },
    ];
    const { container } = render(<SpendPanel spend={spend} />);
    expect(container.querySelector("[data-peak]")!.textContent).toBe("$8.28");
  });
});

describe("compactAge", () => {
  it("names the largest whole unit", () => {
    expect(compactAge(30 * 60)).toBe("<1h");
    expect(compactAge(84_594)).toBe("23h");
    expect(compactAge(3 * 86_400)).toBe("3d");
    expect(compactAge(20 * 86_400)).toBe("2w");
  });
});
