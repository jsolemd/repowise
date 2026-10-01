// @vitest-environment jsdom

import { render, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import captured from "@/lib/make-platform.contract.json";
import type { MakeBuilds, MakeDoctor, MakeEvidence, MakeMedia, MakeMeetings, MakeSpend } from "@/lib/make-platform";
import {
  BuildsPanel,
  DoctorStrip,
  EvidencePanel,
  NeedsYou,
  SpendPanel,
  compactAge,
  meetingShares,
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
    expect(within(container).getByRole("img", { name: "built 3, failed 2" })).toBeTruthy();
  });
});

describe("Meetings", () => {
  it("partitions the recordings, taking stuck and failed ones out of the pipeline's share", () => {
    const meetings = capture<MakeMeetings>("/v1/report/meetings");
    meetings.recordings = { total: 10, summarized: 5, awaiting_summary: 2, in_pipeline: 3, released: 5 };
    meetings.stages = [{ stage: "transcribe", statuses: { completed: 2, failed: 1 } }];
    meetings.stuck = [{ recording: "r1", stage: "diarize", status: "running", updated_at: "2026-10-01T00:00:00Z" }];
    const shares = meetingShares(meetings);
    expect(shares).toEqual({ filed: 5, waiting: 2, flowing: 1, stuck: 2 });
    expect(shares.filed + shares.waiting + shares.flowing + shares.stuck).toBe(meetings.recordings.total);
  });

  it("never draws more stuck recordings than the pipeline holds", () => {
    const meetings = capture<MakeMeetings>("/v1/report/meetings");
    meetings.recordings = { total: 4, summarized: 3, awaiting_summary: 0, in_pipeline: 1, released: 3 };
    meetings.stages = [{ stage: "transcribe", statuses: { failed: 3 } }];
    expect(meetingShares(meetings)).toEqual({ filed: 3, waiting: 0, flowing: 0, stuck: 1 });
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
