// @vitest-environment jsdom

import { render, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import captured from "@/lib/make-platform.contract.json";
import type { MakeBuilds, MakeDoctor, MakeEvidence, MakeMedia } from "@/lib/make-platform";
import { BuildsPanel, DoctorGrid, EvidencePanel, ImagesPanel } from "./make-sections";

// Queries are scoped to each render's container: there is no global cleanup.

const reports = captured.reports as Record<string, unknown>;
const capture = <T,>(path: string) => structuredClone(reports[path]) as T;

describe("Images", () => {
  it("draws an owner's blocked cards beside its pending ones, not out of them", () => {
    // Make counts a blocked card apart from pending (media_review_cards_v2).
    const media = capture<MakeMedia>("/v1/report/media");
    media.owners = [{ ...media.owners[0]!, pending: 0, blocked: 2 }];
    const { container } = render(<ImagesPanel media={media} libraryUrl="#" />);
    const bar = within(container).getByRole("img", { name: "pending 0, blocked 2" });
    expect(bar.closest("li")!.lastElementChild!.textContent).toBe("2");
  });
});

describe("Evidence", () => {
  it("counts only the open questions as open", () => {
    const evidence = capture<MakeEvidence>("/v1/report/evidence");
    evidence.questions = { total: 3, by_state: { open: 1, answered: 1, withdrawn: 1 } };
    const { container } = render(<EvidencePanel evidence={evidence} />);
    expect(container.textContent).toContain("1 open question");
    expect(container.textContent).not.toContain("3 open");
  });
});

describe("Doctor", () => {
  it("prints every open finding's check, detail and action without a hover", () => {
    const doctor: MakeDoctor = {
      schema_version: 1,
      catalog_revision: 1,
      findings: [
        { area: "catalog", check: "Make catalog", severity: "ok", detail: "reachable", action: null },
        { area: "media", check: "Stale renders", severity: "warn", detail: "2 renders derive from replaced bytes", action: "reselect them" },
        { area: "storage", check: "Lab bundles", severity: "fail", detail: "1 file is missing", action: "run `make catalog restore`" },
      ],
      counts: { ok: 1, warn: 1, fail: 1 },
    };
    const { container } = render(<DoctorGrid doctor={doctor} />);
    const details = container.querySelector("details")!;
    expect(within(details).getByText("2 open findings")).toBeTruthy();
    const listed = within(details).getAllByRole("listitem").map((li) => li.textContent);
    expect(listed).toEqual([
      expect.stringMatching(/Lab bundles.*1 file is missing.*Run make catalog restore\./),
      expect.stringMatching(/Stale renders.*2 renders derive from replaced bytes.*Reselect them\./),
    ]);
  });
});

describe("Builds", () => {
  it("prints a group's failed count", () => {
    const builds = capture<MakeBuilds>("/v1/report/builds");
    builds.groups = [{ group: "lectures", documents: 5, failed: 2, latest_built_at: "2026-10-01T00:00:00Z" }];
    const { container } = render(<BuildsPanel builds={builds} />);
    expect(container.textContent).toContain("2 failed");
  });
});
