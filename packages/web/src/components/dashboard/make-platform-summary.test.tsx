// @vitest-environment jsdom

import { render, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import captured from "@/lib/make-platform.contract.json";
import { MakeReportError } from "@/lib/make-platform";
import { MakePlatformSummary } from "./make-platform-summary";

const reports = captured.reports as Record<string, unknown>;

const getters = vi.hoisted(() => ({
  getMakeMedia: vi.fn(),
  getMakeMeetings: vi.fn(),
  getMakeEvidence: vi.fn(),
}));
vi.mock("@/lib/make-platform", async (original) => ({ ...(await original<object>()), ...getters }));

beforeEach(() => {
  getters.getMakeMedia.mockResolvedValue(structuredClone(reports["/v1/report/media"]));
  getters.getMakeMeetings.mockResolvedValue(structuredClone(reports["/v1/report/meetings"]));
  getters.getMakeEvidence.mockResolvedValue(structuredClone(reports["/v1/report/evidence"]));
});

describe("Make platform summary", () => {
  it("shows the page's tiles, counted from the page's reports, with the way in", async () => {
    const { container } = render(await MakePlatformSummary());
    expect(container.textContent).toMatch(/Images to review\s*4/);
    expect(container.textContent).toMatch(/Notes to file\s*3/);
    expect(container.textContent).toMatch(/Copies to fetch\s*96\s*of 142/);
    expect(container.textContent).toMatch(/Stale concepts\s*0/);
    expect(container.textContent).toMatch(/Docs to rebuild\s*0/);
    expect(within(container).getByRole("link", { name: /Open Make platform/ }).getAttribute("href")).toBe("/make-platform");
  });

  it("keeps the tiles that answered when one report does not", async () => {
    getters.getMakeMeetings.mockRejectedValue(new MakeReportError(503));
    const { container } = render(await MakePlatformSummary());
    expect(container.textContent).toMatch(/Notes to file\s*—\s*no answer/);
    expect(container.textContent).toMatch(/Images to review\s*4/);
  });

  it("says so in one line when Make does not answer at all", async () => {
    for (const getter of Object.values(getters)) getter.mockRejectedValue(new MakeReportError(503));
    const { container } = render(await MakePlatformSummary());
    expect(within(container).getByRole("status").textContent).toBe("Make · no answer");
    expect(container.textContent).not.toContain("Images to review");
  });
});
