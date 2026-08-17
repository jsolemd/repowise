import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  asFilePageTab,
  fileTabsFor,
  resolveFileTab,
  FILE_PAGE_TABS,
} from "../../src/files/file-page-tabs.js";
import type { FileDetailResponse } from "@repowise-dev/types/files";

// vitest runs from the package root.
const MODULE_PATH = join(process.cwd(), "src/files/file-page-tabs.ts");

/**
 * The route renders on the server and calls all three of these before the
 * `FilePage` shell mounts. A function exported from a `"use client"` module is
 * a client reference, so calling one during a server render throws
 * "Attempted to call X() from the server" — and because the bundler only draws
 * that boundary in a production build, a dev server renders the route fine
 * while `next build` output 500s. Nothing at type-check or test time notices,
 * which is why the guard is on the source.
 */
describe("file-page-tabs stays callable from a server component", () => {
  it("carries no client directive", () => {
    const source = readFileSync(MODULE_PATH, "utf8");
    expect(source).not.toMatch(/^\s*["']use client["']/);
  });

  it("exports every helper the route calls before the shell mounts", () => {
    expect(typeof asFilePageTab).toBe("function");
    expect(typeof fileTabsFor).toBe("function");
    expect(typeof resolveFileTab).toBe("function");
  });
});

describe("asFilePageTab", () => {
  it("passes through every tab the page renders", () => {
    for (const tab of FILE_PAGE_TABS) {
      expect(asFilePageTab(tab)).toBe(tab);
    }
  });

  it("rejects a ?tab= value that is not a tab", () => {
    expect(asFilePageTab("blame")).toBeUndefined();
    expect(asFilePageTab("")).toBeUndefined();
    expect(asFilePageTab(undefined)).toBeUndefined();
  });
});

function makeFileDetail(): FileDetailResponse {
  return {
    file_path: "src/a.ts",
    health: { metric: null, findings: [] },
    function_blame: [],
    graph: null,
    coverage: null,
    wiki_page: null,
    governing_decisions: [],
  } as unknown as FileDetailResponse;
}

describe("resolveFileTab", () => {
  it("falls back to overview for a tab this file does not have", () => {
    const tabs = fileTabsFor(makeFileDetail());
    // Decisions is the one tab that disappears when nothing governs the file.
    expect(tabs.some((t) => t.id === "decisions")).toBe(false);
    expect(resolveFileTab("decisions", tabs)).toBe("overview");
  });

  it("keeps a tab the file does have", () => {
    const tabs = fileTabsFor(makeFileDetail());
    expect(resolveFileTab("history", tabs)).toBe("history");
  });
});
