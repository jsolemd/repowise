// @vitest-environment jsdom

import React from "react";
import { render, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RetiredPageBanner, isRetiredPage } from "./retired-page-banner";
import type { DocPage, DocPageSummary } from "@repowise-dev/types/docs";

// Queries are scoped to each render's own container rather than `screen`:
// this suite has no global cleanup (there is no vitest setup file), so a
// document-wide query would see every earlier test's markup as well.

function page(over: Partial<DocPage> = {}): DocPage {
  return {
    id: "file_page:src/gone.py",
    repository_id: "r1",
    page_type: "file_page",
    title: "gone.py",
    summary: "",
    target_path: "src/gone.py",
    source_hash: "abc123",
    model_name: "mock",
    provider_name: "mock",
    input_tokens: 0,
    output_tokens: 0,
    confidence: 1,
    freshness_status: "fresh",
    human_notes: null,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
    content: "# gone.py",
    metadata: {},
    ...over,
  } as DocPage;
}

function summary(over: Partial<DocPageSummary> = {}): DocPageSummary {
  return { ...page(), ...over } as DocPageSummary;
}

describe("RetiredPageBanner", () => {
  it("renders nothing for a live page", () => {
    const { container } = render(
      <RetiredPageBanner page={page()} pages={[]} onSelectPage={() => {}} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("announces a deleted file for a tombstoned page", () => {
    const { container } = render(
      <RetiredPageBanner
        page={page({ freshness_status: "tombstone" })}
        pages={[]}
        onSelectPage={() => {}}
      />,
    );

    const banner = within(container).getByRole("note");
    expect(banner.textContent).toMatch(/this file was deleted/i);
    expect(within(container).getByText("Retired")).toBeTruthy();
  });

  it("names where the content moved and links a successor that resolves", () => {
    const onSelectPage = vi.fn();
    const successor = summary({
      id: "file_page:src/main.py",
      target_path: "src/main.py",
      title: "main.py",
    });

    const { container } = render(
      <RetiredPageBanner
        page={page({
          freshness_status: "tombstone",
          metadata: { successor_paths: ["src/main.py"] },
        })}
        pages={[successor]}
        onSelectPage={onSelectPage}
      />,
    );

    expect(within(container).getByRole("note").textContent).toMatch(/moved to/i);
    within(container).getByRole("button", { name: "src/main.py" }).click();
    expect(onSelectPage).toHaveBeenCalledWith(successor);
  });

  it("names an unresolvable successor as text rather than a dead link", () => {
    // Page selection is budgeted, so most files have no page of their own. A
    // link that opens nothing is worse than plain text naming the path.
    const { container } = render(
      <RetiredPageBanner
        page={page({
          freshness_status: "tombstone",
          metadata: { successor_paths: ["src/unpaged.py"] },
        })}
        pages={[]}
        onSelectPage={() => {}}
      />,
    );

    expect(within(container).getByText("src/unpaged.py")).toBeTruthy();
    expect(within(container).queryByRole("button")).toBeNull();
  });
});

describe("isRetiredPage", () => {
  it("is true only for the tombstone status", () => {
    expect(isRetiredPage({ freshness_status: "tombstone" })).toBe(true);
    for (const status of ["fresh", "stale", "outdated"]) {
      expect(isRetiredPage({ freshness_status: status })).toBe(false);
    }
  });
});
