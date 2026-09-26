import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AttentionRows, SeverityRows } from "../../src/overview/attention-rows";

describe("SeverityRows", () => {
  it("names each row's severity in text, not only in the dot's colour", () => {
    render(
      <SeverityRows
        items={[
          { id: "a", severity: "high", label: "Images", title: "Paid generations: 1 interrupted", description: "Run make catalog recover." },
          { id: "b", severity: "medium", label: "Meetings", title: "3 transcripts await a note", description: "", href: "/make-platform#meetings" },
        ]}
      />,
    );
    expect(screen.getByText("high severity.", { exact: false })).toBeTruthy();
    expect(screen.getByText("Paid generations: 1 interrupted")).toBeTruthy();
    expect(screen.getByRole("link").getAttribute("href")).toBe("/make-platform#meetings");
  });

  it("says so when nothing needs attention", () => {
    render(<SeverityRows items={[]} emptyText="Every check passes." />);
    expect(screen.getByText("Every check passes.")).toBeTruthy();
  });

  it("keeps the attention list's labels when it renders through the shared rows", () => {
    render(
      <AttentionRows
        items={[{ id: "x", type: "dead_code", title: "Unused export", description: "d", severity: "low" }]}
        hrefFor={() => null}
      />,
    );
    expect(screen.getByText("Dead code")).toBeTruthy();
  });
});
