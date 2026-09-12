import { describe, expect, it } from "vitest";
import { GLOBAL_NAV, repoNavGroups, repoNavItems } from "./nav-items";

/**
 * Navigation must not offer a destination the server refuses.
 *
 * Under the hard no-generative policy the chat route answers with a notice and
 * the API answers 409, so a Chat entry in the sidebar, the mobile nav, or the
 * palette's "Go to" list is an invitation to a dead end.
 */
describe("repo navigation under the deployment policy", () => {
  it("offers Chat by default", () => {
    const labels = repoNavItems("r1").map((i) => i.label);
    expect(labels).toContain("Chat");
  });

  it("drops Chat when generative surfaces are disabled", () => {
    const labels = repoNavItems("r1", { generativeDisabled: true }).map(
      (i) => i.label,
    );
    expect(labels).not.toContain("Chat");
  });

  it("drops only Chat and preserves the other navigation items", () => {
    const on = repoNavGroups("r1");
    const off = repoNavGroups("r1", { generativeDisabled: true });

    const onLabels = on.flatMap((g) => g.items).map((i) => i.label);
    const offLabels = off.flatMap((g) => g.items).map((i) => i.label);

    expect(onLabels.filter((l) => l !== "Chat")).toEqual(offLabels);
    expect(off.every((g) => g.items.length > 0)).toBe(true);
    expect(on[0].items[1].label).toBe("Chat");
    expect(off).toHaveLength(on.length);
  });

  it("leaves every href intact for the groups that remain", () => {
    for (const item of repoNavItems("r1", { generativeDisabled: true })) {
      expect(item.href.startsWith("/repos/r1")).toBe(true);
    }
  });
});

/**
 * The documentation corpus is workspace-wide, not per-repo, so its only
 * possible entry point is the global nav. Without an entry the page exists at
 * a URL nobody can reach by navigating.
 */
describe("global navigation", () => {
  it("offers the docs library index", () => {
    const entry = GLOBAL_NAV.find((i) => i.href === "/docs-libraries");
    expect(entry).toBeDefined();
    expect(entry?.label).toBe("Docs Libraries");
    expect(entry?.icon).toBeTruthy();
  });

  it("keeps Settings pinned last", () => {
    // The IA's one ordering rule: everything new lands above Settings.
    expect(GLOBAL_NAV.at(-1)?.href).toBe("/settings");
  });
});
