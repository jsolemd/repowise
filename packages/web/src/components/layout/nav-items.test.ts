import { describe, expect, it } from "vitest";
import { repoNavGroups, repoNavItems } from "./nav-items";

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

  it("drops only Chat, and drops its now-empty group with it", () => {
    const on = repoNavGroups("r1");
    const off = repoNavGroups("r1", { generativeDisabled: true });

    const onLabels = on.flatMap((g) => g.items).map((i) => i.label);
    const offLabels = off.flatMap((g) => g.items).map((i) => i.label);

    expect(onLabels.filter((l) => l !== "Chat")).toEqual(offLabels);
    // The Chat group holds nothing else, so it should not survive as an empty
    // separator — a blank group renders as a stray divider.
    expect(off.every((g) => g.items.length > 0)).toBe(true);
    expect(off).toHaveLength(on.length - 1);
  });

  it("leaves every href intact for the groups that remain", () => {
    for (const item of repoNavItems("r1", { generativeDisabled: true })) {
      expect(item.href.startsWith("/repos/r1")).toBe(true);
    }
  });
});
