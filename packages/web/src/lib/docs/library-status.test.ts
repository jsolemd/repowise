import { describe, expect, it, vi } from "vitest";
import {
  classifyLibrary,
  compareLibraries,
  formatNextCheck,
  DOCS_STATUS_UI,
  LIBRARY_FRESHNESS_ORDER,
  remoteHasMoved,
  type LibraryFreshness,
} from "./library-status";
import type { DocsLibrary } from "@/lib/api/docs-libraries";

/** A library the checker read successfully and found current. */
function library(over: Partial<DocsLibrary> = {}): DocsLibrary {
  return {
    library_id: "react-hook-form",
    name: "React Hook Form",
    repo: "react-hook-form/react-hook-form",
    source_type: "git",
    status: "ready",
    branch: "master",
    docs_path: "docs",
    source_subpath: null,
    file_count: 128,
    chunk_count: 3140,
    indexed_at: "2026-09-01T08:00:00+00:00",
    graph_synced_at: "2026-09-01T08:04:00+00:00",
    last_freshness_state: "fresh",
    graph_sync_error: null,
    error_message: null,
    priority: 5,
    current_sha: "a1b2c3d4e5f6a7b8",
    freshness_checked_at: "2026-09-07T06:00:00+00:00",
    next_freshness_check_at: "2026-09-08T06:00:00+00:00",
    last_remote_sha: "a1b2c3d4e5f6a7b8",
    last_freshness_error: null,
    ...over,
  };
}

describe("classifyLibrary", () => {
  it("classifies a library checked within the interval as fresh", () => {
    expect(classifyLibrary(library())).toBe("fresh");
  });

  it("classifies an unknown freshness state as unknown, never as fresh", () => {
    expect(classifyLibrary(library({ last_freshness_state: "unknown" }))).toBe(
      "unknown",
    );
  });

  it("classifies a library that was never checked as unknown", () => {
    expect(
      classifyLibrary(
        library({
          last_freshness_state: null,
          freshness_checked_at: null,
          last_remote_sha: null,
        }),
      ),
    ).toBe("unknown");
  });

  it("classifies a library whose remote sha moved as outdated", () => {
    expect(
      classifyLibrary(library({ last_remote_sha: "ff00ff00ff00ff00" })),
    ).toBe("outdated");
  });

  it("does not read a snapshot's composite sha as a moved remote", () => {
    // Measured on the live corpus 2026-09-07: a snapshot stores
    // `<remote sha>:<content sha>` in current_sha while last_remote_sha holds
    // the remote identity alone. Plain equality called four such libraries
    // outdated when the service reported every one of them fresh.
    expect(
      classifyLibrary(
        library({
          source_type: "snapshot",
          current_sha:
            "cb75c2876f0e5f22e92f59a6092bbc04e20d05f6bd08526347bf8c59a677d623:bad1de551ce1360b6a34797c558e8426250d5b4300a83184aab516e1def3c16d",
          last_remote_sha:
            "cb75c2876f0e5f22e92f59a6092bbc04e20d05f6bd08526347bf8c59a677d623",
        }),
      ),
    ).toBe("fresh");
  });

  it("tolerates a remote identity that is itself colon-separated", () => {
    // The real `/codeatlas/mazehq-homepage` row: both fields hold the same
    // colon-laden identity, which a split-on-first-colon rule got wrong.
    const sha = "local-research:2026-04-18:curated:v2:d5ec9da55766014b9d5dae00e76c6451";
    expect(
      classifyLibrary(
        library({ source_type: "snapshot", current_sha: sha, last_remote_sha: sha }),
      ),
    ).toBe("fresh");
  });

  it("still catches a snapshot whose remote identity moved", () => {
    expect(
      classifyLibrary(
        library({
          source_type: "snapshot",
          current_sha: "aaaaaaaa:bbbbbbbb",
          last_remote_sha: "cccccccc",
        }),
      ),
    ).toBe("outdated");
  });

  it("reports no move when either sha is missing", () => {
    // Absent evidence is not evidence of divergence; the state word decides.
    expect(remoteHasMoved({ current_sha: null, last_remote_sha: "abc" })).toBe(false);
    expect(remoteHasMoved({ current_sha: "abc", last_remote_sha: null })).toBe(false);
    expect(remoteHasMoved({ current_sha: "abcdef", last_remote_sha: "abc" })).toBe(true);
    expect(remoteHasMoved({ current_sha: "abcdef", last_remote_sha: "zzz" })).toBe(true);
  });

  it("lets a moved remote sha outrank an inconclusive check", () => {
    // Measured divergence beats the absence of a verdict: the copy is behind
    // whatever the checker managed to conclude about it.
    expect(
      classifyLibrary(
        library({
          last_freshness_state: "unknown",
          last_remote_sha: "ff00ff00ff00ff00",
        }),
      ),
    ).toBe("outdated");
  });

  it("classifies a reported stale state as stale", () => {
    expect(classifyLibrary(library({ last_freshness_state: "stale" }))).toBe("stale");
  });

  it("classifies a failed index as outdated whatever the checker said", () => {
    expect(
      classifyLibrary(library({ status: "error", error_message: "clone failed" })),
    ).toBe("outdated");
  });

  it("never claims freshness from a state word it does not know", () => {
    expect(classifyLibrary(library({ last_freshness_state: "quantum" }))).toBe(
      "unknown",
    );
  });
});

describe("DOCS_STATUS_UI", () => {
  const verdicts: LibraryFreshness[] = ["fresh", "stale", "outdated", "unknown"];

  it("every classification has a DOCS_STATUS_UI entry", () => {
    for (const verdict of verdicts) {
      const ui = DOCS_STATUS_UI[verdict];
      expect(ui, verdict).toBeDefined();
      expect(ui.label).toBeTruthy();
      expect(ui.hint).toBeTruthy();
      expect(ui.icon).toBeTruthy();
    }
  });

  it("gives the unknown case a real badge variant, not a blank cell", () => {
    // The state that hid a months-stale library must be visible as a chip.
    expect(DOCS_STATUS_UI.unknown.badge).toBe("default");
  });

  it("orders the verdicts worst-first and names each one once", () => {
    expect([...LIBRARY_FRESHNESS_ORDER].sort()).toEqual([...verdicts].sort());
    expect(LIBRARY_FRESHNESS_ORDER[0]).toBe("outdated");
  });
});


describe("inventory ordering and timestamps", () => {
  it.each(["asc", "desc"] as const)("keeps missing dates last when %s", (order) => {
    const rows = [
      library({ name: "Z missing", indexed_at: null }),
      library({ name: "Recent", indexed_at: "2026-09-07T08:00:00" }),
      library({ name: "A invalid", indexed_at: "invalid" }),
      library({ name: "Older", indexed_at: "2026-09-01T08:00:00Z" }),
    ].sort((a, b) => compareLibraries(a, b, "indexed_at", order));
    expect(rows.map((row) => row.name)).toEqual([
      ...(order === "asc" ? ["Older", "Recent"] : ["Recent", "Older"]),
      "A invalid", "Z missing",
    ]);
  });

  it("puts the most concerning freshness first by default", () => {
    const rows = [
      library({ name: "fresh" }),
      library({ name: "unknown", last_freshness_state: null }),
      library({ name: "stale", last_freshness_state: "stale" }),
      library({ name: "outdated", status: "error" }),
    ].sort((a, b) => compareLibraries(a, b, "freshness", "desc"));
    expect(rows.map((row) => row.name)).toEqual(["outdated", "stale", "unknown", "fresh"]);
  });

  it("treats bare API timestamps as UTC for future check times", () => {
    const now = vi.spyOn(Date, "now").mockReturnValue(Date.parse("2026-09-07T08:00:00Z"));
    try {
      expect(formatNextCheck("2026-09-07T12:00:00")).toBe("in 4h");
      expect(formatNextCheck("2026-09-07T07:00:00Z")).toBe("due now");
      expect(formatNextCheck("invalid")).toBe("—");
    } finally {
      now.mockRestore();
    }
  });

  it("requires the complete remote identity before a snapshot content hash", () => {
    expect(remoteHasMoved({ current_sha: "curated:v20:content", last_remote_sha: "curated:v2" })).toBe(true);
    expect(remoteHasMoved({ current_sha: "curated:v2:content", last_remote_sha: "curated:v2" })).toBe(false);
  });
});
