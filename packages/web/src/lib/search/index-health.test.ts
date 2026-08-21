import { describe, expect, it } from "vitest";
import { classifyIndexHealth } from "./index-health";
import type { SourceSearchHealth } from "@/lib/api/types";

/** A healthy index: three stores agreeing, nothing queued, nothing stale. */
function healthy(over: Partial<SourceSearchHealth> = {}): SourceSearchHealth {
  return {
    state: "current",
    degraded: false,
    generation_id: "955a228c4079499fb7905b7c4a0d91b5",
    generation_sequence: 45,
    indexed_commit: "8d1e42e9e7d07029f48d834189c47eaf9f6cd0e8",
    expected_chunks: 8200,
    fts_chunks: 8200,
    vector_chunks: 8200,
    pending_updates: 0,
    ready_updates: 0,
    building_updates: 0,
    blocked_updates: 0,
    stale_files: {},
    recipe_fingerprint: "1ce38488c6575240",
    fts_path: ".repowise/source_search/source_fts_v2.db",
    lance_table: "source_chunks_1ce38488c6575240",
    integrity_errors: [],
    last_error: null,
    ...over,
  };
}

describe("classifyIndexHealth", () => {
  it("calls a fully agreeing, idle index current", () => {
    expect(classifyIndexHealth(healthy())).toBe("healthy");
  });

  it("never calls a degraded index healthy, whatever its state says", () => {
    expect(classifyIndexHealth(healthy({ degraded: true }))).toBe("broken");
  });

  it("treats a reported integrity error as broken", () => {
    expect(
      classifyIndexHealth(healthy({ integrity_errors: ["no field named valid_from"] })),
    ).toBe("broken");
  });

  it("treats a last error as broken even with clean counts", () => {
    expect(classifyIndexHealth(healthy({ last_error: "lance write failed" }))).toBe(
      "broken",
    );
  });

  it("outranks a 'current' state when the stores disagree", () => {
    // The exact shape of the shared-corpus incident: status looked fine while
    // one store had lost rows. Parity is the finding, not the status field.
    expect(classifyIndexHealth(healthy({ vector_chunks: 7721 }))).toBe("broken");
  });

  it("says it cannot tell when a count is missing, rather than guessing", () => {
    expect(classifyIndexHealth(healthy({ vector_chunks: null }))).toBe("unknown");
  });

  it("calls a queued index stale, not broken", () => {
    expect(classifyIndexHealth(healthy({ pending_updates: 3 }))).toBe("stale");
    expect(classifyIndexHealth(healthy({ ready_updates: 1 }))).toBe("stale");
    expect(classifyIndexHealth(healthy({ building_updates: 1 }))).toBe("stale");
    expect(classifyIndexHealth(healthy({ blocked_updates: 1 }))).toBe("stale");
  });

  it("calls a non-current state stale when nothing else is wrong", () => {
    expect(classifyIndexHealth(healthy({ state: "building" }))).toBe("stale");
  });

  it("calls an inconsistent state broken, not stale", () => {
    expect(classifyIndexHealth(healthy({ state: "inconsistent" }))).toBe("broken");
  });

  it("calls an unreadable manifest broken, but leaves a missing one to state", () => {
    // The server distinguishes the two deliberately; collapsing them here
    // would throw away the distinction it was changed to preserve.
    expect(classifyIndexHealth(healthy({ manifest_state: "unreadable" }))).toBe(
      "broken",
    );
    expect(classifyIndexHealth(healthy({ manifest_state: "ok" }))).toBe("healthy");
    expect(
      classifyIndexHealth(healthy({ manifest_state: "missing", state: "absent" })),
    ).toBe("stale");
  });

  it("classifies a server too old to send the newer fields", () => {
    // Optional fields absent entirely — an older server. It must not become
    // "unknown" on their account; the counts it does send are what decide.
    const older = healthy();
    delete (older as Partial<SourceSearchHealth>).manifest_state;
    delete (older as Partial<SourceSearchHealth>).symbol_chunks;
    expect(classifyIndexHealth(older)).toBe("healthy");
  });

  it("counts a file whose parse failed as stale", () => {
    expect(
      classifyIndexHealth(healthy({ stale_files: { "src/a.py": "SyntaxError" } })),
    ).toBe("stale");
  });
});
