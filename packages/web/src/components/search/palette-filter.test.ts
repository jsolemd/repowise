import { describe, expect, it } from "vitest";
import { defaultFilter } from "cmdk";
import { PRE_RANKED, paletteFilter, preRankedKeywords } from "./palette-filter";

describe("paletteFilter", () => {
  it("keeps a search hit whose title shares nothing with the query", () => {
    // The reproducing case: the index answers `reconcile_project_files` with
    // pages titled after the files that hold it. cmdk's own filter scores
    // every one of them zero, so the palette rendered a successful search as
    // an empty one.
    const title = "page-File: codeatlas/code_search/neo4j/writer_mutations.py";
    const query = "reconcile_project_files";

    expect(defaultFilter(title, query, [])).toBe(0);
    expect(paletteFilter(title, query, preRankedKeywords)).toBeGreaterThan(0);
  });

  it("scores pre-ranked items at the top of the range so they sort first", () => {
    const preRanked = paletteFilter("page-anything", "anything", preRankedKeywords);
    const fuzzy = paletteFilter("dashboard", "dash", []);

    expect(preRanked).toBe(1);
    expect(preRanked).toBeGreaterThanOrEqual(fuzzy);
  });

  it("gives every pre-ranked item the same score, leaving their order alone", () => {
    // cmdk sorts by score and nothing else, and `Array.prototype.sort` is
    // stable — equal scores mean the server's ranking is what the reader sees.
    const scores = ["a", "b", "c"].map((v) =>
      paletteFilter(v, "unrelated", preRankedKeywords),
    );
    expect(new Set(scores).size).toBe(1);
  });

  it("still filters the static entries by name", () => {
    expect(paletteFilter("dashboard", "dash", [])).toBeGreaterThan(0);
    expect(paletteFilter("dashboard", "settings", [])).toBe(0);
  });

  it("matches cmdk's own scoring for anything not marked pre-ranked", () => {
    for (const [value, search] of [
      ["dashboard", "dash"],
      ["settings", "set"],
      ["repo-infra-mirror", "infra"],
      ["workspace-contracts", "zzz"],
    ] as const) {
      expect(paletteFilter(value, search, [])).toBe(defaultFilter(value, search, []));
    }
  });

  it("does not treat an unrelated keyword as a pre-ranked marker", () => {
    expect(paletteFilter("dashboard", "settings", ["home"])).toBe(0);
    expect(preRankedKeywords).toEqual([PRE_RANKED]);
  });
});
