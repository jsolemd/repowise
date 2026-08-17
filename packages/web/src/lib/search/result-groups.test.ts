import { describe, expect, it } from "vitest";
import { normalizeSearchResponse } from "@repowise-dev/api-client/search";
import {
  formatLineRange,
  groupSearchResults,
  rankDecisions,
} from "./result-groups";
import type { DecisionRecordResponse } from "@/lib/api/types";

const PREFIX = "/repos/r1";

/** The hybrid host's shape, trimmed to the fields the dashboard reads. */
function envelopeBody(overrides: Record<string, unknown> = {}) {
  return {
    results: [
      {
        file: "codeatlas/code_search/neo4j/writer_mutations.py",
        target_path: "codeatlas/code_search/neo4j/writer_mutations.py",
        name: "reconcile_project_files",
        kind: "method",
        source: "symbol",
        snippet: "async def reconcile_project_files(...)",
        relevance_score: 0.015285,
        evidence: {
          dense_cosine: 0.5611,
          lexical_rank: 9,
          exact_name: true,
          lane: "source",
        },
        start_line: 164,
        end_line: 218,
      },
    ],
    mode: "hybrid",
    confidence: "confident",
    candidates: [{ path: "codeatlas/code_search/neo4j/writer_mutations.py" }],
    selected_owner: {
      file: "codeatlas/code_search/neo4j/writer_mutations.py",
      reason: "exact name match",
    },
    _meta: { timing_ms: 152.15 },
    ...overrides,
  };
}

/** The stock host's shape: a flat list of wiki page rows. */
function pageRow(overrides: Record<string, unknown> = {}) {
  return {
    page_id: "file_page:codeatlas/code_search/sync/utils.py",
    title: "File: codeatlas/code_search/sync/utils.py",
    page_type: "file_page",
    target_path: "codeatlas/code_search/sync/utils.py",
    score: 7.23,
    snippet: "…",
    search_type: "fulltext",
    ...overrides,
  };
}

describe("normalizeSearchResponse", () => {
  it("reads the hybrid host's envelope", () => {
    const envelope = normalizeSearchResponse(envelopeBody());

    expect(envelope.confidence).toBe("confident");
    expect(envelope.mode).toBe("hybrid");
    expect(envelope.selected_owner?.reason).toBe("exact name match");
    expect(envelope.candidates).toHaveLength(1);
    expect(envelope.results[0]).toMatchObject({
      file: "codeatlas/code_search/neo4j/writer_mutations.py",
      name: "reconcile_project_files",
      source: "symbol",
      start_line: 164,
      end_line: 218,
    });
    expect(envelope.results[0]?.evidence?.lane).toBe("source");
  });

  it("reads the stock host's flat list", () => {
    const envelope = normalizeSearchResponse([pageRow()]);

    expect(envelope.results).toHaveLength(1);
    expect(envelope.results[0]).toMatchObject({
      page_id: "file_page:codeatlas/code_search/sync/utils.py",
      file: "codeatlas/code_search/sync/utils.py",
      source: "wiki_page",
      kind: "file_page",
    });
    // No confidence: the stock host classifies nothing, and defaulting to
    // "confident" would be inventing a claim it never made.
    expect(envelope.confidence).toBeUndefined();
    expect(envelope.candidates).toEqual([]);
    expect(envelope.selected_owner).toBeNull();
  });

  it("leaves a page type that names no file without one", () => {
    const envelope = normalizeSearchResponse([
      pageRow({ page_id: "module_page:code_search", page_type: "module_page", target_path: "code_search" }),
    ]);
    expect(envelope.results[0]?.file).toBe("");
  });

  it("takes the file off the head of a symbol spotlight's target", () => {
    const envelope = normalizeSearchResponse([
      pageRow({
        page_id: "symbol_spotlight:a/b.py::Klass.method",
        page_type: "symbol_spotlight",
        target_path: "a/b.py::Klass.method",
      }),
    ]);
    expect(envelope.results[0]?.file).toBe("a/b.py");
  });

  it("survives a body that is neither shape", () => {
    for (const body of [null, undefined, "nope", 42, {}]) {
      const envelope = normalizeSearchResponse(body);
      expect(envelope.results).toEqual([]);
      expect(envelope.confidence).toBeUndefined();
    }
  });

  it("drops a candidate with no path rather than emitting an empty one", () => {
    const envelope = normalizeSearchResponse(
      envelopeBody({ candidates: [{ path: "a.py" }, { path: "" }, {}] }),
    );
    expect(envelope.candidates).toEqual([{ path: "a.py" }]);
  });
});

describe("groupSearchResults — sections", () => {
  it("puts a source hit under Code and opens it at its file", () => {
    const { groups } = groupSearchResults({
      envelope: normalizeSearchResponse(envelopeBody()),
      linkPrefix: PREFIX,
    });

    expect(groups.map((g) => g.id)).toEqual(["code"]);
    expect(groups[0]?.heading).toBe("Code");
    expect(groups[0]?.entries[0]).toMatchObject({
      label: "reconcile_project_files",
      href: "/repos/r1/files/codeatlas/code_search/neo4j/writer_mutations.py",
      detail: "codeatlas/code_search/neo4j/",
      startLine: 164,
      endLine: 218,
    });
  });

  it("puts a wiki hit under Documentation and opens it in the docs reader", () => {
    const { groups } = groupSearchResults({
      envelope: normalizeSearchResponse([pageRow()]),
      linkPrefix: PREFIX,
    });

    expect(groups.map((g) => g.id)).toEqual(["docs"]);
    expect(groups[0]?.entries[0]?.href).toBe(
      "/repos/r1/docs?page=file_page%3Acodeatlas%2Fcode_search%2Fsync%2Futils.py",
    );
  });

  it("rebuilds a page id the envelope dropped, for a file-backed page type", () => {
    // The hybrid host serves no page_id and overwrites the page's own
    // target_path with the file. For a file_page the id is still recoverable.
    const { groups } = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          results: [
            {
              file: "a/b.py",
              target_path: "a/b.py",
              name: "File: a/b.py",
              kind: "file_page",
              source: "wiki_page",
              snippet: "",
              relevance_score: 0.01,
              evidence: { dense_cosine: 0.5, lexical_rank: null, exact_name: false, lane: "wiki" },
            },
          ],
        }),
      ),
      linkPrefix: PREFIX,
    });

    expect(groups[0]?.id).toBe("docs");
    expect(groups[0]?.entries[0]?.href).toBe("/repos/r1/docs?page=file_page%3Aa%2Fb.py");
  });

  it("prefers a page_id the server sent over a rebuilt one", () => {
    const { groups } = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          results: [
            {
              file: "a/b.py",
              page_id: "symbol_spotlight:a/b.py::Klass.method",
              name: "Symbol: Klass.method",
              kind: "symbol_spotlight",
              source: "wiki_page",
              snippet: "",
              relevance_score: 0.01,
            },
          ],
        }),
      ),
      linkPrefix: PREFIX,
    });

    expect(groups[0]?.entries[0]?.href).toContain("symbol_spotlight%3Aa%2Fb.py%3A%3AKlass.method");
  });

  it("drops a wiki hit that carries no way to open it", () => {
    // A module page through the hybrid host: the envelope keeps the title and
    // drops both the page id and the file, so there is nowhere to send anyone.
    const { groups, isEmpty } = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          results: [
            {
              file: "",
              target_path: "",
              name: "Code Search Sync",
              kind: "module_page",
              source: "wiki_page",
              snippet: "",
              relevance_score: 0.01,
            },
          ],
        }),
      ),
      linkPrefix: PREFIX,
    });

    expect(groups).toEqual([]);
    expect(isEmpty).toBe(true);
  });

  it("orders the sections Code, Documentation, Decisions", () => {
    const { groups } = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          results: [
            {
              file: "a/b.py",
              name: "File: a/b.py",
              kind: "file_page",
              source: "wiki_page",
              snippet: "",
              relevance_score: 0.2,
            },
            {
              file: "c/d.py",
              name: "thing",
              kind: "function",
              source: "symbol",
              snippet: "",
              relevance_score: 0.1,
              start_line: 3,
              end_line: 9,
            },
          ],
        }),
      ),
      decisions: [makeDecision("d1", "Use Neo4j for the graph")],
      linkPrefix: PREFIX,
    });

    expect(groups.map((g) => g.id)).toEqual(["code", "docs", "decisions"]);
    expect(groups[2]?.entries[0]).toMatchObject({
      label: "Use Neo4j for the graph",
      href: "/repos/r1/decisions/d1",
      detail: "active",
    });
  });

  it("leaves out a section with nothing in it", () => {
    const { groups } = groupSearchResults({
      envelope: normalizeSearchResponse(envelopeBody()),
      linkPrefix: PREFIX,
    });
    expect(groups.map((g) => g.id)).not.toContain("decisions");
    expect(groups.map((g) => g.id)).not.toContain("docs");
  });
});

describe("groupSearchResults — per-file dedupe", () => {
  it("collapses rows that open the same place and counts what it collapsed", () => {
    // The stock host returns a file's page and its symbol spotlight as two
    // rows; both open the same file page once the id is what opens them.
    const { groups } = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          results: [
            { file: "a/b.py", name: "first", kind: "function", source: "symbol", snippet: "", relevance_score: 0.3, start_line: 1, end_line: 5 },
            { file: "a/b.py", name: "second", kind: "function", source: "symbol", snippet: "", relevance_score: 0.2, start_line: 40, end_line: 60 },
            { file: "a/b.py", name: "third", kind: "function", source: "symbol", snippet: "", relevance_score: 0.1 },
            { file: "c/d.py", name: "other", kind: "function", source: "symbol", snippet: "", relevance_score: 0.05 },
          ],
        }),
      ),
      linkPrefix: PREFIX,
    });

    const code = groups[0]!;
    expect(code.entries).toHaveLength(2);
    // The best-ranked row survives, with its own line bounds intact.
    expect(code.entries[0]).toMatchObject({ label: "first", startLine: 1, alsoMatched: 2 });
    expect(code.entries[1]).toMatchObject({ label: "other" });
    expect(code.entries[1]?.alsoMatched).toBeUndefined();
  });

  it("keeps a page and the file it documents apart", () => {
    // Different open targets — the docs reader and the file page — so these
    // are two answers, not one.
    const { groups } = groupSearchResults({
      envelope: normalizeSearchResponse([
        pageRow({ page_id: "file_page:a/b.py", target_path: "a/b.py" }),
      ]),
      linkPrefix: PREFIX,
    });
    const { groups: codeGroups } = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          results: [{ file: "a/b.py", name: "x", kind: "function", source: "symbol", snippet: "", relevance_score: 1 }],
        }),
      ),
      linkPrefix: PREFIX,
    });

    expect(groups[0]?.entries[0]?.href).not.toBe(codeGroups[0]?.entries[0]?.href);
  });
});

describe("groupSearchResults — confidence", () => {
  it("passes a caution verdict through", () => {
    const grouped = groupSearchResults({
      envelope: normalizeSearchResponse(envelopeBody({ confidence: "caution" })),
      linkPrefix: PREFIX,
    });
    expect(grouped.confidence).toBe("caution");
  });

  it("carries the server's note on a no_match", () => {
    const grouped = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          confidence: "no_match",
          results: [],
          candidates: [],
          selected_owner: null,
          note: "No indexed match for 'x'.",
        }),
      ),
      linkPrefix: PREFIX,
    });
    expect(grouped.confidence).toBe("no_match");
    expect(grouped.emptyMessage).toContain("Nothing in the index matches that.");
    expect(grouped.isEmpty).toBe(true);
  });

  it("renders no rows on a no_match, however many the server sent", () => {
    // The server answers `no_match` with nearest neighbours and says outright
    // they are not evidence. A palette is a list you hit Enter on, so showing
    // them would hand over a wrong destination.
    const grouped = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          confidence: "no_match",
          note: "No indexed match for 'zzzqqxwv'.",
          results: [
            { file: "a/b.py", name: "unrelated", kind: "function", source: "symbol", snippet: "", relevance_score: 0.001 },
            { file: "c/d.py", name: "also unrelated", kind: "function", source: "symbol", snippet: "", relevance_score: 0.001 },
          ],
        }),
      ),
      linkPrefix: PREFIX,
    });

    expect(grouped.groups).toEqual([]);
    expect(grouped.isEmpty).toBe(true);
    expect(grouped.emptyMessage).toContain("Nothing in the index matches that.");
  });

  it("names the indexed commit, so 'nothing found' is a fact about the index", () => {
    const grouped = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          confidence: "no_match",
          results: [],
          _meta: {
            source_search: { indexed_commit: "8d1e42e9e7d07029f48d834189c47eaf9f6cd0e8" },
          },
        }),
      ),
      linkPrefix: PREFIX,
    });
    expect(grouped.emptyMessage).toBe(
      "Nothing in the index matches that. It covers this repository at commit 8d1e42e.",
    );
  });

  it("says it plainly when the server named no commit", () => {
    const grouped = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({ confidence: "no_match", results: [], _meta: {} }),
      ),
      linkPrefix: PREFIX,
    });
    expect(grouped.emptyMessage).toBe(
      "Nothing in the index matches that. It covers this repository as it was last indexed.",
    );
  });

  it("never pastes the server's API-shaped note into the reader's view", () => {
    // The server's `note` names `_meta.source_search` and refers to "the
    // results below", neither of which means anything in this dialog.
    const grouped = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          confidence: "no_match",
          results: [],
          note: "No indexed match for 'x'. … the commit named in _meta.source_search …",
        }),
      ),
      linkPrefix: PREFIX,
    });
    expect(grouped.emptyMessage).not.toContain("_meta");
    expect(grouped.emptyMessage).not.toContain("results below");
  });

  it("still offers decisions on a no_match, which the verdict says nothing about", () => {
    // The confidence is the search host's reading of its own index. Decision
    // records are matched here, out of a list, so it does not cover them.
    const grouped = groupSearchResults({
      envelope: normalizeSearchResponse(
        envelopeBody({
          confidence: "no_match",
          results: [
            { file: "a/b.py", name: "unrelated", kind: "function", source: "symbol", snippet: "", relevance_score: 0.001 },
          ],
        }),
      ),
      decisions: [makeDecision("d1", "Pin the driver")],
      linkPrefix: PREFIX,
    });

    expect(grouped.groups.map((g) => g.id)).toEqual(["decisions"]);
  });

  it("says nothing when the server said nothing", () => {
    const grouped = groupSearchResults({
      envelope: normalizeSearchResponse([pageRow()]),
      linkPrefix: PREFIX,
    });
    expect(grouped.confidence).toBeUndefined();
    expect(grouped.emptyMessage).toBeUndefined();
  });

  it("reports the owner the server chose", () => {
    const grouped = groupSearchResults({
      envelope: normalizeSearchResponse(envelopeBody()),
      linkPrefix: PREFIX,
    });
    expect(grouped.ownerFile).toBe("codeatlas/code_search/neo4j/writer_mutations.py");
  });
});

function makeDecision(
  id: string,
  title: string,
  decision = "",
): DecisionRecordResponse {
  return {
    id,
    title,
    status: "active",
    decision,
    repository_id: "r1",
  } as unknown as DecisionRecordResponse;
}

describe("rankDecisions", () => {
  const records = [
    makeDecision("d1", "Pin the Neo4j driver", "we pin it"),
    makeDecision("d2", "Neo4j is the graph store", "chosen for traversals"),
    makeDecision("d3", "Use Postgres for state", "neo4j was rejected here"),
  ];

  it("ranks a title match above a body match", () => {
    const ranked = rankDecisions(records, "neo4j");
    expect(ranked.map((r) => r.id)).toEqual(["d2", "d1", "d3"]);
  });

  it("prefers an earlier title match", () => {
    // "Neo4j is…" starts with it; "Pin the Neo4j driver" does not.
    expect(rankDecisions(records, "neo4j")[0]?.id).toBe("d2");
  });

  it("stays quiet under two characters", () => {
    expect(rankDecisions(records, "n")).toEqual([]);
    expect(rankDecisions(records, " ")).toEqual([]);
  });

  it("returns nothing when nothing matches", () => {
    expect(rankDecisions(records, "kubernetes")).toEqual([]);
  });

  it("caps the list", () => {
    const many = Array.from({ length: 20 }, (_, i) => makeDecision(`d${i}`, `Neo4j ${i}`));
    expect(rankDecisions(many, "neo4j", 3)).toHaveLength(3);
  });
});

describe("formatLineRange", () => {
  const base = { key: "k", label: "l", detail: "d", href: "/h" };

  it("renders a span", () => {
    expect(formatLineRange({ ...base, startLine: 164, endLine: 218 })).toBe("164-218");
  });

  it("renders a single line once", () => {
    expect(formatLineRange({ ...base, startLine: 20, endLine: 20 })).toBe("20");
    expect(formatLineRange({ ...base, startLine: 20 })).toBe("20");
  });

  it("renders nothing when the hit spans no lines", () => {
    expect(formatLineRange(base)).toBe("");
  });
});
