import { apiGet } from "./client";
import type { SearchResultResponse } from "./types";

/**
 * How much the server believes its own answer.
 *
 * Derived from absolute evidence — an exact name match, a dense cosine over a
 * fixed floor, dense/lexical agreement — never from a normalised window, so
 * `confident` is a claim about the corpus rather than about the top of the
 * list. Absent when the server did not classify at all; see
 * {@link SearchEnvelope.confidence}.
 */
export type SearchConfidence = "confident" | "caution" | "no_match";

/** Why one hit ranked where it did. Present only on the classified shape. */
export interface SearchEvidence {
  /** Cosine against the query embedding, or null when only lexical hit it. */
  dense_cosine: number | null;
  /** 1-based rank in the lexical leg, or null when only dense hit it. */
  lexical_rank: number | null;
  exact_name: boolean;
  /** Which corpus answered: the source index or the generated wiki. */
  lane: string;
}

/**
 * One hit, in the union of the two shapes `/api/search` can serve.
 *
 * The server has two search hosts. The stock one indexes generated wiki pages
 * and answers with a flat list of page rows; the hybrid one also indexes
 * source chunks and answers with an envelope whose results can be a symbol or
 * a window of a file, which no page row can describe. Rather than model them
 * as a union that every caller has to narrow, both are normalised into this
 * one shape and the fields that only one of them can fill are optional.
 *
 * Read `page_id` to open the documentation surface and `file` to open the
 * code; a hit can carry either, both, or — for a wiki page whose type is not
 * backed by a file — only a title.
 */
export interface SearchHit {
  /** Repo-relative path, or "" when nothing on disk backs this hit. */
  file: string;
  /** The wiki page's primary key, when this hit is a page. */
  page_id?: string;
  /** A symbol's bare name, a file's basename, or a page's title. */
  name: string;
  /** A symbol kind ("function", "class"), "file_window", or a page type. */
  kind: string;
  /** Which index produced it: "symbol", "file_window" or "wiki_page". */
  source: string;
  snippet: string;
  /** Comparable within one response only; the two hosts use different scales. */
  relevance_score: number;
  /** 1-based inclusive line bounds, when the hit is a span of a file. */
  start_line?: number;
  end_line?: number;
  evidence?: SearchEvidence;
}

/**
 * A whole search answer: the hits, and what the server can say about them.
 *
 * Everything past `results` is optional because the stock host does not
 * produce it. A caller that renders confidence has to treat "the server did
 * not classify" as its own state rather than as `confident`, which is the
 * whole reason this is `undefined` and not a default.
 */
export interface SearchEnvelope {
  results: SearchHit[];
  /** Openable files behind the whole ranking, best first. */
  candidates: { path: string }[];
  /** The one file the server would open, and why. Null when there is none. */
  selected_owner: { file: string; reason: string } | null;
  /** Absent when the server served the unclassified flat-list shape. */
  confidence?: SearchConfidence;
  /** The retrieval mode that answered, e.g. "hybrid". */
  mode?: string;
  /**
   * The server's own words about a `no_match`.
   *
   * Written for an API consumer — it names `_meta.source_search` and refers to
   * "the results below" — so a user-facing surface should say the same thing
   * in its own words rather than paste this through.
   */
  note?: string;
  /** The commit the index was built from, when the server named one. */
  indexed_commit?: string;
}

const CONFIDENCE_VALUES: readonly string[] = ["confident", "caution", "no_match"];

function asConfidence(value: unknown): SearchConfidence | undefined {
  return typeof value === "string" && CONFIDENCE_VALUES.includes(value)
    ? (value as SearchConfidence)
    : undefined;
}

function asPositiveInt(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/**
 * A flat-list row as a hit.
 *
 * The stock host only ever returns wiki pages, so `source` is fixed and the
 * page's own `page_type` carries the kind. `target_path` is a repo-relative
 * file for the page types that document one and free text (a module name, "")
 * for the rest — which is exactly what `file` means here, so page types that
 * name no file are left with `file: ""` rather than a path that resolves to
 * nothing.
 */
function hitFromPageRow(row: SearchResultResponse): SearchHit {
  const filePath = PAGE_TYPES_BACKED_BY_A_FILE.has(row.page_type)
    ? // A symbol spotlight's target is `file.py::Symbol`; the file is the head.
      (row.target_path.split("::", 1)[0] ?? "")
    : "";
  return {
    file: filePath,
    page_id: row.page_id,
    name: row.title,
    kind: row.page_type,
    source: "wiki_page",
    snippet: row.snippet,
    relevance_score: row.score,
  };
}

/**
 * Page types whose `target_path` is a repo-relative file. Mirrors
 * `_FILE_BACKED_PAGE_TYPES` in `core/source_search/coordinator.py` — the two
 * answer the same question about the same rows and have to agree.
 *
 * This reads a file *out of* a page's target. It is not the inverse: a page id
 * is never rebuilt from a type and a file, because `symbol_spotlight:a/b.py`
 * is a real page about the whole file rather than a miss you would notice.
 * Rows here carry their own `page_id`, so nothing needs to.
 */
const PAGE_TYPES_BACKED_BY_A_FILE: ReadonlySet<string> = new Set([
  "file_page",
  "symbol_spotlight",
  "api_contract",
  "infra_page",
]);

/** An envelope row as a hit, keeping only the fields it actually carried. */
function hitFromEnvelopeRow(raw: Record<string, unknown>): SearchHit {
  // `file` is the envelope's own key; `target_path` mirrors it for callers
  // that already know the stock name. Either will do, so read both rather
  // than depend on which one a given server version fills.
  const file = asString(raw.file) || asString(raw.target_path);
  const evidence = raw.evidence as Record<string, unknown> | undefined;
  const start = asPositiveInt(raw.start_line);
  const end = asPositiveInt(raw.end_line);
  const pageId = asString(raw.page_id);

  return {
    file,
    ...(pageId ? { page_id: pageId } : {}),
    name: asString(raw.name),
    kind: asString(raw.kind),
    source: asString(raw.source),
    snippet: asString(raw.snippet),
    relevance_score: asPositiveInt(raw.relevance_score) ?? 0,
    ...(start !== undefined ? { start_line: start } : {}),
    ...(end !== undefined ? { end_line: end } : {}),
    ...(evidence
      ? {
          evidence: {
            dense_cosine: asPositiveInt(evidence.dense_cosine) ?? null,
            lexical_rank: asPositiveInt(evidence.lexical_rank) ?? null,
            exact_name: evidence.exact_name === true,
            lane: asString(evidence.lane),
          },
        }
      : {}),
  };
}

/**
 * Either wire shape as one {@link SearchEnvelope}.
 *
 * Exported because the shape is decided by a server-side feature flag the
 * dashboard cannot read: whichever one arrives has to be handled at runtime,
 * so the branch is worth testing directly rather than only through `search`.
 */
export function normalizeSearchResponse(raw: unknown): SearchEnvelope {
  if (Array.isArray(raw)) {
    return {
      results: (raw as SearchResultResponse[]).map(hitFromPageRow),
      candidates: [],
      selected_owner: null,
    };
  }

  if (raw === null || typeof raw !== "object") {
    return { results: [], candidates: [], selected_owner: null };
  }

  const body = raw as Record<string, unknown>;
  const rows = Array.isArray(body.results) ? body.results : [];
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const owner = body.selected_owner as Record<string, unknown> | null | undefined;
  const confidence = asConfidence(body.confidence);
  const mode = asString(body.mode);
  const note = asString(body.note);
  const meta = body._meta as Record<string, unknown> | undefined;
  const sourceMeta = meta?.source_search as Record<string, unknown> | undefined;
  const indexedCommit = asString(sourceMeta?.indexed_commit);

  return {
    results: rows.map((row) => hitFromEnvelopeRow(row as Record<string, unknown>)),
    candidates: candidates
      .map((c) => ({ path: asString((c as Record<string, unknown>).path) }))
      .filter((c) => c.path !== ""),
    selected_owner:
      owner && asString(owner.file)
        ? { file: asString(owner.file), reason: asString(owner.reason) }
        : null,
    ...(confidence ? { confidence } : {}),
    ...(mode ? { mode } : {}),
    ...(note ? { note } : {}),
    ...(indexedCommit ? { indexed_commit: indexedCommit } : {}),
  };
}

/**
 * Search the index.
 *
 * Returns the normalised envelope rather than the raw body: `/api/search`
 * serves a flat list of wiki pages by default and a richer object when the
 * server runs its hybrid source+wiki host, and which one a caller gets is
 * decided by server configuration it has no way to query.
 */
export async function search(
  query: string,
  opts?: {
    search_type?: "semantic" | "fulltext";
    limit?: number;
    repo_id?: string;
  },
): Promise<SearchEnvelope> {
  const params: Record<string, string | number> = {
    query,
    search_type: opts?.search_type ?? "semantic",
    limit: opts?.limit ?? 10,
  };
  if (opts?.repo_id) {
    params.repo_id = opts.repo_id;
  }
  return normalizeSearchResponse(await apiGet<unknown>("/api/search", params));
}
