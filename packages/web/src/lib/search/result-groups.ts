import type { SearchEnvelope, SearchHit } from "@repowise-dev/api-client/search";
import {
  docsPagePath,
  fileEntityPath,
} from "@repowise-dev/ui/shared/entity";
import type { DecisionRecordResponse } from "@/lib/api/types";

/**
 * Turning a search answer into the rows the palette renders.
 *
 * The server has two search hosts and the dashboard cannot tell which one it
 * is talking to, so everything here reads the normalised
 * {@link SearchEnvelope} and treats the fields only the hybrid host fills as
 * optional. That is also why grouping is decided by `source` and `kind` rather
 * than by which shape arrived: a page is a page whichever host served it.
 */

export type SearchGroupId = "code" | "docs" | "decisions";

export interface SearchResultEntry {
  /** Stable React key, and the dedupe key: two rows that open the same place. */
  key: string;
  /** A symbol's name, a file's basename, or a page's title. */
  label: string;
  /** Repo-relative directory, or the page's kind — whichever locates the row. */
  detail: string;
  href: string;
  /** 1-based inclusive bounds, when the hit is a span of a file. */
  startLine?: number;
  endLine?: number;
  /**
   * How many further hits collapsed into this row.
   *
   * Deduping is per open target, so a file the index matched three times is
   * one row — but saying so is better than quietly dropping two answers.
   */
  alsoMatched?: number;
}

export interface SearchResultGroup {
  id: SearchGroupId;
  heading: string;
  entries: SearchResultEntry[];
}

export interface GroupedSearchResults {
  groups: SearchResultGroup[];
  /**
   * The server's own reading of its answer, or undefined when it did not say.
   * Undefined is not `confident`: the stock host classifies nothing, and
   * rendering its silence as confidence would be inventing a claim.
   */
  confidence?: SearchEnvelope["confidence"];
  /**
   * What to tell the reader when nothing matched.
   *
   * Composed here rather than taken from the server's `note`, which is written
   * for an API consumer: it names `_meta.source_search` and talks about "the
   * results below", neither of which means anything in a dialog that is
   * showing no results. Only set on a `no_match`.
   */
  emptyMessage?: string;
  /** The one file the server would open, when it named one. */
  ownerFile?: string;
  /** Whether anything at all is renderable. */
  isEmpty: boolean;
}

const GROUP_HEADING: Record<SearchGroupId, string> = {
  code: "Code",
  docs: "Documentation",
  decisions: "Decisions",
};

/** Which section a hit belongs in.
 *
 *  Read off the lane rather than the kind: the source index holds code and the
 *  wiki index holds prose, whatever either one calls its rows. */
function groupOf(hit: SearchHit): SearchGroupId {
  return hit.source === "wiki_page" ? "docs" : "code";
}

function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

function dirname(path: string): string {
  const name = basename(path);
  return path.slice(0, path.length - name.length);
}

/**
 * Where a hit opens.
 *
 * Code opens the file, never a symbol page: a symbol's id carries its parent
 * (`file.py::Class::method`) and the hit carries only the bare name, so a
 * constructed symbol URL 404s for every method in the index. The line bounds
 * ride on the entry instead, which is what tells the reader where to look.
 *
 * Documentation opens the page the server named, and otherwise the file the
 * page documents. A page id is never rebuilt from the type and the path.
 * `symbol_spotlight:a/b.py::Foo` would rebuild as `symbol_spotlight:a/b.py`,
 * which is not a 404 — it is a real page about something else, so the reader
 * gets the wrong document and nothing anywhere says so. A page whose id the
 * server withheld is better left unopenable than opened at a guess.
 */
function hrefOf(hit: SearchHit, linkPrefix: string): string | undefined {
  if (groupOf(hit) === "code") {
    return hit.file ? fileEntityPath(linkPrefix, hit.file) : undefined;
  }
  if (hit.page_id) return docsPagePath(linkPrefix, hit.page_id);
  return hit.file ? fileEntityPath(linkPrefix, hit.file) : undefined;
}

function entryOf(hit: SearchHit, linkPrefix: string): SearchResultEntry | undefined {
  const href = hrefOf(hit, linkPrefix);
  // A hit with nowhere to go is dropped rather than rendered dead. It happens
  // for a wiki page whose type names no file, where the envelope keeps the
  // title and drops the identity the reader would need to open it.
  if (!href) return undefined;

  const isCode = groupOf(hit) === "code";
  const label = hit.name || (hit.file ? basename(hit.file) : "") || hit.kind;

  return {
    key: href,
    label,
    detail: isCode ? dirname(hit.file) : hit.file || hit.kind,
    href,
    ...(hit.start_line !== undefined ? { startLine: hit.start_line } : {}),
    ...(hit.end_line !== undefined ? { endLine: hit.end_line } : {}),
  };
}

/**
 * Rank decision records against a query.
 *
 * Decisions are not in the search index — it holds wiki pages and source
 * chunks and nothing else — so the palette matches the repo's records itself,
 * the same way it matches the file list. Title first, then the decision text,
 * so a record named for what it decided outranks one that merely mentions it.
 */
export function rankDecisions(
  records: readonly DecisionRecordResponse[],
  query: string,
  limit = 5,
): DecisionRecordResponse[] {
  const q = query.trim().toLowerCase();
  if (q.length < 2) return [];

  const scored: { record: DecisionRecordResponse; score: number }[] = [];
  for (const record of records) {
    const title = record.title.toLowerCase();
    const inTitle = title.indexOf(q);
    if (inTitle !== -1) {
      scored.push({ record, score: inTitle });
      continue;
    }
    if (record.decision.toLowerCase().includes(q)) {
      scored.push({ record, score: 1000 });
    }
  }
  scored.sort((a, b) => a.score - b.score || a.record.title.localeCompare(b.record.title));
  return scored.slice(0, limit).map((s) => s.record);
}

function decisionEntries(
  records: readonly DecisionRecordResponse[],
  linkPrefix: string,
): SearchResultEntry[] {
  return records.map((record) => ({
    key: `${linkPrefix}/decisions/${record.id}`,
    label: record.title,
    detail: record.status,
    href: `${linkPrefix}/decisions/${encodeURIComponent(record.id)}`,
  }));
}

/**
 * Collapse rows that open the same place, counting what was collapsed.
 *
 * The hybrid host already returns one hit per file, so this mostly bites on
 * the stock shape, where a file's page and its symbol spotlight are two rows
 * that lead to the same file.
 */
function dedupe(entries: SearchResultEntry[]): SearchResultEntry[] {
  const byKey = new Map<string, SearchResultEntry>();
  for (const entry of entries) {
    const seen = byKey.get(entry.key);
    if (!seen) {
      byKey.set(entry.key, entry);
      continue;
    }
    seen.alsoMatched = (seen.alsoMatched ?? 0) + 1;
  }
  return [...byKey.values()];
}

export interface GroupSearchResultsInput {
  envelope: SearchEnvelope;
  /** Already ranked; see {@link rankDecisions}. */
  decisions?: readonly DecisionRecordResponse[];
  /** `/repos/{id}`. */
  linkPrefix: string;
}

/** Search results as the palette's sections, in reading order. */
export function groupSearchResults({
  envelope,
  decisions = [],
  linkPrefix,
}: GroupSearchResultsInput): GroupedSearchResults {
  const buckets: Record<SearchGroupId, SearchResultEntry[]> = {
    code: [],
    docs: [],
    decisions: decisionEntries(decisions, linkPrefix),
  };

  // A `no_match` answer still carries rows, and the server says in as many
  // words what they are: "the results below (if any) are nearest neighbours,
  // not evidence". A palette is a list you hit Enter on, so rendering them
  // would hand the reader a wrong destination under a caveat they did not
  // read. The note goes out instead of the rows, not above them.
  const isNoMatch = envelope.confidence === "no_match";

  if (!isNoMatch) {
    for (const hit of envelope.results) {
      const entry = entryOf(hit, linkPrefix);
      if (entry) buckets[groupOf(hit)].push(entry);
    }
  }

  const groups: SearchResultGroup[] = (["code", "docs", "decisions"] as const)
    .map((id) => ({ id, heading: GROUP_HEADING[id], entries: dedupe(buckets[id]) }))
    .filter((group) => group.entries.length > 0);

  return {
    groups,
    ...(envelope.confidence ? { confidence: envelope.confidence } : {}),
    ...(isNoMatch ? { emptyMessage: noMatchMessage(envelope.indexed_commit) } : {}),
    ...(envelope.selected_owner?.file ? { ownerFile: envelope.selected_owner.file } : {}),
    isEmpty: groups.length === 0,
  };
}

/**
 * What a `no_match` says to a reader.
 *
 * Naming the commit is the useful half of the server's own note: it turns
 * "nothing found" into "nothing found *in this*", which is the difference
 * between a dead end and a fact about the index.
 */
function noMatchMessage(indexedCommit: string | undefined): string {
  const base = "Nothing in the index matches that.";
  if (!indexedCommit) return `${base} It covers this repository as it was last indexed.`;
  return `${base} It covers this repository at commit ${indexedCommit.slice(0, 7)}.`;
}

/** `src/a.py:164-218`, or `src/a.py:164`, or "" when the hit spans no lines. */
export function formatLineRange(entry: SearchResultEntry): string {
  if (entry.startLine === undefined) return "";
  if (entry.endLine === undefined || entry.endLine === entry.startLine) {
    return `${entry.startLine}`;
  }
  return `${entry.startLine}-${entry.endLine}`;
}
