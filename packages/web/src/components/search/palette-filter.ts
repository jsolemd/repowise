import { defaultFilter } from "cmdk";

/**
 * The keyword that marks a palette item as already ranked.
 *
 * cmdk scores every item against the input and drops the ones that score
 * zero. That is the right behaviour for the palette's static entries — a
 * fixed list of pages and repositories, matched by name as you type — and the
 * wrong behaviour for anything the palette has already ranked itself, because
 * those were chosen by something the input text cannot reproduce.
 *
 * A server-side search is the clear case: ask for `reconcile_project_files`
 * and the index answers with pages titled "File: …/writer_mutations.py",
 * every one of which scores zero against the query that found it. They were
 * fetched, ranked, and then hidden, so the palette reported "no results" for
 * a search that had returned eight.
 *
 * A marker on the item rather than `shouldFilter={false}` on the root: the
 * static entries still want fuzzy matching, and turning filtering off wholesale
 * would mean reimplementing it for them.
 */
export const PRE_RANKED = "cmdk:pre-ranked";

/** The `keywords` prop for an item that has already been ranked. */
export const preRankedKeywords: string[] = [PRE_RANKED];

/**
 * Score a palette item, letting pre-ranked ones through untouched.
 *
 * Pre-ranked items all score 1, which is the top of `defaultFilter`'s range,
 * so they sort above fuzzy matches and hold the order they were rendered in —
 * `Array.prototype.sort` is stable, and cmdk sorts by score alone.
 */
export function paletteFilter(
  value: string,
  search: string,
  keywords?: string[],
): number {
  if (keywords?.includes(PRE_RANKED)) return 1;
  return defaultFilter(value, search, keywords);
}
