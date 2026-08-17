"use client";

import useSWR from "swr";
import { useDebounce } from "./use-debounce";
import { search } from "@/lib/api/search";
import type { SearchEnvelope } from "@repowise-dev/api-client/search";

/** What a search that has not run yet looks like, so callers never branch on
 *  `undefined` before reading `results`. */
const NOTHING: SearchEnvelope = { results: [], candidates: [], selected_owner: null };

export function useSearch(
  query: string,
  opts?: {
    search_type?: "semantic" | "fulltext";
    limit?: number;
    debounce?: number;
    repo_id?: string;
  },
) {
  const debounced = useDebounce(query, opts?.debounce ?? 300);
  const key =
    debounced.trim().length >= 2
      ? `search:${debounced}:${opts?.search_type}:${opts?.repo_id ?? "all"}`
      : null;
  const { data, error, isLoading } = useSWR<SearchEnvelope>(
    key,
    () =>
      search(debounced, {
        search_type: opts?.search_type,
        limit: opts?.limit,
        repo_id: opts?.repo_id,
      }),
    { revalidateOnFocus: false },
  );
  const envelope = data ?? NOTHING;
  return {
    envelope,
    results: envelope.results,
    error,
    isLoading: isLoading && !!key,
    isTyping: query !== debounced,
  };
}
