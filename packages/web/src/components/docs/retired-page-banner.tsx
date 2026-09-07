"use client";

// Explicit, like every other component this app renders under vitest: the web
// tsconfig keeps `jsx: "preserve"` for Next, so esbuild emits classic
// `React.createElement` calls and the runtime import is not injected.
import React from "react";
import { Badge } from "@repowise-dev/ui/ui/badge";
import type { DocPage, DocPageSummary } from "@repowise-dev/types/docs";

/**
 * The notice above a retired page's body.
 *
 * A tombstoned page documents a file that no longer exists. The server keeps
 * serving it — 200, marked `tombstone` — rather than 404ing, because the row
 * carries `successor_paths` and that is the only thing that can tell a reader
 * where the content went. A 404 would throw it away and leave every inbound
 * link stranded on a bare "missing".
 *
 * So the page still renders; this says what it is. The prose below is a true
 * description of a file that was deleted, which makes it history rather than
 * documentation, and the reader has to be told which one they are reading
 * before they act on it.
 */

/** Where a retired page's content moved. Written by the deletion sweep as a
 *  list of repository-relative paths; absent on a plain deletion. */
function successorPaths(page: DocPage): string[] {
  const raw = page.metadata?.successor_paths;
  if (!Array.isArray(raw)) return [];
  return raw.filter((p): p is string => typeof p === "string" && p.length > 0);
}

export function isRetiredPage(page: Pick<DocPage, "freshness_status">): boolean {
  return page.freshness_status === "tombstone";
}

export function RetiredPageBanner({
  page,
  pages,
  onSelectPage,
}: {
  page: DocPage;
  /** The loaded page list, used to resolve a successor path to a real page. */
  pages: DocPageSummary[];
  onSelectPage: (page: DocPageSummary) => void;
}) {
  if (!isRetiredPage(page)) return null;

  const successors = successorPaths(page);
  // A successor path only becomes a link when a page actually exists for it.
  // Page selection is budgeted, so most files have none, and a dead link is
  // worse than plain text naming where the content went.
  const resolved = successors
    .map((path) => ({ path, page: pages.find((p) => p.target_path === path) }))
    .filter((entry) => entry.page !== undefined) as {
    path: string;
    page: DocPageSummary;
  }[];

  return (
    <div
      // `note`, not `status`: this is ancillary context that ships with the
      // page, not a live update — and the Badge inside already carries
      // `role="status"`, so a second one here would nest two live regions.
      role="note"
      aria-label="Retired page"
      className="flex flex-wrap items-center gap-x-2 gap-y-1 border-b border-[var(--color-border-default)] bg-[var(--color-bg-elevated)] px-4 py-2.5 sm:px-6"
    >
      <Badge variant="outdated">Retired</Badge>
      <span className="text-xs text-[var(--color-text-secondary)]">
        {successors.length > 0
          ? "This file was deleted. Its content moved to"
          : "This file was deleted. What follows describes the repository as it was."}
      </span>
      {successors.length > 0 && (
        <span className="text-xs text-[var(--color-text-secondary)]">
          {successors.map((path, i) => {
            const target = resolved.find((entry) => entry.path === path)?.page;
            return (
              <span key={path}>
                {i > 0 && ", "}
                {target ? (
                  <button
                    type="button"
                    onClick={() => onSelectPage(target)}
                    className="font-mono text-[var(--color-accent-primary)] hover:underline"
                  >
                    {path}
                  </button>
                ) : (
                  <span className="font-mono">{path}</span>
                )}
              </span>
            );
          })}
          .
        </span>
      )}
    </div>
  );
}
