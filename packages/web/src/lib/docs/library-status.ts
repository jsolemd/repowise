import { AlertTriangle, CheckCircle2, Clock, HelpCircle } from "lucide-react";
import type { DocsLibrary } from "@/lib/api/docs-libraries";

/**
 * Whether an indexed documentation library still matches its source.
 *
 * Four-valued for the same reason the search index verdict is: a library the
 * checker could not read is not fresh and it is not stale — it is unproven,
 * and only saying so lets a reader act on it. Collapsing `unknown` into
 * `fresh` is what let a five-months-stale library sit in the corpus looking
 * current, because nothing ever succeeded at checking it.
 *
 * Order matters. A moved remote sha is measured evidence that the indexed copy
 * is behind, so it outranks whatever the checker last wrote in
 * `last_freshness_state`; and an unknown state is decided before the fresh
 * case so a stale-but-unchecked library can never fall through to `fresh`.
 */
export type LibraryFreshness = "fresh" | "stale" | "outdated" | "unknown";

export function classifyLibrary(lib: DocsLibrary): LibraryFreshness {
  // A failed index is outdated whatever the freshness checker thinks: its
  // chunks are whatever survived the last successful run, if any.
  if (lib.status === "error") return "outdated";

  // A remote on a different commit is measured evidence that this copy is
  // behind, so it outranks the checker's own verdict — including an
  // inconclusive one. "Unknown with a moved sha" is not unknown.
  if (remoteHasMoved(lib)) return "outdated";

  const state = lib.last_freshness_state;

  // Never checked, or checked and inconclusive.
  if (state === null || state === undefined || state === "unknown") return "unknown";
  if (state === "stale") return "stale";
  if (state === "fresh") return "fresh";

  // A state word this UI does not know is not a licence to claim freshness.
  return "unknown";
}

/**
 * Whether the source moved on since this copy was built.
 *
 * The two fields are not the same kind of string, so equality is the wrong
 * test. A git library stores the bare commit in both. A snapshot stores a
 * composite in ``current_sha`` — the remote identity followed by a content
 * hash — because its source has no commit and only the content proves the
 * copy. What holds in every case is that ``last_remote_sha`` is a *prefix* of
 * ``current_sha`` while the copy is current.
 *
 * Measured against the live corpus on 2026-09-07 (84 libraries, all of which
 * the service itself reports fresh): the prefix rule agrees on all 84. Plain
 * equality invents four outdated libraries, and splitting ``current_sha`` on
 * its first colon still invents one — a snapshot whose remote identity is
 * itself colon-separated (``local-research:2026-04-18:curated:v2:<hash>``).
 * Both would have put a red badge on a current library, on the one column this
 * page exists for.
 *
 * A remote that genuinely moves writes a different hash, which will not be a
 * prefix of a composite built from the old one, so the signal survives.
 */
export function remoteHasMoved(
  lib: Pick<DocsLibrary, "current_sha" | "last_remote_sha">,
): boolean {
  const { current_sha: current, last_remote_sha: remote } = lib;
  if (!current || !remote) return false;
  return !current.startsWith(remote);
}

/**
 * How each verdict presents. Modelled on the search index's `VERDICT_UI` so
 * the two freshness surfaces read as one vocabulary.
 *
 * `badge` names an existing `Badge` variant. There is no `unknown` variant, so
 * it borrows `default` — a filled chip like the other three, because the one
 * thing this column must never do is render the unproven case as empty space.
 */
export const DOCS_STATUS_UI: Record<
  LibraryFreshness,
  {
    icon: typeof CheckCircle2;
    tone: string;
    label: string;
    hint: string;
    badge: "fresh" | "stale" | "outdated" | "default";
  }
> = {
  fresh: {
    icon: CheckCircle2,
    tone: "text-[var(--color-success)]",
    label: "Fresh",
    hint: "Checked against its source and matching. Search answers reflect the published docs.",
    badge: "fresh",
  },
  stale: {
    icon: Clock,
    tone: "text-[var(--color-warning)]",
    label: "Stale",
    hint: "The source has moved on since this copy was indexed. Answers are real but may be behind.",
    badge: "stale",
  },
  outdated: {
    icon: AlertTriangle,
    tone: "text-[var(--color-error)]",
    label: "Outdated",
    hint: "The remote is on a different commit, or the last index failed. Treat answers as incomplete.",
    badge: "outdated",
  },
  unknown: {
    icon: HelpCircle,
    tone: "text-[var(--color-text-tertiary)]",
    label: "Cannot tell",
    hint: "No freshness check has succeeded for this library, so its currency is unproven either way.",
    badge: "default",
  },
};

/** Every verdict, in the order a reader should be shown them. */
export const LIBRARY_FRESHNESS_ORDER: LibraryFreshness[] = [
  "outdated",
  "stale",
  "unknown",
  "fresh",
];

/**
 * How long until a scheduled freshness check, as "in 4h".
 *
 * `formatRelativeTime` cannot express this: it measures backwards, so a
 * timestamp in the future collapses to "just now" — which is why the shared
 * null-safe wrapper refuses future dates outright rather than reporting one.
 * A scheduled check is the one figure on this page that is always ahead of
 * now, so it needs its own direction.
 */
export function formatNextCheck(iso: string | null | undefined, fallback = "—"): string {
  if (!iso) return fallback;
  const at = parseUtc(iso);
  if (at === null) return fallback;

  const minutes = Math.round((at - Date.now()) / 60_000);
  if (minutes <= 0) return "due now";
  if (minutes < 60) return `in ${minutes}m`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `in ${hours}h`;
  return `in ${Math.round(hours / 24)}d`;
}

/**
 * Milliseconds for an API timestamp, or null when it is unparseable.
 *
 * The docs service returns UTC wall-clock strings, some without a `Z`. Same
 * rule as `parseDate` in the shared formatters, restated here so this module
 * stays pure and importable from a test with no UI dependencies.
 */
function parseUtc(iso: string): number | null {
  const trimmed = iso.trim();
  const stamped = /[zZ]|[+-]\d\d?:\d\d$/.test(trimmed) ? trimmed : `${trimmed}Z`;
  const at = new Date(stamped).getTime();
  return Number.isNaN(at) ? null : at;
}

/** The soonest check still ahead of us, or null when nothing is scheduled. */
export function earliestNextCheck(libraries: DocsLibrary[]): string | null {
  const now = Date.now();
  let best: { iso: string; at: number } | null = null;
  for (const lib of libraries) {
    const iso = lib.next_freshness_check_at;
    if (!iso) continue;
    const at = parseUtc(iso);
    // A check whose time has passed is not "next": the scheduler is behind on
    // it, and reporting it as upcoming would read as a promise it is not making.
    if (at === null || at < now) continue;
    if (best === null || at < best.at) best = { iso, at };
  }
  return best?.iso ?? null;
}
