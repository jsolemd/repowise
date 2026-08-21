import type { SourceSearchHealth } from "@/lib/api/types";

/**
 * Whether a search answer from this index can be trusted, decided from the
 * index's own account of itself.
 *
 * Four-valued on purpose. The lifecycle's own doctrine is that a broken store
 * must never be mistakable for an empty one; the same asymmetry applies to a
 * reader deciding whether to believe a result. An index that cannot describe
 * itself is not healthy, and it is not stale either — it is unproven, and
 * saying so is the only honest answer available.
 *
 * Parity outranks `state`. Three stores that disagree are not current whatever
 * the status field claims, because a search reads two of them and the manifest
 * counts the third.
 */
export type IndexVerdict = "healthy" | "stale" | "broken" | "unknown";

export function classifyIndexHealth(h: SourceSearchHealth): IndexVerdict {
  if (h.degraded || h.integrity_errors.length > 0 || h.last_error) return "broken";
  if (h.state === "inconsistent") return "broken";
  // A manifest that exists but cannot be read is corruption, and the whole
  // point of the server distinguishing it from "missing" is that a reader can
  // act on the difference. Missing is a legitimate not-yet-built state and is
  // left to `state` below; unreadable never is.
  if (h.manifest_state === "unreadable") return "broken";

  const counts = [h.expected_chunks, h.fts_chunks, h.vector_chunks];
  if (counts.some((c) => c === null || c === undefined)) return "unknown";
  if (new Set(counts).size > 1) return "broken";

  const queued =
    h.pending_updates + h.ready_updates + h.building_updates + h.blocked_updates;
  if (h.state !== "current" || queued > 0 || Object.keys(h.stale_files).length > 0)
    return "stale";

  return "healthy";
}
