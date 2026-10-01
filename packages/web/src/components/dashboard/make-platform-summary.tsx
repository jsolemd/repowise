import Link from "next/link";
import { OverviewSection, SectionLink } from "@repowise-dev/ui/overview";
import { getMakeEvidence, getMakeMedia, getMakeMeetings, REVIEW_LIBRARY_URL } from "@/lib/make-platform";
import { NeedsYou } from "@/components/make/make-sections";

const value = <T,>(result: PromiseSettledResult<T>): T | null =>
  result.status === "fulfilled" ? result.value : null;

/**
 * Make, beside the repositories: what waits on Jon's hand, and the way in.
 *
 * The same five tiles that open the Make platform page, from the same three
 * reports, so the two never disagree; the summary report counts pending image
 * selections rather than the library's cards and carries no copies or
 * documents, so it cannot stand in. The three answered in about 125 ms in
 * parallel on 2026-10-01, media the slowest. An independent operational
 * section: these figures never enter code-health scores.
 */
export async function MakePlatformSummary() {
  const results = await Promise.allSettled([getMakeMedia(), getMakeMeetings(), getMakeEvidence()] as const);
  const [media, meetings, evidence] = [value(results[0]), value(results[1]), value(results[2])];
  return (
    <OverviewSection
      title="Make platform"
      action={<SectionLink href="/make-platform" LinkComponent={Link}>Open Make platform</SectionLink>}
    >
      {media || meetings || evidence ? (
        <NeedsYou media={media} meetings={meetings} evidence={evidence} libraryUrl={REVIEW_LIBRARY_URL} />
      ) : (
        <p role="status" className="m-0 text-xs text-[var(--color-text-tertiary)]">
          Make · no answer
        </p>
      )}
    </OverviewSection>
  );
}
