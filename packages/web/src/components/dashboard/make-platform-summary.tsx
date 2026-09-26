import Link from "next/link";
import { OverviewSection, SectionLink } from "@repowise-dev/ui/overview";
import { formatCost, formatNumber } from "@repowise-dev/ui/lib/format";
import { getMakeSummary, microsToUsd } from "@/lib/make-platform";

/**
 * Make, beside the repositories: one line and the way in.
 *
 * An independent operational section: these figures never enter code-health
 * scores. Read from the summary alone so the dashboard pays one small request.
 */
export async function MakePlatformSummary() {
  let report;
  try { report = await getMakeSummary(); } catch { report = null; }
  return (
    <OverviewSection
      title="Make platform"
      description="Documents, evidence, images and recordings, reported by Make."
      action={<SectionLink href="/make-platform" LinkComponent={Link}>Open Make platform</SectionLink>}
    >
      <p className="text-sm text-[var(--color-text-secondary)] [text-wrap:pretty]">
        {report
          ? `${formatNumber(report.media.pending ?? 0)} images await review, ${formatNumber(report.meetings.awaiting_summary)} transcripts await a note, and ${formatCost(microsToUsd(report.spend_24h_micros))} was spent in the last 24 hours${report.media.interrupted_dispatches ? `; ${formatNumber(report.media.interrupted_dispatches)} paid generations were interrupted and need reconciling` : ""}.`
          : "Make reporting is unavailable. Repository reporting is unaffected."}
      </p>
    </OverviewSection>
  );
}
