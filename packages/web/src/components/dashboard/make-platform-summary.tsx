import Link from "next/link";
import { getMakeSummary } from "@/lib/make-platform";

/** Independent operational section: these figures never enter code-health scores. */
export async function MakePlatformSummary() {
  let report;
  try { report = await getMakeSummary(); } catch { report = null; }
  return (
    <section aria-labelledby="make-platform-summary" className="space-y-2 border-t border-[var(--color-border)] pt-5">
      <Link id="make-platform-summary" href="/make-platform" className="inline-flex min-h-11 items-center font-medium underline underline-offset-4">
        Make platform
      </Link>
      {report ? (
        <p className="text-sm text-[var(--color-text-secondary)]">
          {report.evidence.works.toLocaleString()} papers · {report.evidence.claims.toLocaleString()} claims · {report.media.pending} images awaiting review · {report.operations.ambiguous ?? 0} interrupted operations needing reconciliation
        </p>
      ) : <p className="text-sm text-[var(--color-text-secondary)]">Make reporting is unavailable. Repository reporting remains available.</p>}
    </section>
  );
}
