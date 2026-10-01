import type { Metadata } from "next";
import { Suspense } from "react";
import { Activity } from "lucide-react";
import { PageShell } from "@repowise-dev/ui/shared/page-shell";
import { ApiError } from "@repowise-dev/ui/shared/api-error";
import { OverviewSection } from "@repowise-dev/ui/overview";
import { formatAgeDays } from "@repowise-dev/ui/lib/format";
import {
  getMakeBuilds,
  getMakeDoctor,
  getMakeEvidence,
  getMakeMedia,
  getMakeMeetings,
  getMakeOperationMix,
  getMakeSpend,
  getMakeStorage,
  type MakeDoctor,
} from "@/lib/make-platform";
import {
  ActivityPanel,
  BuildsPanel,
  DoctorGrid,
  DoctorPending,
  EvidencePanel,
  ImagesPanel,
  MeetingsPanel,
  SpendTile,
  StoragePanel,
} from "@/components/make/make-sections";
import { Label, Meter, RAMP } from "@/components/make/make-charts";
import { LiveRefresh } from "@/components/make/live-refresh";

export const metadata: Metadata = { title: "Make platform" };
export const dynamic = "force-dynamic";

/** Days of spend the tile covers, counted in Jon's time zone. */
const SPEND_DAYS = 30;
/** Days the operation mix covers. */
const MIX_DAYS = 7;
const TIME_ZONE = process.env.MAKE_REPORT_TIME_ZONE || "America/Los_Angeles";
/** Where Jon gives verdicts on renders: SoleMD.Web's review library. */
const REVIEW_LIBRARY_URL = process.env.MAKE_REVIEW_LIBRARY_URL || "https://solemd-web.taild0afc1.ts.net/make";
const DAY_SECONDS = 86_400;

const value = <T,>(result: PromiseSettledResult<T>): T | null =>
  result.status === "fulfilled" ? result.value : null;

/** The doctor is slow (live checks), so it streams in behind the rest. */
async function Doctor({ report }: { report: Promise<MakeDoctor | null> }) {
  const doctor = await report;
  return doctor ? (
    <DoctorGrid doctor={doctor} />
  ) : (
    <ApiError title="The doctor did not answer" message="make-api answered other reports but not /v1/report/doctor." />
  );
}

/**
 * The Make platform: Jon's window onto what Make built, spent and kept.
 *
 * Charts first, with the doctor's open findings in a list that opens by touch
 * and the rest of the prose on hover. Agents read make-api directly and act on
 * the doctor's findings, so this page shows shapes and counts, not a queue.
 * Read-only by contract (SoleMD.Make native-catalog-cutover): every figure is
 * a make-api report reached through a private socket from this server render,
 * and the page re-renders when Make's catalog revision moves (LiveRefresh).
 */
export default async function MakePlatformPage() {
  // Started with the others and awaited only inside its Suspense boundary.
  const doctor = getMakeDoctor().catch(() => null);
  const results = await Promise.allSettled([
    getMakeSpend(SPEND_DAYS, TIME_ZONE),
    // The panel prints only the latest build's time; the groups carry the rest.
    getMakeBuilds({ limit: 1 }),
    getMakeEvidence(),
    getMakeMedia(),
    getMakeMeetings(),
    getMakeStorage(),
    getMakeOperationMix(MIX_DAYS),
  ] as const);
  const [spend, builds, evidence, media, meetings, storage, mix] = [
    value(results[0]),
    value(results[1]),
    value(results[2]),
    value(results[3]),
    value(results[4]),
    value(results[5]),
    value(results[6]),
  ];
  const fulfilled = [spend, builds, evidence, media, meetings, storage, mix].filter((r) => r !== null);
  // The oldest revision any section was drawn at, so a write that landed
  // mid-render still refreshes the page.
  const revision = fulfilled.length ? Math.min(...fulfilled.map((r) => r.catalog_revision)) : null;

  return (
    <PageShell
      title="Make platform"
      icon={<Activity className="h-5 w-5 text-[var(--color-text-tertiary)]" />}
      actions={<LiveRefresh initialRevision={revision} />}
    >
      {!fulfilled.length ? (
        <ApiError
          title="Make reporting is unavailable"
          message="make-api did not answer on its private socket. Repository reporting is unaffected."
        />
      ) : (
        <>
          <Suspense fallback={<DoctorPending />}>
            <Doctor report={doctor} />
          </Suspense>

          <div className="grid grid-cols-2 divide-[var(--color-border-default)] lg:grid-cols-4 lg:divide-x [&>*:first-child]:pl-0 max-lg:[&>*:nth-child(odd)]:pl-0">
            {media ? (
              <Meter label="Images to review" value={media.queue.pending} href={REVIEW_LIBRARY_URL}
                unit={media.queue.oldest_seconds !== null ? `oldest ${formatAgeDays(media.queue.oldest_seconds / DAY_SECONDS)}` : undefined} />
            ) : <div />}
            {meetings ? (
              <Meter label="Notes to file" value={meetings.recordings.awaiting_summary} of={meetings.recordings.total} color="var(--color-node-needs-work)" />
            ) : <div />}
            {evidence ? (
              <Meter label="Papers awaiting a copy" value={evidence.fulltext.waiting} of={evidence.fulltext.needed} color={RAMP[2]} />
            ) : <div />}
            <div className="flex min-w-0 flex-col gap-2 p-4">
              <Label>Spend</Label>
              <SpendTile spend={spend} />
            </div>
          </div>

          <OverviewSection title="Pipelines">
            <div className="grid gap-x-12 gap-y-10 lg:grid-cols-2">
              <EvidencePanel evidence={evidence} />
              <BuildsPanel builds={builds} />
              <ImagesPanel media={media} libraryUrl={REVIEW_LIBRARY_URL} />
              <MeetingsPanel meetings={meetings} />
            </div>
          </OverviewSection>

          <OverviewSection title="Activity">
            <ActivityPanel mix={mix} />
          </OverviewSection>

          <OverviewSection title="Storage">
            <StoragePanel storage={storage} />
          </OverviewSection>
        </>
      )}
    </PageShell>
  );
}
