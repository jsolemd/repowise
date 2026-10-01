import type { Metadata } from "next";
import { Suspense } from "react";
import { Activity } from "lucide-react";
import { PageShell } from "@repowise-dev/ui/shared/page-shell";
import { ApiError } from "@repowise-dev/ui/shared/api-error";
import { OverviewSection } from "@repowise-dev/ui/overview";
import {
  getMakeBuilds,
  getMakeDoctor,
  getMakeEvidence,
  getMakeMedia,
  getMakeMeetings,
  getMakeOperationMix,
  getMakeSpend,
  getMakeStorage,
  REVIEW_LIBRARY_URL,
  type MakeDoctor,
} from "@/lib/make-platform";
import {
  ActivityPanel,
  BuildsPanel,
  DoctorPending,
  DoctorStrip,
  EvidencePanel,
  ImagesPanel,
  MeetingsPanel,
  NeedsYou,
  SpendPanel,
  StoragePanel,
} from "@/components/make/make-sections";
import { LiveRefresh } from "@/components/make/live-refresh";

export const metadata: Metadata = { title: "Make platform" };
export const dynamic = "force-dynamic";

/** Days of spend the panel covers, counted in Jon's time zone. */
const SPEND_DAYS = 30;
/** Days the operation mix covers. */
const MIX_DAYS = 7;
const TIME_ZONE = process.env.MAKE_REPORT_TIME_ZONE || "America/Los_Angeles";

const value = <T,>(result: PromiseSettledResult<T>): T | null =>
  result.status === "fulfilled" ? result.value : null;

/** The doctor is slow (live checks), so it streams in behind the rest. */
async function Doctor({ report }: { report: Promise<MakeDoctor | null> }) {
  return <DoctorStrip doctor={await report} />;
}

/**
 * The Make platform: what waits on Jon's hand, what Make built, what it spent
 * and what it keeps, each a figure and one mark, under the doctor's strip.
 *
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

          <OverviewSection title="Needs you" flush>
            <NeedsYou media={media} meetings={meetings} evidence={evidence} libraryUrl={REVIEW_LIBRARY_URL} />
          </OverviewSection>

          <OverviewSection title="Built">
            <div className="grid gap-x-12 gap-y-10 md:grid-cols-2 lg:grid-cols-3">
              <EvidencePanel evidence={evidence} />
              <BuildsPanel builds={builds} />
              <ImagesPanel media={media} />
              <MeetingsPanel meetings={meetings} />
              <div className="md:col-span-2">
                <ActivityPanel mix={mix} />
              </div>
            </div>
          </OverviewSection>

          <div className="grid gap-x-12 gap-y-[var(--section-gap)] lg:grid-cols-2">
            <OverviewSection title="Spent">
              <SpendPanel spend={spend} />
            </OverviewSection>
            <OverviewSection title="Kept">
              <StoragePanel storage={storage} />
            </OverviewSection>
          </div>
        </>
      )}
    </PageShell>
  );
}
