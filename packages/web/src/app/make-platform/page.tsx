import type { Metadata } from "next";
import Link from "next/link";
import { Activity } from "lucide-react";
import { PageShell } from "@repowise-dev/ui/shared/page-shell";
import { PageLede } from "@repowise-dev/ui/shared/page-lede";
import { ApiError } from "@repowise-dev/ui/shared/api-error";
import { OverviewSection } from "@repowise-dev/ui/overview";
// Direct path, not the `ui/stats` barrel, which re-exports client modules.
import { StatRibbon, type RibbonStat } from "@repowise-dev/ui/stats/stat-ribbon";
import { formatBytes, formatCost, formatNumber } from "@repowise-dev/ui/lib/format";
import {
  getMakeBuilds,
  getMakeDoctor,
  getMakeEvidence,
  getMakeMedia,
  getMakeMeetings,
  getMakeOperationMix,
  getMakeOperations,
  getMakeSpend,
  getMakeStorage,
  getMakeSummary,
  MakeReportError,
  microsToUsd,
  OPERATION_STATUSES,
  operationLabel,
  type MakeDoctor,
} from "@/lib/make-platform";
import {
  actionSentence,
  BuildsPipeline,
  EvidencePipeline,
  MediaPipeline,
  MeetingsPipeline,
  NeedsAttention,
  OperationMixLines,
  RecoverySection,
  SpendSection,
  StorageSection,
} from "@/components/make/make-sections";
import { LiveRefresh } from "@/components/make/live-refresh";
import { OperationsTable } from "@/components/make/operations-table";

export const metadata: Metadata = { title: "Make platform" };
export const dynamic = "force-dynamic";

/** Days of spend the chart covers, counted in Jon's time zone. */
const SPEND_DAYS = 30;
/** Days the operation mix covers. */
const MIX_DAYS = 7;
const TIME_ZONE = process.env.MAKE_REPORT_TIME_ZONE || "America/Los_Angeles";
/** Where Jon gives verdicts on renders: SoleMD.Web's review library. */
const REVIEW_LIBRARY_URL = process.env.MAKE_REVIEW_LIBRARY_URL || "https://solemd-web.taild0afc1.ts.net/make";

const value = <T,>(result: PromiseSettledResult<T>): T | null =>
  result.status === "fulfilled" ? result.value : null;

function ledeBand(doctor: MakeDoctor) {
  if (doctor.counts.fail) return { label: "failing", color: "var(--color-error)" };
  if (doctor.counts.warn) return { label: "warnings", color: "var(--color-warning)" };
  return { label: "all clear", color: "var(--color-success)" };
}

/**
 * The Make platform: what Make built, spent and kept, and what needs Jon.
 *
 * Read-only by contract (SoleMD.Make native-catalog-cutover): every figure is
 * a make-api report over its versioned views, reached through a private socket
 * from this server render. Nothing here changes Make state; the actions a
 * finding names are commands to run in Make, or the review library in Web.
 * The page re-renders when Make's catalog revision moves (LiveRefresh).
 */
export default async function MakePlatformPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const query = await searchParams;
  const status =
    typeof query.status === "string" && (OPERATION_STATUSES as readonly string[]).includes(query.status)
      ? query.status
      : undefined;
  const cursor = typeof query.cursor === "string" ? query.cursor : undefined;

  const results = await Promise.allSettled([
    getMakeSummary(),
    getMakeDoctor(),
    getMakeSpend(SPEND_DAYS, TIME_ZONE),
    getMakeBuilds({ limit: 100 }),
    getMakeEvidence(),
    getMakeMedia(),
    getMakeMeetings(),
    getMakeStorage(),
    getMakeOperations({ ...(cursor ? { cursor } : {}), ...(status ? { status } : {}) }),
    getMakeOperationMix(MIX_DAYS),
  ] as const);
  const [summary, doctor, spend, builds, evidence, media, meetings, storage] = [
    value(results[0]),
    value(results[1]),
    value(results[2]),
    value(results[3]),
    value(results[4]),
    value(results[5]),
    value(results[6]),
    value(results[7]),
  ];
  const operations = results[8];
  const mix = value(results[9]);
  const page = value(operations);
  const reachable = [summary, doctor, spend, builds, evidence, media, meetings, storage].some(Boolean);

  const nextParams = new URLSearchParams();
  if (page?.next_cursor) nextParams.set("cursor", page.next_cursor);
  if (status) nextParams.set("status", status);

  const open = doctor ? doctor.counts.fail + doctor.counts.warn : 0;
  const worst = doctor?.findings.find((f) => f.severity === "fail") ?? doctor?.findings.find((f) => f.severity === "warn");
  const spendTotal = spend ? microsToUsd(spend.total_micros) : null;
  const driveBytes = storage?.tiers.find((t) => t.tier === "drive")?.bytes;
  const buildDocs = builds?.topics.reduce((n, t) => n + t.documents, 0);

  const ribbon: RibbonStat[] = [
    {
      label: `Spend, ${SPEND_DAYS} days`,
      value: spendTotal !== null ? formatCost(spendTotal) : "",
      sub: summary ? `${formatCost(microsToUsd(summary.spend_24h_micros))} in the last 24 hours` : undefined,
      hint: "Provider cost Make recorded on each paid attempt: image and video generation, transcription.",
    },
    {
      label: "Images to review",
      value: media ? formatNumber(media.queue.pending) : "",
      sub: media ? `${formatNumber(media.queue.older)} waiting over a week` : undefined,
    },
    {
      label: "Claims awaiting review",
      value: evidence ? formatNumber(evidence.claims.review_queue.claims) : "",
      sub: evidence ? `of ${formatNumber(evidence.claims.total)} claims` : undefined,
      hint: "Claims with a quote no independent reviewer has checked. A reviewer who drafted the claim, or runs its drafter's model, does not count.",
    },
    {
      label: "Documents built",
      value: buildDocs !== undefined ? formatNumber(buildDocs) : "",
      sub: summary ? `${formatNumber(summary.evidence.works ?? 0)} papers in the catalog` : undefined,
    },
    {
      label: "On Drive",
      value: driveBytes !== undefined ? formatBytes(driveBytes) : "",
      sub: storage ? `${formatBytes(storage.retention.expiring.bytes)} expiring` : undefined,
      hint: "Bytes Make has catalogued on the Drive tier that are still there. Expiring bytes go after their grace period unless something uses them first.",
    },
  ];

  return (
    <PageShell
      title="Make platform"
      icon={<Activity className="h-5 w-5 text-[var(--color-text-tertiary)]" />}
      description="What Make built, spent and kept, and what needs you. Read-only: every figure is a make-api report."
      actions={<LiveRefresh initialRevision={summary?.catalog_revision ?? null} />}
    >
      {!reachable ? (
        <ApiError
          title="Make reporting is unavailable"
          message="make-api did not answer on its private socket, so no count, spend or recovery status can be verified. Repository reporting is unaffected."
        />
      ) : doctor ? (
        <PageLede
          label="Needs attention"
          labelHint="Findings from make doctor's catalog and background-job checks, the same ones the command prints."
          value={formatNumber(open)}
          unit={`of ${formatNumber(doctor.findings.length)} checks`}
          band={ledeBand(doctor)}
          layout="beside"
        >
          <p>
            {worst
              ? `${worst.check}: ${worst.detail}. ${actionSentence(worst.action)}`
              : "Every catalog and background-job check passes."}
          </p>
          <p>
            {summary
              ? `The catalog holds ${formatNumber(summary.objects.count)} files (${formatBytes(summary.objects.bytes)}) and ${formatNumber(summary.evidence.claims ?? 0)} claims. This page redraws itself whenever Make commits a change.`
              : "The catalog summary did not answer; the sections below say which reports did."}
          </p>
        </PageLede>
      ) : (
        <ApiError title="The doctor's findings are unavailable" message="make-api answered other reports but not /v1/report/doctor." />
      )}

      {reachable && <StatRibbon stats={ribbon} />}

      <OverviewSection
        title="Needs attention"
        description="Failures first, then warnings. Each names the command or the place that clears it; nothing here acts on Make."
      >
        <NeedsAttention doctor={doctor} />
      </OverviewSection>

      <OverviewSection
        title="Pipelines"
        description="Where each pipeline stands: evidence toward reviewed claims, documents through their builds, renders toward a verdict, recordings toward a filed note."
      >
        <div className="grid gap-x-12 gap-y-10 lg:grid-cols-2">
          <EvidencePipeline evidence={evidence} />
          <BuildsPipeline builds={builds} />
          <MediaPipeline media={media} libraryUrl={REVIEW_LIBRARY_URL} />
          <MeetingsPipeline meetings={meetings} />
        </div>
      </OverviewSection>

      <OverviewSection
        title="Spend"
        description={`Provider cost recorded per day over ${SPEND_DAYS} days, in ${TIME_ZONE.replace("_", " ")} time; a day with no paid call reads as zero.`}
      >
        <SpendSection spend={spend} />
      </OverviewSection>

      <OverviewSection
        title="Storage and retention"
        description="Bytes Make has catalogued, by what owns them. Unused bytes expire after a grace period; a class's info names who writes it, who reads it and when it goes."
      >
        <StorageSection storage={storage} />
      </OverviewSection>

      <OverviewSection
        title="Recovery"
        description="The nightly catalog dump is the recovery point; the background jobs report failure only through systemd."
      >
        <RecoverySection summary={summary} doctor={doctor} />
      </OverviewSection>

      <OverviewSection
        id="operations"
        title="Operations"
        description="Units of work Make recorded, newest first: builds, generations, transcriptions, retention passes."
      >
        <OperationMixLines mix={mix} />
        <nav aria-label="Filter operations by status" className="flex flex-wrap gap-x-1 gap-y-1 text-xs">
          {[undefined, ...OPERATION_STATUSES].map((s) => {
            const active = s === status;
            return (
              <Link
                key={s ?? "all"}
                href={s ? `/make-platform?status=${s}#operations` : "/make-platform#operations"}
                aria-current={active ? "page" : undefined}
                className={`inline-flex min-h-11 items-center rounded-md px-3 ${
                  active
                    ? "bg-[var(--color-accent-muted)] font-medium text-[var(--color-accent-primary)]"
                    : "text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
                }`}
              >
                {s ? s.replaceAll("_", " ") : "all"}
              </Link>
            );
          })}
        </nav>
        {!page ? (
          <p role="status" className="text-sm text-[var(--color-text-secondary)]">
            {operations.status === "rejected" && operations.reason instanceof MakeReportError && operations.reason.status === 409
              ? "The catalog changed under this list. Open the first page again."
              : "Operation history is unavailable."}
          </p>
        ) : (
          <>
            <OperationsTable
              rows={page.items.map((op) => ({
                id: op.id,
                label: operationLabel(op.kind),
                kind: op.kind,
                status: op.status,
                createdAt: op.created_at,
                attempt: op.attempt_no,
                provider: op.provider,
                errorCode: op.error_code,
              }))}
              empty={status ? `No ${status.replaceAll("_", " ")} operations.` : "No operations recorded."}
            />
            {page.next_cursor && (
              <Link
                href={`/make-platform?${nextParams}#operations`}
                className="inline-flex min-h-11 w-fit items-center gap-1 text-sm font-medium text-[var(--color-accent-primary)] hover:underline"
              >
                Older operations <span aria-hidden>→</span>
              </Link>
            )}
          </>
        )}
      </OverviewSection>
    </PageShell>
  );
}
