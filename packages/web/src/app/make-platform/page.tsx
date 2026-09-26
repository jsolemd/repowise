import type { Metadata } from "next";
import Link from "next/link";
import { Activity } from "lucide-react";
import { PageShell } from "@repowise-dev/ui/shared/page-shell";
import { StatRibbon } from "@repowise-dev/ui/stats/stat-ribbon";
import { getMakeOperations, getMakeSummary, MakeReportError } from "@/lib/make-platform";

export const metadata: Metadata = { title: "Make platform" };
export const dynamic = "force-dynamic";

const timestamp = (value: string | null) => value ? new Date(value).toLocaleString("en-US", { timeZone: "America/Los_Angeles", timeZoneName: "short" }) : "Not yet verified";
const statuses = ["queued", "running", "dispatch_started", "ambiguous", "completed", "failed", "cancelled"];
const readable = (value: string) => value.replaceAll("_", " ");

export default async function MakePlatformPage({ searchParams }: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const query = await searchParams;
  const status = typeof query.status === "string" && statuses.includes(query.status) ? query.status : undefined;
  const cursor = typeof query.cursor === "string" ? query.cursor : undefined;
  const [summary, operations] = await Promise.allSettled([getMakeSummary(), getMakeOperations({ cursor, status })]);
  const report = summary.status === "fulfilled" ? summary.value : null;
  const page = operations.status === "fulfilled" ? operations.value : null;
  const nextParams = new URLSearchParams();
  if (page?.next_cursor) nextParams.set("cursor", page.next_cursor);
  if (status) nextParams.set("status", status);
  return (
    <PageShell title="Make platform" icon={<Activity className="h-5 w-5" />} description="Images, evidence, builds, recordings and the daily catalog dump, reported by Make.">
      <nav aria-label="Make links" className="flex flex-wrap gap-5 text-sm">
        <a className="inline-flex min-h-11 items-center underline underline-offset-4" href="https://solemd-web.taild0afc1.ts.net/make">Open review library</a>
        <Link className="inline-flex min-h-11 items-center underline underline-offset-4" href="/make-platform">Refresh report</Link>
      </nav>
      {!report ? <p role="status">Make reporting is unavailable. No current counts or backup status can be verified.</p> : <>
        <p className="text-sm text-[var(--color-text-secondary)]">Catalog revision {report.catalog_revision} · {timestamp(report.as_of)} · {report.mode}</p>
        <StatRibbon stats={[
          { label: "Papers", value: report.evidence.works.toLocaleString(), sub: `${report.evidence.citation_edges.toLocaleString()} citation connections` },
          { label: "Claims", value: report.evidence.claims.toLocaleString(), sub: `${report.evidence.pending_claim_edits} pending edits` },
          { label: "Image review", value: String(report.media.pending), sub: `${report.media.accepted} accepted current images` },
          { label: "Recordings", value: String(report.meetings.recordings), sub: `${report.meetings.awaiting_summary} awaiting summary` },
        ]} />
        <section className="space-y-3" aria-labelledby="make-attention">
          <h2 id="make-attention" className="text-lg font-medium">Needs attention</h2>
          <ul className="space-y-2 text-sm">
            <li>{report.operations.ambiguous ?? 0} operations interrupted after dispatch. Provider reconciliation is required before retrying.</li>
            <li>{report.media.stale ?? 0} selected images have changed source revisions.</li>
            <li>{report.evidence.pending_claim_edits} claim edits await reconciliation with committed evidence.</li>
            <li>{report.objects.unavailable} catalog objects have no available location.</li>
            <li>{report.media.unresolved_verdicts} historical verdicts reference unavailable image bytes.</li>
          </ul>
        </section>
        <section className="space-y-3" aria-labelledby="make-storage">
          <h2 id="make-storage" className="text-lg font-medium">Storage and cost</h2>
          <p className="text-sm">{report.objects.count.toLocaleString()} immutable objects · {(report.objects.bytes / 1024 ** 3).toFixed(2)} GiB · {report.releases} pinned releases</p>
          <p className="text-sm">Recorded cost in the last 24 hours: {report.costs_24h.length ? report.costs_24h.map((c) => `${(c.amount_micros / 1_000_000).toFixed(2)} ${c.currency}`).join(" · ") : "No charges recorded"}.</p>
        </section>
        <section className="space-y-3" aria-labelledby="make-recovery">
          <h2 id="make-recovery" className="text-lg font-medium">Recovery</h2>
          {report.backup_health.filter((check) => !check.healthy).map((check) => <p key={check.stream} role="status" className="text-sm">{readable(check.stream)}: {readable(check.state)}</p>)}
          <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
            <dt>Daily catalog dump on Drive</dt>
            <dd>{timestamp(report.backup_health[0]?.last_success_at ?? null)}{report.backup_health[0]?.bytes_total ? ` · ${(report.backup_health[0].bytes_total / 1024 ** 2).toFixed(1)} MiB` : ""}</dd>
          </dl>
        </section>
      </>}
      <section className="space-y-4" aria-labelledby="make-operations">
        <h2 id="make-operations" className="text-lg font-medium">Operations</h2>
        <form method="get" className="flex flex-wrap items-center gap-3 text-sm">
          <label htmlFor="make-operation-status">Status</label>
          <select id="make-operation-status" name="status" defaultValue={status ?? ""} className="min-h-11 rounded border border-[var(--color-border)] bg-[var(--color-surface)] px-3">
            <option value="">All statuses</option>{statuses.map((s) => <option key={s} value={s}>{readable(s)}</option>)}
          </select>
          <button className="min-h-11 rounded border border-[var(--color-border)] px-4" type="submit">Filter</button>
        </form>
        {!page ? <p role="status">{operations.status === "rejected" && operations.reason instanceof MakeReportError && operations.reason.status === 409 ? "The catalog changed. Refresh this list to continue." : "Operation history is unavailable."}</p> : <>
          <p className="text-sm text-[var(--color-text-secondary)]">Up to 50 operations, newest first. Catalog revision {page.catalog_revision}.</p>
          {page.items.length === 0 ? <p>No operations match this status.</p> : <ol className="divide-y divide-[var(--color-border)]">
            {page.items.map((op) => <li key={op.id} className="grid gap-2 py-4 text-sm sm:grid-cols-[1fr_1fr]">
              <div><p className="font-medium">{op.kind} · {readable(op.status)}</p><p className="break-all font-mono text-xs text-[var(--color-text-secondary)]">{op.id}</p></div>
              <div><p>{timestamp(op.created_at)}</p><p>{op.attempt_no ? `Attempt ${op.attempt_no}` : "No attempt started"}{op.provider ? ` · ${op.provider}` : ""}{op.error_code ? ` · ${op.error_code}` : ""}</p></div>
            </li>)}
          </ol>}
          {page.next_cursor && <Link href={`/make-platform?${nextParams}`} className="inline-flex min-h-11 items-center underline underline-offset-4">Older operations</Link>}
        </>}
      </section>
    </PageShell>
  );
}
