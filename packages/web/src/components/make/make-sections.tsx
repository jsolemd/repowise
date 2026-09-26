/**
 * The Make platform page's sections. Server components: plain data in, markup
 * out. The page owns fetching; each section owns one subject's reading.
 */
import Link from "next/link";
import { ReadsColumn, SectionLink, SeverityRows, type ReadItem, type SeverityRowItem } from "@repowise-dev/ui/overview";
import { InfoTip } from "@repowise-dev/ui/shared/info-tip";
import { DailySpendChart } from "@repowise-dev/ui/costs/daily-spend-chart";
import {
  formatAgeDays,
  formatBytes,
  formatCost,
  formatNumber,
  formatRelativeTime,
  formatRelativeTimeOrNull,
  stripMarkdown,
} from "@repowise-dev/ui/lib/format";
import {
  microsToUsd,
  operationLabel,
  type MakeBuilds,
  type MakeDoctor,
  type MakeEvidence,
  type MakeFinding,
  type MakeMedia,
  type MakeMeetings,
  type MakeOperationMix,
  type MakeRetention,
  type MakeSpend,
  type MakeStorage,
  type MakeSummary,
} from "@/lib/make-platform";

const SEVERITY: Record<MakeFinding["severity"], string> = { fail: "high", warn: "medium", ok: "low" };
const AREA_LABEL: Record<string, string> = {
  catalog: "Catalog",
  media: "Images",
  meetings: "Meetings",
  evidence: "Evidence",
  recovery: "Recovery",
  storage: "Storage",
  operations: "Operations",
  jobs: "Background job",
};

function Unavailable({ what }: { what: string }) {
  return (
    <p role="status" className="text-xs text-[var(--color-text-tertiary)]">
      {what} is unavailable from make-api right now; nothing here is a guess.
    </p>
  );
}

function SubHeading({ id, children }: { id?: string; children: React.ReactNode }) {
  return (
    <h3 id={id} className="scroll-mt-24 text-sm font-semibold text-[var(--color-text-primary)]">
      {children}
    </h3>
  );
}

/** Label and value lines for a short list inside a pipeline column. */
function Lines({ rows }: { rows: { key: string; label: React.ReactNode; value: React.ReactNode }[] }) {
  if (rows.length === 0) return null;
  return (
    <ul className="m-0 list-none divide-y divide-[var(--color-border-default)] border-t border-[var(--color-border-default)] p-0 text-xs">
      {rows.map((row) => (
        <li key={row.key} className="flex items-baseline justify-between gap-3 py-2">
          <span className="min-w-0 text-[var(--color-text-secondary)] [overflow-wrap:anywhere]">{row.label}</span>
          <span className="shrink-0 text-right tabular-nums text-[var(--color-text-primary)]">{row.value}</span>
        </li>
      ))}
    </ul>
  );
}

// --- Needs attention ------------------------------------------------------------

/** A finding's action as a sentence: Make writes commands in backticks. */
export function actionSentence(action: string | null): string {
  if (!action) return "";
  const text = stripMarkdown(action);
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
}

export function findingRows(findings: MakeFinding[]): SeverityRowItem[] {
  return findings.map((f, i) => ({
    id: `${f.area}-${f.check}-${i}`,
    severity: SEVERITY[f.severity],
    label: AREA_LABEL[f.area] ?? f.area,
    title: `${f.check}: ${f.detail}`,
    description: actionSentence(f.action),
  }));
}

export function NeedsAttention({ doctor }: { doctor: MakeDoctor | null }) {
  if (!doctor) return <Unavailable what="The doctor's findings" />;
  const open = doctor.findings
    .filter((f) => f.severity !== "ok")
    .sort((a, b) => (a.severity === b.severity ? 0 : a.severity === "fail" ? -1 : 1));
  const passing = doctor.findings.filter((f) => f.severity === "ok");
  return (
    <>
      <SeverityRows items={findingRows(open)} LinkComponent={Link} emptyText="Every check passes." />
      {passing.length > 0 && (
        <details className="group text-xs">
          <summary className="inline-flex min-h-11 cursor-pointer list-none items-center gap-1.5 rounded text-[var(--color-text-tertiary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-primary)]">
            <span aria-hidden className="transition-transform group-open:rotate-90">▸</span>
            {passing.length} checks pass
          </summary>
          <ul className="m-0 mt-1 list-none space-y-1 p-0 text-[var(--color-text-tertiary)]">
            {passing.map((f) => (
              <li key={f.check}>
                <span className="text-[var(--color-text-secondary)]">{f.check}</span>: {f.detail}
              </li>
            ))}
          </ul>
        </details>
      )}
    </>
  );
}

// --- Pipelines ----------------------------------------------------------------------

const STAGE_LABEL: Record<string, string> = {
  scoped: "scoped",
  searching: "searching",
  screening: "screening abstracts",
  fulltext: "reading full texts",
};

export function EvidencePipeline({ evidence }: { evidence: MakeEvidence | null }) {
  if (!evidence) return <Unavailable what="Evidence" />;
  const { claims, runs } = evidence;
  const independent = claims.quotes ? claims.quotes_independent / claims.quotes : 0;
  const items: ReadItem[] = [
    {
      key: "queue",
      label: "Independent review queue",
      value: formatNumber(claims.review_queue.claims),
      unit: `of ${formatNumber(claims.total)} claims`,
      why: `${formatNumber(claims.quotes_independent)} of ${formatNumber(claims.quotes)} quotes carry a review by someone other than their drafter; Codex empties this queue through the handoff.`,
      href: "#evidence",
      bar: [
        { fraction: independent, color: "var(--color-accent-primary)", title: "Independently reviewed quotes" },
        { fraction: 1 - independent, color: "var(--color-accent-muted)", title: "Quotes awaiting independent review" },
      ],
    },
    {
      key: "runs",
      label: "Research runs",
      value: formatNumber(runs.total),
      unit: Object.entries(claims.by_status).map(([s, n]) => `${formatNumber(n)} ${s}`).join(" · "),
      why: "Claims by status: proposed claims await ratification; ratified claims can ground a document.",
      href: "#evidence",
    },
  ];
  return (
    <div className="flex flex-col gap-4">
      <SubHeading id="evidence">Evidence</SubHeading>
      <ReadsColumn items={items} LinkComponent={Link} />
      <Lines
        rows={runs.recent.slice(0, 5).map((run) => ({
          key: run.run_slug,
          label: (
            <span className="inline-flex items-center gap-1">
              {run.run_slug}
              <InfoTip content={run.question} label={`Question for ${run.run_slug}`} />
            </span>
          ),
          value: `${STAGE_LABEL[run.stage] ?? run.stage} · ${formatNumber(run.abstracts_included + run.fulltexts_included)} in`,
        }))}
      />
      {evidence.quotas.map((q) => (
        <p key={q.provider} className="text-[11px] text-[var(--color-text-tertiary)]">
          {q.provider} quota on {q.day}: {formatNumber(q.spent)} of {formatNumber(q.daily_limit)} credits spent.
        </p>
      ))}
    </div>
  );
}

export function BuildsPipeline({ builds }: { builds: MakeBuilds | null }) {
  if (!builds) return <Unavailable what="The build record" />;
  const documents = builds.topics.reduce((n, t) => n + t.documents, 0);
  const failed = builds.topics.reduce((n, t) => n + t.failed, 0);
  const latest = builds.items[0];
  const items: ReadItem[] = [
    {
      key: "documents",
      label: "Documents built",
      value: formatNumber(documents),
      unit: failed ? `${formatNumber(failed)} failed last time` : "none failed last time",
      why: latest
        ? `Latest: ${latest.locator.split("/").slice(-2).join("/")} (${latest.engine}), ${formatRelativeTime(latest.built_at)}.`
        : "No build is on record yet.",
      href: "#builds",
    },
  ];
  return (
    <div className="flex flex-col gap-4">
      <SubHeading id="builds">Builds by topic</SubHeading>
      <ReadsColumn items={items} LinkComponent={Link} />
      <Lines
        rows={builds.topics.map((t) => ({
          key: t.topic ?? "untopiced",
          label: t.title ?? "No topic",
          value: `${formatNumber(t.documents)}${t.failed ? ` · ${formatNumber(t.failed)} failed` : ""} · ${formatRelativeTime(t.latest_built_at)}`,
        }))}
      />
      <p className="text-[11px] text-[var(--color-text-tertiary)]">
        Topic and status come from each project&apos;s root note, as authored.
      </p>
    </div>
  );
}

const DAY_SECONDS = 86_400;

export function MediaPipeline({ media, libraryUrl }: { media: MakeMedia | null; libraryUrl: string }) {
  if (!media) return <Unavailable what="The review library" />;
  const { queue } = media;
  const cards = Object.values(media.cards).reduce((a, b) => a + b, 0);
  const items: ReadItem[] = [
    {
      key: "queue",
      label: "Awaiting your verdict",
      value: formatNumber(queue.pending),
      ...(queue.oldest_seconds !== null
        ? { unit: `oldest waiting ${formatAgeDays(queue.oldest_seconds / DAY_SECONDS)}` }
        : {}),
      why: `${formatNumber(queue.under_a_day)} arrived today, ${formatNumber(queue.under_a_week)} this week, ${formatNumber(queue.older)} earlier; ${formatNumber(cards)} cards in the library.`,
      href: libraryUrl,
    },
  ];
  const owners = media.owners.filter((o) => o.pending || o.blocked).slice(0, 6);
  return (
    <div className="flex flex-col gap-4">
      <SubHeading id="media">Images</SubHeading>
      <ReadsColumn items={items} LinkComponent="a" />
      <Lines
        rows={owners.map((o) => ({
          key: o.owner,
          label: o.owner_title,
          value: [o.pending && `${o.pending} pending`, o.blocked && `${o.blocked} blocked`].filter(Boolean).join(" · "),
        }))}
      />
      <RetentionLine retention={media.retention} what="image bytes" />
      <SectionLink href={libraryUrl}>Open the review library</SectionLink>
    </div>
  );
}

const MEETING_STAGES: Record<string, string> = {
  validate: "Check",
  normalize: "Normalize",
  transcribe: "Transcribe",
  diarize: "Speakers",
  finalize: "Assemble",
};

export function MeetingsPipeline({ meetings }: { meetings: MakeMeetings | null }) {
  if (!meetings) return <Unavailable what="The meeting pipeline" />;
  const r = meetings.recordings;
  const items: ReadItem[] = [
    {
      key: "awaiting",
      label: "Transcripts awaiting a note",
      value: formatNumber(r.awaiting_summary),
      unit: `of ${formatNumber(r.total)} recordings`,
      why: `${formatNumber(r.summarized)} filed, ${formatNumber(r.in_pipeline)} still moving through the stages. Last recording admitted ${formatRelativeTimeOrNull(meetings.last_admitted_at, "never")}.`,
      href: "#meetings",
    },
  ];
  return (
    <div className="flex flex-col gap-4">
      <SubHeading id="meetings">Meetings</SubHeading>
      <ReadsColumn items={items} LinkComponent={Link} />
      <Lines
        rows={meetings.stages.map((s) => ({
          key: s.stage,
          label: MEETING_STAGES[s.stage] ?? s.stage,
          value:
            Object.entries(s.statuses)
              .map(([status, n]) => `${formatNumber(n)} ${status.replaceAll("_", " ")}`)
              .join(" · ") || "none on record",
        }))}
      />
      {meetings.stuck.length > 0 && (
        <p role="status" className="text-xs text-[var(--color-warning)]">
          {meetings.stuck.length} stages have not moved in {Math.round(meetings.stuck_after_seconds / 3600)} hours.
        </p>
      )}
      <p className="text-[11px] text-[var(--color-text-tertiary)]">
        {meetings.poll
          ? `The phone poll last ran ${formatRelativeTimeOrNull(meetings.poll.last_finished_at, "not yet")} (${meetings.poll.failed ? "failed" : meetings.poll.result}).`
          : "systemd has no record of the phone poll."}
      </p>
    </div>
  );
}

// --- Spend ------------------------------------------------------------------------

export function SpendSection({ spend }: { spend: MakeSpend | null }) {
  if (!spend) return <Unavailable what="Spend" />;
  const groups = spend.daily.map((d) => ({ group: d.day, cost_usd: microsToUsd(d.micros) }));
  return (
    <div className="flex flex-col gap-5">
      <DailySpendChart groups={groups} height={180} />
      <div className="grid gap-6 sm:grid-cols-2">
        <div className="flex flex-col gap-2">
          <SubHeading>By operation</SubHeading>
          <Lines
            rows={spend.by_kind.map((k) => ({
              key: k.kind,
              label: operationLabel(k.kind),
              value: `${formatCost(microsToUsd(k.micros))} · ${formatNumber(k.attempts)} calls`,
            }))}
          />
          {spend.by_kind.length === 0 && <p className="text-xs text-[var(--color-text-tertiary)]">No paid calls in this window.</p>}
        </div>
        <div className="flex flex-col gap-2">
          <SubHeading>By provider</SubHeading>
          <Lines
            rows={spend.by_provider.map((p) => ({
              key: p.provider,
              label: p.provider,
              value: `${formatCost(microsToUsd(p.micros))} · ${formatNumber(p.attempts)} calls`,
            }))}
          />
        </div>
      </div>
    </div>
  );
}

// --- Storage and retention ----------------------------------------------------------

function ShareBar({ segments }: { segments: { label: string; bytes: number; color: string }[] }) {
  const total = segments.reduce((n, s) => n + s.bytes, 0);
  if (total === 0) return <p className="text-xs text-[var(--color-text-tertiary)]">Nothing stored.</p>;
  return (
    <div className="space-y-1.5">
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-[var(--color-bg-inset)]">
        {segments.map((s) =>
          s.bytes > 0 ? (
            <div
              key={s.label}
              title={`${s.label}: ${formatBytes(s.bytes)}`}
              style={{ width: `${(s.bytes / total) * 100}%`, background: s.color }}
            />
          ) : null,
        )}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-[var(--color-text-tertiary)]">
        {segments.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1 tabular-nums">
            <span aria-hidden className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: s.color }} />
            {formatBytes(s.bytes)} {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function RetentionLine({ retention, what }: { retention: MakeRetention; what: string }) {
  const next = retention.expiring.by_day[0];
  return (
    <p className="text-[11px] text-[var(--color-text-tertiary)]">
      Retention: {formatBytes(retention.expiring.bytes)} of unused {what} expire after their{" "}
      {retention.grace_days}-day grace
      {next ? `, the first on ${next.day}` : ""}; {formatBytes(retention.reclaimed.bytes)} reclaimed in{" "}
      {retention.reclaimed.days} days.
    </p>
  );
}

const CLASS_LABEL: Record<string, string> = {
  "document-output": "Document outputs",
  "media-render": "Image and video renders",
  "media-receipt": "Generation receipts",
  "media-check": "Render checks",
  "website-export": "Website exports",
  "operation-receipt": "Operation receipts",
  "operation-file": "Operation files",
  fulltext: "Paper full texts",
  "extracted-text": "Extracted texts",
  "claim-source": "Claim sources",
  "search-export": "Search exports",
  "pubtator-annotation": "PubTator annotations",
  "meeting-context": "Meeting context",
  "meeting-spool": "Meeting audio and stages",
  "evidence-cache": "Evidence cache",
  "graph-publish": "Graph publications",
  unowned: "Named by no row",
};

export function StorageSection({ storage }: { storage: MakeStorage | null }) {
  if (!storage) return <Unavailable what="Storage" />;
  const available = storage.classes;
  const kept = available.reduce((n, c) => n + c.kept_bytes, 0);
  const expiring = available.reduce((n, c) => n + c.expiring_bytes, 0);
  const total = storage.tiers.reduce((n, t) => n + t.bytes, 0);
  const byClass = new Map<string, { bytes: number; files: number; expiring: number }>();
  for (const c of available) {
    const entry = byClass.get(c.file_class) ?? { bytes: 0, files: 0, expiring: 0 };
    entry.bytes += c.bytes;
    entry.files += c.files;
    entry.expiring += c.expiring_bytes;
    byClass.set(c.file_class, entry);
  }
  const classes = [...byClass.entries()].filter(([, c]) => c.files > 0).sort((a, b) => b[1].bytes - a[1].bytes);
  const last = storage.last_retention;
  return (
    <div className="flex flex-col gap-5">
      <ShareBar
        segments={[
          { label: "in use", bytes: Math.max(0, total - kept - expiring), color: "var(--color-accent-secondary)" },
          { label: "held on purpose", bytes: kept, color: "var(--color-text-tertiary)" },
          { label: "expiring", bytes: expiring, color: "var(--color-warning)" },
        ]}
      />
      <div className="grid gap-6 sm:grid-cols-2">
        <div className="flex flex-col gap-2">
          <SubHeading>By tier</SubHeading>
          <Lines
            rows={storage.tiers.map((t) => ({
              key: t.tier,
              label: t.tier,
              value: `${formatBytes(t.bytes)} · ${formatNumber(t.files)} files`,
            }))}
          />
        </div>
        <div className="flex flex-col gap-2">
          <SubHeading>By file class</SubHeading>
          <Lines
            rows={classes.map(([name, c]) => ({
              key: name,
              label: (
                <span className="inline-flex items-center gap-1">
                  {CLASS_LABEL[name] ?? name}
                  {storage.ledger[name] && (
                    <InfoTip content={storage.ledger[name]} label={`Who writes, reads and expires ${CLASS_LABEL[name] ?? name}`} />
                  )}
                </span>
              ),
              value: `${formatBytes(c.bytes)}${c.expiring ? ` · ${formatBytes(c.expiring)} expiring` : ""}`,
            }))}
          />
        </div>
      </div>
      <RetentionLine retention={storage.retention} what="bytes" />
      <p className="text-[11px] text-[var(--color-text-tertiary)]">
        {last
          ? `The last retention pass ${last.status === "completed" ? "finished" : `is ${last.status}`} ${formatRelativeTimeOrNull(last.finished_at ?? last.started_at, "recently")}${last.deleted ? ` and removed ${formatNumber(last.deleted)} files (${formatBytes(last.deleted_bytes ?? 0)})` : ""}.`
          : "No retention pass is on record."}{" "}
        {storage.unregistered
          ? `It found ${formatNumber(storage.unregistered.files)} files on the watched roots that no row names.`
          : "Unregistered files are counted from the next retention pass on."}
      </p>
    </div>
  );
}

// --- Recovery ---------------------------------------------------------------------

export function RecoverySection({ summary, doctor }: { summary: MakeSummary | null; doctor: MakeDoctor | null }) {
  const dump = summary?.backup_health[0];
  const jobs = doctor?.findings.filter((f) => f.area === "jobs") ?? [];
  if (!dump && jobs.length === 0) return <Unavailable what="Recovery status" />;
  return (
    <Lines
      rows={[
        ...(dump
          ? [{
              key: "dump",
              label: "Daily catalog dump on Drive",
              value: dump.healthy
                ? `verified ${formatRelativeTimeOrNull(dump.last_success_at, "never")}${dump.bytes_total ? ` · ${formatBytes(dump.bytes_total)}` : ""}`
                : <span className="text-[var(--color-error)]">{dump.state}</span>,
            }]
          : []),
        ...jobs.map((j) => ({
          key: j.check,
          label: j.check.replace(/^solemd-/, ""),
          value: j.severity === "ok" ? "last run succeeded" : <span className="text-[var(--color-error)]">{j.detail}</span>,
        })),
      ]}
    />
  );
}

// --- Operation mix ----------------------------------------------------------------

function seconds(value: number): string {
  if (value < 90) return `${Math.round(value)} s`;
  if (value < 5400) return `${Math.round(value / 60)} min`;
  return `${(value / 3600).toFixed(1)} h`;
}

export function OperationMixLines({ mix }: { mix: MakeOperationMix | null }) {
  if (!mix) return <Unavailable what="The operation mix" />;
  const durations = new Map(mix.durations.map((d) => [d.kind, d]));
  return (
    <div className="flex flex-col gap-2">
      <SubHeading>Last {mix.days} days by kind</SubHeading>
      <Lines
        rows={mix.by_kind.slice(0, 8).map((k) => {
          const failed = (k.statuses.failed ?? 0) + (k.statuses.ambiguous ?? 0);
          const took = durations.get(k.kind);
          return {
            key: k.kind,
            label: operationLabel(k.kind),
            value: [
              formatNumber(k.total),
              failed ? `${formatNumber(failed)} failed` : null,
              took ? `median ${seconds(took.median_seconds)}` : null,
            ].filter(Boolean).join(" · "),
          };
        })}
      />
      {mix.by_kind.length === 0 && (
        <p className="text-xs text-[var(--color-text-tertiary)]">Nothing ran in this window.</p>
      )}
      {mix.failures.length > 0 && (
        <p className="text-[11px] text-[var(--color-text-tertiary)]">
          Failures by code:{" "}
          {mix.failures.map((f) => `${operationLabel(f.kind)} ${f.error_code} (${formatNumber(f.count)})`).join("; ")}.
        </p>
      )}
    </div>
  );
}
