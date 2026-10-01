/**
 * The Make platform page's sections. Server components: plain data in, markup
 * out. The page owns fetching; each section owns one subject.
 *
 * The page answers four questions at a glance, in the order Jon needs them: what
 * waits on his hand, what Make built, what it spent, and what it keeps. Each
 * section is a figure and one mark, labelled only where a mark cannot carry the
 * meaning. The doctor's health leads as a strip of one cell per check; its open
 * findings, the one place prose remains, open from that strip by touch or
 * keyboard. Agents read make-api and act on the findings themselves.
 */
import type { ReactNode } from "react";
import { BookOpen, Clock, FileStack, Images, Network, NotebookPen, RefreshCw, Stethoscope, type LucideIcon } from "lucide-react";
import { formatBytes, formatCost, formatNumber, formatRelativeTimeOrNull, stripMarkdown } from "@repowise-dev/ui/lib/format";
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
  type MakeSpend,
  type MakeStorage,
} from "@/lib/make-platform";
import { BarRows, DayColumns, FILL, Figure, Label, Legend, RAMP, StackBar, Tile, type Segment } from "./make-charts";

const DAY_SECONDS = 86_400;

function Unavailable({ what }: { what: string }) {
  return (
    <p role="status" className="m-0 text-xs text-[var(--color-text-tertiary)]">
      {what} · no answer
    </p>
  );
}

function Panel({ label, aside, children }: { label: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="flex h-4 items-center justify-between gap-3">
        <Label>{label}</Label>
        {aside && <span className="text-[11px] tabular-nums text-[var(--color-text-tertiary)]">{aside}</span>}
      </div>
      {children}
    </div>
  );
}

/** A small icon and a short time, with the words a screen reader needs. */
function Stamp({ icon: Icon, said, children, color }: { icon: LucideIcon; said: string; children: ReactNode; color?: string }) {
  return (
    <span className="inline-flex items-center gap-1" style={color ? { color } : undefined}>
      <Icon aria-hidden className="h-3 w-3" />
      <span className="sr-only">{said} </span>
      {children}
    </span>
  );
}

/** How long something has waited, in its largest whole unit: "5h", "2d", "3w". */
export function compactAge(seconds: number): string {
  const hours = seconds / 3600;
  if (hours < 1) return "<1h";
  if (hours < 24) return `${Math.floor(hours)}h`;
  const days = hours / 24;
  return days < 14 ? `${Math.floor(days)}d` : `${Math.floor(days / 7)}w`;
}

const plural = (n: number, noun: string) => `${noun}${n === 1 ? "" : "s"}`;

// --- Doctor -------------------------------------------------------------------------

const AREA_LABEL: Record<string, string> = {
  catalog: "Catalog",
  media: "Images",
  meetings: "Meetings",
  evidence: "Evidence",
  recovery: "Recovery",
  storage: "Storage",
  operations: "Ops",
  jobs: "Jobs",
};

const SEVERITY_RANK: Record<MakeFinding["severity"], number> = { fail: 0, warn: 1, ok: 2 };

/** Severity as a cell, in the node fills. */
const CELL: Record<MakeFinding["severity"], string> = { fail: FILL.fail, warn: FILL.wait, ok: FILL.ok };

/** Severity as type, in the status ink vars. */
const INK: Record<MakeFinding["severity"], string> = {
  fail: "var(--color-error)",
  warn: "var(--color-warning)",
  ok: "var(--color-success)",
};

/** A finding's action as a sentence: Make writes commands in backticks. */
export function actionSentence(action: string | null): string {
  if (!action) return "";
  const text = stripMarkdown(action);
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
}

function findingTitle(f: MakeFinding): string {
  const detail = f.detail.length > 160 ? `${f.detail.slice(0, 157)}…` : f.detail;
  return [`${f.check}: ${detail}`, actionSentence(f.action)].filter(Boolean).join("\n");
}

const STRIP = "flex min-h-11 flex-wrap items-center gap-x-4 gap-y-2";

/**
 * Every make doctor check as one cell, failing first, then warning, then passing,
 * so the share of red and amber reads before any number. Each severity is its own
 * run with a gap before the next, because the at-risk and needs-work fills sit
 * close in hue and one red cell must not vanish into a run of amber. The legend
 * gives the three counts. When anything is open the whole strip is the summary of a native
 * disclosure, so one tap or keypress lists every open finding's check, detail
 * and action, and the strip stays where it was.
 */
export function DoctorStrip({ doctor }: { doctor: MakeDoctor | null }) {
  if (!doctor) {
    return (
      <div className={`${STRIP} border-b border-[var(--color-border-default)] pb-3`}>
        <Unavailable what="Doctor" />
      </div>
    );
  }
  const { fail, warn, ok } = doctor.counts;
  const checks = [...doctor.findings].sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]);
  const open = checks.filter((f) => f.severity !== "ok");
  const row = (
    <>
      <Label className="flex items-center gap-1.5">
        <Stethoscope aria-hidden className="h-3.5 w-3.5" />
        Doctor
      </Label>
      <span aria-hidden className="flex flex-wrap gap-x-2 gap-y-[3px]">
        {(["fail", "warn", "ok"] as const).map((severity) =>
          checks.some((f) => f.severity === severity) ? (
            <span key={severity} className="flex flex-wrap gap-[3px]">
              {checks
                .filter((f) => f.severity === severity)
                .map((f, i) => (
                  <span
                    key={`${f.check}-${i}`}
                    data-severity={f.severity}
                    title={findingTitle(f)}
                    className="block h-3.5 w-3.5 rounded-[3px] sm:h-4 sm:w-4"
                    style={{ background: CELL[f.severity], opacity: f.severity === "ok" ? 0.55 : 1 }}
                  />
                ))}
            </span>
          ) : null,
        )}
      </span>
      <Legend
        items={[
          { key: "fail", label: "fail", color: FILL.fail, value: formatNumber(fail) },
          { key: "warn", label: "warn", color: FILL.wait, value: formatNumber(warn) },
          { key: "ok", label: "ok", color: FILL.ok, value: formatNumber(ok) },
        ]}
      />
    </>
  );
  if (!open.length) {
    return <div className={`${STRIP} border-b border-[var(--color-border-default)] pb-3`}>{row}</div>;
  }
  return (
    <details className="group border-b border-[var(--color-border-default)] pb-3">
      <summary
        className={`${STRIP} cursor-pointer list-none rounded-sm marker:hidden focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-primary)] [&::-webkit-details-marker]:hidden`}
      >
        {row}
        <span aria-hidden className="ml-auto text-[var(--color-text-tertiary)] transition-transform group-open:rotate-90">
          ›
        </span>
      </summary>
      <ul className="m-0 mt-2 flex list-none flex-col gap-3 p-0 pb-1 text-xs">
        {open.map((f, i) => (
          <li key={`${f.check}-${i}`} className="flex flex-col gap-0.5">
            <p className="m-0">
              <span className="font-semibold" style={{ color: INK[f.severity] }}>{f.check}</span>{" "}
              <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--color-text-tertiary)]">
                {f.severity} · {AREA_LABEL[f.area] ?? f.area}
              </span>
            </p>
            <p className="m-0 text-[var(--color-text-secondary)] [overflow-wrap:anywhere]">{f.detail}</p>
            {f.action && <p className="m-0 text-[var(--color-text-primary)]">{actionSentence(f.action)}</p>}
          </li>
        ))}
      </ul>
    </details>
  );
}

/** Holds the strip's height while the doctor runs its live checks. */
export function DoctorPending() {
  return (
    <div className={`${STRIP} border-b border-[var(--color-border-default)] pb-3`} aria-busy>
      <Label className="flex items-center gap-1.5">
        <Stethoscope aria-hidden className="h-3.5 w-3.5" />
        Doctor · checking
      </Label>
    </div>
  );
}

// --- Needs you ----------------------------------------------------------------------

/**
 * What waits on Jon's hand, one tile each, always in the same place so the row
 * is learned once: images to review (the tile opens the review library), notes
 * to file, library copies to fetch against the papers a claim needs, concepts a
 * changed claim made stale, and documents citing a changed claim. `compact` sets
 * the counts at the size of a page's stat ribbon, for the dashboard's Make card.
 */
export function NeedsYou({
  media,
  meetings,
  evidence,
  libraryUrl,
  compact = false,
}: {
  media: MakeMedia | null;
  meetings: MakeMeetings | null;
  evidence: MakeEvidence | null;
  libraryUrl: string;
  compact?: boolean;
}) {
  const oldest = media?.queue.oldest_seconds ?? null;
  const size = compact ? "md" : "xl";
  return (
    <div className="-mx-3 grid grid-cols-2 gap-x-2 gap-y-1 sm:grid-cols-3 lg:grid-cols-5">
      <div className="col-span-2 sm:col-span-1">
        <Tile
          icon={Images}
          label="Images to review"
          value={media?.queue.pending ?? null}
          href={libraryUrl}
          size={size}
          unit={
            media && media.queue.pending > 0 && oldest !== null ? (
              <Stamp icon={Clock} said="oldest waiting">{compactAge(oldest)}</Stamp>
            ) : undefined
          }
        />
      </div>
      <Tile icon={NotebookPen} label="Notes to file" value={meetings?.recordings.awaiting_summary ?? null} size={size} />
      <Tile
        icon={BookOpen}
        label="Copies to fetch"
        value={evidence?.fulltext.waiting ?? null}
        of={evidence?.fulltext.needed}
        size={size}
      />
      <Tile icon={Network} label="Stale concepts" value={evidence?.claims.stale_concepts ?? null} size={size} />
      <Tile icon={FileStack} label="Docs to rebuild" value={evidence?.claims.documents_changed ?? null} size={size} />
    </div>
  );
}

// --- Built --------------------------------------------------------------------------

/**
 * The evidence map: published claims by how clear the literature is on each, with
 * the concepts and open questions beside them. The certainty bar is the ordinal
 * exception make-charts.tsx describes: clear, uncertain and contested step down
 * the ramp in that fixed order whatever their counts. They take the first, third
 * and fourth steps: two apart for the two large shares, and short of the fifth,
 * which all but vanishes into the dark theme's inset.
 */
export function EvidencePanel({ evidence }: { evidence: MakeEvidence | null }) {
  if (!evidence) return <Unavailable what="Evidence" />;
  const { questions, claims } = evidence;
  const certainty = claims.by_certainty;
  // Make totals questions in every state (open, answered, withdrawn).
  const openQuestions = questions.by_state.open ?? 0;
  const concepts = claims.by_kind.concept ?? 0;
  return (
    <Panel label="Evidence">
      <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
        <Figure value={formatNumber(claims.by_kind.claim ?? 0)} unit="claims" />
        <Figure size="sm" value={formatNumber(concepts)} unit={plural(concepts, "concept")} />
        <Figure size="sm" value={formatNumber(openQuestions)} unit={`open ${plural(openQuestions, "question")}`} />
      </div>
      <StackBar
        segments={[
          { key: "clear", value: certainty.clear ?? 0, color: RAMP[0], label: "clear" },
          { key: "uncertain", value: certainty.uncertain ?? 0, color: RAMP[2], label: "uncertain" },
          { key: "contested", value: certainty.contested ?? 0, color: RAMP[3], label: "contested" },
        ]}
        legend
      />
    </Panel>
  );
}

/**
 * Documents by group. Built documents are settled work and recede in the neutral
 * fill; a group's failed builds take the fail fill and are counted beside it.
 */
export function BuildsPanel({ builds }: { builds: MakeBuilds | null }) {
  if (!builds) return <Unavailable what="Builds" />;
  const documents = builds.groups.reduce((n, g) => n + g.documents, 0);
  const latest = builds.items[0];
  return (
    <Panel
      label="Builds"
      aside={latest ? <Stamp icon={Clock} said="last build">{formatRelativeTimeOrNull(latest.built_at, "never")}</Stamp> : undefined}
    >
      <Figure value={formatNumber(documents)} unit={plural(documents, "document")} />
      <BarRows
        rows={builds.groups.map((g) => {
          const label = g.group ?? "No group";
          return {
            key: g.group ?? "none",
            label,
            segments: [
              { key: "built", value: g.documents - g.failed, color: FILL.done, label: "built", title: `${label}: ${g.documents - g.failed} built` },
              { key: "failed", value: g.failed, color: FILL.fail, label: "failed", title: `${label}: ${g.failed} failed` },
            ],
            note: g.failed ? <span className="ml-1.5 text-[var(--color-error)]">· {formatNumber(g.failed)} failed</span> : undefined,
          };
        })}
      />
    </Panel>
  );
}

/** Review cards by verdict: settled ones recede, the pending ones carry the wait fill. */
export function ImagesPanel({ media }: { media: MakeMedia | null }) {
  if (!media) return <Unavailable what="Images" />;
  const { cards } = media;
  return (
    <Panel label="Images">
      <Figure value={formatNumber(Object.values(cards).reduce((a, b) => a + b, 0))} unit="cards" />
      <StackBar
        segments={[
          { key: "accept", value: cards.accept ?? 0, color: FILL.done, label: "accepted" },
          { key: "pending", value: cards.pending ?? 0, color: FILL.wait, label: "pending" },
          { key: "reject", value: cards.reject ?? 0, color: FILL.faint, label: "rejected" },
        ]}
        legend
      />
    </Panel>
  );
}

const MEETING_STAGES: Record<string, string> = {
  validate: "check",
  normalize: "normalize",
  transcribe: "transcribe",
  diarize: "speakers",
  finalize: "assemble",
  speaker: "voice match",
};

/**
 * Recordings by where they stand. Make partitions them into filed, awaiting a
 * note and still in the pipeline; a failed or stuck stage holds its recording in
 * the pipeline, so those come out of that share in the fail fill.
 */
export function meetingShares(meetings: MakeMeetings): { filed: number; waiting: number; flowing: number; stuck: number } {
  const r = meetings.recordings;
  const failed = meetings.stages.reduce((n, s) => n + (s.statuses.failed ?? 0), 0);
  const stuck = Math.min(r.in_pipeline, failed + meetings.stuck.length);
  return { filed: r.summarized, waiting: r.awaiting_summary, flowing: r.in_pipeline - stuck, stuck };
}

export function MeetingsPanel({ meetings }: { meetings: MakeMeetings | null }) {
  if (!meetings) return <Unavailable what="Meetings" />;
  const { filed, waiting, flowing, stuck } = meetingShares(meetings);
  const troubled = [
    ...meetings.stages.filter((s) => s.statuses.failed).map((s) => `${MEETING_STAGES[s.stage] ?? s.stage} failed ×${s.statuses.failed}`),
    ...meetings.stuck.map((s) => `stuck at ${MEETING_STAGES[s.stage] ?? s.stage}`),
  ];
  const poll = meetings.poll;
  return (
    <Panel
      label="Meetings"
      aside={
        poll ? (
          <Stamp
            icon={RefreshCw}
            said={poll.failed ? "phone poll" : "last phone poll"}
            color={poll.failed ? "var(--color-error)" : undefined}
          >
            {poll.failed ? "failed" : formatRelativeTimeOrNull(poll.last_finished_at, "never")}
          </Stamp>
        ) : undefined
      }
    >
      <Figure value={formatNumber(meetings.recordings.total)} unit="recordings" />
      <StackBar
        segments={[
          { key: "filed", value: filed, color: FILL.done, label: "filed" },
          { key: "waiting", value: waiting, color: FILL.wait, label: "to file" },
          { key: "flowing", value: flowing, color: FILL.flight, label: "in pipeline" },
          { key: "stuck", value: stuck, color: FILL.fail, label: "stuck", title: troubled.join(", ") || "stuck" },
        ]}
        legend
      />
    </Panel>
  );
}

const FAMILY_LABEL: Record<string, string> = {
  media: "Images and video",
  meeting: "Meetings",
  build: "Builds",
  evidence: "Evidence",
  catalog: "Catalog",
  brand: "Brand",
  vault: "Vault",
};

/** The window's operations by family, split by how each ended; only failures take colour. */
export function ActivityPanel({ mix }: { mix: MakeOperationMix | null }) {
  if (!mix) return <Unavailable what="Operations" />;
  const families = new Map<string, Record<string, number>>();
  for (const k of mix.by_kind) {
    const family = k.kind.split(".")[0] ?? k.kind;
    const into = families.get(family) ?? {};
    for (const [status, n] of Object.entries(k.statuses)) into[status] = (into[status] ?? 0) + n;
    families.set(family, into);
  }
  const rows = [...families.entries()]
    .map(([family, s]) => ({ family, s, total: Object.values(s).reduce((a, b) => a + b, 0) }))
    .sort((a, b) => b.total - a.total);
  const failed = rows.reduce((n, r) => n + (r.s.failed ?? 0) + (r.s.ambiguous ?? 0), 0);
  const total = rows.reduce((n, r) => n + r.total, 0);
  const segment = (s: Record<string, number>, family: string): Segment[] => [
    { key: "completed", value: s.completed ?? 0, color: FILL.done, label: "completed" },
    { key: "running", value: (s.running ?? 0) + (s.queued ?? 0) + (s.dispatch_started ?? 0), color: FILL.flight, label: "running" },
    { key: "cancelled", value: s.cancelled ?? 0, color: FILL.faint, label: "cancelled" },
    {
      key: "failed",
      value: (s.failed ?? 0) + (s.ambiguous ?? 0),
      color: FILL.fail,
      label: "failed",
      title: `${FAMILY_LABEL[family] ?? family}: ${mix.failures
        .filter((f) => f.kind.startsWith(`${family}.`))
        .map((f) => `${operationLabel(f.kind)} ${f.error_code} ×${f.count}`)
        .join(", ") || "failed"}`,
    },
  ];
  return (
    <Panel
      label={`Operations · ${mix.days} days`}
      aside={failed ? <span className="text-[var(--color-error)]">{formatNumber(failed)} failed</span> : undefined}
    >
      <Figure value={formatNumber(total)} unit="operations" />
      <BarRows
        rows={rows.map((r) => ({ key: r.family, label: FAMILY_LABEL[r.family] ?? r.family, segments: segment(r.s, r.family) }))}
      />
      <Legend items={segment({}, "").map(({ key, label, color }) => ({ key, label, color }))} />
    </Panel>
  );
}

// --- Spent --------------------------------------------------------------------------

/**
 * The window's spend: the total, each day as a column, and the split by provider.
 * The columns are the panel's one series and take the accent; the provider split
 * supports them in the neutral steps, so no provider reads as the columns' colour.
 */
export function SpendPanel({ spend }: { spend: MakeSpend | null }) {
  if (!spend) return <Unavailable what="Spend" />;
  const usd = (micros: number) => formatCost(microsToUsd(micros));
  const providers = spend.by_provider.filter((p) => p.micros > 0);
  // Spread across the neutral steps, so two providers take the two ends.
  const neutrals = providers.length <= 2 ? [FILL.done, FILL.faint] : [FILL.done, FILL.flight, FILL.faint];
  return (
    <>
      <Figure value={usd(spend.total_micros)} unit={`${spend.days} days`} />
      <DayColumns days={spend.daily.map((d) => ({ day: d.day, value: d.micros }))} height={56} format={usd} />
      <StackBar
        height={8}
        segments={providers.map((p, i) => ({ key: p.provider, value: p.micros, color: neutrals[i] ?? FILL.faint, label: p.provider }))}
        format={usd}
        legend
      />
    </>
  );
}

// --- Kept ---------------------------------------------------------------------------

const CLASS_LABEL: Record<string, string> = {
  "document-output": "Documents",
  "media-render": "Renders",
  "media-receipt": "Receipts",
  "media-check": "Render checks",
  "website-export": "Web exports",
  "operation-receipt": "Op receipts",
  "operation-file": "Op files",
  fulltext: "Full texts",
  "extracted-text": "Extracted text",
  "claim-source": "Claim sources",
  "search-export": "Search exports",
  "pubtator-annotation": "PubTator",
  "meeting-context": "Meeting context",
  "meeting-spool": "Meeting audio",
  "evidence-cache": "Evidence cache",
  "graph-publish": "Graph",
  unowned: "Unowned",
};

/** Consecutive days from the first dated entry, zero where nothing falls due. */
function calendar(days: { day: string; bytes: number }[], span: number) {
  const first = days[0];
  if (!first) return [];
  const bytes = new Map(days.map((d) => [d.day, d.bytes]));
  const start = Date.parse(`${first.day}T00:00:00Z`);
  return Array.from({ length: span }, (_, i) => {
    const day = new Date(start + i * DAY_SECONDS * 1000).toISOString().slice(0, 10);
    return { day, bytes: bytes.get(day) ?? 0 };
  });
}

/**
 * Bytes by file class down the ramp with the expiring share hatched, then the days
 * that share expires on, drawn in the same hatch as its legend swatch.
 */
export function StoragePanel({ storage }: { storage: MakeStorage | null }) {
  if (!storage) return <Unavailable what="Storage" />;
  const byClass = new Map<string, { bytes: number; expiring: number }>();
  for (const c of storage.classes) {
    const e = byClass.get(c.file_class) ?? { bytes: 0, expiring: 0 };
    e.bytes += c.bytes;
    e.expiring += c.expiring_bytes;
    byClass.set(c.file_class, e);
  }
  const sorted = [...byClass.entries()].filter(([, c]) => c.bytes > 0).sort((a, b) => b[1].bytes - a[1].bytes);
  const top = sorted.slice(0, 4);
  const rest = sorted.slice(4).reduce((n, [, c]) => ({ bytes: n.bytes + c.bytes, expiring: n.expiring + c.expiring }), { bytes: 0, expiring: 0 });
  const segments: Segment[] = [
    ...top.map(([name, c], i) => ({
      key: name,
      value: c.bytes,
      hatched: c.expiring,
      color: RAMP[i] ?? FILL.done,
      label: CLASS_LABEL[name] ?? name,
      title: `${CLASS_LABEL[name] ?? name}: ${formatBytes(c.bytes)}${c.expiring ? `, ${formatBytes(c.expiring)} expiring` : ""}`,
    })),
    ...(rest.bytes ? [{ key: "rest", value: rest.bytes, hatched: rest.expiring, color: FILL.done, label: "Other" }] : []),
  ];
  const total = sorted.reduce((n, [, c]) => n + c.bytes, 0);
  const { expiring } = storage.retention;
  const upcoming = calendar(expiring.by_day, 28);
  return (
    <>
      <Figure value={formatBytes(total)} />
      <StackBar
        segments={segments}
        height={14}
        format={formatBytes}
        legend
        extra={[{ key: "expiring", label: "expiring", color: FILL.done, value: formatBytes(expiring.bytes), hatched: true }]}
      />
      {upcoming.length > 0 && (
        <DayColumns days={upcoming.map((d) => ({ day: d.day, value: d.bytes }))} height={40} color={FILL.done} hatched format={formatBytes} />
      )}
    </>
  );
}
