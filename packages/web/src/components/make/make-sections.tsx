/**
 * The Make platform page's sections, drawn as charts. Server components: plain
 * data in, markup out. The page owns fetching; each section owns one subject.
 *
 * The page is Jon's window onto Make, not its work queue: agents read make-api
 * and act on findings. So each section shows a shape (what is flowing, what is
 * stuck, what is growing) with labels and figures, and leaves the prose to the
 * `title` a mark carries on hover.
 */
import type { ReactNode } from "react";
import { formatAgeDays, formatBytes, formatCost, formatRelativeTimeOrNull, stripMarkdown } from "@repowise-dev/ui/lib/format";
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
import { BarRows, DayColumns, FILL, Figure, Label, RAMP, StackBar, type Segment } from "./make-charts";

function Unavailable({ what }: { what: string }) {
  return (
    <p role="status" className="text-xs text-[var(--color-text-tertiary)]">
      {what}: no answer from make-api.
    </p>
  );
}

function Panel({ label, aside, children }: { label: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="flex items-baseline justify-between gap-3">
        <Label>{label}</Label>
        {aside && <span className="text-[11px] tabular-nums text-[var(--color-text-tertiary)]">{aside}</span>}
      </div>
      {children}
    </div>
  );
}

const count = (n: number) => n.toLocaleString();

// --- Doctor -------------------------------------------------------------------------

const AREA_ORDER = ["catalog", "media", "evidence", "meetings", "operations", "storage", "recovery", "jobs"];
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

/**
 * Every make doctor check as one cell, grouped by area, with the open count as
 * the page's headline. Failing checks are named under the grid, because a cell
 * cannot be hovered on a phone and these are the ones that matter.
 */
export function DoctorGrid({ doctor }: { doctor: MakeDoctor }) {
  const { fail, warn, ok } = doctor.counts;
  const rank = (area: string) => (AREA_ORDER.includes(area) ? AREA_ORDER.indexOf(area) : AREA_ORDER.length);
  const areas = [...new Set(doctor.findings.map((f) => f.area))].sort((a, b) => rank(a) - rank(b));
  const failing = doctor.findings.filter((f) => f.severity === "fail");
  const color = fail ? "var(--color-error)" : warn ? "var(--color-warning)" : "var(--color-success)";
  return (
    <div className="flex flex-col gap-5 border-b border-[var(--color-border-default)] pb-6 sm:flex-row sm:items-end sm:gap-10">
      <div className="flex shrink-0 flex-col gap-2">
        <Label>Doctor</Label>
        <p className="m-0 flex items-baseline gap-2">
          <span className="text-[56px] font-semibold leading-none tracking-tight tabular-nums" style={{ color }}>
            {fail + warn}
          </span>
          <span className="text-sm text-[var(--color-text-tertiary)]">open of {doctor.findings.length}</span>
        </p>
        <div className="flex gap-3 text-[11px] tabular-nums text-[var(--color-text-secondary)]">
          {([["fail", fail], ["warn", warn], ["ok", ok]] as const).map(([k, n]) => (
            <span key={k} className="inline-flex items-center gap-1">
              <span aria-hidden className="inline-block h-2 w-2 rounded-[2px]" style={{ background: FILL[k] }} />
              {n} {k === "ok" ? "pass" : k}
            </span>
          ))}
        </div>
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <ul className="m-0 flex list-none flex-wrap gap-x-5 gap-y-3 p-0">
          {areas.map((area) => (
            <li key={area} className="flex flex-col gap-1.5">
              <div className="flex gap-[3px]">
                {doctor.findings
                  .filter((f) => f.area === area)
                  .map((f, i) => (
                    <span
                      key={`${f.check}-${i}`}
                      role="img"
                      aria-label={`${f.check}: ${f.severity}`}
                      title={findingTitle(f)}
                      className="block h-6 w-6 rounded-[4px]"
                      style={{ background: FILL[f.severity], opacity: f.severity === "ok" ? 0.55 : 1 }}
                    />
                  ))}
              </div>
              <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-[var(--color-text-tertiary)]">
                {AREA_LABEL[area] ?? area}
              </span>
            </li>
          ))}
        </ul>
        {failing.length > 0 && (
          <p className="m-0 text-xs text-[var(--color-error)]">
            {failing.map((f) => f.check).join(" · ")}
          </p>
        )}
      </div>
    </div>
  );
}

export function DoctorPending() {
  return (
    <div className="flex h-[124px] items-end border-b border-[var(--color-border-default)] pb-6" aria-busy>
      <Label>Doctor · checking</Label>
    </div>
  );
}

// --- Pipelines ----------------------------------------------------------------------

/**
 * The evidence map's state. Its fingerprint first: the published claims by how clear the
 * literature is on each, ordered from clear to contested down the ramp. Then what asks for
 * a hand: concepts a changed claim made stale, documents citing a claim whose sentence
 * moved since they were built, and papers whose body a claim needs that have no copy yet.
 */
export function EvidencePanel({ evidence }: { evidence: MakeEvidence | null }) {
  if (!evidence) return <Unavailable what="Evidence" />;
  const { questions, claims, fulltext } = evidence;
  const certainty = claims.by_certainty;
  const attention = [
    {
      key: "stale",
      value: claims.stale_concepts,
      label: "concepts stale",
      title: "Concepts resting on a claim that changed since they pinned it. A final build refuses them.",
      alert: true,
    },
    {
      key: "documents",
      value: claims.documents_changed,
      label: "documents to rebuild",
      title: "Documents citing a claim whose sentence changed since they were last built.",
      alert: true,
    },
    {
      key: "copies",
      value: fulltext.waiting,
      label: `of ${count(fulltext.needed)} papers await a copy`,
      title: "Papers whose body a claim needs, with no full text yet: a library copy settles them.",
      alert: false,
    },
  ];
  return (
    <Panel label="Evidence" aside={`${count(questions.total)} open questions`}>
      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col gap-2">
          <Figure value={count(claims.by_kind.claim ?? 0)} unit="claims" />
          <StackBar
            segments={[
              { key: "clear", value: certainty.clear ?? 0, color: RAMP[0], label: "clear" },
              { key: "uncertain", value: certainty.uncertain ?? 0, color: RAMP[2], label: "uncertain" },
              { key: "contested", value: certainty.contested ?? 0, color: RAMP[4], label: "contested" },
            ]}
            legend
          />
        </div>
        <div className="flex flex-col gap-2">
          <Figure value={count(claims.by_kind.concept ?? 0)} unit="concepts" />
          <ul className="m-0 flex list-none flex-col gap-1 p-0 text-xs text-[var(--color-text-secondary)]">
            {attention.map((a) => (
              <li key={a.key} title={a.title} className="tabular-nums">
                <span
                  className="font-semibold"
                  style={a.alert && a.value ? { color: FILL.warn } : { color: "var(--color-text-primary)" }}
                >
                  {count(a.value)}
                </span>{" "}
                {a.label}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </Panel>
  );
}

export function BuildsPanel({ builds }: { builds: MakeBuilds | null }) {
  if (!builds) return <Unavailable what="Builds" />;
  const documents = builds.groups.reduce((n, g) => n + g.documents, 0);
  const latest = builds.items[0];
  return (
    <Panel label="Builds by group" aside={latest ? `last ${formatRelativeTimeOrNull(latest.built_at, "never")}` : undefined}>
      <Figure value={count(documents)} unit="documents" />
      <BarRows
        rows={builds.groups.map((g) => ({
          key: g.group ?? "none",
          label: g.group ?? "No group",
          segments: [
            {
              key: "docs",
              value: g.documents,
              color: RAMP[1],
              label: "documents",
              hatched: g.failed,
              title: `${g.group ?? "No group"}: ${g.documents} documents${g.failed ? `, ${g.failed} failed` : ""}`,
            },
          ],
          note: g.failed ? <span className="ml-1 text-[var(--color-error)]">!</span> : undefined,
        }))}
      />
    </Panel>
  );
}

const DAY_SECONDS = 86_400;

export function ImagesPanel({ media, libraryUrl }: { media: MakeMedia | null; libraryUrl: string }) {
  if (!media) return <Unavailable what="Images" />;
  const { cards, queue } = media;
  const owners = media.owners.filter((o) => o.pending || o.blocked).slice(0, 5);
  return (
    <Panel
      label="Images"
      aside={
        <a href={libraryUrl} className="text-[var(--color-accent-primary)] hover:underline">
          review library →
        </a>
      }
    >
      <Figure
        value={count(Object.values(cards).reduce((a, b) => a + b, 0))}
        unit="cards"
      />
      <StackBar
        segments={[
          { key: "accept", value: cards.accept ?? 0, color: RAMP[1], label: "accepted" },
          { key: "pending", value: cards.pending ?? 0, color: FILL.warn, label: "pending" },
          { key: "reject", value: cards.reject ?? 0, color: FILL.quiet, label: "rejected" },
        ]}
        legend
      />
      {owners.length > 0 && (
        <BarRows
          rows={owners.map((o) => ({
            key: o.owner,
            label: o.owner_title,
            segments: [
              { key: "pending", value: o.pending - o.blocked, color: FILL.warn, label: "pending" },
              { key: "blocked", value: o.blocked, color: FILL.fail, label: "blocked" },
            ],
          }))}
        />
      )}
      {queue.oldest_seconds !== null && (
        <p className="m-0 text-[11px] text-[var(--color-text-tertiary)]">
          oldest waiting {formatAgeDays(queue.oldest_seconds / DAY_SECONDS)}
        </p>
      )}
    </Panel>
  );
}

const MEETING_STAGES: Record<string, string> = {
  validate: "check",
  normalize: "normalize",
  transcribe: "transcribe",
  diarize: "speakers",
  finalize: "assemble",
};

/** Recordings as a chain of stages, ending at the filed note. */
export function MeetingsPanel({ meetings }: { meetings: MakeMeetings | null }) {
  if (!meetings) return <Unavailable what="Meetings" />;
  const r = meetings.recordings;
  const stuck = new Set(meetings.stuck.map((s) => s.stage));
  const nodes = [
    ...meetings.stages.map((s) => {
      const done = s.statuses.completed ?? 0;
      const total = Object.values(s.statuses).reduce((a, b) => a + b, 0);
      const failed = (s.statuses.failed ?? 0) + (stuck.has(s.stage) ? 1 : 0);
      return { key: s.stage, label: MEETING_STAGES[s.stage] ?? s.stage, done, total, failed };
    }),
    { key: "note", label: "note filed", done: r.summarized, total: r.total, failed: 0 },
  ];
  return (
    <Panel
      label="Meetings"
      aside={meetings.poll ? `poll ${meetings.poll.failed ? "failed" : "ok"} · ${formatRelativeTimeOrNull(meetings.poll.last_finished_at, "never")}` : undefined}
    >
      <Figure value={count(r.total)} unit="recordings" />
      <ol className="m-0 flex list-none items-start p-0">
        {nodes.map((n, i) => {
          const complete = n.total > 0 && n.done === n.total;
          const color = n.failed ? FILL.fail : complete ? RAMP[1] : FILL.warn;
          return (
            <li key={n.key} className="flex min-w-0 flex-1 flex-col items-center gap-1.5">
              <div className="flex w-full items-center">
                <span className={`h-px flex-1 ${i ? "bg-[var(--color-border-hover)]" : ""}`} />
                <span
                  title={`${n.label}: ${n.done} of ${n.total}`}
                  className="grid h-7 w-7 shrink-0 place-items-center rounded-full text-[10px] font-semibold tabular-nums"
                  style={{
                    background: complete ? color : "var(--color-bg-surface)",
                    border: `2px solid ${color}`,
                    color: complete ? "var(--color-text-on-accent)" : "var(--color-text-primary)",
                  }}
                >
                  {complete ? "✓" : n.total - n.done}
                </span>
                <span className={`h-px flex-1 ${i < nodes.length - 1 ? "bg-[var(--color-border-hover)]" : ""}`} />
              </div>
              <span className="truncate text-center text-[10px] text-[var(--color-text-tertiary)]">{n.label}</span>
            </li>
          );
        })}
      </ol>
    </Panel>
  );
}

// --- Spend --------------------------------------------------------------------------

export function SpendTile({ spend }: { spend: MakeSpend | null }) {
  if (!spend) return <Unavailable what="Spend" />;
  const usd = (micros: number) => formatCost(microsToUsd(micros));
  return (
    <>
      <Figure value={usd(spend.total_micros)} unit={`${spend.days} days`} />
      <DayColumns days={spend.daily.map((d) => ({ day: d.day, value: d.micros }))} height={36} format={usd} />
      <StackBar
        height={6}
        segments={spend.by_provider
          .filter((p) => p.micros > 0)
          .map((p, i) => ({ key: p.provider, value: p.micros, color: RAMP[i * 2] ?? FILL.quiet, label: p.provider }))}
        format={usd}
        legend
      />
    </>
  );
}

// --- Activity -----------------------------------------------------------------------

const FAMILY_LABEL: Record<string, string> = {
  media: "Images and video",
  meeting: "Meetings",
  build: "Builds",
  evidence: "Evidence",
  catalog: "Catalog",
  brand: "Brand",
};

/** Seven days of operations by family, split by how each ended. */
export function ActivityPanel({ mix }: { mix: MakeOperationMix | null }) {
  if (!mix) return <Unavailable what="Activity" />;
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
    { key: "completed", value: s.completed ?? 0, color: FILL.quiet, label: "completed" },
    { key: "running", value: (s.running ?? 0) + (s.queued ?? 0) + (s.dispatch_started ?? 0), color: RAMP[0], label: "running" },
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
    <Panel label={`Operations · ${mix.days} days`} aside={failed ? <span className="text-[var(--color-error)]">{failed} failed</span> : "none failed"}>
      <Figure value={count(total)} unit="operations" />
      <BarRows
        rows={rows.map((r) => ({ key: r.family, label: FAMILY_LABEL[r.family] ?? r.family, segments: segment(r.s, r.family) }))}
      />
      <div className="flex flex-wrap gap-x-3 text-[11px] text-[var(--color-text-secondary)]">
        {segment({}, "").map((s) => (
          <span key={s.key} className="inline-flex items-center gap-1">
            <span aria-hidden className="inline-block h-2 w-2 rounded-[2px]" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </Panel>
  );
}

// --- Storage ------------------------------------------------------------------------

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

/** Bytes by file class (expiring share hatched) and the days they expire on. */
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
      color: RAMP[i] ?? FILL.quiet,
      label: CLASS_LABEL[name] ?? name,
      title: `${CLASS_LABEL[name] ?? name}: ${formatBytes(c.bytes)}${c.expiring ? `, ${formatBytes(c.expiring)} expiring` : ""}`,
    })),
    ...(rest.bytes ? [{ key: "rest", value: rest.bytes, hatched: rest.expiring, color: FILL.quiet, label: "Other" }] : []),
  ];
  const total = sorted.reduce((n, [, c]) => n + c.bytes, 0);
  const { expiring, reclaimed, grace_days } = storage.retention;
  const upcoming = calendar(expiring.by_day, 28);
  return (
    <div className="grid gap-x-12 gap-y-8 lg:grid-cols-[3fr_2fr]">
      <Panel label="Catalogued bytes" aside={`${formatBytes(expiring.bytes)} expiring (hatched)`}>
        <Figure value={formatBytes(total)} />
        <StackBar segments={segments} height={14} format={formatBytes} legend />
      </Panel>
      <Panel label="Expiry calendar" aside={`${grace_days}-day grace · ${formatBytes(reclaimed.bytes)} reclaimed in ${reclaimed.days} d`}>
        {upcoming.length ? (
          <>
            <DayColumns
              days={upcoming.map((d) => ({ day: d.day, value: d.bytes }))}
              height={56}
              color={FILL.warn}
              format={formatBytes}
            />
            <div className="flex justify-between font-mono text-[10px] text-[var(--color-text-tertiary)]">
              <span>{upcoming[0]?.day.slice(5)}</span>
              <span>{upcoming.at(-1)?.day.slice(5)}</span>
            </div>
          </>
        ) : (
          <p className="m-0 text-xs text-[var(--color-text-tertiary)]">Nothing is due to expire.</p>
        )}
      </Panel>
    </div>
  );
}

