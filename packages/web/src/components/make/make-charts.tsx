/**
 * Chart primitives for the Make platform page. Server components, CSS-drawn:
 * no client JavaScript, so a revision-driven refresh redraws them for free.
 *
 * Colour follows the house theme's rules (ui/styles/globals.css): money by day and
 * bytes by class step down the accent ramp, supporting marks take the neutral
 * steps, and state takes the node-* fills, which are tuned as fills where the
 * status ink vars are tuned as type. Counts of work are drawn as state, the same
 * in every chart: work done or in flight recedes in the neutral steps (FILL.done,
 * FILL.flight, FILL.faint), work waiting on Jon takes FILL.wait and a failure
 * FILL.fail, so the eye lands on what is waiting or broken. That keeps the ramp
 * out of the Built panels, the certainty bar below excepted, where its orange
 * would read as the wait fill's.
 * Hatching means one thing everywhere: bytes on their way out.
 *
 * One exception is deliberate: the Evidence panel's certainty bar steps clear,
 * uncertain and contested down the ramp in that fixed order whatever their
 * counts. Certainty is an ordered scale of its own, how clear and how gray the
 * literature is on a claim, so the ramp's position names the rung on that scale
 * rather than the share, and the legend prints the counts.
 *
 * Printed: every headline figure, each legend's values, each BarRows row total,
 * a build group's failed count, and the largest day of a DayColumns run. Hover
 * (`title`) adds only exact values a glance does not need: each other day's
 * amount, the split inside a BarRows row, the failure codes behind an operation
 * family's failed share, and what a passing doctor check found. The aria-label
 * of every chart carries its values for a screen reader.
 */
import type { CSSProperties, ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { formatNumber } from "@repowise-dev/ui/lib/format";

export const RAMP = [1, 2, 3, 4, 5].map((n) => `var(--color-ramp-${n})`);
export const FILL = {
  fail: "var(--color-node-at-risk)",
  wait: "var(--color-node-needs-work)",
  ok: "var(--color-node-good)",
  done: "var(--color-neutral-1)",
  flight: "var(--color-neutral-2)",
  faint: "var(--color-neutral-3)",
} as const;

/** Diagonal hatch over a fill: the house mark for bytes on their way out. */
export const HATCH: CSSProperties = {
  backgroundImage:
    "repeating-linear-gradient(-45deg, color-mix(in srgb, var(--color-bg-surface) 70%, transparent) 0 2px, transparent 2px 6px)",
};

/** A calendar day as the axis prints it: "Oct 7". */
const shortDay = (day: string) =>
  new Date(`${day}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

export function Label({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <p className={`m-0 font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--color-text-tertiary)] ${className}`}>
      {children}
    </p>
  );
}

const FIGURE_SIZE = { xl: "text-[34px]", lg: "text-[28px]", md: "text-xl", sm: "text-lg" } as const;

/** A number and its unit. Proportional figures: a lone number is not a column. */
export function Figure({
  value,
  unit,
  color,
  size = "lg",
}: {
  value: ReactNode;
  unit?: ReactNode;
  color?: string;
  size?: keyof typeof FIGURE_SIZE;
}) {
  return (
    <p className="m-0 flex items-baseline gap-1.5">
      <span className={`${FIGURE_SIZE[size]} font-semibold leading-none tracking-tight`} style={color ? { color } : undefined}>
        {value}
      </span>
      {unit && <> <span className="text-xs text-[var(--color-text-tertiary)]">{unit}</span></>}
    </p>
  );
}

export interface Segment {
  key: string;
  value: number;
  color: string;
  label: string;
  /** Part of `value` drawn hatched: the share on its way out. */
  hatched?: number;
  title?: string;
}

export interface LegendItem {
  key: string;
  label: string;
  color: string;
  value?: string;
  hatched?: boolean;
}

/** Swatches naming a chart's colours, each with its value when the chart has one. */
export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-[var(--color-text-secondary)]">
      {items.map((s) => (
        <span key={s.key} className="inline-flex items-center gap-1 tabular-nums">
          <span
            aria-hidden
            className="inline-block h-2 w-2 rounded-[2px]"
            style={{ background: s.color, ...(s.hatched ? HATCH : {}) }}
          />
          {s.label}
          {s.value !== undefined && <> <span className="text-[var(--color-text-tertiary)]">{s.value}</span></>}
        </span>
      ))}
    </div>
  );
}

/** One horizontal bar split into shares, 2px surface gaps between them. */
export function StackBar({
  segments,
  total,
  height = 10,
  legend = false,
  extra = [],
  format = formatNumber,
}: {
  segments: Segment[];
  /** Scale denominator when the bar should not fill its track. */
  total?: number;
  height?: number;
  legend?: boolean;
  /** Legend entries beyond the segments, such as the hatch's meaning. */
  extra?: LegendItem[];
  format?: (n: number) => string;
}) {
  const sum = segments.reduce((n, s) => n + s.value, 0);
  const scale = Math.max(total ?? sum, 1);
  // With nothing drawn the bar is an empty track: the printed zero or legend says
  // so, and an image with no name would only be noise to a screen reader.
  const named = sum > 0 ? { role: "img", "aria-label": segments.map((s) => `${s.label} ${format(s.value)}`).join(", ") } : { "aria-hidden": true };
  return (
    <div className="flex flex-col gap-1.5">
      <div
        className="flex w-full gap-[2px] overflow-hidden rounded-[4px] bg-[var(--color-bg-inset)]"
        style={{ height }}
        {...named}
      >
        {segments.map((s) =>
          s.value > 0 ? (
            <div
              key={s.key}
              title={s.title ?? `${s.label}: ${format(s.value)}`}
              className="flex h-full min-w-[3px] basis-0 overflow-hidden"
              style={{ flexGrow: s.value, background: s.color }}
            >
              {s.hatched ? (
                <div className="ml-auto h-full" style={{ width: `${(s.hatched / s.value) * 100}%`, ...HATCH }} />
              ) : null}
            </div>
          ) : null,
        )}
        {/* Unfilled track, so a bar scaled to a larger total stops short. */}
        {scale > sum && <div aria-hidden className="basis-0" style={{ flexGrow: scale - sum }} />}
      </div>
      {legend && (
        <Legend items={[...segments.map((s) => ({ key: s.key, label: s.label, color: s.color, value: format(s.value) })), ...extra]} />
      )}
    </div>
  );
}

/** Labelled rows, each a StackBar on a shared scale, the total printed right. */
export function BarRows({
  rows,
  format = formatNumber,
}: {
  rows: { key: string; label: ReactNode; segments: Segment[]; note?: ReactNode }[];
  format?: (n: number) => string;
}) {
  const max = Math.max(1, ...rows.map((r) => r.segments.reduce((n, s) => n + s.value, 0)));
  return (
    <ul className="m-0 grid list-none grid-cols-[minmax(0,7.5rem)_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-2 p-0 text-xs">
      {rows.map((r) => (
        <li key={r.key} className="contents">
          <span className="truncate text-[var(--color-text-secondary)]">{r.label}</span>
          <StackBar segments={r.segments} total={max} height={8} format={format} />
          <span className="text-right tabular-nums text-[var(--color-text-primary)]">
            {format(r.segments.reduce((n, s) => n + s.value, 0))}
            {r.note}
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * A run of day columns, rounded at the data end and anchored to a baseline. The
 * largest day carries its value on its cap, and the first and last days label
 * the axis, so the run reads without a hover.
 */
export function DayColumns({
  days,
  height = 44,
  color = RAMP[0],
  hatched = false,
  format,
}: {
  days: { day: string; value: number }[];
  height?: number;
  color?: string;
  /** Draw the columns in the hatch: days on which bytes go. */
  hatched?: boolean;
  format: (n: number) => string;
}) {
  const max = Math.max(...days.map((d) => d.value), 0);
  const peak = max > 0 ? days.findIndex((d) => d.value === max) : -1;
  // The cap label hangs from its column, so near either end it opens inward.
  const align =
    peak < days.length / 4 ? "left-0" : peak >= (days.length * 3) / 4 ? "right-0" : "left-1/2 -translate-x-1/2";
  const first = days[0];
  const last = days.at(-1);
  return (
    <div className="flex flex-col gap-1">
      <div
        className="flex items-end gap-[2px] border-b border-[var(--color-border-default)]"
        style={{ height, marginTop: 16 }}
        role="img"
        aria-label={days.filter((d) => d.value).map((d) => `${d.day} ${format(d.value)}`).join(", ") || "Nothing in this window"}
      >
        {days.map((d, i) => (
          <div key={d.day} title={`${shortDay(d.day)}: ${format(d.value)}`} className="relative flex h-full min-w-0 flex-1 items-end">
            <div
              className="w-full rounded-t-[3px]"
              style={{
                height: d.value && max ? `${Math.max((d.value / max) * 100, 6)}%` : 2,
                background: d.value ? color : "var(--color-neutral-3)",
                ...(hatched && d.value ? HATCH : {}),
              }}
            />
            {i === peak && (
              <span
                data-peak
                className={`absolute bottom-full mb-1 whitespace-nowrap text-[11px] font-medium tabular-nums text-[var(--color-text-secondary)] ${align}`}
              >
                {format(d.value)}
              </span>
            )}
          </div>
        ))}
      </div>
      {first && last && (
        <div aria-hidden className="flex justify-between font-mono text-[10px] text-[var(--color-text-tertiary)]">
          <span>{shortDay(first.day)}</span>
          <span>{shortDay(last.day)}</span>
        </div>
      )}
    </div>
  );
}

/**
 * One thing waiting on Jon: its count, and a meter when the count is a share of
 * a whole. A tile with work waiting carries its icon in the warning ink and its
 * count in full ink; an empty one recedes, so the eye goes to what is waiting.
 * `value` null means the report did not answer.
 */
export function Tile({
  icon: Icon,
  label,
  value,
  unit,
  of,
  href,
  size = "xl",
}: {
  icon: LucideIcon;
  label: string;
  value: number | null;
  unit?: ReactNode;
  of?: number;
  href?: string;
  /** The count's size: full on the Make page, quieter where Make is a guest. */
  size?: "xl" | "md";
}) {
  const waiting = value !== null && value > 0;
  const body = (
    <>
      <Label className="flex items-center gap-1.5">
        <span className="inline-flex" style={{ color: waiting ? "var(--color-warning)" : undefined }}>
          <Icon aria-hidden className="h-3.5 w-3.5" />
        </span>
        {label}
      </Label>
      <Figure
        size={size}
        value={value === null ? "—" : formatNumber(value)}
        unit={value === null ? "no answer" : (unit ?? (of !== undefined ? `of ${formatNumber(of)}` : undefined))}
        color={waiting ? "var(--color-text-primary)" : "var(--color-text-tertiary)"}
      />
      {of !== undefined && value !== null && (
        <StackBar height={6} total={of} segments={[{ key: "v", value, color: FILL.wait, label }]} />
      )}
    </>
  );
  const className = "flex min-w-0 flex-col gap-2.5 rounded-md p-3";
  return href ? (
    <a href={href} className={`${className} hover:bg-[var(--color-bg-elevated)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-accent-primary)]`}>
      {body}
    </a>
  ) : (
    <div className={className}>{body}</div>
  );
}
