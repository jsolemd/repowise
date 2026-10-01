/**
 * Chart primitives for the Make platform page. Server components, CSS-drawn:
 * no client JavaScript, so a revision-driven refresh redraws them for free.
 *
 * Colour follows the house theme's rules (ui/styles/globals.css): shares of a
 * whole step down the accent ramp, supporting marks use the neutral steps, and
 * state uses the node-* fills, which are tuned as fills where the status ink
 * vars are tuned as type. Hover text rides on `title`; every figure a reader
 * needs is also printed, so nothing depends on hovering.
 */
import type { CSSProperties, ReactNode } from "react";

export const RAMP = [1, 2, 3, 4, 5].map((n) => `var(--color-ramp-${n})`);
export const FILL = {
  fail: "var(--color-node-at-risk)",
  warn: "var(--color-node-needs-work)",
  ok: "var(--color-node-good)",
  quiet: "var(--color-neutral-1)",
  faint: "var(--color-neutral-3)",
} as const;

/** Diagonal hatch over a fill: the house mark for bytes or work on its way out. */
export const HATCH: CSSProperties = {
  backgroundImage:
    "repeating-linear-gradient(-45deg, color-mix(in srgb, var(--color-bg-surface) 70%, transparent) 0 2px, transparent 2px 6px)",
};

export function Label({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <p className={`m-0 font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--color-text-tertiary)] ${className}`}>
      {children}
    </p>
  );
}

export function Figure({ value, unit, color }: { value: ReactNode; unit?: ReactNode; color?: string }) {
  return (
    <p className="m-0 flex items-baseline gap-1.5">
      <span className="text-[28px] font-semibold leading-none tracking-tight tabular-nums" style={color ? { color } : undefined}>
        {value}
      </span>
      {unit && <span className="text-xs text-[var(--color-text-tertiary)]">{unit}</span>}
    </p>
  );
}

export interface Segment {
  key: string;
  value: number;
  color: string;
  label: string;
  /** Part of `value` drawn hatched: the share that is expiring or failed. */
  hatched?: number;
  title?: string;
}

/** One horizontal bar split into shares, 2px surface gaps between them. */
export function StackBar({
  segments,
  total,
  height = 10,
  legend = false,
  format = (n: number) => n.toLocaleString(),
}: {
  segments: Segment[];
  /** Scale denominator when the bar should not fill its track. */
  total?: number;
  height?: number;
  legend?: boolean;
  format?: (n: number) => string;
}) {
  const sum = segments.reduce((n, s) => n + s.value, 0);
  const scale = Math.max(total ?? sum, 1);
  return (
    <div className="flex flex-col gap-1.5">
      <div
        className="flex w-full gap-[2px] overflow-hidden rounded-[4px] bg-[var(--color-bg-inset)]"
        style={{ height }}
        role="img"
        aria-label={segments.map((s) => `${s.label} ${format(s.value)}`).join(", ")}
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
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-[var(--color-text-secondary)]">
          {segments.map((s) => (
            <span key={s.key} className="inline-flex items-center gap-1 tabular-nums">
              <span aria-hidden className="inline-block h-2 w-2 rounded-[2px]" style={{ background: s.color }} />
              {s.label} <span className="text-[var(--color-text-tertiary)]">{format(s.value)}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/** Labelled rows, each a StackBar on a shared scale, the total printed right. */
export function BarRows({
  rows,
  format = (n: number) => n.toLocaleString(),
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

/** A row of day columns, rounded at the data end and anchored to a baseline. */
export function DayColumns({
  days,
  height = 44,
  color = RAMP[0],
  format,
}: {
  days: { day: string; value: number }[];
  height?: number;
  color?: string;
  format: (n: number) => string;
}) {
  const max = Math.max(...days.map((d) => d.value), 0);
  return (
    <div
      className="flex items-end gap-[2px] border-b border-[var(--color-border-default)]"
      style={{ height }}
      role="img"
      aria-label={days.filter((d) => d.value).map((d) => `${d.day} ${format(d.value)}`).join(", ") || "Nothing in this window"}
    >
      {days.map((d) => (
        <div key={d.day} title={`${d.day}: ${format(d.value)}`} className="flex h-full min-w-0 flex-1 items-end">
          <div
            className="w-full rounded-t-[3px]"
            style={{
              height: d.value && max ? `${Math.max((d.value / max) * 100, 6)}%` : 2,
              background: d.value ? color : "var(--color-neutral-3)",
            }}
          />
        </div>
      ))}
    </div>
  );
}

/** A value against a whole: figure above, one bar under it. */
export function Meter({
  label,
  value,
  of,
  unit,
  color = RAMP[0],
  href,
  children,
}: {
  label: string;
  value: number;
  of?: number;
  unit?: string;
  color?: string;
  href?: string;
  children?: ReactNode;
}) {
  const body = (
    <>
      <Label>{label}</Label>
      <Figure value={value.toLocaleString()} unit={unit ?? (of !== undefined ? `of ${of.toLocaleString()}` : undefined)} />
      {of !== undefined && (
        <StackBar
          height={6}
          total={of}
          segments={[{ key: "v", value, color, label }]}
        />
      )}
      {children}
    </>
  );
  const className = "flex min-w-0 flex-col gap-2 p-4";
  return href ? (
    <a href={href} className={`${className} rounded-sm hover:bg-[var(--color-bg-elevated)]`}>
      {body}
    </a>
  ) : (
    <div className={className}>{body}</div>
  );
}
