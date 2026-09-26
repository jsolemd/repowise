"use client";

import { ResponsiveTable, type ResponsiveColumn } from "@repowise-dev/ui/shared/responsive-table";
import { formatRelativeTime } from "@repowise-dev/ui/lib/format";

/** An operation as the page shows it: the kind already named for a person. */
export interface OperationRow {
  id: string;
  label: string;
  kind: string;
  status: string;
  createdAt: string;
  attempt: number | null;
  provider: string | null;
  errorCode: string | null;
}

const STATUS_COLOR: Record<string, string> = {
  completed: "var(--color-success)",
  failed: "var(--color-error)",
  ambiguous: "var(--color-error)",
  dispatch_started: "var(--color-warning)",
  running: "var(--color-warning)",
  queued: "var(--color-text-tertiary)",
  cancelled: "var(--color-text-tertiary)",
};

const readable = (value: string) => value.replaceAll("_", " ");

function Status({ status }: { status: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span
        aria-hidden
        className="h-1.5 w-1.5 shrink-0 rounded-full"
        style={{ background: STATUS_COLOR[status] ?? "var(--color-text-tertiary)" }}
      />
      {readable(status)}
    </span>
  );
}

const columns: ResponsiveColumn<OperationRow>[] = [
  {
    key: "operation",
    header: "Operation",
    priority: 1,
    render: (op) => (
      <span className="flex flex-col">
        <span className="text-[var(--color-text-primary)]">{op.label}</span>
        <span className="font-mono text-[10px] text-[var(--color-text-tertiary)]">{op.kind}</span>
      </span>
    ),
  },
  { key: "status", header: "Status", priority: 1, render: (op) => <Status status={op.status} /> },
  {
    key: "started",
    header: "Started",
    priority: 1,
    render: (op) => (
      <time dateTime={op.createdAt} className="whitespace-nowrap tabular-nums">
        {formatRelativeTime(op.createdAt)}
      </time>
    ),
  },
  {
    key: "attempt",
    header: "Attempt",
    priority: 2,
    render: (op) =>
      op.attempt ? `${op.attempt}${op.provider ? ` · ${op.provider}` : ""}` : "not started",
  },
  {
    key: "error",
    header: "Error",
    priority: 2,
    render: (op) =>
      op.errorCode ? (
        <span className="font-mono text-[11px] text-[var(--color-error)]">{op.errorCode}</span>
      ) : (
        <span className="text-[var(--color-text-tertiary)]">none</span>
      ),
  },
];

export function OperationsTable({ rows, empty }: { rows: OperationRow[]; empty: string }) {
  return (
    <ResponsiveTable
      columns={columns}
      rows={rows}
      rowKey={(op) => op.id}
      stacked="sm"
      bare
      caption="Make operations, newest first"
      empty={<p className="text-sm text-[var(--color-text-secondary)]">{empty}</p>}
    />
  );
}
