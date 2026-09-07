"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { BookOpen } from "lucide-react";
import { ApiError } from "@repowise-dev/ui/shared/api-error";
import { EmptyState } from "@repowise-dev/ui/shared/empty-state";
import { TableSkeleton } from "@repowise-dev/ui/shared/loading-skeletons";
import {
  ResponsiveTable,
  type ResponsiveColumn,
} from "@repowise-dev/ui/shared";
import { OverviewSection } from "@repowise-dev/ui/overview";
import { SettingsRow, SettingsRows } from "@repowise-dev/ui/settings";
import { StatRibbon, type RibbonStat } from "@repowise-dev/ui/stats/stat-ribbon";
import { Badge } from "@repowise-dev/ui/ui/badge";
import { formatNumber, formatRelativeTimeOrNull } from "@repowise-dev/ui/lib/format";
import { ApiClientError } from "@/lib/api/client";
import {
  getDocsLibraries,
  type DocsJob,
  type DocsJobs,
  type DocsLibrariesResponse,
  type DocsLibrary,
} from "@/lib/api/docs-libraries";
import {
  classifyLibrary,
  compareLibraries,
  LIBRARY_SORT_KEYS,
  DOCS_STATUS_UI,
  earliestNextCheck,
  formatNextCheck,
  remoteHasMoved,
  type LibrarySortKey,
} from "@/lib/docs/library-status";

const SWR_OPTS = { revalidateOnFocus: false, revalidateOnReconnect: false };

/** Indexer status → an existing badge variant. `indexing` borrows `accent`
 *  because it is work in progress, not a verdict about the copy on disk. */
const STATUS_BADGE: Record<string, "fresh" | "stale" | "outdated" | "accent" | "default"> =
  {
    ready: "fresh",
    pending: "stale",
    indexing: "accent",
    error: "outdated",
  };

function shortSha(sha: string | null): string | null {
  return sha ? sha.slice(0, 8) : null;
}

/** What went wrong with this library, in one line, or null when nothing did. */
function problem(lib: DocsLibrary): string | null {
  return lib.error_message || lib.last_freshness_error || lib.graph_sync_error || null;
}

const COLUMNS: ResponsiveColumn<DocsLibrary>[] = [
  {
    key: "name",
    header: "Library",
    priority: 1,
    render: (lib) => {
      const source = lib.repo
        ? lib.branch
          ? `${lib.repo}@${lib.branch}`
          : lib.repo
        : lib.source_type;
      const why = problem(lib);
      return (
        <div className="min-w-0">
          <span className="text-sm font-medium text-[var(--color-text-primary)]">
            {lib.name}
          </span>
          {source && (
            <span className="mt-0.5 block font-mono text-xs text-[var(--color-text-tertiary)] [overflow-wrap:anywhere]">
              {source}
            </span>
          )}
          {/* The reason lives with the row's identity rather than in a column
              of its own: it is prose, it is rare, and the stacked card renders
              this block as the card title, so one placement covers both. */}
          {why && (
            <span className="mt-0.5 block text-xs text-[var(--color-error)] [overflow-wrap:anywhere]">
              {why}
            </span>
          )}
        </div>
      );
    },
  },
  {
    key: "status",
    header: "Status",
    priority: 1,
    render: (lib) => (
      <Badge variant={STATUS_BADGE[lib.status] ?? "default"}>{lib.status}</Badge>
    ),
  },
  {
    key: "freshness",
    header: "Freshness",
    priority: 1,
    sortable: true,
    render: (lib) => {
      const ui = DOCS_STATUS_UI[classifyLibrary(lib)];
      // Always a chip, including the unknown case: an empty cell here reads as
      // "fine", which is the exact misreading this column exists to prevent.
      return (
        <Badge variant={ui.badge} title={ui.hint}>
          {ui.label}
        </Badge>
      );
    },
  },
  {
    key: "indexed_at",
    header: "Indexed",
    priority: 2,
    sortable: true,
    render: (lib) => (
      <span className="text-xs tabular-nums text-[var(--color-text-tertiary)]">
        {formatRelativeTimeOrNull(lib.indexed_at, "never")}
      </span>
    ),
  },
  {
    key: "freshness_checked_at",
    header: "Checked",
    priority: 2,
    sortable: true,
    render: (lib) => (
      <span className="text-xs tabular-nums text-[var(--color-text-tertiary)]">
        {formatRelativeTimeOrNull(lib.freshness_checked_at, "never")}
      </span>
    ),
  },
  {
    key: "next_freshness_check_at",
    header: "Next check",
    priority: 3,
    render: (lib) => (
      <span className="text-xs tabular-nums text-[var(--color-text-tertiary)]">
        {formatNextCheck(lib.next_freshness_check_at)}
      </span>
    ),
  },
  {
    key: "counts",
    header: "Files / Chunks",
    mobileLabel: "Files / chunks",
    align: "right",
    priority: 3,
    render: (lib) => (
      <span className="text-xs tabular-nums text-[var(--color-text-tertiary)]">
        {formatNumber(lib.file_count)} / {formatNumber(lib.chunk_count)}
      </span>
    ),
  },
  {
    key: "sha",
    header: "SHA",
    priority: 3,
    render: (lib) => {
      const current = shortSha(lib.current_sha);
      const remote = shortSha(lib.last_remote_sha);
      if (!current && !remote) return null;
      // Not a string comparison of the two cells: the fields hold different
      // kinds of identity. See `remoteHasMoved`.
      const moved = remoteHasMoved(lib);
      return (
        <span className="font-mono text-xs text-[var(--color-text-tertiary)]">
          {current ?? "—"}
          {moved && (
            <>
              {" → "}
              <span className="text-[var(--color-error)]">{remote}</span>
            </>
          )}
        </span>
      );
    },
  },
];

export function DocsLibrariesSection() {
  const { data, error, isLoading, mutate } = useSWR<DocsLibrariesResponse>(
    "docs-libraries",
    getDocsLibraries,
    SWR_OPTS,
  );
  const [sort, setSort] = useState<LibrarySortKey>("freshness");
  const [order, setOrder] = useState<"asc" | "desc">("desc");

  const libraries = useMemo(() => data?.libraries ?? [], [data]);

  const rows = useMemo(
    () => [...libraries].sort((a, b) => compareLibraries(a, b, sort, order)),
    [libraries, sort, order],
  );

  const unknownCount = useMemo(
    () => libraries.filter((lib) => classifyLibrary(lib) === "unknown").length,
    [libraries],
  );

  if (isLoading) {
    return (
      <OverviewSection
        title="Libraries"
        description="Every documentation library the indexer holds, and whether its copy still matches its source."
        flush
      >
        <TableSkeleton rows={8} />
      </OverviewSection>
    );
  }

  if (error || !data) {
    const unreachable = error instanceof ApiClientError && error.status === 502;
    return (
      <OverviewSection title="Libraries" flush>
        <ApiError
          title={unreachable ? "Documentation inventory unavailable" : "Failed to load"}
          message={
            // No backticks: ApiError takes a plain string, so markdown fences
            // would render as literal characters. The docs lane is on-demand,
            // so "down" is a normal state with a one-command fix.
            unreachable
              ? "The documentation service could not provide its inventory. Retry, or start the service with: solemd infra up"
              : "The library inventory could not be read from the server."
          }
          onRetry={() => void mutate()}
        />
      </OverviewSection>
    );
  }

  const { summary, inventory } = data;
  const nextCheck = earliestNextCheck(libraries);

  const ribbon: RibbonStat[] = [
    {
      label: "Libraries",
      value: formatNumber(summary.total_libraries),
      sub: "indexed documentation sources",
    },
    {
      label: "Ready",
      value: formatNumber(summary.ready_libraries),
      sub:
        summary.error_libraries > 0
          ? `${formatNumber(summary.error_libraries)} failed to index`
          : "searchable right now",
    },
    {
      label: "Chunks",
      value: formatNumber(inventory.total_chunks),
      sub: `across ${formatNumber(inventory.total_files)} files`,
    },
    {
      label: "Freshness unknown",
      value: formatNumber(unknownCount),
      // Only coloured when there is something to colour: an unproven library
      // is the state that lets a months-stale copy pass for current.
      valueColor: unknownCount > 0 ? "text-[var(--color-warning)]" : undefined,
      sub: "never successfully checked",
    },
    {
      label: "Next check",
      value: nextCheck ? formatNextCheck(nextCheck) : "—",
      sub: nextCheck ? "soonest scheduled check" : "nothing scheduled",
    },
  ];

  return (
    <>
      <StatRibbon stats={ribbon} />

      <OverviewSection
        title="Libraries"
        description="Every documentation library the indexer holds, and whether its copy still matches its source. Sort by freshness to bring the unproven and the outdated to the top."
      >
        {libraries.length === 0 ? (
          <EmptyState
            icon={<BookOpen className="h-8 w-8" />}
            title="No documentation libraries"
            description="The documentation service is running but holds no libraries yet. Add one with the docs MCP server's add_doc_library tool."
          />
        ) : (
          <ResponsiveTable
            columns={COLUMNS}
            rows={rows}
            rowKey={(lib) => lib.library_id}
            caption="Indexed documentation libraries"
            sortField={sort}
            sortOrder={order}
            onSort={(key) => {
              if (!LIBRARY_SORT_KEYS.includes(key as LibrarySortKey)) return;
              if (key === sort) setOrder((o) => (o === "asc" ? "desc" : "asc"));
              else {
                setSort(key as LibrarySortKey);
                setOrder("desc");
              }
            }}
            stacked="md"
            virtualize={{}}
            bare
          />
        )}
      </OverviewSection>

      <SchedulerSection data={data} />
      <JobsSection jobs={data.jobs} />
    </>
  );
}

function StateFlag({ running, on, off }: { running: boolean; on: string; off: string }) {
  return (
    <Badge variant={running ? "fresh" : "default"}>{running ? on : off}</Badge>
  );
}

/** `subsystem: status` pairs as chips, skipping the ones the service omitted. */
function StatusChips({ statuses }: { statuses: Record<string, string | undefined> }) {
  const entries = Object.entries(statuses).filter(
    (entry): entry is [string, string] => Boolean(entry[1]),
  );
  if (entries.length === 0) {
    return <span className="text-xs text-[var(--color-text-tertiary)]">Not reported</span>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {entries.map(([name, status]) => (
        <Badge key={name} variant={status === "ok" ? "fresh" : "outdated"}>
          {name}: {status}
        </Badge>
      ))}
    </div>
  );
}

function SchedulerSection({ data }: { data: DocsLibrariesResponse }) {
  const { scheduler, worker, runtime, dependencies, unreachable_libraries: unreachable } =
    data;
  const unreachableEntries = Object.entries(unreachable);

  return (
    <OverviewSection
      title="Scheduler & worker"
      description="Who is checking freshness and who is indexing, straight from the documentation service's own readiness report."
    >
      <SettingsRows>
        <SettingsRow
          label="Scheduler"
          hint="Queues freshness checks and re-index jobs on an interval."
        >
          <div className="flex flex-wrap items-center gap-2">
            <StateFlag running={scheduler.running} on="Running" off="Stopped" />
            <span className="text-xs text-[var(--color-text-tertiary)]">
              started {formatRelativeTimeOrNull(scheduler.last_started_at, "never")} ·
              completed {formatRelativeTimeOrNull(scheduler.last_completed_at, "never")}
            </span>
          </div>
        </SettingsRow>

        <SettingsRow label="Worker" hint="Claims queued jobs and writes the chunks.">
          <div className="flex flex-wrap items-center gap-2">
            <StateFlag running={worker.running} on="Running" off="Stopped" />
            {worker.id && (
              <span className="font-mono text-xs text-[var(--color-text-tertiary)]">
                {worker.id}
              </span>
            )}
          </div>
        </SettingsRow>

        <SettingsRow label="Runtime" hint="The service's own subsystem report.">
          <StatusChips statuses={runtime} />
        </SettingsRow>

        <SettingsRow
          label="Dependencies"
          hint="The stores and models indexing needs. Search degrades when one is down."
        >
          <StatusChips statuses={dependencies} />
        </SettingsRow>

        {unreachableEntries.length > 0 && (
          <SettingsRow
            label="Unreachable"
            hint="Consecutive failures reaching a library's remote. These are the libraries whose freshness cannot be proven."
          >
            <ul className="space-y-1">
              {unreachableEntries.map(([libraryId, failures]) => (
                <li key={libraryId} className="text-xs text-[var(--color-text-secondary)]">
                  <span className="font-mono">{libraryId}</span>
                  <span className="text-[var(--color-text-tertiary)]">
                    {" "}
                    — {failures} consecutive {failures === 1 ? "failure" : "failures"}
                  </span>
                </li>
              ))}
            </ul>
          </SettingsRow>
        )}
      </SettingsRows>
    </OverviewSection>
  );
}

function JobList({ jobs }: { jobs: DocsJob[] }) {
  return (
    <ul className="space-y-1">
      {jobs.map((job) => (
        <li key={job.id} className="text-xs text-[var(--color-text-secondary)]">
          <span className="font-mono">{job.library_id}</span>
          <span className="text-[var(--color-text-tertiary)]">
            {" "}
            · {job.job_type}
            {job.files_total
              ? ` · ${formatNumber(job.files_processed ?? 0)}/${formatNumber(job.files_total)} files`
              : ""}
          </span>
          {job.error_message && (
            <span className="block text-[var(--color-error)]">{job.error_message}</span>
          )}
        </li>
      ))}
    </ul>
  );
}

function JobsSection({ jobs }: { jobs: DocsJobs }) {
  const { pending, running, recent_failed: failed } = jobs;
  const idle = pending.length === 0 && running.length === 0 && failed.length === 0;

  return (
    <OverviewSection
      title="Jobs"
      description="What the indexer is working on, waiting on, and last failed at."
    >
      {idle ? (
        // Rule 10: nothing queued, nothing running, nothing broken is one
        // sentence, not an empty-state panel demanding attention.
        <p className="text-sm text-[var(--color-text-tertiary)]">
          Nothing queued, nothing running, and no recent failures.
        </p>
      ) : (
        <SettingsRows>
          {running.length > 0 && (
            <SettingsRow label={`Running (${running.length})`}>
              <JobList jobs={running} />
            </SettingsRow>
          )}
          {pending.length > 0 && (
            <SettingsRow label={`Pending (${pending.length})`}>
              <JobList jobs={pending} />
            </SettingsRow>
          )}
          {failed.length > 0 && (
            <SettingsRow
              label={`Recent failures (${failed.length})`}
              hint="Failed jobs the service still remembers. A library can be ready and still have a failure here."
            >
              <JobList jobs={failed} />
            </SettingsRow>
          )}
        </SettingsRows>
      )}
    </OverviewSection>
  );
}
