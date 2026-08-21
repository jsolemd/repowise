"use client";

import useSWR from "swr";
import { AlertTriangle, CheckCircle2, Clock, HelpCircle } from "lucide-react";
import { getHealth } from "@/lib/api/health";
import { OverviewSection } from "@repowise-dev/ui/overview";
import { SettingsRow, SettingsRows } from "@repowise-dev/ui/settings";
import {
  classifyIndexHealth,
  type IndexVerdict,
} from "@/lib/search/index-health";
import type { SourceSearchHealth } from "@/lib/api/types";
import { toFriendlyMessage } from "@repowise-dev/ui/lib/errors";

const SWR_OPTS = { revalidateOnFocus: false, revalidateOnReconnect: false };

const VERDICT_UI: Record<
  IndexVerdict,
  { icon: typeof CheckCircle2; tone: string; label: string; hint: string }
> = {
  healthy: {
    icon: CheckCircle2,
    tone: "text-[var(--color-success)]",
    label: "Current",
    hint: "Every store agrees and nothing is queued. Search answers reflect the working tree.",
  },
  stale: {
    icon: Clock,
    tone: "text-[var(--color-warning)]",
    label: "Catching up",
    hint: "The index is behind the working tree. Results are still real, but recent edits may be missing.",
  },
  broken: {
    icon: AlertTriangle,
    tone: "text-[var(--color-error)]",
    label: "Not trustworthy",
    hint: "A store failed or the stores disagree. Treat search results as incomplete until this clears.",
  },
  unknown: {
    icon: HelpCircle,
    tone: "text-[var(--color-text-tertiary)]",
    label: "Cannot tell",
    hint: "The index could not fully describe itself, so its freshness is unproven either way.",
  },
};

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-mono text-xs text-[var(--color-text-secondary)] break-all">
      {children}
    </span>
  );
}

function ParityValue({ health }: { health: SourceSearchHealth }) {
  const { expected_chunks: expected, fts_chunks: fts, vector_chunks: vec } = health;
  const agree =
    expected !== null && expected === fts && expected === vec;
  return (
    <span
      className={
        agree
          ? "font-mono text-xs text-[var(--color-text-secondary)]"
          : "font-mono text-xs text-[var(--color-error)]"
      }
    >
      {expected ?? "?"} expected · {fts ?? "?"} full-text · {vec ?? "?"} vector
      {agree ? "" : " — stores disagree"}
    </span>
  );
}

/**
 * Search index health.
 *
 * Lives on Settings rather than a repo page on purpose: this answers "is my
 * install telling me the truth", which is the same question the Server section
 * above it answers one layer down. The in-flow signal an agent or a reader
 * actually needs — "a search lane failed" — is already rendered by the command
 * palette, in the same vocabulary the server uses.
 */
export function SearchIndexSection() {
  const { data, error, isLoading } = useSWR("health:source-search", getHealth, SWR_OPTS);
  const health = data?.source_search ?? null;

  return (
    <OverviewSection
      title="Search index"
      description="What the source-search lane knows about its own freshness, straight from the server."
      flush
    >
      {isLoading && (
        <p className="text-sm text-[var(--color-text-tertiary)]">Checking…</p>
      )}

      {error && (
        <p className="text-sm text-[var(--color-error)]">
          {toFriendlyMessage(error, "Could not reach the server")}
        </p>
      )}

      {!isLoading && !error && !health && (
        <p className="text-sm text-[var(--color-text-tertiary)]">
          The source-search lane is not enabled on this server. Wiki search still
          works; set{" "}
          <span className="font-mono text-xs">REPOWISE_SOURCE_SEARCH=1</span> to
          index source code as well.
        </p>
      )}

      {health && <SearchIndexDetail health={health} />}
    </OverviewSection>
  );
}

function SearchIndexDetail({ health }: { health: SourceSearchHealth }) {
  const verdict = classifyIndexHealth(health);
  const ui = VERDICT_UI[verdict];
  const Icon = ui.icon;
  const staleNames = Object.keys(health.stale_files);
  const queued =
    health.pending_updates +
    health.ready_updates +
    health.building_updates +
    health.blocked_updates;

  return (
    <>
      <div className="flex items-start gap-2.5">
        <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${ui.tone}`} aria-hidden />
        <div>
          <p className={`text-sm font-medium ${ui.tone}`}>{ui.label}</p>
          <p className="mt-0.5 text-sm leading-relaxed text-[var(--color-text-secondary)]">
            {ui.hint}
          </p>
        </div>
      </div>

      <SettingsRows>
        <SettingsRow label="Generation" hint="The published index the server is serving from.">
          <Mono>
            {health.generation_sequence ?? "?"}
            {health.generation_id ? ` · ${health.generation_id}` : ""}
          </Mono>
        </SettingsRow>

        <SettingsRow
          label="Indexed commit"
          hint="The commit this generation was built from. Saved-but-uncommitted edits are indexed on top of it."
        >
          <Mono>{health.indexed_commit ?? "unknown"}</Mono>
        </SettingsRow>

        <SettingsRow
          label="Store parity"
          hint="The manifest, the full-text index and the vector store must hold the same number of chunks."
        >
          <ParityValue health={health} />
        </SettingsRow>

        {(health.symbol_chunks != null ||
          health.file_window_chunks != null ||
          health.files_covered != null) && (
          <SettingsRow
            label="Coverage"
            hint="Parsed symbols, bounded windows over files the parser does not cover, and the files they came from."
          >
            <span className="font-mono text-xs text-[var(--color-text-secondary)]">
              {health.symbol_chunks ?? "?"} symbol ·{" "}
              {health.file_window_chunks ?? "?"} file-window
              {health.files_covered != null
                ? ` · ${health.files_covered} files`
                : ""}
            </span>
          </SettingsRow>
        )}

        {health.embedder && (
          <SettingsRow
            label="Embedder"
            hint="The model every vector in this generation was written with."
          >
            <Mono>
              {health.embedder.provider ?? "?"} · {health.embedder.model ?? "?"}
              {health.embedder.dims ? ` · ${health.embedder.dims}d` : ""}
            </Mono>
          </SettingsRow>
        )}

        {health.built_at && (
          <SettingsRow label="Built" hint="When this generation was published.">
            <Mono>{health.built_at}</Mono>
          </SettingsRow>
        )}

        <SettingsRow
          label="Queue"
          hint="Updates recorded after the active generation, by state."
        >
          <span className="font-mono text-xs text-[var(--color-text-secondary)]">
            {queued === 0
              ? "empty"
              : `${health.pending_updates} pending · ${health.ready_updates} ready · ${health.building_updates} building · ${health.blocked_updates} blocked`}
          </span>
        </SettingsRow>

        <SettingsRow
          label="Stale files"
          hint="Files whose last parse failed. Their previous documents are still served, deliberately."
        >
          {staleNames.length === 0 ? (
            <span className="text-xs text-[var(--color-text-tertiary)]">none</span>
          ) : (
            <span className="flex flex-col gap-0.5">
              <span className="font-mono text-xs text-[var(--color-warning)]">
                {staleNames.length}
              </span>
              {staleNames.slice(0, 10).map((name) => (
                <Mono key={name}>{name}</Mono>
              ))}
              {staleNames.length > 10 && (
                <span className="text-xs text-[var(--color-text-tertiary)]">
                  and {staleNames.length - 10} more
                </span>
              )}
            </span>
          )}
        </SettingsRow>

        <SettingsRow
          label="Recipe"
          hint="Chunking recipe and embedder identity. A change here invalidates every vector."
        >
          <Mono>{health.recipe_fingerprint ?? "unknown"}</Mono>
        </SettingsRow>

        <SettingsRow label="Stores" hint="Where the two derived stores live.">
          <span className="flex flex-col gap-0.5">
            <Mono>{health.fts_path ?? "unknown"}</Mono>
            <Mono>{health.lance_table ?? "unknown"}</Mono>
          </span>
        </SettingsRow>
      </SettingsRows>

      {(health.integrity_errors.length > 0 || health.last_error) && (
        <div className="rounded-md border border-[var(--color-error)]/40 bg-[var(--color-error)]/5 p-3">
          <p className="text-xs font-medium text-[var(--color-error)]">
            Reported by the index
          </p>
          {/* Verbatim: the server's own wording is the diagnostic, and
              summarising it here would be a second vocabulary for one fact. */}
          <ul className="mt-1.5 flex flex-col gap-1">
            {health.integrity_errors.map((err) => (
              <li key={err} className="font-mono text-xs leading-relaxed break-words text-[var(--color-text-secondary)]">
                {err}
              </li>
            ))}
            {health.last_error && (
              <li className="font-mono text-xs leading-relaxed break-words text-[var(--color-text-secondary)]">
                {health.last_error}
              </li>
            )}
          </ul>
        </div>
      )}
    </>
  );
}
