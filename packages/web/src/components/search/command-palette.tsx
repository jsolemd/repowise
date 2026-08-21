"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Command } from "cmdk";
import useSWR from "swr";
import { Search, LayoutDashboard, Settings, BookOpen, FileCode, Layers, Link2, GitMerge, MessageSquare, BookText, Lightbulb, AlertTriangle } from "lucide-react";
import { useSearch } from "@/lib/hooks/use-search";
import { paletteFilter, preRankedKeywords } from "./palette-filter";
import {
  formatLineRange,
  groupSearchResults,
  rankDecisions,
  type SearchGroupId,
} from "@/lib/search/result-groups";
import { truncatePath } from "@repowise-dev/ui/lib/format";
import { commandPaletteShortcutIsClaimed } from "@repowise-dev/ui/lib/command-palette-scope";
import { fileEntityPath } from "@repowise-dev/ui/shared/entity";
import { getFilesIndex } from "@/lib/api/files";
import { listDecisions } from "@/lib/api/decisions";
import { repoNavItems } from "@/components/layout/nav-items";
import { useGenerativeDisabled } from "@/components/layout/deployment-policy-provider";
import type { RepoResponse, WorkspaceResponse } from "@/lib/api/types";

/** One icon per section, so the three read apart at a glance. */
const GROUP_ICON: Record<SearchGroupId, typeof FileCode> = {
  code: FileCode,
  docs: BookText,
  decisions: Lightbulb,
};

interface CommandPaletteProps {
  repos: RepoResponse[];
  workspace?: WorkspaceResponse | null;
}

export function CommandPalette({ repos, workspace }: CommandPaletteProps) {
  const isWorkspace = workspace?.is_workspace ?? false;
  const generativeDisabled = useGenerativeDisabled();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const router = useRouter();
  const pathname = usePathname();

  const { envelope, error: searchError, isLoading } = useSearch(query, { limit: 8 });

  // Active repo: from the URL when inside one, else the only repo.
  const activeRepo = useMemo(() => {
    const m = pathname?.match(/^\/repos\/([^/]+)/);
    const fromPath = m ? repos.find((r) => r.id === m[1]) : undefined;
    return fromPath ?? (repos.length === 1 ? repos[0] : undefined);
  }, [pathname, repos]);

  const repoPages = useMemo(
    () => (activeRepo ? repoNavItems(activeRepo.id, { generativeDisabled }) : []),
    [activeRepo, generativeDisabled],
  );

  // File jump — fetched lazily (only once the palette is open with a repo in
  // scope) and cached; the Files page shares the same SWR key so this is warm
  // after a visit there. We do our own ranking and cap the rendered set so the
  // palette never mounts thousands of cmdk items.
  const { data: filesData } = useSWR(
    open && activeRepo ? `files-index:${activeRepo.id}` : null,
    () => getFilesIndex(activeRepo!.id),
    { revalidateOnFocus: false },
  );

  const fileMatches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!activeRepo || q.length < 2 || !filesData) return [];
    const scored: { path: string; score: number }[] = [];
    for (const f of filesData.files) {
      const path = f.file_path.toLowerCase();
      const idx = path.indexOf(q);
      if (idx === -1) continue;
      const base = f.file_path.split("/").pop()?.toLowerCase() ?? "";
      // Rank: basename match beats mid-path; earlier match beats later.
      const score = (base.includes(q) ? 0 : 1000) + idx;
      scored.push({ path: f.file_path, score });
    }
    scored.sort((a, b) => a.score - b.score || a.path.length - b.path.length);
    return scored.slice(0, 12).map((s) => s.path);
  }, [query, activeRepo, filesData]);

  // Decisions are not in the search index — it holds wiki pages and source
  // chunks — so the palette matches the repo's records itself, on the same
  // lazy-once-open terms as the file list above.
  const { data: decisionsData } = useSWR(
    open && activeRepo ? `decisions-index:${activeRepo.id}` : null,
    () => listDecisions(activeRepo!.id, { limit: 200 }),
    { revalidateOnFocus: false },
  );

  const grouped = useMemo(
    () =>
      groupSearchResults({
        envelope,
        decisions: rankDecisions(decisionsData ?? [], query),
        linkPrefix: `/repos/${activeRepo?.id ?? repos[0]?.id ?? ""}`,
      }),
    [envelope, decisionsData, query, activeRepo, repos],
  );

  // Only when the server said so. The stock search host classifies nothing,
  // and rendering its silence as a verdict would be inventing one.
  //
  // `degraded` outranks the plain caution: the server pins caution whenever a
  // lane failed, so on a degraded answer the usual "no exact match, closest by
  // meaning" would name a cause that is not the cause.
  const noMatch = grouped.confidence === "no_match";
  const degraded = grouped.degraded === true;
  const lowConfidence = grouped.confidence === "caution" && !degraded;
  // A request that never arrived is the same lie as a search that read no
  // corpus, one layer down: SWR hands back no data, the palette renders no
  // rows, and an outage looks like "nothing matches". Both land in the same
  // failure state rather than in the empty one.
  const unavailable = grouped.failureMessage !== undefined || searchError !== undefined;
  const failureDetail =
    grouped.degradedReason ??
    (searchError instanceof Error ? searchError.message : undefined);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        // A surface with its own palette owns the shortcut while it is
        // mounted. Without this both dialogs opened on one keypress, stacked.
        // The sidebar and mobile nav still reach this palette by event, so it
        // stays available where the scoped one cannot answer.
        if (commandPaletteShortcutIsClaimed()) return;
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    const openHandler = () => setOpen(true);
    window.addEventListener("keydown", handler);
    window.addEventListener("repowise:open-command-palette", openHandler);
    return () => {
      window.removeEventListener("keydown", handler);
      window.removeEventListener("repowise:open-command-palette", openHandler);
    };
  }, []);

  const navigate = useCallback(
    (href: string) => {
      router.push(href);
      setOpen(false);
      setQuery("");
    },
    [router],
  );

  return (
    <Command.Dialog
      open={open}
      onOpenChange={setOpen}
      label="Command palette"
      // Search hits and file matches arrive already ranked; everything else is
      // a static entry cmdk should keep matching by name. See `paletteFilter`.
      filter={paletteFilter}
      className="fixed inset-0 z-[calc(var(--z-modal)+1)] flex items-start justify-center pt-[10vh] sm:pt-[20vh] px-4"
    >
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm"
        onClick={() => setOpen(false)}
      />
      <div className="relative z-10 w-full max-w-xl rounded-xl border border-[var(--color-border-default)] bg-[var(--color-bg-overlay)] shadow-2xl overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[var(--color-border-default)]">
          <Search className="h-4 w-4 text-[var(--color-text-tertiary)] shrink-0" />
          <Command.Input
            value={query}
            onValueChange={setQuery}
            placeholder="Jump to a file, search pages, navigate repos…"
            className="flex-1 bg-transparent text-sm text-[var(--color-text-primary)] outline-none placeholder:text-[var(--color-text-tertiary)]"
          />
          <kbd className="hidden sm:inline-flex items-center rounded border border-[var(--color-border-default)] px-1.5 py-0.5 text-xs text-[var(--color-text-tertiary)] font-mono">
            ESC
          </kbd>
        </div>

        <Command.List className="max-h-[60dvh] overflow-y-auto py-2">
          <Command.Empty className="px-4 py-8 text-center text-sm text-[var(--color-text-tertiary)]">
            {isLoading ? "Searching…" : "No results found."}
          </Command.Empty>

          {/* A `no_match` verdict, said plainly. Not `Command.Empty`: the Ask
              row always matches, so cmdk never considers the list empty and
              would never render it. */}
          {noMatch && grouped.emptyMessage && (
            <p className="px-5 py-6 text-center text-sm leading-relaxed text-[var(--color-text-tertiary)] [text-wrap:pretty]">
              {grouped.emptyMessage}
            </p>
          )}

          {/* A search that broke. Deliberately unlike the empty state above:
              that one is a finding about the repository and this one is the
              absence of a finding, and a reader who cannot tell them apart
              will read an outage as an answer. */}
          {unavailable && (
            <div
              role="alert"
              className="mx-3 my-3 rounded-md border border-[color-mix(in_srgb,var(--color-error)_28%,transparent)] bg-[var(--color-error-muted)] px-4 py-3"
            >
              <p className="flex items-center gap-2 text-sm font-medium text-[var(--color-text-primary)]">
                <AlertTriangle className="h-4 w-4 shrink-0 text-[var(--color-error)]" />
                Search is unavailable
              </p>
              <p className="mt-1 text-xs leading-relaxed text-[var(--color-text-secondary)] [text-wrap:pretty]">
                Nothing was searched, so this is not a statement about the repository.
              </p>
              {failureDetail && (
                <p className="mt-1.5 font-mono text-[11px] leading-relaxed text-[var(--color-text-tertiary)] break-words">
                  {failureDetail}
                </p>
              )}
            </div>
          )}

          {/* Partial results. The rows below are real, and the ones a failed
              lane would have contributed are simply missing — which is exactly
              the state that looked like a healthy empty answer for 21 minutes,
              so it gets a banner rather than a footnote. */}
          {degraded && !unavailable && (
            <div className="mx-3 my-2 rounded-md border border-[var(--color-border-default)] bg-[var(--color-bg-elevated)] px-3 py-2">
              <p className="flex items-center gap-2 text-xs font-medium text-[var(--color-text-secondary)]">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-[var(--color-warning)]" />
                Partial results — a search lane failed
              </p>
              {grouped.degradedReason && (
                <p className="mt-1 font-mono text-[11px] leading-relaxed text-[var(--color-text-tertiary)] break-words">
                  {grouped.degradedReason}
                </p>
              )}
            </div>
          )}

          {/* Quick-ask — available when a repo is in scope and this deployment
              permits generative surfaces. Under the hard policy the row is not
              rendered at all: the server refuses the chat route, so offering it
              would only route the user to a 409. */}
          {activeRepo && !generativeDisabled && (
            <Command.Group heading="Ask" className="px-2 pb-1">
              <Command.Item
                value={`ask-repowise ${query}`}
                onSelect={() =>
                  navigate(
                    query.trim()
                      ? `/repos/${activeRepo.id}/chat?q=${encodeURIComponent(query.trim())}`
                      : `/repos/${activeRepo.id}/chat`,
                  )
                }
                className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-[var(--color-text-secondary)] cursor-pointer hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)] data-[selected=true]:bg-[var(--color-bg-elevated)] data-[selected=true]:text-[var(--color-text-primary)]"
              >
                <MessageSquare className="h-4 w-4 text-[var(--color-accent-primary)]" />
                <span className="truncate">
                  {query.trim() ? (
                    <>
                      Ask repowise: <span className="text-[var(--color-text-primary)]">“{query.trim()}”</span>
                    </>
                  ) : (
                    "Ask repowise…"
                  )}
                </span>
              </Command.Item>
            </Command.Group>
          )}

          {/* Per-repo page navigation */}
          {activeRepo && repoPages.length > 0 && (
            <Command.Group heading={`Go to — ${activeRepo.name}`} className="px-2 pb-1">
              {repoPages.map((item) => {
                const Icon = item.icon;
                return (
                  <Command.Item
                    key={item.href}
                    value={`goto ${activeRepo.name} ${item.label}`}
                    onSelect={() => navigate(item.href)}
                    className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-[var(--color-text-secondary)] cursor-pointer hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)] data-[selected=true]:bg-[var(--color-bg-elevated)] data-[selected=true]:text-[var(--color-text-primary)]"
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </Command.Item>
                );
              })}
            </Command.Group>
          )}

          {/* Quick navigation */}
          <Command.Group
            heading="Navigate"
            className="px-2 pb-1"
          >
            <Command.Item
              value="dashboard"
              onSelect={() => navigate("/")}
              className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-[var(--color-text-secondary)] cursor-pointer hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)] data-[selected=true]:bg-[var(--color-bg-elevated)] data-[selected=true]:text-[var(--color-text-primary)]"
            >
              <LayoutDashboard className="h-4 w-4" />
              Dashboard
            </Command.Item>
            <Command.Item
              value="settings"
              onSelect={() => navigate("/settings")}
              className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-[var(--color-text-secondary)] cursor-pointer hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)] data-[selected=true]:bg-[var(--color-bg-elevated)] data-[selected=true]:text-[var(--color-text-primary)]"
            >
              <Settings className="h-4 w-4" />
              Settings
            </Command.Item>
          </Command.Group>

          {/* Workspace */}
          {isWorkspace && (
            <Command.Group heading="Workspace" className="px-2 pb-1">
              <Command.Item
                value="workspace-overview"
                onSelect={() => navigate("/workspace")}
                className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-[var(--color-text-secondary)] cursor-pointer hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)] data-[selected=true]:bg-[var(--color-bg-elevated)] data-[selected=true]:text-[var(--color-text-primary)]"
              >
                <Layers className="h-4 w-4" />
                Workspace Overview
              </Command.Item>
              <Command.Item
                value="workspace-contracts"
                onSelect={() => navigate("/workspace/contracts")}
                className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-[var(--color-text-secondary)] cursor-pointer hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)] data-[selected=true]:bg-[var(--color-bg-elevated)] data-[selected=true]:text-[var(--color-text-primary)]"
              >
                <Link2 className="h-4 w-4" />
                Contracts
              </Command.Item>
              <Command.Item
                value="workspace-co-changes"
                onSelect={() => navigate("/workspace/co-changes")}
                className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-[var(--color-text-secondary)] cursor-pointer hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)] data-[selected=true]:bg-[var(--color-bg-elevated)] data-[selected=true]:text-[var(--color-text-primary)]"
              >
                <GitMerge className="h-4 w-4" />
                Co-Changes
              </Command.Item>
            </Command.Group>
          )}

          {/* Repos */}
          {repos.length > 0 && (
            <Command.Group heading="Repositories" className="px-2 pb-1">
              {repos.map((repo) => (
                <Command.Item
                  key={repo.id}
                  value={`repo-${repo.name}`}
                  onSelect={() => navigate(`/repos/${repo.id}/overview`)}
                  className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm text-[var(--color-text-secondary)] cursor-pointer hover:bg-[var(--color-bg-elevated)] hover:text-[var(--color-text-primary)] data-[selected=true]:bg-[var(--color-bg-elevated)] data-[selected=true]:text-[var(--color-text-primary)]"
                >
                  <BookOpen className="h-4 w-4" />
                  {repo.name}
                </Command.Item>
              ))}
            </Command.Group>
          )}

          {/* File jump */}
          {activeRepo && fileMatches.length > 0 && (
            <Command.Group heading="Files" className="px-2 pb-1">
              {fileMatches.map((path) => {
                const name = path.split("/").pop() ?? path;
                const dir = path.slice(0, path.length - name.length);
                return (
                  <Command.Item
                    key={path}
                    value={`file ${path}`}
                    keywords={preRankedKeywords}
                    onSelect={() =>
                      navigate(fileEntityPath(`/repos/${activeRepo.id}`, path))
                    }
                    className="flex items-center gap-2.5 rounded-md px-3 py-2 text-sm cursor-pointer hover:bg-[var(--color-bg-elevated)] data-[selected=true]:bg-[var(--color-bg-elevated)]"
                  >
                    <FileCode className="h-4 w-4 shrink-0 text-[var(--color-text-tertiary)]" />
                    <span className="min-w-0 truncate font-mono text-[13px]">
                      <span className="text-[var(--color-text-tertiary)]">
                        {truncatePath(dir, 36)}
                      </span>
                      <span className="text-[var(--color-text-primary)]">{name}</span>
                    </span>
                  </Command.Item>
                );
              })}
            </Command.Group>
          )}

          {/* Search results, in sections: Code, Documentation, Decisions. */}
          {grouped.groups.map((group) => {
            const Icon = GROUP_ICON[group.id];
            return (
              <Command.Group
                key={group.id}
                heading={
                  // The caution note rides on the first section rather than
                  // above the list: it qualifies the answers, and a reader who
                  // never scrolls to them does not need warning about them.
                  lowConfidence && group.id === grouped.groups[0]?.id ? (
                    <span className="flex items-baseline gap-2">
                      {group.heading}
                      <span className="text-[10px] font-normal normal-case tracking-normal text-[var(--color-text-tertiary)]">
                        low confidence — no exact match, closest by meaning
                      </span>
                    </span>
                  ) : (
                    group.heading
                  )
                }
                className="px-2 pb-1"
              >
                {group.entries.map((entry) => {
                  const lines = formatLineRange(entry);
                  return (
                    <Command.Item
                      key={entry.key}
                      value={`${group.id}-${entry.key}`}
                      keywords={preRankedKeywords}
                      onSelect={() => navigate(entry.href)}
                      className="flex items-start gap-2.5 rounded-md px-3 py-2 text-sm cursor-pointer hover:bg-[var(--color-bg-elevated)] data-[selected=true]:bg-[var(--color-bg-elevated)]"
                    >
                      <Icon className="mt-0.5 h-4 w-4 shrink-0 text-[var(--color-text-tertiary)]" />
                      <span className="flex min-w-0 flex-col items-start">
                        <span className="truncate text-[var(--color-text-primary)] font-medium">
                          {entry.label}
                        </span>
                        <span className="truncate text-xs text-[var(--color-text-tertiary)] font-mono">
                          {truncatePath(entry.detail, 46)}
                          {lines && `:${lines}`}
                          {entry.alsoMatched
                            ? ` +${entry.alsoMatched} more`
                            : ""}
                        </span>
                      </span>
                    </Command.Item>
                  );
                })}
              </Command.Group>
            );
          })}
        </Command.List>

        <div className="border-t border-[var(--color-border-default)] px-4 py-2 flex items-center gap-4 text-xs text-[var(--color-text-tertiary)]">
          <span>↑↓ navigate</span>
          <span>↵ select</span>
          <span>esc close</span>
        </div>
      </div>
    </Command.Dialog>
  );
}
