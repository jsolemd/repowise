"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState, type FormEvent } from "react";
import useSWR from "swr";
import { Button } from "@repowise-dev/ui/ui/button";
import { Input } from "@repowise-dev/ui/ui/input";
import { ApiError } from "@repowise-dev/ui/shared/api-error";
import { Markdown } from "@repowise-dev/ui/shared/markdown";
import { PaginationControls } from "@repowise-dev/ui/shared/pagination-controls";
import { OverviewSection } from "@repowise-dev/ui/overview";
import { callDocsTool, documentsHref, type DocFilesPage, type DocContent, type DocSearchResults } from "@/lib/api/docs-libraries";
import { RefreshDocLibrary } from "./library-actions";

const OPTIONS = { revalidateOnFocus: false, revalidateOnReconnect: false, shouldRetryOnError: false };

export function DocumentBrowser() {
  const params = useSearchParams();
  const library = params.get("library") || "";
  const path = params.get("path") || "";
  if (!library) return <p>Select a library from the <Link className="underline" href="/docs-libraries">documentation inventory</Link>.</p>;
  return <LibraryDocuments key={library} library={library} path={path} />;
}

function LibraryDocuments({ library, path }: { library: string; path: string }) {
  const [mode, setMode] = useState<"files" | "search">("files");
  const [query, setQuery] = useState("");
  const [draft, setDraft] = useState("");
  const [offset, setOffset] = useState(0);
  const files = useSWR(["doc-files", library, mode === "files" ? query : "", offset],
    ([, id, filter, start]) => callDocsTool<DocFilesPage>("list_doc_files", { library_id: id, query: filter, offset: start, limit: 50 }), OPTIONS);
  const search = useSWR(mode === "search" && query ? ["doc-search", library, query] : null,
    ([, id, text]) => callDocsTool<DocSearchResults>("search_docs", { library_id: id, query: text, limit: 20 }), OPTIONS);
  const document = useSWR(path ? ["doc-content", library, path] : null,
    ([, id, file]) => callDocsTool<DocContent>("read_doc", { library_id: id, path: file, max_tokens: 20000 }), OPTIONS);
  function submit(event: FormEvent) { event.preventDefault(); setQuery(draft.trim()); setOffset(0); }
  const rows = mode === "files" ? files.data?.files.map(file => ({ path: file.file_path, detail: `${file.chunk_count} chunks`, key: file.file_path }))
    : search.data?.results.map(hit => ({ path: hit.file_path, detail: hit.preview, key: hit.chunk_id }));
  const current = mode === "files" ? files : search;
  return <div className="space-y-6">
    <Link href="/docs-libraries" className="text-sm text-[var(--color-accent-primary)] hover:underline">← All libraries</Link>
    <OverviewSection title={files.data?.library.name || "Library documents"}>
      <div className="space-y-4">
        {files.data && <p className="text-sm text-[var(--color-text-secondary)]">
          {files.data.library.status} · {files.data.library.source_type} · Indexed {files.data.library.indexed_at ? new Date(files.data.library.indexed_at).toLocaleString() : "never"}
        </p>}
        <RefreshDocLibrary library={library} onQueued={() => void files.mutate()} />
        <form onSubmit={submit} className="flex flex-col gap-3 sm:flex-row">
          <label className="sr-only" htmlFor="doc-search-mode">Search mode</label>
          <select id="doc-search-mode" value={mode} onChange={event => { setMode(event.target.value as "files" | "search"); setQuery(""); setDraft(""); setOffset(0); }} className="rounded-md border border-[var(--color-border-default)] bg-[var(--color-bg-surface)] px-3 py-2 text-sm">
            <option value="files">File names</option><option value="search">Document contents</option>
          </select>
          <label className="sr-only" htmlFor="doc-query">{mode === "files" ? "Filter file names" : "Search document contents"}</label>
          <Input id="doc-query" value={draft} onChange={event => setDraft(event.target.value)} placeholder={mode === "files" ? "Filter file names…" : "Search document contents…"} />
          <Button type="submit">{mode === "files" ? "Filter" : "Search"}</Button>
        </form>
        {current.error && <ApiError title="Documents unavailable" message={current.error.message} onRetry={() => void current.mutate()} />}
        {current.isLoading && <p role="status">Loading documents…</p>}
        {mode === "search" && !query && <p>Enter a query to search this library.</p>}
        {search.data?.warnings?.map(warning => <p key={warning} role="status" className="text-sm text-[var(--color-warning)]">{warning}</p>)}
        {!current.error && rows?.length === 0 && <p>No documents found.</p>}
        {!!rows?.length && <ul className="divide-y divide-[var(--color-border-default)]">
          {rows.map(row => <li key={row.key} className="py-3">
            <Link href={`${documentsHref(library, row.path)}#doc-preview`} aria-current={path === row.path ? "page" : undefined} className="block font-mono text-sm text-[var(--color-accent-primary)] hover:underline [overflow-wrap:anywhere]">{row.path}</Link>
            <p className="mt-1 line-clamp-3 text-sm text-[var(--color-text-secondary)] [overflow-wrap:anywhere]">{row.detail}</p>
          </li>)}
        </ul>}
        {mode === "files" && files.data && <PaginationControls offset={offset} shown={files.data.files.length} total={files.data.pagination.total} label="documents"
          onPrevious={offset > 0 ? () => setOffset(Math.max(0, offset - 50)) : undefined}
          onNext={files.data.pagination.has_more ? () => setOffset(files.data!.pagination.next_offset!) : undefined} />}
        {mode === "search" && search.data && <p className="text-xs text-[var(--color-text-tertiary)]">Top {search.data.results.length} matching passages. Refine your query to narrow the results.</p>}
      </div>
    </OverviewSection>
    {path && <div id="doc-preview" className="scroll-mt-24"><OverviewSection title="Document preview">
      <div className="min-w-0 space-y-4">
        <p className="font-mono text-sm [overflow-wrap:anywhere]">{path}</p>
        {document.error && <ApiError title="Preview unavailable" message={document.error.message} onRetry={() => void document.mutate()} />}
        {document.isLoading && <p role="status">Loading document…</p>}
        {document.data && <>
          {document.data.truncated && <p role="status" className="text-sm text-[var(--color-warning)]">This preview is truncated.</p>}
          {/\.(md|mdx|markdown|rst|txt)$/i.test(path) ? <Markdown content={document.data.content} /> : <pre className="max-w-full overflow-auto text-sm"><code>{document.data.content}</code></pre>}
        </>}
      </div>
    </OverviewSection></div>}
  </div>;
}
