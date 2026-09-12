"use client";

import { useId, useRef, useState, type FormEvent } from "react";
import { Button } from "@repowise-dev/ui/ui/button";
import { Input } from "@repowise-dev/ui/ui/input";
import { callDocsTool } from "@/lib/api/docs-libraries";

export function AddDocLibrary({ onAdded }: { onAdded: () => void }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const formId = useId();
  const toggle = useRef<HTMLButtonElement>(null);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const fields = new FormData(event.currentTarget);
    const docsPath = String(fields.get("docs_path") ?? "").trim();
    setBusy(true);
    setMessage("");
    try {
      const result = await callDocsTool<{
        disposition: "created" | "exists";
        library_name: string;
        current_status?: string;
      }>("add_doc_library", {
        repo: String(fields.get("repo")).trim(),
        name: String(fields.get("name")).trim(),
        branch: String(fields.get("branch") ?? "").trim() || "main",
        ...(docsPath ? { docs_path: docsPath } : {}),
      });
      setMessage(result.disposition === "exists"
        ? `${result.library_name} is already registered (${result.current_status}).`
        : "Library added. Indexing is queued.");
      setOpen(false);
      toggle.current?.focus();
      onAdded();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Could not add library."); }
    finally { setBusy(false); }
  }
  return (
    <div className="space-y-3">
      <Button ref={toggle} type="button" variant="outline" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls={open ? formId : undefined}>Add library</Button>
      {open && <form id={formId} aria-label="Add documentation library" onSubmit={submit} className="grid gap-3 rounded-lg border border-[var(--color-border-default)] p-4 sm:grid-cols-2">
        <label className="text-sm">Library name<Input name="name" required maxLength={160} placeholder="React" /></label>
        <label className="text-sm">GitHub repository<Input name="repo" required placeholder="facebook/react" /></label>
        <label className="text-sm">Branch<Input name="branch" defaultValue="main" placeholder="main" /></label>
        <label className="text-sm">Docs directory (optional)<Input name="docs_path" placeholder="docs" /></label>
        <div className="flex items-end"><Button type="submit" disabled={busy}>{busy ? "Adding…" : "Add and index"}</Button></div>
      </form>}
      {message && <p role="status" className="text-sm">{message}</p>}
    </div>
  );
}

export function RefreshDocLibrary({ library, onQueued }: { library: string; onQueued: () => void }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  async function refresh() {
    setBusy(true); setMessage("");
    try {
      const result = await callDocsTool<{ disposition: string }>("update_doc_library", { library_id: library });
      setMessage(`Refresh ${result.disposition}.`); onQueued();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Could not queue refresh."); }
    finally { setBusy(false); }
  }
  return <div className="flex flex-wrap items-center gap-3">
    <Button variant="outline" disabled={busy} onClick={() => void refresh()}>{busy ? "Queuing…" : "Refresh library"}</Button>
    {message && <span role="status" className="text-sm">{message}</span>}
  </div>;
}
