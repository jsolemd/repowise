import type { Metadata } from "next";
import { Suspense } from "react";
import { PageShell } from "@repowise-dev/ui/shared/page-shell";
import { DocumentBrowser } from "@/components/docs-libraries/document-browser";

export const metadata: Metadata = { title: "Documents" };

export default function DocumentsPage() {
  return (
    <PageShell title="Documents" description="Browse indexed files and search their contents.">
      <Suspense fallback={<p>Loading documents…</p>}><DocumentBrowser /></Suspense>
    </PageShell>
  );
}
