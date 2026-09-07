import type { Metadata } from "next";
import { BookOpen } from "lucide-react";
import { PageShell } from "@repowise-dev/ui/shared/page-shell";
import { DocsLibrariesSection } from "@/components/docs-libraries/docs-libraries-section";

export const metadata: Metadata = { title: "Documentation libraries" };

/**
 * The indexed documentation corpus and whether it can be believed.
 *
 * Global rather than per-repo: the libraries are shared by every repository an
 * agent asks about, so scoping this under one repo would imply an ownership
 * that does not exist.
 *
 * The page's real subject is the freshness column. A library whose check has
 * never succeeded is not fresh and not stale, and until this page existed
 * nothing said so — a copy months behind its source read exactly like a
 * current one to anyone searching the docs.
 */
export default function DocsLibrariesPage() {
  return (
    <PageShell
      title="Documentation libraries"
      icon={<BookOpen className="h-5 w-5 text-[var(--color-text-tertiary)]" />}
      description="Every external library the documentation service has indexed, how current each copy is, and what the indexer is working on."
    >
      <DocsLibrariesSection />
    </PageShell>
  );
}
