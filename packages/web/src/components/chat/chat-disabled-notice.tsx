import { Ban } from "lucide-react";
import Link from "next/link";

/**
 * What the chat route shows when the deployment forbids generative surfaces.
 *
 * The affordances that lead here are hidden, so nobody should arrive by
 * clicking — but a bookmark, a shared link, or a policy that changed while a
 * tab was open all land here, and "not found" would be a lie. It names the
 * switch, because the person most likely to see this is the one who set it.
 */
export function ChatDisabledNotice({
  repoId,
  policySource,
}: {
  repoId: string;
  policySource: string;
}) {
  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="max-w-md text-center">
        <Ban
          className="mx-auto h-8 w-8 text-[var(--color-text-tertiary)]"
          aria-hidden
        />
        <h1 className="mt-4 text-base font-medium text-[var(--color-text-primary)]">
          Chat is off on this server
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-[var(--color-text-secondary)]">
          This deployment runs without generative models. Indexing, search, the
          knowledge graph and every other surface work as normal — only the ones
          that would call out to a model are disabled.
        </p>
        <p className="mt-3 font-mono text-xs text-[var(--color-text-tertiary)]">
          {policySource}=1
        </p>
        <Link
          href={`/repos/${repoId}/overview`}
          className="mt-5 inline-block text-sm text-[var(--color-accent-primary)] hover:underline"
        >
          Back to the repo
        </Link>
      </div>
    </div>
  );
}
