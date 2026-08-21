import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getRepo } from "@/lib/api/repos";
import { getDeploymentPolicy } from "@/lib/api/meta";
import { ChatInterface } from "@/components/chat/chat-interface";
import { ChatDisabledNotice } from "@/components/chat/chat-disabled-notice";

interface Props {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ q?: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { id } = await params;
  try {
    const repo = await getRepo(id);
    return { title: `${repo.name} — Chat` };
  } catch {
    return { title: "Repository" };
  }
}

/**
 * Thin shell. The page used to render its own header — repo name, local path,
 * branch and SHA — directly under the repo breadcrumb and directly above the
 * chat's own control row, so three hairlines ran before the first word of
 * content and the repo name appeared twice. The breadcrumb already names the
 * repo; branch and SHA are orientation, so they ride the empty state's status
 * line next to the file and doc counts, where the other figures already are.
 */
export default async function RepoChatPage({ params, searchParams }: Props) {
  const { id } = await params;
  const { q } = await searchParams;

  let repo;
  try {
    repo = await getRepo(id);
  } catch {
    notFound();
  }

  // Asked per repo, not read from the layout's server-wide answer: a workspace
  // can set the policy on one repo and not another, and the server refuses per
  // repo, so the page has to agree with the route that would serve it.
  try {
    const policy = await getDeploymentPolicy(id);
    if (policy.generative_disabled) {
      return (
        <ChatDisabledNotice
          repoId={id}
          policySource={policy.generative_policy_source}
        />
      );
    }
  } catch {
    // Policy unknown — fall through and let the chat route answer for itself
    // rather than hiding a working feature behind a failed lookup.
  }

  return (
    <div className="flex h-full flex-col">
      <ChatInterface
        repoId={id}
        repoName={repo.name}
        defaultBranch={repo.default_branch}
        {...(repo.head_commit ? { headCommit: repo.head_commit } : {})}
        {...(q ? { initialQuestion: q } : {})}
      />
    </div>
  );
}
