"use client";

import { createContext, useContext } from "react";
import type { DeploymentPolicy } from "@/lib/api/meta";

/**
 * What this deployment forbids, resolved once server-side in the root layout.
 *
 * Threaded by context rather than by prop, because the consumers are scattered
 * chrome — the command palette, the sidebar and mobile nav, the overview's
 * ask row, the chat route — and drilling one boolean through all of them would
 * put the policy in the signature of components that have nothing to do with it.
 *
 * The default is the restrictive one. If the fetch failed and we genuinely do
 * not know, the dashboard shows one fewer affordance rather than offering a
 * chat the server will refuse; a missing button is a smaller lie than a broken
 * one, and the settings page still names the switch.
 */
const DeploymentPolicyContext = createContext<DeploymentPolicy>({
  generative_disabled: true,
  generative_policy_source: "REPOWISE_TOOLS_NO_GENERATIVE",
});

export function DeploymentPolicyProvider({
  policy,
  children,
}: {
  policy: DeploymentPolicy | null;
  children: React.ReactNode;
}) {
  return (
    <DeploymentPolicyContext.Provider
      value={
        policy ?? {
          generative_disabled: true,
          generative_policy_source: "REPOWISE_TOOLS_NO_GENERATIVE",
        }
      }
    >
      {children}
    </DeploymentPolicyContext.Provider>
  );
}

export function useDeploymentPolicy(): DeploymentPolicy {
  return useContext(DeploymentPolicyContext);
}

/** True when this deployment forbids every generative surface. */
export function useGenerativeDisabled(): boolean {
  return useContext(DeploymentPolicyContext).generative_disabled;
}
