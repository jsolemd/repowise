/**
 * Meta endpoints: repowise version freshness + release changelog.
 * Powers the upgrade banner, what's-new view, and version footer.
 */

import { apiGet } from "./client";

export interface MetaVersion {
  server_version: string;
  latest_version: string | null;
  /** null when PyPI could not be reached (distinct from "up to date"). */
  update_available: boolean | null;
  upgrade_command: string;
  store_format_version: number | null;
  store_compatible: boolean | null;
  reindex_recommended: boolean;
  reindex_command: string | null;
}

/**
 * What this deployment forbids. Cheap enough to fetch on every render, and
 * fetched server-side, so a disabled affordance is never drawn and then removed.
 */
export interface DeploymentPolicy {
  /** True when the hard no-generative policy is in force. Every surface that
   *  would reach a generative provider is refused server-side. */
  generative_disabled: boolean;
  /** The env var that produced the answer, so a missing feature is findable. */
  generative_policy_source: string;
}

export interface ChangelogSection {
  name: string;
  items: string[];
}

export interface ChangelogEntry {
  version: string;
  label: string | null;
  sections: ChangelogSection[];
}

export interface ChangelogData {
  entries: ChangelogEntry[];
}

export async function getMetaVersion(repoId?: string): Promise<MetaVersion> {
  return apiGet<MetaVersion>(
    "/api/meta/version",
    repoId ? { repo_id: repoId } : undefined,
  );
}

export async function getDeploymentPolicy(
  repoId?: string,
): Promise<DeploymentPolicy> {
  return apiGet<DeploymentPolicy>(
    "/api/meta/policy",
    repoId ? { repo_id: repoId } : undefined,
  );
}

export async function getChangelog(limit = 20): Promise<ChangelogData> {
  return apiGet<ChangelogData>("/api/meta/changelog", { limit });
}
