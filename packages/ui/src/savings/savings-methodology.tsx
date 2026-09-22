"use client";

import * as React from "react";

import { CollapsibleSection } from "../shared/collapsible-section";
import { DismissibleNotice } from "../shared/dismissible-notice";

/**
 * The accounting-method version this build reports under.
 *
 * Hosts key the reset notice's dismissal on repository plus this value, so a
 * later methodology change announces itself once more to a reader who
 * dismissed the previous one. Bump it when the way a saving is counted
 * changes, not when a component moves.
 */
export const ACCOUNTING_METHOD_VERSION = 1;

export interface SavingsResetNoticeProps {
  /** Where "See what changed" goes. */
  methodologyHref?: string | undefined;
  onDismiss: () => void;
  LinkComponent?: React.ElementType | undefined;
}

/**
 * Said once, when a repository's reader has not yet seen it.
 *
 * Distillation and hook savings use the upstream event ledger. This fork keeps
 * MCP usage as bounded daily aggregates and preserves older savings records;
 * the notice must distinguish those stores without claiming a history reset.
 *
 * The host decides whether this renders and remembers the dismissal. This
 * component only knows how to say it.
 */
export function SavingsResetNotice({
  methodologyHref,
  onDismiss,
  LinkComponent,
}: SavingsResetNoticeProps) {
  const A = LinkComponent ?? "a";
  return (
    <DismissibleNotice
      tone="info"
      onDismiss={onDismiss}
      dismissLabel="Dismiss the savings accounting notice"
      {...(methodologyHref
        ? {
            action: (
              <A
                href={methodologyHref}
                className="text-[var(--color-accent-primary)] hover:underline"
              >
                See what changed
              </A>
            ),
          }
        : {})}
    >
      <span className="font-medium text-[var(--color-text-primary)]">
        Savings accounting has been upgraded.
      </span>{" "}
      New distillation and hook savings are recorded per interaction. Earlier estimates
      were not deleted, but are excluded from these new totals. MCP usage remains a
      separate rolling 30-day total by tool; individual MCP calls are not stored.
    </DismissibleNotice>
  );
}

export interface SavingsMethodologyProps {
  /** Link to the full accounting document. */
  href?: string | undefined;
  LinkComponent?: React.ElementType | undefined;
}

/**
 * How the numbers on this page were arrived at, folded away by default.
 *
 * Quiet rather than absent: a page that asks to be trusted about token counts
 * has to be able to show its working, but the working is not what a reader
 * came for. Collapsed, at the end, in full.
 */
export function SavingsMethodology({ href, LinkComponent }: SavingsMethodologyProps) {
  const A = LinkComponent ?? "a";
  return (
    <CollapsibleSection title="Method and limits">
      <div className="flex max-w-[72ch] flex-col gap-3 text-[13px] leading-relaxed text-[var(--color-text-secondary)]">
        <p>
          A saving is input tokens your agent never had to read. One logical interaction
          through <code>repowise distill</code> or a replacement hook produces one event
          and contributes to the total once. In this fork, MCP calls contribute only to
          the separate daily usage aggregates, with no individual call history.
        </p>
        <p>
          <strong className="font-medium text-[var(--color-text-primary)]">Measured</strong>{" "}
          figures compare a known before and after from an operation that actually ran.{" "}
          <strong className="font-medium text-[var(--color-text-primary)]">Inferred</strong>{" "}
          figures are a documented estimate of the exploration a Repowise answer replaced.
          They are reported apart because they are different evidence, and the headline says
          Estimated for as long as any inferred figure is in it.
        </p>
        <p>
          Observed opportunities are behaviour that could have been optimised and was not.
          They are never added to achieved savings.
        </p>
        <p>
          Each event is valued at the rate recorded at the moment it happened, so history is
          never repriced from today's rates. An event recorded before a rate could be
          resolved is counted in tokens and left unvalued, which is why priced and unpriced
          totals are shown separately.
        </p>
        <p>
          Token counts are estimates from character length and deliberately undersell.
          Dead ends and errors save nothing and are counted separately from answered
          queries. Everything on this page was recorded on this machine and stays here.
          {href && (
            <>
              {" "}
              <A href={href} className="text-[var(--color-accent-primary)] hover:underline">
                Read the full accounting document
              </A>
            </>
          )}
        </p>
      </div>
    </CollapsibleSection>
  );
}
