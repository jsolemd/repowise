"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

/** A burst of catalog writes (a build publishing several outputs) lands as one refresh. */
const SETTLE_MS = 800;

type State = { connection: "connecting" | "live" | "reconnecting"; revision: number | null };
const CONNECTION_LABEL: Record<State["connection"], string> = {
  connecting: "Connecting",
  live: "Live",
  reconnecting: "Reconnecting",
};

/**
 * Re-renders the page when Make's catalog changes.
 *
 * Make relays each catalog revision (`/v1/events`, through this app's
 * `/make-platform/events`); the page itself stays a server render, so a
 * change is a `router.refresh()` rather than a client-side patch of numbers.
 * The first event is the revision the page was drawn at, not a change.
 */
export function LiveRefresh({ initialRevision }: { initialRevision: number | null }) {
  const router = useRouter();
  const [state, setState] = useState<State>({ connection: "connecting", revision: initialRevision });
  const seen = useRef<number | null>(initialRevision);

  useEffect(() => {
    const source = new EventSource("/make-platform/events");
    let timer: ReturnType<typeof setTimeout> | undefined;
    source.addEventListener("open", () => setState((s) => ({ ...s, connection: "live" })));
    source.addEventListener("error", () => setState((s) => ({ ...s, connection: "reconnecting" })));
    source.addEventListener("revision", (event) => {
      let revision: number | null = null;
      try {
        revision = Number((JSON.parse((event as MessageEvent<string>).data) as { catalog?: unknown }).catalog);
      } catch {
        return;
      }
      if (!Number.isFinite(revision)) return;
      setState({ connection: "live", revision });
      if (seen.current !== null && revision === seen.current) return;
      seen.current = revision;
      clearTimeout(timer);
      timer = setTimeout(() => router.refresh(), SETTLE_MS);
    });
    return () => {
      clearTimeout(timer);
      source.close();
    };
  }, [router]);

  return (
    <p className="inline-flex min-h-11 items-center gap-1.5 text-xs text-[var(--color-text-tertiary)]">
      <span
        aria-hidden
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: state.connection === "live" ? "var(--color-success)" : "var(--color-text-tertiary)" }}
      />
      {CONNECTION_LABEL[state.connection]}
      {state.revision !== null && <span className="tabular-nums">· revision {state.revision}</span>}
    </p>
  );
}
