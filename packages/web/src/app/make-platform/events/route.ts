import { MakeReportError, openMakeEvents } from "@/lib/make-platform";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Make's catalog revisions, relayed to the browser as server-sent events.
 *
 * The browser cannot reach make-api's private socket, so this server route
 * holds one upstream stream per open page and closes it when the page leaves.
 * It carries revision numbers only; the page re-renders from the reports.
 * Outside `/api`, which the middleware rewrites to the RepoWise server.
 */
export async function GET(request: Request): Promise<Response> {
  const upstream = new AbortController();
  request.signal.addEventListener("abort", () => upstream.abort(), { once: true });
  let source;
  try {
    source = await openMakeEvents(upstream.signal);
  } catch (error) {
    const status = error instanceof MakeReportError ? error.status : 503;
    return new Response("Make events are unavailable.\n", { status: status === 200 ? 503 : status });
  }
  let open = true;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      const close = () => {
        if (!open) return;
        open = false;
        controller.close();
      };
      source.on("data", (chunk: Buffer) => {
        if (open) controller.enqueue(new Uint8Array(chunk));
      });
      source.on("end", close);
      source.on("error", close);
    },
    cancel() {
      open = false;
      upstream.abort();
      source.destroy();
    },
  });
  return new Response(body, {
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-store",
      "x-accel-buffering": "no",
    },
  });
}
