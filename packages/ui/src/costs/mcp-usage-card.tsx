"use client";

import { Activity, Clock3 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Skeleton } from "../ui/skeleton";

export interface McpUsageTool {
  tool: string;
  calls: number;
  error_calls: number;
  no_match_calls: number;
  degraded_calls: number;
  avg_duration_ms: number;
  saving_calls: number;
  positive_saving_calls: number;
  saved_tokens: number;
}

export interface McpUsageData {
  available: boolean;
  mcp_usage_calls?: number;
  mcp_usage_error_calls?: number;
  mcp_usage_no_match_calls?: number;
  mcp_usage_degraded_calls?: number;
  mcp_usage_avg_duration_ms?: number;
  mcp_usage_window_days?: number;
  mcp_usage_per_tool?: McpUsageTool[];
}

export interface McpUsageCardProps {
  data?: McpUsageData;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-lg font-semibold tabular-nums text-[var(--color-text-primary)]">
        {value}
      </div>
      <div className="text-xs text-[var(--color-text-tertiary)]">{label}</div>
    </div>
  );
}

/** A factual, privacy-bounded view of which MCP tools agents actually use. */
export function McpUsageCard({ data }: McpUsageCardProps) {
  if (!data) {
    return <Skeleton className="h-[238px] w-full rounded-xl" />;
  }

  const calls = data.mcp_usage_calls ?? 0;
  const windowDays = data.mcp_usage_window_days ?? 30;
  const tools = (data.mcp_usage_per_tool ?? []).slice(0, 10);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2 text-sm">
              <Activity className="h-4 w-4 text-[var(--color-accent-primary)]" />
              RepoWise MCP usage
            </CardTitle>
            <p className="mt-1 text-xs text-[var(--color-text-secondary)]">
              Rolling {windowDays}-day local totals, aggregated by tool and day.
            </p>
          </div>
          <p className="max-w-md text-right text-[11px] leading-relaxed text-[var(--color-text-tertiary)] max-sm:text-left">
            No query text, targets, paths, sessions, or individual call history is stored.
          </p>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {calls === 0 ? (
          <p className="py-5 text-sm text-[var(--color-text-secondary)]">
            No MCP calls have been aggregated in this window yet.
          </p>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-4 border-y border-[var(--color-border-default)] py-4 sm:grid-cols-5">
              <Metric label="calls" value={calls.toLocaleString()} />
              <Metric
                label="no match"
                value={(data.mcp_usage_no_match_calls ?? 0).toLocaleString()}
              />
              <Metric
                label="errors"
                value={(data.mcp_usage_error_calls ?? 0).toLocaleString()}
              />
              <Metric
                label="degraded"
                value={(data.mcp_usage_degraded_calls ?? 0).toLocaleString()}
              />
              <Metric
                label="average latency"
                value={`${Math.round(data.mcp_usage_avg_duration_ms ?? 0).toLocaleString()} ms`}
              />
            </div>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[560px] text-sm">
                <caption className="sr-only">MCP usage by tool</caption>
                <thead>
                  <tr className="border-b border-[var(--color-border-default)] text-xs text-[var(--color-text-tertiary)]">
                    <th className="pb-2 text-left font-medium">Tool</th>
                    <th className="pb-2 text-right font-medium">Calls</th>
                    <th className="pb-2 text-right font-medium">No match</th>
                    <th className="pb-2 text-right font-medium">Errors</th>
                    <th className="pb-2 text-right font-medium">Degraded</th>
                    <th className="pb-2 text-right font-medium">
                      <span className="inline-flex items-center gap-1">
                        <Clock3 className="h-3 w-3" /> Avg
                      </span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {tools.map((tool) => (
                    <tr
                      key={tool.tool}
                      className="border-b border-[var(--color-border-subtle)] last:border-0"
                    >
                      <td className="py-2 font-mono text-xs text-[var(--color-text-primary)]">
                        {tool.tool}
                      </td>
                      <td className="py-2 text-right tabular-nums">{tool.calls.toLocaleString()}</td>
                      <td className="py-2 text-right tabular-nums text-[var(--color-text-secondary)]">
                        {tool.no_match_calls.toLocaleString()}
                      </td>
                      <td className="py-2 text-right tabular-nums text-[var(--color-text-secondary)]">
                        {tool.error_calls.toLocaleString()}
                      </td>
                      <td className="py-2 text-right tabular-nums text-[var(--color-text-secondary)]">
                        {tool.degraded_calls.toLocaleString()}
                      </td>
                      <td className="py-2 text-right tabular-nums text-[var(--color-text-secondary)]">
                        {Math.round(tool.avg_duration_ms).toLocaleString()} ms
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
