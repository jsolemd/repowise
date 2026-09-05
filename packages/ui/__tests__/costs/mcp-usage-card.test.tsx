import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { McpUsageCard, type McpUsageData } from "../../src/costs/mcp-usage-card";

function makeData(overrides: Partial<McpUsageData> = {}): McpUsageData {
  return {
    available: true,
    mcp_usage_calls: 14,
    mcp_usage_error_calls: 1,
    mcp_usage_no_match_calls: 2,
    mcp_usage_degraded_calls: 3,
    mcp_usage_avg_duration_ms: 42,
    mcp_usage_window_days: 30,
    mcp_usage_per_tool: [
      {
        tool: "search_codebase",
        calls: 9,
        error_calls: 0,
        no_match_calls: 2,
        degraded_calls: 1,
        avg_duration_ms: 35,
        saving_calls: 7,
        positive_saving_calls: 6,
        saved_tokens: 12_000,
      },
    ],
    ...overrides,
  };
}

describe("McpUsageCard", () => {
  it("shows retained usage and outcome counts by tool", () => {
    render(<McpUsageCard data={makeData()} />);

    expect(screen.getByText("RepoWise MCP usage")).toBeInTheDocument();
    expect(screen.getByText("search_codebase")).toBeInTheDocument();
    expect(screen.getByText("Rolling 30-day local totals, aggregated by tool and day.")).toBeInTheDocument();
    expect(screen.getByText("14")).toBeInTheDocument();
    expect(screen.getByText("42 ms")).toBeInTheDocument();
  });

  it("states the privacy boundary instead of implying a query ledger", () => {
    render(<McpUsageCard data={makeData()} />);
    expect(
      screen.getByText(/No query text, targets, paths, sessions, or individual call history/),
    ).toBeInTheDocument();
  });

  it("has an honest empty state before the first aggregated call", () => {
    render(<McpUsageCard data={makeData({ mcp_usage_calls: 0, mcp_usage_per_tool: [] })} />);
    expect(screen.getByText(/No MCP calls have been aggregated/)).toBeInTheDocument();
  });
});
