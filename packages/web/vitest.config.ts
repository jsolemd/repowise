import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    // Let jsdom own DOM storage instead of Node 25's originless placeholder.
    execArgv: process.allowedNodeEnvironmentFlags.has("--no-experimental-webstorage")
      ? ["--no-experimental-webstorage"]
      : [],
  },
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
});
