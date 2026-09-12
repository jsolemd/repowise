import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  // `jsx: "preserve"` in tsconfig leaves esbuild on the classic transform, so
  // any component that does not import React itself fails to render under
  // test. Next compiles the app with the automatic runtime; match it here.
  esbuild: { jsx: "automatic", jsxImportSource: "react" },
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
