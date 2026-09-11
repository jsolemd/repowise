import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config.mts";

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: "jsdom",
      // Let jsdom own DOM storage instead of Node 25's originless placeholder.
      execArgv: process.allowedNodeEnvironmentFlags.has("--no-experimental-webstorage")
        ? ["--no-experimental-webstorage"]
        : [],
      include: ["src/**/*.test.{ts,tsx}"],
      globals: false,
      passWithNoTests: true,
    },
  }),
);
