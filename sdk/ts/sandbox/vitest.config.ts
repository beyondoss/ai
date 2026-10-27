import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    testTimeout: 30_000,
    hookTimeout: 60_000,
    globalSetup: ["./__tests__/global-setup.ts"],
    include: ["__tests__/**/*.test.ts"],
  },
});
