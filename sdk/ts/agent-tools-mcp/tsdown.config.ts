import { defineConfig } from "tsdown";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "eve-tools": "src/eve-tools.ts",
    cli: "src/cli.ts",
  },
  format: "esm",
  dts: true,
  clean: true,
  treeshake: true,
});
