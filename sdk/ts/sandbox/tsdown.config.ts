import { defineConfig } from "tsdown";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    driver: "src/driver.ts",
  },
  format: "esm",
  dts: true,
  clean: true,
  treeshake: true,
});
