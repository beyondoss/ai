import { writeFile } from "node:fs/promises";

import type { LoadedTool } from "./load.js";

/** Shape `beyond-agent` reads from a bundle's `.mcp-tools.json` (`CachedTool`). */
export type CachedTool = {
  remote_name: string;
  description: string;
  input_schema: Record<string, unknown>;
};

export type ManifestFile = Record<string, CachedTool[]>;

export function toolsToCached(tools: LoadedTool[]): CachedTool[] {
  // Eve: deferred tools are not in the model's default list (cold-open menu matches).
  return tools
    .filter((t) => !t.def.deferred)
    .map((t) => ({
      remote_name: t.name,
      description: t.def.description,
      input_schema: t.def.jsonInputSchema,
    }));
}

/** Write `{"<connector>": [CachedTool, ...]}` for the bundle root. */
export async function writeManifest(
  tools: LoadedTool[],
  opts: { name: string; out: string },
): Promise<ManifestFile> {
  const file: ManifestFile = { [opts.name]: toolsToCached(tools) };
  await writeFile(opts.out, `${JSON.stringify(file, null, 2)}\n`, "utf8");
  return file;
}
