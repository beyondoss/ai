import { randomBytes } from "node:crypto";
import { rename, unlink, writeFile } from "node:fs/promises";
import path from "node:path";

import type { ToolAnnotations } from "./define-tool.js";
import type { LoadedTool } from "./load.js";
import { modelFacingDescription } from "./rpc.js";

/** Shape `beyond-agent` reads from a bundle's `.mcp-tools.json` (`CachedTool`). */
export type CachedTool = {
  remote_name: string;
  description: string;
  input_schema: Record<string, unknown>;
  output_schema?: Record<string, unknown>;
  annotations?: ToolAnnotations;
  /** `false` hides the tool from subagents. Omitted means they inherit it. */
  available_in_subagents?: boolean;
};

export type ManifestFile = Record<string, CachedTool[]>;

export function toolsToCached(tools: LoadedTool[]): CachedTool[] {
  // Eve: deferred tools are not in the model's default list (cold-open menu matches).
  return tools
    .filter((t) => !t.def.deferred)
    .map((t) => {
      const cached: CachedTool = {
        remote_name: t.name,
        description: modelFacingDescription(t.def.description, t.def.endsTurn),
        input_schema: t.def.jsonInputSchema,
      };
      if (t.def.jsonOutputSchema) cached.output_schema = t.def.jsonOutputSchema;
      if (t.def.annotations) cached.annotations = t.def.annotations;
      if (t.def.availableInSubagents === false) {
        cached.available_in_subagents = false;
      }
      return cached;
    });
}

/**
 * Write `{"<connector>": [CachedTool, ...]}` for the bundle root.
 * The bytes land via a temp file in the same directory and `rename`, so a
 * crash cannot leave the agent a half-written menu.
 */
export async function writeManifest(
  tools: LoadedTool[],
  opts: { name: string; out: string },
): Promise<ManifestFile> {
  const file: ManifestFile = { [opts.name]: toolsToCached(tools) };
  const body = `${JSON.stringify(file, null, 2)}\n`;
  const dir = path.dirname(opts.out);
  const tmp = path.join(
    dir,
    `.${path.basename(opts.out)}.${process.pid}.${randomBytes(4).toString("hex")}.tmp`,
  );
  try {
    await writeFile(tmp, body, "utf8");
    await rename(tmp, opts.out);
  } catch (err) {
    await unlink(tmp).catch(() => {});
    throw err;
  }
  return file;
}
