import { readdir } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { type DefinedTool, defineTool } from "./define-tool.js";
import type { RunnableTool } from "./run-tool.js";
import {
  inputSchemaToJsonSchema,
  isPlainJsonSchema,
  isStandardSchema,
} from "./schema.js";

export type LoadedTool = {
  name: string;
  /** Concrete Zod tools erase here at the load boundary. */
  def: RunnableTool;
  sourcePath: string;
};

const TOOL_EXT = /\.(ts|mts|js|mjs)$/;

function looksLikeToolDef(value: unknown): value is {
  description: string;
  inputSchema: unknown;
  execute: (...args: never[]) => unknown;
} {
  if (value === null || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.description === "string"
    && v.inputSchema !== null
    && v.inputSchema !== undefined
    && typeof v.execute === "function"
  );
}

/** Accept `defineTool(...)` results or a raw Eve-shaped object (normalize through defineTool). */
function asDefinedTool(value: unknown, sourcePath: string): DefinedTool {
  if (looksLikeToolDef(value) && "jsonInputSchema" in (value as object)) {
    return value as DefinedTool;
  }
  if (!looksLikeToolDef(value)) {
    throw new Error(
      `${sourcePath}: default export must be defineTool({ description, inputSchema, execute })`,
    );
  }
  if (
    !isStandardSchema(value.inputSchema)
    && !isPlainJsonSchema(value.inputSchema)
  ) {
    throw new Error(
      `${sourcePath}: inputSchema must be Zod/Standard Schema or a plain JSON Schema object`,
    );
  }
  try {
    return defineTool(value as Parameters<typeof defineTool>[0]);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`${sourcePath}: ${msg}`);
  }
}

/** Load every `tools/*.{ts,js}` default export as a tool named by its file stem. */
export async function loadTools(dir: string): Promise<LoadedTool[]> {
  const root = path.resolve(dir);
  const names = await readdir(root);
  const files = names
    .filter((n) => TOOL_EXT.test(n) && !n.endsWith(".d.ts"))
    .sort();

  const out: LoadedTool[] = [];
  for (const file of files) {
    const sourcePath = path.join(root, file);
    const mod = (await import(pathToFileURL(sourcePath).href)) as {
      default?: unknown;
    };
    const def = asDefinedTool(mod.default, sourcePath);
    // Re-check JSON Schema is materializable (defineTool already did; keep assert for raw paths).
    inputSchemaToJsonSchema(def.inputSchema);
    const name = file.replace(TOOL_EXT, "");
    if (!name) {
      throw new Error(`${sourcePath}: empty tool name`);
    }
    out.push({ name, def, sourcePath });
  }
  return out;
}
