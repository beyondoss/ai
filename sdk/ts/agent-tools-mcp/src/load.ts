import { readdir } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

import type { ToolDefinition } from "./define-tool.js";

export type LoadedTool = {
  name: string;
  def: ToolDefinition;
  sourcePath: string;
};

const TOOL_EXT = /\.(ts|mts|js|mjs)$/;

function isToolDefinition(value: unknown): value is ToolDefinition {
  if (value === null || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.description === "string" &&
    v.inputSchema !== null &&
    typeof v.inputSchema === "object" &&
    typeof v.execute === "function"
  );
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
    if (!isToolDefinition(mod.default)) {
      throw new Error(
        `${sourcePath}: default export must be defineTool({ description, inputSchema, execute })`,
      );
    }
    const name = file.replace(TOOL_EXT, "");
    if (!name) {
      throw new Error(`${sourcePath}: empty tool name`);
    }
    out.push({ name, def: mod.default, sourcePath });
  }
  return out;
}
