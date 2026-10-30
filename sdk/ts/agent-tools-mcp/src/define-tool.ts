/** Context passed to every tool `execute`. Secrets come from the sandbox env (grant-injected). */
export type ToolContext = {
  env: NodeJS.ProcessEnv;
  workspaceRoot: string;
};

/** MCP tool annotations the cell already understands. */
export type ToolAnnotations = {
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  idempotentHint?: boolean;
  openWorldHint?: boolean;
};

/**
 * One tool. `inputSchema` is a JSON Schema object (Eve often uses Zod — convert at the edge or
 * pass a schema object; a Zod bridge can land later without changing this shape).
 */
export type ToolDefinition<I = Record<string, unknown>> = {
  description: string;
  inputSchema: Record<string, unknown>;
  execute: (input: I, ctx: ToolContext) => unknown | Promise<unknown>;
  annotations?: ToolAnnotations;
};

/** Identity helper so tool modules can `export default defineTool({ ... })`. */
export function defineTool<I = Record<string, unknown>>(
  def: ToolDefinition<I>,
): ToolDefinition<I> {
  return def;
}
