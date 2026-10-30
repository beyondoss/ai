import type {
  InferToolInput,
  JsonSchema,
  ToolInputSchema,
} from "./schema.js";
import { inputSchemaToJsonSchema } from "./schema.js";

/** Context passed to every tool `execute`. Secrets come from the sandbox env (grant-injected). */
export type ToolContext = {
  env: NodeJS.ProcessEnv;
  workspaceRoot: string;
  /** MCP tool name (filename stem). */
  toolName?: string;
  /** Abort when the MCP client cancels. */
  signal?: AbortSignal;
};

/** MCP tool annotations the cell already understands. */
export type ToolAnnotations = {
  readOnlyHint?: boolean;
  destructiveHint?: boolean;
  idempotentHint?: boolean;
  openWorldHint?: boolean;
};

export type ToolExecuteResult<O> = O | Promise<O> | AsyncIterable<O>;

/**
 * Eve / Agent-SDK `defineTool` shape.
 *
 * - `inputSchema`: Zod, any Standard Schema, or plain JSON Schema (required).
 * - Zod / Standard Schema infer `execute`'s input type; JSON Schema → `Record<string, unknown>`.
 * - Optional `outputSchema` types the return when it is Zod / Standard Schema.
 * - Input is validated before `execute` (see `runTool`).
 */
export type ToolDefinition<
  S extends ToolInputSchema = ToolInputSchema,
  O = unknown,
> = {
  description: string;
  inputSchema: S;
  outputSchema?: ToolInputSchema;
  execute: (input: InferToolInput<S>, ctx: ToolContext) => ToolExecuteResult<O>;
  annotations?: ToolAnnotations;
  /** Eve: hide from delegated agents. Stored; MCP guest does not enforce subagent policy. */
  availableInSubagents?: boolean;
  /** Eve: omit from the default tool list. MCP guest still lists all loaded files. */
  deferred?: boolean;
  /** Eve: end the turn after success. Beyond agent loop ignores this today. */
  endsTurn?: boolean | ((output: O) => boolean | Promise<boolean>);
  /** Eve: project execute output for the model. Applied in `runTool` when set. */
  toModelOutput?: (output: O) => unknown;
};

/** Normalized tool after `defineTool` — carries JSON Schema for MCP / manifest. */
export type DefinedTool<
  S extends ToolInputSchema = ToolInputSchema,
  O = unknown,
> = ToolDefinition<S, O> & {
  /** JSON Schema for `tools/list` and `.mcp-tools.json` (Eve: no `$schema`). */
  readonly jsonInputSchema: JsonSchema;
};

/**
 * Eve-compatible `defineTool`. Not an identity helper — freezes the def and
 * materializes JSON Schema for the model at definition time.
 */
export function defineTool<S extends ToolInputSchema, O = unknown>(
  def: ToolDefinition<S, O>,
): DefinedTool<S, O> {
  if (def === null || typeof def !== "object") {
    throw new Error("defineTool expects an object");
  }
  if (typeof def.description !== "string" || !def.description) {
    throw new Error("defineTool requires a non-empty description");
  }
  if (def.inputSchema === undefined || def.inputSchema === null) {
    throw new Error(
      "defineTool requires inputSchema (Zod, Standard Schema, or JSON Schema; use z.object({}) for no input)",
    );
  }
  if (typeof def.execute !== "function") {
    throw new Error("defineTool requires execute()");
  }

  const jsonInputSchema = inputSchemaToJsonSchema(def.inputSchema);
  return Object.freeze({
    ...def,
    jsonInputSchema,
  });
}

export type { InferToolInput, JsonSchema, ToolInputSchema };
