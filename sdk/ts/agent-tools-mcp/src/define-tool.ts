import type { ToolContext } from "./context.js";
import type { InferToolInput, JsonSchema, ToolInputSchema } from "./schema.js";
import { inputSchemaToJsonSchema } from "./schema.js";

export type { ToolContext } from "./context.js";
export type {
  CreateToolContextInput,
  ToolChannel,
  ToolSession,
} from "./context.js";
export { createToolContext } from "./context.js";
export type {
  SandboxHandle,
  SandboxProcess,
  SandboxRunResult,
} from "./sandbox-handle.js";

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
 * - `execute` receives Eve-shaped `ctx` (`getSandbox`, `abortSignal`, `callId`, …).
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
  availableInSubagents?: boolean;
  deferred?: boolean;
  endsTurn?: boolean | ((output: O) => boolean | Promise<boolean>);
  toModelOutput?: (output: O) => unknown;
};

/** Normalized tool after `defineTool` — carries JSON Schema for MCP / manifest. */
export type DefinedTool<
  S extends ToolInputSchema = ToolInputSchema,
  O = unknown,
> = ToolDefinition<S, O> & {
  readonly jsonInputSchema: JsonSchema;
};

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
