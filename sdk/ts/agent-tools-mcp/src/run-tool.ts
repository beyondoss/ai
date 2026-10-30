import type { ToolContext } from "./context.js";
import type { ToolAnnotations } from "./define-tool.js";
import type { JsonSchema, ToolInputSchema } from "./schema.js";
import { ToolInputValidationError, validateToolInput } from "./schema.js";

/** Drain async generators (Eve allows them); return the last yielded value. */
async function settleExecute<O>(
  result: O | Promise<O> | AsyncIterable<O>,
): Promise<O> {
  const awaited = await result;
  if (
    awaited !== null
    && typeof awaited === "object"
    && Symbol.asyncIterator in (awaited as object)
  ) {
    let last: O | undefined;
    for await (const chunk of awaited as AsyncIterable<O>) {
      last = chunk;
    }
    return last as O;
  }
  return awaited as O;
}

/**
 * Call-site shape for any concrete `defineTool` result.
 * Execute input uses `any` so Zod generics assign under strict function types
 * (`InferToolInput<any>` collapses to `unknown` and breaks contravariance).
 */
export type RunnableTool = {
  description: string;
  inputSchema: ToolInputSchema;
  jsonInputSchema: JsonSchema;
  jsonOutputSchema?: JsonSchema;
  outputSchema?: ToolInputSchema;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- generic erasure
  execute: (input: any, ctx: ToolContext) => any;
  annotations?: ToolAnnotations;
  availableInSubagents?: boolean;
  deferred?: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- generic erasure
  endsTurn?: boolean | ((output: any) => boolean | Promise<boolean>);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- generic erasure
  toModelOutput?: (output: any) => unknown;
};

export type ToolRun = {
  /** What the model sees (`toModelOutput` when set, otherwise the validated output). */
  value: unknown;
  /** `endsTurn: true`, or the function returned true, on this successful call. */
  endsTurn: boolean;
};

/**
 * Eve order: validate input → execute → validate outputSchema (keep the parsed
 * value) → `endsTurn` on that value → optional `toModelOutput`.
 */
export async function executeTool(
  tool: RunnableTool,
  rawArgs: unknown,
  ctx: ToolContext,
): Promise<ToolRun> {
  const input = await validateToolInput(tool.inputSchema, rawArgs ?? {});
  let output = await settleExecute(tool.execute(input, ctx));
  if (tool.outputSchema) {
    try {
      output = await validateToolInput(tool.outputSchema, output);
    } catch (err) {
      if (err instanceof ToolInputValidationError) {
        throw new ToolInputValidationError(
          `outputSchema: ${err.message}`,
          err.issues,
        );
      }
      throw err;
    }
  }
  const endsTurn = await resolveEndsTurn(tool.endsTurn, output);
  const value = tool.toModelOutput ? tool.toModelOutput(output) : output;
  return { value, endsTurn };
}

/** The model-facing value from {@link executeTool}. */
export async function runTool(
  tool: RunnableTool,
  rawArgs: unknown,
  ctx: ToolContext,
): Promise<unknown> {
  return (await executeTool(tool, rawArgs, ctx)).value;
}

async function resolveEndsTurn(
  endsTurn: RunnableTool["endsTurn"],
  output: unknown,
): Promise<boolean> {
  if (endsTurn === true) return true;
  if (typeof endsTurn === "function") return Boolean(await endsTurn(output));
  return false;
}
