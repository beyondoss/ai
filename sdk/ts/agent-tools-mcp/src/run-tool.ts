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

/**
 * Eve order: validate input → execute → validate outputSchema → optional toModelOutput.
 */
export async function runTool(
  tool: RunnableTool,
  rawArgs: unknown,
  ctx: ToolContext,
): Promise<unknown> {
  const input = await validateToolInput(tool.inputSchema, rawArgs ?? {});
  const output = await settleExecute(tool.execute(input, ctx));
  if (tool.outputSchema) {
    try {
      await validateToolInput(tool.outputSchema, output);
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
  if (tool.toModelOutput) {
    return tool.toModelOutput(output);
  }
  return output;
}
