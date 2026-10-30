import type { DefinedTool, ToolContext } from "./define-tool.js";
import { validateToolInput } from "./schema.js";

/** Drain async generators (Eve allows them); return the last yielded value. */
async function settleExecute<O>(result: O | Promise<O> | AsyncIterable<O>): Promise<O> {
  const awaited = await result;
  if (
    awaited !== null &&
    typeof awaited === "object" &&
    Symbol.asyncIterator in (awaited as object)
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
 * Eve order: validate input → execute → optional toModelOutput.
 * Validation failures throw `ToolInputValidationError`.
 */
export async function runTool(
  tool: DefinedTool,
  rawArgs: unknown,
  ctx: ToolContext,
): Promise<unknown> {
  const input = await validateToolInput(tool.inputSchema, rawArgs ?? {});
  const output = await settleExecute(tool.execute(input, ctx));
  if (tool.toModelOutput) {
    return tool.toModelOutput(output as never);
  }
  return output;
}
