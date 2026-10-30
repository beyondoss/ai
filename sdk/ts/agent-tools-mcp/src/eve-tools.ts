/**
 * Drop-in for `import { defineTool } from "eve/tools"`.
 * Point the agent project's package.json `imports` (or dependency alias) here.
 */
export { defineTool } from "./define-tool.js";
export type {
  DefinedTool,
  ToolAnnotations,
  ToolContext,
  ToolDefinition,
  ToolExecuteResult,
} from "./define-tool.js";
export type { InferToolInput, JsonSchema, ToolInputSchema } from "./schema.js";
export { ToolInputValidationError } from "./schema.js";
