/**
 * Drop-in for `import { defineTool } from "eve/tools"`.
 */
export { defineTool, createToolContext } from "./define-tool.js";
export type {
  DefinedTool,
  ToolAnnotations,
  ToolContext,
  ToolDefinition,
  ToolExecuteResult,
  ToolChannel,
  ToolSession,
  SandboxHandle,
  InferToolInput,
  JsonSchema,
  ToolInputSchema,
} from "./define-tool.js";
export { ToolInputValidationError } from "./schema.js";
