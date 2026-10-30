/**
 * Drop-in for `import { defineTool } from "eve/tools"`.
 */
export { createToolContext, defineTool } from "./define-tool.js";
export type {
  DefinedTool,
  InferToolInput,
  JsonSchema,
  SandboxHandle,
  ToolAnnotations,
  ToolChannel,
  ToolContext,
  ToolDefinition,
  ToolExecuteResult,
  ToolInputSchema,
  ToolSession,
} from "./define-tool.js";
export { ToolInputValidationError } from "./schema.js";
