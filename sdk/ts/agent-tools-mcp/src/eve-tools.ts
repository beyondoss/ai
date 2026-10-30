/**
 * Drop-in shim for `import { defineTool } from "eve/tools"`.
 * Point the agent project's package.json `imports` (or dependencies alias) here.
 */
export { defineTool } from "./define-tool.js";
export type { ToolAnnotations, ToolContext, ToolDefinition } from "./define-tool.js";
