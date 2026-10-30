import type { ToolContext } from "./define-tool.js";
import type { LoadedTool } from "./load.js";
import { runTool } from "./run-tool.js";
import { ToolInputValidationError } from "./schema.js";

type JsonRpcRequest = {
  jsonrpc?: string;
  id?: string | number | null;
  method?: string;
  params?: Record<string, unknown>;
};

type JsonRpcResponse = {
  jsonrpc: "2.0";
  id: string | number | null;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
};

/**
 * Minimal MCP surface: initialize, tools/list, tools/call.
 * Enough for beyond-agent + `.mcp-tools.json` stamp; keep the surface small for AOT+trim.
 */
export async function handleRpc(
  body: JsonRpcRequest,
  tools: LoadedTool[],
  ctx: ToolContext,
): Promise<JsonRpcResponse | null> {
  const id = body.id ?? null;
  const method = body.method;

  if (method === "notifications/initialized" || method === "notifications/cancelled") {
    return null;
  }

  if (method === "initialize") {
    return {
      jsonrpc: "2.0",
      id,
      result: {
        protocolVersion: "2025-06-18",
        capabilities: { tools: {} },
        serverInfo: { name: "beyond-agent-tools-mcp", version: "0.1.0-dev" },
      },
    };
  }

  if (method === "tools/list") {
    // Eve: deferred tools are omitted from the default tool list (still callable by name).
    const listed = tools.filter((t) => !t.def.deferred);
    return {
      jsonrpc: "2.0",
      id,
      result: {
        tools: listed.map((t) => ({
          name: t.name,
          description: modelFacingDescription(t.def.description, t.def.endsTurn),
          inputSchema: t.def.jsonInputSchema,
          ...(t.def.annotations ? { annotations: t.def.annotations } : {}),
        })),
      },
    };
  }

  if (method === "tools/call") {
    const name = body.params?.name;
    if (typeof name !== "string") {
      return {
        jsonrpc: "2.0",
        id,
        error: { code: -32602, message: "tools/call requires params.name" },
      };
    }
    const tool = tools.find((t) => t.name === name);
    if (!tool) {
      return {
        jsonrpc: "2.0",
        id,
        error: { code: -32602, message: `unknown tool: ${name}` },
      };
    }
    const args = body.params?.arguments ?? {};
    try {
      const result = await runTool(tool.def, args, {
        ...ctx,
        toolName: tool.name,
      });
      const text =
        typeof result === "string" ? result : JSON.stringify(result ?? null);
      return {
        jsonrpc: "2.0",
        id,
        result: {
          content: [{ type: "text", text }],
          structuredContent: result,
        },
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const data =
        err instanceof ToolInputValidationError ? { issues: err.issues } : undefined;
      return {
        jsonrpc: "2.0",
        id,
        result: {
          content: [{ type: "text", text: message }],
          isError: true,
          ...(data ? { _meta: data } : {}),
        },
      };
    }
  }

  if (method === "ping") {
    return { jsonrpc: "2.0", id, result: {} };
  }

  return {
    jsonrpc: "2.0",
    id,
    error: { code: -32601, message: `Method not found: ${method ?? ""}` },
  };
}

/** Eve docs: when endsTurn is true, append this after a blank line to the model-facing description. */
const ENDS_TURN_SENTENCE =
  "Calling this tool ends your turn once it succeeds: do not write a reply or call other tools in the same step. If it fails, you will see the error and can continue.";

function modelFacingDescription(
  description: string,
  endsTurn: boolean | ((output: never) => boolean | Promise<boolean>) | undefined,
): string {
  if (endsTurn === true) {
    return `${description}\n\n${ENDS_TURN_SENTENCE}`;
  }
  return description;
}
