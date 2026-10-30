import type { ToolContext } from "./define-tool.js";
import type { LoadedTool } from "./load.js";

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
    return {
      jsonrpc: "2.0",
      id,
      result: {
        tools: tools.map((t) => ({
          name: t.name,
          description: t.def.description,
          inputSchema: t.def.inputSchema,
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
    const args = (body.params?.arguments ?? {}) as Record<string, unknown>;
    try {
      const result = await tool.def.execute(args, ctx);
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
      return {
        jsonrpc: "2.0",
        id,
        result: {
          content: [{ type: "text", text: message }],
          isError: true,
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
