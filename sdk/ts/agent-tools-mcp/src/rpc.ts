import { createToolContext, type CreateToolContextInput } from "./context.js";
import type { LoadedTool } from "./load.js";
import { runTool } from "./run-tool.js";
import { ToolInputValidationError } from "./schema.js";

type JsonRpcRequest = {
  jsonrpc?: string;
  id?: string | number | null;
  method?: string;
  params?: Record<string, unknown>;
};

export type JsonRpcResponse = {
  jsonrpc: "2.0";
  id: string | number | null;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
};

/**
 * Wire bytes for a response whose tool value was already stringified once.
 * `JSON.stringify` of the envelope would walk `structuredContent` again.
 */
const wireByResponse = new WeakMap<JsonRpcResponse, string>();

function serializeOne(message: JsonRpcResponse): string {
  return wireByResponse.get(message) ?? JSON.stringify(message);
}

/** JSON-RPC bytes. Tool successes reuse the stringify from `handleRpc`. */
export function serializeRpc(
  message: JsonRpcResponse | readonly JsonRpcResponse[],
): string {
  if (Array.isArray(message)) {
    return `[${message.map((item) => serializeOne(item)).join(",")}]`;
  }
  // `Array.isArray` does not narrow a readonly array out of this union.
  return serializeOne(message as JsonRpcResponse);
}

function rememberToolWire(
  response: JsonRpcResponse,
  text: string,
  value: unknown,
): void {
  const idJson = JSON.stringify(response.id);
  const encodedText = JSON.stringify(text);
  // `undefined` is omitted, matching JSON.stringify. A string is escaped once
  // and reused for both fields. Any other value's canonical JSON is `text`.
  let structured = "";
  if (value !== undefined) {
    const json = typeof value === "string" ? encodedText : text;
    structured = `,"structuredContent":${json}`;
  }
  wireByResponse.set(
    response,
    `{"jsonrpc":"2.0","id":${idJson},"result":{"content":[{"type":"text","text":${encodedText}}]${structured}}}`,
  );
}

/** Session-wide bits; per-call `toolName` / `callId` are filled in on tools/call. */
export type RpcContextBase = Omit<
  CreateToolContextInput,
  "toolName" | "callId"
>;

/**
 * Minimal MCP surface: initialize, tools/list, tools/call.
 * Enough for beyond-agent + `.mcp-tools.json` stamp; keep the surface small for AOT+trim.
 */
export async function handleRpc(
  body: JsonRpcRequest,
  tools: LoadedTool[],
  base: RpcContextBase,
): Promise<JsonRpcResponse | null> {
  const id = body.id ?? null;
  const method = body.method;

  if (
    method === "notifications/initialized"
    || method === "notifications/cancelled"
  ) {
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
    const listed = tools.filter((t) => !t.def.deferred);
    return {
      jsonrpc: "2.0",
      id,
      result: {
        tools: listed.map((t) => ({
          name: t.name,
          description: modelFacingDescription(
            t.def.description,
            t.def.endsTurn,
          ),
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
    const ctx = createToolContext({
      ...base,
      toolName: tool.name,
      callId: id === null || id === undefined
        ? crypto.randomUUID()
        : String(id),
    });
    try {
      const result = await runTool(tool.def, args, ctx);
      // One walk of the value. `serializeRpc` reuses it for `structuredContent`
      // instead of walking the object again when the envelope is written.
      const text: unknown = typeof result === "string"
        ? result
        : JSON.stringify(result ?? null);
      const response: JsonRpcResponse = {
        jsonrpc: "2.0",
        id,
        result: {
          content: [{ type: "text", text: text as string }],
          structuredContent: result,
        },
      };
      if (typeof text === "string") rememberToolWire(response, text, result);
      return response;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const data = err instanceof ToolInputValidationError
        ? { issues: err.issues }
        : undefined;
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

const ENDS_TURN_SENTENCE =
  "Calling this tool ends your turn once it succeeds: do not write a reply or call other tools in the same step. If it fails, you will see the error and can continue.";

function modelFacingDescription(
  description: string,
  endsTurn:
    | boolean
    | ((output: never) => boolean | Promise<boolean>)
    | undefined,
): string {
  if (endsTurn === true) {
    return `${description}\n\n${ENDS_TURN_SENTENCE}`;
  }
  return description;
}
