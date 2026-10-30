import { createToolContext, type CreateToolContextInput } from "./context.js";
import type { LoadedTool } from "./load.js";
import { executeTool, type RunnableTool } from "./run-tool.js";
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

/** Session-wide bits; per-call `toolName` / `callId` are filled in on tools/call. */
export type RpcContextBase = Omit<
  CreateToolContextInput,
  "toolName" | "callId"
>;

/**
 * `_meta` key on a successful `tools/call`. The agent sets `ToolOutput.terminate`
 * when this is true. A failed call never sets it.
 */
export const ENDS_TURN_META_KEY = "beyond.dev/endsTurn";

/**
 * `tools/list` `_meta` key. `false` means subagents do not inherit the tool.
 * Omitted means they do.
 */
export const AVAILABLE_IN_SUBAGENTS_META_KEY =
  "beyond.dev/availableInSubagents";

type InFlight = {
  controller: AbortController;
  onParent?: () => void;
  parent?: AbortSignal;
};

/**
 * In-flight `tools/call`s for one server. `notifications/cancelled` aborts the
 * matching JSON-RPC id; an HTTP request abort aborts the call it started.
 */
export class RpcCalls {
  private inflight = new Map<string, InFlight>();
  private cancelled = new Set<string>();

  begin(id: string, parent?: AbortSignal): AbortSignal {
    const controller = new AbortController();
    const onParent = () => controller.abort();
    if (parent) {
      if (parent.aborted) controller.abort();
      else parent.addEventListener("abort", onParent, { once: true });
    }
    if (this.cancelled.delete(id)) controller.abort();
    if (!controller.signal.aborted) {
      const entry: InFlight = { controller };
      if (parent) {
        entry.onParent = onParent;
        entry.parent = parent;
      }
      this.inflight.set(id, entry);
    } else if (parent) {
      parent.removeEventListener("abort", onParent);
    }
    return controller.signal;
  }

  end(id: string): void {
    const entry = this.inflight.get(id);
    if (entry?.parent && entry.onParent) {
      entry.parent.removeEventListener("abort", entry.onParent);
    }
    this.inflight.delete(id);
    this.cancelled.delete(id);
  }

  cancel(id: string): void {
    const entry = this.inflight.get(id);
    if (entry) {
      entry.controller.abort();
      return;
    }
    if (this.cancelled.size > 64) this.cancelled.clear();
    this.cancelled.add(id);
  }
}

export type RpcDispatch = {
  calls?: RpcCalls;
  parentSignal?: AbortSignal;
};

/**
 * Minimal MCP surface: initialize, tools/list, tools/call.
 * Enough for beyond-agent + `.mcp-tools.json` stamp; keep the surface small for AOT+trim.
 */
export async function handleRpc(
  body: JsonRpcRequest,
  tools: LoadedTool[],
  base: RpcContextBase,
  dispatch?: RpcDispatch,
): Promise<JsonRpcResponse | null> {
  const id = body.id ?? null;
  const method = body.method;
  const calls = dispatch?.calls ?? new RpcCalls();

  if (method === "notifications/cancelled") {
    const requestId = body.params?.requestId;
    if (typeof requestId === "string" || typeof requestId === "number") {
      calls.cancel(String(requestId));
    }
    return null;
  }

  if (method === "notifications/initialized") {
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
        tools: listed.map((t) => {
          const tool: Record<string, unknown> = {
            name: t.name,
            description: modelFacingDescription(
              t.def.description,
              t.def.endsTurn,
            ),
            inputSchema: t.def.jsonInputSchema,
          };
          if (t.def.jsonOutputSchema) {
            tool.outputSchema = t.def.jsonOutputSchema;
          }
          if (t.def.annotations) tool.annotations = t.def.annotations;
          if (t.def.availableInSubagents === false) {
            tool._meta = { [AVAILABLE_IN_SUBAGENTS_META_KEY]: false };
          }
          return tool;
        }),
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
    const callKey = id === null || id === undefined ? undefined : String(id);
    const abortSignal = callKey
      ? calls.begin(callKey, dispatch?.parentSignal)
      : dispatch?.parentSignal ?? new AbortController().signal;
    const ctx = createToolContext({
      ...base,
      toolName: tool.name,
      callId: callKey ?? crypto.randomUUID(),
      abortSignal,
    });
    try {
      const run = await executeTool(tool.def, args, ctx);
      const text = typeof run.value === "string"
        ? run.value
        : JSON.stringify(run.value ?? null);
      const result: Record<string, unknown> = {
        content: [{ type: "text", text }],
      };
      if (
        run.value !== null
        && typeof run.value === "object"
        && !Array.isArray(run.value)
      ) {
        result.structuredContent = run.value;
      }
      if (run.endsTurn) result._meta = { [ENDS_TURN_META_KEY]: true };
      return { jsonrpc: "2.0", id, result };
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
    } finally {
      if (callKey) calls.end(callKey);
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

/** Description the model sees. Boolean `endsTurn` only; a function is decided after execute. */
export function modelFacingDescription(
  description: string,
  endsTurn: RunnableTool["endsTurn"],
): string {
  if (endsTurn === true) {
    return `${description}\n\n${ENDS_TURN_SENTENCE}`;
  }
  return description;
}
