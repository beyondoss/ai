import { randomUUID } from "node:crypto";

import type { LoadedTool } from "./load.js";
import { handleRpc, RpcCalls, type RpcContextBase } from "./rpc.js";

export type SessionDecision =
  | { ok: true; session: string | null }
  | { ok: false; status: number; message: string };

/**
 * Streamable HTTP session rule. `initialize` mints an id when the client did
 * not send one. Every later POST must carry an id this process issued.
 */
export function sessionGate(
  sessions: Set<string>,
  input: { isInit: boolean; session: string | null },
): SessionDecision {
  let session = input.session;
  if (input.isInit && !session) {
    session = randomUUID();
    sessions.add(session);
    return { ok: true, session };
  }
  if (!input.isInit && !session) {
    return { ok: false, status: 400, message: "missing Mcp-Session-Id" };
  }
  if (session && !sessions.has(session) && !input.isInit) {
    return { ok: false, status: 404, message: "session not found" };
  }
  if (session) sessions.add(session);
  return { ok: true, session };
}

export type ServeHttpOptions = {
  tools: LoadedTool[];
  base: RpcContextBase;
  host?: string;
  port?: number;
  /** When set, print one ready line on stderr. Default true. */
  readyLine?: boolean;
};

type JsonRpc = {
  jsonrpc?: string;
  id?: string | number | null;
  method?: string;
  params?: Record<string, unknown>;
};

/**
 * Minimal streamable-HTTP MCP for beyond-agent's `sandbox: true` connectors:
 * - POST JSON-RPC with Accept: application/json, text/event-stream
 * - initialize without a session → 200 + `Mcp-Session-Id` + JSON body
 * - later POSTs carry `Mcp-Session-Id`
 * - DELETE ends the session
 * - GET is 405 (cell answers SSE as unsupported for sandbox connectors)
 *
 * Path may be `/`, `/mcp`, or `/mcp/` — the demo front forwards to `/mcp`.
 */
export async function serveHttp(
  opts: ServeHttpOptions,
): Promise<{ port: number; stop: () => void }> {
  const host = opts.host ?? "127.0.0.1";
  const port = opts.port ?? 0;
  const sessions = new Set<string>();
  const calls = new RpcCalls();

  const server = Bun.serve({
    hostname: host,
    port,
    async fetch(req) {
      const url = new URL(req.url);
      const path = url.pathname.replace(/\/+$/, "") || "/";

      if (
        req.method === "GET"
        && (path === "/healthz" || path.endsWith("/healthz"))
      ) {
        return new Response("ok");
      }

      // Sandbox connectors do not hold an SSE stream; refuse GET the way a 405 server would.
      if (req.method === "GET") {
        return new Response("method not allowed", { status: 405 });
      }

      if (req.method === "DELETE") {
        const sid = req.headers.get("mcp-session-id");
        if (sid) sessions.delete(sid);
        return new Response(null, { status: 204 });
      }

      if (req.method !== "POST") {
        return new Response("POST JSON-RPC", { status: 405 });
      }

      let body: unknown;
      try {
        body = await req.json();
      } catch {
        return Response.json(
          {
            jsonrpc: "2.0",
            id: null,
            error: { code: -32700, message: "Parse error" },
          },
          { status: 400 },
        );
      }

      const incomingSession = req.headers.get("mcp-session-id");
      const isInit = !Array.isArray(body)
        && body !== null
        && typeof body === "object"
        && (body as JsonRpc).method === "initialize";

      const decision = sessionGate(sessions, {
        isInit,
        session: incomingSession,
      });
      if (!decision.ok) {
        return new Response(decision.message, { status: decision.status });
      }
      const session = decision.session;
      const dispatch = { calls, parentSignal: req.signal };

      const headers: Record<string, string> = {
        "content-type": "application/json",
      };
      if (session) headers["mcp-session-id"] = session;

      if (Array.isArray(body)) {
        const out = [];
        for (const item of body) {
          const r = await handleRpc(
            item as never,
            opts.tools,
            opts.base,
            dispatch,
          );
          if (r) out.push(r);
        }
        return new Response(JSON.stringify(out), { status: 200, headers });
      }

      const r = await handleRpc(body as never, opts.tools, opts.base, dispatch);
      if (r == null) {
        if (session) {
          return new Response(null, {
            status: 202,
            headers: { "mcp-session-id": session },
          });
        }
        return new Response(null, { status: 202 });
      }
      return new Response(JSON.stringify(r), { status: 200, headers });
    },
  });

  if (opts.readyLine ?? true) {
    console.error(`READY ${server.port}`);
  }

  return {
    port: server.port,
    stop: () => {
      void server.stop(true);
    },
  };
}
