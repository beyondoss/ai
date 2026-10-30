import readline from "node:readline";

import type { LoadedTool } from "./load.js";
import { handleRpc, type RpcContextBase, serializeRpc } from "./rpc.js";

/**
 * Line-delimited JSON-RPC on stdin/stdout — what `deploy/demo/mcp-tools.sh` drives
 * to stamp `.mcp-tools.json` without waking a sandbox.
 */
export async function serveStdio(opts: {
  tools: LoadedTool[];
  base: RpcContextBase;
}): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });
  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let body: unknown;
    try {
      body = JSON.parse(trimmed);
    } catch {
      process.stdout.write(
        `${
          JSON.stringify({
            jsonrpc: "2.0",
            id: null,
            error: { code: -32700, message: "Parse error" },
          })
        }\n`,
      );
      continue;
    }
    if (Array.isArray(body)) {
      for (const item of body) {
        const r = await handleRpc(item as never, opts.tools, opts.base);
        if (r) process.stdout.write(`${serializeRpc(r)}\n`);
      }
      continue;
    }
    const r = await handleRpc(body as never, opts.tools, opts.base);
    if (r) process.stdout.write(`${serializeRpc(r)}\n`);
  }
}
