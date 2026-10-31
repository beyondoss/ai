import type { LoadedTool } from "./load.js";
import {
  handleRpc,
  RpcCalls,
  type RpcContextBase,
  serializeRpc,
} from "./rpc.js";

/**
 * Line-delimited JSON-RPC on stdin/stdout — what `deploy/demo/mcp-tools.sh` drives
 * to stamp `.mcp-tools.json` without waking a sandbox.
 */
export async function serveStdio(opts: {
  tools: LoadedTool[];
  base: RpcContextBase;
}): Promise<void> {
  const calls = new RpcCalls();
  for await (const line of stdinLines()) {
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
        const r = await handleRpc(item as never, opts.tools, opts.base, {
          calls,
        });
        if (r) process.stdout.write(`${serializeRpc(r)}\n`);
      }
      continue;
    }
    const r = await handleRpc(body as never, opts.tools, opts.base, { calls });
    if (r) process.stdout.write(`${serializeRpc(r)}\n`);
  }
}

/**
 * The lines of stdin, split on LF alone (a CR before it is dropped). `node:readline` also splits on
 * U+2028 and U+2029, which a JSON-RPC request may carry inside a string, and would answer it twice.
 */
async function* stdinLines(): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of process.stdin) {
    buf += typeof chunk === "string"
      ? chunk
      : decoder.decode(chunk, { stream: true });
    let start = 0;
    let nl: number;
    while ((nl = buf.indexOf("\n", start)) !== -1) {
      yield buf.slice(start, nl).replace(/\r$/, "");
      start = nl + 1;
    }
    buf = buf.slice(start);
  }
  buf += decoder.decode();
  if (buf !== "") yield buf.replace(/\r$/, "");
}
