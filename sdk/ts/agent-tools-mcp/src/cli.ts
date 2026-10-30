#!/usr/bin/env bun
/**
 * beyond-agent-tools-mcp — load tools/* and speak MCP (stdio or HTTP).
 *
 *   beyond-agent-tools-mcp serve --dir ./tools [--http [host:]port]
 *   beyond-agent-tools-mcp manifest --dir ./tools --name tools -o .mcp-tools.json
 *   beyond-agent-tools-mcp call --dir ./tools <name> '<json-args>'
 */
import path from "node:path";

import { loadTools } from "./load.js";
import { writeManifest } from "./manifest.js";
import type { RpcContextBase } from "./rpc.js";
import { handleRpc } from "./rpc.js";
import { serveHttp } from "./serve-http.js";
import { serveStdio } from "./serve-stdio.js";
import { resolveWorkspace } from "./workspace.js";

function usage(): never {
  console.error(`Usage:
  beyond-agent-tools-mcp serve --dir <toolsDir> [--workspace <root>] [--http [host:]port]
  beyond-agent-tools-mcp manifest --dir <toolsDir> --name <connector> [-o .mcp-tools.json]
  beyond-agent-tools-mcp call --dir <toolsDir> [--workspace <root>] <name> '<json-args>'`);
  process.exit(2);
}

function takeFlag(argv: string[], name: string): string | undefined {
  const i = argv.indexOf(name);
  if (i < 0) return undefined;
  const v = argv[i + 1];
  if (v === undefined || v.startsWith("-")) {
    console.error(`missing value for ${name}`);
    process.exit(2);
  }
  argv.splice(i, 2);
  return v;
}

/** `--http` alone, or `--http 8931`, or `--http 127.0.0.1:8931`. */
function takeHttpFlag(argv: string[]): string | true | undefined {
  const i = argv.indexOf("--http");
  if (i < 0) return undefined;
  const v = argv[i + 1];
  if (v === undefined || v.startsWith("-")) {
    argv.splice(i, 1);
    return true;
  }
  argv.splice(i, 2);
  return v;
}

function rpcBase(workspaceRoot = process.cwd()): RpcContextBase {
  return {
    env: process.env,
    workspaceRoot,
  };
}

function parseHttpBind(
  raw: string | undefined,
): { host: string; port: number } {
  // Default 0.0.0.0 so an E2B port mapping can reach the process; local callers use 127.0.0.1 explicitly.
  if (!raw || raw === "true") return { host: "0.0.0.0", port: 8931 };
  if (raw.includes(":")) {
    const idx = raw.lastIndexOf(":");
    const host = raw.slice(0, idx) || "0.0.0.0";
    const portStr = raw.slice(idx + 1);
    return { host, port: Number(portStr) };
  }
  return { host: "0.0.0.0", port: Number(raw) };
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const cmd = argv.shift();
  if (!cmd) usage();

  if (cmd === "serve") {
    const dir = takeFlag(argv, "--dir");
    const workspace = takeFlag(argv, "--workspace");
    if (!dir) usage();
    const httpRaw = takeHttpFlag(argv);
    if (argv.length) usage();
    const tools = await loadTools(dir);
    if (tools.length === 0) {
      console.error(`no tools in ${path.resolve(dir)}`);
      process.exit(1);
    }
    const root = resolveWorkspace(workspace);
    if (httpRaw !== undefined) {
      const bind = parseHttpBind(httpRaw === true ? undefined : httpRaw);
      await serveHttp({
        tools,
        base: rpcBase(root),
        host: bind.host,
        port: bind.port,
      });
      await new Promise(() => {});
    } else {
      await serveStdio({ tools, base: rpcBase(root) });
    }
    return;
  }

  if (cmd === "manifest") {
    const dir = takeFlag(argv, "--dir");
    const name = takeFlag(argv, "--name") ?? "tools";
    const out = takeFlag(argv, "-o") ?? takeFlag(argv, "--out")
      ?? ".mcp-tools.json";
    if (!dir || argv.length) usage();
    const tools = await loadTools(dir);
    const file = await writeManifest(tools, { name, out });
    console.error(
      `wrote ${out} (${
        file[name]?.length ?? 0
      } tools under connector "${name}")`,
    );
    return;
  }

  if (cmd === "call") {
    const dir = takeFlag(argv, "--dir");
    const workspace = takeFlag(argv, "--workspace");
    const name = argv.shift();
    const argsJson = argv.shift() ?? "{}";
    if (!dir || !name || argv.length) usage();
    const tools = await loadTools(dir);
    let args: Record<string, unknown>;
    try {
      args = JSON.parse(argsJson) as Record<string, unknown>;
    } catch {
      console.error("arguments must be JSON");
      process.exit(2);
    }
    const r = await handleRpc(
      {
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: { name, arguments: args },
      },
      tools,
      rpcBase(resolveWorkspace(workspace)),
    );
    process.stdout.write(`${JSON.stringify(r, null, 2)}\n`);
    if (
      r && "result" in r && r.result && typeof r.result === "object"
      && r.result !== null && "isError" in r.result
    ) {
      process.exit(1);
    }
    return;
  }

  usage();
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});
