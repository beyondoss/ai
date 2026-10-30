/**
 * Locks the operational fixes from the agent-tools-mcp audit.
 * Eve-facing claims (endsTurn, enum, cancel, outputSchema) live in the parity suites.
 */
import { mkdtemp, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { defineTool } from "../src/define-tool.js";
import { loadTools } from "../src/load.js";
import { writeManifest } from "../src/manifest.js";
import {
  AVAILABLE_IN_SUBAGENTS_META_KEY,
  handleRpc,
} from "../src/rpc.js";
import { createLocalSandboxHandle } from "../src/sandbox-handle.js";
import { validateToolInput } from "../src/schema.js";
import { sessionGate } from "../src/serve-http.js";
import { resolveWorkspace } from "../src/workspace.js";

describe("audit fixes", () => {
  it("workspace root is cwd or --workspace, not the tools directory", () => {
    expect(resolveWorkspace(undefined, "/bundle")).toBe(path.resolve("/bundle"));
    expect(resolveWorkspace("/explicit", "/bundle")).toBe(path.resolve("/explicit"));
  });

  it("manifest write leaves no temp file", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "audit-manifest-"));
    const tool = defineTool({
      description: "t",
      inputSchema: z.object({}),
      execute: () => ({ ok: true }),
    });
    const out = path.join(dir, ".mcp-tools.json");
    await writeManifest(
      [{ name: "t", def: tool, sourcePath: "t.ts" }],
      { name: "tools", out },
    );
    const names = await readdir(dir);
    expect(names.filter((name) => name.includes(".tmp"))).toEqual([]);
    expect(names).toContain(".mcp-tools.json");
  });

  it("two files with the same stem are an error", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "audit-dup-"));
    const source = `export default {
      description: "t",
      inputSchema: { type: "object", properties: {} },
      execute: () => ({ ok: true }),
    };\n`;
    await writeFile(path.join(dir, "get_weather.ts"), source, "utf8");
    await writeFile(path.join(dir, "get_weather.js"), source, "utf8");
    await expect(loadTools(dir)).rejects.toThrow(/duplicate tool name "get_weather"/);
  });

  it("spawn yields lines and exitCode resolves without a reader", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "audit-spawn-"));
    const sandbox = createLocalSandboxHandle(dir);
    const lined = await sandbox.spawn({ command: "printf 'a\\nb\\n'" });
    const lines: string[] = [];
    for await (const line of lined.stdout) lines.push(line);
    expect(lines).toEqual(["a", "b"]);
    expect(await lined.exitCode).toBe(0);

    const flooded = await sandbox.spawn({
      command: "node -e 'process.stdout.write(\"x\".repeat(200000))'",
    });
    expect(await flooded.exitCode).toBe(0);
  });

  it("tools/call requires a session this process issued", () => {
    const sessions = new Set<string>();
    const init = sessionGate(sessions, { isInit: true, session: null });
    expect(init.ok).toBe(true);
    if (!init.ok || !init.session) throw new Error("initialize did not mint a session");
    expect(
      sessionGate(sessions, { isInit: false, session: null }),
    ).toEqual({ ok: false, status: 400, message: "missing Mcp-Session-Id" });
    expect(
      sessionGate(sessions, { isInit: false, session: "nope" }),
    ).toEqual({ ok: false, status: 404, message: "session not found" });
    const again = sessionGate(sessions, { isInit: false, session: init.session });
    expect(again.ok).toBe(true);
  });

  it("JSON Schema defaults are copied and non-objects are not rewritten", async () => {
    const schema = {
      type: "object",
      properties: {
        opts: {
          type: "object",
          default: { n: 1 },
          properties: { n: { type: "number" } },
        },
      },
    };
    const first = await validateToolInput(schema, {}) as { opts: { n: number } };
    first.opts.n = 9;
    expect(await validateToolInput(schema, {})).toEqual({ opts: { n: 1 } });
    await expect(validateToolInput(schema, "nope")).rejects.toThrow(/object/);
  });

  it("lists outputSchema and hides subagent-private tools in _meta and the manifest", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "audit-flags-"));
    const tool = defineTool({
      description: "private",
      inputSchema: z.object({}),
      outputSchema: z.object({ ok: z.boolean() }),
      annotations: { readOnlyHint: false, destructiveHint: true },
      availableInSubagents: false,
      execute: () => ({ ok: true }),
    });
    const loaded = [{ name: "priv", def: tool, sourcePath: "priv.ts" }];
    const list = await handleRpc(
      { jsonrpc: "2.0", id: 1, method: "tools/list" },
      loaded,
      { workspaceRoot: dir },
    );
    const listed = (list?.result as {
      tools: {
        outputSchema?: { type?: string };
        annotations?: { destructiveHint?: boolean };
        _meta?: Record<string, unknown>;
      }[];
    }).tools[0];
    expect(listed?.outputSchema?.type).toBe("object");
    expect(listed?.annotations?.destructiveHint).toBe(true);
    expect(listed?._meta?.[AVAILABLE_IN_SUBAGENTS_META_KEY]).toBe(false);

    const manifest = await writeManifest(loaded, {
      name: "tools",
      out: path.join(dir, ".mcp-tools.json"),
    });
    expect(manifest.tools?.[0]?.available_in_subagents).toBe(false);
    expect(manifest.tools?.[0]?.annotations?.readOnlyHint).toBe(false);
    expect(manifest.tools?.[0]?.output_schema).toMatchObject({ type: "object" });
  });
});
