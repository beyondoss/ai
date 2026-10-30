/**
 * Prove Eve `ctx` surface — especially `ctx.getSandbox()` from eve.dev/docs/sandbox.
 */
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { createToolContext } from "../src/context.js";
import { defineTool } from "../src/define-tool.js";
import { handleRpc, RpcCalls } from "../src/rpc.js";
import { runTool } from "../src/run-tool.js";

describe("Eve ctx parity", () => {
  it("exposes session, callId, toolName, messages, abortSignal, getSandbox, to", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "eve-ctx-"));
    const ctx = createToolContext({
      workspaceRoot: dir,
      toolName: "demo",
      callId: "call-1",
      session: { id: "sess-9" },
      messages: [{ role: "user", content: "hi" }],
    });

    expect(ctx.session.id).toBe("sess-9");
    expect(ctx.callId).toBe("call-1");
    expect(ctx.toolName).toBe("demo");
    expect(ctx.messages).toEqual([{ role: "user", content: "hi" }]);
    expect(ctx.abortSignal).toBeInstanceOf(AbortSignal);
    expect(typeof ctx.getSandbox).toBe("function");
    expect(typeof ctx.to).toBe("function");
    expect(() => ctx.to("slack", {})).toThrow(/ctx\.to/);
  });

  it("Eve docs sandbox snippet: writeTextFile + run via ctx.getSandbox()", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "eve-sandbox-"));
    // Exact pattern from https://eve.dev/docs/sandbox
    const tool = defineTool({
      description: "Run a Python script.",
      inputSchema: z.object({ script: z.string() }),
      async execute({ script }, ctx) {
        const sandbox = await ctx.getSandbox();
        await sandbox.writeTextFile({ path: "analysis.py", content: script });
        const result = await sandbox.run({ command: "python3 analysis.py" });
        if (result.exitCode !== 0) {
          // Fallback if python3 missing: prove the file write + shell path still works.
          const echo = await sandbox.run({
            command: "cat analysis.py && echo RAN",
          });
          expect(echo.exitCode).toBe(0);
          expect(echo.stdout).toContain("RAN");
          return { stdout: echo.stdout, via: "shell" as const };
        }
        return { stdout: result.stdout, via: "python" as const };
      },
    });

    const ctx = createToolContext({
      workspaceRoot: dir,
      toolName: "run_python",
      callId: "c1",
    });
    const out = await runTool(
      tool,
      { script: "print(40+2)\n" },
      ctx,
    );
    expect(out).toMatchObject({ stdout: expect.any(String) });
    const onDisk = await readFile(path.join(dir, "analysis.py"), "utf8");
    expect(onDisk).toBe("print(40+2)\n");
    if ((out as { via: string }).via === "python") {
      expect((out as { stdout: string }).stdout.trim()).toBe("42");
    }
  });

  it("resolvePath is workspace-rooted; escapes throw", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "eve-path-"));
    const ctx = createToolContext({ workspaceRoot: dir, toolName: "t" });
    const sb = await ctx.getSandbox();
    expect(sb.resolvePath(".")).toBe(path.resolve(dir));
    expect(sb.resolvePath("a/b")).toBe(path.join(path.resolve(dir), "a", "b"));
    expect(() => sb.resolvePath("../outside")).toThrow(/escapes/);
  });

  it("tools/call wires callId + toolName into execute ctx", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "eve-rpc-ctx-"));
    let seen: { callId?: string; toolName?: string; sessionId?: string } = {};
    const tool = defineTool({
      description: "probe ctx",
      inputSchema: z.object({}),
      async execute(_input, ctx) {
        seen = {
          callId: ctx.callId,
          toolName: ctx.toolName,
          sessionId: ctx.session.id,
        };
        const sb = await ctx.getSandbox();
        await sb.writeTextFile({ path: "ok.txt", content: "1" });
        return seen;
      },
    });

    const r = await handleRpc(
      {
        jsonrpc: "2.0",
        id: "rpc-99",
        method: "tools/call",
        params: { name: "probe", arguments: {} },
      },
      [{ name: "probe", def: tool, sourcePath: "probe.ts" }],
      { workspaceRoot: dir, session: { id: "s-1" } },
    );

    expect(r?.result).toMatchObject({
      structuredContent: {
        callId: "rpc-99",
        toolName: "probe",
        sessionId: "s-1",
      },
    });
    expect(await readFile(path.join(dir, "ok.txt"), "utf8")).toBe("1");
  });

  it("notifications/cancelled aborts the tool and its child process", async () => {
    const calls = new RpcCalls();
    const dir = await mkdtemp(path.join(tmpdir(), "eve-cancel-"));
    let seen: AbortSignal | undefined;
    const tool = defineTool({
      description: "sleep",
      inputSchema: z.object({}),
      async execute(_input, ctx) {
        seen = ctx.abortSignal;
        const sandbox = await ctx.getSandbox();
        return sandbox.run({ command: "sleep 30" });
      },
    });
    const pending = handleRpc(
      {
        jsonrpc: "2.0",
        id: 7,
        method: "tools/call",
        params: { name: "sleep", arguments: {} },
      },
      [{ name: "sleep", def: tool, sourcePath: "sleep.ts" }],
      { workspaceRoot: dir },
      { calls },
    );
    for (let i = 0; i < 50 && !seen; i++) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
    expect(seen).toBeInstanceOf(AbortSignal);
    await handleRpc(
      {
        jsonrpc: "2.0",
        method: "notifications/cancelled",
        params: { requestId: 7 },
      },
      [],
      { workspaceRoot: dir },
      { calls },
    );
    const result = await pending;
    expect(seen?.aborted).toBe(true);
    expect(result?.result).toMatchObject({ isError: true });
  }, 8_000);

  it("run nonzero exit does not throw (Eve: check exitCode yourself)", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "eve-exit-"));
    const ctx = createToolContext({ workspaceRoot: dir, toolName: "t" });
    const sb = await ctx.getSandbox();
    const result = await sb.run({ command: "exit 7" });
    expect(result.exitCode).toBe(7);
  });
});
