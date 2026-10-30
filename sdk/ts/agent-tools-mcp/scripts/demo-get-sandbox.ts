/**
 * Unequivocal demo: what ctx.getSandbox() does.
 * Run: DEMO_DIR=$(mktemp -d) bun run scripts/demo-get-sandbox.ts
 */
import { readFileSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { z } from "zod";

import { createToolContext } from "../src/context.ts";
import { defineTool } from "../src/eve-tools.ts";

const dir = process.env.DEMO_DIR
  ?? (await mkdtemp(path.join(tmpdir(), "get-sandbox-")));

console.log("workspace:", dir);

const tool = defineTool({
  description: "Eve sandbox demo",
  inputSchema: z.object({ script: z.string() }),
  async execute({ script }, ctx) {
    console.log("ctx.toolName:", ctx.toolName);
    console.log("ctx.callId:", ctx.callId);
    console.log("ctx.session.id:", ctx.session.id);

    const sandbox = await ctx.getSandbox();
    console.log("getSandbox() resolvePath('.'):", sandbox.resolvePath("."));

    await sandbox.writeTextFile({ path: "analysis.py", content: script });
    console.log("wrote analysis.py");

    const result = await sandbox.run({ command: "python3 analysis.py" });
    console.log("run exitCode:", result.exitCode);
    console.log("run stdout:", JSON.stringify(result.stdout));
    if (result.exitCode !== 0) {
      console.log("run stderr:", result.stderr);
      throw new Error(`python failed: ${result.stderr}`);
    }
    return { answer: result.stdout.trim() };
  },
});

const ctx = createToolContext({
  workspaceRoot: dir,
  toolName: "run_python",
  callId: "demo-1",
  session: { id: "sess-demo" },
});

const out = await tool.execute({ script: "print(40 + 2)\n" }, ctx);
console.log("tool returned:", out);
console.log(
  "file on disk:",
  JSON.stringify(readFileSync(path.join(dir, "analysis.py"), "utf8")),
);
console.log("OK — getSandbox = write/run/read in the tool workspace");
