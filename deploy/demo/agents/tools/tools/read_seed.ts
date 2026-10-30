import { defineTool } from "eve/tools";
import { readFile } from "node:fs/promises";
import { z } from "zod";

// Reads a file from the seeded workspace two ways. tools/seed.txt (next to the modules) holds a
// different value, so a root resolved against the tools dir is visible.
export default defineTool({
  description: "Read a text file from the workspace.",
  inputSchema: z.object({ path: z.string().min(1) }),
  async execute({ path }, ctx) {
    const sb = await ctx.getSandbox();
    const viaSandbox = await sb.readTextFile({ path }).catch((e) =>
      "ERR:" + String(e)
    );
    const viaRoot = await readFile(ctx.workspaceRoot + "/" + path, "utf8")
      .catch((e) => "ERR:" + String(e));
    return { workspaceRoot: ctx.workspaceRoot, viaCtx: viaRoot, viaSandbox };
  },
});
