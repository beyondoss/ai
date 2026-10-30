import { defineTool } from "eve/tools";
import { z } from "zod";

// Sleeps up to 60 s, polling ctx.abortSignal. On abort it writes abort-marker.txt in the workspace.
export default defineTool({
  description: "A slow tool that notices cancellation.",
  inputSchema: z.object({ seconds: z.number().int().min(1).max(60).default(60) }),
  async execute({ seconds }, ctx) {
    const sb = await ctx.getSandbox();
    await sb.writeTextFile({ path: "slow-started.txt", content: String(Date.now()) });
    const end = Date.now() + seconds * 1000;
    while (Date.now() < end) {
      if (ctx.abortSignal.aborted) {
        await sb.writeTextFile({ path: "abort-marker.txt", content: "ABORT-OBSERVED " + Date.now() });
        return { aborted: true };
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    return { aborted: false };
  },
});
