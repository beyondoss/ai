import { defineTool } from "eve/tools";
import { z } from "zod";

// availableInSubagents: false — the parent model sees it, a spawned subagent does not.
export default defineTool({
  description: "Hidden from subagents.",
  inputSchema: z.object({}),
  availableInSubagents: false,
  async execute() {
    return { quiet: true };
  },
});
