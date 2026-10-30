import { z } from "zod";

import { defineTool } from "../../../src/eve-tools.ts";

/** Eve-shaped: `import { defineTool } from "eve/tools"` + Zod inputSchema. */
export default defineTool({
  description: "Get the current weather for a city.",
  inputSchema: z.object({ city: z.string().min(1) }),
  annotations: { readOnlyHint: true, openWorldHint: true },
  async execute({ city }) {
    return {
      city,
      condition: "sunny",
      temperatureF: 72,
      source: "beyond-agent-tools-mcp",
    };
  },
});
