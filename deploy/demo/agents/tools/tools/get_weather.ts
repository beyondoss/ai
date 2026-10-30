import { defineTool } from "eve/tools";
import { z } from "zod";

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
