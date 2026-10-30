/** Plain ToolDefinition (no import): the AOT guest loads this file from the seeded bundle. */
export default {
  description: "Get the weather for a city",
  inputSchema: {
    type: "object",
    properties: {
      city: { type: "string", description: "City name" },
    },
    required: ["city"],
    additionalProperties: false,
  },
  annotations: { readOnlyHint: true, openWorldHint: true },
  async execute({ city }: { city?: unknown }) {
    const name = typeof city === "string" ? city : "unknown";
    return {
      city: name,
      condition: "sunny",
      temperatureF: 72,
      source: "beyond-agent-tools-mcp",
    };
  },
};
