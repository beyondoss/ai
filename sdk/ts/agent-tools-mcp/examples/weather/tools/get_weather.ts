import { defineTool } from "../../../src/eve-tools.ts";

/** Eve-shaped tool: `import { defineTool } from "eve/tools"` via the shim. */
export default defineTool({
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
  async execute({ city }) {
    const name = typeof city === "string" ? city : "unknown";
    return { city: name, condition: "sunny", temperatureF: 72 };
  },
});
