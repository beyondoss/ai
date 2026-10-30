import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { createToolContext } from "../src/context.js";
import { defineTool } from "../src/define-tool.js";
import { loadTools } from "../src/load.js";
import { writeManifest } from "../src/manifest.js";
import { handleRpc } from "../src/rpc.js";
import { runTool } from "../src/run-tool.js";
import { inputSchemaToJsonSchema, validateToolInput } from "../src/schema.js";

const weatherMjs = `export default {
  description: "Get the weather for a city",
  inputSchema: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
  async execute({ city }) {
    return { city, condition: "sunny", temperatureF: 72 };
  },
};
`;

describe("agent-tools-mcp", () => {
  it("loads JSON-Schema tools, stamps CachedTool manifest, and answers tools/call", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "agent-tools-"));
    await writeFile(path.join(dir, "get_weather.mjs"), weatherMjs, "utf8");

    const tools = await loadTools(dir);
    expect(tools.map((t) => t.name)).toEqual(["get_weather"]);
    expect(tools[0]?.def.jsonInputSchema).toMatchObject({ type: "object" });

    const out = path.join(dir, ".mcp-tools.json");
    const file = await writeManifest(tools, { name: "tools", out });
    expect(file.tools?.[0]?.remote_name).toBe("get_weather");
    expect(file.tools?.[0]?.input_schema).toMatchObject({ type: "object" });

    const onDisk = JSON.parse(await readFile(out, "utf8")) as typeof file;
    expect(onDisk).toEqual(file);

    const r = await handleRpc(
      {
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: { name: "get_weather", arguments: { city: "NYC" } },
      },
      tools,
      { workspaceRoot: dir },
    );
    expect(r?.result).toMatchObject({
      structuredContent: { city: "NYC", condition: "sunny", temperatureF: 72 },
    });
  });

  it("defineTool + Zod: JSON Schema for MCP, validate before execute", async () => {
    const tool = defineTool({
      description: "Get the current weather for a city.",
      inputSchema: z.object({ city: z.string().min(1) }),
      async execute({ city }) {
        return { city, condition: "sunny", temperatureF: 72 };
      },
    });

    const json = inputSchemaToJsonSchema(tool.inputSchema);
    expect(json).toMatchObject({ type: "object" });
    expect(json.$schema).toBeUndefined();
    expect(tool.jsonInputSchema).toEqual(json);

    expect(await validateToolInput(tool.inputSchema, { city: "SF" })).toEqual({
      city: "SF",
    });
    await expect(validateToolInput(tool.inputSchema, { city: "" })).rejects
      .toThrow();
    await expect(validateToolInput(tool.inputSchema, {})).rejects.toThrow();

    expect(
      await runTool(
        tool,
        { city: "NYC" },
        createToolContext({ workspaceRoot: "/tmp", toolName: "get_weather" }),
      ),
    ).toMatchObject({ city: "NYC", temperatureF: 72 });

    const bad = await handleRpc(
      {
        jsonrpc: "2.0",
        id: 2,
        method: "tools/call",
        params: { name: "get_weather", arguments: { city: "" } },
      },
      [{ name: "get_weather", def: tool, sourcePath: "get_weather.ts" }],
      { workspaceRoot: "/tmp" },
    );
    expect(bad?.result).toMatchObject({ isError: true });
  });
});
