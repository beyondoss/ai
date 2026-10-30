import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { loadTools } from "../src/load.js";
import { writeManifest } from "../src/manifest.js";
import { handleRpc } from "../src/rpc.js";

const weatherTool = `export default {
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
  it("loads tools, stamps CachedTool manifest, and answers tools/call", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "agent-tools-"));
    // .mjs so Node/vitest can import without a TS loader (Bun loads .ts in the sandbox).
    await writeFile(path.join(dir, "get_weather.mjs"), weatherTool, "utf8");

    const tools = await loadTools(dir);
    expect(tools.map((t) => t.name)).toEqual(["get_weather"]);

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
      { env: process.env, workspaceRoot: dir },
    );
    expect(r?.result).toMatchObject({
      structuredContent: { city: "NYC", condition: "sunny", temperatureF: 72 },
    });
  });
});
