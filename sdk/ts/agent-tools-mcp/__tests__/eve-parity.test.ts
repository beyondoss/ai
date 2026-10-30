/**
 * Eve tools parity — proof, not marketing.
 *
 * Source of truth: https://eve.dev/docs/tools (and vercel/eve docs/tools/overview.mdx).
 * Each `it` is a named Eve claim. Gaps use `it.fails` so the suite stays honest:
 * expected failures = not parity yet. When we implement them, flip to `it`.
 */
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import { z } from "zod";

import { defineTool } from "../src/define-tool.js";
import { loadTools } from "../src/load.js";
import { handleRpc } from "../src/rpc.js";
import { runTool } from "../src/run-tool.js";
import {
  inputSchemaToJsonSchema,
  validateToolInput,
} from "../src/schema.js";

const ctx = { env: process.env, workspaceRoot: "/tmp" };

describe("Eve parity: defineTool authoring (docs/tools)", () => {
  it("docs snippet: Zod inputSchema + execute({ city })", async () => {
    // Exact shape from eve.dev/docs/tools
    const tool = defineTool({
      description: "Get the current weather for a city.",
      inputSchema: z.object({ city: z.string().min(1) }),
      async execute({ city }) {
        return { city, condition: "Sunny", temperatureF: 72 };
      },
    });
    expect(await runTool(tool, { city: "NYC" }, ctx)).toEqual({
      city: "NYC",
      condition: "Sunny",
      temperatureF: 72,
    });
  });

  it("filename stem is the tool name (tools/get_weather.ts → get_weather)", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "eve-parity-"));
    await writeFile(
      path.join(dir, "get_weather.ts"),
      `import { defineTool } from ${JSON.stringify(
        pathToFileURL(path.join(import.meta.dirname, "../src/eve-tools.ts")).href,
      )};

import { z } from "zod";
export default defineTool({
  description: "Get the current weather for a city.",
  inputSchema: z.object({ city: z.string().min(1) }),
  async execute({ city }) { return { city }; },
});
`,
      "utf8",
    );
    const tools = await loadTools(dir);
    expect(tools.map((t) => t.name)).toEqual(["get_weather"]);
  });

  it("inputSchema required — missing throws at defineTool", () => {
    expect(() =>
      defineTool({
        description: "x",
        // @ts-expect-error parity: Eve requires inputSchema
        inputSchema: undefined,
        execute: () => ({}),
      }),
    ).toThrow(/inputSchema/);
  });

  it("no-input tools use z.object({})", async () => {
    const tool = defineTool({
      description: "ping",
      inputSchema: z.object({}),
      execute: () => ({ ok: true }),
    });
    expect(await runTool(tool, {}, ctx)).toEqual({ ok: true });
  });

  it("Zod / Standard Schema: validate before execute (reject bad input)", async () => {
    let ran = false;
    const tool = defineTool({
      description: "t",
      inputSchema: z.object({ city: z.string().min(1) }),
      execute: () => {
        ran = true;
        return {};
      },
    });
    await expect(runTool(tool, { city: "" }, ctx)).rejects.toThrow();
    expect(ran).toBe(false);
  });

  it("plain JSON Schema: reaches model without $schema", () => {
    const tool = defineTool({
      description: "t",
      inputSchema: {
        $schema: "https://json-schema.org/draft/2020-12/schema",
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
      },
      execute: ({ city }: { city: string }) => ({ city }),
    });
    expect(tool.jsonInputSchema.$schema).toBeUndefined();
    expect(inputSchemaToJsonSchema(tool.inputSchema).$schema).toBeUndefined();
  });

  it("plain JSON Schema: validate required props before execute", async () => {
    const tool = defineTool({
      description: "t",
      inputSchema: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
      },
      execute: (input) => input,
    });
    await expect(runTool(tool, {}, ctx)).rejects.toThrow(/city/);
  });

  it("Zod → JSON Schema for tools/list (MCP)", async () => {
    const tool = defineTool({
      description: "Get the current weather for a city.",
      inputSchema: z.object({ city: z.string().min(1) }),
      execute: ({ city }) => ({ city }),
    });
    const list = await handleRpc(
      { jsonrpc: "2.0", id: 1, method: "tools/list" },
      [{ name: "get_weather", def: tool, sourcePath: "x" }],
      ctx,
    );
    const schema = (list?.result as { tools: { inputSchema: Record<string, unknown> }[] })
      .tools[0]?.inputSchema;
    expect(schema?.type).toBe("object");
    expect(schema?.$schema).toBeUndefined();
  });

  it("toModelOutput projects execute return for the model", async () => {
    const tool = defineTool({
      description: "t",
      inputSchema: z.object({ city: z.string() }),
      execute: ({ city }) => ({ city, huge: "secret" }),
      toModelOutput: (out) => ({ city: out.city }),
    });
    expect(await runTool(tool, { city: "SF" }, ctx)).toEqual({ city: "SF" });
  });

  it("execute may be sync", async () => {
    const tool = defineTool({
      description: "t",
      inputSchema: z.object({ n: z.number() }),
      execute: ({ n }) => n * 2,
    });
    expect(await runTool(tool, { n: 21 }, ctx)).toBe(42);
  });

  it("execute may be an async generator (last yield)", async () => {
    const tool = defineTool({
      description: "t",
      inputSchema: z.object({}),
      async *execute() {
        yield "a";
        yield "b";
      },
    });
    expect(await runTool(tool, {}, ctx)).toBe("b");
  });
});

describe("Eve parity: defaults, deferred, endsTurn, outputSchema", () => {
  it("JSON Schema: fill omitted properties from default (Eve docs)", async () => {
    const tool = defineTool({
      description: "t",
      inputSchema: {
        type: "object",
        properties: {
          city: { type: "string", default: "SF" },
        },
      },
      execute: (input) => input,
    });
    expect(await runTool(tool, {}, ctx)).toEqual({ city: "SF" });
  });

  it("deferred: true omits tool from tools/list (still callable)", async () => {
    const tool = defineTool({
      deferred: true,
      description: "Export a finished report as a PDF.",
      inputSchema: z.object({ reportId: z.string() }),
      execute: ({ reportId }) => ({ reportId }),
    });
    const list = await handleRpc(
      { jsonrpc: "2.0", id: 1, method: "tools/list" },
      [{ name: "export_report", def: tool, sourcePath: "x" }],
      ctx,
    );
    const names = (
      list?.result as { tools: { name: string }[] }
    ).tools.map((t) => t.name);
    expect(names).not.toContain("export_report");

    const call = await handleRpc(
      {
        jsonrpc: "2.0",
        id: 2,
        method: "tools/call",
        params: { name: "export_report", arguments: { reportId: "r1" } },
      },
      [{ name: "export_report", def: tool, sourcePath: "x" }],
      ctx,
    );
    expect(call?.result).toMatchObject({
      structuredContent: { reportId: "r1" },
    });
  });

  it("endsTurn: true appends Eve turn-ending sentence to model description", async () => {
    const tool = defineTool({
      description: "React to the current message.",
      endsTurn: true,
      inputSchema: z.object({ emoji: z.string() }),
      execute: ({ emoji }) => ({ emoji }),
    });
    const list = await handleRpc(
      { jsonrpc: "2.0", id: 1, method: "tools/list" },
      [{ name: "react", def: tool, sourcePath: "x" }],
      ctx,
    );
    const desc = (list?.result as { tools: { description: string }[] }).tools[0]
      ?.description;
    expect(desc).toContain("Calling this tool ends your turn once it succeeds");
    // eve info / authored description unchanged on the def
    expect(tool.description).toBe("React to the current message.");
  });

  it("outputSchema validates execute return (Eve)", async () => {
    const tool = defineTool({
      description: "t",
      inputSchema: z.object({ city: z.string() }),
      outputSchema: z.object({ city: z.string(), temperatureF: z.number() }),
      execute: ({ city }) => ({ city }), // missing temperatureF
    });
    await expect(runTool(tool, { city: "X" }, ctx)).rejects.toThrow(/outputSchema/);
  });
});

describe("Eve parity: JSON Schema format annotation", () => {
  it("format is annotation only — do not reject on format (Eve)", async () => {
    const tool = defineTool({
      description: "t",
      inputSchema: {
        type: "object",
        properties: { email: { type: "string", format: "email" } },
        required: ["email"],
      },
      execute: (input) => input,
    });
    expect(
      await validateToolInput(tool.inputSchema, { email: "not-an-email" }),
    ).toEqual({ email: "not-an-email" });
  });
});
