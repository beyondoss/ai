import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { defineTool } from "../src/define-tool.js";
import { loadTools } from "../src/load.js";
import { handleRpc, serializeRpc } from "../src/rpc.js";

const base = { workspaceRoot: "/tmp" };

function toolReturning(result: unknown) {
  return defineTool({
    description: "t",
    inputSchema: { type: "object", properties: {} },
    execute: () => result,
  });
}

async function call(result: unknown, id: string | number = 1) {
  const response = await handleRpc(
    {
      jsonrpc: "2.0",
      id,
      method: "tools/call",
      params: { name: "t", arguments: {} },
    },
    [{ name: "t", def: toolReturning(result), sourcePath: "t.mjs" }],
    base,
  );
  expect(response).not.toBeNull();
  return response!;
}

describe("loadTools compiles JSON Schema once", () => {
  it("defineTool is the only conversion", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "schema-once-"));
    const g = globalThis as { __jsonSchemaInputs?: number };
    g.__jsonSchemaInputs = 0;
    await writeFile(
      path.join(dir, "once.mjs"),
      `const schema = {
  "~standard": {
    version: 1,
    vendor: "test",
    validate: (v) => ({ value: v }),
    jsonSchema: {
      input: () => {
        globalThis.__jsonSchemaInputs = (globalThis.__jsonSchemaInputs ?? 0) + 1;
        return { type: "object", properties: { city: { type: "string" } } };
      },
    },
  },
};
export default {
  description: "once",
  inputSchema: schema,
  execute: () => ({ ok: true }),
};
`,
      "utf8",
    );
    const tools = await loadTools(dir);
    expect(g.__jsonSchemaInputs).toBe(1);
    expect(tools[0]?.def.jsonInputSchema).toMatchObject({ type: "object" });
    delete g.__jsonSchemaInputs;
  });

  it("a module that already carries jsonInputSchema is not converted", async () => {
    const dir = await mkdtemp(path.join(tmpdir(), "schema-prebuilt-"));
    const g = globalThis as { __jsonSchemaInputs?: number };
    g.__jsonSchemaInputs = 0;
    await writeFile(
      path.join(dir, "prebuilt.mjs"),
      `const schema = {
  "~standard": {
    version: 1,
    vendor: "test",
    validate: (v) => ({ value: v }),
    jsonSchema: {
      input: () => {
        globalThis.__jsonSchemaInputs = (globalThis.__jsonSchemaInputs ?? 0) + 1;
        return { type: "object" };
      },
    },
  },
};
export default {
  description: "prebuilt",
  inputSchema: schema,
  jsonInputSchema: { type: "object", properties: {} },
  execute: () => ({ ok: true }),
};
`,
      "utf8",
    );
    const tools = await loadTools(dir);
    expect(g.__jsonSchemaInputs).toBe(0);
    expect(tools[0]?.def.jsonInputSchema).toEqual({
      type: "object",
      properties: {},
    });
    delete g.__jsonSchemaInputs;
  });
});

describe("serializeRpc matches JSON.stringify without a second walk", () => {
  const samples: unknown[] = [
    { city: "NYC", temperatureF: 72 },
    { nested: { flags: [true, false], note: "say \"hi\"\nnext" } },
    ["a", 1, null],
    "plain",
    'say "hi" \\ \u2028',
    "",
    0,
    -0,
    1.5,
    true,
    false,
    null,
    { skip: undefined, keep: 1 },
    [undefined, "x"],
  ];

  it.each(samples)("wire bytes match for %j", async (sample) => {
    const response = await call(sample);
    expect(serializeRpc(response)).toBe(JSON.stringify(response));
  });

  it("omits structuredContent when execute returns undefined", async () => {
    const response = await call(undefined);
    const wire = serializeRpc(response);
    expect(wire).toBe(JSON.stringify(response));
    expect(JSON.parse(wire).result.structuredContent).toBeUndefined();
    expect(JSON.parse(wire).result.content[0].text).toBe("null");
  });

  it("string ids and batches stay byte-identical", async () => {
    const a = await call({ n: 1 }, "abc");
    const b = await call("ok", 2);
    expect(serializeRpc(a)).toBe(JSON.stringify(a));
    expect(serializeRpc([a, b])).toBe(JSON.stringify([a, b]));
  });

  it("does not call toJSON again while writing the envelope", async () => {
    let walks = 0;
    const value = {
      toJSON() {
        walks += 1;
        return { walks };
      },
    };
    const response = await call(value);
    expect(walks).toBe(1);
    const wire = serializeRpc(response);
    expect(walks).toBe(1);
    const parsed = JSON.parse(wire) as {
      result: { content: { text: string }[]; structuredContent: { walks: number } };
    };
    expect(parsed.result.content[0]?.text).toBe('{"walks":1}');
    expect(parsed.result.structuredContent).toEqual({ walks: 1 });
  });

  it("list and error responses still match JSON.stringify", async () => {
    const def = toolReturning({ ok: true });
    const list = await handleRpc(
      { jsonrpc: "2.0", id: 1, method: "tools/list" },
      [{ name: "t", def, sourcePath: "t.mjs" }],
      base,
    );
    expect(list).not.toBeNull();
    expect(serializeRpc(list!)).toBe(JSON.stringify(list));

    const bad = await handleRpc(
      {
        jsonrpc: "2.0",
        id: null,
        method: "tools/call",
        params: { name: "missing", arguments: {} },
      },
      [{ name: "t", def, sourcePath: "t.mjs" }],
      base,
    );
    expect(bad).not.toBeNull();
    expect(serializeRpc(bad!)).toBe(JSON.stringify(bad));
  });
});
