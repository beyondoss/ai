// MCP Apps interop check: the real `agent serve` daemon (a host) against independent servers built
// on the official `@modelcontextprotocol/ext-apps` SDK, with the view side played by the SDK's own
// `App` class and the expected values read straight from each server by the official MCP client.
//
//   npm ci && AGENT_BIN=target/debug/agent node check.mjs
//
// Env: AGENT_BIN (default target/debug/agent), LOG_DIR (frames and agent stderr are saved there).
// Exit code = number of failed assertions.
//
// The agent needs a model to run a turn. A scripted Anthropic-SSE server stands in for it: it answers
// the first request of a turn with a `tool_use` for the tool the prompt names (`OPEN:<tool> <json>`)
// and the follow-up with text. The model decides nothing; everything under test is what the agent
// does with the servers' tool, resource and bridge traffic.
import {
  Client,
  StreamableHTTPClientTransport,
} from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { App } from "@modelcontextprotocol/ext-apps";
import { spawn } from "node:child_process";
import {
  appendFileSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:http";
import { createServer as netServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import WebSocket from "ws";
import { CONTENT_UI, LISTING_UI } from "./consts.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const AGENT = resolve(process.env.AGENT_BIN ?? "target/debug/agent");
const LOG_DIR = resolve(
  process.env.LOG_DIR ?? mkdtempSync(join(tmpdir(), "mcp-apps-interop-")),
);
mkdirSync(LOG_DIR, { recursive: true });
const FRAMES = join(LOG_DIR, "frames.jsonl");
writeFileSync(FRAMES, "");
const bin = (name) => join(HERE, "node_modules", ".bin", name);
const UI_CAP = {
  extensions: {
    "io.modelcontextprotocol/ui": { mimeTypes: ["text/html;profile=mcp-app"] },
  },
};

const results = [];
function check(id, name, ok, detail = "") {
  results.push({ id, name, ok: !!ok });
  console.log(
    `${ok ? "PASS" : "FAIL"} ${id} ${name}${
      ok || !detail ? "" : `\n     ${detail}`
    }`,
  );
}
const j = (v) => JSON.stringify(v);
const freePort = () =>
  new Promise((res) => {
    const s = netServer().listen(0, "127.0.0.1", () => {
      const p = s.address().port;
      s.close(() => res(p));
    });
  });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitPort(port, ms = 30000) {
  const end = Date.now() + ms;
  while (Date.now() < end) {
    const ok = await new Promise((r) => {
      const s = netServer();
      s.once("error", () => r(true)); // in use: the server is listening
      s.listen(port, "127.0.0.1", () => s.close(() => r(false)));
    });
    if (ok) return;
    await sleep(100);
  }
  throw new Error(`port ${port} never opened`);
}

// ---- scripted model -------------------------------------------------------------------------
const sse = (events) => events.map((e) => `data: ${j(e)}\n\n`).join("");
const textTurn = (t) =>
  sse([
    {
      type: "message_start",
      message: { usage: { input_tokens: 10, output_tokens: 1 } },
    },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: t },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn" },
      usage: { output_tokens: 4 },
    },
    { type: "message_stop" },
  ]);
const toolTurn = (id, name, args) =>
  sse([
    {
      type: "message_start",
      message: { usage: { input_tokens: 10, output_tokens: 1 } },
    },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "tool_use", id, name, input: {} },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "input_json_delta", partial_json: args },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "message_delta",
      delta: { stop_reason: "tool_use" },
      usage: { output_tokens: 8 },
    },
    { type: "message_stop" },
  ]);
let toolSeq = 0;
function modelServer() {
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      let reply = textTurn("ok");
      try {
        const last = JSON.parse(body).messages?.at(-1);
        const blocks = typeof last.content === "string"
          ? [{ type: "text", text: last.content }]
          : last.content;
        const text = blocks.filter((b) => b.type === "text").map((b) => b.text)
          .join("\n");
        const m = /OPEN:(\S+) (.*)/.exec(text);
        if (blocks.some((b) => b.type === "tool_result")) {
          reply = textTurn("done");
        } else if (m) reply = toolTurn(`toolu_${++toolSeq}`, m[1], m[2]);
      } catch {}
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(reply);
    });
  });
  return new Promise((res) => server.listen(0, "127.0.0.1", () => res(server)));
}

// ---- the agent ------------------------------------------------------------------------------
class Session {
  constructor(ws) {
    this.ws = ws;
    this.frames = [];
    this.waiters = [];
    ws.on("message", (raw) => {
      const f = JSON.parse(raw.toString());
      this.frames.push(f);
      appendFileSync(FRAMES, j(f) + "\n");
      this.waiters = this.waiters.filter((w) => !w(f));
    });
  }
  send(cmd) {
    appendFileSync(FRAMES, j({ _sent: cmd }) + "\n");
    this.ws.send(j(cmd));
  }
  // The first frame (already seen, or still to come) matching `pred`.
  next(pred, ms = 30000) {
    const seen = this.frames.find(pred);
    if (seen) return Promise.resolve(seen);
    return new Promise((res, rej) => {
      const t = setTimeout(
        () => rej(new Error("timed out waiting for a frame")),
        ms,
      );
      this.waiters.push((
        f,
      ) => (pred(f) ? (clearTimeout(t), res(f), true) : false));
    });
  }
  async command(cmd, name = cmd.type) {
    const from = this.frames.length;
    this.send(cmd);
    const r = await this.next(
      (f) => f.type === "response" && f.command === name && f.id === cmd.id,
      60000,
    ).catch(() => null);
    return r ?? { success: false, error: "no response", from };
  }
}

// ---- the view: the official SDK `App` over a transport into the agent's `mcp_app_request` ----------
// The host side of `ui/initialize` is answered here (that is the renderer's job); every other request
// the view makes is proxied to the agent exactly as a renderer would.
let viewSeq = 0; // response ids must be unique across views: a stale response must never match
function viewTransport(session, server, appId) {
  const t = {
    onmessage: undefined,
    onclose: undefined,
    onerror: undefined,
    async start() {},
    async close() {
      t.onclose?.();
    },
    async send(msg) {
      if (msg.id === undefined || !msg.method) return; // notifications from the view
      const reply = (result, error) =>
        queueMicrotask(() =>
          t.onmessage?.({
            jsonrpc: "2.0",
            id: msg.id,
            ...(error ? { error } : { result }),
          })
        );
      if (msg.method === "ui/initialize") {
        return reply({
          protocolVersion: msg.params.protocolVersion,
          hostInfo: { name: "interop-host", version: "0" },
          hostCapabilities: { serverTools: {}, serverResources: {} },
          hostContext: {},
        });
      }
      const r = await session.command(
        {
          type: "mcp_app_request",
          id: `v${++viewSeq}`,
          server,
          app_id: appId,
          request: { method: msg.method, params: msg.params ?? {} },
        },
        "mcp_app_request",
      );
      if (r.success) reply(r.data.result);
      else reply(undefined, { code: -32601, message: String(r.error) });
    },
  };
  return t;
}
async function viewApp(session, server, appId) {
  const app = new App({ name: "interop-view", version: "0" }, {}, {
    autoResize: false,
  });
  await app.connect(viewTransport(session, server, appId));
  return app;
}

// ---- the official client as the oracle -----------------------------------------------------
async function oracle(connect) {
  const client = new Client({ name: "oracle", version: "0" }, {
    capabilities: UI_CAP,
  });
  await connect(client);
  const resources = (await client.listResources()).resources;
  const tools = (await client.listTools()).tools;
  const read = async (uri) => (await client.readResource({ uri })).contents[0];
  const o = { resources, tools, read };
  o.appTools = tools.filter((t) =>
    (t._meta?.ui?.visibility ?? ["model", "app"]).includes("app")
  );
  o.close = () => client.close();
  return o;
}

// ---- run -----------------------------------------------------------------------------------
const children = [];
const spawnChild = (cmd, args, opts) => {
  const c = spawn(cmd, args, opts);
  children.push(c);
  return c;
};
async function main() {
  const model = await modelServer();
  const mapPort = await freePort();
  const agentPort = await freePort();
  const stdioSrv = (name, args = []) => ({
    command: bin(name),
    args: [...args, "--stdio"],
  });
  const tinyCmd = {
    command: process.execPath,
    args: [join(HERE, "tiny_server.mjs")],
  };
  const sysCmd = stdioSrv("mcp-system-monitor-server");

  // Streamable HTTP: the official map example (CSP on the content item only).
  spawnChild(bin("mcp-map-server"), [], {
    env: { ...process.env, PORT: String(mapPort) },
    stdio: ["ignore", "ignore", openSync(join(LOG_DIR, "map-server.log"), "w")],
  });
  await waitPort(mapPort);
  const mapUrl = `http://127.0.0.1:${mapPort}/mcp`;

  const oracles = {
    tiny: await oracle((c) => c.connect(new StdioClientTransport(tinyCmd))),
    map: await oracle((c) =>
      c.connect(new StreamableHTTPClientTransport(new URL(mapUrl)))
    ),
    sys: await oracle((c) => c.connect(new StdioClientTransport(sysCmd))),
  };

  const home = mkdtempSync(join(tmpdir(), "mcp-apps-interop-home-"));
  mkdirSync(join(home, ".beyond"));
  writeFileSync(
    join(home, ".beyond", "settings.json"),
    j({
      mcp_servers: [
        { name: "tiny", transport: "stdio", ...tinyCmd },
        { name: "sys", transport: "stdio", ...sysCmd },
        { name: "map", transport: "http", url: mapUrl },
      ],
    }),
  );
  const env = {
    ...process.env,
    HOME: home,
    BEYOND_AI_AGENT_MCP_IDLE_SECS: "0",
  };
  for (
    const k of [
      "AI_POOL_KEY_ANTHROPIC",
      "DATABASE_URL",
      "AI_DIRECT",
      "AI_PROVIDER",
      "OPENROUTER_API_KEY",
      "ANTHROPIC_API_KEY",
      "AI_AGENT_LIFECYCLE_URL",
    ]
  ) {
    delete env[k];
  }
  const agent = spawnChild(
    AGENT,
    [
      "serve",
      "--listen",
      `127.0.0.1:${agentPort}`,
      "--gateway-url",
      `http://127.0.0.1:${model.address().port}`,
      "--key",
      "bai_v1.test",
      "--model",
      "claude-test",
      "--session-dir",
      join(home, "sessions"),
    ],
    {
      env,
      stdio: [
        "ignore",
        "ignore",
        openSync(join(LOG_DIR, "agent.stderr.log"), "w"),
      ],
    },
  );
  await waitPort(agentPort);
  const ws = new WebSocket(
    `ws://127.0.0.1:${agentPort}/_beyond/agent?session_id=interop`,
  );
  await new Promise((
    res,
    rej,
  ) => (ws.once("open", res), ws.once("error", rej)));
  const s = new Session(ws);

  const declared = await s.command({
    type: "set_mcp_apps",
    id: "declare",
    enabled: true,
  });
  check(
    "D0",
    "the agent declares the three official/SDK servers as app servers",
    declared.success
      && j(declared.data?.servers?.sort()) === j(["map", "sys", "tiny"]),
    j(declared),
  );

  // Open a view: prompt the scripted model to call `server`'s `tool`.
  async function open(server, tool, args = {}) {
    const from = s.frames.length;
    s.send({
      type: "prompt",
      id: `p-${server}-${tool}`,
      message: `OPEN:mcp__${server}__${tool} ${j(args)}`,
    });
    const isOpen = (f) => f.type === "mcp_app_open" && f.server === server;
    const frame = await s.next(
      (f) => isOpen(f) && s.frames.indexOf(f) >= from,
      30000,
    ).catch(() => null);
    const result = frame
      && (await s.next(
        (f) => f.type === "mcp_app_result" && f.app_id === frame.app_id,
        30000,
      ).catch(() => null));
    const end = frame
      && (await s.next(
        (f) =>
          f.type === "event" && f.event?.kind === "tool_end"
          && f.event.id === frame.app_id,
        30000,
      ).catch(() => null));
    await s.next(
      (f) =>
        f.type === "response" && f.command === "prompt"
        && f.id === `p-${server}-${tool}`,
      30000,
    ).catch(() => null);
    return { frame, result, end, frames: s.frames.slice(from) };
  }

  // (a) the listing carries `_meta.ui`, the content item has none (SDK `registerAppResource` config).
  const listingUi = oracles.tiny.resources.find((r) =>
    r.uri === "ui://interop/listing"
  )?._meta?.ui;
  check(
    "A0",
    "oracle: the SDK server really lists `_meta.ui` and its content item has none",
    isDeepStrictEqual(listingUi, LISTING_UI)
      && (await oracles.tiny.read("ui://interop/listing"))._meta === undefined,
  );
  const a = await open("tiny", "show_listing");
  check(
    "A1",
    "listing-only: mcp_app_open.resource._meta.ui is the listing's CSP + permissions",
    isDeepStrictEqual(a.frame?.resource?._meta?.ui, LISTING_UI),
    j(a.frame?.resource?._meta),
  );
  const aRead = await oracles.tiny.read("ui://interop/listing");
  check(
    "A2",
    "listing-only: uri, mimeType and html are the server's, untouched",
    a.frame?.resource?.uri === aRead.uri
      && a.frame?.resource?.mimeType === aRead.mimeType
      && a.frame?.resource?.text === aRead.text,
  );

  // (b) the content item carries `_meta.ui`.
  const b1 = await open("map", "show-map", {
    west: -0.5,
    south: 51.4,
    east: 0.3,
    north: 51.6,
  });
  const mapRead = await oracles.map.read(oracles.map.resources[0].uri);
  check(
    "B1",
    "official map-server (HTTP): its content-item `_meta.ui` CSP reaches the view whole",
    mapRead._meta?.ui?.csp
      && isDeepStrictEqual(b1.frame?.resource?._meta, mapRead._meta),
    j(b1.frame?.resource?._meta),
  );
  const b2 = await open("tiny", "show_both");
  check(
    "B2",
    "both locations: the content item's `_meta.ui` wins whole (no listing permissions or domains leak in)",
    isDeepStrictEqual(b2.frame?.resource?._meta?.ui, CONTENT_UI),
    j(b2.frame?.resource?._meta),
  );

  // (c) the view lists tools through the SDK `App`.
  const sorted = (xs) => xs.map((t) => t.name).sort();
  const viewTools = async (server, appId) => {
    const app = await viewApp(s, server, appId);
    try {
      return {
        app,
        list: await app.request({ method: "tools/list", params: {} }),
      };
    } catch (e) {
      return { app, error: String(e?.message ?? e) };
    }
  };
  const c1 = await viewTools("tiny", a.frame.app_id);
  check(
    "C1",
    "SDK view `tools/list` (tiny): exactly the app-visible tools, model-only absent",
    c1.list && j(sorted(c1.list.tools)) === j(sorted(oracles.tiny.appTools))
      && !c1.list.tools.some((t) => t.name === "model_only"),
    c1.error ?? j(c1.list),
  );
  const b2c = await open("sys", "get-system-info");
  const c2 = await viewTools("sys", b2c.frame.app_id);
  check(
    "C2",
    "SDK view `tools/list` (official system-monitor): includes its app-only `poll-system-stats`",
    c2.list && j(sorted(c2.list.tools)) === j(sorted(oracles.sys.appTools))
      && c2.list.tools.some((t) => t.name === "poll-system-stats"),
    c2.error ?? j(c2.list),
  );
  check(
    "C3",
    "listed tools keep the server's name and inputSchema",
    !!c1.list && c1.list.tools.every((t) =>
      isDeepStrictEqual(
        t.inputSchema,
        oracles.tiny.tools.find((o) => o.name === t.name)?.inputSchema,
      )
    ),
    c1.error ?? j(c1.list),
  );

  // (d) nothing regresses.
  for (
    const [id, label, r] of [
      ["D1", "tiny/show_listing", a],
      ["D2", "tiny/show_both", b2],
      ["D3", "map/show-map", b1],
      ["D4", "sys/get-system-info", b2c],
    ]
  ) {
    check(
      id,
      `${label}: the view opens, its result follows, the tool call is not an error`,
      r.frame && r.result && r.end && r.end.event.is_error === false,
      r.frame ? j(r.end?.event) : "no mcp_app_open frame",
    );
  }
  const polled = await c2.app.callServerTool({
    name: "poll-system-stats",
    arguments: {},
  }).catch((e) => ({ isError: true, error: String(e) }));
  check(
    "D5",
    "SDK view `callServerTool` of the official app-only `poll-system-stats` succeeds",
    polled.isError !== true && polled.content?.length > 0,
    j(polled),
  );
  const only = await c1.app.callServerTool({
    name: "app_only",
    arguments: { n: 3 },
  }).catch((e) => ({ isError: true, error: String(e) }));
  check(
    "D6",
    "SDK view `callServerTool` of the app-only tool returns its result",
    only.content?.[0]?.text === "app_only:3",
    j(only),
  );
  const refused = await c1.app.callServerTool({
    name: "model_only",
    arguments: {},
  }).catch((e) => ({ isError: true, error: String(e) }));
  check(
    "D7",
    "SDK view `callServerTool` of a model-only tool is refused (visibility)",
    refused.isError === true && refused.content?.[0]?.text !== "model_only",
    j(refused),
  );
  check(
    "D8",
    "no error frames in the session",
    !s.frames.some((f) =>
      f.type === "error" || (f.type === "event" && f.event?.kind === "error")
    ),
    j(s.frames.filter((f) =>
      f.type === "error" || (f.type === "event" && f.event?.kind === "error")
    )),
  );

  ws.close();
  agent.kill("SIGTERM");
  model.close();
  await Promise.all(
    Object.values(oracles).map((o) => o.close().catch(() => {})),
  );
}

try {
  await main();
} catch (e) {
  check("ERR", `harness error: ${e?.stack ?? e}`, false);
} finally {
  for (const c of children) c.kill("SIGTERM");
}
const failed = results.filter((r) => !r.ok);
console.log(
  `\n${
    results.length - failed.length
  }/${results.length} passed; frames in ${FRAMES}`,
);
process.exit(failed.length);
