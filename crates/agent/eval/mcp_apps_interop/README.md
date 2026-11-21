# MCP Apps interop check

The real `agent serve` daemon, as an MCP Apps host, against servers built on the **official**
`@modelcontextprotocol/ext-apps` SDK, with the view side played by the SDK's own `App` class. The
agent's own tests (`tests/mcp_apps*.rs`) run only against our hand-rolled fixture server; this is the
independent counterpart.

## Run

```sh
mise run test:mcp-apps-interop           # builds the agent, npm ci, runs check.mjs
# or by hand
cargo build -p beyond-agent --bin agent
cd crates/agent/eval/mcp_apps_interop && npm ci
AGENT_BIN=../../../../target/debug/agent LOG_DIR=/tmp/interop node check.mjs
```

Node 22+. Exit code is the number of failed assertions. About 3 s once the agent is built. `LOG_DIR`
keeps `frames.jsonl` (every WebSocket frame, both directions), `agent.stderr.log` and the map server's log.

## Independent parts (pinned in `package.json` / `package-lock.json`)

| What                                                                                                            | Pin   |
| --------------------------------------------------------------------------------------------------------------- | ----- |
| `@modelcontextprotocol/ext-apps` (SDK: `registerAppResource`, `registerAppTool`, the view's `App`)              | 2.1.0 |
| `@modelcontextprotocol/server-map` (official example, Streamable HTTP; `_meta.ui` CSP on the content item only) | 2.1.0 |
| `@modelcontextprotocol/server-system-monitor` (official example, stdio; app-only `poll-system-stats`)           | 2.1.0 |
| `@modelcontextprotocol/client` (the oracle: reads what each server really lists and serves)                     | 2.0.0 |
| `tiny_server.mjs`: ours, but built with the SDK (no shipped example puts `_meta.ui` on the listing alone)       | stdio |

The examples are the npm builds of `modelcontextprotocol/ext-apps` examples at v2.1.0.

## No real model

The turn needs a model to call the app's tool. A scripted Anthropic-SSE server (in `check.mjs`) answers
a prompt `OPEN:<tool> <json>` with a `tool_use` and the follow-up with text. It decides nothing under
test. The view's `tools/list` and `tools/call` go through `mcp_app_request`, as a renderer sends them.

## Assertions

| id         | what                                                                                                                                          |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| A0, A1, A2 | listing-only `_meta.ui` (CSP + permissions) is on `mcp_app_open.resource._meta.ui`; the rest of the resource is untouched                     |
| B1         | the official map server's content-item `_meta.ui` is forwarded whole                                                                          |
| B2         | with `_meta.ui` on both, the content item's wins whole: no listing permissions or domains leak in                                             |
| C1, C2, C3 | an SDK view's `tools/list` returns exactly the app-visible tools (tiny, system-monitor), model-only excluded, with the server's `inputSchema` |
| D0-D8      | nothing regresses: views open, results follow, `callServerTool` works for app-only tools and is refused for model-only ones, no error frames  |

Expected before "views get the listing's `_meta.ui` and may call `tools/list`": A1, C1, C2, C3 fail
(`tools/list` is not proxied for MCP Apps); everything else passes.
