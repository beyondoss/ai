# Beyond Agent Harness — Core Architecture

`beyond-ai-agent-core` (lib `agent_core`) is the runtime-agnostic core of the Beyond agent harness,
modeled on [Pi](https://github.com/badlogic/pi-mono) (`pi-agent-core` + the dialect half of `pi-ai`)
and ported to Rust. It holds **no** HTTP, provider, or executor code, so it unit-tests without a
network or a live model — the same discipline the gateway uses to keep its logic testable without
Pingora.

## The Beyond twist

The model layer never manages provider keys or endpoints. It speaks OpenAI/Anthropic **wire** to the
Beyond gateway, which owns routing, auth, and metering. So the only part of `pi-ai` worth porting is
the **dialect-agnostic message model**; provider selection is the gateway's job. The agent crate has
**no dependency on the gateway crate** — its sole contract is HTTP wire to a base URL.

## Modules

| Module      | Type                 | Role                                                                                                                                                                                                                                    |
| ----------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `message`   | data                 | Dialect-agnostic conversation model: `Role`, `ContentBlock` (`Text`/`ToolUse`/`ToolResult`), `Message`, `ToolDef`, `StreamEvent`, `StopReason`. The single internal representation; wire adapters map it to/from each provider's shape. |
| `tool`      | seam (extensibility) | `Tool` trait + `ToolRegistry`. Capabilities are registered values — the core four (Read/Write/Edit/Bash) and Beyond primitives (fork/sync/logs) are all just tools. Last-registration-wins lets an extension override a built-in.       |
| `transport` | seam (network)       | `ModelRequest` + `ModelTransport` trait returning an `EventStream` of `StreamEvent`s. The loop depends only on this; the real gateway client and the test `MockTransport` both implement it.                                            |
| `session`   | data                 | `Session`: message history + token/step counters. `serde`-serializable so a headless run persists and a client can reattach.                                                                                                            |
| `error`     | data                 | `Error` (loop/transport) and `ToolError` (a tool's own failure → an error `tool_result`, not an aborted run).                                                                                                                           |

## The two seams

Everything above the wire is testable with mocks because of two trait boundaries:

- **`Tool`** — tests register a mock (`EchoTool`) to exercise dispatch without a real capability.
- **`ModelTransport`** — tests replay scripted `StreamEvent`s to exercise the loop without a network.

## Observation surface

`Agent::run` exposes only streamed model events (`FnMut(&StreamEvent)`); `Agent::run_events` exposes
the full [`AgentEvent`] stream — `Stream(StreamEvent)`, `ToolStart`/`ToolEnd` (tool boundaries), and
`TurnEnd`. The headless `serve` in the `beyond-ai-agent` crate serializes these to its clients.

## Milestone status

Complete. Built and tested: the type model, the `Tool`/`ToolRegistry` seam, the `ModelTransport`
seam + `GatewayClient` HTTP transport, `MockTransport`, the OpenAI/Anthropic wire dialects, the agent
loop (`run`/`run_events`), and `Session`. The coding tools (read/write/edit/bash/ls/grep/find), the
Beyond platform tools (fork/sync/logs), the `run` CLI, and the headless `serve` control protocol live
in the `beyond-ai-agent` crate. End-to-end proven against the real `beyond-ai` gateway binary
(auth + key-swap + routing) through to a mock upstream.
