# Beyond AI

Beyond's open-source AI stack, a Rust workspace for Linux:

- a **gateway** that sits between apps and LLM providers, authenticates the
  caller, swaps in the real provider key, and bills every request;
- a **coding agent** that runs a model with file, shell and MCP tools: as a
  one-shot CLI, a headless server, or a multi-tenant service;
- the **libraries** they share, and the **contracts, proofs and demos** that
  show the pieces hold together.

## Packages

| Path                                       | Package                  | What it is                                                                              |
| ------------------------------------------ | ------------------------ | --------------------------------------------------------------------------------------- |
| [`crates/gateway`](crates/gateway)         | `beyond-ai`              | The LLM egress proxy (binary `beyond-ai`).                                              |
| [`crates/agent`](crates/agent)             | `beyond-ai-agent`        | The agent: `run`, `serve`, `serve --service` (binary `beyond-ai-agent`).                |
| [`crates/agent-core`](crates/agent-core)   | `beyond-ai-agent-core`   | The agent loop as a library: messages, tool dispatch, compaction, three provider wires. |
| [`crates/providers`](crates/providers)     | `beyond-ai-providers`    | Provider table, model catalog and the pricing contract. No dependencies.                |
| [`crates/exec-notify`](crates/exec-notify) | `beyond-ai-exec-notify`  | Exec protocol v1.4 job-end callback rules as a pure function. No dependencies.          |
| `crates/demo`                              | `beyond-ai-demo`         | Sandbox exec endpoint, key and grant minters, chat client for `deploy/demo`.            |
| `crates/fleet-sim`                         | `beyond-ai-fleet-sim`    | Runs a replica set against a shared filesystem and checks the fleet design's claims.    |
| `crates/test-support`                      | `beyond-ai-test-support` | Test doubles shared by the agent's tests and the simulator.                             |
| `crates/verify`                            | `beyond-ai-verify`       | Claim registry and defect ledger for the gateway ([`verify/`](verify/README.md)).       |
| `crates/rates-sync`                        | `beyond-ai-rates-sync`   | Fetches vendors' pricing sources and generates the rate table.                          |

The last five are not published.

```
 app / SDK ─────────► gateway ───────► OpenAI, Anthropic, OpenRouter, Bedrock, …
                         ▲
 agent ── agent-core ────┘ (virtual key; or straight to a provider with your own key)
   │
   └── exec endpoint (a sandbox that runs the tools' commands) ── exec-notify (job-end callbacks)

 gateway and agent-core both read providers: provider table, catalog, pricing
```

The gateway and the agent are independent. The gateway serves any OpenAI or
Anthropic client, and the agent talks to a provider directly when no gateway is
configured. Together, the agent holds a gateway-issued `bai_v1` virtual key and
never sees a provider key.

## The gateway

An app keeps its stock OpenAI or Anthropic SDK and changes the base URL. The
gateway:

- verifies a signed virtual key without a lookup, or forwards any other token
  to its provider unchanged;
- routes by path (`/groq/openai/v1/…`) to 12 built-in providers, or by model to
  a catalog with failover and OpenAI/Anthropic wire translation;
- streams responses and writes one priced `ai.usage` row per request;
- enforces revocation, quota, per-key rate limits and per-provider circuit
  breakers.

```sh
cargo run -p beyond-ai --release     # BYO keys need no config; listens on :8080
curl localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}'
```

Managed keys, the catalog and configuration: [`crates/gateway`](crates/gateway/README.md).

## The agent

`beyond-ai-agent` reads a task, calls a model, and runs tools (`read`, `write`,
`edit`, `bash`, `grep`, `find`, `ls`, background jobs, MCP servers) until it is
done.

```sh
ANTHROPIC_API_KEY=sk-ant-... cargo run -p beyond-ai-agent -- run "add a CHANGELOG entry"
```

| Mode              | What you get                                                                                              |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| `run`             | One task; output on stdout.                                                                               |
| `serve`           | A JSON-lines protocol over stdio, WebSocket, HTTP POST or a Unix socket; many sessions per process.       |
| `serve --service` | One replica for many tenants; each connection carries a signed grant naming the sandbox its tools run in. |

Static binaries for `linux-amd64` and `linux-arm64` are on the
[releases page](https://github.com/beyondoss/ai/releases), and the service image
is `ghcr.io/beyondoss/ai-agent`. Flags, providers, sessions, the exec protocol
and service mode: [`crates/agent`](crates/agent/README.md).

## Running agents as a service

Service mode splits an agent into a replica that holds the conversation and a
sandbox that holds the workspace.

| Document                                           | Covers                                                                    |
| -------------------------------------------------- | ------------------------------------------------------------------------- |
| [`crates/agent/FLEET.md`](crates/agent/FLEET.md)   | What the edge in front of a fleet must do; sizing, alerts, scaling, cost. |
| [`crates/agent/BEYOND.md`](crates/agent/BEYOND.md) | What Beyond's platform must provide, and the conformance kits.            |
| [`deploy/demo`](deploy/demo/README.md)             | The whole stack on AWS from one model key: `up.sh`, `demo.sh`, `down.sh`. |
| [`deploy/efs-proof`](deploy/efs-proof/README.md)   | The simulator's scenarios on real ECS and EFS, with results.              |

## Proof

| What                                                       | Run                                                                                  |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Gateway claims and defects ([`verify/`](verify/README.md)) | `mise run verify:status`; `mise run verify:live` adds real providers (spends money). |
| Fleet design against a shared filesystem                   | `mise run sim:list`, `mise run sim:matrix`                                           |
| Pricing against vendors' published rates                   | `mise run rates:sync`, `mise run pricing:vectors`                                    |

## Develop

Tasks run with [mise](https://mise.jdx.dev); `mise tasks` lists them.

```sh
mise run ai:mint-dev-key     # a dev-only signing key and bai_v1 token for the gateway
mise run test:unit:rs        # unit tests
mise run bench               # micro-benchmarks and end-to-end throughput
cargo nextest run -p beyond-ai-agent -E 'test(/^serve_approval::/)'   # one test module
```

CI runs on Linux only, the only platform the project supports. Markdown is
formatted by `dprint` (`dprint check`). `mise run audit` checks dependencies
against RustSec advisories.

The agent crates are modeled on [pi](https://github.com/badlogic/pi-mono) (MIT,
Mario Zechner); its license is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
