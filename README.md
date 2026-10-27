# Beyond AI

Beyond's open-source AI stack, a Rust workspace for Linux: a **gateway** between
your apps and LLM providers, and a **coding agent** that runs a model with file,
shell and MCP tools as a one-shot CLI, a headless server, or a multi-tenant
service.

**The hard parts of running agents in production (durable sessions, scale,
security, portability) are built, proven on real infrastructure, and open.**

| The problem you no longer have                  | The answer, in numbers                                                                                                                                                                                                                                                             | Rerun the evidence                                                                                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Sessions die with their process**             | Kill a replica mid-job and another resumes the session from the shared store; an epoch fence keeps two writers out. On real ECS and EFS no acknowledged write was lost across 142,755 turns through rolling deploys and hard kills, and the network-partition takeover passes too. | [`deploy/efs-proof`](deploy/efs-proof/README.md), `mise run sim:matrix`, [fencing](crates/agent/ARCHITECTURE.md#ownership-the-epoch-fence)       |
| **Disconnects lose your place**                 | A connection is only a view. Reattach from another device or watch from several; jobs keep running in the sandbox and report back when they end. Over stdio, SSH, WebSocket, HTTP or a Unix socket.                                                                                | [`crates/agent`](crates/agent/README.md#serve), [`exec-notify`](crates/exec-notify/README.md)                                                    |
| **Scaling is your distributed-systems project** | One 0.5 vCPU, 1 GB Fargate replica held about 2,900 live sessions with 0 failed opens and 520 MB peak anonymous memory (about 170 KB per session). Replicas share a store; routing, failover and sizing are written down.                                                          | [`deploy/demo/load.sh`](deploy/demo/load.sh), [`FLEET.md`](crates/agent/FLEET.md)                                                                |
| **Security will not sign off**                  | The agent never holds a provider or connector secret. The sandbox cannot open a connection to the session store. Sessions are ciphertext at rest under a per-tenant key, and deleting the key erases them.                                                                         | [`deploy/demo`](deploy/demo/README.md) checks `sandbox-cannot-reach-store` and `sealed-at-rest`, [Sealing](crates/agent/ARCHITECTURE.md#sealing) |
| **Lock-in**                                     | Any model, with sessions that resume across providers. Any sandbox: Beyond, E2B, SSH, Docker or your own, behind one exec protocol with a conformance kit. The agent runs inside the sandbox or outside it, from one binary.                                                       | [`BEYOND.md`](crates/agent/BEYOND.md), [`providers`](crates/providers/README.md)                                                                 |
| **LLM egress means a pile of glue**             | Keep your OpenAI or Anthropic SDK and change the base URL. Virtual keys verify in about 28 µs with no lookup, models fail over across providers, and every request writes one priced row at the serving host's cost.                                                               | [`crates/gateway`](crates/gateway/README.md), [`verify/`](verify/README.md)                                                                      |
| **An agent that is actually good**              | Read, write, edit, shell and background-job tools, an MCP client, skills, memory, compaction, approval gates.                                                                                                                                                                      | [`crates/agent`](crates/agent/README.md#tools)                                                                                                   |

None of this rests on our say-so. The claim registry and defect ledger, the
fleet simulator, the real-infrastructure runs and the conformance kits are in
the repo, and you can run them: `mise run verify:status`,
`mise run sim:matrix`, `deploy/demo/demo.sh`.

## Quick start

```sh
# the agent, with your own provider key
ANTHROPIC_API_KEY=sk-ant-... cargo run -p beyond-agent -- run "add a CHANGELOG entry"

# the gateway, in front of any OpenAI client (BYO keys need no config)
cargo run -p beyond-gateway --release &
curl localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}'
```

## Packages

| Path                                       | Package               | What it is                                                                              |
| ------------------------------------------ | --------------------- | --------------------------------------------------------------------------------------- |
| [`crates/gateway`](crates/gateway)         | `beyond-gateway`      | The LLM egress proxy (binary `gateway`).                                                |
| [`crates/agent`](crates/agent)             | `beyond-agent`        | The agent: `run`, `serve`, `serve --service` (binary `agent`).                          |
| [`crates/agent-core`](crates/agent-core)   | `beyond-agent-core`   | The agent loop as a library: messages, tool dispatch, compaction, three provider wires. |
| [`crates/providers`](crates/providers)     | `beyond-providers`    | Provider table, model catalog and the pricing contract. No dependencies.                |
| [`crates/exec-notify`](crates/exec-notify) | `beyond-exec-notify`  | Exec protocol v1.4 job-end callback rules as a pure function. No dependencies.          |
| [`crates/execd`](crates/execd)             | `beyond-execd`        | The exec endpoint for a sandbox: exec protocol v1.1 to v1.4 in one static binary.       |
| `crates/demo`                              | `beyond-demo`         | Sandbox exec endpoint, key and grant minters, chat client for `deploy/demo`.            |
| `crates/fleet-sim`                         | `beyond-fleet-sim`    | Runs a replica set against a shared filesystem and checks the fleet design's claims.    |
| `crates/test-support`                      | `beyond-test-support` | Test doubles shared by the agent's tests and the simulator.                             |
| `crates/verify`                            | `beyond-verify`       | Claim registry and defect ledger for the gateway ([`verify/`](verify/README.md)).       |
| `crates/rates-sync`                        | `beyond-rates-sync`   | Fetches vendors' pricing sources and generates the rate table.                          |

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

To run the agent's tools in a sandbox, put [`execd`](crates/execd/README.md)
in the image (one 5 MB static binary, v1.1 to v1.4 conformance 32/32 live on
AWS Fargate and E2B) and pass `--exec-url`. The recipes are in
[Run the tools somewhere else](crates/agent/README.md#run-the-tools-somewhere-else).

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
cargo run -p beyond-gateway --release     # BYO keys need no config; listens on :8080
curl localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}'
```

Managed keys, the catalog and configuration: [`crates/gateway`](crates/gateway/README.md).

## The agent

`agent` reads a task, calls a model, and runs tools (`read`, `write`,
`edit`, `bash`, `grep`, `find`, `ls`, background jobs, MCP servers) until it is
done.

```sh
ANTHROPIC_API_KEY=sk-ant-... cargo run -p beyond-agent -- run "add a CHANGELOG entry"
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
cargo nextest run -p beyond-agent -E 'test(/^serve_approval::/)'   # one test module
```

CI runs on Linux only, the only platform the project supports. Markdown is
formatted by `dprint` (`dprint check`). `mise run audit` checks dependencies
against RustSec advisories.

The agent crates are modeled on [pi](https://github.com/badlogic/pi-mono) (MIT,
Mario Zechner); its license is in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
