# beyond/ai

Open-source AI at Beyond — a Cargo workspace:

- **`crates/gateway`** (`beyond-ai`) — the LLM egress gateway. Route LLM traffic through one internal proxy; apps use their stock OpenAI or Anthropic SDK unchanged — the gateway authenticates, swaps in the real provider key, and meters every token.
- **`crates/agent-core`** (`beyond-ai-agent-core`) — the agent harness core: a runtime/network-agnostic tool-calling loop (modeled on [pi](https://github.com/badlogic/pi-mono)), the OpenAI/Anthropic wire dialects, and the `ModelTransport`/`Tool` seams. Routes all model traffic through the gateway.
- **`crates/agent`** (`beyond-ai-agent`) — the agent CLI: `run` (one-shot coding task) and `serve` (headless control protocol over stdio, for remote control over SSH). Ships the coding tools (read/write/edit/bash/ls/grep/find) plus the Beyond platform tools (fork/sync/logs).

The gateway is the unified model API: the agent speaks OpenAI/Anthropic wire to it with a `bai_v1` key and never holds a provider key. See each crate's `ARCHITECTURE.md`.

## Gateway Quick Start

```sh
cp crates/gateway/config.example.toml config.toml
# Set at minimum: signing_keys and one pool key
AI_POOL_KEY_OPENAI=sk-... cargo run -p beyond-ai --release
```

Point any OpenAI-wire SDK at `http://ai.internal` with a virtual key:

```python
from openai import OpenAI
client = OpenAI(base_url="http://ai.internal/v1", api_key="bai_v1.1.<payload>.<sig>")
```

Or pass your own provider key directly (BYO — forwarded unchanged, no swap):

```python
client = OpenAI(base_url="http://ai.internal/v1", api_key="sk-your-openai-key")
```

## Agent Quick Start

The agent harness routes through the gateway. Point it at a running gateway with a `bai_v1` key:

```sh
# One-shot coding task
AI_GATEWAY_URL=http://ai.internal AI_AGENT_KEY=bai_v1... \
  cargo run -p beyond-ai-agent -- run "add a CHANGELOG entry for the latest release"

# Headless server — drive it over stdio (e.g. an SSH pipe); newline-delimited JSON
AI_GATEWAY_URL=http://ai.internal AI_AGENT_KEY=bai_v1... \
  cargo run -p beyond-ai-agent -- serve --session-file /tmp/agent.json
# then: {"type":"prompt","message":"…"} → streamed event frames, then a response
```

Tools: `read`, `write`, `edit`, `bash`, `ls`, `grep`, `find` (pi's coding set) plus `fork`, `sync`, `logs` (Beyond platform). See [crates/agent-core/ARCHITECTURE.md](crates/agent-core/ARCHITECTURE.md).

## What It Does

- **Managed keys** (`bai_v1…`) — Ed25519-verified, stateless. Swaps to the pool key. Attributes usage to tenant + VPC. Deny-set checked (spend/fraud).
- **BYO keys** — any other token passes through to the provider untouched. No key-swap, no deny-set, no attribution, no `ai.usage` billing event (aggregate throughput metrics still count it).
- **10 providers, zero config** — openai, anthropic, openrouter, fireworks, groq, deepseek, together, cerebras, mistral, xai. Add more in `config.toml` under `[provider_authorities]`.
- **Never buffers** — request and response stream through; a SIMD scanner extracts `model` in O(1) memory. 64KB tail taps usage without holding the body.
- **Token facts, not pricing** — emits `ai.usage` token-count events as structured logs (stdout → logfwd/OTLP → ClickHouse). A closed downstream consumer prices; slipstream carries only the deny-set.
- **Rate guardrail** — per-key request ceiling (`rate_limit_rps`). Circuit breaker against runaway keys. Deny-set owns spend control.
- **Fail-open NATS** — auth works without NATS. A NATS outage stales the deny-set; existing allows stay allowed.

## Providers

The provider is the **first path segment** of the base URL — no header, nothing tool-specific. Bare
`/v1` defaults to OpenAI (and `/v1/messages` to Anthropic), so the two big providers are a host-only
swap; everything else is `/{provider}/…` using that provider's own path (forwarded verbatim).

```python
# OpenAI (default) — change only the host
client = OpenAI(base_url="http://ai.internal/v1", api_key="bai_v1...")

# Groq — its native base path is /openai/v1, so the gateway path is /groq/openai/v1
client = OpenAI(base_url="http://ai.internal/groq/openai/v1", api_key="bai_v1...")

# Fireworks mounts at /inference/v1 → /fireworks/inference/v1; OpenRouter at /api/v1 → /openrouter/api/v1
```

An unknown first segment is a 404. See `route::KNOWN_PROVIDERS` for each provider's native base path.

## Config

All config keys are overridable by `AI_`-prefixed env vars (`AI_NATS_URL`, `AI_POOL_KEY_OPENAI`, …). See `config.example.toml` for the full reference.

Required to serve managed traffic:

| Key                  | Source        | Purpose                                                 |
| -------------------- | ------------- | ------------------------------------------------------- |
| `signing_keys`       | `config.toml` | Ed25519 public keys by `kid` — verifies `bai_v1` tokens |
| `AI_POOL_KEY_<NAME>` | env (SSM)     | Provider key swapped in for managed requests            |

Optional:

| Key                      | Default   | Purpose                                                                  |
| ------------------------ | --------- | ------------------------------------------------------------------------ |
| `snapshot_path`          | unset     | On-disk deny-set snapshot — set on durable nodes, leave unset on Fargate |
| `rate_limit_rps`         | `100`     | Per-key request ceiling; `0` disables                                    |
| `[provider_authorities]` | built-ins | Override or add upstream hosts                                           |

## Running Tests

```sh
mise run test:unit:rs        # pure-logic unit tests (no network)
mise run test:integration:rs # gateway + mock upstream + NATS
mise run test:smoke          # live providers — needs API keys in env, bills real (tiny) requests
mise run bench               # unit micro-benchmarks + end-to-end throughput
```

## Architecture

[ARCHITECTURE.md](ARCHITECTURE.md) — request flow, module map, key invariants.

## Third-Party Notices

The agent crates are modeled on [pi](https://github.com/badlogic/pi-mono) (MIT, Mario
Zechner). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for its license text.
