# beyond/ai

Route LLM traffic through one internal proxy. Apps use their stock OpenAI or Anthropic SDK unchanged — the gateway authenticates, swaps in the real provider key, and meters every token.

## Quick Start

```sh
cp config.example.toml config.toml
# Set at minimum: signing_keys and one pool key
AI_POOL_KEY_OPENAI=sk-... cargo run --release
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

## What It Does

- **Managed keys** (`bai_v1…`) — Ed25519-verified, stateless. Swaps to the pool key. Attributes usage to tenant + VPC. Deny-set checked (spend/fraud).
- **BYO keys** — any other token passes through to the provider untouched. No key-swap, no deny-set, no attribution, no `ai.usage` billing event (aggregate throughput metrics still count it).
- **10 providers, zero config** — openai, anthropic, openrouter, fireworks, groq, deepseek, together, cerebras, mistral, xai. Add more in `config.toml` under `[provider_authorities]`.
- **Never buffers** — request and response stream through; a SIMD scanner extracts `model` in O(1) memory. 64KB tail taps usage without holding the body.
- **Token facts, not pricing** — emits `ai.usage` token-count events as structured logs (stdout → logfwd/OTLP → ClickHouse). A closed downstream consumer prices; slipstream carries only the deny-set.
- **Rate guardrail** — per-key request ceiling (`rate_limit_rps`). Circuit breaker against runaway keys. Deny-set owns spend control.
- **Fail-open NATS** — auth works without NATS. A NATS outage stales the deny-set; existing allows stay allowed.

## Providers

Select a non-default provider with `x-beyond-provider: <name>`:

```python
client = OpenAI(
    base_url="http://ai.internal/v1",
    api_key="bai_v1...",
    default_headers={"x-beyond-provider": "groq"},
)
```

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
