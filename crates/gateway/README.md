# beyond-gateway

An egress proxy for LLM traffic. Apps keep their stock OpenAI or Anthropic SDK
and change only the base URL. The gateway authenticates the caller, swaps in a
provider key it holds, relays the request and streamed response, and writes one
priced billing row (`ai.usage`) per request.

It is stateless: no database, no session store. Identity is a signed token it
verifies without a lookup; revocation and quota arrive as small sets from NATS.

**One base URL change, and the hard parts of LLM egress are done.**

|                                                | The fact                                                                                                                                                                                                               | The proof                                                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Stateless, so it scales by adding replicas** | No database and no session store. A `bai_v1` virtual key is an Ed25519-signed token verified in about 28 µs with no lookup, and a forged-key flood is rejected in tens of nanoseconds before that cost is paid.        | [Verification cost](ARCHITECTURE.md#core-mechanism), [Benchmarking](ARCHITECTURE.md#benchmarking)          |
| **Your SDKs, unchanged**                       | OpenAI and Anthropic clients keep working, and a request is translated between Chat Completions, Messages and Responses when the chosen model lives on a different wire.                                               | [Routing](#routing), [`verify/`](../../verify/README.md)                                                   |
| **Name a model, get failover**                 | `/auto` and managed `/v1` walk a catalog: the gateway picks the provider, fails over, and opens a per-provider circuit breaker after 20 failures in 10 s.                                                              | [Routing](#routing), [Model routing](ARCHITECTURE.md#model-routing-auto-managed-v1-providerscatalog)       |
| **Every request priced at cost**               | One `ai.usage` row per request, priced pass-through at the serving host's card from rates generated out of the vendors' own pricing sources and re-checked daily. A stream cut short is billed an estimate, flagged.   | [What it records](#what-it-records-and-enforces), [`rates-drift`](../../.github/workflows/rates-drift.yml) |
| **Keys never reach the app**                   | Apps hold a virtual key; the provider key stays in the gateway and is swapped in per request. Revocation and quota arrive as small sets over NATS.                                                                     | [Two kinds of caller](#two-kinds-of-caller)                                                                |
| **Checked against real clients**               | Every claim the gateway makes is a registered claim with tests that carry its id, run against real SDKs and providers; every defect found is in a ledger and settled. Rerun it yourself with `mise run verify:status`. | [`verify/`](../../verify/README.md), [Verification](ARCHITECTURE.md#verification)                          |

## Two kinds of caller

| Key                                          | What the gateway does                                                                                                  |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Managed**: `bai_v1…`, `bai_v2…`, `bai_v3…` | Verifies the Ed25519 signature, checks the deny-set and allowance-set, swaps in a pool key, bills a row to the tenant. |
| **BYO**: any other token                     | Forwards the request unchanged to the provider the key belongs to. No swap, no deny-set, no billing row.               |

Managed keys reach only `POST` generation endpoints: `chat/completions`,
`messages`, `responses`, `embeddings`, and the `count_tokens` and `compact`
sub-resources. Files, batches, stored responses and WebSocket upgrades are
refused. A managed key also gets the model catalog: it names a model and the
gateway picks the provider, fails over, and translates between wires.

## Run it

**BYO only: no config, no NATS.** A binary and an upstream key are enough:

```sh
cargo run -p beyond-gateway --release          # listens on :8080 (data) and :9090 (admin)
curl localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gpt-5-mini","messages":[{"role":"user","content":"hi"}]}'
```

An image is published as `ghcr.io/beyondoss/ai` (`latest` and by commit SHA,
`linux/amd64` and `linux/arm64`); its entrypoint is `beyond-ai`. Configuration
comes from `./config.toml` (`-c`/`AI_CONFIG_PATH`) and `AI_`-prefixed
environment variables.

**Managed keys.** Managed traffic needs a signing key, a pool key, an id-signing
secret and NATS (`nats_url`, default `nats://localhost:4222`). With a dev
keypair:

```sh
cargo run -p beyond-gateway --example mint_dev_key   # prints SIGNING_PUBKEY_B64, KID, DEV_BAI_V1 (dev only)
```

```toml
# config.toml
[signing_keys]
1 = "<SIGNING_PUBKEY_B64>"
```

```sh
AI_POOL_KEY_OPENAI=sk-... AI_ID_SIGNING_KEY_1=$(openssl rand -base64 32) \
  cargo run -p beyond-gateway --release
```

Then use `DEV_BAI_V1` as the API key against `http://localhost:8080/v1`. Until
the allowance-set has been read from NATS (or a `snapshot_path` snapshot),
managed requests get `503 allowance unavailable` and `/readyz` reports
`not_ready`; BYO traffic is unaffected. `beyond-ai doctor` checks NATS, signing
keys, pool keys and provider DNS, and exits non-zero on a failure.

Point an SDK at the gateway:

```python
from openai import OpenAI
client = OpenAI(base_url="http://ai.internal/v1", api_key="bai_v1...")  # or your own provider key
```

## Routing

| Path             | Goes to                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `/v1/…`          | BYO: OpenAI, or Anthropic for `/v1/messages` and `sk-ant-` keys. Managed: the model catalog.                  |
| `/auto/v1/…`     | Managed only: the model catalog.                                                                              |
| `/{provider}/…`  | That provider, with the rest of the path forwarded verbatim. Unknown provider: 404.                           |
| `GET /v1/models` | Managed: the catalog rows this deployment has a pool key for, with list prices. BYO: the provider's own list. |

The provider is the first path segment, never a header:

```python
OpenAI(base_url="http://ai.internal/groq/openai/v1", api_key="bai_v1...")
```

Twelve providers are built in: `openai`, `anthropic`, `openrouter`,
`fireworks`, `groq`, `deepseek`, `together`, `cerebras`, `mistral`, `xai`,
`bedrock` and `openai-codex` (BYO only). Their hosts, wires and auth headers
are rows in [`providers`](../providers/README.md#providers). `[provider_authorities]`,
`[provider_dialects]` and `[provider_auth_schemes]` in `config.toml` add more,
for example Azure OpenAI.

### The catalog

A managed `/v1` or `/auto` request names a model in the `x-beyond-model` header
or the body's `model`. The gateway looks it up in the catalog
([`providers`](../providers/README.md#the-catalog)), tries the row's providers
in order, rewrites `model` per attempt, and translates between Chat
Completions, Messages and Responses when the inbound endpoint and the serving
candidate's differ. A name not in the catalog is a 404. The default order keeps
one caller on one provider, so provider prompt caches stay warm.

Per-request control, managed only, stripped before the upstream and never a 4xx:

| Header              | Effect                                                                  |
| ------------------- | ----------------------------------------------------------------------- |
| `x-beyond-order`    | Try these providers first, then the rest of the row.                    |
| `x-beyond-only`     | Drop every provider not named.                                          |
| `x-beyond-split`    | `name=weight` pairs: a weighted primary.                                |
| `x-beyond-metadata` | A flat JSON object (at most 16 keys, 1 KB) tagged onto the billing row. |
| `x-beyond-capture`  | `on` or `off`: log this request's bodies, or not.                       |
| `x-beyond-cache`    | `off`: skip the response cache.                                         |

Every response carries `x-beyond-request-id` (the billing row's `request_id`),
`x-beyond-provider`, and on catalog walks `x-beyond-upstream-model`.

## What it records and enforces

- **Billing.** Each managed request writes an `ai.usage` row with token counts
  and a price from `providers::pricing`: pass-through at the serving host's
  card, so price equals cost. Rows go to stdout, or to hourly
  `ai-usage-YYYYMMDDTHH.jsonl` files when `usage_dir` is set. A stream cut
  short is billed an estimate flagged `usage_estimated`.
- **Streaming.** SSE is relayed event by event and never held whole; a 64 KB
  tail is read for usage. Request bodies are buffered only where routing needs
  them (catalog walks, bodies over 64 KB).
- **Deny-set and allowance-set.** NATS carries which tenants and keys are
  denied (spend, fraud) or out of quota. A denied caller gets 402 or 403. The
  deny-set fails open during a NATS outage; the allowance-set fails closed
  until first read.
- **Guardrails.** `rate_limit_rps` per credential (default 100), a shared
  `byo_rate_limit_rps` for BYO (default 1000), optional `tenant_max_in_flight`,
  and a per-provider circuit breaker (opens on 20 failures in 10 s, 5xx or
  connect errors, probes after 30 s).
- **Payload capture.** Off for every tenant until the control plane writes an
  `aicapture.{tenant}` entry, or a caller sends `x-beyond-capture: on`.
- **Response cache.** Off by default. `cache_ttl_secs > 0` replays identical
  managed catalog requests from a per-process table.

## Operations

| Port | Serves                                                           |
| ---- | ---------------------------------------------------------------- |
| 8080 | Client traffic (`listen`); `tls_listen` adds a TLS listener.     |
| 9090 | `/metrics` (Prometheus), `/livez`, `/readyz` (`metrics_listen`). |

On SIGTERM the gateway stops accepting, lets in-flight requests finish for up
to `shutdown_grace_period_secs` (default 600), and exits as soon as none remain.

**12 providers, zero config**: openai, anthropic, openrouter, fireworks, groq, deepseek, together, cerebras, mistral, xai, bedrock and openai-codex (BYO only). Add more in `config.toml` under `[provider_authorities]`.

`config.example.toml` documents every key and its default. Scalar keys take an
`AI_` prefix, and pool and signing keys come from `AI_POOL_KEY_<NAME>` and
`AI_SIGNING_KEY_<KID>`. Map keys (`provider_authorities.*`, `provider_dialects.*`,
`provider_auth_schemes.*`) are config-file only.

Required to serve managed traffic:

| Key                  | Source        | Purpose                                                |
| -------------------- | ------------- | ------------------------------------------------------ |
| `signing_keys`       | `config.toml` | Ed25519 public keys by `kid`, verifies `bai_v1` tokens |
| `AI_POOL_KEY_<NAME>` | env (SSM)     | Provider key swapped in for managed requests           |

Optional:

| Key                      | Default   | Purpose                                                                 |
| ------------------------ | --------- | ----------------------------------------------------------------------- |
| `snapshot_path`          | unset     | On-disk deny-set snapshot; set on durable nodes, leave unset on Fargate |
| `rate_limit_rps`         | `100`     | Per-key request ceiling; `0` disables                                   |
| `[provider_authorities]` | built-ins | Override or add upstream hosts                                          |

### Vercel AI SDK

`createOpenAI({ baseURL, apiKey })` works. Its default model uses the Responses
API and sends earlier assistant turns as `item_reference` ids, which only
OpenAI stores. On a managed key and a non-OpenAI model, a continued
conversation is a 400 naming `item_reference`; tool loops are unaffected. Send
history in full with `providerOptions: { openai: { store: false } }`, or use
`@ai-sdk/anthropic` or `@ai-sdk/openai-compatible`
([Stored-response references](ARCHITECTURE.md#stored-response-references-item_reference)).

## Reference

- [`ARCHITECTURE.md`](ARCHITECTURE.md): request flow, key and billing
  contracts, failure modes, metrics.
- [`config.example.toml`](config.example.toml): every configuration key.
- [`verify/README.md`](../../verify/README.md): the claims the gateway makes
  and the tests that prove each.
- `mise run bench`: unit micro-benchmarks and end-to-end throughput.
