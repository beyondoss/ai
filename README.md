# beyond/ai

Open-source AI at Beyond — a Cargo workspace:

- **`crates/gateway`** (`beyond-ai`) — the LLM egress gateway. Route LLM traffic through one internal proxy; apps use their stock OpenAI or Anthropic SDK unchanged — the gateway authenticates, swaps in the real provider key, and meters every token.
- **`crates/agent-core`** (`beyond-ai-agent-core`) — the agent harness core: a runtime/network-agnostic tool-calling loop (modeled on [pi](https://github.com/badlogic/pi-mono)), the OpenAI/Anthropic wire dialects, and the `ModelTransport`/`Tool` seams. Routes all model traffic through the gateway.
- **`crates/agent`** (`beyond-ai-agent`) — the agent CLI: `run` (one-shot coding task) and `serve` (headless control protocol over stdio, for remote control over SSH). Ships the coding tools (read/write/edit/bash/ls/grep/find) plus `todo` and `web`. Vendor-neutral: `--exec-url` points the whole toolset at any remote exec endpoint.

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

**The gateway is optional.** Export a provider key and the agent goes straight to that provider — no
gateway, no extra config. The auth header (`x-api-key` for Anthropic, `Authorization: Bearer` for
everyone else) comes from the provider table; you never spell it out.

```sh
# One-shot coding task, straight to Anthropic
ANTHROPIC_API_KEY=sk-ant-... \
  cargo run -p beyond-ai-agent -- run --model claude-opus-4-8 "add a CHANGELOG entry"

# …or OpenAI, or any of the providers in the table (OPENROUTER_API_KEY, GROQ_API_KEY, …)
OPENAI_API_KEY=sk-... cargo run -p beyond-ai-agent -- run --model gpt-5 "…"

# A subscription login works too, with no gateway and no key at all
cargo run -p beyond-ai-agent -- login anthropic
```

Aggregators serve other vendors' model ids, so they can't be inferred from one — name them:

```sh
AI_PROVIDER=openrouter OPENROUTER_API_KEY=sk-or-... \
  cargo run -p beyond-ai-agent -- run --model anthropic/claude-sonnet-4.5 "…"

# Any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, a corporate proxy)
AI_BASE_URL=http://localhost:8000/v1 AI_API_KEY=local \
  cargo run -p beyond-ai-agent -- run --model qwen3-coder "…"
```

Point it at a gateway instead when you want key-swapping, routing, and metering — that path is
unchanged, and a configured gateway is never bypassed just because a key is in the environment
(`AI_DIRECT=1` overrides that, if you want it):

```sh
AI_GATEWAY_URL=http://ai.internal AI_AGENT_KEY=bai_v1... \
  cargo run -p beyond-ai-agent -- run "add a CHANGELOG entry for the latest release"

# Headless server — drive it over stdio (e.g. an SSH pipe); newline-delimited JSON
AI_GATEWAY_URL=http://ai.internal AI_AGENT_KEY=bai_v1... \
  cargo run -p beyond-ai-agent -- serve --session-file /tmp/agent.json
# then: {"type":"prompt","message":"…"} → streamed event frames, then a response
```

**Credential precedence.** A gateway is _configured_ if `AI_GATEWAY_URL`, a stored `default_gateway_url`,
or `AI_AGENT_KEY` is set — the `http://ai.internal` default is a fallback, not configuration. With one
configured, the gateway is used (unless `AI_DIRECT=1`). Otherwise, routing is direct, in this order:

1. a `models.json` `base_url` override for the exact model id
2. `AI_BASE_URL` + `AI_API_KEY`
3. `AI_PROVIDER` + that provider's key
4. a stored OAuth login (`agent login`) — **above** an ambient key, so a stray `ANTHROPIC_API_KEY` in
   your shell never silently moves a subscription onto pay-per-token billing
5. the provider key for whichever provider natively serves the model id

Tools: `read`, `write`, `edit`, `bash`, `ls`, `grep`, `find` (pi's coding set), plus `todo` and `web`. See [crates/agent-core/ARCHITECTURE.md](crates/agent-core/ARCHITECTURE.md).

### Running the tools against a remote exec endpoint

`--exec-url <URL>` makes the agent's tools — `read`, `write`, `edit`, `ls`, `grep`, `find` **and
`bash`** — run wherever that URL points instead of on the host. The model sees an identical toolset:
same names, descriptions and JSON schemas, so no prompt changes and no prompt-cache miss.

The agent knows one protocol and nothing else. It does not know what is behind the URL, and it does
not manage sandboxes — creating, warming and destroying them is the caller's job.

```sh
AI_GATEWAY_URL=… AI_AGENT_KEY=… \
  cargo run -p beyond-ai-agent -- run \
    --exec-url https://exec.internal/run \
    --exec-header 'Authorization: Bearer $TOKEN' \
    "find the TODOs under /srv/app and fix the one in main.rs"
```

The protocol is one POST, deliberately the smallest thing that carries a command and its result:

```jsonc
// request
{ "command": "rg", "args": ["--files"], "cwd": "/work", "timeout_ms": 120000 }
// response — 200
{ "exit_code": 0, "stdout": "…", "stderr": "…" }
```

Putting that in front of a provider's SDK is a few dozen lines on your side, and that shim is where
vendor specifics belong. `--exec-header` is repeatable and is where auth goes; which scheme the
endpoint wants is the endpoint's business.

#### Multi-tenant `serve`

In `serve`, the endpoint is **per session**, so one server can multiplex tenants with a sandbox each:

```jsonc
{"id":"1","type":"set_exec_endpoint","url":"https://exec.internal/run","headers":["Authorization: Bearer …"]}
{"id":"2","type":"set_exec_endpoint","command":"docker exec tenant-a {}"}   // or a template
{"id":"3","type":"set_exec_endpoint"}                                        // detach, back to the host
```

Three properties this guarantees, each covered by a test:

- **Switching sessions never inherits an endpoint.** The incoming session is detached first, then
  reattached from its own record. Failing open to "runs on the host" is recoverable; failing open to
  "runs in the previous tenant's sandbox" is not.
- **Subagents act on their parent's machine.** A child whose tools ran on the host while its parent
  was sandboxed would let the model escape by delegating.
- **Resuming a session reattaches its endpoint**, so a restart doesn't silently move a tenant's work
  onto the server. A fork does _not_ inherit it — a sandbox belongs to the session it was made for.

`--exec-url`/`--exec-cmd` also work on `serve` as a process-wide default for the single-tenant case;
a per-session `set_exec_endpoint` always wins.

For targets with no HTTP surface, `--exec-cmd` takes an argv template whose `{}` is replaced by the
command:

```sh
--exec-cmd 'ssh build-host -- {}'
--exec-cmd 'docker exec my-container {}'
--exec-cmd 'kubectl exec pod/app -- {}'
```

`{}` expands to **separate argv entries**, never into a shell string, so a path containing spaces,
quotes or `;` stays one argument on the far side and cannot be reparsed as syntax.

**Install `ripgrep` on the target if you can.** With `rg` present, results are byte-identical to
running locally. Without it the fallback is POSIX `grep`, which diverges in two measured ways: it
does not honor `.gitignore` (so it walks `target/` and `node_modules/`), and `grep -I` treats any
file containing non-UTF-8 bytes as binary and **silently skips it**. Both are asserted by name in
`crates/agent/tests/fs_backend_parity.rs`.

One thing to know: **paths are the target's.** Point the endpoint at a machine whose working tree is
at the same absolute path as the host's, or paths the agent reads from skills and the system prompt
won't resolve.

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
