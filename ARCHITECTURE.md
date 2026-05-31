# Beyond AI Gateway — Architecture

Takes HTTP requests carrying an OpenAI- or Anthropic-dialect payload, authenticates the caller via
Ed25519 virtual key or BYO provider token, swaps in a pool key for managed traffic, relays the
request and response byte-for-byte to the upstream provider, and emits a token-usage billing fact
(`ai.usage`) on completion — all without buffering the body or response stream.

**Self-contained:** no `path` deps into the `beyond` repo. Depends only on crates.io + the
published `beyond-slipstream` — clones, CI-builds, and publishes anywhere.

---

## Concepts & Terminology

| Term                                             | What It Controls / Gates                                                                                                                                    | NOT                                                                          |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Managed key** (`bai_v1.…`)                     | Ed25519-verified identity; enables key swap, deny-set check, and `ai.usage` billing                                                                         | A session token or capability grant — just tenant attribution                |
| **BYO key** (anything else)                      | Forwarded as-is to the provider; no swap, no billing, no deny-set                                                                                           | A lesser tier — same proxy, minus attribution and billing                    |
| **Pool key**                                     | Real provider API key held by the gateway; swapped in for managed traffic                                                                                   | Per-tenant — one key per provider, shared by all managed callers             |
| **Tenant**                                       | The billing entity from the virtual key payload (`tenant_id: u64`)                                                                                          | An org, user, or namespace — an opaque integer the gateway doesn't interpret |
| **Dialect**                                      | A provider attribute (OpenAI-wire vs Anthropic-wire) driving usage parsing; for a bare-path request it's derived from the path to pick the default provider | The provider — a prefixed request uses its provider's dialect, not the path  |
| **Provider**                                     | The request's **first path segment** (`/{provider}/…`); a named row in the routing table: authority, dialect, auth scheme                                   | A vendor relationship — just connection facts and auth wiring                |
| **Deny-set**                                     | Sparse map of denied `tenant_id`s → reason; gates managed traffic; default-allow                                                                            | An allowlist or ACL — misses are allowed, not blocked                        |
| **Tail tap**                                     | Bounded 64KB window kept from the end of the response for usage extraction                                                                                  | A buffer or copy — the response is relayed unbuffered; only the tail is kept |
| **Snapshot**                                     | On-disk deny-set cache (entries + NATS cursor) for edge/tunnel deployments                                                                                  | Persistent store — a pure cache; delete it and the gateway re-scans NATS     |
| **Virtual key** (`bai_v1.{kid}.{payload}.{sig}`) | Ed25519-signed token encoding `tenant_id` + `vpc_id` (16-byte fixed payload)                                                                                | A session or auth token — stateless, no server-side lookup, no revocation    |

---

## Data Flow

### Happy Path

```
Client (stock OpenAI/Anthropic SDK)
  │
  ▼  request_filter (proxy.rs)
  │  ├─ Route: first segment → provider row (authority, dialect, auth scheme)
  │  ├─ Extract key from Authorization: Bearer or x-api-key
  │  ├─ Rate guardrails (BEFORE verify — keeps forged-key floods at ns cost)
  │  │    per-credential count-min  ──────────────────────────────► 429
  │  │    global BYO aggregate (managed exempt)  ─────────────────► 429
  │  ├─ Content-Length abuse guard  ──────────────────────────────► 413
  │  └─ Identity branch:
  │       bai_v1.…  → Ed25519 verify → deny-set check (O(1))
  │       │               │                    │
  │       │             401 (bad sig)     402 Spend / 403 Fraud
  │       │                                    │
  │       │           pool key required ───────────────────────── 503
  │       └─ BYO: pass through (no verify, no deny-set, no billing)
  │
  ▼  upstream_peer (proxy.rs)
  │  TTL-cached DNS resolve (60s) → HttpPeer (TLS, H2 pref, timeouts)
  │  DNS fail ──────────────────────────────────────────────────── 502
  │  TCP connect fail (retry 2×) ──────────────────────────────── 502
  │
  ▼  upstream_request_filter (proxy.rs)
  │  Managed: remove both auth headers → inject pool key
  │  BYO: leave auth header unchanged
  │  Set Host; forward path verbatim (/{provider} prefix stripped)
  │
  ▼  request_body_filter (proxy.rs)  — body streamed through, never buffered
  │  Feed chunks → ModelScanner (peek.rs) — extract root-level `model`, O(1) mem
  │  Enforce running size cap (chunked-safe) ──────────────────── 413
  │  Injection-eligible (managed OpenAI chat/responses + stream):
  │    buffer full body → inject stream_options.include_usage → re-frame chunked
  │
  ▼  Provider upstream  (OpenAI / Anthropic / Groq / DeepSeek / …)
  │
  ▼  response_filter (proxy.rs)
  │  Record TTFT; detect streaming (Content-Type: text/event-stream)
  │  Count upstream response by provider + status class
  │  Set x-beyond-request-id header
  │
  ▼  response_body_filter (proxy.rs)  — response relayed chunk-by-chunk, never buffered
  │  Feed chunks → ModelScanner over response head → extract billed model
  │  Append to bounded 64KB tail (compact drain(..half) if tail > 128KB)
  │
  ▼  logging (proxy.rs)
     Parse usage from tail (by dialect + streaming flag)
     Emit ai.usage fact: tenant, vpc, model, requested_model, token counts (managed only)
     Decrement requests_in_flight gauge
```

### Background: Deny-Set Watcher

```
NATS (blackhole.* KV entries)
  │
  ▼  store_watch.rs (Pingora BackgroundService)
  │  On connect: seed from disk snapshot (if snapshot_path set) or full NATS scan
  │  Resume watch from saved revision (gap-free — no entry lost mid-connect)
  │  Reconnect backoff: 1s → 30s exponential
  │
  ▼  ArcSwap<DenySet>  (state.rs)
     Lock-free read on every managed request
     Written only by the watcher on entry add/remove
```

---

## Core Mechanism

### Routing (`route.rs`)

Providers are **data rows**, not code paths. `KNOWN_PROVIDERS` in `route.rs` lists 10 built-in
providers (openai, anthropic, openrouter, fireworks, groq, deepseek, together, cerebras, mistral,
xai); each row carries its authority (host:port), dialect (OpenAI-wire vs Anthropic-wire), and auth
scheme (Bearer vs x-api-key). The `provider_authorities` config key adds or overrides rows at boot
with zero code change.

The routing rule: **first path segment = provider name**. `/groq/openai/v1/chat/completions` routes
to Groq and forwards `/openai/v1/chat/completions` verbatim. A bare `/v1/…` path matches the
dialect default (OpenAI or Anthropic based on which default is set). Unknown segment → 404. Model
is not known at peer-selection time and is never used for routing.

### Identity (`key.rs`)

Virtual key format: `bai_v1.{kid}.{payload}.{sig}` where payload is exactly 16 bytes (8-byte
`tenant_id` + 8-byte `vpc_id`, little-endian u64). Verification is **stateless Ed25519** — no
database, no network call. The keyring holds multiple `kid` → public key mappings simultaneously
(zero-downtime rotation: add the new kid, deploy, remove the old kid). A tampered or forged key
falls through to BYO treatment; it does not error in a way that reveals which part failed.

Verification cost ≈ 28µs per request — this is the gateway's only meaningful per-request CPU cost
(everything else runs in nanoseconds; see Benchmarking). The rate guardrails sit **before** verify
precisely because of this: a forged-key flood is rejected in tens of nanoseconds, not 28µs each.

### Model Extraction (`peek.rs:ModelScanner`)

A streaming structural scanner fed body or response chunks as they arrive. Tracks JSON nesting
depth, string-escape state, and quote boundaries. Captures the **root-level `model` field only**
(depth 0 in the object), ignoring nested `model` keys in tool calls or message content.
SIMD-accelerated via `memchr2` to skip over large string values (base64-encoded images, long
prompts). O(1) memory: one struct, no heap growth with payload size — proven by the unit bench
which shows a single allocation independent of whether the body is 0 bytes, 4 KB, or 256 KB.

The billing fact carries **two model fields**:

- `requested_model` — what the client sent (extracted from the request body)
- `model` — what the provider resolved and billed (extracted from the response head; falls back to
  `requested_model` when the response carries no model field, e.g. an error body)

`model` is what reconciles against the provider's invoice (which itemizes by pinned snapshot, e.g.
`gpt-4o-2024-08-06`, not alias). `requested_model` serves product analytics and as a fallback rate
when the snapshot is newer than the downstream price table.

### Usage Extraction (`usage.rs`)

The tail tap feeds the parser after `logging` fires. Two dialects:

| Dialect   | Format     | Fields                                                                                                            |
| --------- | ---------- | ----------------------------------------------------------------------------------------------------------------- |
| OpenAI    | JSON body  | `usage.prompt_tokens`, `usage.completion_tokens`, `usage.prompt_tokens_details.cached_tokens`                     |
| OpenAI    | SSE stream | Terminal `data:` line (before `[DONE]`), same fields                                                              |
| Anthropic | JSON body  | `usage.input_tokens`, `usage.output_tokens`, `usage.cache_read_input_tokens`, `usage.cache_creation_input_tokens` |
| Anthropic | SSE stream | `message_delta` event with `usage` block                                                                          |

Missing or zero usage fields deserialize to zero (safe default). If the tail is truncated by the
compaction drain, the usage chunk is still present because SSE usage is always the final `data:`
line and the tail keeps the last 64KB.

### Deny-Set (`deny.rs`)

A `HashMap<u64, DenyReason>` (tenant_id → reason). Only denied tenants are stored — the map is
`O(denied)` in memory regardless of total tenant count. Lookup is one hash probe. Written
exclusively by the NATS watcher via `ArcSwap`; reads on the hot path are lock-free.

Reasons: `Spend` (→ 402), `Fraud` (→ 403), `Unknown` (→ 403, fail-safe for unrecognized values).
Restore = explicit delete from NATS KV or TTL expiry — no gateway-side timer.

### Rate Guardrails (`ratelimit.rs`)

Two fixed-memory count-min sketch tiers, checked before Ed25519 verify and before any upstream
connection:

| Tier                 | Key             | Bucket count | Default ceiling | Managed exempt? |
| -------------------- | --------------- | ------------ | --------------- | --------------- |
| Per-credential       | Hash of raw key | 5 MB sketch  | 100 req/s       | No              |
| Global BYO aggregate | Single bucket   | 1 bucket     | 1000 req/s      | **Yes**         |

The per-credential tier is keyed on the **raw presented credential** (not the verified tenant),
which has two consequences: (1) the guard sits ahead of verify, so forged tokens are rejected
before any crypto work; (2) virtual keys are deterministic per `(tenant, app)`, so this is
effectively per-(tenant, app) granularity without a registry lookup.

The global BYO aggregate exists because BYO traffic exits from the gateway's own egress IPs
carrying the caller's raw token. A flood of distinct junk BYO tokens each get their own
per-credential bucket and slip through that tier — the aggregate caps total BYO egress rate to
protect the gateway's IP reputation with providers. Managed traffic is exempt because it's verified
before any upstream connection and cannot be forged.

Both tiers are generous circuit breakers, not quotas. `rate_limit_rps = 0` / `byo_rate_limit_rps =
0` disable them independently.

---

## Why It Behaves This Way

### Why rate guardrails sit before Ed25519 verify

Ed25519 verify is ~28µs — roughly 350–650× more expensive than every other per-request operation.
A flood of forged `bai_v1` tokens could drive unbounded crypto work if the rate limit came after
verify. By checking the per-credential bucket first (keyed on the raw token, no crypto), a
forged-key flood is rejected in tens of nanoseconds per request. Legit traffic is unaffected: the
rate guard passes through, then verify runs as normal. The unit bench (`benches/unit.rs`) asserts
this: `key/verify` ≈ 28µs; `ratelimit::check` ≈ 43–83ns; 0 allocations for either.

### Why the body injection exception exists (`managed + OpenAI + streaming`)

OpenAI streams no usage chunk unless `stream_options.include_usage: true` is set. Without it, a
streaming managed request is unmeterable: no usage block in the response means no billing fact. The
gateway injects this field server-side so callers using stock SDKs get metered without any
cooperation. The request is buffered (`MAX_REQUEST_BODY` cap), the field injected, and the body
re-framed as chunked upstream. Scoped to managed + OpenAI-dialect + streaming only — BYO and
non-streaming requests remain pure passthrough.

### Why the deny-set watch resumes from a saved revision

A plain `watch_prefix` (NATS `DeliverPolicy::New`) would miss any entry written in the window
between the initial seed scan and the live watch attaching. `store_watch.rs` records the stream
revision at which the seed was complete and calls `watch_prefix_from` to resume from that revision
— so a deny written during the gap is delivered, not silently dropped. This revision is also
persisted across reconnects, so a NATS blip resumes from the last-seen point instead of re-scanning
the entire keyspace.

### Why BYO token validity is never checked

Checking a BYO token requires a round-trip to the provider. The provider does that check anyway and
returns 401 if the token is invalid — the client sees the same rejection it would get going direct,
just routed through the gateway. Adding a gateway-side preflight check would double the latency for
every BYO request on the error path with no security benefit at the gateway layer.

### Why pricing is absent from the gateway

The gateway emits token _facts_ (`ai.usage`): counts and model identifiers. Applying prices to
those facts is a downstream concern. Provider pricing changes frequently, varies by contract tier,
and is sometimes retroactively corrected on invoices. A downstream consumer can reprice historical
facts; the gateway's facts cannot be regenerated once the request is gone.

### Why routing uses the first path segment, not a header

Path-based routing makes the target provider explicit in every request URL — visible in logs,
traces, and curl output without inspecting headers. It also survives transparent proxies and load
balancers that strip custom headers. A `/{provider}/` prefix was preferred over a separate header
because SDKs already let callers set the base URL; swapping in the gateway's URL with a provider
prefix requires no SDK modification.

---

## Trust Boundaries

**What the gateway verifies (rejects if invalid):**

- Virtual key signature (Ed25519, stateless — no DB lookup)
- Virtual key format (`bai_v1.{kid}.{payload}.{sig}`, fixed 16-byte payload)
- Tenant not in deny-set (managed traffic only; O(1) HashMap lookup)
- Pool key configured for the requested provider (managed traffic only — else 503)
- Request body size ≤ `MAX_REQUEST_BODY` (declared `Content-Length` + streaming running total)
- Per-credential request rate within ceiling; aggregate BYO rate within ceiling

**What passes through unchecked:**

- Request body content and schema — no validation at the gateway layer
- Model name in the request — extracted for billing facts, never validated against an allowlist
- Provider response content — relayed byte-for-byte
- BYO token validity — forwarded as-is; the provider rejects it if invalid
- `vpc_id` in the virtual key — decoded and emitted in billing facts, not used for access control

**Why these boundaries are where they are:**

- Body schema validation belongs to the provider — duplicate validation adds latency without a
  security benefit at the gateway layer
- Model allowlisting would require a per-provider list coupled to model release cadence
- BYO token validation requires a provider round-trip — the provider does it anyway

---

## Configuration

All fields configurable via `config.example.toml` and environment (`AI_` prefix, flat merge).
Secret-bearing fields (`pool_keys`, `nats_creds`) are held as `Secret<T>` — stray `Debug` or
`Serialize` output redacts to `"***"` and the value is zeroized on drop (`secret.rs`).

| Field                         | Default                           | Runtime Effect                                                                                                                                                                                         |
| ----------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `signing_keys`                | _(required)_                      | Map of kid → base64 Ed25519 public key. Multiple kids enable rotation. Missing → all traffic falls through to BYO treatment.                                                                           |
| `require_signing_keys`        | `false`                           | When `true`, an empty `signing_keys` is a hard boot failure instead of silent BYO-only mode. Set on managed deployments so a typo'd/absent SSM param fails fast rather than silently serving for free. |
| `pool_keys.<name>`            | _(from `AI_POOL_KEY_<NAME>` env)_ | Real provider API key. Missing for a provider → managed requests to that provider return 503 before any upstream connection.                                                                           |
| `provider_authorities.<name>` | _(none)_                          | Override or add a provider's `authority` (host:port). Enables config-added providers beyond `KNOWN_PROVIDERS` with zero code change.                                                                   |
| `snapshot_path`               | _(unset)_                         | Path for the on-disk deny-set cache. Unset → re-scan NATS on every cold boot. Set → load from disk and enforce before NATS reconnects (edge/tunnel deployments).                                       |
| `rate_limit_rps`              | `100`                             | Per-credential request ceiling (count-min, keyed on raw key hash). `0` disables. Exceeded → 429. Checked before Ed25519 verify.                                                                        |
| `byo_rate_limit_rps`          | `1000`                            | Aggregate ceiling for all BYO traffic (single shared bucket). `0` disables. Managed traffic exempt. Exceeded → 429.                                                                                    |
| `connect_timeout_secs`        | `10`                              | TCP connect timeout to the upstream provider. Exceeded → retry up to 2×, then 502.                                                                                                                     |
| `read_timeout_secs`           | `600`                             | Response read timeout (10 min accommodates long-running LLM streams).                                                                                                                                  |
| `write_timeout_secs`          | `60`                              | Upstream request-write timeout (sending the request to the provider).                                                                                                                                  |
| `idle_timeout_secs`           | `90`                              | Idle timeout on a pooled upstream connection before it's closed.                                                                                                                                       |
| `nats_url`                    | `nats://localhost:4222`           | NATS server for the deny-set watcher. Unreachable → fail-open (deny-set stays empty or stale).                                                                                                         |
| `nats_creds`                  | _(unset)_                         | NATS credentials file path. Required for authenticated clusters.                                                                                                                                       |
| `listen_addr`                 | `0.0.0.0:8080`                    | Proxy listener address (client traffic).                                                                                                                                                               |
| `metrics_listen`              | `0.0.0.0:9090`                    | Internal admin/observability listener: `/metrics` (Prometheus scrape), `/livez`, `/readyz`. Separate from the client listener — not externally reachable.                                              |

---

## Failure Modes

| Failure                                     | What Actually Happens                                                                                                          | Recovery                                                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| NATS unreachable at boot                    | Deny-set starts empty (fail-open). Auth still works — keys from config.                                                        | Watcher reconnects; seeds from NATS or disk snapshot on connect.                                                |
| NATS disconnects mid-run                    | Last-known deny-set stays active. New deny entries not applied until reconnect.                                                | Watcher reconnects (1s→30s exponential backoff, reset on success) and resumes from saved revision — no re-scan. |
| NATS history compacted past snapshot cursor | `CursorExpired` → full re-scan from current NATS state.                                                                        | After re-scan, new cursor set; delta watch resumes normally.                                                    |
| Virtual key tampered or forged              | Ed25519 verify fails → falls through to BYO treatment. No billing event. No error reveals which part failed.                   | Billing miss detectable downstream; no security boundary breach.                                                |
| `signing_keys` absent (typo'd/missing SSM)  | Default: warn + BYO-only (silently drops all managed billing + deny-set). With `require_signing_keys=true`: hard boot failure. | Set `require_signing_keys=true` on managed deployments so the mis-deploy fails fast and visibly at boot.        |
| Pool key missing for provider               | Managed request returns 503 before any upstream connection.                                                                    | Add `AI_POOL_KEY_<NAME>` env and redeploy.                                                                      |
| Provider DNS fails                          | `upstream_peer` returns error → 502 to client.                                                                                 | TTL-cached DNS (60s) serves stale; poisoned-lock guard re-resolves on next request.                             |
| Provider TCP connect fails                  | `fail_to_connect` retries up to 2×, then returns 502.                                                                          | Client SDK retries with backoff. No HTTP-status retries (Pingora-idiomatic).                                    |
| Response body > 128KB before usage chunk    | Tail compaction fires: `drain(..half)` discards first half, keeps tail. Usage extracted from retained tail.                    | No action — SSE usage is always in the final `data:` line, which always lands in the tail.                      |
| Gateway crash mid-request                   | In-flight request drops; client receives TCP close. No partial state written.                                                  | Client SDK retries. No DB writes in the request path — no cleanup needed.                                       |

---

## Metrics

Prometheus on the default registry, exposed at `/metrics` on `metrics_listen`.

| Metric                        | Type      | Labels               | What It Measures                                                         |
| ----------------------------- | --------- | -------------------- | ------------------------------------------------------------------------ |
| `ai_requests_total`           | Counter   | —                    | Total admitted requests                                                  |
| `ai_rejections_total`         | Counter   | `reason`             | Rejected requests by cause (auth, deny_spend, deny_fraud, rate_limit, …) |
| `ai_upstream_responses_total` | Counter   | `provider`, `status` | Upstream responses by provider and status class                          |
| `ai_tokens_total`             | Counter   | `kind`               | input / output / cache_read / cache_write token counts                   |
| `ai_ttft_seconds`             | Histogram | `provider`           | Time to first token (50ms–30s buckets)                                   |
| `ai_upstream_latency_seconds` | Histogram | `provider`           | Full request latency (100ms–600s buckets)                                |
| `ai_active_streams`           | Gauge     | —                    | Open SSE streams                                                         |
| `ai_requests_in_flight`       | Gauge     | —                    | All in-flight requests (streaming + non-streaming)                       |
| `ai_deny_set_size`            | Gauge     | —                    | Current number of denied tenants                                         |
| `ai_nats_connected`           | Gauge     | —                    | 1 if NATS watcher is connected, 0 otherwise                              |

---

## Modules

| Module        | Role                                                                                        | Tested    |
| ------------- | ------------------------------------------------------------------------------------------- | --------- |
| `proxy`       | `ProxyHttp` impl — request/response pipeline (request_filter through logging)               | e2e ✓     |
| `key`         | `bai_v1` parse + Ed25519 verify + mint; keyring with multi-kid rotation support             | unit ✓    |
| `route`       | Data-driven provider table (name / authority / auth) + dialect default routing              | unit ✓    |
| `peek`        | `ModelScanner` — streaming structural scan for the root-level `model`; O(1) memory          | unit ✓    |
| `usage`       | Token extraction (OpenAI / Anthropic, body + SSE)                                           | unit ✓    |
| `deny`        | Sparse deny-set, default-allow, reason → HTTP status                                        | unit ✓    |
| `ratelimit`   | Two-tier guardrail: per-credential + global BYO (count-min sketches, fixed memory, no GC)   | unit ✓    |
| `state`       | Keyring + resolved provider registry + watched deny-set (ArcSwap) + TTL DNS cache           | unit ✓    |
| `store_watch` | NATS watcher — gap-free deny-set seeding + delta watch as Pingora `BackgroundService`       | e2e ✓     |
| `config`      | Figment config; build keyring; pool keys / authorities by provider name                     | unit ✓    |
| `secret`      | Redacting, zeroize-on-drop `Secret<T>` newtype for pool keys and NATS creds                 | unit ✓    |
| `admin`       | `ServeHttp` on the metrics listener: `/livez`, `/readyz`, `/metrics`                        | e2e ✓     |
| `metrics`     | Prometheus counter/histogram/gauge registration and update helpers                          | compile ✓ |
| `doctor`      | Boot-time diagnostics (`beyond-ai doctor`)                                                  | compile ✓ |
| `main`        | CLI (`run` / `doctor`), rustls init, config load, Pingora server + three services bootstrap | compile ✓ |

---

## Verification

- **Unit (`cargo test --lib`):** key, route, peek, usage, deny, secret, config. `clippy
  --all-targets -D warnings` clean.
- **End-to-end (`tests/e2e.rs`, `mise run test:integration:rs`):** real `beyond-ai` binary + real
  nats-server + mock upstream. Covers managed key-swap + passthrough fidelity + usage metering
  (OpenAI JSON + SSE, **Anthropic `/v1/messages`** with `x-api-key` swap + metering), **BYO
  passthrough** (raw token unchanged), the **virtual key in either inbound header** (`Bearer` or
  `x-api-key`), and deny-set propagation: spend (write `blackhole.{tenant}` → 402, delete → 200)
  and **fraud** (→ 403). Error/edge paths: **missing key → 401**, **oversized `Content-Length` →
  413**, **managed key for an unconfigured provider → 503**, **streaming tail compaction** (>128KB
  before the usage chunk still meters), **deny-set fail-open** (kill NATS → stale set retained,
  auth still works), and **on-disk snapshot survival** (blackhole a tenant, restart with NATS down
  → the hold is still enforced from disk).
- **Live smoke (`tests/smoke.rs`, `mise run test:smoke`):** the real `beyond-ai` binary against the
  **real** provider hosts over TLS, one per provider in `KNOWN_PROVIDERS`. Proves real TLS/SNI,
  the `/v1` → base-path rewrite landing on a live mount (200, not 404), and BYO auth passthrough.
  Every test is `#[ignore]` and skips unless its provider's API key env var is set — CI stays
  hermetic; you only hit providers you have keys for.

---

## Benchmarking

Two harnesses, mirroring the unit/e2e split of the tests. The framing is **Theory of Constraints**:
a proxy's steady-state constraint is upstream I/O, not gateway CPU. The benches **prove the
gateway's added cost is negligible and bounded** — i.e. it never becomes the constraint.

- **Unit micro (`benches/unit.rs`, `mise run bench:unit`) — `divan`.** Times IO-free hot paths and
  measures allocations natively (divan's `AllocProfiler` reports alloc/dealloc/grow count + bytes
  beside ns/iter, no `unsafe` needed). Coverage: `key` verify/mint; `peek::ModelScanner` over
  0/4KB/256KB bodies with `model` placed last (worst case); `usage` parsers; `route`; `deny`
  (`parse_key`/`parse_reason` off-path + `reason()` on-path); `ratelimit::check` (managed tier
  only vs. BYO which runs both tiers).

  What the alloc numbers assert:
  | Operation           | Cost     | Allocations                  | Claim verified                |
  | ------------------- | -------- | ---------------------------- | ----------------------------- |
  | `key/verify`        | ~28µs    | 0                            | Stack-only Ed25519 decode     |
  | `peek/ModelScanner` | varies   | 1 (independent of body size) | O(1) memory                   |
  | `route`             | ~ns      | 0                            | —                             |
  | `deny::reason`      | ~1–8ns   | 0, flat 0→1M entries         | O(1) lookup, O(denied) memory |
  | `ratelimit::check`  | ~43–83ns | 0                            | Fixed-memory count-min        |

  **Headline: `key/verify` ≈ 28µs is ~350–650× every other per-request op.** This is why the rate
  guardrail sits before verify in `proxy::request_filter`.

- **End-to-end (`benches/e2e.rs`, `mise run bench:e2e`) — `criterion`.** Real `beyond-ai` binary
  - real nats-server + mock upstream (reuses `tests/common`). Four decomposed cases:
    `reject_missing_key_latency` (401, short-circuit before any upstream connection — transport floor),
    `byo_json_latency` (pure passthrough), `managed_json_latency` (verify + deny + key swap),
    `managed_sse_latency` (streaming response tap). Plus a concurrent-throughput group.

  All four cases land in ~110–120µs on loopback with ±15–20µs jitter — larger than the gateway's
  own CPU cost. This harness cannot resolve the verify cost (that's the unit bench's job). Its value:
  catching gross regressions (a buffering mistake, a dropped connection pool, an O(n) path added
  would move the band by far more than 20µs) and saved-baseline RPS trend via `--save-baseline`.

`mise run bench` runs both.
