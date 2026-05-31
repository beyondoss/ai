# Beyond AI Gateway — Architecture

A centralized, internal **egress L7 proxy** to LLM providers, built on **Pingora** + tokio. Apps point their stock
OpenAI/Anthropic SDK at it; the gateway authenticates, swaps in the real provider key, relays the
response untouched, and emits token-usage facts for billing.

**Self-contained:** no `path` deps into the `beyond` repo. Depends only on crates.io + the published
`beyond-slipstream` — so it clones/CI-builds/publishes anywhere.

## Request flow (`proxy.rs`)

```
client (stock SDK, Bearer/ x-api-key)
   │
   ▼ request_filter
   ├─ provider = dialect(path) [+ x-beyond-provider override]   (unknown → 400)
   ├─ extract key
   ├─ Content-Length abuse guard (declared size; streamed total enforced in body filter too)
   ├─ key format branch:
   │    • bai_…  → MANAGED: Ed25519 verify (stateless) → {tenant_id, vpc_id}
   │              → deny-set check (O(1), default-allow) → require pool key
   │    • else   → BYO: the user's own provider token, passed through unchanged
   └─ per-key rate guardrail (tenant / BYO-token hash; over ceiling → 429)
   ▼ upstream_peer        — TTL-cached DNS resolve → HttpPeer (no blocking getaddrinfo)
   ▼ upstream_request_filter — managed: swap auth header to pool key; BYO: leave it. Set Host.
   ▼ request_body_filter  — STREAM BODY THROUGH (never buffered); feed bytes to a structural
   │                         scanner that extracts the exact root-level `model` (O(1), memchr-fast);
   │                         enforce the body cap on the running total (chunked-safe)
   ▼ response_filter      — TTFT; streaming? = response Content-Type is text/event-stream
   ▼ response_body_filter — relay unbuffered; keep a bounded 64KB tail for the usage tap
   ▼ logging              — parse usage from tail (by dialect+streaming); emit `ai.usage` fact
        upstream: a registered provider (openai, anthropic, openrouter, fireworks,
                  groq, deepseek, together, cerebras, mistral, xai — + config-added)
```

## What lives where

- **NATS / slipstream:** exactly one thing — the **deny-set** (`blackhole.{tenant}`). Watched,
  fail-open. Auth and keys do **not** depend on NATS.
- **Config (boot, SSM/env):** `signing_keys` (Ed25519 **public** keys by kid — multiple for
  rotation), `pool_keys` (managed pool keys **by provider name**, from `AI_POOL_KEY_<NAME>` env),
  `provider_authorities` (per-name authority overrides / additions), `rate_limit_rps` (per-key
  request ceiling; 0 disables), `snapshot_path` (optional on-disk deny-set cache; see below),
  timeouts. Secret-bearing fields (`pool_keys`, `nats_creds`) are held as `Secret`, so a stray
  `Debug`/`Serialize` of the config can't leak them. See `config.example.toml`.
- **The virtual key (`bai_v1.{kid}.{payload}.{sig}`):** Ed25519-signed, payload = `{tenant_id,
  vpc_id}`, verified with a public key — stateless, no lookup. Minted by the control plane (it holds
  the private key); a compromised/OSS gateway can verify but not mint.

## Key invariants

- **Managed vs BYO by key format.** `bai_…` → verify + swap to pool key. Anything else → the user's
  real token, passed through (no swap, no deny-set, no per-tenant attribution).
- **Request body is never buffered** — it streams through with original framing; a streaming
  structural scanner (`peek::ModelScanner`, O(1), SIMD `memchr` skip over big values) extracts the
  exact root-level `model`. (Trade-off: OpenAI streaming without `stream_options.include_usage`
  isn't metered — the SDK/platform can set it.)
- **Response is never buffered** — relayed chunk-by-chunk; a bounded 64KB tail feeds the usage tap.
- **Deny-set is `O(denied)`, default-allow, fail-open.** Restore = explicit delete or TTL expiry.
  Seeding is **gap-free**: the seed records the stream revision it reflects, and the watch _resumes
  from that revision_ (`watch_prefix_from`) rather than starting live — so a deny entry written in
  the window between seeding and the watch attaching can't be lost (a plain `watch_prefix` uses NATS
  `DeliverPolicy::New` and would silently drop it). The resume revision is kept across reconnects, so
  a NATS blip resumes from where it left off instead of re-scanning.
- **Deny-set seeding has two modes (`snapshot_path`).** Unset (ephemeral/Fargate): scan
  `blackhole.*` from NATS each cold boot. Set (edge/tunnel, durable disk): load slipstream's on-disk
  snapshot (entries + saved cursor), enforce immediately on restart **before NATS reconnects**, and
  append each applied delta back to the file. The snapshot is a pure cache — delete it and the
  gateway falls back to scanning; a `CursorExpired` (history compacted past the cursor) does the same.
- **Auth works without NATS** (keys from config); a NATS outage only staleens the deny-set.
- **Per-key rate guardrail, not a spend control.** The deny-set is the spend/fraud authority but
  reacts on a lag and never sees floods that don't bill (auth failures, 4xx, BYO). A fixed-memory
  count-min limiter (`ratelimit`, pingora-limits) caps a single tenant's / BYO caller's request
  velocity — bounding a leaked/runaway key during deny-set lag and a retry-storm flood. Generous by
  default (a circuit breaker, not a quota); `rate_limit_rps = 0` disables it.
- **Routing is dialect-based** (model isn't known before peer selection); any non-default provider
  is reached via the `x-beyond-provider: <name>` header. **Providers are data** — a row in
  `route::KNOWN_PROVIDERS` (name, authority, **base path**, auth scheme) or a config entry — so
  adding an OpenAI-wire provider is one line, no new code paths. Each row's connection facts are
  **verified against the provider's official docs (cited inline in `route.rs`)**; the client's `/v1`
  prefix is rewritten to the provider's mount point (Groq `/openai/v1`, Fireworks `/inference/v1`,
  OpenRouter `/api/v1`) so a verbatim passthrough can't 404.
- **Connect retries only** (`fail_to_connect`); no HTTP-status retry (Pingora-idiomatic, SDKs back off).
- **Pricing is never here** — emit token _facts_; a closed downstream consumer prices.

## Modules

| Module                    | Role                                                                        | Tested        |
| ------------------------- | --------------------------------------------------------------------------- | ------------- |
| `key`                     | `bai_v1` parse + Ed25519 verify + mint; stateless identity                  | unit ✓        |
| `route`                   | data-driven provider table (name/authority/auth) + dialect default          | unit ✓        |
| `peek`                    | `ModelScanner` — streaming structural scan for the exact root-level `model` | unit ✓        |
| `usage`                   | token extraction (OpenAI/Anthropic, body + SSE)                             | unit ✓        |
| `deny`                    | sparse deny-set, default-allow, reason → status                             | unit ✓        |
| `ratelimit`               | per-key request guardrail (count-min, fixed memory, no GC)                  | unit ✓        |
| `secret`                  | redacting, zeroize-on-drop `Secret` newtype                                 | unit ✓        |
| `config`                  | Figment config; build keyring; pool keys/authorities by provider name       | unit ✓        |
| `state`                   | keyring + resolved provider registry + watched deny-set + TTL DNS cache     | unit ✓        |
| `store_watch`             | the single NATS watcher (deny-set), as a Pingora `BackgroundService`        | —             |
| `proxy`                   | the `ProxyHttp` impl                                                        | e2e ✓         |
| `metrics`/`doctor`/`main` | Prometheus, diagnostics, bootstrap                                          | e2e/compile ✓ |

## Verification

- **Unit (`cargo test --lib`):** key, route, peek, usage, deny, secret, config. `clippy
  --all-targets -D warnings` clean.
- **End-to-end (`tests/e2e.rs`, `mise run test:integration:rs`):** real `beyond-ai` binary + real
  nats-server + mock upstream. Covers managed key-swap + passthrough fidelity + usage metering
  (OpenAI JSON + SSE, **Anthropic `/v1/messages`** with `x-api-key` swap + metering), **BYO
  passthrough** (raw token unchanged), the **virtual key in either inbound header** (`Bearer` or
  `x-api-key`), and deny-set propagation: spend (write `blackhole.{tenant}` → 402, delete → 200) and
  **fraud** (→ 403). Error/edge paths: **missing key → 401**, **oversized `Content-Length` → 413**,
  **managed key for an unconfigured provider → 503**, **streaming tail compaction** (>128KB before
  the usage chunk still meters), **deny-set fail-open** (kill NATS → stale set retained, auth still
  works), and **on-disk snapshot survival** (blackhole a tenant, restart with NATS down → the hold is
  still enforced from disk). Managed/BYO/streaming seed **nothing** in NATS (signkey/pool keys from
  config), demonstrating auth's independence from NATS.
- **Live smoke (`tests/smoke.rs`, `mise run test:smoke`):** the real `beyond-ai` binary against the
  **real** provider hosts over TLS, one per provider in `KNOWN_PROVIDERS`. Proves what docs and the
  mock can't — real TLS/SNI, the `/v1`→base-path rewrite landing on a live mount (200, not 404), and
  auth passthrough. Traffic is BYO (the env key forwarded as the caller's token). Doubly guarded:
  every test is `#[ignore]` (a plain `cargo test` skips them) **and** skips unless its provider's API
  key env var (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, …) is set — so CI stays
  hermetic and you only hit providers you have keys for.

## Benchmarking

Two harnesses, best-tool-per-job, mirroring the unit/e2e split of the tests:

- **Unit micro (`benches/unit.rs`, `mise run bench:unit`) — `divan`.** Times the IO-free hot paths
  (`key` verify/mint, `peek::ModelScanner` over 0/4KB/256KB bodies with `model` placed _last_ =
  worst case, `usage` parsers, `route`, `deny`) **and** measures allocations natively: divan's
  `AllocProfiler` (installed as the global allocator) reports alloc/dealloc/grow **count + bytes**
  beside ns/iter, no extra plumbing — and stays clear of the crate's `#![deny(unsafe_code)]` (a
  hand-rolled `GlobalAlloc` would need `unsafe impl`). This makes the design's allocation claims
  _assertable_: `key/verify` shows **0 allocs** (stack-only decode — divan omits the alloc rows
  entirely), `peek` a flat **1 alloc** independent of body size (the O(1)-memory claim),
  `route`/`deny::parse_key` **0 allocs**. A regression surfaces as a non-zero / grown number.
- **A-1 end-to-end (`benches/e2e.rs`, `mise run bench:e2e`) — `criterion`.** The real `beyond-ai`
  binary + real nats-server + mock upstream (reuses `tests/common` verbatim), driven over real HTTP
  — measures the whole request path: single-request latency + concurrent throughput. criterion is
  chosen here for its saved-baseline comparison (`--save-baseline`), which tracks latency/RPS drift
  across runs. Allocations are _not_ measured (the gateway is a separate process — its heap is
  invisible to the bench); that's the unit bench's job. Needs `nats-server` on PATH (mise provides
  it).

`mise run bench` runs both.

## Out of scope / deferred

- **Go control plane** (mint/inject virtual keys, write deny entries) — separate workstream; the
  e2e mints keys directly.
- **OpenAI `stream_options` injection** — dropped to keep the request body a pure passthrough.
- **HTTP 5xx/429 response retries + `Retry-After`** — non-idiomatic in Pingora 0.8; SDKs back off.
- **Trickle/cancel e2e** — SSE relay is covered; incremental-timing/cancel assertions are flaky.
- Cross-dialect IR translation; caching; guardrails; ClickHouse ingestion wiring (table exists).
- **Anthropic streaming input tokens** can sit in `message_start` (response head) outside the 64KB
  usage tail on very long streams — a pre-existing tap limitation.
