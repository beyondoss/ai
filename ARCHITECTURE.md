# Beyond AI Gateway — Architecture

A centralized, internal **egress L7 proxy** to LLM providers, built on **Pingora** + tokio. Apps point their stock
OpenAI/Anthropic SDK at it; the gateway authenticates, swaps in the real provider key, relays the
response untouched, and emits token-usage facts for billing.

**Self-contained:** no `path` deps into the `beyond` repo. Depends only on crates.io + the published
`beyond-slipstream` — so it clones/CI-builds/publishes anywhere.

## Concepts & Terminology

| Term                                             | What It Controls / Gates                                                               | NOT                                                                          |
| ------------------------------------------------ | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Managed key** (`bai_v1.…`)                     | Ed25519-verified identity; enables key swap, deny-set check, and `ai.usage` billing    | A session token or capability grant — just tenant attribution                |
| **BYO key** (anything else)                      | Forwarded as-is to the provider; no swap, no billing, no deny-set                      | A lesser tier — same proxy, minus attribution and billing                    |
| **Pool key**                                     | Real provider API key held by the gateway; swapped in for managed traffic              | Per-tenant — one key per provider, shared by all managed callers             |
| **Tenant**                                       | The billing entity from the virtual key payload (`tenant_id: u32`)                     | An org, user, or namespace — an opaque integer the gateway doesn't interpret |
| **Dialect**                                      | Wire protocol implied by the request path (OpenAI `/v1/…` vs Anthropic `/v1/messages`) | The provider — dialect determines auth scheme and usage parsing format       |
| **Provider**                                     | Named row in the routing table: authority, base path, auth scheme                      | A vendor relationship — just connection facts and auth wiring                |
| **Deny-set**                                     | Sparse set of denied `tenant_id`s; gates managed traffic; default-allow                | An allowlist or ACL — misses are allowed, not blocked                        |
| **Tail tap**                                     | Bounded 64KB window kept from the end of the response for usage extraction             | A buffer or copy — the response is relayed unbuffered; only the tail is kept |
| **Snapshot**                                     | On-disk deny-set cache (entries + NATS cursor) for edge/tunnel deployments             | Persistent store — a pure cache; delete it and the gateway re-scans NATS     |
| **Virtual key** (`bai_v1.{kid}.{payload}.{sig}`) | Ed25519-signed token encoding `tenant_id` + `vpc_id`                                   | A session or auth token — stateless, no server-side lookup, no revocation    |

---

## Request flow (`proxy.rs`)

```
client (stock SDK, Bearer/ x-api-key)
   │
   ▼ request_filter
   ├─ provider = dialect(path) [+ x-beyond-provider override]   (unknown → 400)
   ├─ extract key                                               (missing → 401)
   ├─ rate guardrails ← BEFORE verify/connect: per-credential (seeded raw-key hash) +
   │                    global BYO aggregate (managed exempt; protects egress IPs); over → 429
   ├─ Content-Length abuse guard (declared size; streamed total enforced in body filter too)
   ├─ key format branch:
   │    • bai_…  → MANAGED: Ed25519 verify (stateless) → {tenant_id, vpc_id}
   │              → deny-set check (O(1), default-allow) → require pool key
   │    • else   → BYO: the user's own provider token, passed through unchanged
   ▼ upstream_peer        — TTL-cached DNS resolve → HttpPeer (no blocking getaddrinfo)
   ▼ upstream_request_filter — managed: swap auth header to pool key; BYO: leave it. Set Host.
   ▼ request_body_filter  — STREAM BODY THROUGH (never buffered); feed bytes to a structural
   │                         scanner that extracts the exact root-level `model` (O(1), memchr-fast);
   │                         enforce the body cap on the running total (chunked-safe)
   ▼ response_filter      — TTFT; streaming? = response Content-Type is text/event-stream; count
   │                         upstream response by provider+status class; set x-beyond-request-id
   ▼ response_body_filter — relay unbuffered; keep a bounded 64KB tail for the usage tap
   ▼ logging              — parse usage from tail (by dialect+streaming); emit `ai.usage` fact
   │                         (managed only — BYO has no tenant to bill); metrics count all traffic.
   │                         Every terminal path (reject + usage) logs the request_id for correlation
        upstream: a registered provider (openai, anthropic, openrouter, fireworks,
                  groq, deepseek, together, cerebras, mistral, xai — + config-added)
```

## What lives where

- **NATS / slipstream:** exactly one thing — the **deny-set** (`blackhole.{tenant}`). Watched,
  fail-open. Auth and keys do **not** depend on NATS.
- **Config (boot, SSM/env):** `signing_keys` (Ed25519 **public** keys by kid — multiple for
  rotation), `pool_keys` (managed pool keys **by provider name**, from `AI_POOL_KEY_<NAME>` env),
  `provider_authorities` (per-name authority overrides / additions), `rate_limit_rps` (per-credential
  request ceiling; 0 disables), `byo_rate_limit_rps` (aggregate ceiling for _all_ BYO traffic — the
  egress-IP guard; 0 disables), `snapshot_path` (optional on-disk deny-set cache; see below),
  timeouts. Secret-bearing fields (`pool_keys`, `nats_creds`) are held as `Secret`, so a stray
  `Debug`/`Serialize` of the config can't leak them. See `config.example.toml`.
- **The virtual key (`bai_v1.{kid}.{payload}.{sig}`):** Ed25519-signed, payload = `{tenant_id,
  vpc_id}`, verified with a public key — stateless, no lookup. Minted by the control plane (it holds
  the private key); a compromised/OSS gateway can verify but not mint.

## Key invariants

- **Managed vs BYO by key format.** `bai_…` → verify + swap to pool key. Anything else → the user's
  real token, passed through (no swap, no deny-set, no per-tenant attribution, and **no `ai.usage`
  billing event** — it would be an unbillable `tenant_id=0` row; aggregate metrics still count it).
- **Request body is never buffered** — it streams through with original framing; a streaming
  structural scanner (`peek::ModelScanner`, O(1), SIMD `memchr` skip over big values) extracts the
  exact root-level `model`. **One exception:** a _managed_ OpenAI chat/responses request is buffered
  so the gateway can inject `stream_options.include_usage` when the client streams without it —
  otherwise OpenAI emits no usage chunk and the request is unmeterable. Works out of the box (no
  client/SDK cooperation), framed upstream as chunked, bounded by `MAX_REQUEST_BODY`, scoped to that
  one path — BYO and everything else stay pure passthrough.
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
- **Two-tier rate guardrail, checked _before_ verify/connect, not a spend control.** The deny-set is
  the spend/fraud authority but reacts on a lag and never sees floods that don't bill (auth failures,
  4xx, BYO). Two fixed-memory count-min tiers (`ratelimit`, pingora-limits) cap velocity:
  - **Per-credential** — keyed by a seeded hash of the raw presented key (so collisions can't be
    precomputed to false-throttle another caller). Bounds a leaked/runaway key during deny-set lag, a
    retry-storm flood, **and the Ed25519-verify cost of a forged-key flood**: keying on the raw
    credential (not the verified tenant) is what lets the guard sit _ahead of_ the verify (the
    gateway's one ~28µs/req CPU cost; see Benchmarking), so a single bad token can't drive unbounded
    crypto work. Granularity is per-credential ≈ per-(tenant, app), since virtual keys are
    deterministic per that pair — not a per-tenant aggregate.
  - **Global BYO aggregate** — one shared bucket for _all_ BYO traffic. BYO connects outward to
    providers _from our egress IPs_ carrying the caller's token, so a flood of distinct **junk** BYO
    tokens (which slip past per-credential keying — each is its own bucket) would get those IPs
    rate-limited or banned by the provider, hurting _everyone_. This bounds that aggregate regardless
    of token variation. **Managed traffic is exempt** — it's verified before any upstream connect and
    can't be forged, so a random `bai_…` flood fails verify and never reaches a provider; exempting it
    keeps this shared bucket from ever shedding core tenant load. **Per-source-IP was considered and
    rejected** as the primary control: it depends on the calling task's real IP surviving ECS Service
    Connect (unconfirmed), and is worse than nothing if the peer is a collapsed mesh hop — so we chose
    the topology-independent aggregate. The blunt cap's residual (it sheds legit BYO under a flood; the
    default is an untuned guess; the real selective fix is a provider-feedback circuit breaker on
    upstream 401s) is recorded in full in the `ratelimit` **module-doc decision block** — read it
    before changing the knob or reaching for per-IP.

  Both tiers are generous circuit breakers, not quotas; `rate_limit_rps = 0` / `byo_rate_limit_rps = 0`
  disable them independently.
- **Routing is dialect-based** (model isn't known before peer selection); any non-default provider
  is reached via the `x-beyond-provider: <name>` header. **Providers are data** — a row in
  `route::KNOWN_PROVIDERS` (name, authority, **base path**, auth scheme) or a config entry — so
  adding an OpenAI-wire provider is one line, no new code paths. Each row's connection facts are
  **verified against the provider's official docs (cited inline in `route.rs`)**; the client's `/v1`
  prefix is rewritten to the provider's mount point (Groq `/openai/v1`, Fireworks `/inference/v1`,
  OpenRouter `/api/v1`) so a verbatim passthrough can't 404.
- **Connect retries only** (`fail_to_connect`); no HTTP-status retry (Pingora-idiomatic, SDKs back off).
- **`ai.usage` carries _both_ models: `model` (resolved) + `requested_model` (alias).** `model` is
  the id the provider resolved + billed, taken from the _response_ (a second `ModelScanner` over the
  response head; works for SSE — it skips the `data:` prefix and reads the first chunk's root
  `model`). It's the key for pricing **and** for reconciling against the provider's invoice, which
  itemizes by the pinned snapshot (`gpt-4o-2024-08-06`), not the alias. `requested_model` is what the
  client sent (`gpt-4o`) — product analytics, and a fallback rate when a snapshot is newer than the
  downstream price table. The two are equal when the response carried no model (error body), where
  `model` falls back to the alias. Emitting both is additive: a consumer that keyed on the alias
  doesn't break, and reconciliation still gets the exact id.
- **Pricing is never here** — emit token _facts_; a closed downstream consumer prices.

## Trust Boundaries

**What the gateway verifies (rejects if invalid):**

- Virtual key signature (Ed25519, stateless — no DB lookup)
- Virtual key format (`bai_v1.{kid}.{payload}.{sig}`, fixed 16-byte payload)
- Tenant not in deny-set (managed traffic only)
- Pool key configured for the requested provider (managed traffic only)
- Request body size ≤ `MAX_REQUEST_BODY` (declared Content-Length + streaming running total)
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
- Model validation would require a per-provider allowlist coupled to model release cadence
- BYO token validation requires a provider round-trip — the provider does it anyway

---

## Configuration

All fields configurable via `config.example.toml` and environment (`AI_` prefix, flat merge).
Secret-bearing fields (`pool_keys`, `nats_creds`) are held as `Secret` — stray `Debug`/`Serialize`
output redacts values.

| Field                         | Default                           | Runtime Effect                                                                                                                                                   |
| ----------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signing_keys`                | _(required)_                      | Map of kid → base64 Ed25519 public key. Multiple kids enable rotation. Missing → all traffic falls through to BYO treatment.                                     |
| `pool_keys.<name>`            | _(from `AI_POOL_KEY_<NAME>` env)_ | Real provider API key. Missing for a provider → managed requests to that provider return 503.                                                                    |
| `provider_authorities.<name>` | _(none)_                          | Override or add a provider's `authority` (host:port). Enables config-added providers beyond `KNOWN_PROVIDERS` with zero code change.                             |
| `snapshot_path`               | _(unset)_                         | Path for the on-disk deny-set cache. Unset → re-scan NATS on every cold boot. Set → load from disk and enforce before NATS reconnects (edge/tunnel deployments). |
| `rate_limit_rps`              | `100`                             | Per-credential request ceiling (count-min, keyed on raw key hash). `0` disables. Exceeded → 429. Checked before Ed25519 verify.                                  |
| `byo_rate_limit_rps`          | `1000`                            | Aggregate ceiling for all BYO traffic (single shared bucket). `0` disables. Managed traffic exempt.                                                              |
| `connect_timeout_secs`        | `10`                              | TCP connect timeout to the upstream provider. Exceeded → retry up to 2×, then 502.                                                                               |
| `read_timeout_secs`           | `600`                             | Response read timeout. 10 minutes accommodates long-running LLM streams.                                                                                         |
| `nats_url`                    | `nats://localhost:4222`           | NATS server for the deny-set watcher. Unreachable → fail-open (stale or empty set).                                                                              |
| `nats_creds`                  | _(unset)_                         | NATS credentials file path. Required for authenticated clusters.                                                                                                 |
| `listen_addr`                 | `0.0.0.0:8080`                    | Proxy listener address.                                                                                                                                          |
| `metrics_listen`              | `0.0.0.0:9090`                    | Internal admin/observability listener: `/metrics` (Prometheus scrape), `/livez`, `/readyz`. Separate from the client listener — not externally reachable.        |

---

## Failure Modes

| Failure                                     | What Actually Happens                                                                                       | Recovery                                                                                                        |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| NATS unreachable at boot                    | Deny-set starts empty (fail-open). Auth still works — keys from config.                                     | Watcher reconnects; seeds from NATS or disk snapshot on connect.                                                |
| NATS disconnects mid-run                    | Last-known deny-set stays active. New deny entries not applied until reconnect.                             | Watcher reconnects (1s→30s exponential backoff, reset on success) and resumes from saved revision — no re-scan. |
| NATS history compacted past snapshot cursor | `CursorExpired` → full re-scan from current NATS state.                                                     | After re-scan, new cursor set; delta watch resumes normally.                                                    |
| Virtual key tampered or forged              | Ed25519 verify fails → falls through to BYO treatment. No billing event.                                    | Billing miss detectable downstream; no security boundary breach.                                                |
| Pool key missing for provider               | Managed request returns 503 before any upstream connection.                                                 | Add `AI_POOL_KEY_<NAME>` env and redeploy.                                                                      |
| Provider DNS fails                          | `upstream_peer` returns error → 502 to client.                                                              | TTL-cached DNS (60s) serves stale; poisoned-lock guard re-resolves on next request.                             |
| Provider TCP connect fails                  | `fail_to_connect` retries up to 2×, then returns 502.                                                       | Client SDK retries with backoff. No HTTP-status retries (Pingora-idiomatic).                                    |
| Response body > 128KB before usage chunk    | Tail compaction fires: `drain(..half)` discards first half, keeps tail. Usage extracted from retained tail. | No action — O(1) tail tap is designed for this; SSE usage is always in the final data line.                     |
| Gateway crash mid-request                   | In-flight request drops; client receives TCP close, not a structured error. No partial state written.       | Client SDK retries. No DB writes in the request path — no cleanup needed.                                       |

---

## Modules

| Module                    | Role                                                                          | Tested        |
| ------------------------- | ----------------------------------------------------------------------------- | ------------- |
| `key`                     | `bai_v1` parse + Ed25519 verify + mint; stateless identity                    | unit ✓        |
| `route`                   | data-driven provider table (name/authority/auth) + dialect default            | unit ✓        |
| `peek`                    | `ModelScanner` — streaming structural scan for the exact root-level `model`   | unit ✓        |
| `usage`                   | token extraction (OpenAI/Anthropic, body + SSE)                               | unit ✓        |
| `deny`                    | sparse deny-set, default-allow, reason → status                               | unit ✓        |
| `ratelimit`               | two-tier guardrail: per-credential + global BYO (count-min, fixed mem, no GC) | unit ✓        |
| `secret`                  | redacting, zeroize-on-drop `Secret` newtype                                   | unit ✓        |
| `config`                  | Figment config; build keyring; pool keys/authorities by provider name         | unit ✓        |
| `state`                   | keyring + resolved provider registry + watched deny-set + TTL DNS cache       | unit ✓        |
| `store_watch`             | the single NATS watcher (deny-set), as a Pingora `BackgroundService`          | —             |
| `proxy`                   | the `ProxyHttp` impl                                                          | e2e ✓         |
| `admin`                   | `ServeHttp` on the metrics listener: `/livez`, `/readyz`, `/metrics`          | e2e ✓         |
| `metrics`/`doctor`/`main` | Prometheus, diagnostics, bootstrap                                            | e2e/compile ✓ |

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

Two harnesses, best-tool-per-job, mirroring the unit/e2e split of the tests. The framing is
**Theory of Constraints**: a proxy's steady-state constraint is upstream I/O, not gateway CPU — the
whole design exists to _stay off the critical path_. So the benches don't chase micro-optimizations;
they **prove the gateway's added cost is negligible and bounded**, i.e. that we never become the
constraint. Every bench maps to a function that runs on the per-request hot path (`proxy.rs`).

- **Unit micro (`benches/unit.rs`, `mise run bench:unit`) — `divan`.** Times the IO-free hot paths
  **and** measures allocations natively: divan's `AllocProfiler` (installed as the global allocator)
  reports alloc/dealloc/grow **count + bytes** beside ns/iter, no extra plumbing — and stays clear of
  the crate's `#![deny(unsafe_code)]` (a hand-rolled `GlobalAlloc` would need `unsafe impl`). Coverage
  follows the hot path: `key` verify/mint; `peek::ModelScanner` over 0/4KB/256KB bodies with `model`
  placed _last_ = worst case; `usage` parsers; `route`; `deny` (both the off-path ingest parse,
  `parse_key`/`parse_reason`, **and** the on-path `reason()` lookup run on every managed request); and
  `ratelimit::check` (both tiers — `check_managed` runs the per-credential tier only; `check_byo` runs
  the per-credential tier **plus** the global BYO aggregate bucket). This makes the design's
  allocation/complexity claims _assertable_: `key/verify` shows **0 allocs** (stack-only
  decode — divan omits the alloc rows entirely), `peek` a flat **1 alloc** independent of body size
  (the O(1)-memory claim), `route`/`deny::parse_key` **0 allocs**, **`deny::reason` is 0-alloc and flat
  across 0→1M denied tenants** (the O(1)-lookup, `O(denied)`-memory claim — ~1ns/8ns), and
  **`ratelimit::check` is 0-alloc** (~43ns managed / ~83ns BYO — the delta is the second tier's bucket
  `observe` plus hashing a longer token; fixed-memory count-min, no per-credential entry). A regression
  surfaces as a non-zero / grown / size-scaling number. **The headline this bench exists to assert:
  `key/verify` ≈ 28µs is ~350–650× every other per-request op** (deny lookup, ratelimit, route all in
  the **nanoseconds**), so verify is the gateway's one real per-request CPU cost — the constraint that
  motivates checking the rate guardrails _before_ it (`proxy::request_filter`), so a forged-key flood
  is rejected for tens of ns instead of ~28µs each. Everything else is allocation-free and invisible
  against a network round trip.
- **A-1 end-to-end (`benches/e2e.rs`, `mise run bench:e2e`) — `criterion`.** The real `beyond-ai`
  binary + real nats-server + mock upstream (reuses `tests/common` verbatim), driven over real HTTP —
  measures the whole request path across four cases that **decompose** where time goes:
  `reject_missing_key_latency` (401, short-circuited before any upstream connection — the bare
  transport floor), `byo_json_latency` (pure passthrough), `managed_json_latency` (verify + deny +
  key swap), and `managed_sse_latency` (exercises the streaming response tap: tail buffer + bounded
  compaction). Plus a concurrent-throughput group. criterion is chosen for its saved-baseline
  comparison (`--save-baseline`), which tracks latency/RPS drift across runs. Allocations are _not_
  measured (the gateway is a separate process — its heap is invisible to the bench); that's the unit
  bench's job. Needs `nats-server` on PATH (mise provides it).
  - **What the decomposition shows (loopback laptop) — and its limit:** all four cases land in a
    ~110–120µs band, and run-to-run variance is **±15–20µs** (loopback sub-150µs round-trips are
    dominated by OS scheduling jitter). That noise floor is _larger_ than the gateway's own per-request
    CPU (verify ≈28µs, everything else ns) — so this harness **cannot resolve** the verify cost, and the
    reject/BYO/managed cases are statistically indistinguishable here. Two honest conclusions follow:
    (1) the right tool for the gateway's CPU cost is the in-process `unit` bench, not this one; (2) for
    _legitimate_ managed traffic the e2e latency is **expected to be flat** across the verify reorder —
    moving the rate guard before verify doesn't change the legit path (verify still runs); its win is on
    the _throttled_ path (verify skipped, proven at the unit level: 42ns vs 28µs) and in per-request
    allocator pressure (the lazy `resp_tail`, below this harness's resolution). What this harness _is_
    good for: catching gross regressions (a buffering mistake, a dropped connection-pool, an O(n) added
    to the path would move the band by far more than 20µs) and the saved-baseline RPS trend over time.

`mise run bench` runs both.
