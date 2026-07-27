# Beyond AI Gateway — Architecture

Takes HTTP requests carrying an OpenAI- or Anthropic-dialect payload, authenticates the caller via
Ed25519 virtual key or BYO provider token, swaps in a pool key for managed traffic, relays the
request and response byte-for-byte to the upstream provider, and emits a token-usage billing fact
(`ai.usage`) on completion — all without buffering the body or response stream.

**Self-contained:** no `path` deps into the `beyond` repo. Depends only on crates.io + the
published `beyond-slipstream` — clones, CI-builds, and publishes anywhere.

---

## Concepts & Terminology

| Term                                             | What It Controls / Gates                                                                                                                                                          | NOT                                                                          |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Managed key** (`bai_v1.…`)                     | Ed25519-verified identity; enables key swap, deny-set check, and `ai.usage` billing                                                                                               | A session token or capability grant — just tenant attribution                |
| **BYO key** (anything else)                      | Forwarded as-is to the provider; no swap, no billing, no deny-set                                                                                                                 | A lesser tier — same proxy, minus attribution and billing                    |
| **Pool key**                                     | Real provider API key held by the gateway; swapped in for managed traffic                                                                                                         | Per-tenant — one key per provider, shared by all managed callers             |
| **Tenant**                                       | The billing entity from the virtual key payload (`tenant_id: u64`)                                                                                                                | An org, user, or namespace — an opaque integer the gateway doesn't interpret |
| **Dialect**                                      | A provider attribute (OpenAI-wire vs Anthropic-wire) driving usage parsing; for a bare-path request it's derived from the path to pick the default provider                       | The provider — a prefixed request uses its provider's dialect, not the path  |
| **Provider**                                     | The request's **first path segment** (`/{provider}/…`); a named row in the routing table: authority, dialect, auth scheme                                                         | A vendor relationship — just connection facts and auth wiring                |
| **Model route** (`/auto/…`)                      | Reserved first segment; provider, upstream path, and model id all come from the catalog row named by the `x-beyond-model` header, and the body's `model` is rewritten per attempt | A dialect translator — candidates must share a wire format                   |
| **Candidate**                                    | One `(provider, upstream model id, path)` a catalog row will accept, in preference order; tried on a connect failure                                                              | A load-balancing pool — strictly ordered, and only entered on failure        |
| **Deny-set**                                     | Sparse map of denied `tenant_id`s → reason; gates managed traffic; default-allow                                                                                                  | An allowlist or ACL — misses are allowed, not blocked                        |
| **Tail tap**                                     | Bounded 64KB window kept from the end of the response for usage extraction                                                                                                        | A buffer or copy — the response is relayed unbuffered; only the tail is kept |
| **Snapshot**                                     | On-disk deny-set cache (entries + NATS cursor) for edge/tunnel deployments                                                                                                        | Persistent store — a pure cache; delete it and the gateway re-scans NATS     |
| **Virtual key** (`bai_v1.{kid}.{payload}.{sig}`) | Ed25519-signed token encoding `tenant_id` + `vpc_id` (16-byte fixed payload)                                                                                                      | A session or auth token — stateless, no server-side lookup, no revocation    |

---

## Data Flow

### Happy Path

```
Client (stock OpenAI/Anthropic SDK)
  │
  ▼  request_filter (proxy.rs)
  │  ├─ Route: first segment → provider row (authority, dialect, auth scheme)
  │  │    …or `/auto` → x-beyond-model header → catalog row → ordered candidates
  │  │      no/unknown model ──────────────────────────────────► 404
  │  │      BYO key (managed-only route) ─────────────────────► 400
  │  │      no candidate holds a pool key ────────────────────► 503
  │  ├─ Extract key: x-api-key / api-key / x-goog-api-key / Authorization Bearer / ?key= query param
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
  │  └─ Circuit breaker (per provider, all traffic): if OPEN ─────► 503
  │       (claims a half-open probe permit only on an actual attempt)
  │
  ▼  upstream_peer (proxy.rs)   — runs once per attempt, before any body byte
  │  Reset request-body phase state (a retry replays bytes through the body filter)
  │  Model-routed: resolve the outgoing candidate's breaker permit, then walk candidates —
  │    skip any whose breaker is OPEN, allow() the one chosen, set forward_path to that
  │    candidate's own catalog path; DNS failure advances rather than ending the request
  │  TTL-cached DNS resolve (60s) → HttpPeer (TLS, H2 pref, timeouts)
  │  DNS fail ──────────────────────────────────────────────────── 502
  │  TCP connect fail (retry 2× same peer, or next candidate) ──── 502
  │
  ▼  upstream_request_filter (proxy.rs)
  │  Managed: remove every static-key header (authorization, x-api-key, api-key,
  │    x-goog-api-key) UNCONDITIONALLY → inject pool key in the provider's own scheme
  │  BYO: leave auth header unchanged
  │  Set Host; path: verbatim for /{provider} (prefix stripped), or the candidate's
  │    own catalog path for /auto. Model-routed: strip x-beyond-model
  │  OpenRouter + managed only: dashboard-attribution headers (HTTP-Referer, X-OpenRouter-*)
  │
  ▼  request_body_filter (proxy.rs)  — streamed through, except where a rewrite needs the whole body
  │  Enforce running size cap (chunked-safe) ──────────────────── 413
  │  Managed + streamed: feed chunks → ModelScanner (peek.rs), root-level `model`, O(1) mem
  │    (BYO skips it — `model` is only ever read on the managed billing path)
  │  Injection-eligible (managed + OpenAI dialect + path suffix /chat/completions):
  │    buffer full body → ONE fused walk (peek::scan_buffered) yielding `model`, its byte
  │    span, and the splice offset → inject stream_options.include_usage → re-frame chunked
  │  Model-routed: same buffer, and `model` is spliced to the serving candidate's own id
  │    (rewrite first — the injection offset precedes the value, so it cannot move)
  │
  ▼  Provider upstream  (OpenAI / Anthropic / Groq / DeepSeek / …)
  │
  ▼  response_filter (proxy.rs)
  │  Record TTFT; detect streaming (Content-Type: text/event-stream)
  │  Count upstream response by provider + status class
  │  Set x-beyond-request-id header
  │
  ▼  response_body_filter (proxy.rs)  — response relayed chunk-by-chunk, never buffered
  │  Managed only: feed chunks → ModelScanner::for_response → billed model
  │    (accepts Anthropic's nested message.model, so it stops in the first chunk for both dialects)
  │  Append to bounded 64KB tail (copy_within compaction once tail > 128KB)
  │  Anthropic SSE only: also keep a bounded 8KB head — message_start carries input + cache
  │    tokens and would otherwise be compacted out of the tail
  │
  ▼  logging (proxy.rs)
     Parse usage from tail (by dialect + streaming flag)
     Emit ai.usage fact: tenant, vpc, model, requested_model, routed_model, token counts +
       reasoning breakout (managed only)
     Record circuit-breaker outcome, only if one is still owed (breaker_pending): 5xx / upstream
       failure → failure; else → success (429 and client aborts included)
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

Providers are **data rows**, not code paths. `KNOWN_PROVIDERS` in `route.rs` lists 11 built-in
providers (openai, anthropic, openrouter, fireworks, groq, deepseek, together, cerebras, mistral,
xai, openai-codex); each row carries its authority (host:port), dialect (OpenAI-wire vs
Anthropic-wire), and auth scheme (`Bearer`, `x-api-key`, or `api-key`). The `provider_authorities`
config key adds or overrides rows at boot with zero code change; `provider_dialects`/
`provider_auth_schemes` set the dialect/auth scheme for a **config-added** provider (default
OpenAI/Bearer for backward compatibility — see Configuration). A known provider's dialect/scheme is
always fixed in `KNOWN_PROVIDERS`, never overridable from config. This is how Azure OpenAI is
supported: its per-resource host isn't knowable at compile time, so it's always config-added
(`provider_authorities.azure = "..."` + `provider_auth_schemes.azure = "api-key"`), never a
`KNOWN_PROVIDERS` row — see `config.example.toml` and the AWS SigV4 section below for the same
pattern applied to Bedrock's bearer-token mode.

The routing rule: **first path segment = provider name**. `/groq/openai/v1/chat/completions` routes
to Groq and forwards `/openai/v1/chat/completions` verbatim. A bare path that is _exactly_ `/v1` or
starts with `/v1/` (boundary-checked — `route::is_default_prefix`, not a raw string-prefix test)
matches the dialect default (OpenAI or Anthropic based on which default is set); a lookalike like
Google Gemini's `/v1beta/…` does **not** qualify and 404s as an unknown provider instead of being
silently absorbed into the OpenAI default. Unknown segment → 404.

### Model routing (`/auto`, `providers::catalog`)

One reserved first segment routes by **model** instead of provider: `/auto/…` takes the canonical
model name from the `x-beyond-model` header, resolves it in the catalog to an ordered list of
candidate providers, and tries them in order. The name is a header rather than the request body
because the provider must be chosen before `upstream_peer` runs, which is strictly before any body
byte is available. This arm is reached only after a provider-table miss, so `/{provider}/…` traffic
runs exactly the code it always did; `auto` is refused as a provider name at boot so config cannot
shadow it.

Three things differ from the provider-routed path, all consequences of the client no longer naming
the provider:

- **The upstream path comes from the catalog, per candidate.** Providers disagree on where an
  endpoint lives and the disagreement is _not_ a prefix: Anthropic serves Messages at `/v1/messages`
  from a base carrying no path, OpenRouter serves the same wire at `/api/v1/messages`. No client
  suffix is correct for both, so each candidate states its path outright and `forward_path` is set
  from it per attempt. The client points its SDK at `…/auto` and the row decides.
- **The model id is rewritten per attempt.** Providers essentially never share a string —
  `claude-opus-4-8` at Anthropic is `anthropic/claude-opus-4.8` at OpenRouter — so the body's `model`
  is spliced to whatever the serving candidate calls it (`peek::scan_buffered` reports the value's
  byte span). Because the body may change length, `/auto` requests are buffered and re-framed exactly
  as the injection path is; the two are one predicate (`RequestCtx::rewrites_body`).
- **It is managed-only.** A BYO token belongs to one provider, so selecting among candidates would be
  a guess and failing over would hand one vendor's key to another. BYO on `/auto` → 400.

Failover covers both shapes: a candidate that will not **connect** (refused, timed out, or absent
from DNS) and one that **answers with a 5xx**, provided nothing has gone downstream yet and the
request body is still replayable. Candidates whose breaker is open are skipped without an attempt.
See "Status-based failover, and where it stops" for the 5xx path and its one real limit.

Catalog rows live in `providers::catalog`, shared with the agent. Every candidate in a row must
share a wire format, enforced by a test: the gateway rewrites ids but does **not** translate between
API shapes, so a mixed row would send an Anthropic Messages body to a Chat Completions endpoint and
then parse the reply with the wrong dialect's usage extractor — a zero-token billing row rather than
a visible error.

**Wire belongs to the row, not the provider.** `ProviderSpec::wire` is one value per provider and
that is an approximation: OpenRouter serves the OpenAI wire at `/api/v1/chat/completions` _and_ a
genuine Anthropic Messages wire at `/api/v1/messages`, and Fireworks is the same story from the other
side (`agent_core::dialect::is_fireworks_anthropic_wire_model`). So each row declares its own `wire`,
and `request_filter` takes the request's dialect from **the row** rather than from whichever provider
it happens to start on. Reading the provider there fails silently in the worst way — an Anthropic
response meets the OpenAI extractor, trips the dialect-mismatch guard, and bills zero tokens.

That is what makes **Claude failover real today**: `claude-opus-4-8` routes to Anthropic first and
falls back to OpenRouter's Messages endpoint as `anthropic/claude-opus-4.8`. OpenRouter itself fronts
Bedrock for that traffic, so this is Bedrock-backed Claude failover with no SigV4 and no
`application/vnd.amazon.eventstream` decoding. Every row and candidate is verified against the live
providers by `catalog_rows_are_servable` in `tests/smoke.rs`.

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

| Dialect   | Format     | Fields                                                                                                                                                           |
| --------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OpenAI    | JSON body  | `usage.prompt_tokens`, `usage.completion_tokens`, `usage.prompt_tokens_details.cached_tokens`, `usage.completion_tokens_details.reasoning_tokens`                |
| OpenAI    | SSE stream | Terminal `data:` line (before `[DONE]`), same fields (Responses API: nested `response.usage`, `output_tokens_details.reasoning_tokens`)                          |
| Anthropic | JSON body  | `usage.input_tokens`, `usage.output_tokens`, `usage.cache_read_input_tokens`, `usage.cache_creation_input_tokens`, `usage.output_tokens_details.thinking_tokens` |
| Anthropic | SSE stream | `message_delta` event with `usage` block (thinking tokens on the same block)                                                                                     |

Missing or zero usage fields deserialize to zero (safe default) — **except** `reasoning_tokens`
(`Usage::reasoning_tokens: Option<u64>`), which stays `None` when the provider didn't report it at
all, distinct from `Some(0)` when it reported a real zero; that distinction is unrecoverable once the
request completes, so it's never collapsed to a bare zero. If the tail is truncated by the compaction
drain, the usage chunk is still present because SSE usage is always the final `data:` line and the
tail keeps the last 64KB.

**Dialect-mismatch guard:** a config-added provider whose `provider_dialects` value doesn't match its
actual wire (e.g. an Anthropic-wire vendor left at the default OpenAI dialect) would otherwise have
its `usage` block parsed by the wrong dialect's parser. Because both parsers' fields are
`#[serde(default)]`, that misparse used to succeed silently — every field defaults to zero, producing
`Some(Usage::default())`: a zero-token billing row indistinguishable from a real (and wrong)
zero-usage response. `openai_body`/`openai_stream` and `anthropic_body`/`anthropic_stream` now check
for the _other_ dialect's characteristic field names (Anthropic's `input_tokens`/`output_tokens` vs
OpenAI's `prompt_tokens`/`completion_tokens`) before accepting a parse, and return `None` on a match —
tripping `usage_parse_errors_total` (see Metrics) instead of a silent zero-billing row.

### Deny-Set (`deny.rs`)

A `HashMap<u64, DenyReason>` (tenant_id → reason). Only denied tenants are stored — the map is
`O(denied)` in memory regardless of total tenant count. Lookup is one hash probe. Written
exclusively by the NATS watcher via `ArcSwap`; reads on the hot path are lock-free.

Reasons: `Spend` (→ 402), `Fraud` (→ 403), `Unknown` (→ 403, fail-safe for unrecognized values).
Restore = explicit delete from NATS KV or TTL expiry — no gateway-side timer.

### Rate Guardrails (`ratelimit.rs`)

Two fixed-memory tiers, checked before Ed25519 verify and before any upstream connection:

| Tier                 | Key             | State           | Default ceiling | Managed exempt? |
| -------------------- | --------------- | --------------- | --------------- | --------------- |
| Per-credential       | Hash of raw key | 5.24 MB sketch  | 100 req/s       | No              |
| Global BYO aggregate | Single bucket   | one 64 B atomic | 1000 req/s      | **Yes**         |

Only the per-credential tier needs a sketch: its key cardinality is unbounded, so it uses a pair of
count-min estimators (5 × 65536 counters each) rotated at the window boundary, giving fixed memory
with no per-key entry and no GC. The `SLOTS` derivation — peak N, the false-throttle budget against
`rate_limit_rps`, and the cache/rotation cost on the other side — is written out in full at the
constant in `ratelimit.rs`. The global BYO tier is _one_ bucket, so it is one cacheline-isolated
`AtomicU64` packing `(window_index, count)`: exact, contention-minimal, and reset-free (opening a
window is the same CAS as counting a request).

Neither tier uses `pingora_limits::rate::Rate`. `Rate::maybe_reset` subtracts an atomically-loaded
reset timestamp from an independently-taken clock reading, which underflows whenever the two are
inverted by a stall — a panicking worker under the workspace's `overflow-checks`, reproduced in
seconds under oversubscribed threads. The local `WindowedRate` keeps the same red/blue rotation but
compares monotonic window _indices_ instead, so there is nothing to underflow. Both tiers take the
window index from one clock reading per request (`RateLimit::check_at`).

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

### Circuit Breaker (`circuit_breaker.rs`)

A per-provider, lock-free circuit breaker (single packed `AtomicU64`; windowed failure policy) sits
on the upstream path. It protects against a **broken provider**, which is a different failure than
the rate guardrails (which protect against abusive _inbound_ load):

- **Failure = the provider is broken** — a `5xx` response or a connect failure. After
  `circuit_breaker_threshold` failures within `circuit_breaker_window_secs`, the breaker **opens**:
  requests to that provider fast-fail with `503` (`ai_rejections_total{reason="circuit_open"}`)
  instead of piling up against `read_timeout_secs` and exhausting connection / in-flight slots for
  _every_ provider (head-of-line blocking by one sick dependency). After `circuit_breaker_reset_secs`
  it half-opens and admits a probe; success closes it, failure reopens it.
- **A `429` is NOT a failure.** It means the provider is healthy and throttling our pool key — a
  velocity/spend signal the rate limiter and the client's `Retry-After` backoff own. Tripping on it
  would convert a self-healing throttle into a self-inflicted outage. The breaker records any response
  that _arrived_ (2xx/3xx/4xx incl. 429) as a **success**; only 5xx and transport failures count
  against it.
- **A client giving up is NOT a failure.** Pingora tags a client-side abort `ErrorSource::Downstream`,
  and only non-`Downstream` errors count. Cancellation is routine for a coding agent (a user hits ESC
  on a slow turn); counting those opened breakers on perfectly healthy providers, and because
  `half_open_permits` is 1, a cancel-prone request drawn as the recovery probe reopened the breaker
  every time — so it could not recover while users were cancelling.
- **Applies to all traffic** (managed + BYO) — a down provider is down regardless of whose key is
  used. One breaker per provider, built at boot, shared lock-free across callers.
- `circuit_breaker_threshold = 0` disables it.

**The permit ledger.** `RequestCtx::breaker_pending` is true **iff** exactly one `allow()` is
outstanding against whatever `provider` currently points at. `logging` records only when it is set,
and clears it — so one `allow()` yields exactly one `record_*`, and a scarce half-open probe permit
can neither leak nor be resolved twice.

For a provider-routed request that is the old behaviour restated: `allow()` is the last thing in
`request_filter` (after every other rejection, so a permit corresponds to a real upstream attempt),
and `breaker_pending` is simply `breaker.is_some()`.

For a **model-routed** request the ledger is owned by `upstream_peer`, which gates each candidate as
it is chosen and records the outgoing candidate's failure when it moves on. Three reasons it lives
there and not in `fail_to_connect`:

- `upstream_peer` is the only hook that changes `rc.provider`, which is what makes recording the
  wrong candidate structurally impossible.
- It is the only hook that runs on _every_ attempt. Pingora's default `error_while_proxy` marks a
  reused-connection failure retryable on its own, without consulting `fail_to_connect` at all.
- `fail_to_connect` would double-record whenever the candidate list ran out: `retry` stays false, and
  `logging` then also resolves the still-pending permit — tripping the breaker at half its configured
  threshold on the last candidate, precisely where traffic lands once the primaries are sick.

Candidate selection gates with `allow()`, deliberately **not** `state()`: the OPEN → HALF_OPEN
transition happens _inside_ `allow()`, so a `state()` pre-check would report `Open` past the reset
timeout, skip a candidate `allow()` would have admitted as a probe, and leave the breaker with no way
to ever close.

---

## Why It Behaves This Way

### Why rate guardrails sit before Ed25519 verify

Ed25519 verify is ~28µs — roughly 350–650× more expensive than every other per-request operation.
A flood of forged `bai_v1` tokens could drive unbounded crypto work if the rate limit came after
verify. By checking the per-credential bucket first (keyed on the raw token, no crypto), a
forged-key flood is rejected in tens of nanoseconds per request. Legit traffic is unaffected: the
rate guard passes through, then verify runs as normal. The unit bench (`benches/unit.rs`) asserts
this: `key/verify` ≈ 23µs; `ratelimit::check` ≈ 39–60ns single-threaded, ≈ 88–220ns at 16 threads
under a flood of distinct credentials; 0 allocations for either.

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

**Revision 0 is not a resume point.** A seed scan that finds no `blackhole.*` entries yields
revision 0, and slipstream treats a cursor as resumable only when `rev > 0` — at 0 it falls back to
exactly the `watch_prefix`/`DeliverPolicy::New` this design exists to avoid. So an empty deny-set is
deliberately treated as _unseeded_ (`is_resumable`), and the next connect rescans rather than
marking itself seeded and never looking again. Without that, a gateway that booted against an empty
bucket would attach with `New` for the life of the process and silently never pick up the first
deny written while it was starting.

### Why BYO token validity is never checked

Checking a BYO token requires a round-trip to the provider. The provider does that check anyway and
returns 401 if the token is invalid — the client sees the same rejection it would get going direct,
just routed through the gateway. Adding a gateway-side preflight check would double the latency for
every BYO request on the error path with no security benefit at the gateway layer.

### Why AWS SigV4 (Bedrock's default credential chain) is not supported

`AuthScheme` has four variants — `Bearer`, `XApiKey`, `ApiKey` (Azure OpenAI's bare-key `api-key`
header), and `CustomHeader` (a `Bearer`-prefixed value in a differently-named header — Cloudflare AI
Gateway's `cf-aig-authorization` shape; added for the header format's sake but not yet wired to any
built-in or config-added provider, since Cloudflare's own base URL is templated per account+gateway
id, a routing shape this gateway's one-fixed-authority-per-provider-name model doesn't express) —
because every supported provider authenticates with a **static credential string** that the gateway
can swap verbatim into a header. Bedrock's default AWS credential chain (access keys, `AWS_PROFILE`,
the ECS task role, a web identity token) doesn't work that way: each request is signed with
**SigV4**, a signature computed over the method, path, headers, timestamp, and a hash of the body,
using credentials the _signer_ holds. There is no static string to swap in — the signature is
derived fresh, per request, from the exact bytes being sent.

That's structurally incompatible with a byte-relay-plus-key-swap gateway. Supporting it for real
would mean the gateway holds AWS credentials itself and **re-signs every relayed request** — a
SigV4 implementation covering canonical request construction, credential-scope derivation, and
clock-skew handling, running per-request server-side. That's a materially different feature (an AWS
signing proxy), not a config knob or a small patch to `AuthScheme`, and it doesn't fit this gateway's
"provider is a data row" model, where adding a vendor is a struct literal, not a signing engine. We
deliberately do not bolt on a partial implementation (e.g. accepting only unsigned requests, or
signing with a fixed clock skew) — a SigV4 gateway that's subtly wrong fails silently at the provider
with a cryptic signature-mismatch 403, which is worse than not supporting the mode at all.

**`AWS_BEARER_TOKEN_BEDROCK` mode solves the auth half only — it does not make Bedrock a working
route.** Bedrock also accepts a plain long-lived bearer token (no signing), which is exactly the
`Bearer`/`XApiKey` shape this gateway already handles as a config-added provider (see
`provider_authorities`/`provider_dialects` in Configuration), so a bearer-token credential _authenticates_
cleanly. But authenticating is not the same as completing a turn: there is zero Bedrock Converse/
Converse-Stream **wire-format** code anywhere in this gateway or in `agent-core`'s dialects (`grep -ri
bedrock` across both turns up only prose and error-message pattern-matching, never a request/response
shape). Bedrock's request body isn't Anthropic's `/v1/messages` shape and its response isn't SSE — it's
AWS's own binary `application/vnd.amazon.eventstream` framing, which this gateway's usage extractor and
every agent-core dialect decoder assume is never what arrives. Configuring `provider_dialects.bedrock =
"anthropic"` today would relay a request the provider rejects (wrong body shape) or, if that were
somehow fixed client-side, a response this stack can't parse at all (wrong stream framing) — an operator
following this doc's bearer-token instructions alone gets a route that passes auth and then fails to
complete a turn, not a working Bedrock integration. Full support needs both the SigV4 signing infra
described above (for the default credential chain) _and_ a dedicated Bedrock wire dialect (Converse-API
request mapping, event-stream response decoding) — neither exists, and building the latter without the
former only covers Bedrock's less common auth mode. Out of proportion for this pass; revisit as a
dedicated feature if Bedrock support is ever prioritized.

### Why OpenAI Codex traffic only ever goes over HTTP+SSE, never WebSocket

pi's own Codex client (`openai-codex-responses.ts`) defaults `transport` to `"auto"`, which _prefers_ a
persistent WebSocket connection and only falls back to HTTP+SSE on a connection-limit error or a
transport failure — pi treats WebSocket as its primary, tested path, not a fallback. Beyond's agent-core
client only ever speaks HTTP+SSE to Codex (`GatewayClient::send_with_retry`); there is no WebSocket
transport at all.

This was evaluated and deliberately deferred, not overlooked. Two independent reasons, either one
sufficient on its own:

- **Client-side**: pi's WebSocket path is a substantial subsystem (~1000+ lines in
  `openai-codex-responses.ts` alone) — per-session connection caching and reuse, reconnect/continuation
  state across turns, a connection-limit-reached retry loop, its own header construction and binary/text
  frame parsing for the Responses event stream, and a proxy-aware `WebSocket` constructor shim. It would
  also be a new runtime dependency (a WebSocket client crate) agent-core doesn't currently have, and a
  materially different request/response lifecycle than the current one-shot-per-turn `send_with_retry`
  path (a long-lived, session-scoped connection instead of a fresh request per turn). Disproportionate
  next to the HTTP+SSE path already working and covered by tests.
- **Gateway-side**: Codex traffic that goes through this gateway at all uses `RouteOverride::Prefixed`
  (the `/openai-codex` `KNOWN_PROVIDERS` row) — still relayed through Pingora, not bypassed like
  Copilot's `RouteOverride::Direct`. This gateway has no WebSocket-upgrade proxying (no `Connection:
  Upgrade`/`101 Switching Protocols` handling anywhere in `src/`); it is an HTTP request/response (and
  SSE-response) relay only. A client-side WebSocket transport for Codex could not be relayed through this
  gateway as-is — it would have to bypass the gateway entirely (a Copilot-style `Direct` route straight
  to `chatgpt.com`'s WebSocket endpoint), sidestepping this gateway's pooling, metering, and rate-limiting
  for that traffic, which is a real behavior change beyond just "add a transport option."

Net: HTTP+SSE is a real, working, tested path (pi's own fallback, not a hack), and building the
WebSocket path properly requires both a nontrivial client-side subsystem and a gateway-side proxying
capability that doesn't exist. Revisit if Codex's HTTP+SSE path is ever observed to hit the
connection-limit ceiling pi's WebSocket path exists to avoid.

### Status-based failover, and where it stops

A provider that answers badly — a `500`, or Anthropic's `529 overloaded` — is the common outage, and
`upstream_response_filter` handles it: it runs strictly before anything is written downstream (both
`h1_response_filter` and `h2_response_filter` call it ahead of `write_response_tasks`), so returning
a retryable error there re-enters pingora's retry loop and `upstream_peer` picks the next candidate.

Erroring at that hook rather than in `response_filter` keeps the per-attempt state clean for free:
`response_filter` never runs for the abandoned attempt, so nothing increments `active_streams`,
observes TTFT, or sets `upstream_status`, and `response_body_filter` never feeds the tail, head, or
response model scanner. The next attempt starts from the slate the first one did.

Two deliberate non-cases:

- **A `429` is relayed, never failed over.** It is a healthy provider throttling our pool key — the
  same judgement the circuit breaker makes. Re-asking a different vendor converts a self-healing
  throttle into spend somewhere else, and the client's `Retry-After` already owns it.
- **When every candidate 5xxes, the client gets the last provider's own status**, not a synthetic
  error. Better diagnostics than an exhausted retry loop produces.

**Where it stops: bodies that cannot be replayed.** Failover rides pingora's retry, whose request
buffer is `BODY_BUF_LIMIT` = 64 KiB, a private constant with no knob. Past it the body cannot be
re-sent, so the 5xx is relayed rather than attempted — a retry there would hand the next upstream
headers describing a body it never writes, hanging it until `read_timeout_secs`.

The gate is `is_body_done() && !retry_buffer_truncated()`, and the first half deserves an
explanation because it is not the obvious one. Truncation reports on what has been buffered _so
far_, so a provider that 5xxes fast — which is what a failing provider does — answers before a large
body has finished streaming in, and the check reads `false` only because the bytes that would
truncate it have not arrived yet.

Retrying there would not be _unsafe_: pingora replays the buffered prefix with
`end_of_body = is_body_done()` and the duplex loop reads the remainder straight from the socket, so
the next candidate does receive the whole body. It would be **timing-dependent** — the same request
fails over or does not, depending on how quickly the upstream rejected it. On a path that decides
which vendor gets billed, a deterministic rule is worth more than the extra failovers a looser one
would win.

The cost is real: a 5xx arriving while the client is still uploading is relayed rather than retried
even when it would have replayed fine.

`ai_failover_unreplayable_total` counts every request this excludes — both the too-large bodies and
the not-yet-known ones, since both cost the same thing: a failover declined. That number is the input to
whether covering them is worth building, and the options are not cheap: patch `BODY_BUF_LIMIT`
(fork, or upstream a config knob) or drive the retry ourselves via pingora's `Subrequest` API
(`allow_spawning_subrequest` — a full inner proxy request whose downstream is a channel, so the body
comes from our buffer with no cap, at the cost of every filter re-entering on the inner request).
Neither is worth starting before the counter says how often the limit actually bites.

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
- **The request body's `model` on `/auto`.** It is an input the gateway _overwrites_ with the
  serving candidate's id, so it determines nothing — the routing header does. A body that names a
  different model is counted on `ai_model_header_body_mismatch_total` (a client bug worth finding)
  and otherwise ignored; `requested_model` in `ai.usage` reports the catalog name, which is what was
  actually asked for.
- **Which pool key a model-routed request draws on.** The header alone selects the catalog row, and
  there is no per-tenant entitlement check on rows — any managed tenant can route to any row. Not
  price-gameable (billing uses the id the provider echoes back), but worth knowing before rows are
  added whose pool keys differ in cost or contract.
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

| Field                           | Default                           | Runtime Effect                                                                                                                                                                                             |
| ------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signing_keys`                  | _(required)_                      | Map of kid → base64 Ed25519 public key. Multiple kids enable rotation. Missing → all traffic falls through to BYO treatment.                                                                               |
| `require_signing_keys`          | `false`                           | When `true`, an empty `signing_keys` is a hard boot failure instead of silent BYO-only mode. Set on managed deployments so a typo'd/absent SSM param fails fast rather than silently serving for free.     |
| `pool_keys.<name>`              | _(from `AI_POOL_KEY_<NAME>` env)_ | Real provider API key. Missing for a provider → managed requests to that provider return 503 before any upstream connection.                                                                               |
| `provider_authorities.<name>`   | _(none)_                          | Override or add a provider's `authority` (host:port). Enables config-added providers beyond `KNOWN_PROVIDERS` with zero code change.                                                                       |
| `provider_dialects.<name>`      | `"openai"`                        | Wire dialect for a **config-added** provider (`"openai"` or `"anthropic"`, case-insensitive). No effect on a known provider (dialect fixed in code). Unrecognized value → hard boot failure.               |
| `provider_auth_schemes.<name>`  | `"bearer"`                        | Managed auth scheme for a **config-added** provider (`"bearer"`, `"x-api-key"`, or `"api-key"` — the last is Azure OpenAI's shape). No effect on a known provider. Unrecognized value → hard boot failure. |
| `snapshot_path`                 | _(unset)_                         | Path for the on-disk deny-set cache. Unset → re-scan NATS on every cold boot. Set → load from disk and enforce before NATS reconnects (edge/tunnel deployments).                                           |
| `rate_limit_rps`                | `100`                             | Per-credential request ceiling (count-min, keyed on raw key hash). `0` disables. Exceeded → 429. Checked before Ed25519 verify.                                                                            |
| `byo_rate_limit_rps`            | `1000`                            | Aggregate ceiling for all BYO traffic (single shared bucket). `0` disables. Managed traffic exempt. Exceeded → 429.                                                                                        |
| `circuit_breaker_threshold`     | `20`                              | Per-provider upstream failures (5xx / connect; **not** 429) within the window before the breaker opens. While open, requests to that provider fast-fail with 503. `0` disables.                            |
| `circuit_breaker_window_secs`   | `10`                              | Rolling window over which failures are counted (trips on a burst, not a slow trickle).                                                                                                                     |
| `circuit_breaker_reset_secs`    | `30`                              | How long the breaker stays open before admitting a half-open probe. Probe success closes it; failure reopens it.                                                                                           |
| `connect_timeout_secs`          | `10`                              | TCP connect timeout to the upstream provider. Exceeded → retry up to 2×, then 502.                                                                                                                         |
| `read_timeout_secs`             | `600`                             | Response read timeout (10 min accommodates long-running LLM streams).                                                                                                                                      |
| `write_timeout_secs`            | `60`                              | Upstream request-write timeout (sending the request to the provider).                                                                                                                                      |
| `idle_timeout_secs`             | `90`                              | Idle timeout on a pooled upstream connection before it's closed.                                                                                                                                           |
| `shutdown_grace_period_secs`    | `600`                             | SIGTERM drain window for in-flight requests (= `read_timeout_secs` so a deploy never truncates a stream). Capped by the orchestrator's stop timeout (ECS Fargate: 120s).                                   |
| `shutdown_runtime_timeout_secs` | `10`                              | Final runtime-teardown backstop after the drain window.                                                                                                                                                    |
| `nats_url`                      | `nats://localhost:4222`           | NATS server for the deny-set watcher. Unreachable → fail-open (deny-set stays empty or stale).                                                                                                             |
| `nats_creds`                    | _(unset)_                         | NATS credentials file path. Required for authenticated clusters.                                                                                                                                           |
| `listen_addr`                   | `0.0.0.0:8080`                    | Proxy listener address (client traffic).                                                                                                                                                                   |
| `provider_authorities.auto`     | _(rejected)_                      | Reserved: `auto` is the model-routed segment, and a provider of that name would shadow it. Hard boot failure.                                                                                              |
| `metrics_listen`                | `0.0.0.0:9090`                    | Internal admin/observability listener: `/metrics` (Prometheus scrape), `/livez`, `/readyz`. Separate from the client listener — not externally reachable.                                                  |

---

## Failure Modes

| Failure                                                            | What Actually Happens                                                                                                                                                                                                                                              | Recovery                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NATS unreachable at boot                                           | Deny-set starts empty (fail-open). Auth still works — keys from config.                                                                                                                                                                                            | Watcher reconnects; seeds from NATS or disk snapshot on connect.                                                                                                                                                                                                                                                         |
| NATS disconnects mid-run                                           | Last-known deny-set stays active. New deny entries not applied until reconnect.                                                                                                                                                                                    | Watcher reconnects (1s→30s exponential backoff, reset only after a watch that ran ≥30s — _connecting_ is not success, or a reachable NATS with a broken watch loops at 1 Hz forever) and resumes from the saved revision. Rescans instead when the seed found no entries, since revision 0 is not resumable — see above. |
| NATS history compacted past snapshot cursor                        | `CursorExpired` → full re-scan from current NATS state.                                                                                                                                                                                                            | After re-scan, new cursor set; delta watch resumes normally.                                                                                                                                                                                                                                                             |
| Virtual key tampered or forged                                     | Ed25519 verify fails → falls through to BYO treatment. No billing event. No error reveals which part failed.                                                                                                                                                       | Billing miss detectable downstream; no security boundary breach.                                                                                                                                                                                                                                                         |
| `signing_keys` absent (typo'd/missing SSM)                         | Default: warn + BYO-only (silently drops all managed billing + deny-set). With `require_signing_keys=true`: hard boot failure.                                                                                                                                     | Set `require_signing_keys=true` on managed deployments so the mis-deploy fails fast and visibly at boot.                                                                                                                                                                                                                 |
| Pool key missing for provider                                      | Managed request returns 503 before any upstream connection.                                                                                                                                                                                                        | Add `AI_POOL_KEY_<NAME>` env and redeploy.                                                                                                                                                                                                                                                                               |
| `provider_dialects`/`provider_auth_schemes` value unrecognized     | Hard boot failure (`GatewayError::Config`) naming the provider and the bad value.                                                                                                                                                                                  | Fix the typo — `"openai"`/`"anthropic"` and `"bearer"`/`"x-api-key"`/`"api-key"` are the only accepted values.                                                                                                                                                                                                           |
| Config-added provider's dialect misconfigured (wire doesn't match) | `usage::openai_body`/`anthropic_body` (and the stream variants) detect the other dialect's characteristic field names and return `None` instead of a zeroed `Usage`.                                                                                               | `usage_parse_errors_total` fires + a `warn!` log; fix `provider_dialects.<name>` to match the vendor's actual wire.                                                                                                                                                                                                      |
| Provider DNS fails                                                 | `upstream_peer` returns error → 502 to client.                                                                                                                                                                                                                     | TTL-cached DNS (60s) serves stale; poisoned-lock guard re-resolves on next request.                                                                                                                                                                                                                                      |
| Provider TCP connect fails                                         | `fail_to_connect` retries up to 2×, then returns 502. Counts as a circuit-breaker failure.                                                                                                                                                                         | Client SDK retries with backoff. No HTTP-status retries (Pingora-idiomatic).                                                                                                                                                                                                                                             |
| Provider brownout (sustained 5xx)                                  | After `circuit_breaker_threshold` 5xx/connect failures in the window, the breaker opens; requests fast-fail 503 (`circuit_open`) instead of stalling against the read timeout.                                                                                     | Auto: after `circuit_breaker_reset_secs` a half-open probe is admitted — success closes the breaker, failure reopens it. Per-provider, so other providers are unaffected.                                                                                                                                                |
| Provider throttles (429 storm)                                     | Relayed to the client as 429; the client's `Retry-After` backoff applies. Does **not** trip the breaker (provider is healthy).                                                                                                                                     | Backpressure via client + the rate guardrails; no gateway-side circuit action.                                                                                                                                                                                                                                           |
| Response body > 128KB before usage chunk                           | Tail compaction fires: `drain(..half)` discards first half, keeps tail. Usage extracted from retained tail.                                                                                                                                                        | No action — SSE usage is always in the final `data:` line, which always lands in the tail.                                                                                                                                                                                                                               |
| Client cancels mid-request (ESC on a slow turn)                    | Relayed as a downstream abort. **Not** counted against the provider's breaker — pingora tags it `ErrorSource::Downstream`. Was previously recorded as a provider failure, so a burst of cancellations opened the breaker and 503'd everyone.                       | None. `tests/cancellation.rs` pins both halves: aborts do not open the breaker, sustained 5xx still does.                                                                                                                                                                                                                |
| Retry replays a partially-read request body                        | `upstream_peer` resets the body-phase state each attempt, so the replayed prefix replaces rather than appends. Previously it was appended, producing a duplicated JSON fragment the provider rejected with a `400` that `logging` recorded as a breaker _success_. | None — the reset is unconditional and O(1) on the first attempt.                                                                                                                                                                                                                                                         |
| Model-routed candidate refuses the connection                      | `fail_to_connect` advances to the next candidate and pingora re-invokes `upstream_peer`; the abandoned candidate's breaker records the failure. Client sees nothing.                                                                                               | Automatic. `ai_candidate_failovers_total` counts it; the per-provider `ai_connect_retries_total` names the candidate left behind.                                                                                                                                                                                        |
| Model-routed request with every candidate down                     | Each is attempted once, then the request fails with pingora's `5xx`. Each candidate's breaker records its own failure.                                                                                                                                             | Fix whichever providers are down; `doctor`'s `model_catalog` check catches the _configuration_ case (a row with no pool-keyed candidate) at boot.                                                                                                                                                                        |
| Gateway crash mid-request                                          | In-flight request drops; client receives TCP close. No partial state written.                                                                                                                                                                                      | Client SDK retries. No DB writes in the request path — no cleanup needed.                                                                                                                                                                                                                                                |

---

## Metrics

Prometheus on the default registry, exposed at `/metrics` on `metrics_listen`.

| Metric                                | Type      | Labels               | What It Measures                                                                       |
| ------------------------------------- | --------- | -------------------- | -------------------------------------------------------------------------------------- |
| `ai_requests_total`                   | Counter   | —                    | Total admitted requests                                                                |
| `ai_rejections_total`                 | Counter   | `reason`             | Rejected requests by cause (auth, deny_spend, deny_fraud, rate_limit, circuit_open, …) |
| `ai_upstream_responses_total`         | Counter   | `provider`, `status` | Upstream responses by provider and status class                                        |
| `ai_tokens_total`                     | Counter   | `kind`               | input / output / cache_read / cache_write token counts                                 |
| `ai_ttft_seconds`                     | Histogram | `provider`           | Time to first token (50ms–30s buckets)                                                 |
| `ai_upstream_latency_seconds`         | Histogram | `provider`           | Full request latency (100ms–600s buckets)                                              |
| `ai_active_streams`                   | Gauge     | —                    | Open SSE streams                                                                       |
| `ai_requests_in_flight`               | Gauge     | —                    | All in-flight requests (streaming + non-streaming)                                     |
| `ai_deny_set_size`                    | Gauge     | —                    | Current number of denied tenants                                                       |
| `ai_nats_connected`                   | Gauge     | —                    | 1 if NATS watcher is connected, 0 otherwise                                            |
| `ai_usage_parse_errors_total`         | Counter   | —                    | Managed 2xx responses with no parseable usage (emitted as a zero-token billing row)    |
| `ai_candidate_failovers_total`        | Counter   | —                    | Model-routed requests that abandoned a candidate for the next one                      |
| `ai_model_header_body_mismatch_total` | Counter   | —                    | Model-routed requests whose routing header and body `model` disagreed (client bug)     |
| `ai_failover_body_too_large_total`    | Counter   | —                    | 5xx that could not fail over: request body exceeded the 64 KiB replay buffer           |

---

## Modules

| Module            | Role                                                                                                 | Tested         |
| ----------------- | ---------------------------------------------------------------------------------------------------- | -------------- |
| `proxy`           | `ProxyHttp` impl — request/response pipeline (request_filter through logging)                        | e2e ✓          |
| `key`             | `bai_v1` parse + Ed25519 verify + mint; keyring with multi-kid rotation support                      | unit ✓         |
| `route`           | Data-driven provider table (name / authority / auth) + dialect default + model-route re-exports      | unit ✓         |
| `peek`            | `ModelScanner` — streaming structural scan for the root-level `model`; O(1) memory                   | unit ✓         |
| `usage`           | Token extraction (OpenAI / Anthropic, body + SSE)                                                    | unit ✓         |
| `deny`            | Sparse deny-set, default-allow, reason → HTTP status                                                 | unit ✓         |
| `ratelimit`       | Two-tier guardrail: per-credential (count-min sketch, fixed memory, no GC) + global BYO (one atomic) | unit ✓         |
| `circuit_breaker` | Per-provider lock-free breaker (packed `AtomicU64`, windowed policy) — trips on 5xx/connect, not 429 | unit ✓ + e2e ✓ |
| `state`           | Keyring + resolved provider registry + watched deny-set (ArcSwap) + TTL DNS cache                    | unit ✓         |
| `store_watch`     | NATS watcher — gap-free deny-set seeding + delta watch as Pingora `BackgroundService`                | e2e ✓          |
| `config`          | Figment config; build keyring; pool keys / authorities by provider name                              | unit ✓         |
| `secret`          | Redacting, zeroize-on-drop `Secret<T>` newtype for pool keys and NATS creds                          | unit ✓         |
| `admin`           | `ServeHttp` on the metrics listener: `/livez`, `/readyz`, `/metrics`                                 | e2e ✓          |
| `metrics`         | Prometheus counter/histogram/gauge registration and update helpers                                   | compile ✓      |
| `doctor`          | Boot-time diagnostics (`beyond-ai doctor`)                                                           | compile ✓      |
| `main`            | CLI (`run` / `doctor`), rustls init, config load, Pingora server + three services bootstrap          | compile ✓      |

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
- **Model routing (`tests/model_routing.rs`):** the `/auto` route end-to-end against two mocks with
  different mounts, pool keys, and model ids. Covers primary routing, **failover on a refused
  connection** (asserting the fallback's mount, its pool key, and its spelling of the model — the key
  assertion, since forwarding the primary's key would be a credential leak rather than a failed
  request), the breaker ledger (the abandoned candidate's breaker opens while the fallback keeps
  serving), missing/unknown model → 404, BYO → 400, the routing header never reaching an upstream,
  `ai.usage` naming the candidate that served, all-candidates-down, a **256 KiB body surviving a
  failover byte-for-byte**, and provider-routed traffic being unaffected.
- **Cancellation (`tests/cancellation.rs`):** a client that gives up must not open the provider's
  breaker, and a genuinely broken provider still must. Verified non-vacuous — reverting the fix makes
  the first test fail.
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
  only vs. BYO which runs both tiers) — single-threaded/hot-cache _and_ `check_flood_*`, which
  charges 65536 distinct credentials from 1 and 16 threads over ≥ 2 window rotations, plus
  `rotate_window`, which prices the window rotation on its own.

  What the alloc numbers assert:
  | Operation            | Cost      | Allocations                  | Claim verified                   |
  | -------------------- | --------- | ---------------------------- | -------------------------------- |
  | `key/verify`         | ~23µs     | 0                            | Stack-only Ed25519 decode        |
  | `peek/ModelScanner`  | varies    | 1 (independent of body size) | O(1) memory                      |
  | `route`              | ~ns       | 0                            | —                                |
  | `deny::reason`       | ~1–8ns    | 0, flat 0→1M entries         | O(1) lookup, O(denied) memory    |
  | `ratelimit::check`   | ~39–220ns | 0                            | Fixed-memory, no per-key state   |
  | `ratelimit` rotation | ~39µs     | 0                            | Once per window, not per request |

  **Headline: `key/verify` ≈ 23µs is ~100–600× every other per-request op.** This is why the rate
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

  **The model route (`/auto`) costs nothing measurable.** Paired runs against the provider-routed
  path: `managed_json` 107.85 / 108.84 / 108.68 µs vs `auto_json` 107.19 / 108.65 / 108.67 µs — a
  mean delta of −0.3 µs, i.e. `/auto` measured marginally _faster_, which is noise. At 64 KiB the
  two are likewise level (`managed_large_body` 155.23 µs vs `auto_large_body` 155.75 µs) **despite**
  `/auto` buffering the whole body where the path-routed request streams it: a 64 KiB memcpy into a
  pre-sized `Vec` disappears into the network cost, and the `model` splice is a no-op whenever the
  candidate spells the model the way the catalog names it, which the primary candidate usually does.
  A single first run showed `auto_json` +2.26 µs and it did not survive repetition — see the
  paragraph below, which exists because of exactly that.

  **`auto_failover_latency` (~110.6 µs, +2.5 µs) measures the mechanism, not an outage.** Its dead
  primary is an unbound port, so the connect is refused instantly. In production a provider that has
  gone away usually does not refuse — it hangs, and the client pays up to `connect_timeout_secs`
  before the next candidate is tried. The candidate walk itself is microseconds; the wait is
  whatever the failed connect costs, and that is the number to quote to anyone asking what failover
  feels like.

  **Read criterion's verdict on this harness with care below ~3%.** Its p-value models within-run
  sampling noise, not run-to-run drift, and there is plenty of the latter here. Measured directly:
  three consecutive runs of _identical_ code against one saved baseline reported +2.18% ("regressed",
  p=0.00), +0.53% ("no change", p=0.40), and +1.90% ("regressed", p=0.00) — and the run with the
  _highest_ absolute time was the one that reported the smallest delta. So a single flagged run is a
  prompt to re-measure, not a finding. Treat a change as real only if it reproduces across runs, and
  ideally only with a mechanism to point at: the `ModelRouting` boxing was accepted on a +2.31% and
  +2.80% pair _plus_ a 368→432-byte struct measurement that explained why streaming was hit and
  non-streaming was not. For anything at ns scale, use the unit bench — that is what it is for.

`mise run bench` runs both.
