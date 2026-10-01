# Beyond AI Gateway — Architecture

Takes HTTP requests carrying an OpenAI- or Anthropic-dialect payload, authenticates the caller via
Ed25519 virtual key or BYO provider token, swaps in a pool key for managed traffic, relays the
request and response to the upstream provider (byte-for-byte when the inbound path and the catalog
row share an endpoint, including same-endpoint Responses; translated Chat Completions ↔ Messages ↔
Responses when they don't), and
emits a token-usage billing fact (`ai.usage`) on completion. Usage taps the upstream body; the
client sees the inbound dialect.

**Self-contained:** no `path` deps into the `beyond` repo. Depends only on crates.io + the
published `beyond-slipstream` — clones, CI-builds, and publishes anywhere.

---

## Concepts & Terminology

| Term                                       | What It Controls / Gates                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | NOT                                                                                                                                          |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Managed key** (`bai_v1.…` / `bai_v2.…`)  | Ed25519-verified identity; enables key swap, deny-set check, allowance check, and `ai.usage` billing. Reaches only `POST` generation calls: catalog paths take `POST` only (plus `GET`/`HEAD /v1/models`), and `/{provider}/…` only a generation endpoint (see Managed endpoint allowlist)                                                                                                                                                                                               | A session token or capability grant — just tenant attribution                                                                                |
| **BYO key** (anything else)                | Forwarded as-is to the provider; no swap, no billing, no deny-set                                                                                                                                                                                                                                                                                                                                                                                                                        | A lesser tier — same proxy, minus attribution and billing                                                                                    |
| **Pool key**                               | Real provider API key(s) held by the gateway; swapped in for managed traffic. A 429 walks the next unused key on the _same_ provider                                                                                                                                                                                                                                                                                                                                                     | Per-tenant — keys are per provider, shared by all managed callers                                                                            |
| **Tenant**                                 | The billing entity from the virtual key payload (`tenant_id: u64`)                                                                                                                                                                                                                                                                                                                                                                                                                       | An org, user, or namespace — an opaque integer the gateway doesn't interpret                                                                 |
| **Dialect**                                | The wire (OpenAI vs Anthropic) a request is answered on, driving usage parsing: the forwarded path's on `/{provider}/…`, the serving candidate's path's on a catalog walk; for a bare-path BYO request it's derived from the path to pick the default provider                                                                                                                                                                                                                           | The provider — one provider can serve both wires (OpenRouter `/api/v1/messages`)                                                             |
| **Provider**                               | The request's **first path segment** (`/{provider}/…`); a named row in the routing table: authority, dialect, auth scheme                                                                                                                                                                                                                                                                                                                                                                | A vendor relationship — just connection facts and auth wiring                                                                                |
| **Model route** (`/auto/…`, managed `/v1`) | Catalog row named by `x-beyond-model` if present, else the body's root `model`; provider, upstream path, and model id come from that row and the body's `model` is rewritten per attempt. Catalog miss → 404. Same-endpoint walks are a byte relay; Chat Completions ↔ Messages ↔ Responses is translated when the inbound path _or this candidate's path_ names a different one of those three. Inbound Responses with session state walks the GPT row's `/v1/responses` arm (or 400s). | Gemini. Not a per-key grant.                                                                                                                 |
| **Candidate**                              | One `(provider, upstream model id, path)` a catalog row will accept. Default walk is TTFT-ranked (in-process EWMA); `x-beyond-order` / `split` pin, `only` filters. Cannot add a provider the row does not list.                                                                                                                                                                                                                                                                         | A parallel pool — still a sequence, entered on failure. Not a cost sort.                                                                     |
| **Deny-set**                               | Sparse maps of denied `tenant_id`s and `key_id`s → reason; gates managed traffic; default-allow; tenant deny kills every key                                                                                                                                                                                                                                                                                                                                                             | An allowlist or ACL — misses are allowed, not blocked                                                                                        |
| **Allowance-set**                          | Sparse maps of exhausted `tenant_id`s and `key_id`s; remaining-ok vs exhausted; 402 **before** `upstream_peer`. Fail-closed (a retryable 503) until the watcher stores a scan/snapshot (empty = remaining-ok). v1 tokens: tenant grain only. Not a price table — the control plane writes the bit.                                                                                                                                                                                       | A price table, remaining-token counter the gateway decrements, or Redis on the miss path                                                     |
| **Tail tap**                               | Bounded 64KB window kept from the end of the response for usage extraction                                                                                                                                                                                                                                                                                                                                                                                                               | A buffer or copy — the response is relayed unbuffered; only the tail is kept                                                                 |
| **Capture-set**                            | Sparse map of `tenant_id`s with payload logging on; default-**off**; watched under its own prefix by its own watcher                                                                                                                                                                                                                                                                                                                                                                     | Retention policy — the gateway emits and forgets; the store owns TTL/erasure                                                                 |
| **Capture tap**                            | Bounded **head**-keeping copy of each body, taken pre-rewrite; relayed bytes are untouched                                                                                                                                                                                                                                                                                                                                                                                               | A buffer — nothing is withheld, so it costs memcpy, never latency                                                                            |
| **Response cache**                         | **Per-pod** exact-match store: identical managed catalog-walk request (pre-rewrite body + inbound path + `tenant_id` + effective candidate order) replays a stored 2xx on **this process**. Off unless `cache_ttl_secs > 0`. Miss is an unbuffered relay; fill is a tap. Replicas do not share entries.                                                                                                                                                                                  | Redis, semantic cache, a pool-key key, or a fleet-wide cache — none of those                                                                 |
| **Cut-short estimate**                     | A managed request the provider took but whose usage never arrived — a 2xx stream ended early, a non-stream body cut off or answered without usage (D195), a cancel before the response head — is billed an **estimate** flagged `usage_estimated`: input from the prompt text's pre-tokens, a lower bound (Anthropic keeps `message_start`'s exact count), output from the relayed delta events and text. Errs low.                                                                      | A reported count, or a way to see hidden reasoning — both estimates are blind to thinking the stream never shows                             |
| **Tenant slot**                            | One of `tenant_max_in_flight` concurrent requests a tenant may hold **on this process**; over it → 429 before the breaker and upstream. The bound on overspend while the allowance-set lags. Off by default.                                                                                                                                                                                                                                                                             | A rate limit or a quota — short fast requests never hit it; N replicas admit N × the limit                                                   |
| **Control header** (`x-beyond-*`)          | Per-request caller input: `metadata` tags, `capture` on/off, `cache` on/off, catalog `order` / `only` / `split`. Managed only; stripped before the upstream                                                                                                                                                                                                                                                                                                                              | A way to 4xx a request — unusable values are dropped and counted; an `only` that leaves no keyed candidate is the same 503 as an unkeyed row |
| **Smart router**                           | **Per-pod** EWMA of TTFT per catalog candidate. Default walk for managed `/auto` and `/v1` when `order`/`split` are absent. Probe of unmeasured arms every 8th request. Ranks **new** callers only: a caller with a live session pin keeps its provider. `smart_router = false` restores static catalog order. Two replicas can rank the same row differently.                                                                                                                           | Live Redis, cost sort, or a fleet-wide shared ranking — none of those                                                                        |
| **Snapshot**                               | On-disk deny-set cache (entries + NATS cursor) for edge/tunnel deployments. Allowance uses `{snapshot_path}.allowance`.                                                                                                                                                                                                                                                                                                                                                                  | Persistent store — a pure cache; delete it and the gateway re-scans NATS                                                                     |
| **Virtual key** (`bai_v1` / `bai_v2`)      | Ed25519-signed token: v1 is `tenant_id`+`vpc_id` (16 B); v2 adds unique `key_id` (24 B). Same keyring.                                                                                                                                                                                                                                                                                                                                                                                   | A session or auth token — stateless, no server-side lookup                                                                                   |

---

## Data Flow

### Happy Path

```
Client (stock OpenAI/Anthropic SDK)
  │
  ▼  request_filter (proxy.rs)
  │  ├─ Route: first segment → provider row (authority, dialect, auth scheme)
  │  │    `/{provider}/…` is the escape hatch (no catalog)
  │  │    …or `/auto` / managed `/v1` → x-beyond-model if present, else body's root `model`
  │  │      → catalog row → candidate list (default: TTFT rank; cold start = the row's static order)
  │  │      x-beyond-order / split pin; only filters; then the ranker (same wire; no new providers)
  │  │      no/unknown model ──────────────────────────────────► 404 (names the miss)
  │  │      inbound path's endpoint ≠ row
  │  │        Chat Completions ↔ Messages ↔ Responses ─► translate
  │  │        Responses + session state ─► GPT `/v1/responses` arm (relay) or 400 naming the field
  │  │        other mismatch ────────────────────────► 400
  │  │      BYO on `/v1` ─ dialect-default passthrough (no catalog); the forwarded BYO key picks
  │  │        its provider (x-api-key / key shape), else the path; keys for two providers → 400
  │  │      no remaining candidate holds a pool key ──────────► 503
  │  ├─ Extract key: x-api-key / api-key / x-goog-api-key / Authorization Bearer / ?key= query param
  │  │    a managed (bai_v1/v2) value in ANY of them wins: every line of a repeated header,
  │  │    every `key` param (`k%65y` too), `Bearer` with any whitespace; empty = absent
  │  ├─ Rate guardrails (BEFORE verify — keeps forged-key floods at ns cost)
  │  │    per-credential count-min  ──────────────────────────────► 429
  │  │    global BYO aggregate (managed exempt)  ─────────────────► 429
  │  ├─ Content-Length abuse guard (declared > 100 MiB) ──────────► 413
  │  ├─ Identity branch:
  │  │    bai_v1/v2.…  → Ed25519 verify → deny-set (tenant OR key_id, O(1))
  │  │    │               │                    │
  │  │    │             401 (bad sig)     402 Spend / 403 Fraud
  │  │    │                                    │
  │  │    │           allowance-set (tenant OR key_id; fail-closed if unread)
  │  │    │             remaining-ok ───────────────────────────────────
  │  │    │             exhausted ──────────────────────────────────── 402
  │  │    │             unread ───────────────────────── 503 + Retry-After
  │  │    │           pool key required ───────────────────────── 503
  │  │    └─ BYO: pass through (no verify, no deny-set, no billing)
  │  ├─ BYO key on `/auto` (managed-only route) ──────────────────► 400
  │  ├─ GET/HEAD /v1/models (after identity: a managed key must verify) ► catalog list
  │  ├─ Managed endpoint allowlist (method + path) ──────────────► 404 / 405
  │  ├─ Catalog walk: read the body to find `model` (≤ 64 KiB in hand, larger re-run
  │  │    as a `FullBody` subrequest); past 100 MiB ──────────────► 413, before any upstream
  │  ├─ Managed only: parse x-beyond-* control headers (never 4xx; bad values counted)
  │  │    order / only / split, then TTFT rank unless order/split pinned, *before* first_usable / breaker skip
  │  │    capture decision = header (wins both ways) else capture-set rule ∧ 1-in-N sample
  │  ├─ Exact-match cache (managed catalog walk, body already in hand, cache_ttl_secs > 0):
  │  │    key = pre-rewrite body + method + inbound path + tenant_id + catalog row
  │  │      + anthropic-version / anthropic-beta values + effective candidate order
  │  │    per-pod table (ai_cache_scope{kind="process"}); miss does not consult Redis
  │  │    x-beyond-cache: off / Cache-Control: no-store ────────── skip lookup and store
  │  │    hit: write stored 2xx (no upstream, no breaker, no key-walk)
  │  │    miss: unbuffered relay; fill is a tap, insert only on complete 2xx
  │  ├─ Tenant slot (managed, tenant_max_in_flight > 0): at ceiling ──► 429
  │  └─ Circuit breaker (per provider, all traffic): if OPEN ─────► 503
  │       (claims a half-open probe permit only on an actual attempt)
  │
  ▼  upstream_peer (proxy.rs)   — runs once per attempt, before any body byte
  │  Reset request-body phase state (a retry replays bytes through the body filter)
  │  Model-routed: resolve the outgoing candidate's breaker permit, then walk candidates —
  │    skip any whose breaker is OPEN, allow() the one chosen, set forward_path to that
  │    candidate's own catalog path; DNS failure advances rather than ending the request
  │  TTL-cached DNS resolve (60s, every address, 5s lookup bound, serve-stale ≤10 min;
  │    single flight: a due refresh runs once in the background, callers keep the cached answer)
  │    → HttpPeer (TLS, H2 pref, timeouts); a refused connect tries the next address
  │  DNS fail ──────────────────────────────────────────────────── 502
  │  TCP connect fail (retry 2× same peer, or next candidate) ──── 502
  │
  ▼  upstream_request_filter (proxy.rs)
  │  Managed: remove every static-key header (authorization, x-api-key, api-key,
  │    x-goog-api-key; every repeat) UNCONDITIONALLY → inject the pool key in the
  │    provider's own scheme (never provider A's key on provider B); every `key` query
  │    param, under any spelling a provider decodes to `key`, is dropped from the
  │    forwarded path (other params kept)
  │  Managed: forward only allowlisted client headers (content-type, content-length,
  │    transfer-encoding, expect, accept, user-agent, anthropic-version, anthropic-beta
  │    filtered to known-safe tokens); everything else the client sent is dropped
  │  BYO: leave auth header unchanged (and every other client header)
  │  Managed: accept-encoding: identity (the gateway parses the body; gzip billed 0 tokens)
  │  Strip every x-beyond-* header, any name, every route, managed or BYO (ours)
  │  Set Host; path: verbatim for /{provider} (prefix stripped), or the candidate's
  │    own catalog path for a catalog walk (`/auto`, managed `/v1`)
  │  OpenRouter + managed only: dashboard-attribution headers (HTTP-Referer, X-OpenRouter-*)
  │
  ▼  request_body_filter (proxy.rs)  — streamed through, except where a rewrite needs the whole body
  │  Enforce running size cap (chunked-safe) ──────────────────── 413
  │  Capturing: copy chunk into the head-bounded request buffer — PRE-rewrite, so the
  │    capture is what the client sent, not what we spliced (never withheld)
  │  Managed + streamed: feed chunks → ModelScanner (peek.rs), root-level `model`, O(1) mem
  │    (BYO skips it — `model` is only ever read on the managed billing path)
  │  Injection-eligible (managed + OpenAI dialect + path suffix /chat/completions):
  │    buffer full body → ONE fused walk (peek::scan_buffered) yielding `model`, its byte
  │    span, and the splice offset → inject stream_options.include_usage → re-frame chunked
  │    (a client-sent stream_options is rewritten to include_usage:true, never left off)
  │  Model-routed: a client body with two root `model` keys (any spelling) ─────── 400
  │  Model-routed: same buffer, and `model` is spliced to the serving candidate's own id
  │    (rewrite first — the injection offset precedes the value, so it cannot move)
  │  Wire-mismatched catalog walk (Chat Completions ↔ Messages ↔ Responses): map the
  │    buffered JSON *before* the model splice. OpenAI→Anthropic does not inject
  │    `stream_options`. Anthropic→OpenAI (and Responses→Chat Completions) injects
  │    `include_usage` on the translated Chat Completions body if it streams.
  │    Inbound Responses, same-endpoint: byte relay (session fields pass through).
  │    Session fields on a non-Responses candidate: skip or 400.
  │  Stream-only candidate (`catalog::stream_only`), client not streaming: splice
  │    `stream: true` (+ `include_usage`) first; the answer is assembled (below)
  │  Chat Completions candidate: a same-wire `developer` message becomes `system` off
  │    OpenAI / OpenRouter (`catalog::reads_developer_role`); a candidate whose thinking
  │    breaks tools (`catalog::tool_thinking`) gets them with `reasoning.enabled: false`
  │
  ▼  Provider upstream  (OpenAI / Anthropic / Groq / DeepSeek / …)
  │
  ▼  response_filter (proxy.rs)
  │  Record TTFT; detect streaming (Content-Type: text/event-stream)
  │  Count upstream response by provider + status class
  │  Drop any upstream x-beyond-* header, then set x-beyond-request-id, x-beyond-provider,
  │    x-beyond-upstream-model (catalog walk)
  │  Managed: drop any header whose value carries the pool key this attempt sent; a
  │    status >= 400 arms the body scrub below (a JSON one also drops Content-Length: it is
  │    held whole for the account-remedy rewrite)
  │  Translate walk, or a same-endpoint catalog error (re-encoded into the client's
  │    envelope when it is another vendor's shape): drop Content-Length (body length will change)
  │
  ▼  response_body_filter (proxy.rs)  — response relayed chunk-by-chunk; SSE is never fully buffered
  │  Managed >= 400, first (`Redact`): overwrite the pool key with `[redacted]***` (same
  │    length), holding back a key-sized tail per chunk to catch a split key. A JSON error
  │    (<= 64 KiB) is held whole instead and, once masked, has any provider-account remedy
  │    rewritten (`remedy::neutralize`, D174). Everything below (translation, capture, cache,
  │    usage tail) sees the scrubbed bytes. The key searcher is built once per pool key at boot
  │    (`PoolAuth::finder`), not per response
  │  Translate path: convert SSE event-by-event into the inbound dialect (do not wait for `[DONE]`
  │    before forwarding deltas). Non-stream: map the JSON object, including error envelopes.
  │    Assembled walk (stream-only candidate, non-stream client): the stream is read by a quiet
  │    `SseBridge::assembling` and the client's JSON body is written once, at end of stream
  │    Both held buffers (the non-stream body; one not-yet-terminated SSE event) are capped at
  │    MAX_TRANSLATE_BUFFER (32 MiB) — past it the response is aborted
  │    (ai_rejections_total{reason="response_too_large"}). A stream as a whole is never capped.
  │  Managed only: feed *upstream* chunks → ModelScanner::for_response → billed model
  │    (accepts Anthropic's nested message.model, so it stops in the first chunk for both dialects)
  │  Append *upstream* bytes to bounded 64KB tail (copy_within compaction once tail > 128KB)
  │  Anthropic SSE only: also keep a bounded 8KB head — message_start carries input + cache
  │    tokens and would otherwise be compacted out of the tail
  │  Capturing: copy chunk into the head-bounded response buffer (same passive-tap contract
  │    as the usage tail; opposite end, because meaning is at the front). Translate walk: client bytes.
  │  Cache fill (catalog-walk miss): copy chunk into a capped tap — never withheld. Insert
  │    only on a complete 2xx; client abort, 4xx/5xx, truncation → do not store.
  │    On a translate walk the tap is the *client* bytes (post-translate).
  │
  ▼  logging (proxy.rs)
     Parse usage from tail (by dialect + streaming flag)
     Managed 2xx stream cut short before its usage block (client cancel / upstream death), or a
       non-stream 2xx without usage that is not an error object (D195):
       estimate the missing side from the request tally + relayed events → usage_estimated
     Emit ai.usage fact: tenant, vpc, key_id, model, requested_model, routed_model, price_model,
       token counts + usage_wire + reasoning / 1h-cache-write / server-tool / service-tier
       breakouts, upstream_status + outcome, + x-beyond-metadata tags (managed only) → blocking
       stdout, lossless. `provider` is absent when no provider was called
     Cache hit: same row with `cache_hit` and the stored tokens; no parse, no upstream latency
     Capturing: emit ai.payload (both bodies, truncation + completeness flags), correlated by
       request_id → bounded queue, DROPPED on overflow so a stalled sink can't backpressure
     Record circuit-breaker outcome, only if one is still owed (breaker_pending): 5xx / upstream
       failure → failure; no head and no provider outcome (client abort, stalled upload)
       → permit released (`CircuitBreaker::release`), never a success
     Decrement requests_in_flight gauge; release the tenant slot
```

### Background: Control-Plane Watchers

Three sparse per-tenant sets, watched independently. The seed → watch → batch-apply → reconnect loop
is written **once**, over the `WatchedSet` trait, and instantiated per set — that loop carries the
non-obvious correctness properties (revision-0 resume trap, scan→subscribe race, batched `rcu`,
backoff crediting), and a second copy would be a standing invitation for the sets to drift.

```
NATS (blackhole.*)     NATS (allowance.*)      NATS (aicapture.*)
  │                      │                       │
  ▼  WatcherService<Deny>│  WatcherService<Allowance>
  │                      │  WatcherService<Capture>
  │  seed snapshot/scan  │  seed snapshot/scan    │  scan only
  │                      │  (empty scan = ready)  │
  ▼  ArcSwap<DenySet>    ▼  ArcSwap<AllowanceSet> ▼  ArcSwap<CaptureSet>
     fail-open                fail-closed until        fail-open (off)
                              first successful read
```

**One service per set, hence one NATS connection per set.** The extra connections buy independent
failure domains: a capture-set scan that keeps failing backs off on its own schedule and cannot
slow, stall, or reseed deny or allowance enforcement.

**Deny and allowance get on-disk snapshots** (`snapshot_path` and `{snapshot_path}.allowance`). The
snapshot exists so _enforcement_ survives a cold start before NATS reconnects. Capture is not
enforcement — "captured nothing for the first few seconds after a restart" is a non-event — so
`Capture::snapshot_path` returns `None`.

Deny is **fail-open** on an unread store (empty map = allow). Allowance is **fail-closed** until
`from_entries` has run (even on an empty scan): managed traffic gets a retryable `503` with
`ai_rejections_total{reason="allowance_unavailable"}`. After seed, a NATS blip keeps the last-known
set the same way deny does. Auth, signing keys, and pool keys stay in boot config.
`/readyz` is **503** (`"status":"not_ready"`, with the reason) until the allowance-set is seeded
(`ai_allowance_ready` = 1), so the load balancer does not send traffic to a pod that refuses every
managed request; a BYO-only deployment (no `signing_keys`) is exempt. Once seeded the set never goes
unready, and a later NATS outage leaves `/readyz` 200 (body `degraded` while the deny-set watcher is
disconnected).

---

## Core Mechanism

### Routing (`route.rs`)

Providers are **data rows**, not code paths. `KNOWN_PROVIDERS` in `route.rs` lists 12 built-in
providers (openai, anthropic, openrouter, fireworks, groq, deepseek, together, cerebras, mistral,
xai, openai-codex, bedrock); each row carries its authority (host:port), dialect (OpenAI-wire vs
Anthropic-wire), and auth scheme (`Bearer`, `x-api-key`, or `api-key`). The `provider_authorities`
config key adds or overrides rows at boot with zero code change; `provider_dialects`/
`provider_auth_schemes` set the dialect/auth scheme for a **config-added** provider (default
OpenAI/Bearer for backward compatibility — see Configuration). A known provider's dialect/scheme is
always fixed in `KNOWN_PROVIDERS`, never overridable from config. This is how Azure OpenAI is
supported: its per-resource host isn't knowable at compile time, so it's always config-added
(`provider_authorities.azure = "..."` + `provider_auth_schemes.azure = "api-key"`), never a
`KNOWN_PROVIDERS` row — see `config.example.toml`. Bedrock is the opposite case: it has a real
default host (`bedrock-runtime.us-east-1.amazonaws.com`) and a static API key, so it _is_ a
built-in row; override the region with `provider_authorities.bedrock` if you need another one.

The routing rule: **first path segment = provider name**. `/groq/openai/v1/chat/completions` routes
to Groq and forwards `/openai/v1/chat/completions` verbatim. A bare path that is _exactly_ `/v1` or
starts with `/v1/` (boundary-checked — `route::is_default_prefix`, not a raw string-prefix test) is
the drop-in default: BYO dialect-picks OpenAI or Anthropic; a **managed** request there is a catalog
walk (see below). The BYO pick follows the credential the request will forward
(`byo_credential_dialect`), reading **every** value: each `x-api-key` line (Anthropic's header, so
Anthropic unless the key's shape says otherwise) and each `Authorization: Bearer` token, by shape
(`sk-ant-…` Anthropic, any other `sk-…` OpenAI; an opaque token says nothing). Managed and empty
values cast no vote. With no vote the path picks: Anthropic for `/v1/messages…`, else OpenAI.
Picking by path alone sent an Anthropic SDK's `/v1/files` call to OpenAI with its `sk-ant-…` key
attached (D30); picking Anthropic for any stray `x-api-key` sent a `Bearer sk-proj-…` beside it to
Anthropic, and only the first `x-api-key` line was read (D82). BYO keys for **two** providers on one
request are a 400 (every credential is forwarded, so either provider would receive the other's
key). A BYO `x-api-key` on a gateway with no Anthropic provider configured is a 404, never an OpenAI
call. A lookalike like Google Gemini's `/v1beta/…` does **not** qualify and 404s as an
unknown provider instead of being silently absorbed into the OpenAI default. Unknown segment → 404.

### Model routing (`/auto`, managed `/v1`, `providers::catalog`)

One reserved first segment — and the managed bare `/v1` default — routes by **model** instead of
provider. `/auto/…` and managed `/v1/chat/completions` / `/v1/messages` take the canonical model
name from the `x-beyond-model` header if present, else the body's root `model`, resolve it in the
catalog to an ordered list of candidate providers, and try them in order. The catalog **is** the
allowlist: unknown or missing model → 404, with a message that names the miss (`model "…" is not
in the catalog` vs `missing model: …`). There is no parallel grant set. A candidate's own
`upstream_model` spelling (OpenRouter's `anthropic/claude-opus-4.8`, Bedrock's inference-profile
id) is an alias for the row — those are the ids we already rewrite _to_.

The default walk is TTFT-ranked (`smart.rs`): **this process's** EWMA per catalog candidate, measured
from `attempt_start` the same way `ai_ttft_seconds` is. Cold start (no samples) is the row's static
order. A connect failure, a 5xx, or a managed walk's refusal (401/402/403) takes a penalty
floor so a fast error does not outrank a slower 2xx; a 429 is a real answer. A 2xx's sample waits
for the body's first bytes: a `200` whose body is an error object (OpenRouter's error-in-200: a root
`error` key, or Anthropic's `"type":"error"`) or an SSE stream whose first event is an error counts
as a failure, not a fast healthy sample (`settle_health`, at most 1 KiB read). A candidate whose **latest** attempt failed ranks behind every other
candidate — unmeasured ones included — until it answers again or its sample goes stale (30s). That
is what turns a client's own retry into a failover; see "Status-based failover, and where it stops". Unmeasured arms stay failover until a deterministic probe (every 8th
request, skipping seq `0`) promotes the first one that can be dispatched: an arm with no pool key
here (Bedrock on a deployment without it) never gets a sample, so a probe that could pick it would
pick it every time, `upstream_peer` would skip it, and a keyed arm behind it would never be
measured (D119). A sample older than 30s is treated as unmeasured so a
recovered arm is retried. Ranking reads the monotonic clock once per request and reuses that
instant for every candidate's staleness check. `smart_router = false`
restores static catalog order. Samples never leave the pod —
`ai_smart_rank_scope{kind="process"}=1` is the honesty metric; this is not fleet-wide smart
routing. `x-beyond-split` is the only cross-replica pin (hash of the request counter).

**Session pins.** Ranking decides where a _new_ caller goes; a pin keeps an existing one there.
Provider prompt caches are per provider, so re-ranking every request moved agent loops between
Anthropic and Bedrock and re-bought the whole prefix on each move (a Claude cache write is 1.25×
input against 0.1× for a read), and the every-8th probe landed on whoever drew the seed, usually
someone mid-session. After a 2xx that carried an answer (not an error-in-200, see above) on a
candidate walk, `(tenant_id, vpc_id, key_id)` plus the catalog row is pinned to the candidate that
served. While the pin is live the walk puts that
candidate first, keeps the rest in EWMA order as failover, and never probes. One virtual key is one
app, and an app's sessions share their system prompt and tools, so one pin per key per model is
the grain the provider cache wants. A pin yields when its candidate's latest attempt failed (the
walk fails over, and the next 2xx re-pins), after 300s without a 2xx (the provider cache has
expired anyway), and after 1h so a pin taken during an outage drifts back to the ranked primary.
An open breaker or an unkeyed candidate needs no check: `upstream_peer` skips it and the 2xx that
follows re-pins. The table is 16384 packed `AtomicU64`s per pod, direct-mapped by hash: a collision
overwrites and costs one re-rank. Per pod like the EWMA; with a healthy primary two pods rank a new
caller the same way. `ai_session_pinned_total` counts walks a pin decided. `order` / `split` walks skip ranking and so
skip the pin; an `only` walk may follow the key's pin. Neither writes one: a walk the caller shaped
says nothing about where the key's other requests should go, and one debug header must not route
them for an hour.

A managed request may also permute that list with headers, still on the same wire, without adding a
provider the row does not already name (`ProviderSpec::name` on that row). Parsed in `control.rs`,
stripped before the upstream, never a 4xx. `order` and `split` **pin** (the ranker does not run);
`only` filters, then ranking still applies:

| Header           | Value                     | Walk                                                                                                                                   |
| ---------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `x-beyond-order` | `bedrock,anthropic`       | those providers first (stable as written), then any remaining row candidates                                                           |
| `x-beyond-only`  | `bedrock,openrouter`      | drop anyone not named                                                                                                                  |
| `x-beyond-split` | `anthropic=70,bedrock=30` | pick the primary with those weights (hash of the request counter, not `rand` per replica); leftover stay failover in the current order |

Unknown names are dropped. Unparseable values are dropped, counted on
`ai_control_header_errors_total`, and the request uses the unpinned walk (TTFT rank, or catalog
order when the ranker is off). If nothing usable remains (an `only` of an unkeyed or off-row name,
or every remaining candidate unkeyed) → 503, the same as no pool-keyed candidate. Applied
**before** `first_usable` / breaker skip, so failover, breakers, and the 429 key-walk see the
permuted sequence and otherwise behave as they do today.

Not in this surface: cost sort, weighted load-balance across keys, `MODEL_ROUTES` edits, or parsing
Vercel `providerOptions` from the body.

`GET /v1/models` (and `HEAD`) lists the catalog in OpenAI list shape. The Anthropic SDK reads it
too: each row has `display_name`, and the list has `has_more: false`. Each row adds:

| Field                                                      | From                                                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------------------- |
| `wire` (`openai` / `anthropic`)                            | the row; it tells a caller which SDK matches                              |
| `context_window`, `max_output_tokens`                      | `ModelCard`; `max_output_tokens` is `null` on embeddings                  |
| `max_output_published`                                     | `ModelCard`; `false` when `max_output_tokens` is the 32,768 placeholder   |
| `input_modalities` (`text` `image` `file` `audio` `video`) | `ModelCard`                                                               |
| `output_modalities` (`text` / `embeddings`)                | the row's endpoint                                                        |
| `capabilities` (`tools` `reasoning` `structured_outputs`)  | `ModelCard`                                                               |
| `endpoints`                                                | all three generation paths (translation serves each), or `/v1/embeddings` |
| `pricing` (`input` `output` `cache_read` `cache_write`)    | `ListPrice`, USD per million tokens (`pricing_unit` says so once)         |

`providers::catalog::ModelCard` states the limits **every** candidate of the row enforces, so a
request sized from the card is accepted wherever the walk lands: the primary vendor's
`context_window` and `max_output_tokens`, lowered where a candidate enforces less. OpenAI's GPT-5
family counts output inside the published window and caps input at the window less the max
output, so those cards list 272,000 (of 400,000) or 922,000 (of 1,050,000); a failover host with a
smaller window sets the row's, as does a primary that refuses below its vendor's figure (Together's
Qwen3.8 2.4T A95B, 1,010,000). One figure per row,
not per candidate: the gateway counts no prompt tokens, so it could not choose a candidate by
window anyway. The card lists the input kinds and capabilities the vendor lists **and every
candidate serves on the endpoint it is reached on**: `grok-4.20-multi-agent` lists no tools (xAI
gates its client-side tools behind beta access), `gpt-4` lists the tools every candidate calls
though its model page omits them. A bit that one failover candidate refuses but the walk can steer
around stays on the card, and a request using it skips that candidate (below): structured outputs
on Bedrock and on OpenRouter's `z-ai/glm-5.2`, file input on OpenRouter's `x-ai/grok-build-0.1`. A
bit no candidate honors is dropped: Kimi K2.6, Kimi K2.7 Code and Qwen3.6 Plus list no structured
outputs, because their hosts accept a JSON schema and answer outside it (OpenRouter's per-endpoint
`supported_parameters` claims otherwise, so it is not taken as ground truth). Grok rows list file input because
every grok row reaches xAI over `/v1/responses`, where xAI reads PDFs (its Chat Completions answers
400 "File content is not supported on /v1/chat/completions"). Where the vendor publishes no max output, OpenRouter's 0.9x /
0.8x-of-window filler is replaced by `UNPUBLISHED_MAX_OUTPUT` (32,768), a conservative figure and
not a vendor limit: the row lists it with `max_output_published: false`, and the gateway never
enforces it. Prices, `created` and `owned_by` follow the model's maker (OpenAI's and xAI's
own `/v1/models` listings; Moonshot's and Z.ai's price tables for their OpenRouter-only rows),
never OpenRouter's listing; facts no vendor publishes, and open-weight rows whose maker sells no
API, still come from OpenRouter's public card. Each checked value is recorded with its source URL
in `verify/catalog_truth.toml`, with any lower served limit beside it (`input_limit` /
`output_limit` and their evidence), and `catalog_matches_vendor_truth` /
`capability_bits_match_vendor_truth` hold the table to it. The body is built once and cached. It is
served after identity and before the body peek, so an empty GET is not a missing-model 404.

**The card holds the request.** A header-won catalog walk normally relays a small body without
reading it first; on a row where the body decides something (`route::walk_reads_body`: a card
without image input, or a candidate that cannot honor an advertised capability) it reads the whole
body before choosing, as a headerless walk always does. Then:

- An image part (Chat `image_url`, Messages `image`, Responses `input_image`) on a row whose card
  lists no image input is a 400 naming the row (`ai_rejections_total{reason="modality"}`), before
  any upstream: o3-mini would ignore the image and bill an answer about nothing, gpt-4 would
  answer 500. PDFs on a row without file input are not refused: OpenRouter extracts a PDF's text
  for most models.
- A body asking for a JSON-schema output (`response_format` / `output_config.format` /
  `text.format`) leaves Amazon Bedrock out of the walk (`providers::catalog::serves_structured_outputs`):
  Bedrock's Messages surface answers `output_config.format` with a 400 (Opus 4.8) or a 404 (Haiku
  4.5). So is a candidate that accepts the schema but does not hold its answer to it
  (`REFUSES_STRUCTURED_OUTPUTS`: OpenRouter's `z-ai/glm-5.2`, whose hosts include one that answers
  `{\n{\n  "answer": 391\n}`). It is left out of the order, failover and TTFT ranking alike, unless
  nothing else is usable (an `x-beyond-only: bedrock`), when that provider's own answer is the
  client's.
- A body carrying a file part (Chat `file`, Messages `document`, Responses `input_file`) leaves a
  candidate that reads none out of the walk the same way (`providers::catalog::serves_file_input`):
  OpenRouter's `x-ai/grok-build-0.1` answers a PDF with 404 "No endpoints found that support file
  input" (its other grok ids read the same PDF). Both checks are one mask, `route::unserved`.

A large body reaches the same checks in `relay_full_body`, which holds it whole.

A stock OpenAI or Anthropic SDK pointed at `/v1` with `model` in the JSON body is `/auto` without
the header. Same-wire failover is a byte relay — the gateway rewrites ids, not API shapes, across
candidates in a row. When the inbound path names a different Chat Completions / Messages /
Responses endpoint (`POST /v1/chat/completions` with a Claude row, `/v1/messages` with a GPT row,
or `/v1/responses` with either) the gateway **translates** so the stock SDK completes — except
inbound `/v1/responses` on a row with a Responses arm. GPT rows list a parallel OpenAI
`/v1/responses` arm, and every Responses request on such a row walks it, `store: false` one-shots
included: a **byte relay** so `store`, `previous_response_id`, `include`, `truncation`, and
Responses-only tools (Codex's `namespace` groups and `custom` grammars) pass through. A Responses
5xx may walk another Responses candidate; it never walks onto Chat Completions/Messages. Rows with
no Responses arm (Claude, DeepSeek, …) translate a one-shot and have no OpenAI store: an
**omitted** `store` there is the stock `responses.create()` call and translates as a one-shot, while
an explicit `store: true`, a `previous_response_id`, a `conversation` or an `item_reference` that
stands for an earlier turn is a **400** naming the field, not a hollow Messages call
(`translate::responses_session_field`; an `item_reference` inside a tool step is dropped instead,
see [Stored-response references](#stored-response-references-item_reference)). A `conversation` (the Conversations API) is the same OpenAI-held
history `previous_response_id` points into, so it is session state too: relayed on a row's
Responses arm, refused elsewhere, never translated with the history dropped (D128). Grok rows have no Responses arm either (xAI's store is not
OpenAI's, and no failover shares it), but their xAI candidate is `/v1/responses`: a Responses
one-shot is relayed there as sent, except that it goes as `store: false`
(`translate::store_false`), because xAI stores every response for 30 days unless told not to (its
`store` defaults to true). So an `item_reference` points at nothing xAI kept (xAI answered it 422
"unknown item type"): one for an earlier turn is refused before xAI like the other session state,
and a tool step's is cut from the body (`translate::strip_item_references`, D175). Usage/billing
still parse the upstream body/SSE;
`ai.usage.model` is what the provider echoed. Same-wire Responses (`/{provider}/v1/responses`)
stays a byte relay. `/{provider}/…` never translates.

**Stream-only candidates.** Together serves Qwen3.6 Plus, Qwen3.7 Plus, Qwen3.7 Max and Qwen3.8
Flash only as streams: a request without `"stream": true` is a 400 `streaming_required` (D147).
`providers::catalog::stream_only` names those candidates. When one serves an attempt whose client
did not ask for a stream, the body goes out with `"stream":true,"stream_options":{"include_usage":true}`
spliced first and any `stream` / `stream_options` the client sent cut out by span
(`translate::force_stream`; onto a Messages path only `stream`). Every other byte is unchanged.
The streamed answer is
assembled into the client's ordinary JSON body: `TranslateState::assemble` gives the attempt a
`SseBridge::assembling` bridge. That is the bridge a Responses client's stream already uses, which
builds the whole response for `response.completed`, but it writes no event. At the upstream's end
of stream, `assembled` maps that response like any non-stream Responses body (or encodes it as is
for a Responses client). There is one aggregator, not a second one per wire. The client gets
`content-type: application/json` and nothing until the upstream ends, as with a non-stream answer.
The decision is per attempt, so a failover candidate that answers non-streaming requests
(OpenRouter's copy of each row) gets the client's own body. Billing reads the upstream stream:
`ai.usage` has one row with the usage chunk's exact tokens, and its `stream` is `true` because that
is what the provider served and billed. A stream that ends without a finish reason or usage
reaches the client as the `stream_truncated` error in its envelope, and the row is the cut-short
estimate. The status line is already the upstream's 200 by then, as with an error-in-200 from a
non-streaming upstream. The cache stores the assembled JSON (`application/json`) under the
client's non-stream key, so a hit replays JSON. A streaming client on these rows is relayed as
before.

**Thinking and tools on Alibaba's Qwen backend.** Qwen3.7 Plus and Qwen3.8 Flash, at Together and
at OpenRouter (whose only host is Alibaba), think by default and refuse a forced `tool_choice`
(`"required"` or a named function) while thinking: 400 "The tool_choice parameter does not support
being set to required or object in thinking mode" (D172). Thinking, Qwen3.7 Plus also writes about
one tool call in eight as content text ("call\n{\"name\": ...}") with no `tool_calls`, at both
hosts; never with thinking off (D171). Both candidates of each row behave the same, so skipping one
cannot help; instead the body is adapted per attempt (`providers::catalog::tool_thinking`,
`translate::thinking_off_for_tools`): a forced tool on either row, and any request offering tools on
Qwen3.7 Plus whose client asked for no reasoning, goes with every root `reasoning` /
`reasoning_effort` cut out by span and `"reasoning":{"enabled":false}` spliced first (Together's
documented switch for hybrid models, and OpenRouter's unified one). A Qwen3.7 Plus client that did
ask for reasoning gets it, with the risk. Applied after translation, so a Messages `tool_choice`
`any` or a named tool is covered too.

**`developer` off OpenAI.** A same-wire Chat Completions `developer` message is sent as `system` to
any host but OpenAI and OpenRouter (`providers::catalog::reads_developer_role`,
`translate::developer_as_system`: only the role values change). For a model that is not OpenAI's
the two are one role; Together's Qwen backend refuses `developer` (400 "developer is not one of
['system', 'assistant', 'user', 'tool', 'function']") and its Kimi K3, DeepSeek V4 Pro and GLM 5.2
accept it but did not follow it, while OpenRouter maps it per upstream itself (D173).

**Tool-count limits.** OpenAI Chat Completions takes at most 128 tools (400
`array_above_max_length` "Expected an array with maximum length 128"); OpenAI's Responses API took
600 in TOOL-1, xAI documents 350 per request on Responses, and Anthropic publishes no count limit
(600 tested). A Messages client offering more than 128 tools on a row whose primary is Chat
Completions and that has a Responses arm (the GPT rows before 5.4) walks that arm instead,
translated onto `/v1/responses` (`route::tools_need_responses_arm`), so Claude Code with a few MCP
servers works on `gpt-5.1`. That walk has no OpenRouter failover: the arm is OpenAI only. The
count is read from the body before
the walk: a header-won Messages walk on such a row reads the body first (`route::walk_reads_tools`),
and a large one is counted in `relay_full_body`. A Chat Completions client keeps its own endpoint,
so above 128 it gets OpenAI's own 400 naming the limit, as it would calling OpenAI.

**Chat Completions streams from vendors other than OpenAI** are the one same-endpoint walk that
is not a pure byte relay. A stock SDK accumulates every string in a delta except `index` and
`type`. OpenRouter repeats `delta.role` on every chunk and `format` on every `reasoning_details`
entry, so openai-python's `.stream()` built a role of `"assistantassistant…"` for the next turn to
send back. These streams go through `SseBridge` in relay mode: an event is forwarded byte for byte
unless it repeats an identity field, which `translate::ChatIdentity` drops. The identity fields are
`role`, a reasoning entry's `id`/`format`, and a tool call's `id`/name, each tracked per choice and
per entry. OpenAI's own streams stay a zero-copy relay.

**Embeddings rows.** `text-embedding-3-small` and `-large` are catalog rows whose candidates are
embeddings paths (OpenAI `/v1/embeddings`, then OpenRouter `/api/v1/embeddings`), so a stock
`client.embeddings.create` on managed `/v1` walks, fails over, and bills input tokens like any other
row. A batch past 64 KiB (openai-python puts `input` before `model`) is read in full and re-run
(see the peek below), which is also what lets it fail over on a 5xx or walk keys on a 429.
The row's endpoint is `Endpoint::of_row`: `Embeddings` when its primary's
path is, else what `wire` says.

**Managed endpoint allowlist.** A managed key spends Beyond's shared pool key, so it reaches only
metered generation calls. On `/v1` and `/auto` the method must be `POST` (`GET`/`HEAD /v1/models` is
served before this check). On `/{provider}/…` the method must be `POST` and the forwarded path, query
string excluded, must end in a generation endpoint: `/chat/completions`, `/messages`, `/responses`,
`/embeddings`, `/messages/count_tokens`, `/responses/input_tokens` or `/responses/compact`. The
suffix match covers every mount prefix (`/api/v1`, `/openai/v1`, `/inference/v1`, `/anthropic/v1`,
`/backend-api/codex`). Everything else is refused before any upstream contact: 404 for an endpoint
outside the list, whatever the method, and 405 for a wrong method on an allowed endpoint or a catalog
path. Each refusal is a JSON error with `x-beyond-request-id`, counted as
`ai_rejections_total{reason="managed_endpoint"}`. Without this, one tenant could list, read or delete
files, stored responses and batches that another tenant created through the same pool key, and run
unmetered batches, fine-tuning and images. BYO keys belong to the caller and are not checked.

A managed request carrying `Upgrade` (a WebSocket: OpenAI Realtime, Codex) is a 400 on any path,
checked first and counted under the same `managed_endpoint` reason. An upgraded connection is an
opaque relay that no usage tap can meter, so it is refused by name rather than left to the path list.

**Which paths name an endpoint.** `route::implied_endpoint` is an exact table:
`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/embeddings` (under `/auto` the `/v1`
is optional; a trailing slash is ignored). Bare `/v1` and `/auto` name none and relay onto the row's
primary path. Everything else is a **400** naming the row's endpoint: embeddings against a
generation row or the reverse (translation never involves embeddings), any other API path
(`/v1/moderations`, …), and stateful sub-resources such as `/v1/responses/{id}`. Sub-resources
used to match their parent by prefix and were forwarded to the candidate's generation path, where
they ran and billed as a generation; `/auto/embeddings` and `/auto/responses` used to skip the
check entirely.

**Sub-resources** (`route::SubResource`, also an exact table) are a provider's own API under its
generation endpoint, so they are forwarded, never translated:

| Path                         | Walks                                  | Billed                    |
| ---------------------------- | -------------------------------------- | ------------------------- |
| `/v1/messages/count_tokens`  | Anthropic candidates on `/v1/messages` | no — no ai.usage row      |
| `/v1/responses/input_tokens` | OpenAI candidates on `/v1/responses`   | no — no ai.usage row      |
| `/v1/responses/compact`      | OpenAI candidates on `/v1/responses`   | yes — its top-level usage |

The walk keeps only the candidates that serve the path (OpenAI's `/v1/responses` is in a GPT row's
Responses arm, or first in a Responses-first row's candidates), appends the suffix to the serving
candidate's path, and keeps failover and key rotation. It skips the TTFT ranker and pins: a token
count answers in a fraction of a generation's time and would skew both. A row with no serving
candidate is a 400 naming the missing provider (`count_tokens` on a GPT row, `compact` on a Claude
row). OpenRouter candidates are never used for them.

The billing column holds on every route, not only the catalog walk: a `/{provider}` request whose
forwarded path ends in one of these suffixes (`route::SubResource::of_forward_path`) is billed the
same way, so a provider-routed token count writes no ai.usage row and does not count toward
`ai_usage_parse_errors_total`.

v1 mapping is lossy on extras a stock SDK does not need for a tool loop: Responses-only
fields (`store`, `previous_response_id`, `include`, `truncation`, …) are dropped when leaving
Responses; they are **not** dropped when the walk stays on `/v1/responses`. Images convert both
ways: base64 data URIs ↔ Anthropic `base64` sources, and `http(s)` URLs ↔ Anthropic `url` sources
(passed through unchanged between Chat Completions and Responses). The gateway fetches nothing —
each upstream downloads the URL itself. Amazon Bedrock rejects `url` sources, so a URL image that
fails over onto a Bedrock candidate gets Bedrock's 400 rather than an answer about a picture the
model never saw (the walk leaves Bedrock out only for structured outputs, above). `thinking` / `redacted_thinking` blocks, `cache_control`,
`parallel_tool_calls: false` ↔ `tool_choice.disable_parallel_tool_use`, and `user` ↔
`metadata.user_id` pass both ways so an agent workload round-trips. Onto Messages, thinking crosses
only as a block Anthropic can verify — signed, or redacted — from the gateway's `thinking` array,
thinking content parts, or OpenRouter's `reasoning_details` (a turn a Chat client got relayed from
OpenRouter on failover). Bare `reasoning_content` / `reasoning` text is never signed and never
becomes a block: an unsigned block is a 400 on every later turn ("thinking.signature: Field
required"), while a turn without its thinking is accepted (measured on Haiku 4.5 and Sonnet 4.6).
Onto a Claude model behind Chat Completions (OpenRouter's `anthropic/…`), signed blocks also ride
`reasoning_details` (`reasoning.text` with its signature, `reasoning.encrypted` for a redacted
block, `format: "anthropic-claude-v1"`): OpenRouter replays nothing else, and older Claude models
400 a tool turn without its thinking ("a final `assistant` message must start with a thinking
block", measured on claude-sonnet-4). A Responses client echoes the `reasoning` items the gateway
minted (`rs_gw…` id, `encrypted_content` = `rs_gw:` + the Anthropic signature; the prefix still
marks an item a client replays without its id, as Codex and the Agents SDK do); onto a Claude
upstream they become that turn's signed thinking again, while OpenAI's own reasoning items stay
dropped. The reverse holds on a same-wire Responses walk: a catalog walk that relays a Responses body
to an OpenAI Responses upstream (a Responses-first row, a GPT row's Responses arm, a mixed-row
failover) first strips the gateway's own reasoning items, whose id and Anthropic signature OpenAI
rejects; one `memmem` for `rs_gw` keeps every other body a byte relay. The items are cut out by
span (`peek::remove_items`), so every other byte of a stripped body (key order, a strict schema's
property order, spacing) is still the client's, and a prompt-cache prefix still matches. Budget-thinking Claude (before 4.6) 400s a tool loop whose final assistant turn does not
open with its thinking block; when a client sent none back (Vercel, LangChain, a Responses client
that drops reasoning items), that request goes without `thinking`, which Anthropic accepts. The
same rule covers a Claude model behind Chat Completions (OpenRouter's `anthropic/…`: any
Claude row's OpenRouter candidate), on a same-wire
relay and a translated walk alike: when the request asks for reasoning (`reasoning_effort`,
`reasoning`, `include_reasoning`, `thinking`) and its last assistant tool-call turn carries no
`reasoning_details` OpenRouter can replay (a signed `reasoning.text` or an encrypted entry), those
keys are dropped for that request — pi, the stock OpenAI SDK and LangChain echo at most a plain
`reasoning` string. The inverse also holds: a request that does not think loses the thinking its
assistant messages carry, since Anthropic on Bedrock, directly and behind OpenRouter, rejects it in
the final turn ("When thinking is disabled, an `assistant` message in the final position cannot
contain `thinking`"), and a tool loop is one turn: its first call 400s as much as its last
(Anthropic's own API accepts both, and Bedrock accepts thinking in a turn a later user message
answered; measured 2026-10-01). A request that does not think has no use for any of it and
Anthropic ignores it, so it goes from every assistant message rather than from a turn boundary the
gateway would have to guess (a message of only thinking keeps it: an empty one is a 400). "Does not think" is no reasoning key onto Chat Completions, and `thinking:
disabled`, or no `thinking` before 4.7, onto Messages; an adaptive-generation model thinks without
being asked, so its blocks stay. A Responses client with thinking off (pi) replays the gateway's
`reasoning` items, so without this its tool turn 400ed on claude-sonnet-4 (D79). A relayed body
that never says `reasoning` or `thinking` is not parsed (one `memmem` each). Structured output maps
`response_format` `json_schema` ↔ `output_config.format` ↔ Responses `text.format`; OpenAI JSON mode
(`json_object`) has no schema to give Anthropic and is dropped (OpenAI already requires the prompt to
ask for JSON). Inline PDFs map Chat `file` ↔ Anthropic base64 `document` ↔ Responses `input_file`.

**Reasoning and sampling follow the upstream model.** `translate::request` takes the id this
attempt's candidate receives and parses its Claude generation (`ClaudeGen`, any spelling:
`claude-opus-4-8`, `anthropic/claude-opus-4.8`, Bedrock's `global.anthropic.…`). On 4.7 and later
(and Fable, Mythos), `budget_tokens` and non-default `temperature` / `top_p` are 400s, so
`reasoning_effort` becomes `thinking: {type: adaptive}` plus `output_config.effort`, "none" becomes an
omitted `thinking` at effort `low` (several of these models reject `thinking: disabled`), and sampling
is dropped. Sonnet 5.5 is the exception for "none": it rejects `disabled` and an omitted `thinking`
is adaptive thinking, so "none" is `thinking: {type: between_tools}` at effort `low` (no extended
thinking; nothing else may sit in that object, and effort must be `high` or below). When the
history holds thinking and the request carries `block_binding` (below), which `between_tools`
rejects, it stays adaptive at `low`: keeping the conversation answerable outranks skipping the
thinking. 4.6 is the same with `xhigh` → `max`. Before 4.6, and for non-Claude models behind a
Messages-compatible API, effort becomes `budget_tokens` held below `max_tokens` (at most half, at
least 1024, none at all when `max_tokens` ≤ 1024), and sampling is dropped only alongside thinking.
Before this, an OpenAI SDK sending `reasoning_effort` or `temperature` to `claude-opus-4-8` got a
400. A client's own Anthropic `thinking` object on the OpenAI wire goes through the same mapping
(`{type: enabled, budget_tokens}` is a 400 on 4.7+, `{type: enabled}` without a budget a 400
everywhere). Where sampling survives it is clamped to Anthropic's 0–1, and `top_p` yields to
`temperature` (Claude 4.5-era models reject both together). Budget thinking with a forced
`tool_choice` is a 400 ("Thinking may not be enabled when tool_choice forces tool use"); the forced
call is the client's contract and reasoning a hint, so the thinking goes. Adaptive thinking takes
forced tool use (measured on 4.6, 4.8 and Opus 5).

The other direction has the same problem. `translate::OpenAiModel` parses OpenAI's own ids
(`gpt-4*`, `gpt-5*`, `gpt-6*`, `o*`; measured against Chat Completions and Responses on
2026-09-30) and every OpenAI-bound mapping (Messages → Chat Completions, Responses → Chat
Completions, Chat Completions / Messages → Responses) applies it: `max_tokens` becomes
`max_completion_tokens` (a 400 otherwise on every reasoning model), `temperature` / `top_p` go
unless the effort sent is `none` (reasoning models take only the default), and the effort is
clamped to what the family accepts (next value up, else the highest): GPT-4.x has none (the field
is "Unrecognized"), GPT-5 `minimal`–`high`, 5.1 `none`–`high`, 5.2+ `none`–`xhigh`, GPT-6
`low`–`xhigh`, `-pro` variants `medium`–`xhigh` (`gpt-5-pro`: `high` only), o-series `low`–`high`.
So `thinking: disabled` is `minimal` on GPT-5 and omitted on GPT-4.1, and `effort: max` is `xhigh`
or `high`. OpenRouter's `openai/…` ids get only the effort clamp: OpenRouter renames `max_tokens`
and drops sampling itself (measured) but passes `none` and `xhigh` through to a 400. gpt-oss, on
any host and in any spelling, gets the same clamp to `low`–`high` (reasoning is mandatory: `none` is
"Reasoning is mandatory for this endpoint", measured on OpenRouter). Other OpenAI-wire hosts (xAI,
DeepSeek, Kimi, …) keep the body as sent, except that an effort only OpenAI's newer families define
becomes the classic one: `xhigh` / `max` (what a large Anthropic `budget_tokens` or `effort: max`
maps to) → `high`, `minimal` → `low`. `stop` on a
model that rejects it (GPT-5+, o3, o4-mini) is still forwarded: dropping it would return text past
the stop sequence the client asked for.

**Forced tool use on models that reject it.** Claude Fable 5.1, Mythos 5.1, Opus 5.5 and Sonnet 5.5
400 on `tool_choice` `any` / `tool` (`ClaudeModel::forced_tool_choice`). For those, Chat `required`
or a named function becomes `auto` plus a closing mid-conversation system message ("Respond by
calling one of the provided tools." / "Respond by calling the `get_time` tool."), which is
Anthropic's documented migration for these models. The message is appended after the cache
breakpoints, so the cached prefix is untouched, and only when the request ends on a user turn (the
only place such a message is valid). It is an instruction, not a guarantee. Measured live
(2026-09-30): `required` with no fitting tool ("tell me a joke") still called a tool 6/6 on Sonnet
5.5 and Opus 5.5; a named tool that conflicts with the question was called 8/8, and Sonnet 5.5 also
called the tool that fit the question alongside it 4/4. Limiting it to one call made Sonnet pick the
fitting tool instead, so parallel calls stay allowed. Every other model keeps the hard `any` /
`tool`. `allowed_tools` has no Messages subset: `required` over one tool forces that tool, over
several forces any tool (trimming `tools` instead would rewrite the cached, thinking-bound prefix).

**Preserved thinking on a translated conversation.** Fable 5.1, Opus 5.5 and Sonnet 5.5 bind each
thinking block to the conversation that produced it (system, tools, every earlier message); a
replayed block whose prefix changed is a 400 ("bound to a different conversation") on accounts
created on or after 2026-08-31, and on any account whose request sets
`thinking.block_binding.prefix_mismatch_behavior`. The translator keeps the prefix append-only
where it can: a mid-conversation `system` / `developer` message stays in `messages` as a
`role: "system"` message (Opus 4.8+, Fable, Mythos, Sonnet 5.5+) instead of being folded into
top-level `system`, which used to rewrite the prefix, and the prompt cache, every time a client
appended one. Messages accepts such a message only right after a user turn and before an assistant
turn or at the end, so one placed elsewhere moves past the next user turn; with no user turn after
it (the request ends on an assistant turn) it joins top-level `system`, as it does on every other
model. The forced-tool instruction above cannot be made append-only: it exists only in the
gateway's request, so the client's next turn replays that turn's thinking without it. So every
translated request to one of those models on Anthropic's own API carries `block_binding:
{prefix_mismatch_behavior: "drop_block"}` (on an explicit `{type: adaptive}` when the client sent
no thinking; never with `between_tools`, which rejects it), and `proxy`'s `upstream_request_filter`
adds the `thinking-binding-controls-2026-08-01` beta, merged with any the client sent, from the
same model-id fact (`translate::messages_beta`), since headers leave before the body is
translated. The API then drops that block and every thinking block after it instead of failing.
Always set, not only when the history holds thinking: a thinking parameter that changes between
turns restarts the prompt cache. Measured live on Sonnet 5.5 with enforcement on: turn 2 after a
forced turn was the 400 without the field and a 200 with `thinking_dropped` with it; a client
appending a system message kept every block valid across three turns, where hoisting 400ed.
Bedrock and OpenRouter spellings get neither the field nor the header: whether those hosts accept
the beta is unverified, and the field without it is a 400. No Bedrock candidate serves one of those
models today; one that does would 400 a turn after a forced one on an enforced account.

**Unmappable input is forwarded, not dropped.** `input_audio`, a `file_id` or URL document, a
Files API image, a non-base64 data URI, `n` > 1, `logprobs`, audio output, Anthropic server and
Anthropic-defined tools (`web_search_…`, `bash_…`, `text_editor_…`: as an empty-schema function the
model would call one and nothing would run it), Responses hosted tools (`file_search`, `mcp`,
…), a Chat Completions client's `custom` tools onto Messages, `mcp_servers`, a Responses `prompt`
template and `top_logprobs` (onto Chat Completions and, through the intermediate Chat body, onto
Messages: D129), `stop` onto Responses, and Responses input items with no Chat Completions shape
(`computer_call_output`, `local_shell_call`, …, forwarded whole in their place in `messages`), and
a message whose role no dialect has (a typo, a framework's private role: forwarded whole, in place,
role unchanged, never turned into a user turn) have no equivalent on the other wire and change what
the client gets back. Only records of a hosted tool the provider ran itself (`web_search_call`,
`mcp_call`, …) are dropped: the client wrote none of it, and the answer that used it follows as a
message. So is a `compaction` item, OpenAI-held state (a summary only OpenAI can decrypt) that no
translated upstream can resolve: a compacted Codex session that fails over onto a translated
candidate runs on the history the client holds, a degraded answer rather than a 400. An
`item_reference` reaches translation only from inside a tool step, and is dropped there; one that
stands for an earlier turn is refused before any upstream (D175, see
[Stored-response references](#stored-response-references-item_reference)). The distinction from forwarding:
an item the gateway does not know is forwarded (the provider names it), and a known item that only OpenAI can read is
dropped when translating. A same-wire Responses relay keeps both. OpenAI's hosted search
(`web_search`, `web_search_preview` and dated spellings) is the one tool dropped leaving Responses,
unless the `tool_choice` names it: Codex offers it on every turn, by default over OpenAI's cached
index (`external_web_access: false`), which no other provider has; the model may use it or not and
the client runs nothing for it, so Codex works on with its other tools (it searches only when it
chooses to). Forwarded, it made Codex unusable on every Claude row (D78). Anthropic's server
`web_search` is not substituted: it always fetches the live web (what Codex's default opts out of),
bills per search, and a Claude row's OpenRouter and Bedrock candidates would each need their own
spelling, so a failover would change what the tool does. An explicit `null` (how OpenAI SDKs send
an unset option) is "not set" and is never forwarded; that includes Responses `instructions: null`,
which becomes no system message rather than one with null content (D102). A same-wire Chat
Completions relay to a host other than OpenAI (OpenRouter, xAI, Together, …) drops root-level nulls
too, by span (`peek::remove_root_nulls`; every other byte is the client's): OpenRouter 400s
`user: null`, which openai-python sends for `user=None`, so a Claude row's failover used to change
the outcome (D101). Translation runs in `request_body_filter`, after the request headers went
upstream, so the gateway cannot answer 400 itself; the field is passed through and the provider's
400 names it (each verified live, 2026-09-30). A non-http(s) image or file URL (`file://`) is never
forwarded in any shape, on any wire pair (Chat Completions ↔ Responses included); inline `data:`
URIs are not URLs and pass. Hints that change nothing about the response's shape (`seed`, penalties,
`logit_bias`, `top_k`, a message's `name`) are dropped, and so are `prompt_cache_key`,
`service_tier` and `verbosity` everywhere but OpenAI's own Chat Completions (which takes all three).
Equivalents are mapped instead: legacy `functions` / `function_call` become the `tools` loop (a
legacy call gets an id from its message position, reused by its `function` result); a
Responses `custom` tool and its calls map to Chat Completions' `custom`, in requests and in
responses both ways (a Responses `custom_tool_call` item with its raw `input` ↔ a Chat
`tool_calls` entry of `type: "custom"`, streamed as `response.custom_tool_call_input.*` ↔
`custom.input` deltas; a grammar format's flat `{type, syntax, definition}` ↔ Chat's nested
`{type, grammar: {syntax, definition}}`, which OpenAI requires). Onto Messages, which has no
free-form tool, a Responses client's `custom` tool becomes a tool taking one `input` string, its
description carrying the grammar, and a call to it comes back as a `custom_tool_call` whose raw
input is that string (streamed whole once the arguments object is). A Responses `namespace` tool
(Codex groups its multi-agent and MCP tools in one) is flattened onto Chat Completions and Messages:
each member tool is sent as `{namespace}__{name}` (`{namespace}{name}` when the namespace already
ends in `_`), with the namespace's description ahead of its own, and a replayed call carrying
`namespace` is sent under the same flat name. The flat names are per-request state
(`translate::ToolNames`, built from the client's tools as the body is translated): each call that
comes back is named `{name, namespace}` again, in JSON and in streams. The tool message answering a `custom` call becomes a
`custom_tool_call_output` onto Responses (OpenAI rejects a `function_call_output` for it); a custom
call in history onto Messages keeps its name, its raw text as `tool_use.input: {"input": …}` (the
input must be an object); `tool_choice` and `parallel_tool_calls` go onto Chat Completions only
alongside `tools` (OpenAI 400s either without them; Messages and Responses accept both); a Responses named
`tool_choice` or `allowed_tools` nests its name the Chat Completions way and back; consecutive
Responses `function_call` items become one assistant message (OpenAI rejects the split form),
joined to that turn's text; consecutive same-role messages share one Messages turn, each keeping its
own text block (two strings are never fused into one); an assistant refusal is text on Messages; a mid-conversation Anthropic
`system` message stays a Chat Completions `system` message in place; a Responses `developer`
message stays `developer` on OpenAI's own API and becomes `system` on every other Chat Completions
host (DeepSeek, Mistral, OpenRouter, … know no `developer` role; a same-wire Chat body gets the same off OpenAI and OpenRouter, see "`developer` off OpenAI"); tool `strict` crosses both
ways; structured output is `strict: true` onto OpenAI only when the schema qualifies (every object
closed with `additionalProperties: false`, every property required; Anthropic allows optional
ones); an Anthropic `tool_result` that is an error says so in the tool text (`Error: …`); a
Responses `reasoning.summary` asks current Claude for `display: "summarized"`. Images or files a
tool returned stay with it: inside the `tool_result` on Messages, and on Chat Completions (whose
tool message holds text) as a user message right after the run of tool messages. Other hosts' tool
call ids (`functions.get_weather:0`) become `^[a-zA-Z0-9_-]+$` on Messages, identically on the
call and its result, with a hash of the original. An email address as `user` (Anthropic rejects
it as `metadata.user_id`) becomes its FNV-1a hash, stable per user. Chat Completions or Messages → Responses
sends `store: false` unless the client asked to store (OpenAI and xAI both store a response by
default). Tools, text, and usage still
round-trip. Anthropic requires `max_tokens`; a missing OpenAI value becomes 4096. On every catalog
walk, translated or not, a root `max_tokens` / `max_completion_tokens` / `max_output_tokens` above
the row's **published** max output (`ModelCard::output_cap`) is capped to it (never raised; spans
found by the same `peek::scan_buffered` walk that finds `model`, spliced in place). Claude Code sends
32000 or 64000, which gpt-4o-class rows (16384) answered with a 400. A row whose vendor publishes no
max output (Grok, Kimi, MiniMax, …) lists the `UNPUBLISHED_MAX_OUTPUT` placeholder and is not
capped: cutting 64000 to a figure that is not the vendor's would end answers early with
`finish_reason: length`. The catalog models one card per row, not a limit per candidate, so the
row's card is the limit on every candidate. The cap runs before translation, so a
thinking budget derived from the limit fits under it too. When the serving candidate is native
OpenAI Chat Completions (provider `openai`, `/v1/chat/completions`), a same-wire relay also respells
a root `max_tokens` as `max_completion_tokens`, which every OpenAI chat model accepts and which
its reasoning models require; with both present, the explicit `max_completion_tokens` stays and
the `max_tokens` member is removed (key offsets from the same scan, edited in place). Other
OpenAI-wire hosts keep `max_tokens`, the spelling they document. OpenAI→Anthropic
does not inject `stream_options`. Anthropic→OpenAI injects `include_usage` on the translated
Chat Completions body when streaming; Responses→Chat Completions does the same because injection
follows the **upstream** endpoint. A stock OpenAI SDK also does not send `anthropic-version`; the
gateway injects `2023-06-01` on a walk that lands on Messages. Usage/billing still parse the
**upstream** body. Responses ↔ Messages is composed through Chat Completions.

**Default cache breakpoints onto Messages.** OpenAI caches a repeated prefix by itself; Anthropic
caches only up to an explicit `cache_control` marker, and a stock OpenAI SDK never sends one. So a
Chat Completions or Responses client on a Claude row used to pay full input price for its whole
prefix every turn. When a request translated onto Messages carries no `cache_control` anywhere,
the gateway adds at most two `{"type":"ephemeral"}` markers: one on the last system block (or the
last tool when there is no system; tools render first, so one marker covers both), and, once the
request holds an assistant turn, one on the last non-thinking block of the last message so the next
turn reads the conversation back. A single-turn request gets only the prefix marker, since a write
costs 1.25× input and a one-shot never reads it. One client marker anywhere disables all of this.
A Chat client's marker on a whole message (rather than a content part) is kept on every role: on
each system or user text block, on an assistant turn's last non-thinking block, and on a tool
message's `tool_result`.
Same-wire Messages traffic is a byte relay and is never touched.

**Who pays for a cache write: whoever chose to cache.** When the gateway added the markers
(`translate::request_with_tools` reports it per attempt, into `TranslateState::gateway_cache`), the
client never asked for caching, so the writes they cause are the gateway's optimization and bill at
the input rate. The row folds them into `input_tokens` (on the `anthropic` wire, whose
`input_tokens` excludes writes; the `openai` wire's already includes them), reports
`cache_write_tokens` and `cache_write_1h_tokens` as 0, and records the count in
`gateway_cache_write_tokens` so the row still reconciles against the provider's usage, which calls
them cache writes (`Usage::bill_gateway_cache_writes`). The client is shown the same: the translated
`prompt_tokens` / Responses `input_tokens` is the whole prompt as before, with
`cache_write_tokens: 0`. An OpenAI-wire harness (the AI SDK's openai-compatible provider, pi's
Responses provider) reads no cache-write field and prices the whole prompt less cache reads at
input, which now matches the bill. A client that sent `cache_control` itself asked for the writes
and is billed and shown true cache writes, exactly as calling Anthropic directly; whether its wire
can display them is a limitation it shares with direct use. Cache reads are unaffected either way.

**What the client gets back.** Responses are translated in `translate::response_json_status`
(non-stream, withheld until end of stream) and `translate::SseBridge` (event by event). Every stream
pairing meets in Chat Completions chunks held as values, so Messages → Responses parses each event
once. Every Responses-upstream path (Responses → Chat Completions / Messages) is live for the
Responses-only models the catalog routes to `/v1/responses`, and is held to the same mapping:

- **Usage.** Anthropic's `input_tokens` counts only uncached prompt tokens; OpenAI's
  `prompt_tokens` and Responses `input_tokens` count the whole prompt. So `prompt_tokens` =
  `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens`, with
  `prompt_tokens_details.cached_tokens` = cache reads and `cache_write_tokens` = cache writes (the
  field OpenAI and OpenRouter use); the reverse subtracts. Responses usage always carries
  `input_tokens_details` and `output_tokens_details` (the schema requires them). A `message_delta`'s
  cumulative counts replace `message_start`'s. Billing parses the upstream body. On a request whose
  cache breakpoints the gateway added, cache writes show as uncached input to the client and on the
  row alike (see above).
- **Stop reasons.** `end_turn`/`stop_sequence` ↔ `stop`, `tool_use` ↔ `tool_calls`, `max_tokens` ↔
  `length`, `refusal` ↔ `content_filter`. `model_context_window_exceeded` and `pause_turn` become
  `length`: both leave the turn unfinished, and resending it is the remedy for either. A Responses
  client gets `status: "incomplete"` with `incomplete_details.reason` `max_output_tokens` or
  `content_filter`, and a terminal `response.incomplete`. OpenRouter's `native_finish_reason`, when
  it is an Anthropic stop reason, reaches an Anthropic client exactly. OpenAI's `message.refusal`
  becomes refusal text plus `stop_reason: "refusal"` (and a Responses `refusal` part); Anthropic's
  `stop_details.explanation` becomes `message.refusal`. Tool calls under a plain `stop` are still
  `tool_use`. No empty text block is invented.
- **Thinking.** An Anthropic client only ever receives signed `thinking` or `redacted_thinking`.
  OpenRouter streams Claude's signature in a later `reasoning_details` entry, so the bridge holds a
  thinking block (up to 8 MiB) until its signature arrives and emits it whole; `reasoning.encrypted`
  becomes `redacted_thinking`. Only payloads whose `format` is Anthropic's are trusted. Reasoning
  that never gets a signature — DeepSeek, gpt-oss, OpenAI summaries, a block cut off mid-way — is
  **dropped**: echoed back unsigned, it would 400 every later turn Anthropic serves. A Responses
  client gets each thinking block as a `reasoning` item (text as a summary, signature as
  `encrypted_content`); `redacted_thinking` has no Responses form and is dropped there. A Chat
  client gets what it must send back: `reasoning_content` (the text) plus the `thinking` array
  (the blocks). Streamed, the text arrives as `reasoning_content` deltas and each finished block —
  signed, or redacted — as one `thinking` list entry with its own `index`, so openai-python's
  accumulator (which requires an `index` on every list entry) keeps blocks apart and
  `messages.append(final.choices[0].message)` echoes exactly the signed blocks. A block that never
  got its signature is not offered for replay.
- **The Responses stream** is the full lifecycle: `response.created`, `response.in_progress`, and per
  item `output_item.added` → content events (`content_part.*`, `output_text.*`, `refusal.*`,
  `function_call_arguments.*`, `reasoning_summary_part.*` / `reasoning_summary_text.*`) →
  `output_item.done`, each with `sequence_number`, `output_index` and `item_id`, ending in
  `response.completed` carrying the whole `output` and usage. An upstream error is an `error` event
  (with the envelope under `error`, so an OpenAI SDK raises it) followed by `response.failed`, whose
  `error.code` is one of the closed set the Responses schema allows, plus the two real OpenAI also
  sends there, `context_length_exceeded` and `insufficient_quota` (`server_error` when the
  upstream's code is none of those; the `error` event keeps the upstream's own).
- **Streams that fail or stop mid-way.** OpenRouter reports a provider that died mid-generation as
  a chunk carrying `choices` _and_ an `error` (with `finish_reason: "error"`); that is an error, not
  a finish, for a Messages or Responses client (a Chat client on the same wire gets the relay). A
  stream that ends without saying how — no stop reason, no end marker — was cut short: the client
  gets an error (`code: "stream_truncated"`; a Chat client then its `[DONE]`), never an invented
  `end_turn` / `response.completed`. `proxy` reaches that flush only on a clean end of the upstream
  body (a close-delimited body, or a provider that stopped writing); an upstream that drops a
  chunked, sized or HTTP/2 body mid-way fails the request in Pingora, and the client's connection is
  cut instead.
- **Nothing follows the client's terminal event.** Once a translated client has its `[DONE]`,
  `message_stop` or `response.completed`, nothing more reaches it: an upstream that writes past its
  own end (an error between a Chat usage chunk and its `[DONE]`, an event after `message_stop`) is
  dropped, where it used to hand the client a second ending on a turn that had completed (D124).
  The stream the client was sent then ends on its terminal event, which is what
  `terminal::TerminalTracker` reads to bill a close after the answer as `ok` (D120).
- **Tool-call deltas** are keyed by `index`, else by `id`: an id repeated on every delta is one call,
  a reused `index` with a new id is a new call, interleaved deltas land on their own call, and a call
  opens only once its name is known. A `tool_use` block that closes with no argument bytes still
  gives a Chat or Responses client JSON: the `input` its `content_block_start` carried (a
  Messages-compatible host may send it whole there), else `{}` (a zero-argument call), as the
  non-stream body does. Onto Messages, parallel calls stream as sequential `tool_use` blocks
  (Anthropic never opens a block before closing the last): a call that appears while another's
  block is open waits, its arguments gathered, and opens once the open call's arguments are complete
  JSON (no valid byte can follow) or the stream ends. The end is found incrementally, scanning only
  the argument bytes new since the last step, with one parse to confirm it when the outer value
  closes. An open call with no argument bytes is done once a waiting call's arguments start
  arriving (a zero-argument call; the upstream has moved on). A sequential upstream streams each
  call as it comes; one that interleaves argument deltas has the later calls held back.
- **Tool arguments keep their numbers as written.** Translation moves arguments between a JSON
  string (Chat Completions and Responses `arguments`) and a JSON value (Messages `tool_use.input`).
  Parsed into a `serde_json::Value`, an integer past `u64` became an `f64` and a long decimal was
  rounded (`-1.32417719719006e-11` came out `-1.3241771971900598e-11`, D127), so the model's own
  arguments were altered before reaching the client's tool or its replayed history. Arguments
  holding a number a `Value` keeps as an `f64` (a decimal, an exponent, a huge integer) now keep
  their text: a Chat `arguments` string becomes a verbatim stand-in that `translate::Exact` writes
  back as the original text, and a Messages body whose `tool_use` inputs hold one is parsed a
  second time keeping each `input`'s text (`KeepInputs`). Integer-only arguments, the common case,
  take the path they always did. serde_json's `arbitrary_precision` would have kept every number,
  but it is crate-wide and breaks `untagged` enums holding numbers, which async-nats' JetStream
  responses are. Measured (`benches/unit.rs`, `translate`): a Messages tool call with a decimal
  onto a Chat client 4.0 → 5.4 µs, the same with integers and 20 replayed Chat calls onto Messages
  unchanged. Streams already passed argument text through untouched.
- **Errors.** Any non-2xx JSON body is an error, whatever its shape (Bedrock's `{"message"}` has no
  `error` key); `message`, `type`, `code` and `param` survive, a string `error` is the message,
  OpenRouter's `metadata.raw` is quoted after its message with the provider's name, and a numeric
  `code` is a string on the OpenAI wire. OpenAI types with an Anthropic name get it
  (`server_error` → `api_error`, rate limits → `rate_limit_error`), and a body that names no type is
  typed from its status (Anthropic's closed set on Messages: 400 `invalid_request_error`, 404
  `not_found_error`, 429 `rate_limit_error`, 529 `overloaded_error`…; on the OpenAI endpoints 4xx
  `invalid_request_error`, 429 `rate_limit_error`, else `api_error`). The same endpoint on another
  vendor gets this too: a catalog walk buffers a non-stream error from a same-endpoint candidate and
  re-encodes it unless it is already the client API's envelope, so xAI's `{"code", "error":
  "<string>"}` reaches an OpenAI SDK as `{"error": {"message", "type", "code"}}` (D100), while
  OpenAI's own errors and OpenRouter's are relayed verbatim except provider-account remedies (see
  "Provider-account remedies" below). A context overflow carries what
  each client's harness compacts on: OpenAI's `context_length_exceeded` reaches a Messages client
  with its message prefixed "prompt is too long: " (Claude Code's trigger), and Anthropic's "prompt
  is too long" reaches a Chat Completions or Responses client with code `context_length_exceeded`
  (Codex's and the Agents SDK's).
- **Fields.** Every `chat.completion` and chunk has `created` and one `id`; Responses objects have
  `created_at`, items have `id`s (`msg_`, `fc_`, `rs_`) and function calls a `call_id`. An id the
  upstream did not give is minted (`…_gw…`), never a shared constant.
- **Not mapped:** server-tool blocks and hosted-tool items (only reachable through tools a
  translated client cannot declare), `choices` past the first, logprobs.

**Provider-account remedies (D174).** A provider error reaches the client verbatim — its status,
`type`, `code`, `param`, `Retry-After` and message — except advice about Beyond's own account with
that provider. OpenRouter's shared-pool 429 says "add your own key to accumulate your rate limits:
https://openrouter.ai/settings/integrations" beside `is_byok`, `limit_source:
upstream_provider_shared_pool` and `remedy_hint`; a gateway client has no OpenRouter account to add
a key to, and the text names the upstream behind the row and describes Beyond's account with it.
Billing and quota remedies do the same with Beyond's account state. So on a managed JSON error the
pool-key scrub (`Redact`, which already rewrites error bodies, before translation, capture and the
cache) also runs `remedy::neutralize` at end of body:

- **What counts.** A `message`, a string `error` (xAI) or OpenRouter's `metadata.raw` containing,
  case-insensitively, a phrase from one of two lists, and always OpenRouter's `metadata.is_byok`,
  `limit_source` and `remedy_hint`.
  - _Out of credit or quota_ (`remedy::UNFUNDED`): "check your plan and billing details",
    `platform.openai.com/account/billing` (OpenAI's `insufficient_quota`, also matched by that code
    or type; Gemini uses the same words); "credit balance is too low",
    `console.anthropic.com/settings` (Anthropic's billing 400); "purchase more credits", "raise your
    spending limit", `console.x.ai` (xAI's spent credits or spending limit, which name Beyond's team
    id); "insufficient credits", `openrouter.ai/settings/credits` (OpenRouter's 402); "insufficient
    balance" (DeepSeek's 402).
  - _Other account advice_ (`remedy::REMEDIES`): "add your own key", `openrouter.ai/settings`
    (OpenRouter's shared pool, its key and BYOK pages); `platform.openai.com/account` (OpenAI's
    organization rate limit, which names Beyond's org id); `anthropic.com/contact-sales` (Anthropic's
    organization rate limit, which names Beyond's org id); `console.groq.com/settings` (Groq's
    "Upgrade to Dev Tier").
- **What it becomes.** The message says what the error is, without the account behind it. A rate
  limit (a 429 that is not out of credit) reads "The provider is rate-limited upstream; retry
  later.": waiting clears it. A 429 that says the request alone is over the limit (OpenAI's TPM
  "Request too large ... The input or output tokens must be reduced in order to run successfully")
  is the exception, since no wait admits it: it reads "The request is larger than the provider's
  per-minute token limit admits; reduce the input or output tokens. Retrying it unchanged will not
  succeed." (D196; "retry later" sent clients round the same 429). Out of credit or quota, whatever its status (Anthropic's is a 400,
  OpenAI's a 429), or account advice on any other status, reads "The provider cannot serve this
  request right now; retry later or use another model.": true without saying why, and a retry is
  served, since the walk now leaves that provider out (D180, below). A `raw` carrying a phrase is
  removed (a translation quotes `raw` after the message, so it would repeat), and the three metadata
  keys are removed. `type`, `code`, `param`, `metadata.provider_name` and the status and headers are
  kept, so an SDK raises the same typed error and retries on the same `Retry-After`.
- **What does not count.** Anything the client can act on carries none of those phrases and is
  relayed as sent: a context overflow, an unknown or invalid parameter, a content-policy refusal, a
  plain rate limit, OpenRouter quoting an upstream's own error. An OpenAI organization rate limit
  does lose its "try again in 2s" with the rest of the message; `Retry-After` still says when.
- **Scope and cost.** Managed only: a BYO error is the caller's own account talking, and its remedy
  is one they can follow. Error paths only: a status >= 400 with a JSON body (<= 64 KiB) is held to
  its end and parsed once; a 2xx is never held or parsed, and the response head drops its
  `Content-Length` only on that error path (the rewrite changes the length). A body that is not a
  JSON object, an SSE error, or an error inside a 2xx stream is not rewritten. OpenRouter's root
  `user_id` on its errors is not a remedy and is relayed.

**Managed responses are never compressed upstream** (`accept-encoding: identity`; see "Usage
Extraction"). For translation it matters as much as for billing: a gzipped body reached a translated
client untranslated, or as plain text under `content-encoding: gzip`.

`/{provider}/…` is the escape hatch and does not consult the catalog. This arm is reached only after
a provider-table miss, so `/{provider}/…` traffic runs exactly the code it always did; `auto` is
refused as a provider name at boot so config cannot shadow it.

The name can live in the body because the gateway peeks it itself. Pingora's 64 KiB retry buffer is
enabled and `ModelScanner` is fed each new chunk once (a `model` after a long prompt is linear, not a
rescan per chunk). The whole body is read, never a stop at `model`: inbound Responses' `store` /
`previous_response_id` can sit after `input`, the cache hashes the whole body, and stock Python SDKs
put `model` **after** `messages` / `input` anyway.

- **A small body** (declared under 64 KiB) fits the buffer, so pingora replays it on the first
  attempt and on every failover.
- **A larger body** (or one of unknown length) outgrows the buffer. That leaves pingora nothing to
  send, since its retry buffer is gone and the client has nothing more to read, so the request is
  re-run as a pingora **subrequest** carrying the body
  (`AiProxy::relay_full_body`), whose response is piped to the client (`pipe_full_body`). The
  subrequest runs this same proxy with a gateway-only context (`FullBody`: the chosen row and
  Responses session field), so it does not peek again, does not charge the rate guardrail twice,
  and does not count as a second request. It does its own auth, walk, translation and `ai.usage`.
  Counted on `ai_full_body_relays_total`.
- **No `model` in the whole body:** 404. **Past `MAX_REQUEST_BODY`:** 413.

The pipe idle-watches the client while the subrequest runs, the way pingora's own proxy loop does
(pingora's `pipe_subrequest` does not poll the client when the body is preset). A client that hangs
up closes the subrequest's channels, so its upstream is aborted at once and a cut-short stream
bills its estimate, instead of a hidden-reasoning model generating for minutes after an ESC.

What the parent does for every attempt, so a relayed request behaves like any other:

- **One request.** Every attempt carries the parent's request id and sequence (`FullBody`), so the
  client's `x-beyond-request-id` names the row that bills, and `x-beyond-split` or the probe seed
  cannot pick a different primary per attempt (a 429 key walk stays on its vendor). An abandoned
  attempt (one that recorded a `RelayRetry`) still feeds the breaker and the ranker but writes no
  `ai.usage` or `ai.payload` row.
- **One tenant slot, taken before the read.** `tenant_max_in_flight` is checked before the body is
  read in full (`SlotGuard`), so it bounds the bodies held in memory, not only requests in flight: a
  tenant at its cap gets its 429 before uploading. The parent holds the slot across every attempt;
  attempts neither take nor release one.
- **Abandoned attempts wind down first.** The next attempt starts after the abandoned one finishes
  (bounded at 2s), since it still holds that candidate's breaker permit and a half-open breaker has
  one.
- **A reset retries only before delivery.** An upstream connection that fails before any response
  header _and before the upstream had the whole body_ (a reset mid-upload, a write error) is retried
  on the same candidate once, then the next, which pingora cannot do for a body past its buffer.
  After delivery it is not retried at all — see "A delivered body is never sent twice" below.
- **HTTP/2 clients.** Pingora builds a subrequest by rendering the parent's header as HTTP/1.1; an H2
  parent renders as `HTTP/2`, which that parser rejects. The header is rendered as HTTP/1.1 with the
  body's real `Content-Length` and a `Host` from `:authority`, then restored.
- **No silent hang.** An attempt that ends with neither a response nor an error (a panic) is a 502,
  never a connection left open with nothing written.
- **`Expect: 100-continue`** is answered before the body is read (curl waits a second for it).

Before this, a body whose `model` was past 64 KiB was a 404, and one where the read that found it
also ended the body hung until the client timed out (openai-python batches of ~150 embeddings
inputs, or any long agent turn). A Responses turn with `previous_response_id` past 64 KiB was taken
for a one-shot and translated onto Chat Completions, which dropped the conversation. And no body
past 64 KiB could fail over (see "Status-based failover, and where it stops").

Three things differ from the provider-routed path, all consequences of the client no longer naming
the provider:

- **The upstream path comes from the catalog, per candidate.** Providers disagree on where an
  endpoint lives and the disagreement is _not_ a prefix: Anthropic serves Messages at `/v1/messages`
  from a base carrying no path, OpenRouter serves Claude at `/api/v1/chat/completions`. No client
  suffix is correct for both, so each candidate states its path outright and `forward_path` is set
  from it per attempt. The client points its SDK at `/v1` (or `…/auto`) and the row decides.
- **The model id is rewritten per attempt.** Providers essentially never share a string —
  `claude-opus-4-8` at Anthropic is `anthropic/claude-opus-4.8` at OpenRouter — so the body's `model`
  is spliced to whatever the serving candidate calls it (`peek::scan_buffered` reports the value's
  byte span). Because the body may change length, catalog-walk requests are buffered and re-framed
  exactly as the injection path is; the two are one predicate (`RequestCtx::rewrites_body`).
- **OpenRouter is asked not to compress.** OpenRouter's context compression is on by default for
  endpoints of 8K context or less: an over-window prompt has its middle dropped and is answered
  (gpt-4 through OpenRouter returned 200 to a 10,601-word prompt). An attempt on an OpenRouter
  Chat Completions candidate gets `"plugins":[{"id":"context-compression","enabled":false}]`
  spliced just inside the root object, OpenRouter's documented switch, so it answers the context
  error every other host does. A body that already names `plugins` is the client's choice and is
  left as sent.
- **It is managed-only on `/auto`, and on `/v1` only for managed keys.** A BYO token belongs to one
  provider, so selecting among candidates would be a guess and failing over would hand one vendor's
  key to another. BYO on `/auto` → 400. BYO on `/v1` is unchanged dialect-default passthrough.

Failover covers both shapes: a candidate that will not **connect** (refused, timed out, or absent
from DNS) and one that **answers with a 5xx**, provided nothing has gone downstream yet and the
request body is still replayable. Candidates whose breaker is open are skipped without an attempt.
See "Status-based failover, and where it stops" for the 5xx path and its one real limit.

Catalog rows live in `providers::catalog`, shared with the agent. A row's `wire` is the _client
default_ (what the primary speaks, and what a bare `/v1` caller is assumed to send). Its `price` is
the standard list card published on `GET /v1/models`, not a rate the proxy applies to `ai.usage`.
Candidates may list Messages and Chat Completions (or Responses) together — Claude's OpenRouter arm is Chat
Completions. Each attempt translates the **original** client body onto **this candidate's** path
(`endpoint_of_path` / `wire_of_path`); injection (`stream_options`) follows the upstream candidate,
not the client. Billing dialect is the serving candidate's path, never the row or the provider
table. Sending the wrong wire would trip the dialect-mismatch guard and emit a zero-token billing
row. `/{provider}/…` never translates. Do not invent mixed endpoint types inside one candidate
path. GPT session-state Responses is a **parallel arm** (`ModelRoute::responses`), not mixed into
`candidates`, so Chat Completions TTFT ranking and `stream_options` injection stay on that walk.

**Wire belongs to the serving candidate's path, not the provider and not only the row.**
`ProviderSpec::wire` is one value per provider and that is an approximation: OpenRouter serves the
OpenAI wire at `/api/v1/chat/completions` _and_ a genuine Anthropic Messages wire at
`/api/v1/messages`, and Fireworks is the same story from the other side
(`agent_core::dialect::is_fireworks_anthropic_wire_model`). `request_filter` seeds dialect from the
row; `upstream_peer` overwrites it from the candidate about to be dialed. Reading the provider there
fails silently in the worst way — an Anthropic response meets the OpenAI extractor, trips the
dialect-mismatch guard, and bills zero tokens. A provider-routed `/{provider}/…` request follows the same rule
with no catalog: its dialect, and with it `stream_options` eligibility, is the **forwarded path's**
wire (`route::Endpoint::of_upstream_path`, a sub-resource reading as its parent), so
`/openrouter/api/v1/messages` meters with the Anthropic extractor and `/anthropic/v1/chat/completions`
with the OpenAI one (and gets `include_usage`).

That is what makes **Claude failover real today**: every Claude row Anthropic still serves
(Fable 5 / 5.1, Opus 5 / 5.5, Sonnet 5 / 5.5, Haiku 4.5, plus the 4.x snapshots Opus 4.5 / 4.6 /
4.7 / 4.8 and Sonnet 4.5 / 4.6) routes to Anthropic first and falls back to OpenRouter's Chat Completions endpoint
under the vendor-slug spelling (`claude-opus-5` → `anthropic/claude-opus-5`; `claude-opus-4-8` →
`anthropic/claude-opus-4.8`). `claude-haiku-4-5` and `claude-opus-4-8` insert Amazon Bedrock's
Messages API as an independent second source (`us.anthropic.claude-haiku-4-5-20251001-v1:0` /
`us.anthropic.claude-opus-4-8`) before OpenRouter. GPT rows do the same shape on the Chat
Completions wire (`gpt-5.2` → `openai/gpt-5.2`) plus an OpenAI-only `/v1/responses` arm —
the table covers the current 4 / 4.1 / 4o / 5 / 5.x / 6 and o-series ids OpenRouter listed on
2026-09-19. From GPT-5.4 on (`gpt-5.4*`, `gpt-5.5`, `gpt-5.6-*`, `gpt-6-*`, `gpt-6.1-sol`) the OpenAI
candidate is `/v1/responses` too: OpenAI's Chat Completions answers function tools with any
reasoning effort with a 400 ("use /v1/responses or set reasoning_effort to 'none'"), and GPT-5.6
and GPT-6 reason by default, so every tool call failed there; a Chat Completions or Messages
client is translated onto Responses. xAI `grok-*` rows reach xAI over `/v1/responses` too
(`xai_responses_first`; OpenRouter Chat Completions failover, `grok-4.6` → `x-ai/grok-4.6`): xAI
serves multi-agent only there, reads PDFs only there, and calls its Chat Completions deprecated
(docs.x.ai "Migrating to Responses API"). Chat Completions and Messages clients are translated
onto it with `store: false`; checked 2026-10-01 against xAI's docs and API, its Responses usage
counts cached tokens inside `input_tokens` and reasoning inside `output_tokens`, which xAI's own
per-request `cost_in_usd_ticks` matches exactly on every grok row (catalog sweep CAT-7). So a Chat
client on a grok row now sees OpenAI's convention, `completion_tokens` with reasoning inside, where
xAI's own Chat Completions reports reasoning beside it; and the translated answer carries no
`cost_in_usd_ticks` (a Responses client gets xAI's usage as sent). The same
Chat Completions helper covers every other pool-keyed host: DeepSeek (`deepseek-flash` fails
over to Together's `deepseek-ai/DeepSeek-V4.1-Flash`, then `deepseek/deepseek-v4.1-flash`;
`deepseek-v4-pro` to Together's `deepseek-ai/DeepSeek-V4-Pro-0813`, then OpenRouter's
`deepseek/deepseek-v4-pro-0813`, the same snapshot: OpenRouter's bare `deepseek/deepseek-v4-pro` is
the older 0423), and the Groq/Together llama /
qwen / open-weight ids people send (`openai/gpt-oss-120b` names Groq, Together and Fireworks:
one of OpenRouter's hosts answers a forced tool call with an empty `finish_reason: "error"` 200;
Kimi K3 / MiniMax M3 do the Together + Fireworks + OpenRouter shape, and GLM-5.2 Together +
OpenRouter since Fireworks ended its serverless on 2026-09-25). A
fallback must serve the model the row names, so a retired vendor id is removed with its row
(`deepseek-chat`, `deepseek-reasoner` and `mistral-nemo`, whose OpenRouter fallbacks were other
models). The Mistral `-latest` rows (Large, Medium, Small, Ministral 3B / 8B / 14B, Codestral)
were removed by owner decision until there is an `AI_POOL_KEY_MISTRAL` (D156): with no Mistral
key, their only reachable candidate was OpenRouter, whose only host for each is Mistral on
OpenRouter's own upstream key. That limit is shared with every OpenRouter customer, so it 429s
whatever our traffic does, and a pool-key walk cannot help. The `mistral` provider and its
`/mistral/…` route remain, so bringing them back is the key plus the rows. The Together-primary
Qwen rows (Qwen3.6 Plus, 3.7 Plus, 3.7 Max, 3.8 Flash) keep OpenRouter as their failover, though
OpenRouter's only host for them is Alibaba on that same kind of shared upstream limit. Together
serves them first, and the failover works when OpenRouter's pool has room (live, 2026-10-01: all
of their OpenRouter cells pass). When it is saturated, the failover attempt gets OpenRouter's 429,
relayed with its `Retry-After` and with the account remedy rewritten (D174). Expect that
occasionally on these failovers. It is not a gateway fault. Likewise, a candidate the host
reserves for Enterprise or dedicated deployments is not listed
(Groq `llama-3.1-8b-instant` / `llama-3.3-70b-versatile` / `minimaxai/minimax-m2.7`, the
Fireworks Llama 4 / Kimi K2.6 / GLM 5.1 / GLM 5.2 / Llama 3.3 ids, Together Kimi K2.7 Code, GPT-OSS 20B,
Gemma 4 31B and Qwen2.5 7B Turbo).
Rows left with only OpenRouter keep their names as OpenRouter-only rows. These ids are recorded in
`verify/catalog_truth.toml`, and `no_candidate_is_retired_or_not_serverless` keeps them out. Those
rows have no Responses arm — `previous_response_id` is OpenAI's store. OpenAI serves some GPT ids only
on the Responses API (`gpt-5-pro`, `gpt-5.x-pro`, `gpt-5.3-codex`, `o1-pro`): their OpenAI candidate
is `/v1/responses` (a Chat Completions or Messages client is translated onto it), with OpenRouter
Chat Completions as the failover (not on `o1-pro`: OpenRouter's one endpoint for it takes no tools).
Rows whose first-party API does not serve our keys (the `-pro` ids OpenAI does not serve this
account; the Groq/Fireworks/Together ids above that a serverless key cannot reach) are
OpenRouter-only. Every row and candidate is
verified against the live providers by `catalog_rows_are_servable` in `tests/smoke.rs`, and the
failover itself by `model_route_fails_over_to_a_real_provider`.

**A model its maker retires leaves the catalog, even where another host still runs it.** Claude
Opus 4.1 and Sonnet 4 (retired at Anthropic 2026-08-05 and 2026-06-15) and GPT-5.1-Codex / -Max /
-Mini and GPT-5.2-Codex (shut down by OpenAI 2026-07-23, still in its `/v1/models` listing) were
OpenRouter-only rows served from Bedrock or Azure; they were removed (D181), and a request for one
is a catalog miss (404). The gateway never remaps a requested model to the vendor's successor.
MiniMax M2.7 went too (D182): its one candidate, OpenRouter, sends every forced tool call and
JSON-schema request for it to hosts that answer 410 Gone, so no candidate could serve its card.
`verify/catalog_truth.toml` `[[retired]]` records each retirement with the vendor's notice: a
`retired` date keeps the row and its candidates from coming back
(`no_candidate_is_retired_or_not_serverless`, `no_catalog_row_outlives_its_retirement`), and a
scheduled `retires` date keeps the row until that day, when the same test fails until it is
removed (Claude Sonnet 4.5 on 2026-11-30; gpt-4, gpt-4-turbo, gpt-4.1-nano, o1, o1-pro, o3-mini and
o4-mini on 2026-10-23; gpt-5, -mini, -nano, -pro, o3 and o3-pro on 2026-12-11). `GET /v1/models`
does not carry the date: neither the OpenAI list shape nor Anthropic's `/v1/models` has a
deprecation field. The live cells `CAT-16::raw::{provider}::{row}` (`crates/verify/tests/catalog_live.rs`,
listing calls only) hold every candidate to its vendor today: listed by the vendor's own models
API (Together: in its serverless table; Bedrock: an `ACTIVE` inference profile over a model that
is not `LEGACY`), or, on OpenRouter, at least one endpoint up in the last 30 minutes and one that
takes each capability the card lists; and every deprecation notice the vendor (or, for an
OpenRouter slug, its maker) publishes for the id is recorded. `CAT-16::raw::{anthropic,openai,xai}::new-models`
fails on a model those vendors list in a family the catalog carries (Claude; GPT and the o-series;
Grok; text generation only, aliases not snapshots) that is neither a row nor recorded in
`[[not_carried]]` with a reason, so a release shows up as a red cell. Together and OpenRouter list
hundreds of models, so their gaps (recent models in the namespaces the catalog carries) are a
report, `VERIFY_CATALOG_GAPS=1`, not a failure.

**What that failover does and does not cover.** On the two Bedrock-backed rows, Bedrock is the
independent second source: a different account, a different network path, and AWS's own serving of
Claude. OpenRouter is the last candidate and covers failures that are on our side of the wire
(egress blocked, our Anthropic or Bedrock key throttled) but is not independently guaranteed — it
picks its own backend per request and has been observed serving these ids from Anthropic directly
_and_ from Bedrock. A 5xx from Anthropic on those rows therefore fails over to Bedrock first;
OpenRouter is what is left if Bedrock is down or unkeyed too. Other Claude rows still go
Anthropic → OpenRouter until a live-verified Bedrock inference-profile id is added.

### Stored-response references (`item_reference`)

**What it is.** A Responses input item `{"type":"item_reference","id":"msg_…"}` stands in for an
item of an earlier response that the upstream stored: OpenAI expands it from its own store. The
Vercel AI SDK's default OpenAI model (`openai(model)`, the Responses API) sends no `store`, so it
takes the upstream to be keeping every response, and on every later call it sends each earlier
assistant text or reasoning item that has an id as an `item_reference`
(`convertToOpenAIResponsesInput` in `@ai-sdk/openai`, gated only on
`providerOptions.openai.store ?? true` and the part's `itemId`). Client-executed tool calls always go
in full: a `function_call` and its `function_call_output`.

**Why the gateway cannot resolve it.** The gateway is stateless: it stores no customer content, so
it holds no copy of an answer it translated. On a row with no Responses arm (Claude, Bedrock,
OpenRouter, Together, the failover rows, grok) no upstream holds the item either: Anthropic and
Together have no such store, a translated response was never stored anywhere, and grok goes to xAI
as `store: false` (D145; xAI answers an `item_reference` 422 "unknown item type"). Nor can the response
steer the SDK into sending items in full: it takes `itemId` from each output item's required `id`,
fixed at `response.output_item.added` before the content exists, and never reads
`response.store`. A translated response still says `store: false`, truthfully.

**The rule** (`translate::turn_reference_in_input`, in the same single pass over `input` as the other
session fields; no extra parse). Read `input` as model steps: a **step** is a maximal run of
`item_reference`, `reasoning`, `function_call` and `custom_tool_call` items, ended by any other item
(a user or assistant message, a tool output) or by the end of `input`.

- A reference in a step that holds a call is the preamble text or reasoning of the step that made
  the call. It is **dropped**: translation skips it, and a Responses relay off the row (grok's xAI
  candidate) cuts it from the body (`translate::strip_item_references`). The call and its output
  are sent in full. This covers several references in a row, a reference then a reasoning item then
  the call, parallel calls, text the model wrote after the call (a reference between the call and
  its output), and every step of a multi-step loop.
- A reference in a step with no call stands for an earlier **turn**: followed by a user message,
  at the end of `input`, the whole `input`, or before a tool output with no call in its step. It is
  a **400** `invalid_request_error` that names `item_reference` and the remedy, before any upstream
  is contacted (`translate::session_field_refusal`).

**Why.** Dropping a turn means the model answers a conversation without its own earlier answers, and
says so with confidence: live, Claude replied "NO-REPLY" when asked to repeat what it had said, and
Llama invented a different number. That is never acceptable silently, so it is a named refusal. A
tool step's preamble ("Let me check the weather.") or reasoning is not a turn: the call and its
result say what the step did, the model loses nothing it needs, and refusing it would break every AI
SDK tool loop whose model wrote a sentence before calling a tool (D175's first fix did exactly
that).

**GPT rows** relay every `item_reference`, a tool step's and a turn's alike, to OpenAI's Responses
arm byte for byte: OpenAI holds the items it stored, and resolves them.

**OpenRouter's own Responses API** (verified live 2026-10-01) silently drops an `item_reference` it
does not know and answers anyway: the silent loss this rule exists to prevent. Claude rows reach
OpenRouter over Chat Completions, so the gateway's rule applies before it.

**Client remedy.** Send earlier items in full. With the AI SDK's OpenAI provider on a non-GPT model,
set `providerOptions: { openai: { store: false } }` (the SDK then sends every item as content), or
use a provider that never sends references: `@ai-sdk/anthropic` (the `/v1/messages` wire) or
`@ai-sdk/openai-compatible` (Chat Completions). The 400 says this.

### Identity (`key.rs`)

Virtual key format: `bai_vN.{kid}.{payload}.{sig}`. Both versions are **stateless Ed25519** — no
database, no network call. The keyring holds multiple `kid` → public key mappings simultaneously
(zero-downtime rotation: add the new kid, deploy, remove the old kid). A tampered or forged key
that carries a managed prefix (`bai_v1` / `bai_v2`) is **fail-closed: 401**, never BYO. One
branch in `request_filter`: prefix match → verify; verify failure → 401. Anything else is BYO.

**v1** payload is 16 bytes little-endian: `tenant_id u64 || vpc_id u64`. `mint(tenant, vpc)` is
deterministic. There is no per-credential identity, so a v1 token can only be cut off by
`blackhole.{tenant}`.

**v2** payload is 24 bytes little-endian: `tenant_id u64 || vpc_id u64 || key_id u64`. `mint_v2`
takes an **explicit** `key_id` (not derived from tenant+vpc). Two credentials for the same
tenant are distinct tokens. Control-plane mint (Go) must match this layout exactly:

```
token       = "bai_v2" "." kid "." payload_b64 "." sig_b64
payload     = tenant_id || vpc_id || key_id     // 24 bytes, little-endian u64s
payload_b64 = base64url(payload)                // no padding, 32 chars
signed      = "bai_v2" "." kid "." payload_b64  // Ed25519 message
sig_b64     = base64url(sig)                    // no padding, 86 chars
```

`kid` must be the canonical decimal `mint` writes: digits only, no sign, no leading zero. The
signed bytes are rebuilt from the parsed kid, so without this `bai_v1.01.…` and `bai_v1.+1.…` would
verify as the same key while the rate guard, which keys on the raw token, gave each spelling its own
bucket.

`vpc_id` is still not an access check — decoded and emitted on `ai.usage` only. `key_id` is
emitted on `ai.usage` (`None` for v1).

Verification cost ≈ 28µs per request — this is the gateway's only meaningful per-request CPU cost
(everything else runs in nanoseconds; see Benchmarking). The rate guardrails sit **before** verify
precisely because of this: a forged-key flood is rejected in tens of nanoseconds, not 28µs each.

### Model Extraction (`peek.rs:ModelScanner`)

A streaming structural scanner fed body or response chunks as they arrive. Tracks JSON nesting
depth, string-escape state, and quote boundaries. Captures the **root-level `model` field only**
(depth 0 in the object), ignoring nested `model` keys in tool calls or message content.
SIMD-accelerated via `memchr2` to skip over large string values (base64-encoded images, long
prompts). Root keys and the `model` value are accumulated only up to `MAX_CAPTURE` (256) bytes, so
a multi-megabyte `model` string is never materialized — past the cap it can only miss the catalog
or log as `unknown`. O(1) memory: one struct, no heap growth with payload size — proven by the unit bench
which shows a single allocation independent of whether the body is 0 bytes, 4 KB, or 256 KB.

The billing fact carries **two model fields**:

- `requested_model` — what the client sent (extracted from the request body)
- `model` — what the provider resolved and billed (extracted from the response head; falls back to
  `requested_model` when the response carries no model field, e.g. an error body)

`model` is what reconciles against the provider's invoice (which itemizes by pinned snapshot, e.g.
`gpt-4o-2024-08-06`, not alias). `requested_model` serves product analytics and as a fallback rate
when the snapshot is newer than the downstream price table. The response scanner reads a root
`model`, or one nested a single level under a root `message` (Anthropic `message_start`) or
`response` (Responses `response.created` / `response.completed`) key.

A third field, `price_model`, names the catalog row the row prices at. A model-routed row already
knows it (`routed_model`); a provider-routed one resolves `model`, then `requested_model`, through
`providers::catalog::for_model`, each as spelled and then without a dated snapshot suffix
(`-YYYY-MM-DD`, `-YYYYMMDD`), since the catalog lists aliases and vendor slugs, never snapshots.
Absent when neither names a row: an unpriced model.

### Usage Extraction (`usage.rs`)

The tail tap feeds the parser after `logging` fires. It reads plain bytes, so managed requests go
upstream with `Accept-Encoding: identity`: stock Python and Node SDKs send `gzip`, OpenAI and
OpenRouter honor it, and a gzipped body parsed as no usage (a **zero-token** billing row) and filled
the cache with gzip bytes that a hit replayed without their `Content-Encoding`. BYO requests keep
the caller's own header. Two dialects:

| Dialect   | Format     | Fields                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| OpenAI    | JSON body  | `usage.prompt_tokens`, `usage.completion_tokens`, `usage.prompt_tokens_details.cached_tokens`, `usage.prompt_tokens_details.cache_write_tokens` (OpenRouter's Claude cache writes), `usage.completion_tokens_details.reasoning_tokens` (Responses API: `usage.input_tokens`, `usage.output_tokens`, `usage.input_tokens_details.cached_tokens`, `usage.input_tokens_details.cache_write_tokens`, `usage.output_tokens_details.reasoning_tokens`) |
| OpenAI    | SSE stream | Terminal `data:` line (before `[DONE]`), same fields (Responses API: nested `response.usage`, `output_tokens_details.reasoning_tokens`)                                                                                                                                                                                                                                                                                                          |
| Anthropic | JSON body  | `usage.input_tokens`, `usage.output_tokens`, `usage.cache_read_input_tokens`, `usage.cache_creation_input_tokens`, `usage.output_tokens_details.thinking_tokens`                                                                                                                                                                                                                                                                                 |
| Anthropic | SSE stream | `message_start` input and cache counts; `message_delta` event with `usage` block (thinking tokens on the same block). Its cumulative input and cache counts, when present (server tools), supersede `message_start`'s                                                                                                                                                                                                                            |

**Reasoning tokens outside `completion_tokens`.** OpenAI counts reasoning inside
`completion_tokens`. xAI reports it beside it (`total_tokens = prompt_tokens + completion_tokens +
reasoning_tokens`) and bills it at the output rate. When a body's arithmetic shows the second
convention, `output_tokens` on the row is `completion_tokens + reasoning_tokens`. Before this, grok
reasoning rows billed only the visible answer (a live grok-4.3 call: 7 of 174 output tokens). On
Responses the same arithmetic runs on `input_tokens` / `output_tokens` / `total_tokens`: xAI's
Responses API counts reasoning inside `output_tokens` (measured against its `cost_in_usd_ticks`),
but its API reference example shows it beside, and either shape bills it once.

**Provider counts are untrusted numbers.** Every sum of provider-supplied counts (this arithmetic,
the translated `usage` a client is shown, and the cut-short estimators' scaling) saturates at
`u64::MAX` instead of overflowing. Release builds keep `overflow-checks` on, so an unchecked `+` on
a garbage count would panic in logging and lose the billing row; a nonsense count now bills a
saturated value instead.

Missing or zero usage fields deserialize to zero (safe default) — **except** `reasoning_tokens`
(`Usage::reasoning_tokens: Option<u64>`), which stays `None` when the provider didn't report it at
all, distinct from `Some(0)` when it reported a real zero; that distinction is unrecoverable once the
request completes, so it's never collapsed to a bare zero. If the tail is truncated by the compaction
drain, the usage chunk is still present because SSE usage is always the final `data:` line and the
tail keeps the last 64KB. OpenAI's parser walks those lines backwards and stops at the first usage
block; the split is `memrchr`, the same reason the forward Anthropic walk uses `memchr` — a tail
with no usage block still has to scan all 64 KiB.

SSE lets one event carry its data on several `data:` lines, joined with `\n` (JSON whitespace), and
the translator and every spec-following SDK join them. Both parsers read line by line, so a line
that names `usage` but is not JSON alone sends them to a slow path that re-reads the view event by
event, each event's `data:` lines joined (each line alone if the joined text is not one JSON value,
for a stream with no blank lines between events). Before this, a usage event written that way
billed an estimate or zero while the client was shown exact usage (D126). OpenAI and Anthropic
write one line per event, so their streams take the slow path only when a retained head or tail
was cut through a usage line (OpenAI's walk, which stops at the first usage block, not even then),
and it bills what the fast path would. The 64 KiB Anthropic tail still measures 9.5 µs.

**`input_tokens` follows its wire, and the row says which.** OpenAI's `prompt_tokens` (and the
Responses API's `input_tokens`) include cached tokens; OpenRouter's also include its Claude cache
writes. Anthropic's `input_tokens` excludes both cache reads and cache writes. So the same Claude
prompt served by Anthropic and by OpenRouter Chat Completions reports `input_tokens` 100 and 1000
for 900 cached. The rows keep each wire's own meaning (downstream consumers rely on it) and carry
`usage_wire` (`openai` | `anthropic`) naming the convention: the whole prompt is `input_tokens` on
`openai`, and `input_tokens + cache_read_tokens + cache_write_tokens` on `anthropic`. It is set by
the extractor that read the usage, so a cache hit replays the convention its fill was parsed under;
an estimate with nothing parsed takes the request's wire.

**A zero-token row says why.** Every row carries `outcome` and, when a response head arrived,
`upstream_status` (the provider's HTTP status; absent on a cache hit, which made no call):

| `outcome`          | When                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------ |
| `ok`               | A complete response, or a cache hit                                                  |
| `upstream_error`   | The provider answered 4xx/5xx, or failed before any response head (connect, timeout) |
| `client_cancelled` | The client went away before the stream's terminal event reached it                   |
| `no_candidate`     | No provider was called — every candidate's breaker was open                          |
| `cut_short`        | A response started and then died upstream                                            |

**A stream is complete when its protocol says so.** Chat Completions ends at `data: [DONE]`,
Messages at `message_stop`, Responses at `response.completed` (or `response.incomplete`). Stock
clients close as soon as they have it: openai-python breaks out at `[DONE]`, Codex closes at
`response.completed`. When that close reaches the gateway before the provider's own end of stream,
pingora ends the request with a downstream error, though the client has the whole answer. So on a
managed stream `response_body_filter` feeds the bytes the client is sent (after translation) to a
`terminal::TerminalTracker`, which reports whether they end with a whole terminal event (its first
line, and the blank line that dispatches it; reading only each chunk's last event, three bytes of
state). In `logging`, a downstream error after that is the request completing: `outcome` is `ok`,
the cache fill runs, the capture is `complete` (D120, D122). No timer is involved. Pingora awaits
each chunk's write to the client before it polls the client again, so a terminal event the tracker
saw was written, unless that write failed, and then the error is the write's
(`WriteError`/`WriteTimedout`), which stays `client_cancelled`.

On `no_candidate` the row has no `provider`: the request's provider is only the walk's seed then,
and naming it would bill a call that never happened. `RequestCtx::upstream_phase` (none → attempted
in `upstream_peer` → connected in `upstream_request_filter`) is what tells the cases apart.

**Priced variants and per-call fees** ride on the row next to the token counts:

| Row field                    | Source                                                                                                                                   |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `cache_write_1h_tokens`      | Anthropic `usage.cache_creation.ephemeral_1h_input_tokens` — a subset of `cache_write_tokens` (2×)                                       |
| `gateway_cache_write_tokens` | Cache writes caused by breakpoints the gateway added: already in `input_tokens`, not in `cache_write_tokens` (billed at input)           |
| `server_tool_calls`          | Anthropic `usage.server_tool_use.web_search_requests` (cumulative, on `message_delta` when streamed)                                     |
| `service_tier`               | `service_tier` as echoed: Chat Completions root, Responses `response`, Anthropic `usage`; absent if not echoed or not `[a-z0-9_-]{1,16}` |

OpenAI reports no hosted-tool call count in `usage`, so `server_tool_calls` is 0 there. A
malformed `service_tier` never fails the usage parse: it reads as absent.

A final event can itself be bigger than the tail: a Responses `response.completed` echoes the
request's instructions and tools ahead of `usage`, so a Codex-sized prompt puts it past 64 KiB and
the tail begins mid-way through it. When no whole line carries usage, the tail's first line — the
only one that can be front-truncated — is searched for its last `"usage"` and the object after it
is deserialized, the same recovery a front-truncated non-stream body gets. One `memrchr` over at
most the tail, and only on a miss.

**Dialect-mismatch guard:** a config-added provider whose `provider_dialects` value doesn't match its
actual wire (e.g. an Anthropic-wire vendor left at the default OpenAI dialect) would otherwise have
its `usage` block parsed by the wrong dialect's parser. Because both parsers' fields are
`#[serde(default)]`, that misparse used to succeed silently — every field defaults to zero, producing
`Some(Usage::default())`: a zero-token billing row indistinguishable from a real (and wrong)
zero-usage response. `openai_body`/`openai_stream` and `anthropic_body`/`anthropic_stream` now check
for the _other_ dialect's characteristic field names (Anthropic's `input_tokens`/`output_tokens` vs
OpenAI's `prompt_tokens`/`completion_tokens`) before accepting a parse, and return `None` on a match —
tripping `usage_parse_errors_total` (see Metrics) instead of a silent zero-billing row. A non-stream
Responses body (`/v1/responses`, `/v1/responses/compact`) uses Anthropic's key names too; its
`total_tokens`, which Anthropic never sends, is what tells it apart. Before that check, every
non-stream `/v1/responses` call billed zero.

### Streams cut short (`usage.rs` estimates)

A stream's usage block is its **last** event. A client that hangs up first — a cancelled agent
turn, routine for a coding agent — or an upstream that dies mid-stream leaves nothing to parse,
while the provider still bills us for what it generated before it noticed. Emitting zero there made
"stream a long answer, disconnect one event before the end" free.

A managed 2xx stream is **cut short** when its usage never arrived: no parseable usage on the
OpenAI wire, no parseable usage or no `message_delta` event on the Anthropic wire (`message_start`
alone parses, so a successful parse is not the same as a finished stream). The event is found
structurally — an `event: message_delta` line or a `"type":"message_delta"` member — never as the
bare words, which generated text can contain but cannot forge as a line or an unescaped member.
The 2xx head is the provider's word that it took the request, and it bills the prompt from there, so
a stream that goes silent or dies before its first event is estimated too (D123: it used to need
`message_start`, a relayed delta or a clean finish, and a provider that answered 200 and stalled
billed a generation the row wrote as 0/0). The one exception is a 200 stream carrying only an error
event (`"error":{`, an `overloaded_error` before any output): not work we were billed for. A stream
that **finished** cleanly without readable usage (a provider shape change, a final event past
recovery) is a billed turn too. Its row carries an estimate and `usage_estimated=true`;
`ai_usage_estimated_total` counts them.

- **Input:** Anthropic's `message_start` is the first event and carries exact input and cache
  counts, so those are kept. Otherwise the prompt text's **pre-tokens**, counted as the body streams
  past (`InputTally`, 12 bytes of state). A BPE tokenizer splits text with a pre-tokenizer before
  merging and never merges across a split, so the split count is at most the token count whatever
  the vocabulary. The splits counted are the subset of the GPT tokenizers' stated regex
  (`cl100k_base`, `o200k_base`, Llama 3's) every measured tokenizer honors: words split at
  whitespace, one per letter run (an apostrophe between letters stays in it), one per three digits,
  one per punctuation run unless it is a single byte before letters (`(foo`); a byte past ASCII is
  left out, which can only join what it separated, and a word of only such bytes counts one. Only the
  string values of prompt keys count (`content`, `text`, `role`, `system`, `instructions`, `input`,
  `output`, `arguments`, `name`, `description`). Keys, structure, the model id and every other
  parameter do not: counted as bytes ÷ 5 they billed a 24-token prompt at 40 (D99). Binary payloads
  (a data URI under `url`, an Anthropic source's or `input_audio`'s `data`) are values of other
  keys, skipped with `memchr` at ~100 GB/s; text is classified four bytes per table lookup, ~1.3
  GB/s on source code and ~3 GB/s on prose (`benches/unit.rs` `input_tally`). Measured 2026-10-01
  on prose, code, Markdown, JSON, numbers, hex, base64-like text, French, Russian, CJK, emoji and
  punctuation runs: never above the count of `o200k_base`, `cl100k_base`, grok-4.3 (xAI's tokenize
  endpoint) or Claude Haiku 4.5 (`count_tokens`), about 0.8× of it on English and code for the GPT
  and grok tokenizers, 0.65× for Claude's denser one, and far less on non-Latin scripts. The
  provider's own template tokens (role markers, a system preamble) are not in the body at all.
- **Output:** only the 64 KiB tail is retained, so the tail is measured (delta events and text bytes
  per byte of stream) and scaled up to the bytes relayed over the whole stream — one add per managed
  stream chunk, no scan on the relay path (counting `data:` per chunk measured +9.5% on a 600 KiB
  Anthropic stream). The estimate is the larger of one token per delta
  event (exact on OpenAI) and text bytes ÷ 4.5 (the floor for providers that batch tokens).

The output divisor was measured against live providers and rounded to under-count: an estimate is a
bill the customer cannot check. On five real streams each cut at 50% and 99% of their bytes, the output
estimate ranged 0.94–1.05× on OpenAI Chat Completions and 0.86–1.05× on Responses (the one over-count:
a Responses stream cut halfway, where the preamble events skew the byte scaling), 0.84–0.85× on
Claude Haiku 4.5, 0.75–0.84× on Claude via OpenRouter, and 0.57–0.60× on
Claude Sonnet 5 (a denser tokenizer). Hidden reasoning — an OpenAI reasoning model, Claude with
thinking display omitted — is invisible to both.

Two more endings the provider bills and we used to write as zero get the same treatment:

- **Given up on before the response head**, after the provider had the whole request: a long
  reasoning turn or a huge prompt the client gave up on, or that outlasted the gateway's own
  `read_timeout_secs` (its 504) on a connection that was still up (D130). The provider cannot tell
  who hung up and bills the prompt either way; billing only the client's cancel made the SDK's retry
  of a 504 hide one (BIL-14). Input is estimated from the tally, output is 0 (`gave_up_waiting`).
  "Had the whole request" is `body_delivered`: the connection was up (`upstream_request_filter`
  ran), the client's body was read to its end, and nothing failed writing it upstream. A request
  that never reached a provider (every breaker open, a connect failure, a reset mid-upload) stays
  unbilled, and so does one the upstream **closed or reset** before any head (the gateway's 502): the
  peer declined to answer, and nothing says it processed the request. The real edges in front of
  Anthropic, OpenAI and OpenRouter answer a request they forwarded and then lost with an HTTP error
  (Cloudflare's 52x, Envoy's 503 local reply), not a bare close, so the close is read as a refusal,
  the side an estimate the customer cannot check errs on (BIL-20). A dead peer (TCP keepalive or
  `TCP_USER_TIMEOUT`) is the same: it may never have read the request.
- **A non-stream 2xx body cut off** before its `usage` (which is last): the provider generated the
  whole answer. Input from the tally; output from the bytes inside JSON string values in the tail
  (generated text, not keys or structure), scaled to the bytes relayed and divided by 4.5. Managed
  responses count relayed bytes whether or not they stream.
- **A non-stream 2xx that ended cleanly without usage** (D195): OpenRouter answers a generation
  that failed partway with a 200 whose `usage` is `null` beside an empty or partial message
  (live 2026-10-01 on `x-ai/grok-4.20-multi-agent`), and documents that the upstream may still bill
  the prompt. Such a body wrote a 0/0 row while the same turn streamed was estimated. It is
  estimated as a cut-off body is: input from the tally, output from the body's string values. A
  body that is only an error object (OpenRouter's non-stream error-in-200, `{"error":{…}}` with no
  answer; read whole when the body fits the tail) is the exception, as an error-only stream is: no
  generation, a 0/0 row, not an estimate. A free sub-resource (a token count) is never estimated.

A stream or non-stream body that ends **cleanly** without usage still counts on
`ai_usage_parse_errors_total` (the wire-shape-change alarm) even though it is now billed an
estimate. An estimated response is never
stored in the response cache.

### Deny-Set (`deny.rs`)

Two `HashMap<u64, DenyReason>`s — tenants (`blackhole.{tenant}`) and credentials
(`blackhole.key.{id}`). Only denied ids are stored: `O(denied)` memory, not `O(tenants)`. A
request is denied if the tenant is in the set **or** (v2) its `key_id` is. Tenant deny wins when
both match and kills every key for that tenant. v1 tokens have no `key_id`, so only the tenant
map can deny them. Both key shapes live under the `blackhole.` prefix; one `WatchedSet` watcher
applies both. Written exclusively via `ArcSwap`; reads on the hot path are lock-free.

Reasons: `Spend` (→ 402), `Fraud` (→ 403), `Unknown` (→ 403, fail-safe for unrecognized values).
Restore = explicit delete from NATS KV or TTL expiry — no gateway-side timer.

### Allowance-Set (`allowance.rs`)

Same WatchedSet / ArcSwap shape as deny, inverted fail direction. Membership = exhausted
(`allowance.{tenant}`, `allowance.key.{id}`). After a successful scan or snapshot — **including
an empty one** — absence is remaining-ok. Until that read, every managed request gets a `503`
with `Retry-After: 5` (`allowance_unavailable`) and does not call `upstream_peer`. A `503`, not a
`402`: the set not being read yet is transient, and stock SDKs treat a `402` as final, while
"quota exhausted" (an exhausted tenant or key) stays the final `402`. v1 tokens have no `key_id`, so only
the tenant grain applies. The control plane writes the bit; this is not a price table and the
gateway does not decrement a remaining counter. Restore = delete the KV entry.

The exhaust bit is written only after usage has shipped and been summed downstream, so a tenant
keeps being served for that lag. The overshoot is `lag × requests in flight × cost per request`;
`tenant_max_in_flight` (`concurrency.rs`) bounds the middle term per process — a sharded, sparse,
exact counter per tenant, claimed after the cache and before the breaker, released in `logging`.
Off by default: the right ceiling depends on how many agents a tenant legitimately runs in parallel.

Checked after deny, before the exact-match cache, so an exhausted key cannot be served from a
cached 2xx.

### Revocation: how fast a change lands, and streams in flight

The contract for a written or deleted `blackhole.{tenant}`, `blackhole.key.{id}`,
`allowance.{tenant}` or `allowance.key.{id}` entry (SEC-17, TEN-1):

- **A connected gateway applies it within 2s.** The path has no gateway-side timer: the NATS KV
  watch pushes the delta (a JetStream push consumer, `watch_prefix_from`), `recv_many` takes it as
  soon as one is queued, and one `rcu` publishes the new set. Every `request_filter` loads the set
  afresh, so the first request admitted after the publish sees it. The real cost is one NATS
  delivery: 17–50 ms live (`tenancy_live.rs`). The 2s is the stated bound with that headroom, and
  `a_deny_lands_within_the_bound_and_in_flight_streams_finish` holds it. A delete restores service
  the same way.
- **A disconnected gateway applies it after reconnecting.** While the watch is down the last-known
  set keeps serving (fail-open for deny; allowance after its seed), so a change written during the
  outage is not in force. The watcher reconnects on a backoff of 1s doubling to 30s
  (`RECONNECT_BACKOFF_MAX`), then resumes strictly after its saved revision and replays what it
  missed (or rescans if the history was compacted). Once NATS is reachable again, the change lands
  within one backoff step, at most 30s plus the connect. A connection that dies silently is
  detected by the NATS client's own ping (async-nats 0.46 pings every 60s and drops the connection
  at the third tick with two pings unanswered, so within about three minutes), and until then the
  set is stale. `/readyz` reports `degraded` while the deny watcher is disconnected.
- **A stream already in flight runs to completion and is billed exactly.** Deny and allowance are
  checked once, at admission in `request_filter`, before the cache and `upstream_peer`. Nothing
  later consults them: not a pingora retry, a catalog failover or a key walk inside the admitted
  request, the response relay, or `logging`, which writes the normal row with the provider's own
  usage. The control plane revokes the next request, not a generation already paid for. The one
  re-check is a large-body request's next attempt (`relay_full_body`, after a 5xx, 429 or reset),
  which is a subrequest of its own and goes through `request_filter` again, so it is refused if the
  change landed in between. The attempt it replaces was never relayed to the client, so no stream
  is cut.

### Control surface (`control.rs`) and payload capture (`capture.rs`)

The `x-beyond-*` headers are the per-request control surface, parsed once in `request_filter` after
identity is verified. Every `x-beyond-*` request header — these, `x-beyond-model`, and any name the
gateway does not define — is stripped in `upstream_request_filter` before the request leaves, on
every route, BYO included.
**Managed only** — a BYO request carries no verified `tenant_id`, so a tag on it would be an
unattributable row, the same reason `ai.usage` is managed-only.

| Header              | Value                                       | Effect                                          |
| ------------------- | ------------------------------------------- | ----------------------------------------------- |
| `x-beyond-metadata` | flat JSON object of scalars, ≤1KB, ≤16 keys | tags `ai.usage` + `ai.payload`                  |
| `x-beyond-capture`  | `on` / `off`                                | enables or suppresses capture for this request  |
| `x-beyond-cache`    | `on` / `off`                                | `off` skips exact-match cache lookup and store  |
| `x-beyond-order`    | comma-separated `ProviderSpec::name`s       | those providers first, then the rest of the row |
| `x-beyond-only`     | comma-separated `ProviderSpec::name`s       | drop anyone on the row not named                |
| `x-beyond-split`    | `name=weight` pairs                         | weighted primary; leftover stay failover        |

**Nothing here can fail a request with a 4xx.** Malformed, oversize, or unrecognized values are
dropped and counted on `ai_control_header_errors_total`; the request proceeds as if the header were
absent. An observability header that can 400 a customer's inference call is a worse bug than the
missing observability — and on a proxy, the header we would be rejecting was written by _their_
SDK. An `only` filter that leaves no pool-keyed candidate is a routing 503, the same as a row with
no keyed provider — not a parse rejection.

**Metadata is re-serialized, never passed through.** The client's JSON is parsed, validated, and
re-emitted from the parsed values with keys sorted. Log injection is therefore structurally
impossible rather than filtered-for, and identical tags always render identically.

Capture has **two enablers and one suppressor**, because they serve different people:

| Source                       | Enable | Suppress | Serves                                                            |
| ---------------------------- | :----: | :------: | ----------------------------------------------------------------- |
| Control plane (`aicapture.`) |   ✓    |    —     | operator; retroactive; needs no cooperation from the client       |
| `x-beyond-capture`           |   ✓    |    ✓     | caller; per-request precision — "log this trace" / "not this one" |

The header wins in both directions. The control-plane path is the one the founding use case needs:
when a user reports that the agent did something stupid, the request has already happened and the
client cannot be changed. A header-only design would require the customer to ship code before we
could help them, and would make debugging a _misbehaving_ client impossible.

An explicitly requested capture is **never sampled away** — a caller who asks to log one trace and
silently gets nothing is the single outcome that makes the feature useless. Sampling (`sample_n`)
bounds only control-plane-enabled capture.

**Enablement expiry costs zero gateway code**: the control plane writes the `aicapture.{tenant}`
entry with a slipstream TTL, and its expiry arrives as an ordinary `Delete` delta.

### Served-by response headers

Every response says what served it, so a client sees a failover or a cache replay without a log
search:

| Header                    | When                                        | Value                                                                       |
| ------------------------- | ------------------------------------------- | --------------------------------------------------------------------------- |
| `x-beyond-request-id`     | every response                              | the `request_id` on `ai.usage` / `ai.payload`                               |
| `x-beyond-provider`       | every upstream response, and a cache replay | the provider that answered (on a replay, the one that originally served it) |
| `x-beyond-upstream-model` | catalog walks (`/auto`, managed `/v1`)      | the model id as the gateway sent it to that provider                        |
| `x-beyond-cache-status`   | cache replays only                          | `hit`                                                                       |

The namespace is the gateway's. Any `x-beyond-*` header on the upstream response is removed before
the gateway adds its own, so a provider (or anything between us and it) cannot tell the client who
served a request or that it was a cache replay.

Cost is deliberately **not** returned. The gateway never prices a request (see "Why the catalog has
a list price and the request does not"): the billed amount is decided downstream and can differ
from list price, so a gateway-computed figure could disagree with the invoice. A per-request cost
lookup belongs to the control plane, keyed by `x-beyond-request-id`. The provider header value is
precomputed per provider at boot (`Provider::name_header`), so it is a refcount bump per response.

### Capture is a tap, not a buffer

Capture copies each chunk as it passes and **never withholds a byte** — the same passive-tap
contract as the usage tail beside it. Nothing is rewritten, so there is no added client latency,
only bounded memcpy and memory. `tests/capture.rs` asserts byte-identical relay in both directions
with capture on; that is the invariant that would make the feature unshippable if it broke.

Two details that follow from where the taps sit:

- **Head, not tail** — deliberately the inverse of `UsageTail`. Usage lives at the _end_ of a
  response; meaning lives at the _start_ of a request. Truncating from the front would discard the
  system prompt, which is the part that explains what the agent was told to do. Truncation is
  flagged, because a capture that reads as complete when it isn't produces confident wrong
  conclusions mid-incident.
- **Pre-rewrite bytes** — the request tap runs on the chunk as it arrives, before both end-of-stream
  rewrites (`stream_options` injection and the model-routed `model` re-spelling). Capturing a
  `stream_options` the client's SDK never sent, or a `model` re-spelled for whichever failover
  candidate served, sends an engineer hunting through their own code for something the gateway put
  there.

Bodies only, never headers — so no pool key or `Authorization` value can reach a capture. Partial
captures are kept and marked (`complete: false`): `logging` runs on upstream errors and client
disconnects too, and "the stream died at token 400" is frequently the answer.

### Payload egress (`capture_sink.rs`)

`ai.usage` keeps the blocking stdout writer and is **lossless** — a dropped billing row is money we
can't account for. `ai.payload` gets a **bounded, lossy** queue drained by its own OS thread, and
overflow drops the line rather than waiting. `init_tracing` installs three layers whose target
filters are exact complements: `ai.usage`, `ai.payload`, and every other target.

`AI_LOG` filters the diagnostic and payload layers only. The `ai.usage` layer has no level filter:
billing rows are not diagnostics, and when one global filter covered every layer, `AI_LOG=warn`
silently dropped every row. It does give `tracing` a max level hint of INFO (the rows' level,
`usage::usage_log_filter`): a filter with no hint made the whole subscriber's max level TRACE, so
`LogTracer` dispatched every pingora `debug!`/`trace!` record only for each layer to drop it. A row
whose stdout write fails (a closed or broken pipe) is counted on
`ai_usage_write_errors_total` and reported on stderr, the only record left of it.

The hazard this removes is real: a synchronous multi-KB write means that if the log shipper stops
draining the stdout pipe, `write(2)` blocks and a _log sink_ is applying backpressure to the proxy.
Observability that can stall the data plane is a worse bug than the missing observability.

Drops are counted on `ai_capture_dropped_total`. That counter is what makes a missing payload
diagnosable ("capture was on and we lost it") rather than ambiguous ("was capture even on?") —
which is exactly the question asked during the incident capture exists to serve.

**Downstream schema.** Payload rows ship on the same logfwd/OTLP path as `ai.usage` and correlate by
`request_id`, which is also returned to the client in `x-beyond-request-id` — so a user quoting that
id resolves straight to their conversation with no join table. Retention and deletion are the
store's concern, not the gateway's: a `TTL captured_at + INTERVAL <n> DAY` clause on the table, and
`ALTER TABLE ai_payload DELETE WHERE tenant_id = ?` for a tenant erasure request. The gateway emits
and forgets. `metadata` arrives as a JSON object string; materialize it as `Map(String, String)` so
`GROUP BY metadata['feature']` answers "which feature is burning money" against the same row that
carries the tokens.

### Response cache (`cache.rs`)

An **in-process** exact-match store — this pod, not the fleet. Identical managed catalog-walk
requests replay a stored 2xx and skip the provider **on the replica that filled it**. Another
replica starts empty. Off unless `cache_ttl_secs > 0`. There is no Redis (or other shared store) on
the miss path: a miss is always the unbuffered upstream relay. `ai_cache_scope{kind="process"}=1`
is the honesty metric; do not alert as if a shared cache were in front of the providers.

**The key is the client request plus the walk, not the upstream attempt.** Hash of the pre-rewrite
body + method + inbound path + `tenant_id` + the resolved catalog row + the `anthropic-version` and
`anthropic-beta` values (`cache::VARY_HEADERS`) + the effective candidate order (`ProviderId`
indices). The row matters because `x-beyond-model` names it, not the body: the body's `model` is
overwritten per candidate, so without the row a `gpt-4o-mini` request whose body matched an earlier
`gpt-4o` one replayed the `gpt-4o` answer whenever the two rows shared a provider set. The two
Anthropic headers change what the provider generates for the same body; the other forwarded headers
do not. Two ahash
passes with a process-secret key — not `DefaultHasher`, which is SipHash-1-3 with zero keys, run
over the whole body on the lookup in front of `upstream_peer`. Not the pool key, not the serving
candidate, not the raw virtual key: a 429 that walks to a second key and then 200s is still one
client request and is stored under that request's body hash. Split/order permute the walk, so each
arm is its own cache entry rather than pinning A/B traffic to whichever provider filled first.
Tenants cannot read each other's entries.

**Lookup only where the body is already in hand** before `upstream_peer` — the headerless managed
catalog walk on `/v1` and `/auto`, the same peek that resolves `model`. BYO and `/{provider}`
passthrough are not cached this round: those paths do not have the client body before connect.
A `Content-Length` at or past the 64 KiB peek cap, or a missing one, skips lookup rather than
draining into pingora's truncated-complete hang.

**A hit short-circuits in `request_filter`.** The stored status, Content-Type, and body are written
to the client; there is no upstream, no breaker permit, no key-walk. `logging` still runs: `ai.usage`
carries `cache_hit` and the tokens stored on the fill, and `ai_cache_hits_total` counts it.

**A miss is still an unbuffered relay.** The fill is a tap — copy, never withhold, the same contract
as payload capture. Insert only on a complete 2xx (a stream whose terminal event reached the
client is complete even if the client then closed first; see "A stream is complete when its
protocol says so"). Client abort, 4xx/5xx, and truncation (the tap
hit `cache_max_bytes`) are all skips: serving a cut or error body as a cached 2xx would be a silent
wrong answer. Cache errors never fail the request: a poisoned lock is recovered, a full store
evicts the oldest insertion, an oversized body is dropped. One TTL means insertion order is expiry
order, so a fill drops only the expired prefix and does not scan the live entries to learn that a
new key is absent. Refreshing a key moves it to the back.

`x-beyond-cache: off` or `Cache-Control: no-store` skips lookup **and** store. `x-beyond-cache: on`
cannot turn a disabled cache on; `cache_ttl_secs = 0` is the operator's off switch.

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
window index from one clock reading per request (`RateLimit::check_at`). `request_filter` passes
the `Instant` it already took at admission, so the limiter does not read the clock again.

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

- **Failure = the provider is broken** — a `5xx` response or a connect failure. The breaker
  **opens** on a failure _rate_: at least `circuit_breaker_threshold` failures within
  `circuit_breaker_window_secs`, **and** at least as many failures as successes since the window's
  first failure. A success no longer wipes the count, so a partial brownout (every other request a
  `5xx`) opens it, which a success-resets rule never did; a busy healthy provider's background
  errors do not, because its successes outnumber them. The window is **fixed**, not rolling: it
  starts at its first failure, and the first failure after it ends starts a new one with both
  counts at zero. Both counts saturate at the same cap (16383), so a failure majority can always
  trip it (the success count once ran to 65535 while failures stopped at 16383, and a busy window
  past that could never open). While open, requests to that provider
  fast-fail with `503` and `Retry-After` set to the seconds left until it half-opens
  (`ai_rejections_total{reason="circuit_open"}`) instead of piling up against the read timeout and
  exhausting connection / in-flight slots for _every_ provider (head-of-line blocking by one sick
  dependency). After `circuit_breaker_reset_secs` it half-opens and admits a probe; success closes
  it, failure reopens it.
- **The probe resolves at its response header**, not at the end of its stream: a `2xx`–`4xx` head
  closes the breaker, a `5xx` reopens it. A probe resolved only at end of stream let one long or
  stalled stream (up to the read timeout) 503 that provider for everyone. A probe permit that is
  still unresolved a whole `circuit_breaker_reset_secs` after it was handed out (a header that
  never came) is reclaimed and a fresh probe admitted; the stalled one's late outcome still lands.
- **A `429` is NOT a failure.** It means the provider is healthy and throttling _that credential_ — a
  velocity/spend signal the rate limiter, the same-provider key walk, and the client's `Retry-After`
  backoff own. Tripping on it would convert a self-healing throttle into a self-inflicted outage. The
  breaker records any response that _arrived_ (2xx/3xx/4xx incl. 429) as a **success**; only 5xx and
  transport failures count against it. A key-walk retry does not record a breaker failure and does
  not claim a second permit.
- **A client giving up is NOT a failure.** Pingora tags a client-side abort `ErrorSource::Downstream`,
  and only non-`Downstream` errors count. Cancellation is routine for a coding agent (a user hits ESC
  on a slow turn); counting those opened breakers on perfectly healthy providers, and because
  `half_open_permits` is 1, a cancel-prone request drawn as the recovery probe reopened the breaker
  every time — so it could not recover while users were cancelling.
- **A client's bad upload is NOT a failure.** A chunked body that crosses the 100 MiB cap is aborted
  with an error tagged `Downstream` (pingora answers 413). An upstream error that ends a request whose
  body was still arriving (the attempt had fed body bytes and the client had not finished) means the
  provider was waiting on the client: a stalled or abandoned upload. Neither counts. Before this, any
  caller, BYO included, could open a provider's breaker for every tenant with a few oversized or
  stalled uploads. A connect failure feeds no body byte, so it still counts. The cost: a provider
  that resets the connection mid-upload is not blamed for it either; one that answers 5xx is.
- **Nor is either of those a success.** An attempt that ends with no provider outcome (no response
  head, and the error is the client's: an abort, a stalled or abandoned upload) gives its permit back
  without one (`CircuitBreaker::release`). Recording a success there closed a half-open breaker on a
  probe that never heard from the provider, and every caller then flooded a provider that may still
  be broken. Released, the probe permit goes to the next request, and the stalled-probe reclaim
  above counts from that new handout, so it stays one probe at a time.
- **Applies to all traffic** (managed + BYO) — a down provider is down regardless of whose key is
  used. One breaker per provider, built at boot, shared lock-free across callers.
- `circuit_breaker_threshold = 0` disables it.

**The permit ledger.** `RequestCtx::breaker_pending` is true **iff** exactly one `allow()` is
outstanding against whatever `provider` currently points at. `response_filter` (a response head
arrived) and `logging` (no head ever did) record only when it is set, and clear it — so one
`allow()` yields exactly one `record_*`, and a scarce half-open probe permit can neither leak nor be
resolved twice.

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

Ed25519 verify is ~26µs — roughly 300–1000× more expensive than every other **always-on**
per-request operation (deny, allowance, rank, the rate-limit check). A flood of forged `bai_v1`
tokens could drive unbounded crypto work if the rate limit came after verify. By checking the
per-credential bucket first (keyed on the raw token, no crypto), a forged-key flood is rejected in
tens of nanoseconds per request. Legit traffic is unaffected: the rate guard passes through, then
verify runs as normal. The unit bench (`benches/unit.rs`) asserts this: `key/verify` ≈ 26µs;
`ratelimit::check` ≈ 70ns single-threaded, ≈ 130–190ns at 16 threads under a flood of distinct
credentials; 0 allocations for either. Two paths are not in that "always-on" set and can sit next
to verify: a cross-wire `translate` of a 64 KiB body (~34µs, once) and, only when the response
cache is enabled, fingerprinting a 256 KiB body (~35µs). See the benchmarking table.

### Why the body injection exception exists (`managed + OpenAI + streaming`)

OpenAI streams no usage chunk unless `stream_options.include_usage: true` is set. Without it, a
streaming managed request is unmeterable: no usage block in the response means no billing fact. The
gateway injects this field server-side so callers using stock SDKs get metered without any
cooperation. The request is buffered (`MAX_REQUEST_BODY` cap), the field injected, and the body
re-framed as chunked upstream. Scoped to managed + OpenAI-dialect + streaming only: a BYO or
non-streaming body is relayed as sent, without the field.

A client that sends `stream_options` itself does not get to turn metering off: `include_usage:
false` is rewritten to `true`, an object without it gains it, and a non-object value is replaced
by `{"include_usage":true}` (`proxy::force_include_usage`, which parses only that value, and only on
the requests that send one). Without the usage chunk the row would be an estimate that cannot see
hidden reasoning. The visible cost: such a client receives one chunk it did not ask for, OpenAI's
usage chunk with `choices: []`, which every OpenAI SDK already accepts.

Nor does a second key. OpenAI's parser keeps the **last** of duplicate keys and decodes escaped key
names, while the scan reads the first raw `stream_options`. So when the root carries more than one
`stream_options`, or any spelled with escapes (`"stream\u005foptions"`), every such member is cut
out by span (`peek::remove_root_members`) and the usual injection adds the one that reaches OpenAI.
`stream` itself is read the same way: a key that decodes to `stream` (`"str\u0065am"`) counts, and
the last one decides, so a client cannot hide `stream: true` from the scan (D88).

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

### Why AWS SigV4 (Bedrock's IAM credential chain) is not supported — and what is

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

**The built-in `bedrock` row is Bedrock's Anthropic Messages API, keyed with an Amazon Bedrock API
key.** AWS now serves Claude on the same `/anthropic/v1/messages` wire Anthropic does, at
`https://bedrock-runtime.{region}.amazonaws.com/anthropic/v1/messages`, authenticated with
`x-api-key: $AWS_BEARER_TOKEN_BEDROCK` (see the Messages API and API-keys docs). That is exactly a
data row: static host (default `us-east-1`, overridable), Anthropic dialect, `XApiKey` scheme, pool
key from `AI_POOL_KEY_BEDROCK` / `AWS_BEARER_TOKEN_BEDROCK`. The catalog puts it second on the
Claude rows whose Bedrock inference-profile ids were live-verified (`claude-haiku-4-5`,
`claude-opus-4-8`), so a 5xx from `api.anthropic.com` on those models fails over to a genuinely
independent supply. Other Claude rows stay Anthropic → OpenRouter until those ids are checked.
The client still sends `anthropic-version: 2023-06-01`; that header is required on Bedrock's
Messages path the same way it is on Anthropic's.

**What this row is not.** It is not Converse, Converse-Stream, or InvokeModel — those use AWS's own
body shape and `application/vnd.amazon.eventstream` framing, which this gateway's usage extractor
and every agent-core dialect decoder assume never arrives. It is not SigV4. It is not the
OpenAI-compat surface at `/openai/v1/chat/completions`, which wants `Authorization: Bearer` and the
OpenAI wire: one `ProviderSpec` has one auth scheme and one `wire`, so that path is a config-added
alias (`provider_authorities.bedrock-openai` + Bearer + OpenAI dialect) rather than a second
built-in. See `config.example.toml`.

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
  Copilot's `RouteOverride::Direct`. This gateway has no WebSocket-upgrade proxying (a managed `Upgrade` request is a 400;
  see "Managed endpoint allowlist"); it is an HTTP request/response (and SSE-response) relay only. A client-side WebSocket transport for Codex could not be relayed through this
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

Two deliberate non-cases, plus one same-provider retry:

- **A `429` is not a vendor failover.** It is a healthy provider throttling _that credential_ — the
  same judgement the circuit breaker makes. Re-asking a different vendor converts a self-healing
  throttle into spend somewhere else. If another unused pool key remains for this provider and the
  body is replayable (`is_body_done && !retry_buffer_truncated`), `upstream_response_filter` returns
  a retryable error and the next attempt stays on the same provider with the next key. Works for
  `/{provider}` and `/auto`. BYO does not walk. The last 429 is relayed, with `Retry-After` if the
  upstream sent one. Counted on `ai_key_walks_total`, never on `ai_candidate_failovers_total`.
- **A managed `401` walks keys first, like a `429`.** It is that pool key revoked, never the
  caller's fault, and the provider refused it before processing anything, so resending the body
  under the next key is safe. It walks the next key on the same provider (never
  a vendor switch at this step) on `/{provider}` and `/auto` alike, counted on
  `ai_key_walks_total`. The key is also cooled off for `KEY_COOLDOWN` (60s,
  `Provider::mark_key_bad`, counted on `ai_key_auth_failures_total`): a new request, or a walk
  entering the provider, starts on its first key not cooling off (`Provider::first_key`), or key 0
  when all are. So rotating by appending the new key and revoking the old one costs one extra round
  trip per pod per minute, not one per request. On the last key, a provider route relays the 401,
  and a catalog walk goes on to the rule below.
- **A managed `403` neither walks nor cools a key.** A 403 is usually about the request: a
  moderation block, a model the key's project may not use, Anthropic's `permission_error`. Walking
  and cooling on it let one tenant move the whole pool off a healthy key for a minute, and resent
  each such request once per key (D84). A provider route relays it; a catalog walk treats it as a
  candidate refusal (below). Only a relayed 403 whose body names the key itself (OpenAI's
  `invalid_api_key` code, Anthropic's `authentication_error` type) cools that key, from the body in
  `logging`; the walk decides on the response head, so that request does not walk.
- **A `5xx` does not walk keys.** Vendor walk already owns that on `/auto`. Keys stay with their
  provider.
- **A managed walk's `401` on its last key, `402` or `403` is a candidate failure.** It is that
  candidate refusing (a revoked or unfunded key, a 403 its vendor may not share), and the next
  candidate holds a different key at a different vendor, so the walk fails over exactly as on a `5xx` (replayable
  body, or the `FullBody` re-run). The candidate takes the ranker's failure penalty and is never pinned, so one revoked key
  cannot black-hole a row by answering fastest. It is **not** a breaker failure: the provider
  answered, so its permit resolves as a success. When every candidate fails this way, the last
  candidate's own status is relayed.
- **A key that is out of credit is cooled, and the walk leaves its provider out (D180).** A
  provider can say "unfunded" under a status the head cannot tell from the request's fault or a
  rate limit: Anthropic's credit-balance `400`, OpenAI's `insufficient_quota` `429`. The walk decides
  on the response head, which reaches the client before the first body byte is read, so that
  request is relayed (its message rewritten, above). `Redact` reads the body
  (`remedy::Neutralized::unfunded`), and `logging` cools that key as a `401` is cooled
  (`ai_key_auth_failures_total`). A catalog walk then leaves out every candidate whose provider has
  all its keys cooling (`Provider::cooling`) while another candidate can take the request, so the
  next request goes to the row's next candidate for `KEY_COOLDOWN` instead of failing on the same
  account; when nothing else can, the cooled provider is tried and its answer relayed. The same
  holds for a provider whose keys all drew a `401`. A provider with several keys that are all out
  of credit is cooled one key per relayed answer (a key walk abandons the earlier keys' bodies
  unread).
- **When every candidate 5xxes, the client gets the last provider's own status**, not a synthetic
  error. Better diagnostics than an exhausted retry loop produces.

**Bodies pingora cannot replay.** Pingora's retry replays its request buffer, which is
`BODY_BUF_LIMIT` = 64 KiB, a private constant with no knob (checked against pingora 0.9.0 and main,
2026-09-30; upstream PR cloudflare/pingora#816 would lift it). Past it pingora sends no body on a
retry and never calls `request_body_filter`, so its own retry would hand the next upstream headers
for a body it never writes.

**On a managed catalog walk this no longer limits failover.** Every body past the buffer is read in
full before the walk (see the peek above) and run as a `FullBody` subrequest. Inside it the body is
still unreplayable to pingora, so where an ordinary walk would retry (a 5xx with another candidate
left, a 429 with another pool key) the subrequest records the decision in its context (`RelayRetry`)
and relays its response. The parent's pipe sees the decision before the response header reaches the
client, abandons that attempt, and runs a new subrequest that skips the failed candidate or resumes
the key walk. Same rules as above: one attempt per subrequest, the last candidate's own status when
every one fails, `ai_candidate_failovers_total` / `ai_key_walks_total` as usual. The breaker and the
TTFT ranker see each failed attempt, since each is a real request. The cost is holding the body in
memory (the model splice already buffered it) and connecting only after it has fully arrived.

**Where it still stops: `/{provider}/…` and BYO.** Those are not catalog walks, so a body past the
buffer takes pingora's path. A 429 there is relayed rather than key-walked, counted on
`ai_failover_unreplayable_total`. That also counts a small body whose 5xx or 429 arrived while it was
still uploading: the gate is `is_body_done() && !retry_buffer_truncated()`, and truncation only
reports on what has been buffered so far, so retrying before the upload finished would make the
same request fail over or not depending on how fast the upstream rejected it. On a path that decides
which vendor gets billed, a deterministic rule is worth more than the extra retries.

**The client's own retry is still a second line.** The stock OpenAI and Anthropic SDKs retry 5xx and
529, and a relayed 5xx marks the candidate failed in the TTFT ranker, which puts it behind every
alternative, so that retry lands on a fallback. `smart_router = false` or a pinned walk
(`x-beyond-order` / `split`) turns the demotion off.

**Pingora 0.9's default refuses to retry a non-idempotent method** — every LLM call is a `POST` —
which silently disabled both walks above. `error_while_proxy` is overridden to keep 0.8's policy:
our walks are marked retryable only after `body_replayable` has proven the body can be resent, and a
reused-connection failure retries only when the replay buffer holds the whole body — and the body
was not yet delivered.

**A delivered body is never sent twice.** A failure with no response — a read timeout, a reset, a
reused connection closed — after the upstream had the whole request means the provider may already
be generating, and billing, the answer. Resending it, to the same candidate or the next, duplicates
that spend (a 200 KiB body with a silent upstream used to reach each of two candidates twice, with
waits up to 2 × `read_timeout_secs`). So `error_while_proxy` resends — pingora's reused-connection
retry, or a `FullBody` reset retry — only when `body_delivered` is false: the connection never came
up, the client's body was not read to its end, or writing it upstream failed (a reset mid-upload
surfaces as a write error). Otherwise the walk ends, and `fail_to_proxy` answers with the gateway's
JSON error and request id: 504 (`upstream timed out after receiving the request`) for a read
timeout, 502 (`upstream failed after receiving the request`) for anything else, with `connection:
close` since pingora drops the client connection after a proxy error. Small and large bodies follow
the one rule, and a small body is read straight through to the upstream (nothing reads it ahead of
connecting), so "the client's body was read" tracks what went out. A timeout is never resent: it is
a liveness verdict on a peer that may have the request.

**A pooled connection reset with the request unread is not delivered.** A **reused** HTTP/1.1
connection reset before any response byte means the provider's kernel answered with RST because the
socket was closed with our request unread in it: the idle-close race on a pooled connection, which
reached no server. `reset_before_reading` resends it on a fresh connection to the same candidate,
like pingora's own reused-connection retry (D80). A clean end-of-file after the body is not
resent: a server that read the request and hung up looks the same as one that closed first. That
is the remaining stale keep-alive cost; pingora's pool watches idle connections for a close, which
keeps it rare, and a client SDK's own retry covers it; a duplicated generation is not recoverable.
The rule trusts the RST: a kernel resets on `close()` only when unread data is left in the socket,
or on an abortive close (`SO_LINGER` 0). A server that read the request and then gave up would have
to abort that way before any response byte to be resent wrongly, and the edges in front of the
managed providers do not: Anthropic, OpenAI, OpenRouter, xAI and Together all sit behind Cloudflare,
which answers a request it forwarded and lost with a 52x status (an origin reset is its 520), and all
of them, Bedrock too, negotiate `h2` (checked 2026-10-01), where this HTTP/1.1 rule never applies
and only an RFC 9113 refusal is resent. Narrower signals were weighed and rejected: "reset before the
request was fully written" loses D80 itself (a small request is fully written into the socket buffer
before the RST arrives), and the peer's TCP ACKs cannot tell the idle-close race (the kernel ACKs a
request the application then closes on unread) from a request that was read. Each resend takes another
connection, and is resent again only if that one fails the same way, within pingora's retry limit.

**Upstream H2 is multiplexed.** A provider that negotiates `h2` carries up to 100 concurrent
streams on one connection (`UPSTREAM_H2_MAX_STREAMS`, lowered by the provider's
`SETTINGS_MAX_CONCURRENT_STREAMS`), and a full connection makes the next request open another.
Pingora's default is one stream per connection, which opened a TLS connection per concurrent
request (D160). Requests that arrive together, before the open connection is back in pingora's pool
as one with room, can each still open their own (a burst of 32 opened 0-10, measured). A connection
that dies takes every stream on it: a refused one is resent (below), the rest fail like any
delivered request whose connection died.

**A stream the provider refused is not delivered.** An upstream HTTP/2 stream refused with
`RST_STREAM(REFUSED_STREAM)` (RFC 9113 §8.7), or left above a GOAWAY's `last_stream_id` (§6.8), is
the provider's guarantee that it processed none of the request, so `error_while_proxy` resends it
however much of the body went out: on a fresh connection to the same candidate (a walk keeps its
breaker permit; a large body gets its `FullBody` reset retry), logged as `upstream refused the
stream unprocessed`. `upstream_refused_stream` reads the `h2` error under pingora's: a remote
GOAWAY error is only ever given to a stream above `last_stream_id` (one at or below it that dies
later fails with an I/O error, which stays under the rule above). Without it, a provider's routine
connection recycling surfaced as client 502s whenever a request was in flight at the GOAWAY (D72).
What stays a 502 is the narrower race the `h2` client allows: a stream opened on a connection
whose GOAWAY has already arrived, which the server ignores and which then fails as a broken pipe,
indistinguishable from a processed stream whose connection died.

**A refused stream is resent once per candidate.** Refused again on the same candidate, the
provider is failing to take work (sitting at its concurrent-stream limit, a drain that never ends),
which is a provider failure: a walk fails over to the next candidate (whose breaker records the
failure as it moves on), and with nowhere to go the request ends and `logging` records the failure
against the breaker. A large body's `FullBody` reset retry already allowed one per candidate; its
abandoned attempt now gives its breaker permit back (`release`) instead of recording a failure, the
same as a small body keeping its permit for the resend. The client gets a JSON 502 that says what
happened, `the provider refused the stream; it was not processed`, so it knows its own retry is
safe. Before (D91) every refusal was resent at once, up to pingora's retry limit (16 sends), the
breaker never heard of it, and the client read `upstream failed after receiving the request`.

**An undelivered body fails over, whatever its size.** When the failure came before the upstream
had the whole request (a reset mid-upload, a failed write), a catalog walk moves to the next
candidate: a small body replays from pingora's buffer, a large one is re-run by its `FullBody`
parent. A reused connection that fails that early is tried once more on the same candidate first
(its breaker permit kept: the connection failed, not the provider). Before this the two paths
disagreed: a small body made one attempt and got an empty 502, a large one re-sent its body to
each candidate twice and ended on a misleading 503 "no provider key available". A `FullBody` retry
is recorded only when another candidate remains, so the walk ends on an accurate error; a 429 key
walk on a large body resumes on its candidate first (`FullBody::resume` moves it to the front of
the re-run's walk), so a re-ranked row cannot turn it into a vendor switch, while the other
candidates stay behind it, so the 5xx (or last key's auth failure) that ends the key walk still
fails over exactly as a small body does (D81: pinning the walk to that one candidate relayed the
5xx); a re-run with no key walk on a candidate starts past its cooling keys
(`Provider::first_key`), as any request does (D83: re-runs always started on key 0, so every large
body paid a revoked key's 401 and a second upload); and the last attempt the parent allows records
no retry, so its answer is relayed rather than lost.

### Why the catalog has a list price and the request does not

`GET /v1/models` carries a standard list price on every catalog row (`providers::catalog::ListPrice`):
USD per million tokens for input, output, cache read, and cache write. That is the card a client
estimates with, and the card a downstream consumer uses when it has nothing better. A public card
that omits a cache rate is stored as the input rate — no discount, no write premium. Omission is
not zero; a missing rate is how cache tokens used to bill free.

The rate is the **primary candidate's** vendor standard published rate: not batch or flex, and not
OpenRouter's cheapest host (OpenRouter's card is used only when OpenRouter is the primary). A
vendor that tiers its rate is listed at the standard tier, which one card cannot fully express:
DeepSeek's peak rate (off-peak is half, so off-peak traffic is over-listed 2x), xAI's < 200k-prompt
tier (≥ 200k is 2x on every token), and OpenAI's ≤ 272K-input tier (above it, 2x input and 1.5x
output). Each direct-vendor row's rates are recorded with their source URL in
`verify/catalog_truth.toml`, and `catalog_matches_vendor_truth` fails when a row drifts from it or a
direct-vendor row has no entry.

The gateway still does not multiply those rates into `ai.usage`. It emits token facts: counts and
model identifiers. Provider pricing changes frequently, varies by contract tier, and is sometimes
retroactively corrected on invoices. Batch, fast mode, the 1-hour Claude cache write (2× input),
and long-context overrides are not this card. A downstream consumer can reprice historical facts;
the gateway's facts cannot be regenerated once the request is gone.

### Why routing uses the first path segment, not a header

Path-based routing makes the target provider explicit in every request URL — visible in logs,
traces, and curl output without inspecting headers. It also survives transparent proxies and load
balancers that strip custom headers. A `/{provider}/` prefix was preferred over a separate header
because SDKs already let callers set the base URL; swapping in the gateway's URL with a provider
prefix requires no SDK modification.

The managed `/v1` catalog walk is the other half of that: a stock SDK that can only set a host and
put `model` in the JSON body does not have to learn `x-beyond-model` or `/auto`. `/{provider}/…`
remains the escape hatch when the catalog should not apply.

### Why the response cache is a tap, and why the key is the client body

Buffering the miss to fill a cache would add latency to the path the cache is supposed to make
faster next time — and would break the gateway's unbuffered-relay contract on every first request.
A tap copies as the bytes pass; a truncated or aborted response is simply not stored. Redis,
embeddings, and semantic match are a different product: this is "the same bytes in, the same 2xx
out" for a managed catalog walk whose body we already had to peek.

The key cannot be the pool key or the candidate. Those change across a 429 walk and a vendor
failover, but the client sent one body. Hashing the pre-rewrite body (and the method, inbound path,
catalog row, Anthropic version and beta headers, and the tenant) keeps "identical request" meaning
what the caller sent, not which credential happened
to serve.

---

## Trust Boundaries

**What the gateway verifies (rejects if invalid):**

- Virtual key signature (Ed25519, stateless — no DB lookup)
- Virtual key format (`bai_v1` 16-byte payload, `bai_v2` 24-byte payload with `key_id`)
- Tenant / key not in deny-set (managed traffic only; O(1) HashMap lookup; tenant deny kills every key)
- Tenant / key remaining-ok on the allowance-set (managed only; fail-closed with a 503 until the set has been read; 402 before `upstream_peer` when exhausted)
- Pool key configured for the requested provider (managed traffic only — else 503). On a catalog
  walk this is per candidate: none keyed → 503, not a request-wide missing openai key.
- Catalog model on managed `/v1` and `/auto` (unknown or missing → 404 naming the miss). Candidate
  spellings are aliases. Chat Completions ↔ Messages ↔ Responses is translated when the inbound
  path names a different one of those three; any other inbound-path vs row mismatch is a 400.
  `/{provider}/…` is not allowlisted and never translates. `GET /v1/models` lists the catalog.
- Request body size ≤ `MAX_REQUEST_BODY` (declared `Content-Length` + streaming running total)
- One root `model` key on a catalog walk. The walk routes on one and rewrites one, while most JSON
  parsers take the _last_, so `{"model":"cheap",…,"model":"o1-pro"}` would route as the cheap row
  and be served as o1-pro. `peek::scan_buffered` counts root `model` keys on the client body, before
  any translation, and decodes a key spelled with escapes (`"mod\u0065l"`), since the provider
  would. A second one is refused (`ai_rejections_total{reason="duplicate_model"}`). Where the
  whole body is in hand before connecting (a headerless walk, which reads it to choose the row,
  and a header-won Responses or large-body walk), the refusal is the gateway's JSON 400
  (`invalid_request_error`, with `x-beyond-request-id`) and the upstream gets nothing. A body that
  streams through (a header-won small body on Chat Completions or Messages) is checked in
  `request_body_filter` once it is all in: the request aborts before a body byte goes upstream,
  but its headers have already left, so the 400 is pingora's bare status.
- Per-credential request rate within ceiling; aggregate BYO rate within ceiling

**What passes through unchecked:**

- Request body content and schema — no validation at the gateway layer
- Model name on `/{provider}/…` — extracted for billing facts, never validated against an allowlist.
  That path is the escape hatch.
- **The request body's `model` on a catalog walk when `x-beyond-model` is set.** It is an input the
  gateway _overwrites_ with the serving candidate's id, so it determines nothing — the header does.
  A body that names a different model is counted on `ai_model_header_body_mismatch_total` (a client
  bug worth finding) and otherwise ignored; `requested_model` in `ai.usage` reports the catalog
  name, which is what was actually asked for. Header still wins.
- **Which pool key a model-routed request draws on.** The header or body selects the catalog row,
  and there is no per-tenant entitlement check on rows — any managed tenant can route to any row. Not
  price-gameable (billing uses the id the provider echoes back), but worth knowing before rows are
  added whose pool keys differ in cost or contract.
- Provider response content — relayed byte-for-byte on a same-endpoint walk (a managed error:
  verbatim except provider-account remedies, below); Chat Completions ↔
  Messages ↔ Responses is translated so the client sees the inbound dialect. Usage taps stay on
  the upstream body.
- BYO token validity — forwarded as-is; the provider rejects it if invalid

**Which client headers ride on the pool key.** A managed request is sent with Beyond's credentials,
so it forwards only the client headers the gateway can vouch for, and drops the rest before adding
its own (pool key, `Host`, `accept-encoding`, OpenRouter attribution, a translated walk's
`anthropic-version` and `anthropic-beta`):

| Forwarded           | Why                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------ |
| `content-type`      | the body's media type                                                                      |
| `content-length`    | framing (re-framed by the gateway when it rewrites the body)                               |
| `transfer-encoding` | framing                                                                                    |
| `expect`            | a client waiting on `100 Continue` is answered by the provider, not by its own timeout     |
| `accept`            | SSE vs JSON negotiation                                                                    |
| `user-agent`        | provider-side diagnostics                                                                  |
| `anthropic-version` | Anthropic's required API version                                                           |
| `anthropic-beta`    | only the tokens below; a header left with none is removed, and repeated header lines merge |

`anthropic-beta` is an allowlist of tokens that change how a request is parsed or streamed and
nothing about price or server-side execution: `claude-code-20250219`, `prompt-caching-2024-07-31`,
`interleaved-thinking-2025-05-14`, `fine-grained-tool-streaming-2025-05-14`,
`context-management-2025-06-27`, `token-efficient-tools-2025-02-19`, `output-128k-2025-02-19`, and
the gateway's own `thinking-binding-controls-2026-08-01`. Everything else is dropped: `context-1m-*`
turns on premium long-context pricing, `mcp-client-*`, `code-execution-*` and `files-api-*` reach
servers, sandboxes and storage on Beyond's account, and `oauth-*` means nothing beside a pool API
key. The gateway's own beta for a translated walk is merged in after the filter, so it is never
dropped. Adding a token is a one-line change to `MANAGED_ANTHROPIC_BETAS` in `proxy.rs`.

Dropped, among others: `openai-organization` and `openai-project` (they switch the org or project the
pool key bills to, and an SDK user with `OPENAI_ORG_ID` set got a 401), `cookie`,
`x-goog-user-project`, SDK telemetry (`x-stainless-*`). `proxy-authorization` is hop-by-hop and never
crosses the proxy for anyone. BYO requests forward the client's headers, credentials included,
less every `x-beyond-*` header (the gateway's own control headers, swept by prefix on every route,
managed or BYO) and the hop-by-hop ones.

**Where a credential may travel:**

- A managed key reaches no provider in any location. Every static-key header and `Authorization`
  are stripped (every repeat), and every `key` query param is removed from the forwarded path. When
  the credential locations disagree (`x-api-key: junk` beside `Bearer bai_v1…`, or an empty
  `x-api-key`), a managed value in **any** location makes the request managed. First-location-wins
  classified that as BYO, and BYO credential headers are forwarded, so the virtual key reached the
  provider. "Any location" means every spelling a provider would read: each line of a repeated
  header, each repeated `?key=`, a percent-encoded name (`k%65y`, which Google decodes as `key`) or
  value, and `Bearer` followed by any run of whitespace. A value managed only once decoded is
  verified as written, so it 401s rather than being forwarded.
- A pool key reaches no client. A provider, or a proxy in between, that echoes the credential it
  was sent would otherwise hand Beyond's key to a tenant. On a managed response, any header whose
  value carries the key this attempt sent is dropped, and an error body (status >= 400) has every
  occurrence overwritten with a same-length `[redacted]***` marker as it streams, before
  translation (so a translated error envelope, or OpenRouter quoting an upstream error in
  `metadata.raw`, is scrubbed too) and before capture and the cache. A 2xx body is an answer and is
  **not** scanned: a streamed answer would pay the scan on every chunk, and an echo of a credential
  belongs in an error. Real providers already mask all but the last 4 characters; this does not
  depend on it.
- Pingora's own error line prints `ProxyHttp::request_summary`, overridden to log the path without
  its query, so a `?key=` credential (managed, or a BYO Google key) never reaches the log.
- A BYO key reaches only the provider it belongs to: on bare `/v1` the forwarded credential picks the provider (`x-api-key` or an `sk-ant-…` key → Anthropic, any other `sk-…` → OpenAI), and keys for two providers on one request are a 400.
- `vpc_id` in the virtual key — decoded and emitted in billing facts, not used for access control

**What the catalog allowlists (managed `/v1` and `/auto` only):**

- The model name. A catalog miss is a 404. This is not a per-key grant list and not a parallel set
  beside the catalog — the catalog row _is_ the allowlist.

**Why these boundaries are where they are:**

- Body schema validation belongs to the provider — duplicate validation adds latency without a
  security benefit at the gateway layer
- A per-provider model allowlist coupled to release cadence is what the catalog already is, for the
  drop-in `/v1` and `/auto` paths. `/{provider}/…` stays unlisted so an operator can still send an
  id the catalog does not carry
- BYO token validation requires a provider round-trip — the provider does it anyway

---

## Configuration

Every field is set in the TOML config file (`config.example.toml` is the reference; an unknown key
there is a boot failure). Scalar fields are also overridable by `AI_`-prefixed env vars (`AI_NATS_URL`,
…), pool and signing keys by `AI_POOL_KEY_<NAME>` / `AI_SIGNING_KEY_<KID>`; the provider map fields
(`provider_authorities.*`, `provider_dialects.*`, `provider_auth_schemes.*`) are file-only.
Secret-bearing fields (`pool_keys`, `nats_creds`) are held as `Secret<T>` — stray `Debug` or
`Serialize` output redacts to `"***"` and the value is zeroized on drop (`secret.rs`). The
pool key's precomputed `HeaderValue` is marked sensitive, so HPACK never indexes it and its `Debug`
prints `Sensitive`. Booting with pool keys and `upstream_tls = false` logs a loud warning: every
managed request would carry Beyond's provider key in cleartext, which is valid only against the
local plaintext mock.

| Field                           | Default                           | Runtime Effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signing_keys`                  | _(required for managed)_          | Map of kid → base64 Ed25519 public key. Multiple kids enable rotation. Missing → every `bai_v1` token 401s (fail-closed); BYO still works.                                                                                                                                                                                                                                                                                                                                              |
| `require_signing_keys`          | `false`                           | When `true`, an empty `signing_keys` is a hard boot failure. Set on managed deployments so a typo'd/absent SSM param fails at boot rather than 401-ing every managed client.                                                                                                                                                                                                                                                                                                            |
| `pool_keys.<name>`              | _(from `AI_POOL_KEY_<NAME>` env)_ | Real provider API key(s). TOML array (a string is a list of one); env stays one key = list of one. `<NAME>` is lowercased, and when it names no provider its `_` → `-` spelling is tried, so `AI_POOL_KEY_OPENAI_CODEX` reaches `openai-codex` (an exact match wins). Missing or empty → managed requests to that provider return 503 before any upstream connection. A managed 429 walks the next unused key on the same provider.                                                     |
| `provider_authorities.<name>`   | _(none)_                          | Override or add a provider's `authority` (host:port). Enables config-added providers beyond `KNOWN_PROVIDERS` with zero code change.                                                                                                                                                                                                                                                                                                                                                    |
| `provider_dialects.<name>`      | `"openai"`                        | Wire dialect for a **config-added** provider (`"openai"` or `"anthropic"`, case-insensitive). No effect on a known provider (dialect fixed in code). Unrecognized value → hard boot failure.                                                                                                                                                                                                                                                                                            |
| `provider_auth_schemes.<name>`  | `"bearer"`                        | Managed auth scheme for a **config-added** provider (`"bearer"`, `"x-api-key"`, or `"api-key"` — the last is Azure OpenAI's shape). No effect on a known provider. Unrecognized value → hard boot failure.                                                                                                                                                                                                                                                                              |
| `snapshot_path`                 | _(unset)_                         | Path for the on-disk deny-set cache. Allowance uses `{path}.allowance`. Unset → re-scan NATS on every cold boot. Set → load from disk and enforce before NATS reconnects (edge/tunnel deployments).                                                                                                                                                                                                                                                                                     |
| `rate_limit_rps`                | `100`                             | Per-credential request ceiling (count-min, keyed on raw key hash). `0` disables. Exceeded → 429 with `Retry-After: 1`. Checked before Ed25519 verify.                                                                                                                                                                                                                                                                                                                                   |
| `byo_rate_limit_rps`            | `1000`                            | Aggregate ceiling for all BYO traffic (single shared bucket). `0` disables. Managed traffic exempt. Exceeded → 429 with `Retry-After: 1`.                                                                                                                                                                                                                                                                                                                                               |
| `circuit_breaker_threshold`     | `20`                              | Per-provider upstream failures (5xx / connect; **not** 429) within the window before the breaker opens, provided failures are also at least half the window's outcomes (a failure rate, not a count a success resets). While open, requests to that provider fast-fail with 503. `0` disables. Max 16383 (the packed count); above → hard boot failure.                                                                                                                                 |
| `circuit_breaker_window_secs`   | `10`                              | Fixed window over which failures are counted, starting at its first failure (trips on a burst, not a slow trickle). `0` with the breaker on → hard boot failure (it could never open).                                                                                                                                                                                                                                                                                                  |
| `circuit_breaker_reset_secs`    | `30`                              | How long the breaker stays open before admitting a half-open probe. Probe success closes it; failure reopens it. `0` with the breaker on → hard boot failure.                                                                                                                                                                                                                                                                                                                           |
| `connect_timeout_secs`          | `10`                              | TCP connect timeout to the upstream provider. Exceeded → retry up to 2×, then 502.                                                                                                                                                                                                                                                                                                                                                                                                      |
| `read_timeout_secs`             | `600`                             | The longest the provider may stay silent: before the response head, or between body reads (pingora has one per-read upstream timeout). 600s is the OpenAI and Anthropic SDKs' default request timeout, so the gateway never gives up on a slow-but-alive provider before the client would. Silence is not a failure signal (a model thinking without emitting looks exactly like a stuck provider); a dead connection is caught by `h2_ping_interval_secs` / `tcp_keepalive_*` instead. |
| `write_timeout_secs`            | `60`                              | Upstream request-write timeout (sending the request to the provider).                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `idle_timeout_secs`             | `90`                              | Idle timeout on a pooled upstream connection before it's closed.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `client_write_timeout_secs`     | `60`                              | Downstream write timeout: a write to the client that makes no progress for this long ends the request (tagged downstream, so no breaker failure), releasing its in-flight and tenant slots, upstream connection and any probe permit. Per write, so a slow steady reader never trips it. Not a guess at model behavior: it bounds a client that stopped consuming bytes the gateway already holds, which transport liveness cannot see (see "Transport liveness"). `0` disables.        |
| `h2_ping_interval_secs`         | `15`                              | HTTP/2 PING interval on upstream connections; `0` disables. A peer that stops acknowledging fails the connection, and every stream on it, within the interval plus pingora's fixed 5s ACK deadline (≤ 20s). See "Transport liveness".                                                                                                                                                                                                                                                   |
| `tcp_keepalive_idle_secs`       | `15`                              | TCP keepalive on upstream and client sockets: first probe after this many idle seconds; `0` disables.                                                                                                                                                                                                                                                                                                                                                                                   |
| `tcp_keepalive_interval_secs`   | `5`                               | Seconds between keepalive probes.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `tcp_keepalive_count`           | `3`                               | Unanswered probes before the kernel drops the peer: a vanished host fails within idle + interval × count (30s). Upstream sockets also get `TCP_USER_TIMEOUT` of the same bound.                                                                                                                                                                                                                                                                                                         |
| `max_buffered_body_bytes`       | `536870912` (512 MiB)             | Most request-body bytes the process holds at once. A catalog walk's body past the 64 KiB replay buffer reserves twice its size (its read, and the `FullBody` re-run's copy), a `/{provider}` usage-splice body its size; a declared length reserves before a byte is read, a chunked body as it grows. Over it → 503 with `Retry-After` (`ai_rejections_total{reason="body_memory"}`). Bodies within the replay buffer are not counted. `0` disables.                                   |
| `shutdown_grace_period_secs`    | `600`                             | SIGTERM drain window for in-flight requests (= `read_timeout_secs` so a deploy never truncates a stream). An upper bound, not a wait: the process exits as soon as the last request context drops (its billing row written), so an idle gateway stops in well under a second. Capped by the orchestrator's stop timeout (ECS Fargate: 120s).                                                                                                                                            |
| `shutdown_runtime_timeout_secs` | `10`                              | Final runtime-teardown backstop after the drain window.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `capture_max_bytes`             | `262144`                          | Per-direction cap on a captured payload; the default a per-tenant entry overrides. Bounded by the log pipeline's per-record limit. This and per-tenant values clamp to 4 MiB (`MAX_CAPTURE_BYTES`).                                                                                                                                                                                                                                                                                     |
| `capture_default_sample_n`      | `1`                               | Default sampling for control-plane-enabled capture (keep 1 request in N). `1` captures every request. A capture requested via `x-beyond-capture: on` is never sampled away.                                                                                                                                                                                                                                                                                                             |
| `capture_queue_depth`           | `1024`                            | Depth of the bounded `ai.payload` sink queue. When full, captures are **dropped** (`ai_capture_dropped_total`) rather than blocking — a stalled log sink must never backpressure the data plane.                                                                                                                                                                                                                                                                                        |
| `cache_ttl_secs`                | `0`                               | Per-pod exact-match response cache TTL. `0` disables. Only managed catalog walks whose body is already in hand before `upstream_peer`. A hit on **this process** replays the stored 2xx; a miss stays an unbuffered relay (no Redis).                                                                                                                                                                                                                                                   |
| `cache_max_entries`             | `1024`                            | Cap on stored cache entries. Oldest insertion is dropped when a new one would exceed it.                                                                                                                                                                                                                                                                                                                                                                                                |
| `cache_max_bytes`               | `65536`                           | Cap on a single stored response body. Oversize complete 2xxs are relayed but not stored.                                                                                                                                                                                                                                                                                                                                                                                                |
| `smart_router`                  | `true`                            | Rank managed catalog walks by **this process's** TTFT EWMA. `false` restores static catalog order. `x-beyond-order` / `split` pin either way. Not a fleet-wide ranking.                                                                                                                                                                                                                                                                                                                 |
| `tenant_max_in_flight`          | `0`                               | Most requests one tenant may hold open **on this process**. `0` disables. Over it → 429 with `Retry-After: 1` (`ai_rejections_total{reason="tenant_concurrency"}`), before the breaker. Bounds overspend while the allowance-set lags. Managed only.                                                                                                                                                                                                                                    |
| `nats_url`                      | `nats://localhost:4222`           | NATS server for the control-plane watchers. Unreachable → deny-set stale (fail-open), capture off, allowance fail-closed until a scan or `{snapshot_path}.allowance` lands.                                                                                                                                                                                                                                                                                                             |
| `nats_creds`                    | _(unset)_                         | Base64 NATS `.creds` contents (ECS via SOPS). Held as `Secret`. Takes priority over `nats_creds_file`.                                                                                                                                                                                                                                                                                                                                                                                  |
| `nats_creds_file`               | _(unset)_                         | Path to a NATS `.creds` file, used when `nats_creds` is unset. One of the two is required for authenticated clusters.                                                                                                                                                                                                                                                                                                                                                                   |
| `config_bucket`                 | `ai-gateway`                      | slipstream bucket holding the `blackhole.*` (deny), `allowance.*` and `aicapture.*` sets.                                                                                                                                                                                                                                                                                                                                                                                               |
| `listen`                        | `0.0.0.0:8080`                    | Proxy listener address (client traffic). Plain HTTP; production keeps it internal (no public ingress).                                                                                                                                                                                                                                                                                                                                                                                  |
| `downstream_h2c`                | `true`                            | Also accept HTTP/2 cleartext on `listen` (Pingora peeks the preface; HTTP/1.1 clients are unaffected). `false` forces HTTP/1.1.                                                                                                                                                                                                                                                                                                                                                         |
| `worker_threads`                | `0`                               | Tokio worker threads for the proxy service. `0` = one per available core (Pingora alone would default to one). Set it explicitly under a CPU quota: the core count is the host's, not the cgroup's.                                                                                                                                                                                                                                                                                     |
| `upstream_tls`                  | `true`                            | TLS to the provider. `false` only for the plaintext test mock; with pool keys set it logs a loud warning.                                                                                                                                                                                                                                                                                                                                                                               |
| `upstream_http2`                | `true`                            | Offer ALPN `h2` (HTTP/1.1 fallback) to TLS upstreams. `false` forces HTTP/1.1 without recompiling.                                                                                                                                                                                                                                                                                                                                                                                      |
| `upstream_verify_cert`          | `true`                            | Verify the upstream certificate and SNI. `false` only for the bench's self-signed TLS mock; never against a real provider.                                                                                                                                                                                                                                                                                                                                                              |
| `provider_authorities.auto`     | _(rejected)_                      | Reserved: `auto` is the model-routed segment, and a provider of that name would shadow it. Hard boot failure.                                                                                                                                                                                                                                                                                                                                                                           |
| `metrics_listen`                | `0.0.0.0:9090`                    | Admin/observability listener: `/metrics` (Prometheus scrape), `/livez`, `/readyz` (503 until the allowance-set is seeded on a managed deployment). Separate from the client listener and unauthenticated. The default binds every interface, so anything that can reach the host on that port can read it; keep it off public ingress (or bind it to a private address).                                                                                                                |

---

### Transport liveness

The gateway cannot tell a model thinking silently from a stuck provider, so it does not cut silence
below the client's own timeout: `read_timeout_secs` is 600s, the OpenAI and Anthropic SDKs' default.
What it can tell is a dead peer, and that is caught by transport liveness, which the peer's HTTP/2
stack and kernel answer, not the model:

- **HTTP/2 PING** (`h2_ping_interval_secs`, 15s) on every upstream connection, via pingora's
  `PeerOptions::h2_ping_interval`. An unanswered PING fails the connection after pingora's fixed
  5s ACK deadline, so a dead H2 provider is detected within 20s. 5s is over ten worst-case
  intercontinental round trips (~300ms) and fits several TCP retransmissions, so a live peer never
  misses it; one 17-byte frame and its ACK per connection per 15s is negligible next to a stream.
- **TCP keepalive** (`tcp_keepalive_*`: 15s idle, 5s × 3 probes) on upstream sockets
  (`PeerOptions::tcp_keepalive`) and accepted client sockets (`TcpSocketOptions`). It covers
  HTTP/1.1 providers, which have no PING, within 30s. Upstream sockets also get `TCP_USER_TIMEOUT`
  of 30s, because keepalive does not probe a connection with unacknowledged data (a partition
  mid-request). Toward clients it frees a vanished client's slots during a silent model turn, when
  the gateway has nothing to write. The probe interval is ~15 round trips and well past Linux's
  200ms minimum retransmission timeout; three probes ride out two lost ones. The kernel's own
  default (2h idle, 9 × 75s) is far too slow to matter.

A client that stops **reading** is a different question, and `client_write_timeout_secs` (60s)
answers it. A live process that stops reading still ACKs at the kernel and advertises a zero
window, so keepalive (which probes only an idle connection) reports it healthy forever, and its own
timeout cannot fire because it is not waiting on anything. The bound is per write: any progress
resets it, and a client that is reading at all drains a full socket buffer within a few round trips,
so 60s of zero progress is hundreds of them. Client sockets get no `TCP_USER_TIMEOUT`, so that
judgement stays with this one setting.

## Failure Modes

Every error the gateway makes itself, whether a `reject` before the upstream or a failure that
ends the proxy loop (`fail_to_proxy`), is JSON: `content-type: application/json`,
`{"error":{"message","type"}}`, and `x-beyond-request-id`. The status follows the cause: a connect
failure on every candidate is `502`, every candidate's breaker open is `503` with `Retry-After`, an
upstream timeout is `504`, and a chunked body that crosses the cap is `413`. A client that is
already gone gets nothing, and a response that has started cannot be replaced.

| Failure                                                                                    | What Actually Happens                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Recovery                                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NATS unreachable at boot                                                                   | Deny-set starts empty (fail-open). Allowance is unread → managed traffic gets a retryable 503 (`Retry-After: 5`), fail-closed. Auth still works — keys from config.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Watcher reconnects; seeds from NATS or disk snapshot on connect. Allowance flips remaining-ok once `from_entries` runs (empty scan included). Until then `/readyz` is 503 so the pod receives no traffic.                                                                                                                |
| NATS disconnects mid-run                                                                   | Last-known deny-set and allowance-set stay active. New entries not applied until reconnect. Unready allowance (never seeded) keeps refusing managed traffic with a retryable 503.                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Watcher reconnects (1s→30s exponential backoff, reset only after a watch that ran ≥30s — _connecting_ is not success, or a reachable NATS with a broken watch loops at 1 Hz forever) and resumes from the saved revision. Rescans instead when the seed found no entries, since revision 0 is not resumable — see above. |
| NATS history compacted past snapshot cursor                                                | `CursorExpired` → full re-scan from current NATS state.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | After re-scan, new cursor set; delta watch resumes normally.                                                                                                                                                                                                                                                             |
| Virtual key tampered or forged                                                             | Prefix matches (`bai_v1`) and Ed25519 verify fails → **401**, never BYO. No billing event. The error does not name which part of the token failed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Client retries with a valid key. A public listener that forwarded a failed verify as BYO would open junk-auth egress while the rate guard had already exempted the token from the BYO aggregate.                                                                                                                         |
| `signing_keys` absent (typo'd/missing SSM)                                                 | Default: warn; every `bai_v1` token 401s (fail-closed); BYO still works. With `require_signing_keys=true`: hard boot failure.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Set `require_signing_keys=true` on managed deployments so the mis-deploy fails at boot rather than 401-ing every tenant.                                                                                                                                                                                                 |
| Pool key missing for provider                                                              | Managed request returns 503 before any upstream connection.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Add `AI_POOL_KEY_<NAME>` env and redeploy.                                                                                                                                                                                                                                                                               |
| `provider_dialects`/`provider_auth_schemes` value unrecognized                             | Hard boot failure (`GatewayError::Config`) naming the provider and the bad value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Fix the typo — `"openai"`/`"anthropic"` and `"bearer"`/`"x-api-key"`/`"api-key"` are the only accepted values.                                                                                                                                                                                                           |
| Config-added provider's dialect misconfigured (wire doesn't match)                         | `usage::openai_body`/`anthropic_body` (and the stream variants) detect the other dialect's characteristic field names and return `None` instead of a zeroed `Usage`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `usage_parse_errors_total` fires + a `warn!` log; fix `provider_dialects.<name>` to match the vendor's actual wire.                                                                                                                                                                                                      |
| Provider DNS fails                                                                         | A lookup is bounded at 5s. With no earlier answer: provider-routed → JSON 502; a catalog walk records the candidate's failure and moves on. When re-resolution fails or times out, the last good answer is served for up to 10 min after it resolved (re-tried every 5s, `warn` logged). Lookups are single-flight per authority: a due refresh runs once in the background while every caller keeps the cached answer (none waits on it), concurrent cold callers share one lookup, and a lookup hung past its 5s stays the one in flight until it returns, so a resolver hang never piles `getaddrinfo` calls onto the blocking pool (D89).              | Fix the resolver or the authority; serve-stale covers a resolver outage for providers whose addresses did not change.                                                                                                                                                                                                    |
| A name resolves to several addresses and the first is dead                                 | Every address is kept in the lookup's order. A refused connect tries the next one: provider-routed via the connect retries, a catalog candidate before the walk moves to the next candidate (same breaker permit: one address refusing is not the provider failing).                                                                                                                                                                                                                                                                                                                                                                                       | None. `tests/reliability_lifecycle.rs` pins both (`localhost` → `::1` first).                                                                                                                                                                                                                                            |
| Provider TCP connect fails                                                                 | `fail_to_connect` retries up to 2×, then returns a JSON 502. Counts as a circuit-breaker failure.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Client SDK retries with backoff. No HTTP-status retries (Pingora-idiomatic).                                                                                                                                                                                                                                             |
| Provider connection dies (host gone, partition, wedged edge)                               | Upstream HTTP/2 PING goes unacknowledged → the connection and its streams fail within `h2_ping_interval_secs` + 5s (≤ 20s); on HTTP/1.1, TCP keepalive drops the peer within 30s (and `TCP_USER_TIMEOUT` fails unacknowledged request bytes as fast). Before the head: a JSON 502; mid-stream: the client gets an error, never a clean end, and the row is `cut_short`.                                                                                                                                                                                                                                                                                    | Client retry.                                                                                                                                                                                                                                                                                                            |
| Provider alive but silent (no head yet, or a quiet gap: a model thinking without emitting) | Never cut before `read_timeout_secs` (600s), the clients' own default timeout: the gateway cannot tell thinking from stuck, so it does not guess. Then a JSON 504 (`upstream timed out after receiving the request`), not resent. No total deadline: a stream that keeps sending lives until it ends.                                                                                                                                                                                                                                                                                                                                                      | Client timeout or retry.                                                                                                                                                                                                                                                                                                 |
| Provider brownout (sustained 5xx)                                                          | After `circuit_breaker_threshold` 5xx/connect failures in the window that are also at least half its outcomes (so a 50% brownout counts), the breaker opens; requests fast-fail 503 (`circuit_open`) instead of stalling against the read timeout.                                                                                                                                                                                                                                                                                                                                                                                                         | Auto: after `circuit_breaker_reset_secs` a half-open probe is admitted — success closes the breaker, failure reopens it. Per-provider, so other providers are unaffected.                                                                                                                                                |
| Provider throttles (429 storm)                                                             | Walk the next unused pool key on the same provider when the body is replayable; the last 429 is relayed with `Retry-After` if the upstream sent one. Does **not** trip the breaker (provider is healthy). Does **not** fail over to another vendor.                                                                                                                                                                                                                                                                                                                                                                                                        | Client `Retry-After` backoff after keys are exhausted; no gateway-side circuit action.                                                                                                                                                                                                                                   |
| Pool key out of credit (a credit-balance 400, `insufficient_quota`)                        | Relayed with a neutral message (the head is relayed before the body says why), the key cooled off for 60s (`ai_key_auth_failures_total`); a catalog walk leaves the provider out while all its keys cool, so later requests go to the row's next candidate (D180). Not a breaker failure.                                                                                                                                                                                                                                                                                                                                                                  | Fund the account or replace the key; the warn line `pool key is out of credit` names provider and key index.                                                                                                                                                                                                             |
| Pool key revoked (401)                                                                     | Walk the next pool key on the same provider when the body is replayable, and cool the refused key off for 60s so later requests start on a good one (`ai_key_auth_failures_total`). The last key's 401 is relayed on a provider route, a candidate failure on a catalog walk. A 403 never walks or cools (a catalog walk fails over on it), except that one whose body names the key cools it. Not a breaker failure.                                                                                                                                                                                                                                      | Replace the key in config; the metric says which provider (log line names the key index).                                                                                                                                                                                                                                |
| Response body > 128KB before usage chunk                                                   | Tail compaction fires: `drain(..half)` discards first half, keeps tail. Usage extracted from retained tail.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | No action — SSE usage is always in the final `data:` line, which always lands in the tail.                                                                                                                                                                                                                               |
| Client cancels mid-request (ESC on a slow turn)                                            | Relayed as a downstream abort. **Not** counted against the provider's breaker — pingora tags it `ErrorSource::Downstream`. A streaming 2xx cut short this way is billed an estimate (`usage_estimated=true`), not zero. The tenant slot is released.                                                                                                                                                                                                                                                                                                                                                                                                       | None. `tests/cancellation.rs` pins the breaker halves; `tests/cut_short.rs` the billing.                                                                                                                                                                                                                                 |
| Retry replays a partially-read request body                                                | `upstream_peer` resets the body-phase state each attempt, so the replayed prefix replaces rather than appends. Previously it was appended, producing a duplicated JSON fragment the provider rejected with a `400` that `logging` recorded as a breaker _success_.                                                                                                                                                                                                                                                                                                                                                                                         | None — the reset is unconditional and O(1) on the first attempt.                                                                                                                                                                                                                                                         |
| Provider drops the connection without answering                                            | Before the whole body went out (a reset mid-upload, a failed write): a catalog walk fails over to the next candidate, any body size (a reused connection first gets one more try on the same candidate). After it went out: not resent anywhere, reused connection or not (it may be generating and billing); a JSON 502, or 504 on a timeout. Except an H2 stream the provider refused (REFUSED_STREAM, or above a GOAWAY's last stream id), or a reused H1 connection reset before any response byte: resent once on a fresh connection, since it was not processed; refused again, a provider failure (failover, or a 502 saying it was not processed). | Client retry, which is a fresh request. `tests/reliability_large_body.rs` pins both halves for small and large bodies.                                                                                                                                                                                                   |
| Model-routed candidate refuses the connection                                              | `fail_to_connect` advances to the next candidate and pingora re-invokes `upstream_peer`; the abandoned candidate's breaker records the failure. Client sees nothing.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Automatic. `ai_candidate_failovers_total` counts it; the per-provider `ai_connect_retries_total` names the candidate left behind.                                                                                                                                                                                        |
| Model-routed request with every candidate down                                             | Each is attempted once, then the request fails with a JSON `502` (`no provider could be reached`), or a `503` with `Retry-After` when every candidate was skipped because its breaker is open. Each candidate's breaker records its own failure.                                                                                                                                                                                                                                                                                                                                                                                                           | Fix whichever providers are down; `doctor`'s `model_catalog` check catches the _configuration_ case (a row with no pool-keyed candidate) at boot.                                                                                                                                                                        |
| Many large uploads at once                                                                 | Each body past the replay buffer reserves its buffered bytes against `max_buffered_body_bytes` (512 MiB) before it is read. One that would cross it gets a JSON 503 with `Retry-After: 1` and costs nothing more; the others proceed. Before, eight concurrent 90 MiB uploads grew the process by ~1.4 GiB.                                                                                                                                                                                                                                                                                                                                                | Client retry after the in-flight bodies finish. `tenant_max_in_flight` bounds one tenant's share.                                                                                                                                                                                                                        |
| Gateway crash mid-request                                                                  | In-flight request drops; client receives TCP close. No partial state written.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Client SDK retries. No DB writes in the request path — no cleanup needed.                                                                                                                                                                                                                                                |
| Panic inside a proxy phase (a gateway bug)                                                 | Tokio catches it on the request task; the connection drops. The request context's drop (`proxy::Ctx`) gives back what `logging` would have: the in-flight and SSE gauges, the tenant concurrency slot, and an unresolved breaker permit (returned without an outcome, so a half-open probe permit is not stranded). No billing row is written for that request.                                                                                                                                                                                                                                                                                            | Fix the bug from the panic in the log. Debug builds take `AI_FAULT_PANIC=<phase>` to panic the first request reaching that phase; `tests/reliability_lifecycle.rs` proves the release.                                                                                                                                   |

---

## Metrics

Prometheus on the default registry, exposed at `/metrics` on `metrics_listen`.

| Metric                                | Type      | Labels               | What It Measures                                                                                                                                                           |
| ------------------------------------- | --------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ai_requests_total`                   | Counter   | —                    | Every client request received, rejected ones included (a `FullBody` re-run is not counted again)                                                                           |
| `ai_rejections_total`                 | Counter   | `reason`             | Rejected requests by cause (auth, deny_spend, quota, allowance_unavailable, deny_fraud, rate_limit, tenant_concurrency, managed_endpoint, duplicate_model, modality, etc.) |
| `ai_upstream_responses_total`         | Counter   | `provider`, `status` | Upstream responses by provider and status class                                                                                                                            |
| `ai_tokens_total`                     | Counter   | `kind`               | input / output / cache_read / cache_write token counts                                                                                                                     |
| `ai_ttft_seconds`                     | Histogram | `provider`           | Time to first token (50ms–30s buckets)                                                                                                                                     |
| `ai_upstream_latency_seconds`         | Histogram | `provider`           | Full request latency (100ms–600s buckets)                                                                                                                                  |
| `ai_active_streams`                   | Gauge     | —                    | Open SSE streams                                                                                                                                                           |
| `ai_requests_in_flight`               | Gauge     | —                    | All in-flight requests (streaming + non-streaming)                                                                                                                         |
| `ai_deny_set_size`                    | Gauge     | —                    | Current number of denied tenants                                                                                                                                           |
| `ai_nats_connected`                   | Gauge     | —                    | 1 if the **deny-set** watcher is connected, 0 otherwise                                                                                                                    |
| `ai_allowance_set_size`               | Gauge     | —                    | Exhausted tenants + keys in the allowance-set                                                                                                                              |
| `ai_allowance_ready`                  | Gauge     | —                    | 1 after a successful allowance scan/snapshot (empty = remaining-ok); 0 = fail-closed                                                                                       |
| `ai_allowance_nats_connected`         | Gauge     | —                    | 1 if the **allowance-set** watcher is connected                                                                                                                            |
| `ai_capture_set_size`                 | Gauge     | —                    | Tenants with payload capture enabled (climbing and never falling ⇒ missing TTLs)                                                                                           |
| `ai_capture_nats_connected`           | Gauge     | —                    | 1 if the **capture-set** watcher is connected — separate watcher, separate connection                                                                                      |
| `ai_captures_total`                   | Counter   | —                    | Requests whose payloads were captured (post-sampling)                                                                                                                      |
| `ai_capture_bytes_total`              | Counter   | —                    | Payload bytes handed to the sink — the cost signal, ahead of the storage bill                                                                                              |
| `ai_capture_dropped_total`            | Counter   | —                    | Captures dropped on a full sink queue — distinguishes "lost it" from "capture was off"                                                                                     |
| `ai_control_header_errors_total`      | Counter   | —                    | `x-beyond-*` headers present but unusable (dropped; request still served)                                                                                                  |
| `ai_usage_parse_errors_total`         | Counter   | —                    | Managed 2xx responses that ended cleanly without parseable usage (billed an estimate or zero; free token counts excluded)                                                  |
| `ai_cache_hits_total`                 | Counter   | —                    | Exact-match cache hits that replayed a stored 2xx and skipped the provider                                                                                                 |
| `ai_cache_scope`                      | Gauge     | `kind`               | Constant `1` with `kind="process"`: this pod's cache table, not a fleet store                                                                                              |
| `ai_smart_rank_scope`                 | Gauge     | `kind`               | Constant `1` with `kind="process"`: this pod's TTFT EWMA, not a fleet-wide ranking                                                                                         |
| `ai_candidate_failovers_total`        | Counter   | —                    | Model-routed requests that abandoned a candidate for the next one                                                                                                          |
| `ai_key_walks_total`                  | Counter   | —                    | Managed 429s and 401s that retried the same provider with the next unused pool key                                                                                         |
| `ai_key_auth_failures_total`          | Counter   | —                    | A pool key drew a 401, a 403 naming the key, or an out-of-credit answer (D180); that key is cooled off for 60s. Any rate means a key needs replacing or funding            |
| `ai_session_pinned_total`             | Counter   | —                    | Catalog walks whose primary came from a session pin instead of the TTFT rank                                                                                               |
| `ai_full_body_relays_total`           | Counter   | —                    | Managed requests re-run as a subrequest because routing needed the whole body past 64 KiB                                                                                  |
| `ai_model_header_body_mismatch_total` | Counter   | —                    | Catalog-walk requests whose `x-beyond-model` and body `model` disagreed (header wins; client bug)                                                                          |
| `ai_failover_unreplayable_total`      | Counter   | —                    | 5xx/429 retries declined on `/{provider}` or a still-uploading body: not provably replayable (catalog walks re-run instead)                                                |
| `ai_usage_estimated_total`            | Counter   | —                    | Managed requests billed estimated tokens: a stream or body cut short, or a cancel before the response head                                                                 |
| `ai_usage_write_errors_total`         | Counter   | —                    | `ai.usage` billing rows whose stdout write failed (the row is lost; also reported on stderr)                                                                               |

---

## Modules

| Module            | Role                                                                                                                            | Tested         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `proxy`           | `ProxyHttp` impl — request/response pipeline (request_filter through logging)                                                   | e2e ✓          |
| `key`             | `bai_v1` parse + Ed25519 verify + mint; keyring with multi-kid rotation support                                                 | unit ✓         |
| `route`           | Data-driven provider table (name / authority / auth) + dialect default + model-route re-exports                                 | unit ✓         |
| `peek`            | `ModelScanner` — streaming structural scan for the root-level `model`; O(1) memory                                              | unit ✓         |
| `usage`           | Token extraction (OpenAI / Anthropic, body + SSE)                                                                               | unit ✓         |
| `terminal`        | Whether the bytes sent to a client end with the stream's terminal event (`[DONE]`, `message_stop`, `response.completed`)        | unit ✓ + e2e ✓ |
| `deny`            | Sparse deny-set, default-allow, reason → HTTP status                                                                            | unit ✓         |
| `allowance`       | Sparse remaining-ok / exhausted set; fail-closed until seeded (503); 402 when exhausted                                         | unit ✓ + e2e ✓ |
| `capture`         | Sparse capture-set (default-off) + head-bounded `CaptureBufs` and 1-in-N sampling                                               | unit ✓ + e2e ✓ |
| `capture_sink`    | Bounded, lossy `ai.payload` writer — drops on a full queue so a stalled log sink can't backpressure                             | unit ✓         |
| `control`         | `x-beyond-*` header parse/validate; metadata canonicalized and re-serialized; catalog walk permute (`order` / `only` / `split`) | unit ✓ + e2e ✓ |
| `translate`       | Chat Completions ↔ Messages ↔ Responses mapping for a catalog endpoint mismatch; SSE event-by-event                             | unit ✓ + e2e ✓ |
| `smart`           | Per-pod TTFT EWMA table; ranks unpinned catalog walks; probe of unmeasured arms; not fleet-wide                                 | unit ✓ + e2e ✓ |
| `cache`           | Per-pod exact-match response store (TTL + max entries + max bytes/entry); tap, never a buffer; miss does not consult Redis      | unit ✓ + e2e ✓ |
| `concurrency`     | Per-tenant in-flight cap (`tenant_max_in_flight`): sharded, sparse, exact counters; the overspend bound                         | unit ✓ + e2e ✓ |
| `ratelimit`       | Two-tier guardrail: per-credential (count-min sketch, fixed memory, no GC) + global BYO (one atomic)                            | unit ✓         |
| `circuit_breaker` | Per-provider lock-free breaker (packed `AtomicU64`, windowed policy) — trips on 5xx/connect, not 429                            | unit ✓ + e2e ✓ |
| `state`           | Keyring + provider registry + watched deny-/allowance-/capture-sets (ArcSwap) + TTL DNS cache (all addresses, serve-stale)      | unit ✓         |
| `store_watch`     | Generic `WatchedSet` driver — gap-free seeding + delta watch, instantiated per set                                              | e2e ✓          |
| `config`          | Figment config; build keyring; pool keys / authorities by provider name                                                         | unit ✓         |
| `secret`          | Redacting, zeroize-on-drop `Secret<T>` newtype for pool keys and NATS creds                                                     | unit ✓         |
| `admin`           | `ServeHttp` on the metrics listener: `/livez`, `/readyz`, `/metrics`                                                            | e2e ✓          |
| `metrics`         | Prometheus counter/histogram/gauge registration and update helpers                                                              | compile ✓      |
| `doctor`          | Boot-time diagnostics (`beyond-ai doctor`)                                                                                      | compile ✓      |
| `main`            | CLI (`run` / `doctor`), rustls init, config load, Pingora server + proxy/watchers/admin bootstrap                               | compile ✓      |

---

## Verification

- **Unit (`cargo test --lib`):** key, route, peek, usage, deny, allowance, secret, config, cache, control,
  smart, translate. `clippy --all-targets -D warnings` clean. The response side of translate
  (`src/translate_response_tests.rs`) feeds every stream whole, byte by byte and in 7-byte chunks
  and requires the same events from each, and rebuilds client state the way the OpenAI and
  Anthropic SDKs accumulate it (tool calls by `index`, blocks by `content[index]`, the Responses
  item lifecycle, and openai-python's `accumulate_delta` for the message a Chat client echoes on
  its next turn).
- **Property tests (`tests/props_*.rs`):** generated, adversarial inputs instead of examples —
  arbitrary unicode and escapes, nested tool arguments with boundary numbers, every content kind,
  SSE framed with CRLF, comments, multi-line data and cut at any byte, errors and truncation at
  any event. They check that `SseBridge` never panics and always hands each client a well-formed
  stream of its own dialect (block nesting, the Responses item lifecycle and `sequence_number`,
  Chat `[DONE]`), whatever the upstream sent; that chunking never changes the client's bytes; that
  text, tool calls and signed thinking survive every pairing, request and response, and back;
  that the usage a translated client is shown equals what `usage.rs` bills for the same upstream
  bytes, stream and not; that `peek`'s scans agree with serde_json (the provider's reading, last
  duplicate wins) on `model`, `stream`, `stream_options` and the output limits; that no field that
  changes the answer is silently dropped; and that the route tables classify any path as
  documented. Each property runs `PROPTEST_CASES` cases (default 2000), stopped after
  `PROPS_SECS` (default 60); the weekly deep job (`.github/workflows/deep.yml`, nextest profile
  `deep`) runs 100,000 each with the box raised so all of them run. A failure prints the
  `PROPS_SEED` that replays it. Every
  counterexample that was a real defect is a named regression in `tests/props_regressions.rs`
  (and `tests/billing_streams.rs` for the metering bypass); the generators skip that exact shape,
  naming the defect, until it is fixed.
- **Translated responses end to end (`tests/translate_response.rs`):** through the real proxy with
  provider-shaped fixtures — the full Responses stream for a Claude and a GPT row, a custom tool
  call reaching a Responses client, an OpenRouter thinking signature reaching a Messages client on
  failover, cache-inclusive `prompt_tokens`, truncation as `incomplete`, a Bedrock-shaped 400, an
  OpenRouter provider error and mid-stream failure keeping their message, a stream cut before its
  end reaching the client as an error, and `accept-encoding: identity` on every managed upstream
  request. `tests/translate_request.rs` reads what the upstream received, including a Chat
  client's next turn built from what the gateway streamed it and signed thinking bound for
  OpenRouter as `reasoning_details`.
- **End-to-end (`tests/e2e.rs`, `mise run test:integration:rs`):** real `beyond-ai` binary + real
  nats-server + mock upstream. Covers managed key-swap + passthrough fidelity + usage metering
  (OpenAI JSON + SSE, **Anthropic `/v1/messages`** with `x-api-key` swap + metering), **BYO
  passthrough** (raw token unchanged), the **virtual key in either inbound header** (`Bearer` or
  `x-api-key`), and deny-set propagation: spend (write `blackhole.{tenant}` → 402, delete → 200),
  **fraud** (→ 403), and **per-credential** (write `blackhole.key.{id}` → 402 for that `bai_v2`
  key only; a sibling key for the same tenant still serves; tenant deny then 402s both).
  **Allowance:** exhaust one `bai_v2` key (`allowance.key.{id}`) → 402 for that credential only
  (sibling still serves; no pool connect on 402); tenant exhaust 402s both; keyed deny still works;
  unready (NATS never seeded) 503s with `allowance_unavailable` and `mock.hits()==0`.
  Error/edge paths: **missing key → 401**, **oversized `Content-Length` →
  413**, **managed key for an unconfigured provider → 503**, **managed 429 key-walk** (two keys,
  first throttled, second serves; one key and an unreplayable body still relay 429 with
  `Retry-After`; BYO does not walk), **streaming tail compaction** (>128KB
  before the usage chunk still meters), **deny-set fail-open** (kill NATS → stale set retained,
  auth still works), and **on-disk snapshot survival** (blackhole a tenant, restart with NATS down
  → the hold is still enforced from disk).
- **Model routing (`tests/model_routing.rs`):** the `/auto` route and managed `/v1` catalog walk
  end-to-end against two mocks with different mounts, pool keys, and model ids. Covers primary
  routing, **failover on a refused connection** (asserting the fallback's mount, its pool key, and
  its spelling of the model — the key assertion, since forwarding the primary's key would be a
  credential leak rather than a failed request), the breaker ledger (the abandoned candidate's
  breaker opens while the fallback keeps serving), missing/unknown model → 404 (named), Chat
  Completions ↔ Messages **translate** (OpenAI body + `claude-opus-4-8` on `/v1/chat/completions`
  reaches Anthropic `/v1/messages` with the pool key; client SSE is `chat.completion.chunk`;
  `ai.usage` has non-zero Anthropic tokens including cache/reasoning from the upstream parser;
  `cache_control` / `reasoning_effort` reach Anthropic fields and thinking blocks reappear on the
  client stream; the reverse with a GPT id on `/v1/messages`; same-wire walks still byte-relay;
  `/{provider}` still 400s a Claude body to OpenAI; `/v1/embeddings` with a Claude or GPT row, and a
  chat body against an embeddings row, are wire-mismatch 400s; `/v1/embeddings` on an embeddings
  row reaches `/v1/embeddings`, fails over to OpenRouter's path and id, and bills input tokens). **Mixed-wire rows:** Anthropic 5xx fails onto OpenRouter Chat Completions with
  a Chat Completions body spliced from the original client; billing dialect is the serving
  candidate. **Card-held walks** (`tests/catalog_capabilities.rs`, `tests/catalog_responses_arms.rs`):
  a JSON-schema output skips Bedrock, a PDF skips OpenRouter's grok-build-0.1, a grok one-shot
  reaches xAI's `/v1/responses` as `store: false`, and a Messages client with more than 128 tools
  on a GPT row walks its Responses arm. **Responses** (`tests/translate.rs`): a stock `/v1/responses` body with `store: false`
  and a GPT catalog id is translated onto Chat Completions; the same body with `claude-*` lands on
  Messages. Managed `/v1/responses` + `gpt-4o` + `previous_response_id` hits OpenAI `/v1/responses`
  with the field intact; Claude + `previous_response_id` is 400 naming the field, no upstream
  (`responses_session_field` classes a `conversation` the same way);
  `/{provider}/v1/responses` stays a relay. `ai.usage` still comes from the upstream parser. `GET /v1/models` lists the catalog, a candidate
  spelling is an alias, BYO on
  `/auto` → 400, BYO on `/v1` still forwarded, `/openai/…` ignoring the catalog, a stock SDK shape
  against `/v1` with only `model` in the body, the routing header never reaching an upstream,
  `ai.usage` naming the candidate that served, all-candidates-down, a
  **256 KiB body surviving a failover byte-for-byte**, a **429 walking keys not vendors**,
  provider-routed traffic being unaffected, **`x-beyond-order` hitting Bedrock's mount/key/id on
  an Anthropic-first row**, **`only` of an unkeyed provider → 503**, a junk walk header keeping
  catalog order and incrementing `ai_control_header_errors_total`, a **split over N requests
  hitting both primaries**, and a **TTFT ranker that, after a probe, prefers the faster of two
  live candidates**.
- **Response cache (`tests/cache.rs`):** two identical managed `/v1` requests hit once upstream,
  the replayed body and status are byte-identical, a different tenant misses, `x-beyond-cache: off`
  always goes upstream and never fills, a 429-then-200 is still one cacheable client-body hash, and
  two candidate orders (default vs `x-beyond-order`) do not cross-hit. `ai_cache_scope{kind="process"}`
  and `ai_smart_rank_scope{kind="process"}` are `1` — rank and cache are per-pod, not fleet-wide.
- **Cut short (`tests/cut_short.rs`):** a real stream cancelled mid-flight through the binary —
  OpenAI bills estimated input (the prompt text's pre-tokens) and one token per relayed delta; Anthropic keeps
  `message_start`'s exact input and estimates output; a base64 image does not inflate the input
  estimate; a stream that finishes is billed exactly, `usage_estimated=false`.
- **Tenant concurrency (`tests/tenant_concurrency.rs`):** at the ceiling a tenant gets a 429 without
  the provider being hit, another tenant is unaffected, the slot comes back on completion, and
  cancelled requests do not strand slots.
- **Cancellation (`tests/cancellation.rs`):** a client that gives up must not open the provider's
  breaker, and a genuinely broken provider still must. Verified non-vacuous — reverting the fix makes
  the first test fail.
- **Live smoke (`tests/smoke.rs`, `mise run test:smoke`):** the real `beyond-ai` binary against the
  **real** provider hosts over TLS, one per pool-keyed provider (every one in `KNOWN_PROVIDERS`
  except `openai-codex`, whose subscription token is not an API key), plus Responses usage
  metering, every catalog row and candidate (`catalog_rows_are_servable`) and a real catalog
  failover. Each runs the **managed** path: the real key is the gateway's pool key and the client
  presents a minted `bai_…` key, so it proves verify → deny-check → pool-key swap, real TLS/SNI,
  and the base-path rewrite landing on a live mount (200, not 404). Every test is `#[ignore]` and
  skips unless its provider's API key env var is set — CI stays hermetic; you only hit providers
  you have keys for.

---

## Benchmarking

Two harnesses, mirroring the unit/e2e split of the tests. The framing is **Theory of Constraints**:
a proxy's steady-state constraint is upstream I/O, not gateway CPU. The benches **prove the
gateway's added cost is negligible and bounded** — i.e. it never becomes the constraint.

- **Unit micro (`benches/unit.rs`, `mise run bench:unit`) — `divan`.** Times IO-free hot paths and
  measures allocations natively (divan's `AllocProfiler` reports alloc/dealloc/grow count + bytes
  beside ns/iter, no `unsafe` needed). Coverage: `key` verify/mint; `peek::ModelScanner` over
  0/4KB/256KB bodies with `model` placed last (worst case); `usage` parsers; `route`; `deny`
  (`parse_key`/`parse_reason` off-path + `reason()` on-path); `allowance::reason_for` (v1 and v2,
  miss and hit, empty and 1M entries); `ratelimit::check` (managed tier only vs. BYO which runs
  both tiers) — single-threaded/hot-cache _and_ `check_flood_*`, which charges 65536 distinct
  credentials from 1 and 16 threads over ≥ 2 window rotations, plus `rotate_window`, which prices
  the window rotation on its own; `smart::rank` / `observe` (unmeasured and fully measured);
  `cache::key` over 0/4KB/64KB/256KB plus `ResponseCache::get` miss, hit, and a 16-thread shared
  hit; `translate` request and response (chat ↔ messages, including a 64KB body) and one SSE
  `text_delta`.

  Left unbenched on purpose. The open and half-open breaker are the failure path; the closed
  `allow` is what every request pays, and it is already measured. A hung failover waits out
  `connect_timeout_secs` — that number is configuration, not gateway CPU, and the instant-refuse
  case below already isolates the walk. The H2-vs-H1 and worker-thread sweeps stay on a small body
  so they measure protocol and cores; body-sized work is the unit benches plus
  `managed_large_anthropic_sse_throughput`.

  What the alloc numbers assert:
  | Operation                          | Cost                                | Allocations                   | Claim verified                   |
  | ---------------------------------- | ----------------------------------- | ----------------------------- | -------------------------------- |
  | `key/verify`                       | ~26µs                               | 0                             | Stack-only Ed25519 decode        |
  | `peek/ModelScanner`                | ~0.2µs at 4 KiB, ~3.6µs at 256 KiB  | 1 (independent of body size)  | O(1) memory                      |
  | `route`                            | ~ns                                 | 0                             | —                                |
  | `deny::reason`                     | ~0.3–2ns                            | 0, flat 0→1M entries          | O(1) lookup, O(denied) memory    |
  | `allowance::reason_for`            | ~1.3–3.3ns                          | 0, flat 0→1M entries          | Same claim; v2 probes two maps   |
  | `smart::rank` / `observe`          | ~110–150ns                          | 0                             | Atomics only, no lock            |
  | `ratelimit::check`                 | ~70ns; ~130–190ns at 16 threads     | 0                             | Fixed-memory, no per-key state   |
  | `ratelimit` rotation               | ~81µs                               | 0                             | Once per window, not per request |
  | `cache::key` (cache on)            | ~9µs at 64 KiB, ~35µs at 256 KiB    | 0                             | Two SipHash passes, ~7 GB/s      |
  | `ResponseCache::get` hit           | ~130ns; ~450ns median at 16 threads | 4 × 55 B, flat at 64 KiB body | `Bytes` clone, not a body copy   |
  | `translate` request, chat→messages | ~2µs small, ~34µs at 64 KiB         | ~46 allocs (serde DOM, freed) | Once per cross-wire request      |
  | `translate` SSE `text_delta`       | ~1.4µs                              | ~38 allocs / ~6 KiB           | Per event, not per request       |

  Fastest sample from one full `divan` run. Ratios against `key/verify` on that run are the claim;
  absolute µs move with the host.

  **Headline: on the always-on path, `key/verify` ≈ 26µs is still ~100–1000× deny, allowance, rank,
  and the rate-limit check.** That is why the rate guardrail sits before verify in
  `proxy::request_filter`. Cross-wire translate and an enabled response-cache fingerprint are the
  exceptions, and both are off the path a same-wire request with the default config pays. The SSE
  bridge is the one that adds up: ~1.4µs and ~38 allocations **per event**. A long translated
  stream pays that once a token, which is still small next to provider time-to-first-byte, and it
  is the first hot path in this suite that allocates per event rather than per request.

- **End-to-end (`benches/e2e.rs`, `mise run bench:e2e`) — `criterion`.** Real `beyond-ai` binary
  - real nats-server + mock upstream (reuses `tests/common`). Latency group:
    `reject_missing_key_latency` (401, short-circuit before any upstream connection — transport floor),
    `byo_json_latency` (BYO relay: no verify, no key swap), `managed_json_latency` (verify + deny + key swap),
    `managed_sse_latency` (a 3-line stream), `managed_large_sse_latency` and
    `managed_large_anthropic_sse_latency` (streams big enough to wrap the response tail, the
    Anthropic one splitting usage across head and tail), `managed_large_body_latency` /
    `byo_large_body_latency` (60 KiB, `model` last, under the 64 KiB replay cap), and the model route
    (`auto_json_latency`, `auto_large_body_latency`, `auto_failover_latency`). Throughput at 32
    in-flight for both a tiny JSON body and the large Anthropic stream. Then
    `e2e_concurrency` (HTTP/2 vs HTTP/1.1 to the upstream at 1/8/32/128/512) and
    `e2e_worker_threads` (1 worker vs one per core, same sweep).

  On one host the four small cases landed in ~110–120µs. A later run of the same harness did not:
  BYO ~73µs, reject ~103µs, managed and `/auto` ~113µs, a 60 KiB body ~149µs managed / ~105µs BYO.
  The managed−BYO gap is on the order of `key/verify`, which a noisy loopback floor can hide. This
  harness still cannot resolve anything at nanosecond scale (that's the unit bench). Its value:
  catching gross regressions (a buffering mistake, a dropped connection pool, an O(n) path added
  would move the band by far more than the run-to-run jitter) and saved-baseline RPS trend via
  `--save-baseline`. At 32 in flight the same later run did ~25k req/s of tiny JSON and ~6k req/s
  of the large Anthropic stream. Four workers against one roughly doubled throughput by concurrency
  128 (~31k vs ~15k req/s) and did not differ at concurrency 1. HTTP/2 to the upstream did not beat
  HTTP/1.1 at any point in the sweep on that loopback mock.

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
