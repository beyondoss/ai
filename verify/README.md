# Gateway verification

This directory holds every claim the gateway makes, and every suspected or confirmed defect against
those claims. Nobody writes down whether a claim holds. The `verify` tool (`crates/verify`) computes
it from the tests and their results.

| File           | What it holds                                                                  |
| -------------- | ------------------------------------------------------------------------------ |
| `claims.toml`  | Each claim's id, priority, layer, clients, `touches` paths and linked defects. |
| `defects.toml` | Each defect's severity, evidence, linked claims, state, test, fix and note.    |

## Tagging a test

```rust
/// claim: SEC-1, SEC-5
/// defect: D01
#[tokio::test]
#[ignore = "D01 reproduced: managed /{provider} forwards any path"]
async fn provider_route_rejects_files_api() {
    // asserts the CORRECT behavior
}
```

A test may claim several claims. The tag lines go directly above the attributes and the `fn`.

Every test in a live binary (`tests/*_live.rs`, `tests/live.rs`) whose name holds `::` is read as a
live cell, `CLAIMS::client::...`, and its claims must be known ids. A hermetic check inside one (a
`libtest-mimic` trial, not a `#[test]`) is named exactly after its function, with no `::`, and
the function carries the `/// claim:` tag, so it counts as a tagged hermetic test
(`oracle_empty_reasoning_item` in `parity_live.rs`).

## Defect lifecycle

| State        | What the gate requires                                                          |
| ------------ | ------------------------------------------------------------------------------- |
| `suspected`  | Nothing yet. The finding comes from reading code.                               |
| `reproduced` | `test` asserts the correct behavior, is `#[ignore]`d, and **fails** when run.   |
| `fixed`      | `test` passes, is no longer ignored, and `fix` names the commit.                |
| `refuted`    | `test` passes and `note` records why the behavior was already correct.          |
| `accepted`   | `note` records the decision, and `test` asserts the behavior as now documented. |

A defect that only a real client can settle (`refuted` or `fixed` by a live cell) names the cell as
`test = "live:<cell name>"`; the gate checks that cell's outcome in the last live run.

When a reproduction starts passing, the gate stops and asks for the defect to be marked `fixed` or
`refuted`. This keeps the ledger from saying "broken" after a fix, or "fixed" while the code is
still broken. An `#[ignore]`d test that reproduces no open defect is also an error. That is the only
reason a tagged test may be ignored.

## Claim status

| Status     | Meaning                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------ |
| `PROVEN`   | Every tagged test passed and no linked defect is open.                                     |
| `PARTIAL`  | Hermetic tests pass, but live cells are pending or INCONCLUSIVE, or some tests didn't run. |
| `RED`      | A tagged test fails, or a linked defect is still `suspected` or `reproduced`.              |
| `UNTESTED` | No test carries the id.                                                                    |

## Commands

```sh
mise run verify:status            # run tagged tests (ignored ones too), then print status
mise run verify:status -- --json  # the same as JSON (feeds the status page)
mise run verify:gate              # registry/tag/result consistency; non-zero on any problem
```

## One live run, one proof

`mise run verify:live` runs in two phases, so that what it reports holds for this tree and
nothing else:

1. Every tagged test and every live cell except the reconciled ones, concurrently
   (`verify filter`).
2. The reconciled live cells, the ones whose claims include BIL-5, after phase 1 has finished
   (`verify filter --isolated`, nextest profile `verify-isolated`, its own JUnit report). They run
   side by side; each reconciles its own model, and `long_live.rs` refuses to list two on the same
   one.

A reconciled cell compares the ledger with a provider's usage report for its key, model and
minutes, and that report holds every request anyone made there. No model is free of other suites:
the catalog sweep drives every row. So isolation is in time, not in the choice of model.
`tests/common/live.rs` enforces it between processes as well, so a cell run some other way (or a
second worktree's run on the same host) still can't overlap: every live process holds a shared
`flock` on `$TMPDIR/beyond-verify-live/traffic` while it sends, and a reconciled cell opens its
window only once it holds `window` and has seen `traffic` free. It then waits for the next whole
minute (the report's bucket), and holds the window until a minute past its last request, so no
earlier or later request shares a bucket with it. Live traffic that starts while a window is open
waits for it to close.

Ports come from `tests/common/live.rs` too. A port is reserved with an exclusive `flock` on
`$TMPDIR/beyond-verify-ports/<port>` for the life of the test process, then bind-checked, so no two
live processes on the host are ever handed the same one. A bind check alone races: Pingora binds
with `SO_REUSEPORT`, so two gateways that picked one port share it silently, and a nats-server
holding it instead makes the gateway give up.

### Provider unavailable: INCONCLUSIVE

A cell proves the gateway, so a provider that is overloaded at that moment is neither a pass nor
a gateway failure. The cells retry what the stock OpenAI and Anthropic SDKs retry (408, 409, 429,
any 5xx, or what `x-should-retry` says), as they do: two retries, waiting out `Retry-After` up to
two minutes, otherwise 0.5s doubling to at most 8s less up to 25% jitter (openai-python and
anthropic-sdk-python `_constants.py`). If the last answer is still the provider's own (it names
itself in `x-beyond-provider` and the ledger row records the same upstream status) on a request no
other provider could take (a one-provider route, a candidate forced with `x-beyond-only`, a direct
call), the cell fails with a message that starts `INCONCLUSIVE:`. `verify status` lists those cells
apart and keeps their claims from `PROVEN` without turning them `RED`; the gate rejects a defect
whose closing live cell was inconclusive. The cell's other problems stay in the message. An answer
the gateway made itself stays a failure, as does a relayed one where failover was possible.

Three more answers count as the provider's:

- **An error in a 200.** OpenRouter answers an error during a non-streaming generation with
  `200 OK`: a body of only an `error` object, whose `code` is the status it means, or a choice with
  `finish_reason: "error"` ([its errors page](https://openrouter.ai/docs/api-reference/errors)).
  The catalog sweep reads such a body as that status (502 when it names none), retries it as the
  SDKs retry that status, and never takes it for an answer.
- **A connection the provider ended.** The gateway answers 502 "upstream failed after receiving the
  request" when the upstream closed, reset or sent GOAWAY without answering. The catalog sweep
  reads the error the gateway logged for that request (`upstream request errored`): when it is the
  candidate under test ending the exchange (a refused or dropped connection, a reset or GOAWAY the
  peer sent), the cell is INCONCLUSIVE. A frame error against what the gateway sent, or a gateway
  timeout (`WriteTimedout`, D118), stays a failure.
- **A best-effort feature that didn't happen.** A probe whose oracle rests on something the provider
  documents as best effort says so (`detail.best_effort`) when it misses: OpenAI's prompt-cache hits
  (`auto_cache` on the GPT row). The cell is INCONCLUSIVE when its ledger is clean, a failure when
  it is not.

**Retries fit the test's time budget.** nextest terminates a test at its profile's `slow-timeout`
period × `terminate-after` (180s under `verify`; the reconciliation and long-session binaries have
their own, as do live.rs's `long_output*` cells, matched by test name: 6 minutes for a ~32k-token
generation measured at 135-177s). The retry loop (`common::sdk_retry`, `sdk_backoff`) reads that
budget from `.config/nextest.toml` for `NEXTEST_PROFILE`, `NEXTEST_BINARY_ID` and
`NEXTEST_TEST_NAME`, and gives up a retry whose
wait plus one more attempt as long as the last would not finish inside it, keeping five seconds to
read the billing row and report. The catalog sweep also caps each request's timeout at the time
left. So a cell reports its last answer (INCONCLUSIVE, where it is the provider's) instead of being
killed as a TIMEOUT. A client process (a `live.rs` probe, a harness in `live.rs` or `long_live.rs`)
gets the same budget less ten seconds (`common::run_client`), as `VERIFY_DEADLINE_SECS`:
`harness.py` stops the agent there and reports with `detail.timed_out` and what it printed, and a
client still running five seconds later is killed with its process group. A stopped session fails
saying so, or is INCONCLUSIVE when its provider had the last request whole and sent no response
head for a minute (`common::note_if_stalled_on_provider`): `E7::pi::gpt::pi:responses` was killed
as a TIMEOUT on 2026-10-01 while OpenAI held pi's fifth request for 172s, after answering its
first four in 1-2s (15 re-runs answered in 8-14s). Outside nextest there is no budget. A failed catalog run keeps its gateway's
directory (`target/catalog-live/gw-<pid>/gateway.log`), so a failure can be read after the fact.

| Suite                                                         | How a cell retries                                                                                                                                      |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `live.rs` probes (SDK and raw HTTP)                           | The probe's client runs with retries off (one HTTP call, one row); the whole cell runs again on a new gateway.                                          |
| `catalog_live.rs`                                             | Each request is retried in place.                                                                                                                       |
| `parity_live.rs`                                              | Each request (direct or through the gateway) is retried in place.                                                                                       |
| `tenancy_live.rs`, `reconcile_live.rs`, TOOL-1                | The whole cell runs again (no `Retry-After` to read: the backoff). A provider holding only revoked keys (REL-14) is not counted as one to fail over to. |
| LNG sessions, MCP cases, `live.rs` harness and recorded cells | The harness already retried with its own defaults; a session that ended on its provider's retryable answer is INCONCLUSIVE.                             |
| `fault_live.rs`, `session_live.rs`                            | Not applied: their providers fail on purpose (injected faults, killed upstreams), so a relayed failure is the subject.                                  |

A cell that is INCONCLUSIVE on every run is worth a look: a provider 500 can be its deterministic
answer to a request shape the gateway produced.

## Billing reconciliation (BIL-5)

`crates/verify/tests/reconcile_live.rs` checks the ledger against the providers' own books. Each
trial (`BIL-5::raw::openai::reconcile`, `BIL-5::raw::anthropic::reconcile`) boots a gateway with
that provider's pool key, sends a fixed batch through it (non-stream and stream, a translated
pairing, a ~3k-token system prompt reused so cache writes and reads occur, and Responses on
OpenAI), and sums the batch's `ai.usage` rows, normalized by `usage_wire`. It then reads the
provider's organization usage report for the same minutes, filtered to the pool key's id and the
batch's model (`gpt-4.1-mini`, `claude-sonnet-5-5`; a report row matches the model or a dated
snapshot of it, never another model it prefixes). It runs in verify:live's isolated phase (see
[One live run, one proof](#one-live-run-one-proof)), so no other request is in those minutes. It
passes only when uncached input, cache reads, cache writes and output agree exactly (and the request
count, where OpenAI reports one). So it proves that every token the provider charged our key for is
in a billing row, and that no row bills a token the provider didn't charge.

A trial is listed only with `VERIFY_LIVE=1` and both of that provider's keys set: `OPENAI_ADMIN_KEY`
and `OPENAI_API_KEY`, or `ANTHROPIC_ADMIN_KEY` and `ANTHROPIC_API_KEY`. A missing key means the
trial isn't listed, never that it fails. The admin keys are used only on read-only endpoints (key
listings and usage reports).

Usage reports lag traffic: Anthropic documents "typically within 5 minutes", but on 2026-10-10 its
report lagged 45-60 minutes. A trial polls every three minutes (the admin API allows 90 requests a
window, shared by the organization) until the provider's totals match and hold, for up to 90
minutes.

Once the tokens agree, the trial checks the money (BIL-24). It prices the provider's own totals
with the reference pricer (`providers::pricing`, this build's `RATE_VERSION`) and compares them
with the sum of the rows' `cost_micros`. They may differ by at most one micro-dollar per row,
because each row rounds half-up once.

## Pricing reconciliation (BIL-24, BIL-25)

`crates/verify/tests/pricing_live.rs` holds the gateway's logged `cost_micros` to what vendors
actually bill. Pricing is pass-through, so every case also checks `price_micros == cost_micros`.
It runs one case per dimension:

- Anthropic `inference_geo: us`, fast mode, a 1-hour cache write then read, web search and web
  fetch;
- OpenAI priority (Fast mode), flex and Responses web search;
- an OpenRouter call and an OpenRouter cancelled stream;
- a Bedrock `us.` profile (list × 1.1);
- an xAI web search;
- the long-context tiers on each vendor's cheapest long-context row (OpenAI gpt-6-luna past 272K,
  against its usage report; xAI grok-build-0.1 at 200K, against its own billed cost);
- a cancelled stream on Bedrock (and Groq, with a key), drained by the gateway: the same
  deterministic prompt run to completion and cut after a second must write the same row
  (`usage_settled=drained`, priced), recorded in `target/verify-cancel-gap.jsonl` (BIL-26).

The file's header lists each case's budget, about $0.65 in all. The Anthropic admin API allows 90
requests a window, shared by the whole organization. The cases read it six minutes after their request, then every three, through one per-process gate, and wait out
its `retry-after`.

- **Dollars, where the vendor reports them, must match exactly.** On OpenRouter the row's
  `upstream_cost_usd` must equal `usage.cost` and the generation API's `total_cost`, and
  `cost_micros` must equal that × 1.055. On xAI the rate table's price of the row's tokens and
  tools must equal xAI's `cost_in_usd_ticks` to the micro-dollar.
- **Usage, where the vendor reports only usage, must match exactly.** First the response's usage
  block, then the admin usage report for the pool key, model and minutes, filtered by
  `inference_geos[]` / `speeds[]` where the dimension is one (a case with the admin key claims
  BIL-5 too and runs in the isolated phase). The logged cost must then be the rate table's price
  of those facts.
- **A disagreement is a finding to fix in the rate table**, never a tolerance to widen.
- **The cancelled-stream cases** hold a drained row to the vendor's own bill: on OpenRouter the
  row's `cost_micros` must equal the generation API's `total_cost` × 1.055, on Bedrock the cut row
  must equal the full run.
- **Long-context boundaries** (exactly at and one past each threshold) are pinned by the golden
  vectors (`verify/pricing_vectors.json`). The live cases prove one request on each side's tier.

**BIL-25 (invoice level)** is listed when `VERIFY_INVOICE_ROWS` names a JSONL file of rows (keep
one with `VERIFY_ROWS_OUT`). For every whole UTC day in it, the trial compares the rows' summed
`cost_micros` with Anthropic's daily cost report for the key's workspace, or OpenAI's costs for the
key's project, within one cent or 0.1%. It prints a per-day diff table. Both reports are daily and
group by workspace or project, not key, so they reconcile only a key whose workspace or project
carries nothing else.

Other vendors are covered differently:

- OpenRouter is reconciled per generation instead.
- Bedrock, Together, Fireworks, Groq and DeepSeek expose no per-key cost API this suite can read.
  Their costs are held by the rate table and its sources (`verify/catalog_truth.toml`).

```sh
VERIFY_LIVE=1 cargo nextest run -p beyond-ai-verify --test pricing_live --profile verify
VERIFY_INVOICE_ROWS=$PWD/target/rows.jsonl VERIFY_LIVE=1 cargo test -p beyond-ai-verify --test pricing_live -- BIL-25
```

## Live cells

`crates/verify/tests/live.rs` runs real SDKs and agent harnesses through a gateway built from this
tree to real providers, one cell per `(claims, client, route, probe)`. Each probe
(`verify/clients/py/probe.py`, `verify/clients/node/probe.mjs`) asserts its claim's oracle and
reports every HTTP call it made. The cell then holds each call to exactly one `ai.usage` row with
the tokens the client saw. A call whose row is not the ordinary one says so with `expect`: no row (a
free token count, a BYO key), a refusal billed nothing, a cut-short estimate bounded by the same
request completed, the provider it must land on, or a row field's minimum (cache reads, 1-hour
writes, server tool calls). Image and PDF fixtures live in `verify/clients/fixtures/`. A probe
shared by routes that can and can't serve a feature reads the cell's claims (`VERIFY_CLAIMS`) and
asserts the feature only where its cell claims it (`ai_sdk_responses`'s structured output, T4).

A `raw` cell is the same with no SDK: httpx on the wire (`raw_*` probes in `probe.py`), streams
read as SSE by hand. A coding agent's calls are invisible to a probe, so its `W*` and `E7` cells
check the ledger in aggregate. Its other cells run behind `harness_long.py`'s recording proxy
(`harness_long.py cell <harness>+<scenario>`), and every call it made is held to the ledger by
request id: a served call has exactly one row on the route's provider, a free or refused one bills
nothing, and no row is left over. A scenario can ask more of a call (an estimate for a stream the
agent was interrupted in, a cache read from turn 2, one provider for the whole session). Where the
cell's claims hold the client's usage to the ledger (E1, E2, B3, B1), the usage each response showed
the harness on the wire (input, output, cache reads) must equal its row, as a probe's must. Where
they claim T1, a served turn must replay one of the agent's own tool calls with its result, paired
by id, and no turn feeding a result back may be refused. Claude
Code's S1 cell streams one answer straight to Anthropic (through a second recorder) and then
through the gateway, and compares time to first token and how spread out each stream arrives. A
claim's `client_note` says why a client it would name can't exercise it, or what part of the claim
it can.

```sh
VERIFY_LIVE=1 cargo nextest run -p beyond-ai-verify --test live -E 'test(/::stream_abort$/)'
VERIFY_ROWS_OUT=$PWD/target/rows.jsonl VERIFY_LIVE=1 cargo nextest run ...  # also keep every row
```

## Differential parity

`crates/verify/tests/parity_live.rs` checks the promise "same as calling the provider". A seeded
corpus (`PARITY_SEED`, default 1) of 36 short logical requests covers text, system prompts,
multi-turn, unicode, stop sequences, max-token cut-offs, every `tool_choice`, parallel calls, tool
results, `json_schema`, a base64 PNG carrying a code word, reasoning effort and two invalid requests,
each streamed and not. Each request is sent directly to the provider with the real key and through a
gateway that holds only that provider's pool key, on the same model. Same-dialect paths (OpenAI Chat
and Responses, Anthropic Messages, xAI Responses, OpenRouter Chat) compare the answers' structure:
status class, the client's error envelope (an error not already in it, xAI's, is re-encoded: D100)
and its type and code, the type skeleton, finish reason, tool names and argument validity,
structured-output validity, whether the code word came back, usage, and for streams the event-type
sequence, each event's keys, and that events still arrive spread out. Cross-dialect paths (Chat and
Responses clients on Claude, a Messages client on GPT, a Chat client on grok, which reaches xAI over
Responses) compare the gateway's translation with the
same request sent natively: status class, the client's error envelope, finish class, tools,
structured output, the code word and input size. Every gateway answer must also match its
`ai.usage` row.

A trial is named `CLAIMS::raw::ROUTE::parity_CASE` and is listed only with `VERIFY_LIVE=1` and the
path's key. A mismatch is retried once before it fails, so one nondeterministic answer isn't
reported as a defect. Run the suite twice, and file only differences that reproduce. Allowed
differences: values (only types are compared), headers, the usage chunk the gateway injects into a
Chat stream, OpenRouter's repeated `delta.role` (dropped by `ChatIdentity`), and grok's reasoning
visibility (a Chat `reasoning_content`, a Responses `reasoning` item and its summary events), which
varies even between two direct calls, as does whether an OpenAI reasoning model that did no
reasoning (zero reasoning tokens on both answers) emits an empty `reasoning` output item (5 direct
calls of 120 of one gpt-5-mini body came back without it). The GPT paths run on gpt-5.1, sent
effort `none` (its lowest) where a case asks for no reasoning. A run costs about $0.30; it prints an
estimate first and the measured cost last.

```sh
VERIFY_LIVE=1 cargo test -p beyond-ai-verify --test parity_live              # one process, shared gateways
PARITY_DUMP=1 VERIFY_LIVE=1 cargo test -p beyond-ai-verify --test parity_live -- chat-to-claude --nocapture
```

## Catalog sweep

`crates/verify/tests/catalog_live.rs` forces every in-scope catalog candidate (OpenAI, Anthropic,
OpenRouter, xAI, Bedrock, Together) through a gateway with `x-beyond-only` and checks CAT-1..CAT-8
and CAT-13 (BIL-9 and BIL-13 on the way) with raw HTTP. Its cells are named `CLAIMS::raw::ROUTE::ROW`
and count toward claim status like the client cells. `mise run verify:catalog` runs only the sweep
and prints its cost (each billed call is priced from the ledger into
`target/catalog-live/<VERIFY_CATALOG_SWEEP>.jsonl`). `VERIFY_CATALOG_PLAN=1` prints the trials, the
estimated cost, and every trial left out with its reason (no key, over the per-call cost cap, a
provider window an over-limit prompt would be billed against, or a CAT-3 prompt OpenAI's rate
limiter can never admit on the pool key's project).

CAT-3 on OpenAI is held to the project's own tokens-per-minute limits, read with `OPENAI_ADMIN_KEY`
from `GET /v1/organization/projects/{id}/rate_limits` (the project whose key hint matches
`OPENAI_API_KEY`). The limiter answers before the context check: a prompt over the TPM is a 429
at once ("Request too large" on Chat Completions; "You've exceeded the rate limit, please slow
down" on Responses), and one over half of it a 429 "Rate limit reached ... Used N, Requested N"
after up to a minute (the limiter counts the request against the minute before comparing; measured
2026-10-01 on gpt-4.1, gpt-5-pro, gpt-5.4-pro and gpt-5.5-pro). So an over-limit or near-limit
prompt is planned only when twice its size fits the model's own TPM. A larger `-long-context`
limit listed beside it doesn't raise that: gpt-5.5-pro (500K, long-context 2M) refuses a 970K
prompt at once, as gpt-5.4-pro (200K) does, while gpt-5.5 (2M) answers the context error. A listed
long-context limit can still refuse a prompt over it, so the prompt must fit that too. A 429 is never
read as a context error: "exceeded the rate limit" used to match the context words, and
`CAT-3::raw::openai::{gpt-4.1,gpt-5.2-pro,gpt-5.4-pro,gpt-5.5-pro,o1-pro}` passed on rate limits
until 2026-10-01. Those over-limit trials are now left out with that reason.

CAT-16 keeps the catalog current, from listing calls only (no completions, no gateway, free).
`CAT-16::raw::PROVIDER::ROW`, one per candidate, fails when the vendor no longer offers it: not in
its own models API (Anthropic by snapshot, xAI by alias), not in Together's serverless table, a
Bedrock inference profile that is not `ACTIVE` or a foundation model that is `LEGACY`, or, on
OpenRouter, no endpoint up in the last 30 minutes (one at 0% uptime answers 410) or none that takes a
capability the card lists. An endpoint counts as up when it answered in the last 30 minutes, or had
no traffic to measure and OpenRouter has not flagged it: one with no recent traffic and a negative
`status` (Fireworks' dead `z-ai/glm-5.2` lists `status` -5 and no uptime) vouches for nothing.
These are listings: a capability an endpoint lists but fails to serve for a while (OpenRouter's
error-in-200s on `x-ai/grok-4.20` for a minute on 2026-10-01, with every endpoint still listing
`tools` and `structured_outputs`) is the provider being unavailable, which the CAT-6 cells report
as INCONCLUSIVE. It also fails when the vendor's deprecation page (Anthropic, OpenAI,
Together; for an OpenRouter slug, its maker's) retires the id, or a dated snapshot of it, without a
`[[retired]]` entry in `catalog_truth.toml`, or when a recorded retirement is due. So a
retirement shows up as a red cell, not a customer 404. `CAT-16::raw::{anthropic,openai,xai}::new-models`
fails on any model those vendors list (xAI's `/v1/models`, image and video models included) in a
family the catalog carries (Claude; GPT, `chatgpt-` and `chat-latest`, and the o-series; Grok)
that is not a row, a vendor-listed alias of a row, a dated snapshot of one, or recorded in
`[[not_carried]]` (with a reason) or `[[retired]]`, so a release is a visible gap. Nothing in a
family is skipped by name or modality: an audio, realtime, image or video model is recorded in
`[[not_carried]]` one by one, saying which endpoint it needs that the gateway doesn't serve. Together and OpenRouter list
hundreds of models, so their gaps are a report, not a failure: `VERIFY_CATALOG_GAPS=1` prints recent
models in the namespaces the catalog carries that are not in it (`mise run verify:catalog` prints
it last). The hermetic `no_catalog_row_outlives_its_retirement` always fails on a row or candidate
named by a `retired` entry. For a recorded `retires` date it prints a warning naming the row and
date from 14 days before that date, and fails only under `CATALOG_RETIREMENT_STRICT=1`: the weekly
`deep.yml` job sets it, so a scheduled retirement is acted on before the vendor's 404 while the
calendar never fails an unrelated PR. Run it strict locally with
`CATALOG_RETIREMENT_STRICT=1 cargo test -p beyond-ai-providers --lib no_catalog_row_outlives_its_retirement -- --nocapture`.

```sh
VERIFY_LIVE=1 cargo nextest run -p beyond-ai-verify --test catalog_live --profile verify -E 'test(/^CAT-16::/)'
VERIFY_CATALOG_GAPS=1 cargo test -q -p beyond-ai-verify --test catalog_live
```

## Tenancy sessions (TEN-1, TEN-2)

`crates/verify/tests/tenancy_live.rs` runs real SDK sessions while the control plane changes under
them, and two tenants side by side. Each trial (`CLAIMS::client::route::scenario`) boots its own
nats-server and gateway, mints `bai_v2` keys for two tenants from the dev signing key, and runs one
scenario of `verify/clients/py/tenancy.py`. The scenario drives the clients and the timing; the cell
writes the `ai-gateway` KV bucket (`blackhole.*`, `allowance.*`) when the scenario asks, over a bare
NATS connection, and checks the ledger afterwards: each call has one row with the tenant and key it
used, the tokens it was shown and the cache hit and provider it saw; a refused call bills nothing;
no row is unaccounted for.

| Scenario                 | What it proves                                                                                                                                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `revoke_tenant`          | `blackhole.{tenant}` written mid-stream: the stream finishes and is billed exactly; the next call 402s within 2s, the tenant's other key too, the other tenant doesn't; deleting it restores within 2s                                                              |
| `revoke_key`             | the same for `blackhole.key.{id}`: the tenant's sibling key keeps working                                                                                                                                                                                           |
| `exhaust_allowance`      | the same for `allowance.{tenant}`: 402 `insufficient_quota`                                                                                                                                                                                                         |
| `claude_code_revoked`    | Claude Code's key is denied after its first billed call: it exits promptly with the 402 named, no retry loop                                                                                                                                                        |
| `cache_isolation`        | with `cache_ttl_secs`, identical sessions: a tenant replays its own fills, never the other tenant's                                                                                                                                                                 |
| `pin_isolation`          | a tenant-A key pinned to the provider a new caller isn't ranked to; tenant B's key with the same vpc and key id is ranked, not steered                                                                                                                              |
| `tenant_limit`           | `tenant_max_in_flight = 2`: A's third concurrent call 429s; B's two are served                                                                                                                                                                                      |
| `rate_limit`             | `rate_limit_rps = 3`: a burst on one credential 429s; the tenant's other key and the other tenant don't                                                                                                                                                             |
| `responses_id_isolation` | SEC-25: tenant B sends tenant A's real Responses ids (as `previous_response_id`, as an `item_reference`): each is a 400 naming the field, billing nothing; B's full item carrying A's id is answered from B's content, never A's codename; A's own chain recalls it |

A run costs about $0.10.

```sh
VERIFY_LIVE=1 cargo test -p beyond-ai-verify --test tenancy_live -- --nocapture --test-threads=4
```

## Faults in front of real providers (FLT-1)

`crates/verify/tests/fault_live.rs` puts a fault proxy (`tests/common/fault_proxy.rs`) between the
gateway and each real provider a route uses. The gateway is pointed at it with
`provider_authorities` and `upstream_verify_cert = false`, so it still dials TLS with its own pool,
ALPN and error handling; the proxy's throwaway cert offers only `http/1.1` (H2 is out of scope).
The proxy re-originates TLS to the real host, reads each request in full and, per a script, answers
with a provider-shaped `5xx` or `429` (with `Retry-After`), resets the connection before the TLS
handshake, after reading the request, or after the provider answered, holds the provider's answer
back past the gateway's read timeout, cuts or stalls a stream after N events, or slows every event
down. It records each request: whether the provider processed it,
the provider's status, how much of the response reached the gateway, and the usage the provider
reported. A response it cuts off is still drained from the provider, so that usage is what the
provider billed.

A trial is `FLT-1+<claims>::<client>::<route>::<fault>`: the four stock SDKs at their default
`max_retries` (two retries), on a Claude row (Anthropic, then OpenRouter), a GPT row (`gpt-4o-mini`:
OpenAI, then OpenRouter), and single-provider versions of both holding their key twice (so a `429`
can key-walk). The fault goes on the primary. Each trial checks the client's final outcome and
attempt count against `crates/gateway/ARCHITECTURE.md` (failover, relayed status, a JSON error with
`x-beyond-request-id`, `Retry-After` waited out, a cut stream raised as an error), that providers
processed at most one generation per client attempt, and that every generation whose response
reached the gateway, or that the gateway waited out with the request delivered, has exactly one
billed row with the provider's tokens (an estimate, never above them and never zero, when cut
short or waited out) and nothing else is billed. A generation whose connection was reset before
any head is not billed: the gateway reads a bare reset as the peer declining to answer (D130). The gateway's `read_timeout_secs` is 20 in
these trials (default 600) so a stall costs seconds.

```sh
VERIFY_LIVE=1 cargo nextest run -p beyond-ai-verify --test fault_live -j 12
VERIFY_FAULT_VERBOSE=1 VERIFY_LIVE=1 cargo nextest run ... --no-capture  # print every witness
```

## Long sessions and large tool sets (LNG-1, LNG-2, TOOL-1)

`crates/verify/tests/long_live.rs` drives clients through `verify/clients/harness_long.py`, which
puts a recording proxy between the client and the gateway. So every HTTP call a coding agent makes
is held to the ledger: one `ai.usage` row per successful billed call, none for a free one, nothing
billed for a refusal, and no row without a call.

- LNG-1: Claude Code, Codex and pi work through `ledgerlib`, a fixture repo with eight ordered
  steps, each with its own tests (30-80 model calls). Each harness's compaction knob is turned
  down so auto-compaction happens mid-session. The trial checks that the repo's tests pass, that
  compaction shows in both the harness's events and on the wire, and that the ledger is complete.
  Two sessions reconcile against the provider's usage report (the BIL-5 method): pi on
  `claude-sonnet-5` and Claude Code (translated) on `gpt-5.1`. Where the provider's admin key is
  set they claim BIL-5 too (`LNG-1+BIL-5::claude-code::gpt-5.1::long_task_uncompacted`,
  `LNG-1+LNG-2+BIL-5::pi::claude-sonnet-5::long_task_messages`), which runs them in verify:live's
  isolated phase.
- LNG-2: the per-turn cache share (`cache_read / input_total`) over the same sessions, plus
  opencode, must clear a floor after warm-up. The floor and how it was chosen are in the file's
  doc comment.
- TOOL-1: SDK requests with N tools on every wire (Chat, Responses, Messages, Codex's `namespace`
  tool), below and above OpenAI Chat Completions' 128-tool limit, plus large schemas. Two
  harnesses are also offered 150 MCP tools.

A long session takes 3-10 minutes, and a reconciled one waits up to 15 more for the usage report.
A full run costs about $3. Harness homes live in `/var/tmp/verify-long-*`, outside the user's home:
Claude Code reads every `CLAUDE.md` on the way up from its working directory. `VERIFY_LONG_KEEP=1`
keeps them (with every call and its response tail) and each gateway log under
`target/verify-long/`.

```sh
VERIFY_LIVE=1 cargo nextest run -p beyond-ai-verify --test long_live --profile verify
```

`STALE` status will land with the run ledger.

## Session trials (SES-1..3)

`crates/verify/tests/session_live.rs` runs whole conversations where something changes between
turns, named `CLAIMS::client::route::scenario`. SDK scenarios live in
`verify/clients/py/session_probe.py`, coding-agent ones in `verify/clients/session_harness.py`.

| Claim | What changes                          | How the trial does it                                                                                                                                                                                                                                 |
| ----- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SES-1 | The primary provider dies mid-session | Every pool provider sits behind `verify/clients/py/upstream_proxy.py`, an HTTP to HTTPS reverse proxy (the gateway runs with `upstream_tls = false`). The primary's proxy exits after serving `k` requests, so later turns are refused and fail over. |
| SES-2 | The client switches models            | Claude and GPT alternate each user turn (SDKs), or `pi --continue` / `opencode run --continue` run the next step on the other model and dialect.                                                                                                      |
| SES-3 | Nothing; state lives upstream         | `previous_response_id` chains (openai-py, the Agents SDK), the Responses arm's only upstream dying, and `codex exec resume --last`.                                                                                                                   |

Each SDK call names the provider that must serve it, and is held to exactly one `ai.usage` row there,
with the tokens the client saw. A coding agent's rows are held to the order of its steps: the models
it ran, or the primary first and the fallback last. Every scenario asserts recall of a per-run
codename from turn 1, so a turn that lost its history fails. `VERIFY_SESSION_TRACE=1` prints what a
passing trial saw (rows, proxy request logs, the client's detail).

```sh
VERIFY_LIVE=1 cargo test -p beyond-ai-verify --test session_live -- SES-1
```
