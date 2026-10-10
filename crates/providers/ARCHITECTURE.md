# beyond-ai-providers — Architecture

A pure library with no I/O, below both the gateway and the agent. It holds three tables and the
lookups over them:

- **Providers** (`lib.rs`): which upstream a name, host or model id means, and how to talk to it
  (authority, auth header, wire).
- **The catalog** (`catalog.rs`): canonical model name → the ordered candidates that serve it, plus
  the list price and card `GET /v1/models` publishes.
- **The pricing contract** (`pricing.rs`, `rates.rs`, generated `rates/generated.rs`): what one `ai.usage` row costs us and what
  we charge for it.

The first two are documented in their module docs. This file is the contract for the third.

---

## Pricing contract

This repo owns the billing contract. `providers::pricing::price` is the reference implementation.
`verify/pricing_vectors.json` holds the golden vectors any other implementation (beyond's billing,
a repricer, an invoice audit) must reproduce exactly. The rates are generated from snapshots of
their primary sources (`verify/rates_sources/`; see "Rate data and versions"). If this section and the code disagree, the vectors settle it. The code and
vectors are fixed first, then this text.

### The model

One row in, two amounts out, both in integer micro-dollars (1e-6 USD):

- **cost**: what we owe the vendor that served the row.
- **price**: what we charge the customer.

**Pricing is pass-through** (owner decision, 2026-10-10). The customer pays exactly what we pay
for each request, fees included, so `price == cost` on every row. Both come from the card of the
host that **actually served** the request, under that host's own rules (long-context tier, fast
mode, data residency, off-peak hours, server-tool fees):

| Serving candidate                                               | Card                                                                                                            |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| The row's own vendor (Anthropic, OpenAI, xAI, DeepSeek, Groq)   | The row's list card (`CostCard::List`)                                                                          |
| Another host (Bedrock, Together, Fireworks, a non-primary Groq) | That host's own published card (`CostCard::Own`); Bedrock's `us.` profile is list + 10%                         |
| OpenRouter                                                      | `usage.cost` as reported × 1.055 (the credit fee); with no reported cost, the dearest matching endpoint × 1.055 |
| A candidate whose rates no primary source gives (none today)    | Refused (`CostCard::Unverified`)                                                                                |

The **list card** (`RowRates::list`) is the catalog row's primary vendor's published card. Its
standard tier is exactly the catalog's `ListPrice`, which a test enforces, so `GET /v1/models`
publishes what the primary vendor charges. A failover host prices at its own rates, which can be
above or below the list. An estimated row (a stream cut short, a cancelled stream on a host that
may keep generating) is priced from the same estimate as its cost. `pass_through.rs` asserts
`price == cost` across every vector, and across every card under a sweep of usage shapes.

Anything the tables cannot price exactly is an `Unpriced` error with a stable reason code. It is
never a zero. A zero is only ever a priced zero: a cache hit, or a row whose exact cost rounds
below half a micro-dollar.

### Input: the row

`UsageRow` reads the `ai.usage` row's fields by their row names (the row's contract is
`crates/gateway/ARCHITECTURE.md`, "The `ai.usage` row"). The pricer prices what was **served**,
never the `requested_*` values.

| Field                        | Meaning                                                                                                                   | Default     |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `price_model`                | The catalog row to price at                                                                                               | none        |
| `provider`                   | The provider that served (`anthropic`, `bedrock`, `openrouter`, …)                                                        | none        |
| `price_variant`              | `regional` / `global`. Accepted: `regional` on Bedrock (every Bedrock candidate is a `us.` profile), either on OpenRouter | none        |
| `usage_wire`                 | `openai` or `anthropic`: the convention `input_tokens` follows                                                            | `anthropic` |
| `input_tokens`               | As the wire reports it (see Normalization)                                                                                | 0           |
| `output_tokens`              | Reasoning included                                                                                                        | 0           |
| `cache_read_tokens`          |                                                                                                                           | 0           |
| `cache_write_tokens`         | All cache writes, 1-hour included                                                                                         | 0           |
| `cache_write_1h_tokens`      | The 1-hour subset of `cache_write_tokens`                                                                                 | 0           |
| `gateway_cache_write_tokens` | Writes from breakpoints the gateway added: inside `input_tokens`, not in `cache_write_tokens`                             | 0           |
| `server_tools`               | The row's text, `kind=count` joined with `,` (see Server tools). An unknown kind or bad count is `unknown_tool`           | empty       |
| `service_tier`               | The tier the provider says it served at                                                                                   | none        |
| `speed`                      | The speed the provider says it served at (`standard` / `fast`)                                                            | none        |
| `inference_geo`              | Where the provider says inference ran (`us` / `global`)                                                                   | none        |
| `upstream_cost_usd`          | The vendor's own reported cost, decimal USD (OpenRouter `usage.cost`; xAI `cost_in_usd_ticks` ÷ 10^10)                    | none        |
| `upstream_tool_cost_usd`     | OpenRouter's metered tool and plugin cost                                                                                 | none        |
| `served_by`                  | The host OpenRouter routed to (its provider name)                                                                         | none        |
| `usage_estimated`            | Some count is the gateway's estimate                                                                                      | false       |
| `upstream_may_continue`      | The stream ended early on a provider that keeps generating: the tokens are a lower bound                                  | false       |
| `cache_hit`                  | Served from the gateway's response cache                                                                                  | false       |
| `unix_secs`                  | Request start, Unix seconds UTC: the row's timestamp less `latency_ms`                                                    | 0           |

`reasoning_tokens` is on the row for audit, but the pricer never reads it. Reasoning is already
inside `output_tokens` on every wire the gateway meters, and the extractor folds xAI's
beside-count convention in.

### Output

```text
Ok(Priced { status, basis, cost: Side, price: Side })
Side { micros, class, long, multiplier, parts: { input, cache_read, cache_write_5m, cache_write_1h, output, tools } }
Err(Unpriced)
```

| `status`    | When                                                                                                                             |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `priced`    | From the provider's reported usage, or a cache hit                                                                               |
| `estimated` | `usage_estimated`, `upstream_may_continue`, or an OpenRouter row with no reported cost (priced at the dearest matching endpoint) |

| `basis`            | Cost from                                                    |
| ------------------ | ------------------------------------------------------------ |
| `tokens`           | The candidate's card times the tokens                        |
| `reported`         | `upstream_cost_usd`: OpenRouter's × 1.055, xAI's as reported |
| `dearest_endpoint` | OpenRouter, no reported cost: an upper bound, plus the 5.5%  |
| `cache_hit`        | Nothing owed                                                 |

`Unpriced` codes: `no_price_model`, `unknown_model`, `unknown_provider`, `not_a_candidate`,
`unverified`, `unknown_variant`, `unknown_class`, `unknown_geo`, `no_rate`, `no_tool_fee`,
`unknown_tool`, `inconsistent_tokens`, `bad_reported_cost`, `unknown_host`, `unknown_calendar`,
`overflow`. The
gateway logs the code as the row's `price_status` reason, never a zero.

`parts` are each rounded on their own and explain an invoice line. `micros` is rounded once from
the exact sum, so the parts can differ from it by a few micro-dollars. A reported cost has no
parts.

### Normalization: every prompt token in exactly one bucket

| Bucket           | `anthropic` wire               | `openai` wire                                           |
| ---------------- | ------------------------------ | ------------------------------------------------------- |
| uncached input   | `input_tokens`                 | `input_tokens − cache_read_tokens − cache_write_tokens` |
| cache read       | `cache_read_tokens`            | `cache_read_tokens`                                     |
| 5-minute write   | `cache_write − cache_write_1h` | `cache_write − cache_write_1h`                          |
| 1-hour write     | `cache_write_1h_tokens`        | `cache_write_1h_tokens`                                 |
| gateway's writes | inside uncached input          | inside uncached input                                   |

On the OpenAI wire, input includes cache reads, and OpenRouter's (and GPT-5.6+'s) cache writes.
On the Anthropic wire it includes neither. Subtracting on the wrong wire double-bills a cached
token or bills it free. A row whose subtraction would go negative is `inconsistent_tokens`, as is
one with more 1-hour writes than writes, or more gateway writes than uncached input.

The gateway's own cache writes (`gateway_cache_write_tokens`) sit inside the uncached input. The
vendor bills them as **5-minute writes**, and pass-through bills them the same.

### Choosing the rates

1. **Class**, from `service_tier` and `speed`:

   | Row says                                                                            | Class           |
   | ----------------------------------------------------------------------------------- | --------------- |
   | none, `default`, `standard`, `on_demand` (Groq)                                     | standard        |
   | `priority` or `fast` (OpenAI, xAI, Fireworks, OpenRouter)                           | fast            |
   | `speed: fast` (Anthropic; with `service_tier: priority` on OpenRouter)              | fast            |
   | `ultrafast`                                                                         | ultrafast       |
   | `flex`                                                                              | flex            |
   | anything else, two classes at once, or a non-`standard` tier from Anthropic/Bedrock | `unknown_class` |

   Anthropic's own `service_tier: priority` is Priority Tier, contract-priced committed capacity
   that can no longer be bought. It is not fast mode, so it is refused.

2. **Long-context tier.** If the card has one, the whole prompt (uncached + cache read + both
   writes) is compared with its threshold: `>=` where the vendor says "reaches" (xAI, 200,000),
   `>` where it says "more than" (OpenAI, 272,000; OpenRouter's `min_prompt_tokens`). Above it,
   **every** token in the request is billed at the long rates. Each side decides on its own card,
   so a list card can be long while a host's flat card is not.

3. **Off-peak** (DeepSeek), for a standard-class, short request. Peak is 01:00–04:00 and
   06:00–10:00 UTC, Monday to Friday, with windows half-open. Everything else is off-peak, at half
   the rates: weekends, and Chinese public holidays in full. The holiday list covers a span of days
   (2026-10-10 to 2026-12-31 today, which holds no statutory holiday). A weekday peak minute
   outside the span is `unknown_calendar` until the next year's holidays are recorded. The
   off-peak price applies only when DeepSeek itself served; a host with a flat rate bills its flat rate.

4. **The rate set** for (class, tier) must exist on the card, or the row is `no_rate`. Examples:
   fast mode on Opus 4.7; fast mode above 272K on gpt-5.5, which OpenAI does not publish; 1-hour
   writes on a card that sells none.

5. **Data residency.** `inference_geo: us` multiplies every token rate by the card's `geo_us`
   (Anthropic 1.1× on Claude 4.6+). Per-call fees are not multiplied. A card without it is
   `unknown_geo`.

6. **Server tools.** Each kind's count × the card's per-call fee. A kind the card has no fee for is
   `no_tool_fee`. A `0` fee is a published "no charge" (Anthropic web fetch).

### Reported costs (OpenRouter, xAI)

Where the vendor reports its own cost (`upstream_cost_usd`), that is the cost, parsed exactly from
its decimal text (exponent forms included, rounded half-up below one attodollar). It already
carries the host's price, the tier, any regional premium and tool fees. OpenRouter's is then
× 1.055: the 5.5% is its fee on the credits that paid for it (Standard plan; Business is 8%). xAI's
is taken as reported. The price is the same figure (pass-through).

On OpenRouter with none (a stream cut before its final usage chunk), the cost is the dearest endpoint that could
have served the row, × 1.055. "Could have served" means listed for the served class and, if the row
names a host, that host's endpoints, regional variants included. The result is an upper bound,
and the row is `estimated`. Each endpoint's long-context override applies. A time-of-day schedule
(DeepSeek's hosts) is folded to its dearest window.

### Cache hits

A response-cache hit makes no vendor call. Cost is 0, and the customer is charged what the hit cost
us, so the price is 0 too (owner decision, 2026-10-10). This is a rule, not a setting. The row keeps
`cache_hit=true` and the stored token facts for audit.

### Rounding

The pricer works in exact integers: an attodollar numerator times at most two multipliers in basis
points, in 128 bits. Each side (cost, price) is rounded **once, half-up, to the micro-dollar, per
row**. There is no rounding inside a row, no floating point, and no systematic rounding down. Half-up
on a non-negative amount is the same as round-half-away-from-zero. Rates are decimal strings with at
most six places per million tokens, so every published rate is exact. An overflow is `overflow`,
never a wrapped number.

### Rate data and versions

The table is generated, never typed. `rates/generated.rs` (every card, every OpenRouter endpoint
list, `ROW_RATES` and `OPENROUTER_CREDIT_FEE`) is written by `crates/rates-sync` from two inputs:

1. **Snapshots of the primary sources**, `verify/rates_sources/`. `manifest.toml` records each
   file's URL, the date its content was fetched, and the sha256 of the raw response and of the
   snapshot. A snapshot is the response normalized: churn that is not a price is dropped
   (OpenRouter's uptime and latency, xAI's build fingerprints, an HTML page's navigation), and
   order that carries no meaning is sorted.
2. **The hand-entered rules**, `verify/catalog_truth.toml`: which source row each card is, the
   facts a vendor states only in prose, per-call tool fees, recorded source conflicts and
   overrides, and which card each row's candidates bill from (`[[pricing]]`). Every prose fact and
   fee carries its URL and a verbatim quote, and generation fails unless the quote is in that
   URL's snapshot.

| Vendor     | Source the numbers come from (chosen)                                              | Cross-checked against (generation fails on disagreement)                                     | Hand-entered, with a checked quote                                                                                     |
| ---------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Anthropic  | `pricing.md`: model table (input, 5m and 1h writes, hits, output), fast mode table | Bedrock's Global SKUs (must equal the list); the organization's billed costs (`rates:audit`) | fast mode's cache rates ride the model's own multipliers; `inference_geo` 1.1× (4.6+); web search $10/1K, web fetch $0 |
| Bedrock    | AWS Price List offer `AmazonBedrockFoundationModels` (us-east-1): Regional SKUs    | Global SKUs = Anthropic's list; Regional = Global × 1.1 (Anthropic's pricing page)           | the 10% regional premium                                                                                               |
| OpenAI     | `pricing.md`: Standard, Flex, Fast, Ultrafast tables; specialized table            | the organization's billed costs per line item (`rates:audit`)                                | long context `> 272K`; tool fees                                                                                       |
| xAI        | `GET /v1/language-models` (1e-10 USD ticks per token; long tier and threshold)     | `pricing.md` text table, row for row                                                         | `>=` (reaches) at the threshold; Priority 2×; tool fees                                                                |
| DeepSeek   | the pricing page's HTML table (peak and off-peak, both read)                       | none published                                                                               | peak windows; the holiday calendar (gov.cn, quoted)                                                                    |
| Groq       | `models.md` price column; `prompt-caching.md` supported-model table                | none published                                                                               | cached input 50%; Flex priced as on-demand                                                                             |
| Fireworks  | `serverless/pricing.md` table (Standard and Priority)                              | none published                                                                               | the Priority column is the `fast` class                                                                                |
| Together   | `GET /v1/models` `pricing`                                                         | the docs' chat-models table (`serverless/models.md`)                                         | `[[conflict]]` per disagreement; the Kimi K3 promotion `[[override]]`                                                  |
| OpenRouter | `GET /api/v1/models/{slug}/endpoints` (×10^6; overrides folded)                    | the slug is in `GET /api/v1/models`                                                          | the 5.5% credit fee (its pricing page's FAQ, quoted)                                                                   |

Every reader is strict: it finds its table by exact header and label and parses a cell only when a
card asks for its row, so an unrelated row never breaks generation, while a renamed column, a
missing or duplicated row, or a malformed cell on a priced row always does. It never guesses. An
unpublished cache read or write is the input rate, by rule. A rate finer than six decimal places
per million fails.

Maintenance is one command:

- `mise run rates:sync` re-fetches every source, rewrites the changed snapshots, regenerates the
  table, prints the diff, and sets `RATE_VERSION` (`rates-sync rate-version`: `{today}.{hash}` of
  the table a fresh build links, also written to `verify/pricing_vectors.json`'s `rate_version`;
  unchanged when the hash already matches). Review the diff, and recompute any golden vector
  whose rates moved (`golden_vectors_price_exactly` names it).
- `mise run rates:generate` regenerates from the committed snapshots, offline.
  `rates_generated_from_sources` (`crates/rates-sync/tests/regenerate.rs`, per PR) does the same
  and diffs it with the committed file, so the table can only be what the snapshots and rules say.
- `.github/workflows/rates-drift.yml` (daily and on dispatch, never per PR) re-fetches into a
  scratch copy and classifies the result (`rates-sync classify`, `crates/rates-sync/src/classify.rs`;
  `mise run rates:classify` runs it locally, writing nothing):

  | Class   | When                                                                                                                                                                        | The workflow                                                                                   |
  | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
  | none    | the generated table is unchanged (a snapshot may have changed outside any priced cell)                                                                                      | nothing                                                                                        |
  | routine | only rate values (per-token rates, per-call fees) moved, each by at most 2× either way and none to or from zero; no card, endpoint, row, tier, or tool fee added or removed | regenerates, runs `rate-version`, opens a `rates/sync-YYYY-MM-DD` PR labeled `rates-routine`   |
  | review  | the new snapshots generate a table, but a routine rule fails; any non-rate value (a host, a threshold, a premium, a schedule, the credit fee) changing is one               | the same, labeled `rates-review`, the failed rules listed first                                |
  | broken  | a fetch, a reader (a layout change), a quote, a cross-check, or a stale `[[override]]`/`[[conflict]]` fails                                                                 | regenerates nothing; updates the open `rates-broken` issue for that kind (or opens one); fails |

  Every reader, cross-check, quote and override check runs inside generation, so a table that
  generates has passed them all. The PR body tables every changed rate (old, new, source URL). If
  an open `rates/sync-*` PR exists, the workflow force-pushes a fresh commit to it (only when its
  tip is the bot's own commit; a human's commit gets a comment instead), and does nothing when it
  already carries the same snapshots and table. A human merges every rates PR. Everything runs
  on `GITHUB_TOKEN`, whose pushes start no `pull_request` run, so after each push the workflow
  dispatches `ci.yml` on the branch (`workflow_dispatch` is the one event such a token can start),
  and the PR's head commit gets `Check`. A rates PR whose moved rate a golden vector prices fails
  CI until that vector is recomputed.
- `mise run rates:audit` prints LiteLLM's and models.dev's disagreements (leads to check against
  the primary source, never corrections), and checks the cards against the vendors' own invoices:
  every (day, model, token kind) line of OpenAI's and Anthropic's admin cost reports must equal
  tokens × rate, except that a line from before a vendor's announced price change is checked at
  the old rate (`[[superseded]]`). The reports are the organization's spend, so they are read live
  and never committed.

`crates/providers/tests/rates_truth.rs` checks the compiled table against the catalog: every row
and candidate provider is priced, every list card's standard tier is the `ListPrice`, every card is
well formed, and `RATE_VERSION` (`{date}.{hash of the table}`) changes whenever any rate does. The
catalog's `ListPrice` (and its `[[row]]` in the truth file) is the one hand copy of a list rate
left, and that test fails the moment it differs from the generated list card.

Every priced row should log `RATE_VERSION`. Repricing a historical row means running its token
facts through a table version of your choice: the facts are kept, and the version says which table
produced the logged amounts.

### Golden vectors: how to consume them

`verify/pricing_vectors.json`:

```json
{ "rate_version": "2026-10-10.…", "vectors": [ { "name": "…", "why": "…", "row": { … }, "expect": { … } } ] }
```

- `row` uses the field names above. An absent field takes the default in the Input table.
  `server_tools` is `{ kind: count }` with the kinds in Server tools below.
- `expect` is either `{ "unpriced": code }` or
  `{ status, basis, cost_micros, price_micros, cost_class, price_class, cost_long, price_long }`.
- An implementation passes when every vector matches exactly, against the rate table named by
  `rate_version`. A new table means new vectors. `crates/providers/tests/pricing_vectors.rs` is
  the reference consumer.

The vectors were computed by a second, independent implementation in exact rationals, not by this
crate, and this crate's test checks every one. They cover every refusal code, every class, every
cost basis, both wires, the tier boundaries exactly at and one past the threshold (OpenAI `>`, xAI
`>=`), the off-peak window edges, cache reads plus 5-minute plus 1-hour writes plus tools plus fast
mode plus data residency together, estimated rows, cache hits, and rounding at exactly half a
micro-dollar.

### Server tools

The row's `server_tools` kinds, and each card's fee per call (or per item). "—" means no fee, so
the row is refused. `tool_calls` (OpenRouter, every kind together) is never priced on its own.
With a reported cost, OpenRouter's tools and plugins (`upstream_tool_cost_usd`) are inside it and
pass through. Without one, a row whose `tool_calls` exceed its `web_search` is refused: a tool ran
whose fee no endpoint lists.

| Kind                 | Anthropic                 | OpenAI                                                                | xAI                           | OpenRouter endpoint              |
| -------------------- | ------------------------- | --------------------------------------------------------------------- | ----------------------------- | -------------------------------- |
| `web_search`         | $0.01                     | $0.01; refused on gpt-4o-mini and gpt-4.1-mini (fixed 8K-token block) | $0.005                        | the endpoint's ($0.01 or $0.005) |
| `web_search_preview` | —                         | $0.01 reasoning models, $0.025 non-reasoning                          | —                             | —                                |
| `web_search_page`    | —                         | $0                                                                    | —                             | —                                |
| `web_fetch`          | $0                        | —                                                                     | —                             | —                                |
| `code_execution`     | refused (container-hours) | refused (container session, per minute by size)                       | $0.005                        | —                                |
| `file_search`        | —                         | $0.0025                                                               | $0.0025                       | —                                |
| `computer_use`       | —                         | $0                                                                    | —                             | —                                |
| `mcp`                | —                         | $0                                                                    | $0                            | —                                |
| `tool_search`        | —                         | $0                                                                    | —                             | —                                |
| `shell`              | —                         | refused (container session)                                           | —                             | —                                |
| `image_generation`   | —                         | refused (image tokens unreported)                                     | refused                       | —                                |
| `x_search`           | —                         | —                                                                     | $0 (items billed)             | —                                |
| `x_posts`            | —                         | —                                                                     | $0.005 / post                 | —                                |
| `x_users`            | —                         | —                                                                     | $0.01 / profile               | —                                |
| `document_search`    | —                         | —                                                                     | refused (no fee by that name) | —                                |
| `sources`            | —                         | —                                                                     | refused (legacy Live Search)  | —                                |

OpenAI's $0 kinds follow its pricing page: fee-bearing tools are the Tools table, and "tokens used
for built-in tools are billed at the chosen model's per-token rates". A `web_search_call` made with
`web_search_preview` is counted apart on the row, so each tool gets its own fee. gpt-4o-mini and
gpt-4.1-mini add a fixed 8,000-token content block per `web_search` call, which the usage may not
show, so no `web_search` fee is held there (and the gateway refuses the request, D268).
So no fee is held there.

### Provider × dimension matrix

Every billable dimension found in each provider's primary pricing pages and docs, read on
2026-10-10. Each row says how the pricer handles it and how the row learns it applied. "Held"
means the rate is in the generated table. Where the row cannot observe a dimension, the
**Prevent/bound** column says how it is kept from costing money unseen.

#### Anthropic (direct): https://platform.claude.com/docs/en/about-claude/pricing.md

| Dimension           | Rate                                                                                                    | Observed via                                                       | Row field                                    | Pricer / prevent                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Input / output      | per model (= `ListPrice`)                                                                               | `usage`                                                            | `input_tokens`, `output_tokens`              | Held                                                                                              |
| Cache read          | 0.1× input; 0.025× Fable 5.1; 0.05× Opus 5.5 and Sonnet 5.5                                             | `usage.cache_read_input_tokens`                                    | `cache_read_tokens`                          | Held. Sonnet 5.5: $0.10 since 2026-10-07; invoices before then bill $0.20                         |
| 5-minute write      | 1.25× input                                                                                             | `usage.cache_creation.ephemeral_5m_input_tokens`                   | `cache_write_tokens − cache_write_1h_tokens` | Held                                                                                              |
| 1-hour write        | 2× input                                                                                                | `usage.cache_creation.ephemeral_1h_input_tokens`                   | `cache_write_1h_tokens`                      | Held                                                                                              |
| Fast mode           | 2× every rate, caches included; Opus 4.8, Opus 5, Opus 5.5 only (fast-mode.md)                          | `usage.speed`                                                      | `speed`                                      | Held; refused on other models. Today unreachable: the gateway's beta filter drops `fast-mode-*`   |
| US-only inference   | 1.1× every token category; Claude 4.6+ (data-residency.md)                                              | `usage.inference_geo`                                              | `inference_geo`                              | Held; refused on 4.5 models (the API 400s); a reported `not_available` (4.5 models) is no premium |
| Long context        | none: 4.6+ have 1M at standard rates; 4.5 models have a 200K window                                     | —                                                                  | —                                            | No tier on any Claude row                                                                         |
| Priority Tier       | contract-priced; no longer sold (api/service-tiers.md)                                                  | `usage.service_tier: priority`                                     | `service_tier`                               | Refused (`unknown_class`); we hold no commitment                                                  |
| Web search          | $10 / 1,000                                                                                             | `usage.server_tool_use.web_search_requests`                        | `server_tools.web_search`                    | Held                                                                                              |
| Web fetch           | $0 beyond tokens                                                                                        | `usage.server_tool_use.web_fetch_requests`                         | `server_tools.web_fetch`                     | Held at $0                                                                                        |
| Code execution      | $0.05 per container-hour after 1,550 free hours per org per month; free with web search/fetch 20260209+ | `usage.server_tool_use.code_execution_requests` (calls, not hours) | `server_tools.code_execution`                | Refused. Unreachable today: its beta header is filtered                                           |
| Advisor tool        | the advisor model's rates, **outside** top-level `usage`                                                | `usage.iterations[]`                                               | —                                            | Must stay blocked (beta filter): top-level usage would under-bill it                              |
| MCP connector       | no fee stated (tokens only; unverified)                                                                 | content blocks                                                     | —                                            | Blocked by the beta filter                                                                        |
| Files API           | free; file content billed as input                                                                      | `usage.input_tokens`                                               | `input_tokens`                               | Managed `/files` is refused                                                                       |
| Citations           | `cited_text` not billed; slight input increase                                                          | `usage`                                                            | `input_tokens`                               | Nothing extra                                                                                     |
| Thinking            | output rate; `output_tokens` includes the full thinking                                                 | `usage.output_tokens`                                              | `output_tokens`                              | Never priced on its own                                                                           |
| Images / PDFs       | input tokens                                                                                            | `usage.input_tokens`                                               | `input_tokens`                               | Nothing extra                                                                                     |
| Batch               | 50%                                                                                                     | —                                                                  | —                                            | Not applicable: managed batches are refused                                                       |
| Per-request minimum | none                                                                                                    | —                                                                  | —                                            | —                                                                                                 |

#### Amazon Bedrock: AWS Price List `AmazonBedrockFoundationModels` (us-east-1), https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock.md

| Dimension                                       | Rate                                                                      | Observed via                                  | Row field                 | Pricer / prevent                                                        |
| ----------------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------- | ------------------------- | ----------------------------------------------------------------------- |
| `us.` geo profile                               | the Regional SKUs: Global + 10% on every category, 1-hour writes included | the request's model id (static per candidate) | `price_variant: regional` | Held: Haiku 4.5 1.1/5.5/0.11/1.375/2.2; Opus 4.8 5.5/27.5/0.55/6.875/11 |
| `global.` / `eu.` profiles                      | Global = list; geo + 10%                                                  | request model id                              | `price_variant`           | Not candidates: refused (`unknown_variant`)                             |
| Latency-optimized                               | Claude 3.5 Haiku only                                                     | —                                             | —                         | Not applicable                                                          |
| Fast mode, `inference_geo`, server tools, batch | not offered on the Messages endpoint                                      | —                                             | —                         | A row claiming one is refused                                           |

#### OpenAI: https://developers.openai.com/api/docs/pricing.md

| Dimension                 | Rate                                                                                | Observed via                                                                             | Row field                                 | Pricer / prevent                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------- |
| Input / cached / output   | per model; "-" cached = no discount                                                 | `usage`                                                                                  | tokens                                    | Held                                                                                    |
| Cache writes              | 1.25× input on GPT-5.6 and later, a third input rate inside `input_tokens`          | `input_tokens_details.cache_write_tokens`                                                | `cache_write_tokens`                      | Held                                                                                    |
| Long context              | > 272K input: 2× input and cache, 1.5× output, whole request (gpt-5.4, 5.5, 5.6, 6) | derived from tokens                                                                      | tokens                                    | Held                                                                                    |
| Fast mode (Priority)      | the Fast table, per model (2×; 2.5× gpt-5.5; 1.67–1.8× older)                       | `service_tier: priority` (or `fast`); a downgrade says `default`                         | `service_tier`                            | Held. Fast above 272K held for 5.6/6; unpublished for 5.4/5.5, so refused               |
| Ultrafast                 | 6× (gpt-6-astra, gpt-6.1-sol)                                                       | `service_tier: ultrafast`                                                                | `service_tier`                            | Held                                                                                    |
| Flex                      | the Flex table (= Batch rates)                                                      | `service_tier: flex`                                                                     | `service_tier`                            | Held                                                                                    |
| Scale tier                | contract                                                                            | `service_tier: scale`                                                                    | `service_tier`                            | Refused                                                                                 |
| Regional processing       | +10% (models from 2026-03-05)                                                       | the host (`us.api.openai.com`); no response field                                        | —                                         | Never applies: the pool key goes to `api.openai.com`                                    |
| Web search                | $10 / 1k search actions + content tokens at model rates                             | `web_search_call` output items (search actions)                                          | `server_tools.web_search`                 | Held; refused on gpt-4o-mini / gpt-4.1-mini (the gateway refuses the request too, D268) |
| Web search preview        | $10 / 1k reasoning models; $25 / 1k non-reasoning                                   | `web_search_call` from a request offering `web_search_preview` (the row counts it apart) | `server_tools` `web_search_preview`       | Held                                                                                    |
| File search               | $2.50 / 1k calls; storage $0.10 / GB-day                                            | `file_search_call`                                                                       | `server_tools.file_search`                | Calls held; storage cannot accrue (managed vector stores are refused)                   |
| Containers                | $0.03–$1.92 per 20-minute session by memory, billed per minute, 5-minute minimum    | `code_interpreter_call` / `shell_call` (no minutes, no size)                             | `server_tools` `code_execution` / `shell` | Refused. Gateway: strip or refuse the tool on managed traffic                           |
| Image generation tool     | image-token rates, not in `usage`                                                   | `image_generation_call` (no usage)                                                       | `server_tools.image_generation`           | Refused. Gateway: strip or refuse the tool on managed traffic                           |
| Image input               | input tokens (per-model multipliers already in the count)                           | `usage.input_tokens`                                                                     | `input_tokens`                            | Nothing extra                                                                           |
| Reasoning                 | inside output                                                                       | `output_tokens_details.reasoning_tokens`                                                 | `output_tokens`                           | Never priced on its own                                                                 |
| Pro mode (GPT-5.6, GPT-6) | the model's standard rates                                                          | —                                                                                        | —                                         | The seven `*-pro` rows (OpenRouter-only) are priced at the base model's card            |
| Promotion                 | gpt-5.6-sol's current $4 / $20 is promotional "at least through November 21, 2026"  | —                                                                                        | —                                         | Re-check that day                                                                       |

#### xAI: https://docs.x.ai/developers/pricing.md

| Dimension                                     | Rate                                                                          | Observed via                         | Row field                        | Pricer / prevent                                 |
| --------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------ | -------------------------------- | ------------------------------------------------ |
| Input / cached / output                       | per model; no cache-write price                                               | `usage`                              | tokens                           | Held                                             |
| Long context                                  | prompt ≥ 200K (cached included): 2× every rate, whole request                 | derived                              | tokens                           | Held (`>=`; see Known gaps)                      |
| Priority                                      | 2× every token type, after caching; billed only when the response confirms it | `service_tier: priority`             | `service_tier`                   | Held short; priority + long unpublished, refused |
| US endpoint                                   | 1.1× (grok-4.6, 4.7)                                                          | the host (`us.api.x.ai`)             | —                                | Never applies: the gateway uses `api.x.ai`       |
| Web search, code execution, attachment search | $5 / 1k                                                                       | `server_side_tool_usage_details`     | `server_tools.*`                 | Held                                             |
| X search                                      | $5 / 1k posts, $10 / 1k profiles, per item returned; the call itself free     | `x_posts_fetched`, `x_users_fetched` | `x_posts`, `x_users`, `x_search` | Held                                             |
| Collections / file search                     | $2.50 / 1k                                                                    | `file_search_calls`                  | `server_tools.file_search`       | Held                                             |
| Image generation                              | $0.02–0.05 per image by model                                                 | `image_generation_calls`             | `server_tools.image_generation`  | Refused                                          |
| Usage-guideline violation                     | $0.05 per request                                                             | an error response                    | —                                | Unobserved; bounded at $0.05 a request           |
| Reported cost                                 | `cost_in_usd_ticks` (1e-10 USD)                                               | `usage`                              | `upstream_cost_usd`              | The cost of goods when present                   |

#### DeepSeek: https://api-docs.deepseek.com/quick_start/pricing

| Dimension        | Rate                                                                                        | Observed via | Row field   | Pricer / prevent                  |
| ---------------- | ------------------------------------------------------------------------------------------- | ------------ | ----------- | --------------------------------- |
| Peak / off-peak  | card = peak; off-peak half; peak 01–04 and 06–10 UTC Mon–Fri except Chinese public holidays | request time | `unix_secs` | Held, with a covered holiday span |
| Cache hit / miss | per model; no write price                                                                   | `usage`      | tokens      | Held                              |

#### Together: https://www.together.ai/pricing, https://docs.together.ai/docs/serverless/models

| Dimension               | Rate                                                     | Observed via | Row field | Pricer / prevent                                                                         |
| ----------------------- | -------------------------------------------------------- | ------------ | --------- | ---------------------------------------------------------------------------------------- |
| Input / cached / output | per model; no cached price = no discount; no write price | `usage`      | tokens    | Held, from `/v1/models`; a disagreement with the docs table is a recorded `[[conflict]]` |
| Long context, tiers     | none published                                           | —            | —         | —                                                                                        |
| Images                  | input tokens (1,601 per 560px tile)                      | `usage`      | tokens    | Nothing extra                                                                            |

#### Fireworks: https://docs.fireworks.ai/serverless/pricing

| Dimension | Rate                        | Observed via             | Row field      | Pricer / prevent                              |
| --------- | --------------------------- | ------------------------ | -------------- | --------------------------------------------- |
| Tokens    | per model, cached per model | `usage`                  | tokens         | Held                                          |
| Priority  | per model (1.2–1.5×)        | `service_tier: priority` | `service_tier` | Held: priced at Fireworks' own priority rates |

#### Groq: https://console.groq.com/docs/models

| Dimension                        | Rate                                       | Observed via         | Row field      | Pricer / prevent |
| -------------------------------- | ------------------------------------------ | -------------------- | -------------- | ---------------- |
| Tokens                           | per model; cached 50% on gpt-oss only      | `usage`              | tokens         | Held             |
| Flex                             | = on-demand                                | `service_tier: flex` | `service_tier` | Held             |
| Performance tier                 | provisioned, contract                      | —                    | —              | Not requested    |
| browser_search, code_interpreter | unpublished                                | `executed_tools`     | `server_tools` | Refused (no fee) |
| Images                           | 2,048 input tokens per image (qwen3.8-27b) | `usage`              | tokens         | Nothing extra    |

#### OpenRouter: https://openrouter.ai/pricing, https://openrouter.ai/docs/faq, `GET /api/v1/models/{slug}/endpoints`

| Dimension                 | Rate                                                                                      | Observed via                                                         | Row field                                         | Pricer / prevent                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Credit fee                | 5.5% of purchases (Standard), 8% (Business)                                               | —                                                                    | —                                                 | × 1.055 on every OpenRouter cost                                                           |
| Inference                 | the host's list price, which varies by host (up to ~10× on open models)                   | `usage.cost` (always included now)                                   | `upstream_cost_usd`                               | Reported cost used; otherwise the dearest endpoint                                         |
| Regional endpoints        | Anthropic, OpenAI, xAI hosts 1.1×; others vary; only on a regional domain or explicit pin | in `usage.cost`; generation `data_region`                            | —                                                 | In the reported cost; in the fallback's dearest endpoint                                   |
| Service tiers             | each tier its own endpoint (`/fast` 2×, `/flex` 0.5×, `/ultrafast` 6×)                    | `service_tier` (fast reported as `priority`; Messages `usage.speed`) | `service_tier`, `speed`                           | Customer card class; fallback filtered by class                                            |
| Long context, time of day | per-endpoint overrides (`min_prompt_tokens`, `utc_*`)                                     | derived / in `usage.cost`                                            | tokens                                            | Held per endpoint (time of day at its dearest window)                                      |
| Native web search         | pass-through ($0.01 or $0.005)                                                            | `server_tool_use_details.web_search_requests`                        | `server_tools.web_search`                         | Held per endpoint                                                                          |
| Web plugin / `:online`    | Exa $0.007–0.015 per request + $0.001 per result over 10                                  | `num_search_results` (generation API)                                | `upstream_tool_cost_usd`                          | A nonzero `upstream_tool_cost_usd` is refused. Gateway: strip `plugins` on managed traffic |
| `:nitro` / `:floor`       | admits priority / flex endpoints                                                          | `service_tier`                                                       | `service_tier`                                    | Priced by the served class                                                                 |
| BYOK                      | 5% over $25k/month                                                                        | `usage.is_byok`                                                      | —                                                 | Not used                                                                                   |
| Cancelled stream          | hosts that cannot cancel bill the whole completion                                        | `GET /api/v1/generation?id=` (`total_cost`, `cancelled`)             | `upstream_generation_id`, `upstream_may_continue` | `estimated`: the row's cost is a lower bound; reconcile through the generation API         |

#### Hosts the catalog reaches only through OpenRouter

Google (Vertex), Azure, Alibaba, Moonshot, Z.ai, and the open-model hosts are priced as
OpenRouter endpoints. Their makers' own cards, where a row's list card follows one, match the
catalog: Moonshot Kimi K2.6/K2.7 Code, Z.ai GLM 5.1–5.3, Alibaba Qwen. Alibaba tiers qwen3.6-plus
and qwen3.7-plus at 256K. Those rows' list card is Together's flat rate (the primary), and
OpenRouter's Alibaba endpoint carries the tier on the cost side.

### Known gaps

- **xAI at exactly 200,000 prompt tokens**: the pricing table says "reaches" (≥, which we use).
  The caching page says "exceed". At most one request in 200K tokens straddles this.
- **OpenRouter `usage.cost` and plugin fees**: "the total amount charged" is read as including
  them. Unverified until the live reconciliation compares it with the generation API.
- **Together Kimi K3** is held at list ($3 / $15 / cached $0.30) by an `[[override]]`: Together's
  API and pricing page show a promotion ($2.70 / $13.50 / $0.27) that ends 2026-10-11. The first
  sync after it ends fails until the override is removed.
- **Together retires Llama 3.3 70B Turbo on 2026-10-22.** Its row needs a decision before then.
- **DeepSeek's holiday calendar** must be extended before 2027-01-01, or weekday peak rows refuse.
- **The seven OpenAI `*-pro` ids for GPT-5.6 and GPT-6 are not OpenAI models.** OpenAI makes "pro" a
  `reasoning.mode`. The rows exist only on OpenRouter, priced at the base model's card.
