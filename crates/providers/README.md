# beyond-ai-providers

The facts about LLM providers that the gateway and the agent must agree on,
as static tables and the lookups over them. No I/O, no dependencies.

| Table         | Answers                                                                              | Module             |
| ------------- | ------------------------------------------------------------------------------------ | ------------------ |
| **Providers** | Which upstream does this name, host or model id mean, and how do I talk to it?       | `lib.rs`           |
| **Catalog**   | Which providers serve this model, in what order, and what is its list price?         | `catalog`          |
| **Pricing**   | What does one `ai.usage` row cost us and what do we charge for it, in micro-dollars? | `pricing`, `rates` |

The gateway routes `/{provider}/…` and swaps keys from the provider rows. The
agent, with no gateway configured, dials `base_url` directly and sends the
user's key in the header from the same row. One table keeps a provider's auth
scheme from being right in one and wrong in the other.

```toml
[dependencies]
providers = { package = "beyond-ai-providers", path = "crates/providers" }
```

## Providers

`providers::PROVIDERS` has 17 rows. The first 12 are gateway routes
(`/{name}/…`); the last five are reachable only through `AI_PROVIDER` or a
`models.json` `base_url` in the agent.

| Name                                         | Direct `base_url`                                           | Wire                      | Auth header     | Key variable               |
| -------------------------------------------- | ----------------------------------------------------------- | ------------------------- | --------------- | -------------------------- |
| `openai`                                     | `https://api.openai.com/v1`                                 | OpenAI                    | `Authorization` | `OPENAI_API_KEY`           |
| `anthropic`                                  | `https://api.anthropic.com`                                 | Anthropic                 | `x-api-key`     | `ANTHROPIC_API_KEY`        |
| `openrouter`                                 | `https://openrouter.ai/api/v1`                              | OpenAI                    | `Authorization` | `OPENROUTER_API_KEY`       |
| `fireworks`                                  | `https://api.fireworks.ai/inference/v1`                     | OpenAI                    | `Authorization` | `FIREWORKS_API_KEY`        |
| `groq`                                       | `https://api.groq.com/openai/v1`                            | OpenAI                    | `Authorization` | `GROQ_API_KEY`             |
| `deepseek`                                   | `https://api.deepseek.com/v1`                               | OpenAI                    | `Authorization` | `DEEPSEEK_API_KEY`         |
| `together`                                   | `https://api.together.ai/v1`                                | OpenAI                    | `Authorization` | `TOGETHER_API_KEY`         |
| `cerebras`                                   | `https://api.cerebras.ai/v1`                                | OpenAI                    | `Authorization` | `CEREBRAS_API_KEY`         |
| `mistral`                                    | `https://api.mistral.ai/v1`                                 | OpenAI                    | `Authorization` | `MISTRAL_API_KEY`          |
| `xai`                                        | `https://api.x.ai/v1`                                       | OpenAI                    | `Authorization` | `XAI_API_KEY`              |
| `bedrock`                                    | `https://bedrock-runtime.us-east-1.amazonaws.com/anthropic` | Anthropic                 | `x-api-key`     | `AWS_BEARER_TOKEN_BEDROCK` |
| `openai-codex`                               | none (OAuth bearer only)                                    | OpenAI                    | `Authorization` | none                       |
| `huggingface`                                | `https://router.huggingface.co/v1`                          | OpenAI                    | `Authorization` | `HF_TOKEN`                 |
| `nvidia`                                     | `https://integrate.api.nvidia.com/v1`                       | OpenAI                    | `Authorization` | `NVIDIA_API_KEY`           |
| `kimi-coding`, `opencode-go`, `opencode-zen` | none (account-templated)                                    | Anthropic, OpenAI, OpenAI | `Authorization` | none                       |

`for_model_id("claude-opus-4-8")` returns the provider that natively serves an
id by prefix. It returns `None` for an aggregator, because
`moonshotai/kimi-k2.6` is equally a Fireworks, Together and OpenRouter id;
callers name an aggregator. The provider decides the wire, not the model id:
`anthropic/claude-sonnet-4.5` is an OpenAI-wire request on OpenRouter.

## The catalog

`catalog::MODEL_ROUTES` has 90 rows. A `ModelRoute` holds the canonical model
name, the wire its clients speak, the ordered `Candidate`s (provider, that
provider's spelling of the model id, and the upstream path), and a `ListPrice`
in USD per million tokens. On the gateway the catalog is the allowlist for
managed `/v1` and `/auto`: a name that is not a row is a 404. `for_model`
resolves a name, a candidate's own spelling, or a retired alias.

## Pricing

`pricing::price` turns one `ai.usage` row into integer micro-dollars. Pricing
is pass-through: price equals cost, taken from the card of the host that served
the request, under that host's rules (long-context tier, fast mode, data
residency, off-peak hours, server-tool fees). A row the tables cannot price
exactly returns an `Unpriced` error, never zero.

```rust
use providers::pricing::{UsageRow, price};
use providers::{WireFormat, for_model};

let route = for_model("claude-opus-4-8").unwrap();
assert_eq!(route.price.input, "5");   // USD per million tokens

let row = UsageRow {
    price_model: Some("claude-opus-4-8"),
    provider: Some("anthropic"),
    usage_wire: WireFormat::Anthropic,
    input_tokens: 1000,
    output_tokens: 500,
    ..Default::default()
};
let p = price(&row).unwrap();
assert_eq!((p.cost.micros, p.price.micros), (17_500, 17_500));
```

`verify/pricing_vectors.json` holds golden rows with their expected results.
It is the contract for any other implementation (a repricer, an invoice
audit): reproduce it exactly. An independent reference pricer in Python
(`verify/pricing_reference.py`) writes the vectors, and a test holds this
crate to them.

The rate table (`rates/generated.rs`) is generated from snapshots of each
vendor's pricing source in `verify/rates_sources/`; it is never edited by hand.

```sh
mise run rates:sync        # re-fetch every source, regenerate the table, update vectors
mise run rates:generate    # regenerate offline from the committed snapshots
mise run pricing:vectors   # recompute the golden vectors' expected results
```

## Reference

[`ARCHITECTURE.md`](ARCHITECTURE.md) states every pricing rule: token
normalization per wire, rate selection, reported costs, rounding, server-tool
fees, each vendor's source, and how the catalog stays current.
