//! The model catalog — canonical model name → the ordered upstreams that serve it.
//!
//! This is the inverse of [`crate::for_model_id`], and the general form of it. `for_model_id` answers
//! "which provider serves this id *shape*, natively, when nobody said otherwise" and deliberately
//! returns nothing for an aggregator, because `moonshotai/kimi-k2.6` is equally a Fireworks,
//! Together and OpenRouter id and guessing between them is the mis-route this crate exists to
//! prevent. The catalog is where that guess becomes a *decision*: a named model, the wire its
//! clients speak, and the ordered upstreams we are willing to serve it from. On the gateway, that
//! table **is** the allowlist for managed `/v1` and `/auto` — a name that is not a row is a 404.
//! A candidate's `upstream_model` spelling is an alias for the same row. `/{provider}/…` does
//! not consult it.
//!
//! A row carries routing facts — provider, the id that provider spells it with, and the path to
//! send it to — plus one published list price ([`ListPrice`]). Model *capability* facts (context
//! window, thinking shape) stay in `agent_core::models`; a test keeps a [`Candidate`] from growing
//! them. The price is the public standard card, not the invoice: `ai.usage` still emits token
//! counts, and a downstream consumer applies (or replaces) this card.
//!
//! # Wire format belongs to the row, not the provider
//!
//! [`ProviderSpec::wire`](crate::ProviderSpec::wire) is a single value per provider, and that is an
//! approximation. OpenRouter is the clearest case: it serves the **OpenAI** wire at
//! `/api/v1/chat/completions` *and* the **Anthropic** wire at `/api/v1/messages`, and both are real
//! — the Anthropic one returns `message_start`/`message_delta` SSE with `input_tokens`,
//! `cache_read_input_tokens` and `output_tokens_details.thinking_tokens`, which is exactly what the
//! gateway's Anthropic usage extractor reads. (Fireworks is the same story from the other side: see
//! `agent_core::dialect::is_fireworks_anthropic_wire_model`.)
//!
//! So a row declares its own [`ModelRoute::wire`] (the *client default* / primary endpoint) and
//! each candidate carries the [`Candidate::path`] that serves it. Deriving the wire from the
//! provider would have been wrong in a specifically nasty way: an Anthropic-wire response parsed
//! by the OpenAI extractor trips the dialect-mismatch guard and emits a **zero-token billing row**,
//! not an error.
//!
//! Candidates in a row may disagree on the endpoint (Messages vs Chat Completions). The gateway
//! translates the original client body onto **this candidate's** path each attempt, so Claude can
//! fail onto an OpenAI-compat host without sending a Messages body at Chat Completions.
//! GPT rows also list a parallel [`ModelRoute::responses`] arm (OpenAI `/v1/responses`) that every
//! inbound Responses request walks, `store: false` one-shots included; Chat Completions / Messages
//! inbound walk [`ModelRoute::candidates`].
//! `/{provider}/…` never translates.
//!
//! # Maintenance
//!
//! These rows are product data and they go stale — providers rename ids, deprecate models, change
//! what they host, and reprice. Every id and path below was verified against the live API before
//! being added, and `catalog_rows_are_servable` (in `crates/gateway/tests/smoke.rs`) re-verifies the
//! whole table against real providers whenever the keys are present. Add a row the same way: check
//! it, then add it, **with a list price**. A wrong route does not fail loudly — it routes to a 404
//! that looks like the client's fault. A missing price used to fail the same way in the other
//! direction: `GET /v1/models` named the model and a consumer priced it at zero.
//!
//! A row whose primary is a direct vendor also needs an entry in `verify/catalog_truth.toml`: the
//! vendor's published price and card values with the URL they came from (or a reasoned entry in
//! its `unverified` list). `catalog_matches_vendor_truth` holds the table to that file, and a
//! retired or non-serverless id recorded there can never come back as a candidate.

use crate::{ProviderId, WireFormat};

/// One upstream that can serve a catalog model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub provider: ProviderId,
    /// The model id **as this provider spells it**. The gateway splices this into the request body's
    /// `model` field before forwarding, so it may differ from the row's canonical name and from
    /// every other candidate's — `claude-opus-4-8` at Anthropic is `anthropic/claude-opus-4.8` at
    /// OpenRouter, dots and all.
    pub upstream_model: &'static str,
    /// The full upstream path for this candidate.
    ///
    /// Absolute and complete, not a suffix to be composed: providers do not agree on where an
    /// endpoint lives, and the disagreement is not a simple prefix. Anthropic serves Messages at
    /// `/v1/messages` from a base URL carrying no path; OpenRouter may serve the same model at
    /// `/api/v1/chat/completions`. There is no client-supplied suffix that is correct for both, so
    /// the catalog states each one outright. Mixed-wire rows are translated per candidate.
    pub path: &'static str,
}

/// A canonical model name, the wire its clients speak, and the ordered upstreams that serve it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelRoute {
    /// The catalog name a client puts in `x-beyond-model` or, on a managed `/v1` (and headerless
    /// `/auto`) request, the body's root `model`. Lowercase and restricted to `[a-z0-9._/-]`,
    /// which is what lets the gateway log it verbatim without sanitizing.
    pub model: &'static str,
    /// The API shape the *primary* speaks, and the shape a bare `/v1` client is assumed to send.
    /// Failover candidates may speak a different endpoint; the gateway translates the original
    /// client body onto each candidate's path. Inbound `/v1/responses` on a row without a
    /// [`Self::responses`] arm is translated (a one-shot only) onto Chat Completions or Messages
    /// according to this field for the primary, then per candidate; a row with one walks it.
    /// `/{provider}/…` never translates.
    pub wire: WireFormat,
    /// Preference order: `[0]` is primary, the rest are failover candidates. Non-empty, at most
    /// [`MAX_CANDIDATES`], no provider repeated. Chat Completions / Messages inbound, and one-shot
    /// Responses on a row with no [`Self::responses`] arm, walk this list.
    pub candidates: &'static [Candidate],
    /// OpenAI `/v1/responses` arms. Walked by every inbound Responses request when non-empty,
    /// `store: false` one-shots included, so Responses-only fields and tools are never lost to a
    /// translation. Empty on Claude rows — those have no OpenAI store, so session-state Responses
    /// (`previous_response_id`, or an explicit `store: true`) is a 400, not a hollow Messages call.
    /// Same-endpoint: byte relay (`store` / `previous_response_id` / `include` / `truncation` /
    /// tools pass through). A Responses 5xx may walk another entry here; it never walks onto
    /// [`Self::candidates`].
    pub responses: &'static [Candidate],
    /// Standard public list price for this model. See [`ListPrice`].
    pub price: ListPrice,
    /// What `GET /v1/models` tells a client about the model. See [`ModelCard`].
    pub card: ModelCard,
}

/// The model facts `GET /v1/models` publishes beside the price, so a client can size a prompt,
/// cap its output and pick a model by what it accepts and supports.
///
/// Sourcing, checked 2026-10-01 against the primary vendor's docs (`verify/catalog_truth.toml`
/// records each checked value and its URL; a test holds this table to it):
///
/// - `context_window` and `max_output_tokens` are the limits **every** candidate of the row
///   enforces: the smallest, so a request sized from the card is accepted wherever the walk lands
///   (failover and TTFT ranking may serve any of them). They start from the **primary** vendor's
///   model docs and come down where a limit is lower in practice. OpenAI's GPT-5 family counts the
///   output inside the published window and caps input at the window less the 128K max output
///   (272,000 of 400,000; 922,000 of 1,050,000; the API says "Input tokens exceed the configured
///   limit of 272000 tokens"). A failover host that serves a smaller window (OpenRouter's
///   Ministral 3B, 131,072) sets the row's. `verify/catalog_truth.toml` records the vendor's figure
///   and, beside it, the lower `input_limit` / `output_limit` with its evidence. One limit per row
///   rather than one per candidate: the gateway does not count prompt tokens, so it could not
///   choose between candidates by window anyway, and a client only ever sees the row.
/// - Where the vendor publishes no max output, a figure that was a fraction of the window
///   (OpenRouter's 0.9x / 0.8x filler) is not kept: the row lists [`UNPUBLISHED_MAX_OUTPUT`]
///   instead. A max output that is not such a fraction is kept.
/// - `input` and `features` list what the vendor lists **and every candidate serves on the
///   endpoint the catalog sends it to**. A bit the vendor's model page does not name (structured
///   outputs on `gpt-4`) is removed; so is one a candidate's endpoint refuses (file input on grok
///   rows, which xAI serves on Responses only while these rows reach xAI over Chat Completions;
///   function calling on `grok-4.20-multi-agent`, which xAI gates behind beta access; tools on
///   `meta-llama/llama-4-scout`, which no OpenRouter host serves). One the page omits but every
///   candidate serves is listed: function calling on `gpt-4`, the snapshot that introduced it.
///   The gateway refuses image input on a row whose card omits it, rather than let a candidate
///   ignore the image (o3-mini) or answer 500 (gpt-4).
/// - `created` and `owned_by` are the vendor's own listing (OpenAI's and xAI's `/v1/models`),
///   never the day OpenRouter listed the model.
/// - Anything the vendor does not publish still comes from OpenRouter's public card for the row's
///   candidate (`context_length`, `top_provider.max_completion_tokens`,
///   `architecture.input_modalities`, `supported_parameters`), fetched 2026-09-30.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCard {
    /// Human-readable name, e.g. `Claude Sonnet 5.5`.
    pub name: &'static str,
    /// The vendor that made the model (`anthropic`, `openai`, `meta-llama`, …).
    pub owned_by: &'static str,
    /// Release time, Unix seconds.
    pub created: u64,
    /// Most input tokens one request may carry, on every candidate of the row.
    pub context_window: u32,
    /// Most output tokens one request may ask for, on every candidate of the row. Zero on an
    /// embeddings row, which generates none. Where the vendor publishes no max output this is
    /// [`UNPUBLISHED_MAX_OUTPUT`] and [`Self::max_output_published`] is false: a figure to size
    /// requests by, never a limit to enforce. The gateway caps a request's output limit at
    /// [`Self::output_cap`].
    pub max_output_tokens: u32,
    /// Whether [`Self::max_output_tokens`] is the vendor's published limit.
    pub max_output_published: bool,
    /// `IN_*` bits: what a request may contain.
    pub input: u8,
    /// Capability bits: [`TOOLS`], [`REASONING`], [`STRUCTURED_OUTPUTS`].
    pub features: u8,
}

impl ModelCard {
    /// The output limit the gateway may enforce: the vendor's published maximum, `None` where it
    /// publishes none (the card's [`UNPUBLISHED_MAX_OUTPUT`] is a placeholder, and a request past
    /// it may well be served) and on an embeddings row.
    pub const fn output_cap(&self) -> Option<u32> {
        if self.max_output_published && self.max_output_tokens > 0 {
            Some(self.max_output_tokens)
        } else {
            None
        }
    }
}

/// The max output a card lists when the primary vendor publishes none: a conservative figure, not
/// a vendor limit, so it is never enforced ([`ModelCard::output_cap`]). A client that sizes `max_tokens` from the card stays under every host's real
/// cap; one that asks for more may still be accepted. It replaced OpenRouter's derived
/// 0.9x / 0.8x-of-window values, which advertised outputs as large as the whole prompt budget.
pub const UNPUBLISHED_MAX_OUTPUT: u32 = 32_768;

pub const IN_TEXT: u8 = 1;
pub const IN_IMAGE: u8 = 1 << 1;
/// Documents (PDF).
pub const IN_FILE: u8 = 1 << 2;
pub const IN_AUDIO: u8 = 1 << 3;
pub const IN_VIDEO: u8 = 1 << 4;
/// Function calling.
pub const TOOLS: u8 = 1;
/// Thinking / reasoning tokens.
pub const REASONING: u8 = 1 << 1;
/// Output constrained to a JSON schema.
pub const STRUCTURED_OUTPUTS: u8 = 1 << 2;

/// The `/v1/models` names of the `IN_*` bits and the capability bits, in output order.
const INPUT_NAMES: [(u8, &str); 5] = [
    (IN_TEXT, "text"),
    (IN_IMAGE, "image"),
    (IN_FILE, "file"),
    (IN_AUDIO, "audio"),
    (IN_VIDEO, "video"),
];
const FEATURE_NAMES: [(u8, &str); 3] = [
    (TOOLS, "tools"),
    (REASONING, "reasoning"),
    (STRUCTURED_OUTPUTS, "structured_outputs"),
];

const fn card(
    name: &'static str,
    owned_by: &'static str,
    created: u64,
    context_window: u32,
    max_output_tokens: u32,
    input: u8,
    features: u8,
) -> ModelCard {
    ModelCard {
        name,
        owned_by,
        created,
        context_window,
        max_output_tokens,
        max_output_published: true,
        input,
        features,
    }
}

/// A card whose vendor publishes no max output: it lists [`UNPUBLISHED_MAX_OUTPUT`], unenforced.
const fn card_unpublished_output(
    name: &'static str,
    owned_by: &'static str,
    created: u64,
    context_window: u32,
    input: u8,
    features: u8,
) -> ModelCard {
    ModelCard {
        max_output_published: false,
        ..card(
            name,
            owned_by,
            created,
            context_window,
            UNPUBLISHED_MAX_OUTPUT,
            input,
            features,
        )
    }
}

/// Standard list price, USD per million tokens.
///
/// Decimal strings, not `f64`: `0.075` is not binary-exact, and these bytes are copied into
/// `GET /v1/models`. At most six digits after the point (one micro-dollar). The four rates are the
/// standard card only — not batch, not fast mode, not a long-context override, and not the 1-hour
/// Claude cache write (2× input). `cache_write` here is the 5-minute / default write rate.
///
/// The rate is the **primary candidate's** vendor standard published rate, checked 2026-10-01 and
/// recorded with its source URL in `verify/catalog_truth.toml` (a test holds this table to it).
/// Not batch or flex, and not OpenRouter's cheapest host. Where a vendor tiers the rate, the row
/// lists the standard tier and says so in a comment: DeepSeek's peak rate (off-peak is half),
/// xAI's < 200k-prompt tier, OpenAI's ≤ 272K-input tier. A promotional rate is not a list rate:
/// gpt-5.6-sol lists OpenAI's standard $4 / $20, which OpenAI bills the pool key, while OpenRouter
/// passes on OpenAI's announced half-price promotion (`[[promo]]` in the truth file). A row whose
/// primary is OpenRouter lists its maker's own published rate where there is one (Moonshot's Kimi,
/// Z.ai's GLM, MiniMax's M2.7, Anthropic's and OpenAI's retired-at-source ids), never OpenRouter's
/// cheapest host; only a row whose maker publishes none (open weights; OpenAI ids its pricing page
/// no longer lists) uses OpenRouter's public `https://openrouter.ai/api/v1/models` card.
///
/// `claude-3-haiku`, `claude-opus-4` and `gpt-5.2-chat` were removed on 2026-09-30: no provider
/// serves them any more (the catalog smoke reported 404s from every candidate). `deepseek-chat`,
/// `deepseek-reasoner` and `mistral-nemo` were removed on 2026-10-01: their vendors retired them,
/// and the fallbacks served a different model under the name.
///
/// A card that omits `cache_read` or `cache_write` is filled with the **input** rate: no discount,
/// no write premium. Omission is not $0. A consumer that subtracted cache tokens and then multiplied
/// the remainder by a missing rate was billing those tokens free, which is how most of the catalog
/// used to be unpriced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListPrice {
    /// Uncached input, USD per million tokens.
    pub input: &'static str,
    /// Output, USD per million tokens. Reasoning tokens are already inside the output count the
    /// gateway meters; this rate is not an extra charge on top of them.
    pub output: &'static str,
    /// Cache hits, USD per million tokens. Equal to [`Self::input`] when the public card publishes
    /// no separate read rate.
    pub cache_read: &'static str,
    /// Cache writes (5-minute / default), USD per million tokens. Equal to [`Self::input`] when the
    /// public card publishes no separate write rate. On the OpenAI wire the gateway's
    /// `cache_write_tokens` is always zero, so this rate is unused there.
    pub cache_write: &'static str,
}

const fn price(
    input: &'static str,
    output: &'static str,
    cache_read: &'static str,
    cache_write: &'static str,
) -> ListPrice {
    ListPrice {
        input,
        output,
        cache_read,
        cache_write,
    }
}

/// Whether a candidate honors a JSON-schema output constraint (`response_format` `json_schema`,
/// Messages `output_config.format`, Responses `text.format`). Amazon Bedrock's Anthropic Messages
/// surface does not: it answers `output_config.format` (and the beta `output_format`) with 400
/// "Extra inputs are not permitted" on Opus 4.8 and a 404 "The model doesn't exist or doesn't
/// support this API" on Haiku 4.5 (measured 2026-10-01), so a structured-output request on a row
/// whose card advertises [`STRUCTURED_OUTPUTS`] must not be served there. The gateway drops such a
/// candidate from that request's walk.
pub const fn serves_structured_outputs(c: &Candidate) -> bool {
    !matches!(c.provider, ProviderId::Bedrock)
}

/// Upper bound on candidates per row, so the gateway can track which are usable in a single `u8`
/// bitmask with no per-request allocation.
pub const MAX_CANDIDATES: usize = 8;

/// The wire a path serves — `…/messages` is Anthropic, everything else is OpenAI-shaped.
///
/// An independent read of the same fact a candidate's path declares, which is what lets the
/// gateway pick a usage extractor **per attempt** rather than from the row or the provider.
pub fn wire_of_path(path: &str) -> WireFormat {
    if path.ends_with("/messages") {
        WireFormat::Anthropic
    } else {
        WireFormat::OpenAi
    }
}

/// Chat Completions vs Messages vs Responses vs Embeddings, from a candidate path.
///
/// Distinct from [`wire_of_path`]: `/v1/chat/completions` and `/v1/responses` are both OpenAI-wire
/// but different endpoints. The gateway translates when this candidate's path differs from the
/// client's; it never sends a Messages body at Chat Completions (or the reverse).
pub fn endpoint_of_path(path: &str) -> &'static str {
    if path.ends_with("/embeddings") {
        "embeddings"
    } else if path.ends_with("/messages") {
        "messages"
    } else if path.contains("/responses") {
        "responses"
    } else if path.contains("chat/completions") {
        "chat/completions"
    } else {
        "other"
    }
}

/// Anthropic-native primary, OpenRouter Chat Completions failover. OpenRouter spells Claude with a
/// vendor prefix and dots (`anthropic/claude-opus-4.8`), not dashes. The OpenRouter arm is the
/// mixed-wire case: Claude fails onto an OpenAI-compat host; the gateway translates per candidate.
const fn claude(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    [
        Candidate {
            provider: ProviderId::Anthropic,
            upstream_model: native,
            path: "/v1/messages",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: openrouter,
            path: "/api/v1/chat/completions",
        },
    ]
}

/// Anthropic → Bedrock Messages → OpenRouter Chat Completions.
///
/// `bedrock` is a US geo inference-profile id, not a mechanical rewrite of `native`. Only call this
/// with a live-verified string — a guessed id 404s and looks like the client's fault.
const fn claude_bedrock(
    native: &'static str,
    bedrock: &'static str,
    openrouter: &'static str,
) -> [Candidate; 3] {
    [
        Candidate {
            provider: ProviderId::Anthropic,
            upstream_model: native,
            path: "/v1/messages",
        },
        Candidate {
            provider: ProviderId::Bedrock,
            upstream_model: bedrock,
            path: "/anthropic/v1/messages",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: openrouter,
            path: "/api/v1/chat/completions",
        },
    ]
}

/// OpenAI-native primary, OpenRouter Chat Completions failover. Same Chat Completions mount on
/// both sides (`/v1` vs `/api/v1`). The matching Responses arm lives on [`ModelRoute::responses`]
/// rather than here: mixing `/v1/chat/completions` and `/v1/responses` in one walk would break
/// `stream_options.include_usage` injection. Every inbound Responses request on such a row,
/// `store: false` one-shots included, walks that arm as a byte relay; only Chat Completions and
/// Messages clients walk these candidates.
///
/// Not for GPT-5.4 and later: OpenAI's Chat Completions refuses function tools with any reasoning
/// effort on those families (and GPT-5.6 / GPT-6 reason by default), so they use
/// [`openai_responses_first`].
const fn openai_chat(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    [
        Candidate {
            provider: ProviderId::OpenAi,
            upstream_model: native,
            path: "/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: openrouter,
            path: "/api/v1/chat/completions",
        },
    ]
}

/// OpenAI `/v1/embeddings`, then OpenRouter's `/api/v1/embeddings`. An embeddings row is only ever
/// embeddings candidates: the gateway recognizes the row by its primary's path and never translates
/// it to or from a generation endpoint. Ids and paths verified live 2026-09-30.
const fn openai_embeddings(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    [
        Candidate {
            provider: ProviderId::OpenAi,
            upstream_model: native,
            path: "/v1/embeddings",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: openrouter,
            path: "/api/v1/embeddings",
        },
    ]
}

/// OpenRouter Chat Completions as the only candidate: models whose first-party API no longer serves
/// them to our keys (retired at Anthropic, unavailable on OpenAI's API, or reserved for Enterprise
/// or dedicated deployments at Groq, Fireworks or Together) but OpenRouter still does.
/// Live-verified 2026-09-30 by `catalog_rows_are_servable`.
const fn openrouter_only(openrouter: &'static str) -> [Candidate; 1] {
    [Candidate {
        provider: ProviderId::OpenRouter,
        upstream_model: openrouter,
        path: "/api/v1/chat/completions",
    }]
}

/// OpenAI's `/v1/responses` first, then OpenRouter Chat Completions: models OpenAI serves only on
/// the Responses API (the `-pro` and later `-codex` ids 404 on Chat Completions), and GPT-5.4 and
/// later, whose Chat Completions answers function tools with any reasoning effort with a 400
/// ("Function tools with reasoning_effort are not supported … use /v1/responses"; GPT-5.6 and
/// GPT-6 Astra reason by default, so every tool call failed there). A Chat Completions or Messages
/// client is translated onto Responses for the first candidate.
const fn openai_responses_first(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    [
        Candidate {
            provider: ProviderId::OpenAi,
            upstream_model: native,
            path: "/v1/responses",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: openrouter,
            path: "/api/v1/chat/completions",
        },
    ]
}

/// OpenAI `/v1/responses` for a GPT catalog row. One provider: OpenAI's store is not OpenRouter's,
/// so a `previous_response_id` failover across vendors would be another hollow call. A second
/// Responses candidate can be appended when it actually shares that store.
const fn openai_responses(native: &'static str) -> [Candidate; 1] {
    [Candidate {
        provider: ProviderId::OpenAi,
        upstream_model: native,
        path: "/v1/responses",
    }]
}

/// OpenAI-compat Chat Completions primary + OpenRouter Chat Completions failover.
///
/// Same shape as [`openai_chat`], for every other provider that already has a pool key. `path` is
/// the primary's absolute mount — Groq is `/openai/v1/chat/completions`, Fireworks
/// `/inference/v1/chat/completions`, everyone else here `/v1/chat/completions`. No Responses arm:
/// `previous_response_id` is OpenAI's store, not these vendors'.
const fn compat_chat(
    provider: ProviderId,
    native: &'static str,
    path: &'static str,
    openrouter: &'static str,
) -> [Candidate; 2] {
    [
        Candidate {
            provider,
            upstream_model: native,
            path,
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: openrouter,
            path: "/api/v1/chat/completions",
        },
    ]
}

const fn xai(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    compat_chat(ProviderId::XAi, native, "/v1/chat/completions", openrouter)
}

/// xAI's `/v1/responses`, then OpenRouter Chat Completions: xAI serves its multi-agent models on
/// Responses only (Chat Completions answers 400 "Multi Agent requests are not allowed on chat
/// completions"). A Chat Completions or Messages client is translated onto Responses for the first
/// candidate, as for [`openai_responses_first`]; there is no Responses arm (`previous_response_id`
/// would name xAI's store, which no failover shares).
const fn xai_responses_first(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    [
        Candidate {
            provider: ProviderId::XAi,
            upstream_model: native,
            path: "/v1/responses",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: openrouter,
            path: "/api/v1/chat/completions",
        },
    ]
}

const fn deepseek(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    compat_chat(
        ProviderId::DeepSeek,
        native,
        "/v1/chat/completions",
        openrouter,
    )
}

const fn mistral(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    compat_chat(
        ProviderId::Mistral,
        native,
        "/v1/chat/completions",
        openrouter,
    )
}

const fn groq(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    compat_chat(
        ProviderId::Groq,
        native,
        "/openai/v1/chat/completions",
        openrouter,
    )
}

const fn together(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    compat_chat(
        ProviderId::Together,
        native,
        "/v1/chat/completions",
        openrouter,
    )
}

/// DeepSeek V4 Pro, then Together's copy of the **same snapshot** (`DeepSeek-V4-Pro-0813`).
/// OpenRouter's `deepseek/deepseek-v4-pro` is the older 0423 snapshot, so it is not a failover for
/// this row: a fallback must serve the model the row names, not a neighbour.
const fn deepseek_v4_pro() -> [Candidate; 2] {
    [
        Candidate {
            provider: ProviderId::DeepSeek,
            upstream_model: "deepseek-v4-pro",
            path: "/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::Together,
            upstream_model: "deepseek-ai/DeepSeek-V4-Pro-0813",
            path: "/v1/chat/completions",
        },
    ]
}

/// Groq + Together + Fireworks for GPT-OSS 120B. Catalog name is the shared
/// `openai/gpt-oss-120b` Groq/Together id; Fireworks keeps its own spelling as an alias.
/// Cerebras's bare `gpt-oss-120b` is not listed — `for_model_id` prefix-matches `gpt-` to OpenAI.
///
/// No OpenRouter candidate: OpenRouter spreads `openai/gpt-oss-120b` across hosts, and one of them
/// (CoreWeave) answers any forced `tool_choice` (a named function or `required`) with a 200 whose
/// `finish_reason` is `error` and whose message is empty (measured 2026-10-01: two of three forced
/// calls), so the row's advertised tools failed whenever the walk reached it. Three hosts that
/// serve the same weights remain.
const fn gpt_oss_120b() -> [Candidate; 3] {
    [
        Candidate {
            provider: ProviderId::Groq,
            upstream_model: "openai/gpt-oss-120b",
            path: "/openai/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::Together,
            upstream_model: "openai/gpt-oss-120b",
            path: "/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::Fireworks,
            upstream_model: "accounts/fireworks/models/gpt-oss-120b",
            path: "/inference/v1/chat/completions",
        },
    ]
}

/// Together + Fireworks + OpenRouter for Kimi K3. Canonical name is the OpenRouter slug people send.
const fn kimi_k3() -> [Candidate; 3] {
    [
        Candidate {
            provider: ProviderId::Together,
            upstream_model: "moonshotai/Kimi-K3",
            path: "/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::Fireworks,
            upstream_model: "accounts/fireworks/models/kimi-k3",
            path: "/inference/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: "moonshotai/kimi-k3",
            path: "/api/v1/chat/completions",
        },
    ]
}

/// Together + Fireworks + OpenRouter for GLM-5.2. Fireworks spells the version separator as `p`.
const fn glm_5_2() -> [Candidate; 3] {
    [
        Candidate {
            provider: ProviderId::Together,
            upstream_model: "zai-org/GLM-5.2",
            path: "/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::Fireworks,
            upstream_model: "accounts/fireworks/models/glm-5p2",
            path: "/inference/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: "z-ai/glm-5.2",
            path: "/api/v1/chat/completions",
        },
    ]
}

/// Together + Fireworks + OpenRouter for MiniMax M3.
const fn minimax_m3() -> [Candidate; 3] {
    [
        Candidate {
            provider: ProviderId::Together,
            upstream_model: "MiniMaxAI/MiniMax-M3",
            path: "/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::Fireworks,
            upstream_model: "accounts/fireworks/models/minimax-m3",
            path: "/inference/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: "minimax/minimax-m3",
            path: "/api/v1/chat/completions",
        },
    ]
}

/// Every routable model, **sorted by `model`** — [`for_model`] binary-searches it.
///
/// Native ids are the providers' own published aliases (Anthropic Models overview, OpenAI
/// Models catalog, xAI / DeepSeek / Mistral / Groq / Together / Fireworks catalogs, 2026-09-19).
/// OpenRouter spellings were taken from the live `https://openrouter.ai/api/v1/models` list the
/// same day (446 models). `catalog_rows_are_servable` re-verifies each pair against the real
/// providers whenever the keys are present.
pub const MODEL_ROUTES: &[ModelRoute] = &[
    // Claude on the Anthropic wire.
    //
    // Default shape is Anthropic first-party, then OpenRouter Chat Completions (`claude()`). OpenRouter
    // chooses its own backend per request — observed serving these ids from both Anthropic
    // directly and Amazon Bedrock — so that second candidate is *not* a guaranteed independent
    // supply. It covers failures that are ours: egress blocked, our Anthropic key throttled,
    // api.anthropic.com unreachable from us.
    //
    // Two rows have a live-verified independent second source: Amazon Bedrock's Messages API
    // (`claude_bedrock()`, `/anthropic/v1/messages`, `x-api-key`). Ids are the US geo inference
    // profiles the default `bedrock-runtime.us-east-1.amazonaws.com` host serves. OpenRouter
    // stays third on those rows. Do not add a Bedrock candidate without a live-verified
    // inference-profile id — a wrong id 404s and looks like the client's fault.
    //
    // Current lineup (Fable 5.1 / Opus 5 / Sonnet 5 / Haiku 4.5) plus the still-served 4.x
    // snapshots Anthropic lists as legacy. Dateless 4.6+ ids are pinned snapshots, not aliases.
    ModelRoute {
        model: "claude-fable-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-fable-5", "anthropic/claude-fable-5"),
        responses: &[],
        price: price("10", "50", "1", "12.5"),
        card: card(
            "Claude Fable 5",
            "anthropic",
            1780790400,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-fable-5-1",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-fable-5-1", "anthropic/claude-fable-5.1"),
        responses: &[],
        price: price("10", "50", "0.25", "12.5"),
        card: card(
            "Claude Fable 5.1",
            "anthropic",
            1787875200,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-haiku-4-5",
        wire: WireFormat::Anthropic,
        candidates: &claude_bedrock(
            "claude-haiku-4-5",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic/claude-haiku-4.5",
        ),
        responses: &[],
        price: price("1", "5", "0.1", "1.25"),
        card: card(
            "Claude Haiku 4.5",
            "anthropic",
            1760486400,
            200_000,
            64_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-opus-4-1",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("anthropic/claude-opus-4.1"), // retired at Anthropic
        responses: &[],
        price: price("15", "75", "1.5", "18.75"),
        card: card(
            "Claude Opus 4.1",
            "anthropic",
            1754411591,
            200_000,
            32_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING,
        ),
    },
    ModelRoute {
        model: "claude-opus-4-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-4-5", "anthropic/claude-opus-4.5"),
        responses: &[],
        price: price("5", "25", "0.5", "6.25"),
        card: card(
            "Claude Opus 4.5",
            "anthropic",
            1763942400,
            200_000,
            64_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-opus-4-6",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-4-6", "anthropic/claude-opus-4.6"),
        responses: &[],
        price: price("5", "25", "0.5", "6.25"),
        card: card(
            "Claude Opus 4.6",
            "anthropic",
            1770163200,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-opus-4-7",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-4-7", "anthropic/claude-opus-4.7"),
        responses: &[],
        price: price("5", "25", "0.5", "6.25"),
        card: card(
            "Claude Opus 4.7",
            "anthropic",
            1776124800,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-opus-4-8",
        wire: WireFormat::Anthropic,
        candidates: &claude_bedrock(
            "claude-opus-4-8",
            "us.anthropic.claude-opus-4-8",
            "anthropic/claude-opus-4.8",
        ),
        responses: &[],
        price: price("5", "25", "0.5", "6.25"),
        card: card(
            "Claude Opus 4.8",
            "anthropic",
            1779926400,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-opus-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-5", "anthropic/claude-opus-5"),
        responses: &[],
        price: price("5", "25", "0.5", "6.25"),
        card: card(
            "Claude Opus 5",
            "anthropic",
            1784851200,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-opus-5-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-5-5", "anthropic/claude-opus-5.5"),
        responses: &[],
        price: price("4", "20", "0.2", "5"),
        card: card(
            "Claude Opus 5.5",
            "anthropic",
            1790007840,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-sonnet-4",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("anthropic/claude-sonnet-4"), // retired at Anthropic
        responses: &[],
        price: price("3", "15", "0.3", "3.75"),
        card: card(
            "Claude Sonnet 4",
            "anthropic",
            1747930371,
            200_000,
            64_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING,
        ),
    },
    ModelRoute {
        model: "claude-sonnet-4-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-sonnet-4-5", "anthropic/claude-sonnet-4.5"),
        responses: &[],
        price: price("3", "15", "0.3", "3.75"),
        card: card(
            "Claude Sonnet 4.5",
            "anthropic",
            1759104000,
            200_000,
            64_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-sonnet-4-6",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-sonnet-4-6", "anthropic/claude-sonnet-4.6"),
        responses: &[],
        price: price("3", "15", "0.3", "3.75"),
        card: card(
            "Claude Sonnet 4.6",
            "anthropic",
            1771286400,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-sonnet-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-sonnet-5", "anthropic/claude-sonnet-5"),
        responses: &[],
        price: price("2", "10", "0.2", "2.5"),
        card: card(
            "Claude Sonnet 5",
            "anthropic",
            1782691200,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "claude-sonnet-5-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-sonnet-5-5", "anthropic/claude-sonnet-5.5"),
        responses: &[],
        price: price("2", "10", "0.2", "2.5"),
        card: card(
            "Claude Sonnet 5.5",
            "anthropic",
            1790553600,
            1_000_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    // Mistral `-latest` aliases (GA only). Magistral and Devstral are retired as of 2026-09;
    // a guessed still-served alias 404s and looks like the client's fault. `mistral-nemo`
    // (`open-mistral-nemo-2407`) was retired 2026-07-31 and its row removed. Mistral publishes no
    // per-model cache rate ("up to 90%" is not a rate) and no max output, so cache rates equal input
    // and max output is `UNPUBLISHED_MAX_OUTPUT`. Its "256k" / "128k" windows are 262,144 / 131,072.
    ModelRoute {
        model: "codestral-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("codestral-latest", "mistralai/codestral-2508"),
        responses: &[],
        price: price("0.3", "0.9", "0.3", "0.3"), // no published cache rates; both equal input
        card: card_unpublished_output(
            "Codestral 2508",
            "mistralai",
            1754079630,
            131_072,
            IN_TEXT | IN_FILE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    // DeepSeek. The only served names are `deepseek-flash` (V4.1-Flash) and `deepseek-v4-pro`
    // (V4-Pro-0813). `deepseek-chat` and `deepseek-reasoner` were retired 2026-07-24 and their rows
    // removed: the native id 404s, and the OpenRouter slugs they fell back to are V3 and R1, so the
    // row would have served a different model under the name.
    //
    // DeepSeek prices peak hours (01:00–04:00 and 06:00–10:00 UTC, Mon–Fri) at 2x off-peak. One
    // list price cannot say that, so these rows list the peak (standard) rate and over-list
    // off-peak traffic 2x.
    ModelRoute {
        model: "deepseek-flash",
        wire: WireFormat::OpenAi,
        candidates: &deepseek("deepseek-flash", "deepseek/deepseek-v4.1-flash"),
        responses: &[],
        price: price("0.3", "1.2", "0.006", "0.3"), // peak rate (off-peak is half); cache_write unpublished, equals input
        card: card(
            "DeepSeek V4.1 Flash",
            "deepseek",
            1789021285,
            1_048_576,
            384_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "deepseek-v4-pro",
        wire: WireFormat::OpenAi,
        candidates: &deepseek_v4_pro(),
        responses: &[],
        price: price("1.32", "3.96", "0.044", "1.32"), // peak rate (off-peak is half); cache_write unpublished, equals input
        card: card(
            "DeepSeek V4 Pro 0813",
            "deepseek",
            1786579200,
            1_048_576,
            384_000,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    // Gemma 4 on OpenRouter. Together lists `google/gemma-4-31B-it` with a price but answers a
    // serverless key 400 model_not_available ("Unable to access non-serverless model"), and its
    // serverless models table omits it, so it is not a candidate. Google sells no Gemma API, so the
    // price is OpenRouter's (the only host). Not a Gemini dialect: Chat Completions.
    ModelRoute {
        model: "google/gemma-4-31b-it",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("google/gemma-4-31b-it"), // Together: not serverless
        responses: &[],
        price: price("0.09", "0.34", "0.05", "0.09"), // cache_write unpublished; equals input
        card: card(
            "Gemma 4 31B",
            "google",
            1775148486,
            262_144,
            16_384,
            IN_TEXT | IN_IMAGE | IN_VIDEO,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    // The same shape on the OpenAI wire, where the two mounts differ as well (`/v1` vs `/api/v1`).
    // Flagships first in the *id* sort: 4.x, then 5 / 5.4 / 5.5 / 5.6, then 6 Astra, then o-series.
    // `responses` is the arm every inbound `/v1/responses` walks; Chat Completions / Messages
    // inbound walks `candidates`. From GPT-5.4 on, `candidates` reach OpenAI over Responses too
    // (`openai_responses_first`): OpenAI's Chat Completions refuses function tools with reasoning
    // on those families. GPT-5-family cards list OpenAI's input cap (the window less the max
    // output), which is what the API enforces, not the window.
    //
    // OpenAI bills prompts over 272K input tokens at 2x input and 1.5x output on gpt-5.4, gpt-5.4-pro,
    // gpt-5.5, gpt-5.5-pro, gpt-5.6-* and gpt-6-astra. These rows list the base tier only; such a
    // request is under-listed. A cache-write rate is published only for gpt-6-astra and gpt-5.6-*.
    ModelRoute {
        model: "gpt-4",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4", "openai/gpt-4"),
        responses: &openai_responses("gpt-4"),
        price: price("30", "60", "30", "30"), // no separate cache card; both rates equal input
        card: card("GPT-4", "openai", 1687882411, 8_191, 4_096, IN_TEXT, TOOLS),
    },
    ModelRoute {
        model: "gpt-4-turbo",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4-turbo", "openai/gpt-4-turbo"),
        responses: &openai_responses("gpt-4-turbo"),
        price: price("10", "30", "10", "10"), // no separate cache card; both rates equal input
        card: card(
            "GPT-4 Turbo",
            "openai",
            1712361441,
            128_000,
            4_096,
            IN_TEXT | IN_IMAGE,
            TOOLS,
        ),
    },
    ModelRoute {
        model: "gpt-4.1",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4.1", "openai/gpt-4.1"),
        responses: &openai_responses("gpt-4.1"),
        price: price("2", "8", "0.5", "2"), // cache_write unpublished; equals input
        card: card(
            "GPT-4.1",
            "openai",
            1744316542,
            1_047_576,
            32_768,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-4.1-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4.1-mini", "openai/gpt-4.1-mini"),
        responses: &openai_responses("gpt-4.1-mini"),
        price: price("0.4", "1.6", "0.1", "0.4"), // cache_write unpublished; equals input
        card: card(
            "GPT-4.1 Mini",
            "openai",
            1744318173,
            1_047_576,
            32_768,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-4.1-nano",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4.1-nano", "openai/gpt-4.1-nano"),
        responses: &openai_responses("gpt-4.1-nano"),
        price: price("0.1", "0.4", "0.025", "0.1"), // cache_write unpublished; equals input
        card: card(
            "GPT-4.1 Nano",
            "openai",
            1744321707,
            1_047_576,
            32_768,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-4o",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4o", "openai/gpt-4o"),
        responses: &openai_responses("gpt-4o"),
        price: price("2.5", "10", "1.25", "2.5"), // cache_write unpublished; equals input
        card: card(
            "GPT-4o",
            "openai",
            1715367049,
            128_000,
            16_384,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-4o-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4o-mini", "openai/gpt-4o-mini"),
        responses: &openai_responses("gpt-4o-mini"),
        price: price("0.15", "0.6", "0.075", "0.15"), // cache_write unpublished; equals input
        card: card(
            "GPT-4o-mini",
            "openai",
            1721172741,
            128_000,
            16_384,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5", "openai/gpt-5"),
        responses: &openai_responses("gpt-5"),
        price: price("1.25", "10", "0.125", "1.25"), // cache_write unpublished; equals input
        card: card(
            "GPT-5",
            "openai",
            1754425777,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5-mini", "openai/gpt-5-mini"),
        responses: &openai_responses("gpt-5-mini"),
        price: price("0.25", "2", "0.025", "0.25"), // cache_write unpublished; equals input
        card: card(
            "GPT-5 Mini",
            "openai",
            1754425928,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5-nano",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5-nano", "openai/gpt-5-nano"),
        responses: &openai_responses("gpt-5-nano"),
        price: price("0.05", "0.4", "0.005", "0.05"), // cache_write unpublished; equals input
        card: card(
            "GPT-5 Nano",
            "openai",
            1754426384,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5-pro", "openai/gpt-5-pro"),
        responses: &openai_responses("gpt-5-pro"),
        price: price("15", "120", "15", "15"), // no separate cache card; both rates equal input
        card: card(
            "GPT-5 Pro",
            "openai",
            1759469822,
            128_000,
            272_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.1",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.1", "openai/gpt-5.1"),
        responses: &openai_responses("gpt-5.1"),
        price: price("1.25", "10", "0.125", "1.25"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.1",
            "openai",
            1762800673,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.1-codex",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("openai/gpt-5.1-codex"), // not served to our OpenAI key
        responses: &[],
        price: price("1.25", "10", "0.13", "1.25"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.1-Codex",
            "openai",
            1762988221,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.1-codex-max",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("openai/gpt-5.1-codex-max"), // not served to our OpenAI key
        responses: &[],
        price: price("1.25", "10", "0.125", "1.25"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.1-Codex-Max",
            "openai",
            1763671532,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.1-codex-mini",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("openai/gpt-5.1-codex-mini"), // not served to our OpenAI key
        responses: &[],
        price: price("0.25", "2", "0.03", "0.25"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.1-Codex-Mini",
            "openai",
            1763007109,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.2",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.2", "openai/gpt-5.2"),
        responses: &openai_responses("gpt-5.2"),
        price: price("1.75", "14", "0.175", "1.75"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.2",
            "openai",
            1765313051,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.2-codex",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("openai/gpt-5.2-codex"), // not served to our OpenAI key
        responses: &[],
        price: price("1.75", "14", "0.175", "1.75"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.2-Codex",
            "openai",
            1766164985,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.2-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.2-pro", "openai/gpt-5.2-pro"),
        responses: &openai_responses("gpt-5.2-pro"),
        price: price("21", "168", "21", "21"), // no separate cache card; both rates equal input
        card: card(
            "GPT-5.2 Pro",
            "openai",
            1765343983,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING,
        ),
    },
    ModelRoute {
        model: "gpt-5.3-codex",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.3-codex", "openai/gpt-5.3-codex"),
        responses: &openai_responses("gpt-5.3-codex"),
        price: price("1.75", "14", "0.175", "1.75"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.3-Codex",
            "openai",
            1770537915,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.4",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.4", "openai/gpt-5.4"),
        responses: &openai_responses("gpt-5.4"),
        price: price("2.5", "15", "0.25", "2.5"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.4",
            "openai",
            1772691852,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.4-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.4-mini", "openai/gpt-5.4-mini"),
        responses: &openai_responses("gpt-5.4-mini"),
        price: price("0.75", "4.5", "0.075", "0.75"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.4 Mini",
            "openai",
            1773451123,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.4-nano",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.4-nano", "openai/gpt-5.4-nano"),
        responses: &openai_responses("gpt-5.4-nano"),
        price: price("0.2", "1.25", "0.02", "0.2"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.4 Nano",
            "openai",
            1773450870,
            272_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.4-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.4-pro", "openai/gpt-5.4-pro"),
        responses: &openai_responses("gpt-5.4-pro"),
        price: price("30", "180", "30", "30"), // no separate cache card; both rates equal input
        card: card(
            "GPT-5.4 Pro",
            "openai",
            1772659601,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING,
        ),
    },
    ModelRoute {
        model: "gpt-5.5",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.5", "openai/gpt-5.5"),
        responses: &openai_responses("gpt-5.5"),
        price: price("5", "30", "0.5", "5"), // cache_write unpublished; equals input
        card: card(
            "GPT-5.5",
            "openai",
            1776824847,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.5-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.5-pro", "openai/gpt-5.5-pro"),
        responses: &openai_responses("gpt-5.5-pro"),
        price: price("30", "180", "30", "30"), // no separate cache card; both rates equal input
        card: card(
            "GPT-5.5 Pro",
            "openai",
            1776894349,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.6-luna",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.6-luna", "openai/gpt-5.6-luna"),
        responses: &openai_responses("gpt-5.6-luna"),
        price: price("0.2", "1.2", "0.02", "0.25"),
        card: card(
            "GPT-5.6 Luna",
            "openai",
            1782228658,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.6-luna-pro",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("openai/gpt-5.6-luna-pro"), // not served to our OpenAI key
        responses: &[],
        price: price("0.2", "1.2", "0.02", "0.25"),
        card: card(
            "GPT-5.6 Luna Pro",
            "openai",
            1783590867,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.6-sol",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.6-sol", "openai/gpt-5.6-sol"),
        responses: &openai_responses("gpt-5.6-sol"),
        price: price("4", "20", "0.4", "5"), // OpenAI standard tier, not the Batch/Flex row
        card: card(
            "GPT-5.6 Sol",
            "openai",
            1782228018,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.6-sol-pro",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("openai/gpt-5.6-sol-pro"), // not served to our OpenAI key
        responses: &[],
        price: price("4", "20", "0.4", "5"),
        card: card(
            "GPT-5.6 Sol Pro",
            "openai",
            1783590854,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.6-terra",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-5.6-terra", "openai/gpt-5.6-terra"),
        responses: &openai_responses("gpt-5.6-terra"),
        price: price("2", "12", "0.2", "2.5"),
        card: card(
            "GPT-5.6 Terra",
            "openai",
            1782228459,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-5.6-terra-pro",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("openai/gpt-5.6-terra-pro"), // not served to our OpenAI key
        responses: &[],
        price: price("2", "12", "0.2", "2.5"),
        card: card(
            "GPT-5.6 Terra Pro",
            "openai",
            1783590861,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-6-astra",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("gpt-6-astra", "openai/gpt-6-astra"),
        responses: &openai_responses("gpt-6-astra"),
        price: price("10", "50", "1", "12.5"),
        card: card(
            "GPT-6 Astra",
            "openai",
            1787853604,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "gpt-6-astra-pro",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("openai/gpt-6-astra-pro"), // not served to our OpenAI key
        responses: &[],
        price: price("10", "50", "1", "12.5"),
        card: card(
            "GPT-6 Astra Pro",
            "openai",
            1788552835,
            922_000,
            128_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    // xAI Grok. Native ids from the 2026-09-17 xAI models table plus `grok-4.20-multi-agent`
    // (OpenRouter `x-ai/grok-4.20-multi-agent`, 2026-09-19). No Responses arm — session state
    // is OpenAI's store. xAI reads PDFs on `/v1/responses` only, and these rows reach it over Chat
    // Completions, so their cards list no file input. Multi-agent is Responses-only at xAI, so that
    // row's xAI candidate is `/v1/responses` (`xai_responses_first`).
    //
    // xAI bills a prompt of 200k tokens or more at 2x input, cache and output for the whole request.
    // These rows list the < 200k tier; such a request is under-listed by half. xAI publishes no max
    // output, so every Grok row uses `UNPUBLISHED_MAX_OUTPUT`.
    ModelRoute {
        model: "grok-4.20",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-4.20", "x-ai/grok-4.20"),
        responses: &[],
        price: price("1.25", "2.5", "0.2", "1.25"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Grok 4.20",
            "xai",
            1773014400,
            1_000_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "grok-4.20-multi-agent",
        wire: WireFormat::OpenAi,
        candidates: &xai_responses_first("grok-4.20-multi-agent", "x-ai/grok-4.20-multi-agent"),
        responses: &[],
        price: price("1.25", "2.5", "0.2", "1.25"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Grok 4.20 Multi-Agent",
            "xai",
            1773014400,
            1_000_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "grok-4.3",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-4.3", "x-ai/grok-4.3"),
        responses: &[],
        price: price("1.25", "2.5", "0.2", "1.25"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Grok 4.3",
            "xai",
            1776384000,
            1_000_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "grok-4.5",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-4.5", "x-ai/grok-4.5"),
        responses: &[],
        price: price("2", "6", "0.3", "2"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Grok 4.5",
            "xai",
            1782691200,
            500_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "grok-4.6",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-4.6", "x-ai/grok-4.6"),
        responses: &[],
        price: price("2", "6", "0.5", "2"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Grok 4.6",
            "xai",
            1785974400,
            500_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "grok-build-0.1",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-build-0.1", "x-ai/grok-build-0.1"),
        responses: &[],
        price: price("1", "2", "0.2", "1"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Grok Build 0.1",
            "xai",
            1776297600,
            256_000,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    // Groq / Together / Fireworks llama + qwen + open-weight ids people send. No Meta row, so
    // primary is a host that serves the model to a standard serverless key: Groq for GPT-OSS and
    // Qwen3.8 27B, Together for Llama 3.3 / Qwen / Kimi / GLM / MiniMax / Inkling. A candidate the
    // host reserves for Enterprise or dedicated deployments is not listed (Groq
    // `llama-3.1-8b-instant`, `llama-3.3-70b-versatile` and `minimaxai/minimax-m2.7`; Fireworks
    // Llama 4, Kimi K2.6, GLM 5.1 and Llama 3.3; Together Kimi K2.7 Code, GPT-OSS 20B, Gemma 4 31B
    // and Qwen2.5 7B Turbo): it fails over on every request. Rows left with only OpenRouter are
    // `openrouter_only`, and keep their names so existing clients still resolve. Groq caches only
    // GPT-OSS; Together publishes no cache-write rate, so those rates equal input.
    ModelRoute {
        model: "llama-3.1-8b-instant",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("meta-llama/llama-3.1-8b-instruct"), // Groq: Enterprise-only
        responses: &[],
        price: price("0.05", "0.08", "0.025", "0.05"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Llama 3.1 8B Instruct",
            "meta-llama",
            1721692800,
            131_072,
            IN_TEXT,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "llama-3.3-70b-versatile",
        wire: WireFormat::OpenAi,
        candidates: &together(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/llama-3.3-70b-instruct",
        ),
        responses: &[],
        price: price("1.04", "1.04", "1.04", "1.04"), // no published cache rates; both equal input
        card: card(
            "Llama 3.3 70B Instruct",
            "meta-llama",
            1733506137,
            131_072,
            16_384,
            IN_TEXT,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "meta-llama/llama-4-maverick",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("meta-llama/llama-4-maverick"), // Fireworks: not serverless
        responses: &[],
        price: price("0.1875", "0.6525", "0.1875", "0.1875"), // no separate cache card; both rates equal input
        card: card(
            "Llama 4 Maverick",
            "meta-llama",
            1743881822,
            1_048_576,
            16_384,
            IN_TEXT | IN_IMAGE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "meta-llama/llama-4-scout",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("meta-llama/llama-4-scout"), // Fireworks: not serverless
        responses: &[],
        price: price("0.1", "0.3", "0.1", "0.1"), // no separate cache card; both rates equal input
        card: card(
            "Llama 4 Scout",
            "meta-llama",
            1743881519,
            1_048_576,
            16_384,
            IN_TEXT | IN_IMAGE,
            STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "meta/muse-glimmer-30b",
        wire: WireFormat::OpenAi,
        candidates: &together("meta-models/Muse-Glimmer-30B", "meta/muse-glimmer-30b"),
        responses: &[],
        price: price("0.35", "1.5", "0.04", "0.35"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Muse Glimmer 30B",
            "meta",
            1786302394,
            131_072,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "minimax/minimax-m3",
        wire: WireFormat::OpenAi,
        candidates: &minimax_m3(),
        responses: &[],
        price: price("0.3", "1.2", "0.06", "0.3"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "MiniMax M3",
            "minimax",
            1780245374,
            524_288,
            IN_TEXT | IN_IMAGE | IN_VIDEO,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "minimaxai/minimax-m2.7",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("minimax/minimax-m2.7"), // Groq: Enterprise-only
        responses: &[],
        price: price("0.3", "1.2", "0.06", "0.375"), // MiniMax's own pay-as-you-go rate
        card: card_unpublished_output(
            "MiniMax M2.7",
            "minimax",
            1773836697,
            204_800,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "ministral-14b-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("ministral-14b-latest", "mistralai/ministral-14b-2512"),
        responses: &[],
        price: price("0.2", "0.2", "0.2", "0.2"), // no published cache rates; both equal input
        card: card_unpublished_output(
            "Ministral 3 14B 2512",
            "mistralai",
            1764681735,
            262_144,
            IN_TEXT | IN_IMAGE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "ministral-3b-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("ministral-3b-latest", "mistralai/ministral-3b-2512"),
        responses: &[],
        price: price("0.1", "0.1", "0.1", "0.1"), // no published cache rates; both equal input
        card: card_unpublished_output(
            "Ministral 3 3B 2512",
            "mistralai",
            1764681560,
            131_072,
            IN_TEXT | IN_IMAGE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "ministral-8b-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("ministral-8b-latest", "mistralai/ministral-8b-2512"),
        responses: &[],
        price: price("0.15", "0.15", "0.15", "0.15"), // no published cache rates; both equal input
        card: card_unpublished_output(
            "Ministral 3 8B 2512",
            "mistralai",
            1764681654,
            262_144,
            IN_TEXT | IN_IMAGE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "mistral-large-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("mistral-large-latest", "mistralai/mistral-large-2512"),
        responses: &[],
        price: price("0.5", "1.5", "0.5", "0.5"), // no published cache rates; both equal input
        card: card_unpublished_output(
            "Mistral Large 3 2512",
            "mistralai",
            1764624472,
            262_144,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "mistral-medium-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("mistral-medium-latest", "mistralai/mistral-medium-3-5"),
        responses: &[],
        price: price("1.5", "7.5", "1.5", "1.5"), // no published cache rates; both equal input
        card: card_unpublished_output(
            "Mistral Medium 3.5",
            "mistralai",
            1777570439,
            262_144,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "mistral-small-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("mistral-small-latest", "mistralai/mistral-small-2603"),
        responses: &[],
        price: price("0.15", "0.6", "0.15", "0.15"), // no published cache rates; both equal input
        card: card_unpublished_output(
            "Mistral Small 4",
            "mistralai",
            1773695685,
            262_144,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "moonshotai/kimi-k2.6",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("moonshotai/kimi-k2.6"), // Fireworks: serverless deprecated
        responses: &[],
        price: price("0.95", "4", "0.16", "0.95"), // Moonshot's own rate; cache_write unpublished, equals input
        card: card_unpublished_output(
            "Kimi K2.6",
            "moonshotai",
            1776699402,
            262_144,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "moonshotai/kimi-k2.7-code",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("moonshotai/kimi-k2.7-code"), // Together: dedicated only
        responses: &[],
        price: price("0.95", "4", "0.19", "0.95"), // Moonshot's own rate; cache_write unpublished, equals input
        card: card_unpublished_output(
            "Kimi K2.7 Code",
            "moonshotai",
            1781266361,
            262_144,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "moonshotai/kimi-k3",
        wire: WireFormat::OpenAi,
        candidates: &kimi_k3(),
        responses: &[],
        price: price("3", "15", "0.3", "3"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Kimi K3",
            "moonshotai",
            1784215858,
            1_048_576,
            IN_TEXT | IN_IMAGE | IN_VIDEO,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "o1",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o1", "openai/o1"),
        responses: &openai_responses("o1"),
        price: price("15", "60", "7.5", "15"), // cache_write unpublished; equals input
        card: card(
            "o1",
            "openai",
            1734375816,
            200_000,
            100_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "o1-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_responses_first("o1-pro", "openai/o1-pro"),
        responses: &openai_responses("o1-pro"),
        price: price("150", "600", "150", "150"), // no separate cache card; both rates equal input
        card: card(
            "o1-pro",
            "openai",
            1742251791,
            200_000,
            100_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "o3",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o3", "openai/o3"),
        responses: &openai_responses("o3"),
        price: price("2", "8", "0.5", "2"), // cache_write unpublished; equals input
        card: card(
            "o3",
            "openai",
            1744225308,
            200_000,
            100_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "o3-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o3-mini", "openai/o3-mini"),
        responses: &openai_responses("o3-mini"),
        price: price("1.1", "4.4", "0.55", "1.1"), // cache_write unpublished; equals input
        card: card(
            "o3 Mini",
            "openai",
            1737146383,
            200_000,
            100_000,
            IN_TEXT | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "o3-pro",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("openai/o3-pro"), // not served to our OpenAI key
        responses: &[],
        price: price("20", "80", "20", "20"), // no separate cache card; both rates equal input
        card: card(
            "o3 Pro",
            "openai",
            1749598352,
            200_000,
            100_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "o4-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o4-mini", "openai/o4-mini"),
        responses: &openai_responses("o4-mini"),
        price: price("1.1", "4.4", "0.275", "1.1"), // cache_write unpublished; equals input
        card: card(
            "o4 Mini",
            "openai",
            1744225351,
            200_000,
            100_000,
            IN_TEXT | IN_IMAGE | IN_FILE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "openai/gpt-oss-120b",
        wire: WireFormat::OpenAi,
        candidates: &gpt_oss_120b(),
        responses: &[],
        price: price("0.15", "0.6", "0.075", "0.15"), // cache_write unpublished; equals input
        card: card(
            "gpt-oss-120b",
            "openai",
            1754414231,
            131_072,
            65_536,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "openai/gpt-oss-20b",
        wire: WireFormat::OpenAi,
        candidates: &groq("openai/gpt-oss-20b", "openai/gpt-oss-20b"),
        responses: &[],
        price: price("0.075", "0.3", "0.0375", "0.075"), // cache_write unpublished; equals input
        card: card(
            "gpt-oss-20b",
            "openai",
            1754414229,
            131_072,
            65_536,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "openai/gpt-oss-safeguard-20b",
        wire: WireFormat::OpenAi,
        candidates: &groq(
            "openai/gpt-oss-safeguard-20b",
            "openai/gpt-oss-safeguard-20b",
        ),
        responses: &[],
        price: price("0.075", "0.3", "0.0375", "0.075"), // cache_write unpublished; equals input
        card: card(
            "gpt-oss-safeguard-20b",
            "openai",
            1761752836,
            131_072,
            65_536,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "qwen/qwen-2.5-7b-instruct",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("qwen/qwen-2.5-7b-instruct"), // Together: not serverless
        responses: &[],
        price: price("0.1", "0.2", "0.1", "0.1"), // OpenRouter's (the only host); no cache rates, both equal input
        card: card(
            "Qwen2.5 7B Instruct",
            "qwen",
            1729036800,
            32_768,
            8_192,
            IN_TEXT,
            TOOLS | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "qwen/qwen3.5-9b",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.5-9B", "qwen/qwen3.5-9b"),
        responses: &[],
        price: price("0.17", "0.25", "0.17", "0.17"), // no published cache rates; both equal input
        card: card(
            "Qwen3.5-9B",
            "qwen",
            1773152396,
            262_144,
            32_768,
            IN_TEXT | IN_IMAGE | IN_VIDEO,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "qwen/qwen3.6-plus",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.6-Plus", "qwen/qwen3.6-plus"),
        responses: &[],
        price: price("0.5", "3", "0.5", "0.5"), // no published cache rates; both equal input
        card: card(
            "Qwen3.6 Plus",
            "qwen",
            1775133557,
            1_000_000,
            65_536,
            IN_TEXT | IN_IMAGE | IN_VIDEO,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "qwen/qwen3.7-max",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.7-Max", "qwen/qwen3.7-max"),
        responses: &[],
        price: price("1.5", "4.5", "0.3", "1.5"), // cached: pricing page $0.30, docs table $0.50. cache_write unpublished; equals input
        card: card(
            "Qwen3.7 Max",
            "qwen",
            1779376861,
            1_000_000,
            131_072,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "qwen/qwen3.7-plus",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.7-Plus", "qwen/qwen3.7-plus"),
        responses: &[],
        price: price("0.32", "1.28", "0.32", "0.32"), // no published cache rates; both equal input
        card: card(
            "Qwen3.7 Plus",
            "qwen",
            1780491783,
            1_000_000,
            131_072,
            IN_TEXT | IN_IMAGE,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "qwen/qwen3.8-2.4t-a95b",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.8-2.4T-A95B", "qwen/qwen3.8-2.4t-a95b"),
        responses: &[],
        price: price("2", "6", "0.25", "2"), // cached: pricing page $0.25, docs table $0.50. cache_write unpublished; equals input
        card: card(
            "Qwen3.8 2.4T A95B",
            "qwen",
            1786551702,
            1_010_000,
            131_072,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "qwen/qwen3.8-27b",
        wire: WireFormat::OpenAi,
        candidates: &groq("qwen/qwen3.8-27b", "qwen/qwen3.8-27b"),
        responses: &[],
        price: price("0.8", "4", "0.8", "0.8"), // Groq lists no prompt caching for this model; both equal input
        card: card(
            "Qwen3.8 27B",
            "qwen",
            1786722910,
            131_072,
            16_384,
            IN_TEXT | IN_IMAGE | IN_VIDEO,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "qwen/qwen3.8-flash",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.8-Flash", "qwen/qwen3.8-flash"),
        responses: &[],
        price: price("0.09", "0.282", "0.09", "0.09"), // docs table and /v1/models $0.282 out (pricing page rounds to $0.28); no cache rates, both equal input
        card: card(
            "Qwen3.8 Flash",
            "qwen",
            1787773060,
            1_000_000,
            131_072,
            IN_TEXT | IN_IMAGE | IN_VIDEO,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    // Embeddings: input only. Output is priced 0 and there is no cache, so both cache rates are
    // the input rate (the `ListPrice` rule for an unpublished rate). `text-embedding-ada-002` is
    // left out: both providers answer it as `text-embedding-ada-002-v2`, a billed id no row names.
    ModelRoute {
        model: "text-embedding-3-large",
        wire: WireFormat::OpenAi,
        candidates: &openai_embeddings("text-embedding-3-large", "openai/text-embedding-3-large"),
        responses: &[],
        price: price("0.13", "0", "0.13", "0.13"),
        card: card(
            "Text Embedding 3 Large",
            "openai",
            1705953180,
            8_192,
            0,
            IN_TEXT,
            0,
        ),
    },
    ModelRoute {
        model: "text-embedding-3-small",
        wire: WireFormat::OpenAi,
        candidates: &openai_embeddings("text-embedding-3-small", "openai/text-embedding-3-small"),
        responses: &[],
        price: price("0.02", "0", "0.02", "0.02"),
        card: card(
            "Text Embedding 3 Small",
            "openai",
            1705948997,
            8_192,
            0,
            IN_TEXT,
            0,
        ),
    },
    ModelRoute {
        model: "thinkingmachines/inkling",
        wire: WireFormat::OpenAi,
        candidates: &together("thinkingmachines/Inkling", "thinkingmachines/inkling"),
        responses: &[],
        price: price("1", "4.05", "0.17", "1"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "Inkling",
            "thinkingmachines",
            1784325956,
            524_288,
            IN_TEXT | IN_IMAGE | IN_AUDIO,
            TOOLS | REASONING,
        ),
    },
    ModelRoute {
        model: "z-ai/glm-5.1",
        wire: WireFormat::OpenAi,
        candidates: &openrouter_only("z-ai/glm-5.1"), // Fireworks: not serverless
        responses: &[],
        price: price("1.4", "4.4", "0.26", "1.4"), // Z.ai's own rate; cache_write unpublished, equals input
        card: card(
            "GLM 5.1",
            "z-ai",
            1775578025,
            204_800,
            131_072,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "z-ai/glm-5.2",
        wire: WireFormat::OpenAi,
        candidates: &glm_5_2(),
        responses: &[],
        price: price("1.4", "4.4", "0.26", "1.4"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "GLM 5.2",
            "z-ai",
            1781631930,
            1_048_575,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "z-ai/glm-5.3",
        wire: WireFormat::OpenAi,
        candidates: &together("zai-org/GLM-5.3", "z-ai/glm-5.3"),
        responses: &[],
        price: price("1.4", "4.4", "0.26", "1.4"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "GLM 5.3",
            "z-ai",
            1787086655,
            1_048_575,
            IN_TEXT,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
    ModelRoute {
        model: "z-ai/glm-5.3-flash",
        wire: WireFormat::OpenAi,
        candidates: &together("zai-org/GLM-5.3-Flash", "z-ai/glm-5.3-flash"),
        responses: &[],
        price: price("0.15", "0.5", "0.03", "0.15"), // cache_write unpublished; equals input
        card: card_unpublished_output(
            "GLM 5.3 Flash",
            "z-ai",
            1787752741,
            1_048_575,
            IN_TEXT | IN_IMAGE | IN_VIDEO,
            TOOLS | REASONING | STRUCTURED_OUTPUTS,
        ),
    },
];

/// The catalog row for a model name, or `None` if we do not serve it.
///
/// [`MODEL_ROUTES`] is sorted (asserted), so the common path is a binary search over `&'static str`
/// — no allocation, and the result is `&'static` so the caller stores a thin pointer. The
/// case-insensitive linear fallback covers a client that upcased the header; it never runs for a
/// well-formed request.
///
/// A miss then tries each row's candidate `upstream_model` ids (OpenRouter's
/// `anthropic/claude-opus-4.8`, Bedrock's inference-profile id, …). Those are not extra products —
/// they are the spellings we already rewrite *to* — so accepting them as aliases is how a caller
/// who copied a vendor slug still hits the row.
pub fn for_model(name: &str) -> Option<&'static ModelRoute> {
    match MODEL_ROUTES.binary_search_by(|r| r.model.cmp(name)) {
        Ok(i) => MODEL_ROUTES.get(i),
        Err(_) => MODEL_ROUTES
            .iter()
            .find(|r| r.model.eq_ignore_ascii_case(name))
            .or_else(|| {
                MODEL_ROUTES.iter().find(|r| {
                    r.candidates
                        .iter()
                        .any(|c| c.upstream_model.eq_ignore_ascii_case(name))
                })
            }),
    }
}

/// OpenAI-shaped `GET /v1/models` body for the catalog, readable by the Anthropic SDK too
/// (`display_name`, `has_more`). Beyond OpenAI's fields each model carries `wire` (`"openai"` /
/// `"anthropic"`, so a caller can pick the matching SDK), its [`ModelCard`] (`context_window`,
/// `max_output_tokens`, `input_modalities`, `output_modalities`, `capabilities`), the `endpoints` it
/// answers on (any generation row serves all three through translation), and `pricing` (USD per
/// million tokens — see [`ListPrice`]). Ids are log-safe (`[a-z0-9._/-]`), display names are
/// tested free of quotes and backslashes, and prices are decimal strings, so this needs no JSON
/// escaping.
pub fn models_list_json() -> &'static str {
    static JSON: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    JSON.get_or_init(|| {
        use std::fmt::Write as _;
        fn names(out: &mut String, bits: u8, table: &[(u8, &str)]) {
            out.push('[');
            let mut first = true;
            for &(bit, name) in table {
                if bits & bit != 0 {
                    if !first {
                        out.push(',');
                    }
                    first = false;
                    let _ = write!(out, "\"{name}\"");
                }
            }
            out.push(']');
        }
        let mut out = String::from(
            "{\"object\":\"list\",\"pricing_unit\":\"usd_per_million_tokens\",\"has_more\":false,\"data\":[",
        );
        for (i, r) in MODEL_ROUTES.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let c = r.card;
            let embeddings = r.candidates[0].path.ends_with("/embeddings");
            let _ = write!(
                out,
                "{{\"id\":\"{}\",\"object\":\"model\",\"type\":\"model\",\"created\":{},\"owned_by\":\"{}\",\"display_name\":\"{}\",\"wire\":\"{}\",\"context_window\":{},\"max_output_tokens\":",
                r.model,
                c.created,
                c.owned_by,
                c.name,
                r.wire.as_str(),
                c.context_window,
            );
            if embeddings {
                out.push_str("null");
            } else {
                let _ = write!(
                    out,
                    "{},\"max_output_published\":{}",
                    c.max_output_tokens, c.max_output_published
                );
            }
            out.push_str(",\"input_modalities\":");
            names(&mut out, c.input, &INPUT_NAMES);
            out.push_str(if embeddings {
                ",\"output_modalities\":[\"embeddings\"],\"capabilities\":"
            } else {
                ",\"output_modalities\":[\"text\"],\"capabilities\":"
            });
            names(&mut out, c.features, &FEATURE_NAMES);
            out.push_str(if embeddings {
                ",\"endpoints\":[\"/v1/embeddings\"]"
            } else {
                ",\"endpoints\":[\"/v1/chat/completions\",\"/v1/messages\",\"/v1/responses\"]"
            });
            let p = r.price;
            let _ = write!(
                out,
                ",\"pricing\":{{\"input\":\"{}\",\"output\":\"{}\",\"cache_read\":\"{}\",\"cache_write\":\"{}\"}}}}",
                p.input, p.output, p.cache_read, p.cache_write,
            );
        }
        out.push_str("]}");
        out
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::{by_id, gateway_providers};

    /// Product floor: managed `/v1` should list a full current generation, not a handful of
    /// flagships. Count is the guard; new rows still have to pass the uniqueness / wire tests below.
    ///
    /// The floor was 100 until 2026-10-01, when three retired models (`deepseek-chat`,
    /// `deepseek-reasoner`, `mistral-nemo`) left the table. A row must never serve a different model
    /// under its name, so the floor came down rather than padding the table with rows to meet it.
    #[test]
    fn catalog_lists_at_least_95_models() {
        assert!(
            MODEL_ROUTES.len() >= 95,
            "MODEL_ROUTES has {} rows; keep the managed catalog at 95+",
            MODEL_ROUTES.len(),
        );
    }

    /// The binary search in `for_model` is only correct on a sorted table, and duplicate names would
    /// make which row wins depend on where the search landed.
    #[test]
    fn model_routes_are_sorted_and_unique() {
        for pair in MODEL_ROUTES.windows(2) {
            assert!(
                pair[0].model < pair[1].model,
                "MODEL_ROUTES must be sorted by `model` and free of duplicates, but {:?} \
                 is not strictly before {:?}",
                pair[0].model,
                pair[1].model,
            );
        }
    }

    /// The gateway logs the matched route's name into the `ai.usage` billing row without running it
    /// through `sanitize_model`. That is only sound because the name is *ours*, from this table, and
    /// restricted to a charset that cannot break out of a JSON string or inject a log line.
    #[test]
    fn model_names_are_lowercase_and_log_safe() {
        for route in MODEL_ROUTES {
            assert!(!route.model.is_empty(), "a route name must not be empty");
            for b in route.model.bytes() {
                assert!(
                    b.is_ascii_lowercase()
                        || b.is_ascii_digit()
                        || matches!(b, b'.' | b'_' | b'/' | b'-'),
                    "route name {:?} contains {:?}, outside the log-safe [a-z0-9._/-] set",
                    route.model,
                    b as char,
                );
            }
        }
    }

    #[test]
    fn every_route_has_between_one_and_max_candidates() {
        for route in MODEL_ROUTES {
            let n = route.candidates.len();
            assert!(
                (1..=MAX_CANDIDATES).contains(&n),
                "route {:?} has {n} candidates; must be 1..={MAX_CANDIDATES} \
                 (the gateway tracks usable candidates in a u8 bitmask)",
                route.model,
            );
        }
    }

    /// A candidate the gateway has no registry entry for is dead on arrival — it would be filtered
    /// out at request time and silently reduce the row's failover depth.
    #[test]
    fn candidates_are_gateway_routable() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                assert!(
                    gateway_providers().any(|p| p.id == c.provider),
                    "route {:?} names {:?}, which is not a gateway-routable provider",
                    route.model,
                    c.provider,
                );
            }
        }
    }

    /// A provider listed twice would burn two connect attempts on the same dead upstream.
    #[test]
    fn candidates_within_a_row_are_distinct() {
        for route in MODEL_ROUTES {
            for (i, c) in route.candidates.iter().enumerate() {
                assert!(
                    !route.candidates[..i]
                        .iter()
                        .any(|p| p.provider == c.provider),
                    "route {:?} lists {:?} more than once",
                    route.model,
                    c.provider,
                );
            }
        }
    }

    /// The primary speaks the row's declared wire; failover candidates may speak another endpoint.
    /// The gateway translates the original client body onto **this** candidate's path, so a mixed
    /// row never forwards a Messages body at Chat Completions (or the reverse).
    #[test]
    fn primary_candidate_matches_the_rows_declared_wire() {
        for route in MODEL_ROUTES {
            let Some(first) = route.candidates.first() else {
                continue;
            };
            assert_eq!(
                wire_of_path(first.path),
                route.wire,
                "route {:?} declares {:?} but primary {:?} points at {:?}",
                route.model,
                route.wire,
                first.provider,
                first.path,
            );
        }
    }

    /// Every candidate path must name Chat Completions, Messages, or Responses so the gateway can
    /// translate onto it. Mixed Messages + Chat Completions in one row is allowed (Claude →
    /// OpenRouter); an unrecognized path would be forwarded as the wrong wire.
    #[test]
    fn every_candidate_path_names_a_known_endpoint() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                let ep = endpoint_of_path(c.path);
                assert_ne!(
                    ep, "other",
                    "route {:?} candidate {:?} path {:?} is not Chat Completions, Messages, Responses, or Embeddings",
                    route.model, c.provider, c.path,
                );
            }
        }
    }

    /// The two OpenAI-wire endpoints stay distinguishable so a translate walk can target one
    /// without inventing a mixed type *inside* a single candidate path.
    #[test]
    fn chat_completions_and_responses_are_distinct_endpoints() {
        assert_eq!(
            endpoint_of_path("/v1/chat/completions"),
            endpoint_of_path("/api/v1/chat/completions")
        );
        assert_ne!(
            endpoint_of_path("/v1/chat/completions"),
            endpoint_of_path("/v1/responses"),
            "the two OpenAI-wire endpoints must be distinguishable",
        );
        assert_eq!(endpoint_of_path("/v1/messages"), "messages");
        assert_eq!(
            endpoint_of_path("/api/v1/chat/completions"),
            "chat/completions"
        );
    }

    #[test]
    fn candidate_paths_are_absolute() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                assert!(
                    c.path.starts_with('/') && !c.path.contains("://"),
                    "route {:?} candidate {:?} path {:?} must be an absolute path, not a URL",
                    route.model,
                    c.provider,
                    c.path,
                );
            }
        }
    }

    /// A candidate's path must sit under its provider's own base URL, where the provider publishes
    /// one. Catches a path copied from the wrong row — the failure mode a 404 at request time.
    #[test]
    fn candidate_paths_sit_under_their_providers_base() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                let spec = by_id(c.provider);
                let Some(base) = spec.base_url else { continue };
                let mount = spec.base_path();
                assert!(
                    c.path.starts_with(mount),
                    "route {:?}: {} serves from {base} (mount {mount:?}), but the candidate path is \
                     {:?}",
                    route.model,
                    spec.name,
                    c.path,
                );
            }
        }
    }

    /// If an id is one a provider serves *natively* by shape, the candidate holding it had better be
    /// that provider. Catches `claude-opus-4-8` listed under Groq. Vendor-slug ids
    /// (`anthropic/claude-opus-4.8`) resolve to no native provider and are correctly skipped.
    #[test]
    fn native_ids_are_not_misrouted() {
        for route in MODEL_ROUTES {
            for c in route.candidates {
                if let Some(native) = crate::for_model_id(c.upstream_model) {
                    assert_eq!(
                        native.id, c.provider,
                        "route {:?} asks {:?} for {:?}, but that id is natively {:?}'s",
                        route.model, c.provider, c.upstream_model, native.id,
                    );
                }
            }
        }
    }

    /// The catalog is routing data. Capability facts belong in `agent_core::models`, and the way
    /// this file would grow them is by someone adding a field — so pin the field set.
    #[test]
    fn a_candidate_carries_routing_facts_only() {
        let c = Candidate {
            provider: ProviderId::OpenAi,
            upstream_model: "gpt-4o-mini",
            path: "/v1/chat/completions",
        };
        // Destructured exhaustively: adding a field fails to compile here, which is the prompt to
        // ask whether it is really routing knowledge.
        let Candidate {
            provider: _,
            upstream_model: _,
            path: _,
        } = c;
    }

    /// The gateway decides a row is embeddings from its primary's path and never translates it,
    /// so a row that mixed an embeddings candidate with a generation one would fail over onto a
    /// different endpoint with the wrong body.
    #[test]
    fn embeddings_rows_are_only_embeddings() {
        for route in MODEL_ROUTES {
            let embeds = route
                .candidates
                .iter()
                .filter(|c| c.path.ends_with("/embeddings"))
                .count();
            assert!(
                embeds == 0 || (embeds == route.candidates.len() && route.responses.is_empty()),
                "{} mixes embeddings and generation candidates",
                route.model
            );
        }
    }

    /// claim: CAT-10
    #[test]
    fn for_model_finds_every_row() {
        for route in MODEL_ROUTES {
            assert_eq!(
                for_model(route.model),
                Some(route),
                "{:?} must resolve to its own row",
                route.model,
            );
        }
    }

    /// claim: CAT-10
    #[test]
    fn for_model_is_case_insensitive() {
        assert_eq!(
            for_model("GPT-4O-MINI").map(|r| r.model),
            Some("gpt-4o-mini"),
        );
    }

    /// claim: CAT-10
    #[test]
    fn for_model_is_none_for_unknown_ids() {
        for unknown in ["", "gpt-4o-min", "gpt-4o-mini-x", "claude", "nonesuch"] {
            assert_eq!(
                for_model(unknown),
                None,
                "{unknown:?} must not resolve — a near-miss is not a match",
            );
        }
    }

    /// A candidate's upstream id is an alias for its row, not a second product. OpenRouter slugs
    /// and Bedrock inference-profile ids must resolve to the same row as the canonical name.
    /// claim: CAT-10
    #[test]
    fn for_model_accepts_candidate_spellings_as_aliases() {
        assert_eq!(
            for_model("anthropic/claude-opus-4.8").map(|r| r.model),
            Some("claude-opus-4-8"),
        );
        assert_eq!(
            for_model("us.anthropic.claude-opus-4-8").map(|r| r.model),
            Some("claude-opus-4-8"),
        );
        assert_eq!(
            for_model("openai/gpt-4o-mini").map(|r| r.model),
            Some("gpt-4o-mini"),
        );
        assert_eq!(
            for_model("ANTHROPIC/CLAUDE-OPUS-4.8").map(|r| r.model),
            Some("claude-opus-4-8"),
        );
        for route in MODEL_ROUTES {
            for c in route.candidates {
                assert_eq!(
                    for_model(c.upstream_model).map(|r| r.model),
                    Some(route.model),
                    "{:?} must alias to {:?}",
                    c.upstream_model,
                    route.model,
                );
            }
        }
    }

    /// Two rows cannot share a candidate id, or `for_model` would have to guess. Canonical names
    /// already unique (`model_routes_are_sorted_and_unique`); this extends that to aliases.
    #[test]
    fn candidate_ids_are_unambiguous_aliases() {
        use std::collections::HashMap;
        let mut seen: HashMap<&str, &str> = HashMap::new();
        for route in MODEL_ROUTES {
            for c in route.candidates {
                if c.upstream_model == route.model {
                    continue;
                }
                assert!(
                    seen.insert(c.upstream_model, route.model).is_none(),
                    "{:?} is a candidate of more than one catalog row",
                    c.upstream_model
                );
                assert!(
                    !MODEL_ROUTES.iter().any(|r| r.model == c.upstream_model),
                    "{:?} cannot be both a catalog name and another row's candidate id",
                    c.upstream_model
                );
            }
        }
    }

    /// claim: E4
    #[test]
    fn models_list_json_describes_every_row() {
        let v: serde_json::Value = serde_json::from_str(models_list_json()).expect("valid JSON");
        assert_eq!(v["object"], "list");
        assert_eq!(v["pricing_unit"], "usd_per_million_tokens");
        assert_eq!(v["has_more"], false);
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), MODEL_ROUTES.len());
        for (m, r) in data.iter().zip(MODEL_ROUTES) {
            let (c, p) = (r.card, r.price);
            assert_eq!(m["id"], r.model);
            assert_eq!(m["object"], "model");
            assert_eq!(m["type"], "model");
            assert_eq!(m["created"], c.created);
            assert_eq!(m["owned_by"], c.owned_by);
            assert_eq!(m["display_name"], c.name);
            assert_eq!(m["wire"], r.wire.as_str());
            assert_eq!(m["context_window"], c.context_window);
            assert_eq!(m["pricing"]["input"], p.input);
            assert_eq!(m["pricing"]["output"], p.output);
            assert_eq!(m["pricing"]["cache_read"], p.cache_read);
            assert_eq!(m["pricing"]["cache_write"], p.cache_write);
            let inputs = m["input_modalities"].as_array().unwrap();
            assert_eq!(inputs.len(), c.input.count_ones() as usize, "{m}");
            assert_eq!(inputs[0], "text", "every model reads text: {m}");
            let caps = m["capabilities"].as_array().unwrap();
            assert_eq!(caps.len(), c.features.count_ones() as usize, "{m}");
            if r.candidates[0].path.ends_with("/embeddings") {
                assert!(m["max_output_tokens"].is_null(), "{m}");
                assert_eq!(m["output_modalities"], serde_json::json!(["embeddings"]));
                assert_eq!(m["endpoints"], serde_json::json!(["/v1/embeddings"]));
                assert!(caps.is_empty(), "{m}");
            } else {
                assert_eq!(m["max_output_tokens"], c.max_output_tokens);
                assert_eq!(m["max_output_published"], c.max_output_published, "{m}");
                assert_eq!(m["output_modalities"], serde_json::json!(["text"]));
                assert_eq!(m["endpoints"].as_array().unwrap().len(), 3);
            }
        }
    }

    /// The card is what a client sizes requests by: a zero window or output cap would make every
    /// request look oversized. An output cap may exceed the input cap where the vendor's window
    /// holds both (gpt-5-pro: 128K in, 272K out of 400K), but not the whole of any window we list.
    /// Names go into the list unescaped.
    #[test]
    fn every_route_has_a_plausible_card() {
        for r in MODEL_ROUTES {
            let c = r.card;
            let embeddings = r.candidates[0].path.ends_with("/embeddings");
            assert!(
                c.context_window >= 512,
                "{} window {}",
                r.model,
                c.context_window
            );
            assert_eq!(
                c.max_output_tokens == 0,
                embeddings,
                "{} max output",
                r.model
            );
            assert!(
                c.max_output_tokens <= 400_000,
                "{} output {} over any vendor window it shares",
                r.model,
                c.max_output_tokens
            );
            assert!(c.input & IN_TEXT != 0, "{} reads no text", r.model);
            assert!(
                c.created > 1_600_000_000,
                "{} created {}",
                r.model,
                c.created
            );
            for s in [c.name, c.owned_by] {
                assert!(
                    !s.is_empty()
                        && !s
                            .chars()
                            .any(|ch| ch == '"' || ch == '\\' || ch.is_control()),
                    "{} card string {s:?} needs escaping",
                    r.model
                );
            }
        }
    }

    /// Every catalog row has a real standard card. A zero rate is how a consumer bills tokens free;
    /// a cache rate above input is not a discount; a write rate of zero is the omission this table
    /// exists to close. Six decimal places is one micro-dollar, which covers every card we store.
    #[test]
    fn every_route_has_a_positive_list_price() {
        fn micros(label: &str, model: &str, raw: &str) -> u64 {
            assert!(
                !raw.is_empty()
                    && raw.bytes().all(|b| b.is_ascii_digit() || b == b'.')
                    && raw.matches('.').count() <= 1
                    && !raw.starts_with('.')
                    && !raw.ends_with('.'),
                "{model} {label} {raw:?} is not a plain decimal",
            );
            let (whole, frac) = raw.split_once('.').unwrap_or((raw, ""));
            assert!(
                frac.len() <= 6,
                "{model} {label} {raw} has more than 6 decimal places",
            );
            let mut padded = frac.to_string();
            padded.extend(std::iter::repeat_n('0', 6 - frac.len()));
            let w = whole.parse::<u64>();
            let f = padded.parse::<u64>();
            assert!(w.is_ok() && f.is_ok(), "{model} {label} {raw:?}");
            let w = w.unwrap_or(0);
            let f = f.unwrap_or(0);
            let scaled = w.checked_mul(1_000_000).and_then(|n| n.checked_add(f));
            assert!(scaled.is_some(), "{model} {label} overflow");
            scaled.unwrap_or(0)
        }
        for route in MODEL_ROUTES {
            let p = route.price;
            let input = micros("input", route.model, p.input);
            let output = micros("output", route.model, p.output);
            let cache_read = micros("cache_read", route.model, p.cache_read);
            let cache_write = micros("cache_write", route.model, p.cache_write);
            assert!(input > 0, "{} input", route.model);
            // Embeddings produce no output tokens; everything else must price them.
            let embeddings = route
                .candidates
                .first()
                .is_some_and(|c| c.path.ends_with("/embeddings"));
            assert!(output > 0 || embeddings, "{} output", route.model);
            assert!(
                cache_read > 0 && cache_read <= input,
                "{} cache_read {cache_read} vs input {input}",
                route.model
            );
            assert!(cache_write > 0, "{} cache_write", route.model);
        }
    }

    /// The two cards the eval harness already freezes, plus the first-party Claude flagship, so a
    /// refresh of this table cannot silently reprice the models we bill against in tests.
    #[test]
    fn pinned_list_prices_match_the_published_cards() {
        // `unwrap_or` of the first row is only reached if the assert above it failed and
        // didn't abort — the id is in `MODEL_ROUTES`, so the fallback is dead.
        let glm = for_model("z-ai/glm-5.3");
        assert!(glm.is_some(), "glm-5.3");
        let glm = glm.unwrap_or(&MODEL_ROUTES[0]).price;
        assert_eq!(
            (glm.input, glm.cache_read, glm.output),
            ("1.4", "0.26", "4.4")
        );
        let kimi = for_model("moonshotai/kimi-k3");
        assert!(kimi.is_some(), "kimi-k3");
        let kimi = kimi.unwrap_or(&MODEL_ROUTES[0]).price;
        assert_eq!(
            (kimi.input, kimi.cache_read, kimi.output),
            ("3", "0.3", "15")
        );
        let opus = for_model("claude-opus-4-8");
        assert!(opus.is_some(), "claude-opus-4-8");
        let opus = opus.unwrap_or(&MODEL_ROUTES[0]).price;
        assert_eq!(
            (opus.input, opus.output, opus.cache_read, opus.cache_write),
            ("5", "25", "0.5", "6.25")
        );
        let opus55 = for_model("claude-opus-5-5");
        assert!(opus55.is_some(), "claude-opus-5-5");
        let opus55 = opus55.unwrap_or(&MODEL_ROUTES[0]).price;
        assert_eq!(
            (
                opus55.input,
                opus55.output,
                opus55.cache_read,
                opus55.cache_write
            ),
            ("4", "20", "0.2", "5")
        );
    }

    /// `wire_of_path` is what per-candidate usage parsing leans on, so prove it discriminates
    /// rather than always answering the same thing.
    #[test]
    fn wire_of_path_discriminates_messages_from_chat_completions() {
        assert_eq!(wire_of_path("/v1/messages"), WireFormat::Anthropic);
        assert_eq!(wire_of_path("/api/v1/messages"), WireFormat::Anthropic);
        assert_eq!(wire_of_path("/v1/chat/completions"), WireFormat::OpenAi);
        assert_eq!(wire_of_path("/api/v1/chat/completions"), WireFormat::OpenAi);
        assert_eq!(wire_of_path("/v1/responses"), WireFormat::OpenAi);
    }

    /// Claude has two real fallbacks, and they are not the same kind. Bedrock stays Messages.
    /// OpenRouter is the mixed-wire arm: an OpenAI-compat Chat Completions host. The gateway
    /// translates the original client body onto that path; billing follows the serving candidate.
    #[test]
    fn claude_fails_over_to_bedrock_then_openrouter_chat_completions() {
        let want = [
            Candidate {
                provider: ProviderId::Anthropic,
                upstream_model: "claude-opus-4-8",
                path: "/v1/messages",
            },
            Candidate {
                provider: ProviderId::Bedrock,
                upstream_model: "us.anthropic.claude-opus-4-8",
                path: "/anthropic/v1/messages",
            },
            Candidate {
                provider: ProviderId::OpenRouter,
                // OpenRouter spells Claude with a vendor prefix and dots, not dashes.
                upstream_model: "anthropic/claude-opus-4.8",
                path: "/api/v1/chat/completions",
            },
        ];
        assert_eq!(
            for_model("claude-opus-4-8").map(|r| (r.wire, r.candidates)),
            Some((WireFormat::Anthropic, &want[..])),
            "Claude must fail over to Bedrock Messages, then OpenRouter Chat Completions",
        );
        assert_eq!(by_id(ProviderId::Bedrock).wire, WireFormat::Anthropic);
        assert_eq!(by_id(ProviderId::OpenRouter).wire, WireFormat::OpenAi);
        assert_eq!(
            endpoint_of_path(want[0].path),
            "messages",
            "primary stays Messages"
        );
        assert_eq!(
            endpoint_of_path(want[2].path),
            "chat/completions",
            "OpenRouter arm is Chat Completions, not Messages"
        );
    }

    /// The two Claude rows whose Bedrock inference-profile ids were live-verified. Other Claude
    /// rows from the 2026-09 lineup stay on Anthropic → OpenRouter until those ids are checked.
    #[test]
    fn verified_claude_bedrock_rows_fail_over_through_bedrock() {
        for name in ["claude-haiku-4-5", "claude-opus-4-8"] {
            assert!(for_model(name).is_some(), "{name} must be in the catalog");
            if let Some(route) = for_model(name) {
                assert_eq!(route.wire, WireFormat::Anthropic, "{name}");
                assert!(
                    route.candidates.len() >= 3,
                    "{name} must have Anthropic + Bedrock + OpenRouter"
                );
                assert_eq!(
                    route.candidates[0].provider,
                    ProviderId::Anthropic,
                    "{name} primary"
                );
                assert_eq!(
                    route.candidates[1].provider,
                    ProviderId::Bedrock,
                    "{name} second source must be Bedrock, not a proxy"
                );
                assert_eq!(
                    route.candidates[1].path, "/anthropic/v1/messages",
                    "{name} Bedrock path"
                );
                assert_eq!(
                    route.candidates[2].provider,
                    ProviderId::OpenRouter,
                    "{name} third"
                );
            }
        }
    }

    /// Spot-check the current-generation rows: native id matches the catalog name, OpenRouter uses
    /// the vendor-slug + (for Claude) dot-spelled form published on 2026-09-12.
    ///
    /// These flagships still use the two-candidate `claude()` / `openai_chat()` shape. Haiku 4.5
    /// and Opus 4.8 insert Bedrock and are pinned separately.
    #[test]
    fn current_flagships_have_native_plus_openrouter_candidates() {
        let cases = [
            (
                "claude-opus-5",
                WireFormat::Anthropic,
                ProviderId::Anthropic,
                "claude-opus-5",
                "anthropic/claude-opus-5",
            ),
            (
                "claude-fable-5-1",
                WireFormat::Anthropic,
                ProviderId::Anthropic,
                "claude-fable-5-1",
                "anthropic/claude-fable-5.1",
            ),
            (
                "claude-sonnet-5",
                WireFormat::Anthropic,
                ProviderId::Anthropic,
                "claude-sonnet-5",
                "anthropic/claude-sonnet-5",
            ),
            (
                "gpt-6-astra",
                WireFormat::OpenAi,
                ProviderId::OpenAi,
                "gpt-6-astra",
                "openai/gpt-6-astra",
            ),
            (
                "gpt-5.6-sol",
                WireFormat::OpenAi,
                ProviderId::OpenAi,
                "gpt-5.6-sol",
                "openai/gpt-5.6-sol",
            ),
            (
                "grok-4.6",
                WireFormat::OpenAi,
                ProviderId::XAi,
                "grok-4.6",
                "x-ai/grok-4.6",
            ),
            (
                "deepseek-flash",
                WireFormat::OpenAi,
                ProviderId::DeepSeek,
                "deepseek-flash",
                "deepseek/deepseek-v4.1-flash",
            ),
            (
                "mistral-large-latest",
                WireFormat::OpenAi,
                ProviderId::Mistral,
                "mistral-large-latest",
                "mistralai/mistral-large-2512",
            ),
            (
                "qwen/qwen3.8-27b",
                WireFormat::OpenAi,
                ProviderId::Groq,
                "qwen/qwen3.8-27b",
                "qwen/qwen3.8-27b",
            ),
        ];
        for (name, wire, primary, native, openrouter) in cases {
            assert!(for_model(name).is_some(), "{name} must be in the catalog");
            if let Some(row) = for_model(name) {
                assert_eq!(row.wire, wire, "{name}");
                assert_eq!(row.candidates.len(), 2, "{name}");
                assert_eq!(row.candidates[0].provider, primary, "{name}");
                assert_eq!(row.candidates[0].upstream_model, native, "{name}");
                assert_eq!(row.candidates[1].provider, ProviderId::OpenRouter, "{name}");
                assert_eq!(row.candidates[1].upstream_model, openrouter, "{name}");
            }
        }
    }

    /// GPT-OSS 120B is served by Groq, Together and Fireworks. Catalog name is the shared
    /// Groq/Together id; Fireworks keeps `accounts/fireworks/models/gpt-oss-120b` as an alias. No
    /// OpenRouter candidate: one of its hosts answers forced tool calls with an empty
    /// `finish_reason: "error"` 200. Llama 4 Scout advertises no tools: no OpenRouter host serves
    /// them ("Filter by Tool Compatibility removed deepinfra/fp8, novita/bf16").
    /// claim: CAT-6
    /// defect: D115
    #[test]
    fn advertised_tools_have_a_tool_capable_candidate() {
        assert!(
            for_model("openai/gpt-oss-120b").is_some(),
            "openai/gpt-oss-120b must be in the catalog"
        );
        if let Some(row) = for_model("openai/gpt-oss-120b") {
            assert_eq!(row.wire, WireFormat::OpenAi);
            assert_eq!(row.candidates.len(), 3);
            assert_eq!(row.candidates[0].provider, ProviderId::Groq);
            assert_eq!(row.candidates[1].provider, ProviderId::Together);
            assert_eq!(row.candidates[2].provider, ProviderId::Fireworks);
            assert_eq!(
                row.candidates[2].upstream_model,
                "accounts/fireworks/models/gpt-oss-120b"
            );
            assert!(
                row.candidates
                    .iter()
                    .all(|c| c.provider != ProviderId::OpenRouter)
            );
            assert_eq!(
                for_model("accounts/fireworks/models/gpt-oss-120b").map(|r| r.model),
                Some("openai/gpt-oss-120b"),
            );
        }
        let scout = for_model("meta-llama/llama-4-scout");
        assert!(scout.is_some_and(|r| r.card.features & TOOLS == 0));
        assert!(scout.is_some_and(|r| r.card.features & STRUCTURED_OUTPUTS != 0));
    }

    #[test]
    fn kimi_k3_names_together_fireworks_and_openrouter() {
        assert!(
            for_model("moonshotai/kimi-k3").is_some(),
            "moonshotai/kimi-k3 must be in the catalog"
        );
        if let Some(row) = for_model("moonshotai/kimi-k3") {
            assert_eq!(row.candidates.len(), 3);
            assert_eq!(row.candidates[0].provider, ProviderId::Together);
            assert_eq!(row.candidates[0].upstream_model, "moonshotai/Kimi-K3");
            assert_eq!(row.candidates[1].provider, ProviderId::Fireworks);
            assert_eq!(
                row.candidates[1].upstream_model,
                "accounts/fireworks/models/kimi-k3"
            );
            assert_eq!(row.candidates[2].provider, ProviderId::OpenRouter);
        }
    }

    #[test]
    fn glm_5_2_and_minimax_m3_name_together_fireworks_and_openrouter() {
        for (name, together_id, fireworks_id, openrouter) in [
            (
                "z-ai/glm-5.2",
                "zai-org/GLM-5.2",
                "accounts/fireworks/models/glm-5p2",
                "z-ai/glm-5.2",
            ),
            (
                "minimax/minimax-m3",
                "MiniMaxAI/MiniMax-M3",
                "accounts/fireworks/models/minimax-m3",
                "minimax/minimax-m3",
            ),
        ] {
            assert!(for_model(name).is_some(), "{name} must be in the catalog");
            if let Some(row) = for_model(name) {
                assert_eq!(row.candidates.len(), 3, "{name}");
                assert_eq!(row.candidates[0].provider, ProviderId::Together, "{name}");
                assert_eq!(row.candidates[0].upstream_model, together_id, "{name}");
                assert_eq!(row.candidates[1].provider, ProviderId::Fireworks, "{name}");
                assert_eq!(row.candidates[1].upstream_model, fireworks_id, "{name}");
                assert_eq!(row.candidates[2].provider, ProviderId::OpenRouter, "{name}");
                assert_eq!(row.candidates[2].upstream_model, openrouter, "{name}");
            }
        }
    }

    /// Llama 3.3 is served by Together and OpenRouter. Groq's `llama-3.3-70b-versatile` is
    /// Enterprise-only and Fireworks' copy is not serverless, so neither is a candidate. The Groq id
    /// stays the catalog name so clients that send it still resolve.
    #[test]
    fn llama_3_3_names_together_and_openrouter() {
        let row = for_model("llama-3.3-70b-versatile");
        assert!(
            row.is_some(),
            "llama-3.3-70b-versatile must be in the catalog"
        );
        if let Some(row) = row {
            assert_eq!(row.wire, WireFormat::OpenAi);
            assert_eq!(row.candidates.len(), 2);
            assert_eq!(row.candidates[0].provider, ProviderId::Together);
            assert_eq!(
                row.candidates[0].upstream_model,
                "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            );
            assert_eq!(row.candidates[1].provider, ProviderId::OpenRouter);
            assert_eq!(
                for_model("meta-llama/Llama-3.3-70B-Instruct-Turbo").map(|r| r.model),
                Some("llama-3.3-70b-versatile"),
            );
        }
    }

    /// GPT / o-series only. Other OpenAI-wire rows (Grok, DeepSeek, Mistral, llama, qwen) have no
    /// OpenAI store, so a Responses session-state walk must not list an arm.
    #[test]
    fn gpt_rows_carry_an_openai_responses_arm() {
        for route in MODEL_ROUTES.iter().filter(|r| {
            r.candidates.first().is_some_and(|c| {
                c.provider == ProviderId::OpenAi && !c.path.ends_with("/embeddings")
            })
        }) {
            assert_eq!(
                route.responses.len(),
                1,
                "{:?} must list OpenAI /v1/responses",
                route.model
            );
            let c = &route.responses[0];
            assert_eq!(c.provider, ProviderId::OpenAi, "{:?}", route.model);
            assert_eq!(c.path, "/v1/responses", "{:?}", route.model);
            assert_eq!(c.upstream_model, route.model, "{:?}", route.model);
            assert!(
                (1..=MAX_CANDIDATES).contains(&route.responses.len()),
                "{:?}",
                route.model
            );
        }
    }

    #[test]
    fn non_openai_primary_rows_have_no_responses_arm() {
        for route in MODEL_ROUTES.iter().filter(|r| {
            !r.candidates
                .first()
                .is_some_and(|c| c.provider == ProviderId::OpenAi)
        }) {
            assert!(
                route.responses.is_empty(),
                "{:?} is not an OpenAI store; Responses session state must not list an arm",
                route.model
            );
        }
    }

    #[test]
    fn claude_rows_have_no_openai_responses_arm() {
        for route in MODEL_ROUTES
            .iter()
            .filter(|r| r.wire == WireFormat::Anthropic)
        {
            assert!(
                route.responses.is_empty(),
                "{:?} has no OpenAI store; Responses session state must 400, not list an arm",
                route.model
            );
        }
    }

    #[test]
    fn responses_arms_are_routable_absolute_and_openai_wire() {
        for route in MODEL_ROUTES {
            for c in route.responses {
                assert!(
                    gateway_providers().any(|p| p.id == c.provider),
                    "route {:?} responses names {:?}",
                    route.model,
                    c.provider
                );
                assert!(
                    c.path.starts_with('/') && !c.path.contains("://"),
                    "{:?} responses path {:?}",
                    route.model,
                    c.path
                );
                assert_eq!(
                    wire_of_path(c.path),
                    WireFormat::OpenAi,
                    "{:?} responses must be OpenAI-wire",
                    route.model
                );
                assert!(
                    c.path.ends_with("/responses"),
                    "{:?} responses path {:?}",
                    route.model,
                    c.path
                );
                let spec = by_id(c.provider);
                if let Some(base) = spec.base_url {
                    let mount = spec.base_path();
                    assert!(
                        c.path.starts_with(mount),
                        "route {:?}: {} serves from {base} (mount {mount:?}), but responses path is \
                         {:?}",
                        route.model,
                        spec.name,
                        c.path,
                    );
                }
            }
        }
    }

    #[test]
    fn responses_arms_within_a_row_share_one_endpoint() {
        for route in MODEL_ROUTES {
            let endpoint = |p: &str| p.rsplit_once("/v1").map_or(p, |(_, tail)| tail).to_string();
            let Some(first) = route.responses.first() else {
                continue;
            };
            let want = endpoint(first.path);
            for c in route.responses {
                assert_eq!(
                    endpoint(c.path),
                    want,
                    "route {:?}: responses mix {:?} and {:?}",
                    route.model,
                    first.path,
                    c.path
                );
            }
        }
    }

    // --- Vendor truth (`verify/catalog_truth.toml`) ------------------------------------------

    fn truth() -> toml::Table {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../verify/catalog_truth.toml"
        );
        let raw = std::fs::read_to_string(path).expect("read verify/catalog_truth.toml");
        raw.parse::<toml::Table>()
            .expect("verify/catalog_truth.toml parses")
    }

    fn truth_array<'a>(t: &'a toml::Table, key: &str) -> &'a [toml::Value] {
        t.get(key)
            .and_then(toml::Value::as_array)
            .map_or(&[], Vec::as_slice)
    }

    /// A USD decimal (`"0.30"`, `"12.5"`) as micro-dollars, so `"0.30"` equals `"0.3"`.
    fn usd_micros(raw: &str) -> u64 {
        let (whole, frac) = raw.split_once('.').unwrap_or((raw, ""));
        assert!(frac.len() <= 6, "{raw}: more than 6 decimal places");
        let frac = format!("{frac:0<6}");
        whole.parse::<u64>().expect(raw) * 1_000_000 + frac.parse::<u64>().expect(raw)
    }

    fn exact_row(model: &str) -> Option<&'static ModelRoute> {
        MODEL_ROUTES.iter().find(|r| r.model == model)
    }

    fn bits_of(names: &[toml::Value], table: &[(u8, &str)], model: &str) -> u8 {
        names.iter().fold(0, |acc, n| {
            let n = n.as_str().expect("bit names are strings");
            let bit = table.iter().find(|(_, s)| *s == n).map(|(b, _)| *b);
            assert!(bit.is_some(), "{model}: unknown bit name {n:?}");
            acc | bit.unwrap_or(0)
        })
    }

    /// Every list price and card size the primary vendor publishes, as recorded with its source URL
    /// in `verify/catalog_truth.toml`, equals the catalog row. A rate the vendor does not publish is
    /// the input rate (the `ListPrice` rule). And no direct-vendor row escapes the check: each has a
    /// recorded entry, or an `unverified` entry that says why not.
    /// claim: CAT-7, CAT-3, CAT-4
    /// defect: D60
    #[test]
    fn catalog_matches_vendor_truth() {
        let t = truth();
        let mut recorded = std::collections::BTreeSet::new();
        for entry in truth_array(&t, "row") {
            let model = entry["model"].as_str().expect("row.model");
            assert!(recorded.insert(model), "{model}: recorded twice");
            let sources = truth_array(entry.as_table().expect("row is a table"), "source");
            assert!(
                !sources.is_empty()
                    && sources
                        .iter()
                        .all(|s| s.as_str().is_some_and(|s| s.starts_with("https://"))),
                "{model}: needs https source URLs"
            );
            assert!(
                entry.get("date").and_then(toml::Value::as_str).is_some(),
                "{model}: needs a date"
            );
            let row = exact_row(model);
            assert!(row.is_some(), "{model}: recorded but not a catalog row");
            let Some(row) = row else { continue };
            // An OpenRouter-primary row is recorded only from its maker's own pages.
            assert!(
                sources
                    .iter()
                    .all(|s| s.as_str().is_some_and(|s| !s.contains("openrouter.ai"))),
                "{model}: vendor truth never comes from OpenRouter"
            );
            let p = row.price;
            let rate = |k: &str| entry.get(k).and_then(toml::Value::as_str).map(usd_micros);
            for (k, ours) in [
                ("input", p.input),
                ("output", p.output),
                ("cache_read", p.cache_read),
                ("cache_write", p.cache_write),
            ] {
                if let Some(vendor) = rate(k) {
                    assert_eq!(usd_micros(ours), vendor, "{model} {k}: ours {ours}");
                } else if k.starts_with("cache") && rate("input").is_some() {
                    assert_eq!(
                        usd_micros(ours),
                        usd_micros(p.input),
                        "{model} {k}: unpublished, so it must equal input"
                    );
                }
            }
            let int = |k: &str| entry.get(k).and_then(toml::Value::as_integer);
            // The card states the limit every candidate enforces: the vendor's figure, or the
            // lower one recorded beside it with its evidence.
            for (published, limit, card) in [
                (
                    "context_window",
                    "input_limit",
                    i64::from(row.card.context_window),
                ),
                (
                    "max_output_tokens",
                    "output_limit",
                    i64::from(row.card.max_output_tokens),
                ),
            ] {
                if let Some(l) = int(limit) {
                    assert!(
                        entry
                            .get("limit_evidence")
                            .and_then(toml::Value::as_str)
                            .is_some_and(|e| !e.trim().is_empty()),
                        "{model}: {limit} needs limit_evidence"
                    );
                    assert!(
                        int(published).is_none_or(|v| l < v),
                        "{model}: {limit} {l} is not below {published}"
                    );
                    assert_eq!(card, l, "{model} card {published} must be its {limit}");
                } else if let Some(v) = int(published) {
                    assert_eq!(card, v, "{model} {published}");
                }
            }
            if let Some(v) = int("created") {
                assert_eq!(
                    i64::try_from(row.card.created).ok(),
                    Some(v),
                    "{model} created"
                );
            }
            if let Some(v) = entry.get("owned_by").and_then(toml::Value::as_str) {
                assert_eq!(row.card.owned_by, v, "{model} owned_by");
            }
        }
        let mut unverified = std::collections::BTreeSet::new();
        for u in truth_array(&t, "unverified") {
            let model = u["model"].as_str().expect("unverified.model");
            let reason = u.get("reason").and_then(toml::Value::as_str).unwrap_or("");
            assert!(
                !reason.trim().is_empty(),
                "{model}: unverified needs a reason"
            );
            assert!(
                exact_row(model).is_some(),
                "{model}: unverified but not a row"
            );
            assert!(
                !recorded.contains(model),
                "{model}: both recorded and unverified"
            );
            unverified.insert(model);
        }
        for r in MODEL_ROUTES {
            if r.candidates[0].provider == ProviderId::OpenRouter {
                continue;
            }
            assert!(
                recorded.contains(r.model) || unverified.contains(r.model),
                "{}: direct-vendor primary with no entry in verify/catalog_truth.toml",
                r.model
            );
        }
    }

    /// No candidate (or Responses arm) names an id its vendor has retired or does not serve to a
    /// standard serverless key, and no row is named after a retired id. Such a candidate fails over
    /// on every request; a retired name served by a fallback is a different model under that name.
    /// claim: CAT-1, CAT-2
    /// defect: D61, D108
    #[test]
    fn no_candidate_is_retired_or_not_serverless() {
        let t = truth();
        let mut dead = std::collections::BTreeSet::new();
        let mut retired_ids = std::collections::BTreeSet::new();
        for (list, retired) in [("retired", true), ("not_serverless", false)] {
            for e in truth_array(&t, list) {
                let provider = e["provider"].as_str().expect("provider");
                let id = e["id"].as_str().expect("id");
                assert!(
                    e.get("source")
                        .and_then(toml::Value::as_str)
                        .is_some_and(|s| s.starts_with("https://")),
                    "{list} {provider}/{id}: needs a source URL"
                );
                assert!(
                    gateway_providers().any(|p| p.name == provider),
                    "{list} {provider}/{id}: unknown provider"
                );
                dead.insert((provider, id));
                if retired {
                    retired_ids.insert(id);
                }
            }
        }
        assert!(!dead.is_empty(), "the truth file lists no dead ids");
        for r in MODEL_ROUTES {
            assert!(
                !retired_ids.contains(r.model),
                "{}: row is named after a retired model",
                r.model
            );
            for c in r.candidates.iter().chain(r.responses) {
                let key = (by_id(c.provider).name, c.upstream_model);
                assert!(
                    !dead.contains(&key),
                    "{}: candidate {}/{} is retired or not serverless",
                    r.model,
                    key.0,
                    key.1
                );
            }
        }
    }

    /// deepseek-v4-pro's failover is the same snapshot (V4-Pro-0813 on Together), not OpenRouter's
    /// 0423. The rows whose fallbacks served V3 and R1 under DeepSeek's retired names are gone.
    /// claim: CAT-2
    /// defect: D18
    #[test]
    fn deepseek_v4_pro_fails_over_to_the_same_snapshot() {
        let row = for_model("deepseek-v4-pro");
        assert!(row.is_some(), "deepseek-v4-pro must be in the catalog");
        if let Some(row) = row {
            let got: Vec<_> = row
                .candidates
                .iter()
                .map(|c| (c.provider, c.upstream_model))
                .collect();
            assert_eq!(
                got,
                [
                    (ProviderId::DeepSeek, "deepseek-v4-pro"),
                    (ProviderId::Together, "deepseek-ai/DeepSeek-V4-Pro-0813"),
                ]
            );
            assert_eq!(row.card.name, "DeepSeek V4 Pro 0813");
        }
        for gone in ["deepseek-chat", "deepseek-reasoner", "mistral-nemo"] {
            assert!(for_model(gone).is_none(), "{gone} is retired at its vendor");
        }
    }

    /// The D19 examples: a direct-vendor primary is priced at that vendor's standard rate, not
    /// OpenRouter's cheapest host and not an off-peak or base-model rate.
    /// claim: CAT-7
    /// defect: D19
    #[test]
    fn direct_primaries_are_priced_at_their_vendor_rate() {
        for (model, primary, want) in [
            // Together's Llama 3.3 70B Turbo (Groq's id is Enterprise-only); was 0.1 / 0.32.
            (
                "llama-3.3-70b-versatile",
                ProviderId::Together,
                ("1.04", "1.04", "1.04"),
            ),
            // DeepSeek's peak rate; was 0.95526 / 1.91052, matching neither tier.
            (
                "deepseek-v4-pro",
                ProviderId::DeepSeek,
                ("1.32", "3.96", "0.044"),
            ),
            // Groq's own rates; were OpenRouter's cheapest host (0.018 / 0.09, 0.037 / 0.17).
            (
                "openai/gpt-oss-20b",
                ProviderId::Groq,
                ("0.075", "0.3", "0.0375"),
            ),
            (
                "openai/gpt-oss-120b",
                ProviderId::Groq,
                ("0.15", "0.6", "0.075"),
            ),
        ] {
            let row = for_model(model);
            assert!(row.is_some(), "{model}");
            if let Some(row) = row {
                assert_eq!(row.candidates[0].provider, primary, "{model}");
                let p = row.price;
                assert_eq!((p.input, p.output, p.cache_read), want, "{model}");
            }
        }
    }

    /// Input and capability bits agree with what the primary vendor lists, wherever the truth file
    /// records it: a bit the vendor does not list is not advertised, and one it does list is.
    /// claim: CAT-6
    /// defect: D62
    #[test]
    fn capability_bits_match_vendor_truth() {
        let t = truth();
        let mut checked = 0;
        for entry in truth_array(&t, "row") {
            let model = entry["model"].as_str().expect("row.model");
            let Some(row) = exact_row(model) else {
                continue;
            };
            let e = entry.as_table().expect("row is a table");
            for (key, table, have, present) in [
                ("input_present", &INPUT_NAMES[..], row.card.input, true),
                ("input_absent", &INPUT_NAMES[..], row.card.input, false),
                (
                    "features_present",
                    &FEATURE_NAMES[..],
                    row.card.features,
                    true,
                ),
                (
                    "features_absent",
                    &FEATURE_NAMES[..],
                    row.card.features,
                    false,
                ),
            ] {
                let names = truth_array(e, key);
                if names.is_empty() {
                    continue;
                }
                checked += 1;
                let bits = bits_of(names, table, model);
                if present {
                    assert_eq!(have & bits, bits, "{model}: {key} {names:?}");
                } else {
                    assert_eq!(have & bits, 0, "{model}: {key} {names:?}");
                }
            }
        }
        assert!(checked > 0, "the truth file records no capability bits");
    }

    /// The D54 cards: gpt-4 claims no structured outputs, deepseek-flash keeps image input (DeepSeek
    /// lists vision; D54's suspicion was wrong), and no row advertises a max output that is
    /// OpenRouter's 0.9x / 0.8x-of-window filler.
    /// claim: CAT-6, CAT-4
    /// defect: D54
    #[test]
    fn suspect_cards_are_corrected() {
        let gpt4 = for_model("gpt-4").map(|r| r.card);
        assert!(gpt4.is_some(), "gpt-4");
        if let Some(c) = gpt4 {
            assert_eq!(c.features & STRUCTURED_OUTPUTS, 0);
        }
        for m in ["gpt-4-turbo", "gpt-5.2-pro", "gpt-5.4-pro"] {
            let f = for_model(m).map(|r| r.card.features);
            assert!(f.is_some_and(|f| f & STRUCTURED_OUTPUTS == 0), "{m}");
        }
        let flash = for_model("deepseek-flash").map(|r| r.card);
        assert!(
            flash.is_some_and(|c| c.input & IN_IMAGE != 0),
            "deepseek-flash reads images"
        );
        assert!(
            flash.is_some_and(|c| c.max_output_tokens == 384_000),
            "deepseek-flash max output"
        );
        for r in MODEL_ROUTES {
            let ctx = u64::from(r.card.context_window);
            let max = u64::from(r.card.max_output_tokens);
            for (num, den) in [(9, 10), (8, 10)] {
                assert!(
                    max.abs_diff(ctx * num / den) > 1,
                    "{}: max output {max} is {num}/{den} of the {ctx} window",
                    r.model
                );
            }
        }
    }

    /// OpenAI's GPT-5 family caps input at the window less the max output, and says so ("Input
    /// tokens exceed the configured limit of 272000 tokens"): a card that listed the whole window
    /// sent clients sizing prompts from `/v1/models` into a 400 between the two numbers. Every
    /// GPT-5-family row served by OpenAI (directly or through OpenRouter) states the cap.
    /// claim: CAT-3
    /// defect: D104
    #[test]
    fn gpt5_cards_state_openais_input_cap() {
        for r in MODEL_ROUTES {
            let Some(rest) = r.model.strip_prefix("gpt-") else {
                continue;
            };
            if !(rest.starts_with('5') || rest.starts_with('6')) {
                continue;
            }
            let c = r.card;
            assert!(
                matches!(c.context_window, 128_000 | 272_000 | 922_000),
                "{}: context_window {} is not OpenAI's input cap",
                r.model,
                c.context_window
            );
            assert!(
                matches!(
                    u64::from(c.context_window) + u64::from(c.max_output_tokens),
                    400_000 | 1_050_000
                ),
                "{}: input cap {} plus max output {} is not OpenAI's window",
                r.model,
                c.context_window,
                c.max_output_tokens
            );
        }
    }

    /// A failover host that serves a smaller window than the primary sets the row's card: the
    /// walk may land there, and a prompt sized to the larger figure would be refused.
    /// claim: CAT-3
    /// defect: D110
    #[test]
    fn cards_state_a_failover_hosts_smaller_window() {
        for (model, window) in [
            ("ministral-3b-latest", 131_072),
            ("qwen/qwen3.8-2.4t-a95b", 1_010_000),
        ] {
            assert_eq!(
                for_model(model).map(|r| r.card.context_window),
                Some(window),
                "{model}"
            );
        }
    }

    /// gpt-4's 8,192-token window holds prompt and output together, so the vendor's 8,192 max
    /// output was refused with any prompt at all. The card states a max output a prompt fits
    /// beside, and the smaller window its OpenRouter candidate enforces.
    /// claim: CAT-4, CAT-3
    /// defect: D111
    #[test]
    fn gpt4_max_output_leaves_room_for_a_prompt() {
        let c = for_model("gpt-4").map(|r| r.card);
        assert_eq!(
            c.map(|c| (c.context_window, c.max_output_tokens)),
            Some((8_191, 4_096))
        );
    }

    /// gpt-4 calls tools on every candidate (OpenAI Chat and Responses, OpenRouter), though its
    /// model page lists none: the card lists what is served.
    /// claim: CAT-6
    /// defect: D113
    #[test]
    fn gpt4_card_lists_the_tools_it_serves() {
        let f = for_model("gpt-4").map(|r| r.card.features);
        assert_eq!(f, Some(TOOLS));
    }

    /// xAI serves multi-agent models on Responses only, and refuses their client-side tools
    /// without beta access: the row's xAI candidate is a Responses path and the card has no tools.
    /// claim: CAT-1, CAT-6
    /// defect: D105
    #[test]
    fn multi_agent_reaches_xai_over_responses() {
        let row = for_model("grok-4.20-multi-agent");
        assert!(row.is_some());
        if let Some(row) = row {
            assert_eq!(row.candidates[0].provider, ProviderId::XAi);
            assert_eq!(row.candidates[0].path, "/v1/responses");
            assert_eq!(endpoint_of_path(row.candidates[0].path), "responses");
            assert!(row.responses.is_empty(), "xAI's store is not OpenAI's");
            assert_eq!(row.card.features & TOOLS, 0);
        }
    }

    /// xAI's Chat Completions answers file content with 400 "File content is not supported on
    /// /v1/chat/completions. Please use /v1/responses instead.", so a row that reaches xAI there
    /// cannot advertise file input.
    /// claim: CAT-5
    /// defect: D106
    #[test]
    fn no_row_advertises_files_over_xai_chat_completions() {
        let mut grok = 0;
        for r in MODEL_ROUTES {
            for c in r.candidates {
                if c.provider == ProviderId::XAi && endpoint_of_path(c.path) == "chat/completions" {
                    grok += 1;
                    assert_eq!(r.card.input & IN_FILE, 0, "{}", r.model);
                }
            }
        }
        assert!(grok >= 5, "the grok rows are still on xAI Chat Completions");
    }

    /// OpenAI's Chat Completions refuses function tools with any reasoning effort on GPT-5.4 and
    /// later ("use /v1/responses or set reasoning_effort to 'none'"), and GPT-5.6 and GPT-6 Astra
    /// reason by default, so every tool call on their Chat primary failed. Those rows reach OpenAI
    /// over Responses, which takes both; their Responses arm is unchanged.
    /// claim: CAT-6
    /// defect: D114
    #[test]
    fn gpt_5_4_and_later_reach_openai_over_responses() {
        let mut checked = 0;
        for r in MODEL_ROUTES {
            let late = ["gpt-5.4", "gpt-5.5", "gpt-5.6", "gpt-6"]
                .iter()
                .any(|p| r.model.starts_with(p));
            if !late {
                continue;
            }
            for c in r
                .candidates
                .iter()
                .filter(|c| c.provider == ProviderId::OpenAi)
            {
                checked += 1;
                assert_eq!(c.path, "/v1/responses", "{}", r.model);
            }
        }
        assert!(checked >= 8, "only {checked} GPT-5.4+ OpenAI candidates");
    }

    /// Amazon Bedrock's Messages surface refuses `output_config.format`: a structured-output request
    /// must skip it. Every other candidate type honors the constraint.
    /// claim: CAT-6
    #[test]
    fn only_bedrock_refuses_structured_outputs() {
        for r in MODEL_ROUTES {
            for c in r.candidates.iter().chain(r.responses) {
                assert_eq!(
                    serves_structured_outputs(c),
                    c.provider != ProviderId::Bedrock,
                    "{}",
                    r.model
                );
            }
        }
    }

    /// Prices are the model maker's own published rate, never OpenRouter's listing (its cheapest
    /// host), even on an OpenRouter-primary row whose maker sells an API. Together's Qwen3.8 Flash
    /// output is $0.282 (docs table and `/v1/models`; the pricing page rounds).
    /// claim: CAT-7
    /// defect: D116
    #[test]
    fn prices_follow_the_maker_not_openrouter() {
        for (model, want) in [
            ("z-ai/glm-5.1", ("1.4", "4.4", "0.26")),
            ("moonshotai/kimi-k2.7-code", ("0.95", "4", "0.19")),
            ("moonshotai/kimi-k2.6", ("0.95", "4", "0.16")),
            ("minimaxai/minimax-m2.7", ("0.3", "1.2", "0.06")),
            ("qwen/qwen3.8-flash", ("0.09", "0.282", "0.09")),
            // OpenAI's standard rate, which it bills the pool key; the promotion is recorded apart.
            ("gpt-5.6-sol", ("4", "20", "0.4")),
        ] {
            let p = for_model(model).map(|r| r.price);
            assert_eq!(
                p.map(|p| (p.input, p.output, p.cache_read)),
                Some(want),
                "{model}"
            );
        }
        let t = truth();
        for promo in truth_array(&t, "promo") {
            let model = promo["model"].as_str().expect("promo.model");
            let row = exact_row(model);
            assert!(
                row.is_some(),
                "{model}: promo for a row that is not in the catalog"
            );
            let Some(row) = row else { continue };
            let until = promo["until"].as_str().expect("promo.until");
            assert!(
                until.len() == 10 && until.as_bytes()[4] == b'-',
                "{model}: until {until:?}"
            );
            for (k, card) in [("input", row.price.input), ("output", row.price.output)] {
                let rate = promo[k].as_str().map(usd_micros).expect("promo rate");
                assert!(
                    rate < usd_micros(card),
                    "{model}: a promo {k} rate is a discount on the card"
                );
            }
        }
    }

    /// `created` and `owned_by` are the vendor's own listing (OpenAI's and xAI's `/v1/models`), not
    /// the day OpenRouter listed the model or OpenRouter's `x-ai` slug.
    /// claim: CAT-8
    /// defect: D117
    #[test]
    fn cards_carry_the_vendors_created_and_owner() {
        let t = truth();
        let recorded = |model: &str| {
            truth_array(&t, "row")
                .iter()
                .find(|e| e["model"].as_str() == Some(model))
                .and_then(|e| e.get("created"))
                .and_then(toml::Value::as_integer)
        };
        for r in MODEL_ROUTES {
            match r.candidates[0].provider {
                ProviderId::OpenAi | ProviderId::XAi => {
                    assert_eq!(
                        recorded(r.model),
                        i64::try_from(r.card.created).ok(),
                        "{}: created must be the vendor listing's",
                        r.model
                    );
                }
                _ => {}
            }
            if r.candidates[0].provider == ProviderId::XAi {
                assert_eq!(r.card.owned_by, "xai", "{}", r.model);
            }
        }
    }
}
