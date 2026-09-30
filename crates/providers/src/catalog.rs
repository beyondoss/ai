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
//! GPT rows also list a parallel [`ModelRoute::responses`] arm (OpenAI `/v1/responses`) used when
//! the inbound path is Responses **and** the body uses session state; Chat Completions / Messages
//! inbound, and `store: false` one-shot Responses, still walk [`ModelRoute::candidates`].
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
    /// client body onto each candidate's path. Inbound `/v1/responses` without session state is
    /// translated onto Chat Completions or Messages according to this field for the primary, then
    /// per candidate. Session-state Responses walks [`Self::responses`] instead.
    /// `/{provider}/…` never translates.
    pub wire: WireFormat,
    /// Preference order: `[0]` is primary, the rest are failover candidates. Non-empty, at most
    /// [`MAX_CANDIDATES`], no provider repeated. Chat Completions / Messages inbound, and one-shot
    /// Responses (`store: false`) that may still translate onto Chat Completions, walk this list.
    pub candidates: &'static [Candidate],
    /// OpenAI `/v1/responses` arms. Walked when the inbound path is Responses **and** the body uses
    /// session state (`previous_response_id` set, or `store` not explicitly `false`). Empty on
    /// Claude rows — those have no OpenAI store, so session-state Responses is a 400, not a hollow
    /// Messages call. Same-endpoint: byte relay (`store` / `previous_response_id` / `include` /
    /// `truncation` pass through). A Responses 5xx may walk another entry here; it must not walk
    /// onto [`Self::candidates`] while session fields are in play.
    pub responses: &'static [Candidate],
    /// Standard public list price for this model. See [`ListPrice`].
    pub price: ListPrice,
}

/// Standard list price, USD per million tokens.
///
/// Decimal strings, not `f64`: `0.075` is not binary-exact, and these bytes are copied into
/// `GET /v1/models`. At most six digits after the point (one micro-dollar). The four rates are the
/// standard card only — not batch, not fast mode, not a long-context override, and not the 1-hour
/// Claude cache write (2× input). `cache_write` here is the 5-minute / default write rate.
///
/// Taken from OpenRouter's public `https://openrouter.ai/api/v1/models` card on 2026-09-30, which
/// matched Anthropic's first-party table on every Claude row OpenRouter still lists. Two retired
/// Claude rows OpenRouter has dropped (`claude-3-haiku`, `claude-opus-4`) use Anthropic's published
/// card instead (`claude-opus-4` is still on the first-party pricing page; Haiku 3 keeps the
/// long-standing $0.25 / $1.25 card with the same 0.1× read and 1.25× write multipliers as the
/// rest of the Claude table).
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

/// Chat Completions vs Messages vs Responses, from a candidate path.
///
/// Distinct from [`wire_of_path`]: `/v1/chat/completions` and `/v1/responses` are both OpenAI-wire
/// but different endpoints. The gateway translates when this candidate's path differs from the
/// client's; it never sends a Messages body at Chat Completions (or the reverse).
pub fn endpoint_of_path(path: &str) -> &'static str {
    if path.ends_with("/messages") {
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
/// OpenAI-native primary, OpenRouter Chat Completions failover. Same Chat Completions mount on
/// both sides (`/v1` vs `/api/v1`). The matching Responses arm lives on [`ModelRoute::responses`]
/// rather than here: mixing `/v1/chat/completions` and `/v1/responses` in one walk would break
/// `stream_options.include_usage` injection. `store: false` one-shot inbound Responses still
/// translates onto this Chat Completions row.
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

const fn fireworks(native: &'static str, openrouter: &'static str) -> [Candidate; 2] {
    compat_chat(
        ProviderId::Fireworks,
        native,
        "/inference/v1/chat/completions",
        openrouter,
    )
}

/// Groq + Together + Fireworks + OpenRouter for the same Llama 3.3 70B instruct. Canonical name is
/// the Groq id people send; the other three are that vendor's own spelling of the same model.
const fn llama_3_3() -> [Candidate; 4] {
    [
        Candidate {
            provider: ProviderId::Groq,
            upstream_model: "llama-3.3-70b-versatile",
            path: "/openai/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::Together,
            upstream_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            path: "/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::Fireworks,
            upstream_model: "accounts/fireworks/models/llama-v3p3-70b-instruct",
            path: "/inference/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: "meta-llama/llama-3.3-70b-instruct",
            path: "/api/v1/chat/completions",
        },
    ]
}

/// Groq + Together + OpenRouter for GPT-OSS 20B. Same shared id on every host; Fireworks was not
/// live-verified for the 20B spelling the way 120B was.
const fn gpt_oss_20b() -> [Candidate; 3] {
    [
        Candidate {
            provider: ProviderId::Groq,
            upstream_model: "openai/gpt-oss-20b",
            path: "/openai/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::Together,
            upstream_model: "openai/gpt-oss-20b",
            path: "/v1/chat/completions",
        },
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: "openai/gpt-oss-20b",
            path: "/api/v1/chat/completions",
        },
    ]
}

/// Groq + Together + Fireworks + OpenRouter for GPT-OSS 120B. Catalog name is the shared
/// `openai/gpt-oss-120b` Groq/Together/OpenRouter id; Fireworks keeps its own spelling as an alias.
/// Cerebras's bare `gpt-oss-120b` is not listed — `for_model_id` prefix-matches `gpt-` to OpenAI.
const fn gpt_oss_120b() -> [Candidate; 4] {
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
        Candidate {
            provider: ProviderId::OpenRouter,
            upstream_model: "openai/gpt-oss-120b",
            path: "/api/v1/chat/completions",
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
        model: "claude-3-haiku",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-3-haiku", "anthropic/claude-3-haiku"),
        responses: &[],
        price: price("0.25", "1.25", "0.025", "0.3125"),
    },
    ModelRoute {
        model: "claude-fable-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-fable-5", "anthropic/claude-fable-5"),
        responses: &[],
        price: price("10", "50", "1", "12.5"),
    },
    ModelRoute {
        model: "claude-fable-5-1",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-fable-5-1", "anthropic/claude-fable-5.1"),
        responses: &[],
        price: price("10", "50", "0.25", "12.5"),
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
    },
    ModelRoute {
        model: "claude-opus-4",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-4", "anthropic/claude-opus-4"),
        responses: &[],
        price: price("15", "75", "1.5", "18.75"),
    },
    ModelRoute {
        model: "claude-opus-4-1",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-4-1", "anthropic/claude-opus-4.1"),
        responses: &[],
        price: price("15", "75", "1.5", "18.75"),
    },
    ModelRoute {
        model: "claude-opus-4-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-4-5", "anthropic/claude-opus-4.5"),
        responses: &[],
        price: price("5", "25", "0.5", "6.25"),
    },
    ModelRoute {
        model: "claude-opus-4-6",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-4-6", "anthropic/claude-opus-4.6"),
        responses: &[],
        price: price("5", "25", "0.5", "6.25"),
    },
    ModelRoute {
        model: "claude-opus-4-7",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-4-7", "anthropic/claude-opus-4.7"),
        responses: &[],
        price: price("5", "25", "0.5", "6.25"),
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
    },
    ModelRoute {
        model: "claude-opus-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-opus-5", "anthropic/claude-opus-5"),
        responses: &[],
        price: price("5", "25", "0.5", "6.25"),
    },
    ModelRoute {
        model: "claude-sonnet-4",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-sonnet-4", "anthropic/claude-sonnet-4"),
        responses: &[],
        price: price("3", "15", "0.3", "3.75"),
    },
    ModelRoute {
        model: "claude-sonnet-4-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-sonnet-4-5", "anthropic/claude-sonnet-4.5"),
        responses: &[],
        price: price("3", "15", "0.3", "3.75"),
    },
    ModelRoute {
        model: "claude-sonnet-4-6",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-sonnet-4-6", "anthropic/claude-sonnet-4.6"),
        responses: &[],
        price: price("3", "15", "0.3", "3.75"),
    },
    ModelRoute {
        model: "claude-sonnet-5",
        wire: WireFormat::Anthropic,
        candidates: &claude("claude-sonnet-5", "anthropic/claude-sonnet-5"),
        responses: &[],
        price: price("2", "10", "0.2", "2.5"),
    },
    // Mistral `-latest` aliases (GA only). Magistral and Devstral are retired as of 2026-09;
    // a guessed still-served alias 404s and looks like the client's fault.
    ModelRoute {
        model: "codestral-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("codestral-latest", "mistralai/codestral-2508"),
        responses: &[],
        price: price("0.3", "0.9", "0.03", "0.3"), // cache_write unpublished; equals input
    },
    // DeepSeek. Official current names are `deepseek-flash` / `deepseek-v4-pro`. `deepseek-chat`
    // and `deepseek-reasoner` are the ids stock SDKs still send; OpenRouter still lists the chat
    // slug. Reasoner's OpenRouter arm is `deepseek/deepseek-r1` — they never published
    // `deepseek/deepseek-reasoner`.
    ModelRoute {
        model: "deepseek-chat",
        wire: WireFormat::OpenAi,
        candidates: &deepseek("deepseek-chat", "deepseek/deepseek-chat"),
        responses: &[],
        price: price("0.2574", "1.0287", "0.2574", "0.2574"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "deepseek-flash",
        wire: WireFormat::OpenAi,
        candidates: &deepseek("deepseek-flash", "deepseek/deepseek-v4.1-flash"),
        responses: &[],
        price: price("0.3", "1.2", "0.006", "0.3"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "deepseek-reasoner",
        wire: WireFormat::OpenAi,
        candidates: &deepseek("deepseek-reasoner", "deepseek/deepseek-r1"),
        responses: &[],
        price: price("0.7", "2.5", "0.7", "0.7"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "deepseek-v4-pro",
        wire: WireFormat::OpenAi,
        candidates: &deepseek("deepseek-v4-pro", "deepseek/deepseek-v4-pro"),
        responses: &[],
        price: price("0.95526", "1.91052", "0.079605", "0.95526"), // cache_write unpublished; equals input
    },
    // Gemma 4 on Together (vision table, `google/gemma-4-31B-it`) with OpenRouter failover.
    // Not a Gemini dialect — Chat Completions like every other third-party row.
    ModelRoute {
        model: "google/gemma-4-31b-it",
        wire: WireFormat::OpenAi,
        candidates: &together("google/gemma-4-31B-it", "google/gemma-4-31b-it"),
        responses: &[],
        price: price("0.09", "0.34", "0.05", "0.09"), // cache_write unpublished; equals input
    },
    // The same shape on the OpenAI wire, where the two mounts differ as well (`/v1` vs `/api/v1`).
    // Flagships first in the *id* sort: 4.x, then 5 / 5.4 / 5.5 / 5.6, then 6 Astra, then o-series.
    // `responses` is the arm used when inbound is `/v1/responses` with session state; Chat
    // Completions / Messages inbound still walks `candidates`.
    ModelRoute {
        model: "gpt-4",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4", "openai/gpt-4"),
        responses: &openai_responses("gpt-4"),
        price: price("30", "60", "30", "30"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "gpt-4-turbo",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4-turbo", "openai/gpt-4-turbo"),
        responses: &openai_responses("gpt-4-turbo"),
        price: price("10", "30", "10", "10"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "gpt-4.1",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4.1", "openai/gpt-4.1"),
        responses: &openai_responses("gpt-4.1"),
        price: price("2", "8", "0.5", "2"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-4.1-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4.1-mini", "openai/gpt-4.1-mini"),
        responses: &openai_responses("gpt-4.1-mini"),
        price: price("0.4", "1.6", "0.1", "0.4"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-4.1-nano",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4.1-nano", "openai/gpt-4.1-nano"),
        responses: &openai_responses("gpt-4.1-nano"),
        price: price("0.1", "0.4", "0.025", "0.1"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-4o",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4o", "openai/gpt-4o"),
        responses: &openai_responses("gpt-4o"),
        price: price("2.5", "10", "1.25", "2.5"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-4o-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-4o-mini", "openai/gpt-4o-mini"),
        responses: &openai_responses("gpt-4o-mini"),
        price: price("0.15", "0.6", "0.075", "0.15"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5", "openai/gpt-5"),
        responses: &openai_responses("gpt-5"),
        price: price("1.25", "10", "0.125", "1.25"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5-mini", "openai/gpt-5-mini"),
        responses: &openai_responses("gpt-5-mini"),
        price: price("0.25", "2", "0.025", "0.25"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5-nano",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5-nano", "openai/gpt-5-nano"),
        responses: &openai_responses("gpt-5-nano"),
        price: price("0.05", "0.4", "0.005", "0.05"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5-pro", "openai/gpt-5-pro"),
        responses: &openai_responses("gpt-5-pro"),
        price: price("15", "120", "15", "15"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "gpt-5.1",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.1", "openai/gpt-5.1"),
        responses: &openai_responses("gpt-5.1"),
        price: price("1.25", "10", "0.125", "1.25"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.1-codex",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.1-codex", "openai/gpt-5.1-codex"),
        responses: &openai_responses("gpt-5.1-codex"),
        price: price("1.25", "10", "0.13", "1.25"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.1-codex-max",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.1-codex-max", "openai/gpt-5.1-codex-max"),
        responses: &openai_responses("gpt-5.1-codex-max"),
        price: price("1.25", "10", "0.125", "1.25"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.1-codex-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.1-codex-mini", "openai/gpt-5.1-codex-mini"),
        responses: &openai_responses("gpt-5.1-codex-mini"),
        price: price("0.25", "2", "0.03", "0.25"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.2",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.2", "openai/gpt-5.2"),
        responses: &openai_responses("gpt-5.2"),
        price: price("1.75", "14", "0.175", "1.75"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.2-chat",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.2-chat", "openai/gpt-5.2-chat"),
        responses: &openai_responses("gpt-5.2-chat"),
        price: price("1.75", "14", "0.175", "1.75"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.2-codex",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.2-codex", "openai/gpt-5.2-codex"),
        responses: &openai_responses("gpt-5.2-codex"),
        price: price("1.75", "14", "0.175", "1.75"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.2-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.2-pro", "openai/gpt-5.2-pro"),
        responses: &openai_responses("gpt-5.2-pro"),
        price: price("21", "168", "21", "21"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "gpt-5.3-codex",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.3-codex", "openai/gpt-5.3-codex"),
        responses: &openai_responses("gpt-5.3-codex"),
        price: price("1.75", "14", "0.175", "1.75"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.4",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.4", "openai/gpt-5.4"),
        responses: &openai_responses("gpt-5.4"),
        price: price("2.5", "15", "0.25", "2.5"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.4-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.4-mini", "openai/gpt-5.4-mini"),
        responses: &openai_responses("gpt-5.4-mini"),
        price: price("0.75", "4.5", "0.075", "0.75"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.4-nano",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.4-nano", "openai/gpt-5.4-nano"),
        responses: &openai_responses("gpt-5.4-nano"),
        price: price("0.2", "1.25", "0.02", "0.2"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.4-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.4-pro", "openai/gpt-5.4-pro"),
        responses: &openai_responses("gpt-5.4-pro"),
        price: price("30", "180", "30", "30"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "gpt-5.5",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.5", "openai/gpt-5.5"),
        responses: &openai_responses("gpt-5.5"),
        price: price("5", "30", "0.5", "5"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "gpt-5.5-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.5-pro", "openai/gpt-5.5-pro"),
        responses: &openai_responses("gpt-5.5-pro"),
        price: price("30", "180", "30", "30"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "gpt-5.6-luna",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.6-luna", "openai/gpt-5.6-luna"),
        responses: &openai_responses("gpt-5.6-luna"),
        price: price("0.2", "1.2", "0.02", "0.25"),
    },
    ModelRoute {
        model: "gpt-5.6-luna-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.6-luna-pro", "openai/gpt-5.6-luna-pro"),
        responses: &openai_responses("gpt-5.6-luna-pro"),
        price: price("0.2", "1.2", "0.02", "0.25"),
    },
    ModelRoute {
        model: "gpt-5.6-sol",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.6-sol", "openai/gpt-5.6-sol"),
        responses: &openai_responses("gpt-5.6-sol"),
        price: price("2", "10", "0.2", "2.5"),
    },
    ModelRoute {
        model: "gpt-5.6-sol-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.6-sol-pro", "openai/gpt-5.6-sol-pro"),
        responses: &openai_responses("gpt-5.6-sol-pro"),
        price: price("4", "20", "0.4", "5"),
    },
    ModelRoute {
        model: "gpt-5.6-terra",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.6-terra", "openai/gpt-5.6-terra"),
        responses: &openai_responses("gpt-5.6-terra"),
        price: price("2", "12", "0.2", "2.5"),
    },
    ModelRoute {
        model: "gpt-5.6-terra-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-5.6-terra-pro", "openai/gpt-5.6-terra-pro"),
        responses: &openai_responses("gpt-5.6-terra-pro"),
        price: price("2", "12", "0.2", "2.5"),
    },
    ModelRoute {
        model: "gpt-6-astra",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-6-astra", "openai/gpt-6-astra"),
        responses: &openai_responses("gpt-6-astra"),
        price: price("10", "50", "1", "12.5"),
    },
    ModelRoute {
        model: "gpt-6-astra-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("gpt-6-astra-pro", "openai/gpt-6-astra-pro"),
        responses: &openai_responses("gpt-6-astra-pro"),
        price: price("10", "50", "1", "12.5"),
    },
    // xAI Grok. Native ids from the 2026-09-17 xAI models table plus `grok-4.20-multi-agent`
    // (OpenRouter `x-ai/grok-4.20-multi-agent`, 2026-09-19). No Responses arm — session state
    // is OpenAI's store.
    ModelRoute {
        model: "grok-4.20",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-4.20", "x-ai/grok-4.20"),
        responses: &[],
        price: price("1.25", "2.5", "0.2", "1.25"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "grok-4.20-multi-agent",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-4.20-multi-agent", "x-ai/grok-4.20-multi-agent"),
        responses: &[],
        price: price("1.25", "2.5", "0.2", "1.25"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "grok-4.3",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-4.3", "x-ai/grok-4.3"),
        responses: &[],
        price: price("1.25", "2.5", "0.2", "1.25"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "grok-4.5",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-4.5", "x-ai/grok-4.5"),
        responses: &[],
        price: price("2", "6", "0.3", "2"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "grok-4.6",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-4.6", "x-ai/grok-4.6"),
        responses: &[],
        price: price("2", "6", "0.5", "2"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "grok-build-0.1",
        wire: WireFormat::OpenAi,
        candidates: &xai("grok-build-0.1", "x-ai/grok-build-0.1"),
        responses: &[],
        price: price("1", "2", "0.2", "1"), // cache_write unpublished; equals input
    },
    // Groq / Together / Fireworks llama + qwen + open-weight ids people send. No Meta row, so
    // primary is the host whose id is the catalog name (Groq for the short llama-3.x ids,
    // Fireworks for Llama 4, Together for Qwen / Kimi / GLM / MiniMax / Gemma / Inkling).
    // OpenRouter (or Groq/Fireworks) is failover. Llama 3.3 and GPT-OSS 120B name every host
    // we already route them on.
    ModelRoute {
        model: "llama-3.1-8b-instant",
        wire: WireFormat::OpenAi,
        candidates: &groq("llama-3.1-8b-instant", "meta-llama/llama-3.1-8b-instruct"),
        responses: &[],
        price: price("0.05", "0.08", "0.025", "0.05"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "llama-3.3-70b-versatile",
        wire: WireFormat::OpenAi,
        candidates: &llama_3_3(),
        responses: &[],
        price: price("0.1", "0.32", "0.1", "0.1"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "meta-llama/llama-4-maverick",
        wire: WireFormat::OpenAi,
        candidates: &fireworks(
            "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "meta-llama/llama-4-maverick",
        ),
        responses: &[],
        price: price("0.1875", "0.6525", "0.1875", "0.1875"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "meta-llama/llama-4-scout",
        wire: WireFormat::OpenAi,
        candidates: &fireworks(
            "accounts/fireworks/models/llama4-scout-instruct-basic",
            "meta-llama/llama-4-scout",
        ),
        responses: &[],
        price: price("0.1", "0.3", "0.1", "0.1"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "meta/muse-glimmer-30b",
        wire: WireFormat::OpenAi,
        candidates: &together("meta-models/Muse-Glimmer-30B", "meta/muse-glimmer-30b"),
        responses: &[],
        price: price("0.35", "1.5", "0.04", "0.35"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "minimax/minimax-m3",
        wire: WireFormat::OpenAi,
        candidates: &minimax_m3(),
        responses: &[],
        price: price("0.3", "1.2", "0.06", "0.3"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "minimaxai/minimax-m2.7",
        wire: WireFormat::OpenAi,
        candidates: &groq("minimaxai/minimax-m2.7", "minimax/minimax-m2.7"),
        responses: &[],
        price: price("0.21", "0.84", "0.042", "0.21"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "ministral-14b-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("ministral-14b-latest", "mistralai/ministral-14b-2512"),
        responses: &[],
        price: price("0.2", "0.2", "0.02", "0.2"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "ministral-3b-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("ministral-3b-latest", "mistralai/ministral-3b-2512"),
        responses: &[],
        price: price("0.1", "0.1", "0.01", "0.1"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "ministral-8b-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("ministral-8b-latest", "mistralai/ministral-8b-2512"),
        responses: &[],
        price: price("0.15", "0.15", "0.015", "0.15"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "mistral-large-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("mistral-large-latest", "mistralai/mistral-large-2512"),
        responses: &[],
        price: price("0.5", "1.5", "0.05", "0.5"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "mistral-medium-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("mistral-medium-latest", "mistralai/mistral-medium-3-5"),
        responses: &[],
        price: price("1.5", "7.5", "1.5", "1.5"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "mistral-nemo",
        wire: WireFormat::OpenAi,
        candidates: &mistral("mistral-nemo", "mistralai/mistral-nemo"),
        responses: &[],
        price: price("0.019", "0.03", "0.019", "0.019"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "mistral-small-latest",
        wire: WireFormat::OpenAi,
        candidates: &mistral("mistral-small-latest", "mistralai/mistral-small-2603"),
        responses: &[],
        price: price("0.15", "0.6", "0.015", "0.15"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "moonshotai/kimi-k2.6",
        wire: WireFormat::OpenAi,
        candidates: &fireworks(
            "accounts/fireworks/models/kimi-k2p6",
            "moonshotai/kimi-k2.6",
        ),
        responses: &[],
        price: price("0.65", "3.41", "0.15", "0.65"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "moonshotai/kimi-k2.7-code",
        wire: WireFormat::OpenAi,
        candidates: &together("moonshotai/Kimi-K2.7-Code", "moonshotai/kimi-k2.7-code"),
        responses: &[],
        price: price("0.6562", "3.3", "0.18", "0.6562"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "moonshotai/kimi-k3",
        wire: WireFormat::OpenAi,
        candidates: &kimi_k3(),
        responses: &[],
        price: price("3", "15", "0.3", "3"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "o1",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o1", "openai/o1"),
        responses: &openai_responses("o1"),
        price: price("15", "60", "7.5", "15"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "o1-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o1-pro", "openai/o1-pro"),
        responses: &openai_responses("o1-pro"),
        price: price("150", "600", "150", "150"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "o3",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o3", "openai/o3"),
        responses: &openai_responses("o3"),
        price: price("2", "8", "0.5", "2"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "o3-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o3-mini", "openai/o3-mini"),
        responses: &openai_responses("o3-mini"),
        price: price("1.1", "4.4", "0.55", "1.1"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "o3-pro",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o3-pro", "openai/o3-pro"),
        responses: &openai_responses("o3-pro"),
        price: price("20", "80", "20", "20"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "o4-mini",
        wire: WireFormat::OpenAi,
        candidates: &openai_chat("o4-mini", "openai/o4-mini"),
        responses: &openai_responses("o4-mini"),
        price: price("1.1", "4.4", "0.275", "1.1"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "openai/gpt-oss-120b",
        wire: WireFormat::OpenAi,
        candidates: &gpt_oss_120b(),
        responses: &[],
        price: price("0.037", "0.17", "0.037", "0.037"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "openai/gpt-oss-20b",
        wire: WireFormat::OpenAi,
        candidates: &gpt_oss_20b(),
        responses: &[],
        price: price("0.018", "0.09", "0.009", "0.018"), // cache_write unpublished; equals input
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
    },
    ModelRoute {
        model: "qwen/qwen-2.5-7b-instruct",
        wire: WireFormat::OpenAi,
        candidates: &together(
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "qwen/qwen-2.5-7b-instruct",
        ),
        responses: &[],
        price: price("0.1", "0.2", "0.1", "0.1"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "qwen/qwen3.5-9b",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.5-9B", "qwen/qwen3.5-9b"),
        responses: &[],
        price: price("0.1", "0.15", "0.1", "0.1"), // no separate cache card; both rates equal input
    },
    ModelRoute {
        model: "qwen/qwen3.6-plus",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.6-Plus", "qwen/qwen3.6-plus"),
        responses: &[],
        price: price("0.325", "1.95", "0.325", "0.40625"), // cache_read unpublished; equals input
    },
    ModelRoute {
        model: "qwen/qwen3.7-max",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.7-Max", "qwen/qwen3.7-max"),
        responses: &[],
        price: price("1.475", "4.425", "0.295", "1.84375"),
    },
    ModelRoute {
        model: "qwen/qwen3.7-plus",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.7-Plus", "qwen/qwen3.7-plus"),
        responses: &[],
        price: price("0.32", "1.28", "0.064", "0.4"),
    },
    ModelRoute {
        model: "qwen/qwen3.8-2.4t-a95b",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.8-2.4T-A95B", "qwen/qwen3.8-2.4t-a95b"),
        responses: &[],
        price: price("2", "6", "0.25", "2"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "qwen/qwen3.8-27b",
        wire: WireFormat::OpenAi,
        candidates: &groq("qwen/qwen3.8-27b", "qwen/qwen3.8-27b"),
        responses: &[],
        price: price("0.42", "3", "0.085", "0.42"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "qwen/qwen3.8-flash",
        wire: WireFormat::OpenAi,
        candidates: &together("Qwen/Qwen3.8-Flash", "qwen/qwen3.8-flash"),
        responses: &[],
        price: price("0.15", "0.47", "0.016", "0.2"),
    },
    ModelRoute {
        model: "thinkingmachines/inkling",
        wire: WireFormat::OpenAi,
        candidates: &together("thinkingmachines/Inkling", "thinkingmachines/inkling"),
        responses: &[],
        price: price("1", "4.05", "0.17", "1"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "z-ai/glm-5.1",
        wire: WireFormat::OpenAi,
        candidates: &fireworks("accounts/fireworks/models/glm-5p1", "z-ai/glm-5.1"),
        responses: &[],
        price: price("1.4", "4.4", "0.26", "1.4"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "z-ai/glm-5.2",
        wire: WireFormat::OpenAi,
        candidates: &glm_5_2(),
        responses: &[],
        price: price("0.1739", "3.99", "0.1391", "0.1739"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "z-ai/glm-5.3",
        wire: WireFormat::OpenAi,
        candidates: &together("zai-org/GLM-5.3", "z-ai/glm-5.3"),
        responses: &[],
        price: price("1.4", "4.4", "0.26", "1.4"), // cache_write unpublished; equals input
    },
    ModelRoute {
        model: "z-ai/glm-5.3-flash",
        wire: WireFormat::OpenAi,
        candidates: &together("zai-org/GLM-5.3-Flash", "z-ai/glm-5.3-flash"),
        responses: &[],
        price: price("0.15", "0.5", "0.03", "0.15"), // cache_write unpublished; equals input
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

/// OpenAI-shaped `GET /v1/models` body for the catalog. Extra `wire` (`"openai"` / `"anthropic"`)
/// so a caller can pick the matching SDK, and `pricing` (USD per million tokens — see [`ListPrice`]).
/// Names are log-safe (`[a-z0-9._/-]`) and prices are decimal strings, so this needs no JSON escaping.
pub fn models_list_json() -> &'static str {
    static JSON: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    JSON.get_or_init(|| {
        use std::fmt::Write as _;
        let mut out = String::from(
            "{\"object\":\"list\",\"pricing_unit\":\"usd_per_million_tokens\",\"data\":[",
        );
        for (i, r) in MODEL_ROUTES.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let p = r.price;
            let _ = write!(
                out,
                "{{\"id\":\"{}\",\"object\":\"model\",\"type\":\"model\",\"owned_by\":\"system\",\"wire\":\"{}\",\"pricing\":{{\"input\":\"{}\",\"output\":\"{}\",\"cache_read\":\"{}\",\"cache_write\":\"{}\"}}}}",
                r.model,
                r.wire.as_str(),
                p.input,
                p.output,
                p.cache_read,
                p.cache_write,
            );
        }
        out.push_str("]}");
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{by_id, gateway_providers};

    /// Product floor: managed `/v1` should list a full current generation, not a handful of
    /// flagships. Count is the guard; new rows still have to pass the uniqueness / wire tests below.
    #[test]
    fn catalog_lists_at_least_100_models() {
        assert!(
            MODEL_ROUTES.len() >= 100,
            "MODEL_ROUTES has {} rows; keep the managed catalog at 100+",
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
                    "route {:?} candidate {:?} path {:?} is not Chat Completions, Messages, or Responses",
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

    #[test]
    fn for_model_is_case_insensitive() {
        assert_eq!(
            for_model("GPT-4O-MINI").map(|r| r.model),
            Some("gpt-4o-mini"),
        );
    }

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

    #[test]
    fn models_list_json_names_every_row_its_wire_and_its_price() {
        let json = models_list_json();
        assert!(
            json.starts_with(
                "{\"object\":\"list\",\"pricing_unit\":\"usd_per_million_tokens\",\"data\":["
            ),
            "OpenAI list envelope: {json}"
        );
        assert!(json.ends_with("]}"), "{json}");
        for route in MODEL_ROUTES {
            let p = route.price;
            assert!(
                json.contains(&format!(
                    "\"id\":\"{}\",\"object\":\"model\",\"type\":\"model\",\"owned_by\":\"system\",\"wire\":\"{}\",\"pricing\":{{\"input\":\"{}\",\"output\":\"{}\",\"cache_read\":\"{}\",\"cache_write\":\"{}\"}}",
                    route.model,
                    route.wire.as_str(),
                    p.input,
                    p.output,
                    p.cache_read,
                    p.cache_write,
                )),
                "{:?} wire/price missing from the models list",
                route.model
            );
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
            assert!(output > 0, "{} output", route.model);
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
        let haiku3 = for_model("claude-3-haiku");
        assert!(haiku3.is_some(), "claude-3-haiku");
        let haiku3 = haiku3.unwrap_or(&MODEL_ROUTES[0]).price;
        assert_eq!(
            (
                haiku3.input,
                haiku3.output,
                haiku3.cache_read,
                haiku3.cache_write
            ),
            ("0.25", "1.25", "0.025", "0.3125")
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
                "deepseek-chat",
                WireFormat::OpenAi,
                ProviderId::DeepSeek,
                "deepseek-chat",
                "deepseek/deepseek-chat",
            ),
            (
                "mistral-large-latest",
                WireFormat::OpenAi,
                ProviderId::Mistral,
                "mistral-large-latest",
                "mistralai/mistral-large-2512",
            ),
            (
                "llama-3.1-8b-instant",
                WireFormat::OpenAi,
                ProviderId::Groq,
                "llama-3.1-8b-instant",
                "meta-llama/llama-3.1-8b-instruct",
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

    /// GPT-OSS 120B is the other four-host row. Catalog name is the shared Groq/Together/OpenRouter
    /// id; Fireworks keeps `accounts/fireworks/models/gpt-oss-120b` as an alias.
    #[test]
    fn gpt_oss_120b_names_groq_together_fireworks_and_openrouter() {
        assert!(
            for_model("openai/gpt-oss-120b").is_some(),
            "openai/gpt-oss-120b must be in the catalog"
        );
        if let Some(row) = for_model("openai/gpt-oss-120b") {
            assert_eq!(row.wire, WireFormat::OpenAi);
            assert_eq!(row.candidates.len(), 4);
            assert_eq!(row.candidates[0].provider, ProviderId::Groq);
            assert_eq!(row.candidates[1].provider, ProviderId::Together);
            assert_eq!(row.candidates[2].provider, ProviderId::Fireworks);
            assert_eq!(
                row.candidates[2].upstream_model,
                "accounts/fireworks/models/gpt-oss-120b"
            );
            assert_eq!(row.candidates[3].provider, ProviderId::OpenRouter);
            assert_eq!(
                for_model("accounts/fireworks/models/gpt-oss-120b").map(|r| r.model),
                Some("openai/gpt-oss-120b"),
            );
        }
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

    /// Llama 3.3 is the one row that names every host we already route it on. Groq's id is the
    /// catalog name; Together / Fireworks / OpenRouter keep their own spellings as aliases.
    #[test]
    fn llama_3_3_names_groq_together_fireworks_and_openrouter() {
        assert!(
            for_model("llama-3.3-70b-versatile").is_some(),
            "llama-3.3-70b-versatile must be in the catalog"
        );
        if let Some(row) = for_model("llama-3.3-70b-versatile") {
            assert_eq!(row.wire, WireFormat::OpenAi);
            assert_eq!(row.candidates.len(), 4);
            assert_eq!(row.candidates[0].provider, ProviderId::Groq);
            assert_eq!(row.candidates[0].path, "/openai/v1/chat/completions");
            assert_eq!(row.candidates[1].provider, ProviderId::Together);
            assert_eq!(
                row.candidates[1].upstream_model,
                "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            );
            assert_eq!(row.candidates[2].provider, ProviderId::Fireworks);
            assert_eq!(
                row.candidates[2].upstream_model,
                "accounts/fireworks/models/llama-v3p3-70b-instruct"
            );
            assert_eq!(row.candidates[2].path, "/inference/v1/chat/completions");
            assert_eq!(row.candidates[3].provider, ProviderId::OpenRouter);
            assert_eq!(
                for_model("accounts/fireworks/models/llama-v3p3-70b-instruct").map(|r| r.model),
                Some("llama-3.3-70b-versatile"),
            );
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
            r.candidates
                .first()
                .is_some_and(|c| c.provider == ProviderId::OpenAi)
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
}
