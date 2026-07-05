//! A minimal per-model capability table.
//!
//! The agent — like the gateway it sits behind — holds no model catalog: a model id is forwarded
//! verbatim and only its *wire dialect* is derived from the name (see [`dialect`]). But a handful of
//! wire-shaping decisions genuinely need per-model knowledge the request can't carry on its own:
//!
//! - which output-ceiling field a model expects (`max_tokens` vs `max_completion_tokens`),
//! - whether it honors the 1-hour prompt-cache TTL,
//! - what extended-thinking shape it accepts,
//! - a sane context-window default for compaction.
//!
//! This is the *smallest* table those decisions need, matched by id prefix. It is deliberately **not**
//! pricing (the gateway meters tokens; a downstream consumer prices them) and **not** a routing
//! registry (the gateway routes). When a model id is unknown the table returns conservative defaults
//! that keep every wire decision safe.
//!
//! [`dialect`]: crate::dialect

/// Which output-ceiling field a model's wire expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxTokensField {
    /// `max_tokens` — Anthropic, and OpenAI chat-completions on non-reasoning models.
    MaxTokens,
    /// `max_completion_tokens` — OpenAI reasoning models (o-series, gpt-5) reject `max_tokens`.
    MaxCompletionTokens,
}

/// Which OpenAI-wire API surface a model speaks. Anthropic ids ignore this field entirely (they
/// always speak `Dialect::Anthropic`); it only disambiguates within the OpenAI-wire branch of
/// [`crate::dialect::Dialect::for_model`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKind {
    /// `/v1/chat/completions` — every third-party OpenAI-compatible provider (OpenRouter, DeepSeek,
    /// Together, Cerebras, xAI, Groq, Fireworks, Mistral), and the fallback for any unrecognized id.
    ChatCompletions,
    /// `/v1/responses` — every native OpenAI model id (gpt-4/4.1/4o, gpt-5 family, o-series). Ported
    /// from pi's live catalogue: every entry under `packages/ai/src/providers/openai.models.ts`
    /// carries `api: "openai-responses"`, with no exceptions among current ids.
    Responses,
}

/// Which OpenAI Chat-Completions-wire reasoning/thinking toggle a model expects — mirrors pi's
/// `compat.thinkingFormat` tag (`packages/ai/src/api/openai-completions.ts`). Only consulted by
/// [`crate::dialect::openai::build_body`]; the Anthropic dialect has its own shape
/// ([`ThinkingShape`]) and the OpenAI Responses dialect always speaks native OpenAI (`Standard`).
/// Every native OpenAI id and every model with no vendor-specific quirk stays `Standard` — the bare
/// `reasoning_effort` string this dialect emitted before third-party coverage existed, unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiReasoningFormat {
    /// Bare `reasoning_effort: "<level>"` (or `"none"` when [`ModelCaps::reasoning_disableable`] and
    /// no level is requested) — OpenAI's own shape, and pi's default for every provider it doesn't
    /// special-case (xAI, Groq, Cerebras, and any unrecognized id).
    Standard,
    /// DeepSeek — and Moonshot/Kimi, which pi tags with this identical shape: `thinking: {"type":
    /// "enabled"}` / `{"type": "disabled"}`, plus — only when [`ModelCaps::reasoning_effort`] is also
    /// `true` (DeepSeek; not Kimi, which has no effort vocabulary at all) — a sibling top-level
    /// `reasoning_effort` string.
    DeepSeek,
    /// Z.ai/GLM: `thinking: {"type": "enabled", "clear_thinking": false}` / `{"type": "disabled"}`,
    /// sent unconditionally (unlike `DeepSeek`, pi never gates this on a disableable check), plus —
    /// only when [`ModelCaps::reasoning_effort`] is also `true` (GLM-5.2+ only) — a sibling top-level
    /// `reasoning_effort` string.
    Zai,
    /// Together: a nested `reasoning: {"enabled": bool}`, sent unconditionally, plus — only when
    /// [`ModelCaps::reasoning_effort`] is also `true` — a sibling top-level `reasoning_effort` string.
    Together,
    /// OpenRouter — and the generic vendor/model-shaped fallback for an uncatalogued third-party id:
    /// a nested `reasoning: {"effort": "<level>"}`, or `{"effort": "none"}` when
    /// [`ModelCaps::reasoning_disableable`] and no level is requested.
    OpenRouter,
}

/// The extended-thinking request shape a model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingShape {
    /// No Anthropic-style thinking control — omit the `thinking` field (OpenAI reasoning models use
    /// `reasoning_effort` instead; see [`ModelCaps::reasoning_effort`]).
    None,
    /// Anthropic `{type:"enabled", budget_tokens}` — Claude 3.7 / 4.x extended thinking.
    Budget,
    /// Anthropic `{type:"adaptive", display}` (plus a sibling top-level `output_config.effort`) — the
    /// shape generation-6+ Claude/Fable models require (`Budget`'s `{type:"enabled", budget_tokens}` is
    /// rejected). This is our own default model's shape (`claude-opus-4-8`); the live smoke test
    /// exercises it directly rather than assuming `Budget` covers everything current.
    Adaptive,
}

/// Per-model capability flags the wire adapters and compaction consult.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCaps {
    /// Context window in tokens — the compaction default when the caller doesn't override it.
    pub context_window: u32,
    /// A safe default per-turn output ceiling for the model.
    pub max_output: u32,
    /// Which output-ceiling field name to emit on the wire.
    pub max_tokens_field: MaxTokensField,
    /// Whether the model honors the 1-hour prompt-cache TTL (`cache_control.ttl = "1h"`).
    pub supports_long_cache: bool,
    /// Whether the model accepts image input.
    pub supports_vision: bool,
    /// Whether the model accepts a `temperature` parameter at all. `true` for almost every model;
    /// `false` for `claude-opus-4-7`/`claude-opus-4-8` specifically, which reject it outright (pi:
    /// `compat.supportsTemperature: false`, `anthropic.models.ts`) — distinct from the existing
    /// thinking-enabled gate (`ModelRequest::temperature`'s own doc comment), which omits the field
    /// only while extended thinking is on. A model can need both gates at once.
    pub supports_temperature: bool,
    /// The extended-thinking shape the model accepts.
    pub thinking: ThinkingShape,
    /// Whether the model takes an OpenAI `reasoning_effort` parameter.
    pub reasoning_effort: bool,
    /// Whether the model accepts an *explicit* "reasoning off" signal — Anthropic
    /// `{"type":"disabled"}`, or OpenAI `{"effort":"none"}` — when the caller isn't requesting
    /// thinking/reasoning this turn. A model that supports thinking/`reasoning_effort` at all doesn't
    /// necessarily support turning it off explicitly (e.g. `claude-fable-5`); one that supports
    /// neither has nothing to disable, so this is `false` there too.
    pub reasoning_disableable: bool,
    /// Whether the model's tool definitions should be marked with `eager_input_streaming: true` on the
    /// Anthropic wire (ignored outside the Anthropic dialect). `true` for every current Anthropic id;
    /// exists as a named capability (rather than a blanket constant) so a future model needing the
    /// mutually-exclusive `fine-grained-tool-streaming-2025-05-14` beta header instead has somewhere to
    /// say so.
    pub supports_eager_tool_streaming: bool,
    /// Whether tool definitions on the OpenAI Chat-Completions wire should be sent alongside a
    /// top-level `tool_stream: true` (ignored outside that dialect, and only ever emitted when the
    /// request actually carries tools — see [`crate::dialect::openai::build_body`]). Mirrors pi's
    /// `compat.zaiToolStream` (`openai-completions.ts:582-586`): every current Z.ai/GLM id sets it
    /// except `glm-4.5-air`, the one id pi's own `zai.models.ts`/`zai-coding-cn.models.ts` catalogue
    /// leaves it off for; no other family sets it at all.
    pub supports_tool_stream: bool,
    /// Which OpenAI-wire API surface the model speaks (ignored for Anthropic ids). See [`ApiKind`].
    pub api: ApiKind,
    /// The lowest [`crate::transport::ReasoningEffort`] this model's wire actually accepts — a request
    /// below this floor is clamped up (e.g. `gpt-5.5-pro` rejects both `minimal` and `low`, so its floor
    /// is `Medium`). Only consulted for a model that takes `reasoning_effort`/adaptive thinking at all;
    /// meaningless (and unread) otherwise. See [`clamp_reasoning_effort`].
    pub min_reasoning_effort: crate::transport::ReasoningEffort,
    /// Whether this model's wire has a distinct `xhigh` tier at all. Several current OpenAI reasoning
    /// models (o-series, bare/gpt-5.1-family gpt-5) and two Anthropic adaptive models (sonnet-4-6,
    /// sonnet-5) top out at `high` — requesting `xhigh` there must clamp down to `high`, not send a
    /// value the provider doesn't recognize. See [`clamp_reasoning_effort`].
    pub supports_xhigh_reasoning: bool,
    /// The Anthropic adaptive-thinking wire string for `xhigh`, once [`clamp_reasoning_effort`] has
    /// confirmed the model supports it at all. Only `claude-opus-4-6` differs (`"max"` — pi: "effort
    /// 'max' is only valid on Opus 4.6, while Opus 4.7+ and Fable 5 support 'xhigh'"); every other
    /// adaptive-shape id sends the literal `"xhigh"`. Unread outside `ThinkingShape::Adaptive`.
    pub adaptive_xhigh_effort_wire: &'static str,
    /// Which OpenAI Chat-Completions-wire reasoning-toggle shape this model expects. Only read by
    /// [`crate::dialect::openai::build_body`]; every other dialect ignores it. See
    /// [`OpenAiReasoningFormat`].
    pub openai_reasoning_format: OpenAiReasoningFormat,
}

impl ModelCaps {
    /// Conservative defaults for an unknown model id: the universally-accepted `max_tokens` field, no
    /// long cache, no thinking — every choice that can't make a request invalid.
    const fn unknown() -> Self {
        Self {
            context_window: 128_000,
            max_output: 4_096,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: false,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        }
    }
}

/// Whether a `gpt-5.*` id's wire has a distinct `xhigh` reasoning tier. True from generation 5.2
/// onward (`gpt-5.2`/`.3`/`.4`/`.5`, any suffix — codex/pro/spark/chat-latest all included); false for
/// bare `gpt-5` and every `gpt-5.1` variant, none of which list `xhigh` in pi's live catalogue at all.
/// Shared across all three `gpt-5` return sites below (chat-latest, the codex-spark special case, and
/// the general bucket) rather than duplicated per branch.
fn gpt5_supports_xhigh(m: &str) -> bool {
    m.starts_with("gpt-5.2")
        || m.starts_with("gpt-5.3")
        || m.starts_with("gpt-5.4")
        || m.starts_with("gpt-5.5")
}

/// Every id in pi's Mistral catalogue (`packages/ai/src/providers/mistral.models.ts`) starts with one
/// of these — there's no single common prefix for the whole provider the way there is for e.g.
/// DeepSeek: codestral/devstral/ministral/magistral/mistral itself/pixtral/open-mistral/open-mixtral are
/// all separate id families under this one native `KNOWN_PROVIDERS` gateway route
/// (`crates/gateway/src/route.rs`). `m` must already be lowercased (every caller below goes through
/// [`is_mistral_model`], the public, self-lowercasing wrapper).
const MISTRAL_ID_PREFIXES: &[&str] = &[
    "mistral",
    "codestral",
    "devstral",
    "ministral",
    "magistral",
    "pixtral",
    "open-mistral",
    "open-mixtral",
    "labs-devstral",
];

fn is_mistral_id(m: &str) -> bool {
    MISTRAL_ID_PREFIXES.iter().any(|p| m.starts_with(p))
}

/// Whether `model` is a Mistral id. Shared between this table's Mistral `capabilities()` branch and
/// `dialect::openai`'s Mistral-specific tool-call-id reshaping (Mistral's real API rejects a
/// `tool_call_id` that isn't exactly 9 alphanumeric characters) — both need the identical "is this a
/// Mistral model" check, so it lives here once rather than being duplicated per call site.
pub fn is_mistral_model(model: &str) -> bool {
    is_mistral_id(&model.to_ascii_lowercase())
}

/// Whether `model` is a DeepSeek id — `dialect::openai::build_body`'s assistant-replay path needs this
/// specifically (not the broader `OpenAiReasoningFormat::DeepSeek` shape, which Moonshot/Kimi shares
/// too) to backfill an empty `reasoning_content` on replay, matching pi's `compat
/// .requiresReasoningContentOnAssistantMessages` (`isDeepSeek`, `openai-completions.ts::detectCompat`) —
/// a DeepSeek-only quirk, not shared with any other family tagged with the same wire *shape*.
pub fn is_deepseek_model(model: &str) -> bool {
    model.to_ascii_lowercase().starts_with("deepseek")
}

/// Every current id belonging to a provider pi's `openai-completions.ts::detectCompat` marks
/// `isNonStandard` (hence `supportsStore: false` — the provider rejects or simply doesn't understand
/// OpenAI's `store` extension field): DeepSeek, Z.ai/GLM, Moonshot/Kimi, xAI/Grok, Together-hosted
/// Qwen, and Cerebras's native (unprefixed) id set. `false` — meaning `store: false` gets sent — for
/// everything else, including native OpenAI-via-Chat-Completions, OpenRouter, Groq, Fireworks, Mistral,
/// and any uncatalogued id, matching pi's own default.
///
/// pi's real exclusion list also covers NVIDIA, Cloudflare (Workers AI and AI Gateway), and Ant-Ling —
/// none of which have an id shape of their own this table could recognize (they're reached via
/// arbitrary vendor-native ids through a NIM/gateway proxy, not a fixed prefix), the same known
/// limitation already documented at this table's own "Third-party OpenAI-compatible providers" section
/// header: a route/provider-level distinction with no matching id-level signature can't be told apart
/// from a generic third-party id by name alone.
pub fn is_non_standard_store_provider(model: &str) -> bool {
    const CEREBRAS_NATIVE: &[&str] = &[
        "gpt-oss-120b",
        "gpt-oss-20b",
        "gpt-oss-safeguard-20b",
        "gemma-4-31b",
        "zai-glm-4.7",
    ];
    let m = model.to_ascii_lowercase();
    m.starts_with("deepseek")
        || m.starts_with("glm")
        || m.starts_with("kimi")
        || m.starts_with("grok")
        || (m.starts_with("qwen") && m != "qwen/qwen3-32b")
        || CEREBRAS_NATIVE.contains(&m.as_str())
}

/// Resolve a model id to its [`ModelCaps`]. Matching is by id prefix (most-specific first); unknown
/// ids fall back to [`ModelCaps::unknown`] (logged, since a silent conservative fallback can otherwise
/// mask a model we should have taught this table about).
///
/// The per-family numbers below are ported from the reference agent's live model catalogue
/// (`packages/ai/src/providers/{anthropic,openai}.models.ts` in `badlogic/pi-mono`) rather than
/// invented — re-check that catalogue when adding a new model family, since it's regenerated upstream
/// and may have moved on by the time you read this.
pub fn capabilities(model: &str) -> ModelCaps {
    use crate::transport::ReasoningEffort as RE;
    let m = model.to_ascii_lowercase();
    // The id the third-party family branches below (DeepSeek, GLM, Kimi, MiniMax, Qwen, MiMo) match
    // their bare-id patterns against: the suffix after the last `/` for a vendor-slug id — the shape
    // aggregator hosts (Together, HuggingFace, NVIDIA NIM, …) prefix a model with the org that trained
    // it (`"moonshotai/Kimi-K2.6"`, `"zai-org/GLM-5.2"`, `"XiaomiMiMo/MiMo-V2.5-Pro"`) — or the id
    // itself when it isn't slug-prefixed at all. Using this (rather than the raw id, or a
    // `.contains(family)` substring check) fixes two failure modes at once: a slug-prefixed id whose
    // org slug doesn't happen to start with the family name (`"moonshotai/kimi-k2.6"` doesn't start
    // with `"kimi"`) used to fall all the way through to the generic 128k/32k vendor-slug fallback
    // further below; and an org slug that *does* happen to share a literal prefix with a different
    // family's own name — `"MiniMaxAI/MiniMax-M3"` lowercases to `"minimaxai/minimax-m3"`, which
    // itself begins with the literal string `"minimax"` one character early — used to land in the
    // right family branch but silently match the wrong per-id sub-case inside it (the flat "else"
    // shape, not `minimax-m3`'s real one). Deliberately not applied to Anthropic/OpenAI/Mistral/
    // xAI/Cerebras-native/Groq/Llama below: none of those are reachable through an aggregator's
    // vendor-slug id shape in pi's live catalogues, and blanket-applying it there would let a
    // vendor-slug id's *suffix* alone decide family membership for providers that never actually
    // appear slug-prefixed — an unnecessary widening of what each of those branches matches.
    let family_id: &str = match m.rfind('/') {
        Some(idx) => &m[idx + 1..],
        None => &m,
    };

    // ---- Anthropic Claude (+ Fable, which speaks the Anthropic wire) ----
    if m.starts_with("claude") || m.starts_with("fable") {
        // Generation 6+ (opus-4-6/4-7/4-8, sonnet-4-6, sonnet-5, fable-5) require the newer `adaptive`
        // thinking shape (`Budget`'s `{type:"enabled", budget_tokens}` is rejected) and ship a 1M-token
        // context window — a full step up from every earlier generation. Matched by the exact
        // family+generation token so "claude-opus-4-5" can't collide with "claude-opus-4-6" and later.
        const GEN6_PLUS: &[&str] = &[
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable-5",
            "fable-5",
        ];
        if GEN6_PLUS.iter().any(|p| m.starts_with(p)) {
            // sonnet-4-6 caps output at 64k; every other gen6+ family caps at 128k.
            let max_output = if m.starts_with("claude-sonnet-4-6") {
                64_000
            } else {
                128_000
            };
            // Every gen6+ id can be told to disable thinking explicitly, except claude-fable-5 (pi:
            // `thinkingLevelMap: {"off": null}` — there's no "off" wire shape for it at all).
            let reasoning_disableable =
                !(m.starts_with("claude-fable-5") || m.starts_with("fable-5"));
            // sonnet-4-6/sonnet-5 carry no `thinkingLevelMap` at all in pi's catalogue, and an unmapped
            // "xhigh" isn't a value their wire accepts — only opus-4-6/4-7/4-8 and fable-5 do (see
            // `adaptive_xhigh_effort_wire`'s doc comment).
            let supports_xhigh_reasoning =
                !(m.starts_with("claude-sonnet-4-6") || m.starts_with("claude-sonnet-5"));
            // Only opus-4-6 remaps xhigh to "max"; every other adaptive id that supports it sends the
            // literal "xhigh" (the field's own default).
            let adaptive_xhigh_effort_wire = if m.starts_with("claude-opus-4-6") {
                "max"
            } else {
                "xhigh"
            };
            // opus-4-7 and opus-4-8 reject `temperature` outright — pi's `compat.supportsTemperature:
            // false` is set on exactly these two ids in `anthropic.models.ts` and no other gen6+ entry
            // (opus-4-6, sonnet-4-6, sonnet-5, fable-5 all default to supported). Our own DEFAULT_MODEL
            // is `claude-opus-4-8`, so this is a live, reachable 400 without the gate.
            let supports_temperature =
                !(m.starts_with("claude-opus-4-7") || m.starts_with("claude-opus-4-8"));
            return ModelCaps {
                context_window: 1_000_000,
                max_output,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature,
                thinking: ThinkingShape::Adaptive,
                reasoning_effort: false,
                reasoning_disableable,
                supports_eager_tool_streaming: true,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning,
                adaptive_xhigh_effort_wire,
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
            };
        }

        // Generation 3 (pre-4.x, excluding 3.7-sonnet which *does* support extended thinking): no
        // thinking support at all — sending a `thinking` field would be rejected.
        let is_legacy_gen3 = m.starts_with("claude-3-5")
            || m.starts_with("claude-3-haiku")
            || m.starts_with("claude-3-opus")
            || m.starts_with("claude-3-sonnet");
        if is_legacy_gen3 {
            let max_output = if m.starts_with("claude-3-5") {
                8_192
            } else {
                4_096
            };
            return ModelCaps {
                context_window: 200_000,
                max_output,
                max_tokens_field: MaxTokensField::MaxTokens,
                // Prompt caching (the 1-hour TTL included) is supported on every current Claude 3.x
                // id — pi's catalogue defaults `supportsLongCacheRetention` to `true` and no gen-3
                // entry overrides it. `false` here would silently downgrade a `cache_long` request on
                // these ids to the standard 5-minute TTL instead of the 1-hour one asked for.
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                // No thinking support at all here — nothing to explicitly disable, so the `thinking`
                // field stays omitted entirely rather than sending a `{"type":"disabled"}` a model
                // that never supported thinking might reject.
                reasoning_disableable: false,
                supports_eager_tool_streaming: true,
                supports_tool_stream: false,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
            };
        }

        // Everything else current (opus-4-0/4-1/4-5, sonnet-3-7/4-0/4-5, haiku-4-5, and future ids we
        // haven't special-cased above): `Budget`-shape extended thinking, 200k context.
        let max_output = if m.contains("sonnet") || m.contains("haiku") || m.contains("opus-4-5") {
            64_000
        } else {
            32_000 // opus-4-0 / opus-4-1 / generic opus and fable ids
        };
        return ModelCaps {
            context_window: 200_000,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            supports_vision: true,
            supports_temperature: true,
            thinking: ThinkingShape::Budget,
            reasoning_effort: false,
            reasoning_disableable: true,
            supports_eager_tool_streaming: true,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            // Budget-shape thinking sends a numeric `budget_tokens`, never a named effort string, so
            // `min_reasoning_effort`/`adaptive_xhigh_effort_wire` are never read for this shape — left
            // at their harmless defaults. `supports_xhigh_reasoning` is *not* one of those unread
            // fields, despite the shape never putting a named effort string on the wire: it's read
            // shape-agnostically by `available_thinking_levels`/`clamp_reasoning_effort` to decide
            // whether to offer/accept the `xhigh` *portable* thinking-level rung at all (which then
            // maps to a numeric budget via `budget_for_effort`). pi's `thinkingLevelMap.xhigh` is only
            // defined for the four gen6+ Adaptive-shape ids — none of these classic Budget-shape models
            // (sonnet-4-5, opus-4-5, sonnet-3-7, haiku-4-5, opus-4-0/4-1, etc.) offer it.
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }

    // ---- OpenAI reasoning models: o-series ----
    // Reject `max_tokens` (require `max_completion_tokens`); 200k context, 100k max output.
    //
    // `supports_long_cache` below (and on every other OpenAI branch): pi's actual default is
    // *permissive* — `prompt_cache_retention`/24h is supported unless the specific provider opts out
    // (a denylist: Together, Cloudflare, NVIDIA, Ant-Ling — everything else, including native OpenAI,
    // OpenRouter, DeepSeek, Cerebras, Groq, xAI, Mistral, defaults to supporting it). Our table is
    // keyed by *model id*, not by which gateway route/provider is actually serving it, so it can't
    // reproduce that per-provider denylist — we default to `true` (matching pi's default) and accept
    // that a route to a genuinely non-supporting provider (Together is the one we route to) will send
    // a `prompt_cache_retention` field it ignores rather than one it needs but doesn't get.
    let is_o_series = m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4");
    if is_o_series {
        return ModelCaps {
            context_window: 200_000,
            max_output: 100_000,
            // Every o-series id speaks the Responses API (`api` below), which always sends
            // `max_output_tokens` regardless of this field (see `dialect/openai_responses.rs::build_body`)
            // — `max_tokens_field` is only ever read by the Chat Completions dialect. Still set correctly
            // (o-series genuinely rejects `max_tokens`, requiring `max_completion_tokens` instead) rather
            // than left at a value that would be actively wrong if a future routing change ever sent one
            // of these ids through Chat Completions.
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            // o3-mini is text-only (pi's catalogue: `input: ["text"]`) — the one o-series id that isn't
            // vision-capable. o1-mini used to be excluded here too, but pi's live catalogue has fully
            // retired it (no entry at all, as of this writing) — nothing upstream still serves it, so
            // there's no longer a live model to special-case.
            supports_vision: !m.starts_with("o3-mini"),
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            // o-series ids are disable-capable by default in pi's catalogue (no override).
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::Responses,
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            // No o-series id carries a `thinkingLevelMap` at all in pi's catalogue, and an unmapped
            // "xhigh" is excluded from `getSupportedThinkingLevels` — none of them accept it.
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh", // unread: o-series never uses Adaptive shape.
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }

    // ---- OpenAI GPT-5 family (reasoning) ----
    if m.starts_with("gpt-5") {
        // The narrower "-chat-latest" variants share the family name but cap at the older
        // chat-completions ceiling (128k/16384), not the reasoning-model one, and aren't uniformly
        // `reasoning_effort`-driven — treat them like a non-reasoning chat model. Two of the four
        // current ids (5.1/5.2) are still `reasoning_effort`-driven per pi's catalogue, though;
        // gpt-5-chat-latest/gpt-5.3-chat-latest are not. None of the four support an explicit "off"
        // signal (pi: `"off": null` for this whole bucket).
        if m.contains("chat-latest") {
            let reasoning_effort =
                m.starts_with("gpt-5.1-chat-latest") || m.starts_with("gpt-5.2-chat-latest");
            return ModelCaps {
                context_window: 128_000,
                max_output: 16_384,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::Responses,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: gpt5_supports_xhigh(&m),
                adaptive_xhigh_effort_wire: "xhigh", // unread: this bucket never uses Adaptive shape.
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
            };
        }
        // "gpt-5.3-codex-spark" is a narrower model than the rest of the family — 128k context, 32k
        // output — not the generic 400k/128k every other gpt-5 id gets below. Not in pi's
        // disable-capable allowlist (that's `gpt-5.3-codex`, a different id).
        if m == "gpt-5.3-codex-spark" {
            return ModelCaps {
                context_window: 128_000,
                max_output: 32_000,
                // Unread: Responses-routed (see the o-series branch above for why this is still set
                // correctly rather than to a harmless-but-wrong value).
                max_tokens_field: MaxTokensField::MaxCompletionTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: true,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::Responses,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
            };
        }
        // "-pro" ships a much larger 1.05M context — but only the 5.4/5.5 generation of it;
        // gpt-5-pro/gpt-5.2-pro are 400k like the rest of the family. The bare "gpt-5.4"/"gpt-5.5"
        // release (no mini/nano/pro suffix) runs a smaller 272k window; every other family member is
        // 400k.
        let context_window = if m == "gpt-5.4-pro" || m == "gpt-5.5-pro" {
            1_050_000
        } else if m == "gpt-5.4" || m == "gpt-5.5" {
            272_000
        } else {
            400_000
        };
        // Disable-capability is a per-exact-id allowlist in pi's catalogue, not a blanket rule — an
        // id under this generic branch that isn't listed here (e.g. bare "gpt-5", "gpt-5-mini",
        // "gpt-5.3" without "-codex", any "-pro"/"-nano" variant not listed) has no "off" signal.
        const GPT5_DISABLE_CAPABLE: &[&str] = &[
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.3-codex",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-5.5",
        ];
        let reasoning_disableable = GPT5_DISABLE_CAPABLE.iter().any(|id| m == *id);
        // gpt-5.5-pro excludes both "minimal" and "low" (floor: medium); gpt-5.5 excludes just
        // "minimal" (floor: low) — pi's `thinkingLevelMap` nulls those out explicitly. Every earlier
        // gpt-5 generation accepts the full ladder from minimal up.
        let min_reasoning_effort = if m == "gpt-5.5-pro" {
            crate::transport::ReasoningEffort::Medium
        } else if m == "gpt-5.5" {
            crate::transport::ReasoningEffort::Low
        } else {
            crate::transport::ReasoningEffort::Minimal
        };
        return ModelCaps {
            context_window,
            max_output: 128_000,
            // Unread: Responses-routed (see the o-series branch above for why this is still set
            // correctly rather than to a harmless-but-wrong value).
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: true,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::Responses,
            min_reasoning_effort,
            supports_xhigh_reasoning: gpt5_supports_xhigh(&m),
            adaptive_xhigh_effort_wire: "xhigh", // unread: this bucket never uses Adaptive shape.
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }

    // ---- OpenAI GPT-4 family (bare gpt-4 / 4-turbo / 4o / 4.1) ----
    // None of these take `reasoning_effort` at all, so there's nothing to explicitly disable.
    if m.starts_with("gpt-4") {
        // 4.1 shipped a ~1M-token context window, a full step up from the rest of the family.
        if m.starts_with("gpt-4.1") {
            return ModelCaps {
                context_window: 1_047_576,
                max_output: 32_768,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::Responses,
                // The gpt-4 family never takes `reasoning_effort` at all — these three are unread.
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
            };
        }
        // This one pinned snapshot caps output tighter (4096) than every other 4o-family id (16384).
        if m == "gpt-4o-2024-05-13" {
            return ModelCaps {
                context_window: 128_000,
                max_output: 4_096,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                supports_temperature: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                supports_tool_stream: false,
                api: ApiKind::Responses,
                // The gpt-4 family never takes `reasoning_effort` at all — these three are unread.
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
                openai_reasoning_format: OpenAiReasoningFormat::Standard,
            };
        }
        // Bare "gpt-4" (no suffix) is the original 8k-context model; 4-turbo caps output tighter than
        // 4o; everything else (4o, 4o-mini, and dated snapshots) shares a 128k/16384 ceiling.
        let (context_window, max_output) = if m == "gpt-4" {
            (8_192, 8_192)
        } else if m.starts_with("gpt-4-turbo") {
            (128_000, 4_096)
        } else {
            (128_000, 16_384)
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            // Bare "gpt-4" (pi's catalogue: `input: ["text"]`) predates GPT-4's vision support
            // entirely — that shipped separately as "gpt-4-vision-preview" and was later folded into
            // "gpt-4-turbo" (`input: ["text","image"]`) and every 4o-family id. Every other id in this
            // branch does accept image input.
            supports_vision: m != "gpt-4",
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::Responses,
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }

    // ---- Third-party OpenAI-compatible providers ----
    //
    // The gateway's own provider table (`crates/gateway/src/route.rs::KNOWN_PROVIDERS`, plus several
    // Anthropic-wire vendors reachable only via config) actively routes to every family below; without
    // an entry here each one silently fell through to `ModelCaps::unknown()` — 128k context, a 4096
    // output ceiling regardless of the model's real one (`Agent::new` seeds `max_tokens` from
    // `max_output`), and no reasoning/thinking wire support at all (`--thinking high` on, say,
    // `deepseek-v4-pro`, sent no thinking parameter whatsoever). Numbers below are ported from pi's
    // live catalogue (`packages/ai/src/providers/{deepseek,zai,moonshotai,xai,groq,cerebras,together,
    // minimax}.models.ts`, `openai-completions.ts::detectCompat`) — re-check those when adding a new
    // model, same caveat as the Anthropic/OpenAI tables above. Coverage here is intentionally
    // family-level (matched by a recognizable id prefix/substring), not an exhaustive per-exact-id
    // table the way the native Anthropic/OpenAI branches above are: these providers' catalogues
    // (OpenRouter's especially, which mirrors nearly every model that exists) are large and change
    // often, and a reasonably-accurate family default is what actually closes the truncation/no-thinking
    // gap — an exact number a few percent off doesn't reopen it. A model id genuinely reachable through
    // more than one host (e.g. a Together-hosted "deepseek-ai/deepseek-r1" also starting with the
    // literal prefix "deepseek") can still land on the wrong sub-branch; this is a known limitation of a
    // table keyed on model id alone, with no route/provider context to disambiguate by.

    // DeepSeek: 1M context, 384k output, a real reasoning-effort vocabulary (floor `high`, `xhigh`
    // wired as `"max"`) — pi: `compat.thinkingFormat: "deepseek"`, `thinkingLevelMap: {high:"high",
    // xhigh:"max"}`, `supportsReasoningEffort` left at its (`!isZai && …`) default of `true`.
    // DeepSeek's own auto-detected `maxTokensField` is `max_completion_tokens` (not in pi's
    // `useMaxTokens` allowlist), matching this table's other reasoning-model families. Also matches a
    // vendor-slug id whose suffix is a DeepSeek id (e.g. Together/HuggingFace's
    // `"deepseek-ai/DeepSeek-V4-Pro"`) via `family_id` — see its own doc comment above.
    if m.starts_with("deepseek") || family_id.starts_with("deepseek") {
        return ModelCaps {
            context_window: 1_000_000,
            max_output: 384_000,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::High,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "max",
            openai_reasoning_format: OpenAiReasoningFormat::DeepSeek,
        };
    }

    // Xiaomi/MiMo: no bare "mimo"/"xiaomi" branch existed at all before this — every current id
    // (`packages/ai/src/providers/xiaomi.models.ts`, identical across the ams/cn/sgp token-plan
    // variants — only `baseUrl`/`provider` differ) fell all the way through to `ModelCaps::unknown()`,
    // capping `max_output` at 4096 regardless of the model's real 65536-384000 one and disabling
    // reasoning entirely. pi tags every id `compat: {"requiresReasoningContentOnAssistantMessages":
    // true, "thinkingFormat": "deepseek"}` — the identical `thinking:{enabled/disabled}` toggle
    // DeepSeek and Kimi share, with a sibling `reasoning_effort` string since `supportsReasoningEffort`
    // resolves `true` for this provider (xiaomi isn't in pi's `isGrok`/`isZai`/`isMoonshot`/`isTogether`/
    // `isCloudflareAiGateway`/`isNvidia`/`isAntLing` exclusion list) — the exact same
    // `OpenAiReasoningFormat::DeepSeek` shape, reused rather than adding a new enum variant. No id
    // carries a `thinkingLevelMap` at all, so (unlike real DeepSeek) there's no floor/remap: every
    // effort level is legal and sent under its own literal name. `"mimo-v2.5-pro"` starts with the same
    // string as the vision-capable bare `"mimo-v2.5"`, so vision is matched by exact id, not prefix.
    // Also matches a vendor-slug id (HuggingFace's `"XiaomiMiMo/MiMo-V2.5-Pro"`) via `family_id`.
    if m.starts_with("mimo") || family_id.starts_with("mimo") {
        // Keyed on whether `m` is slug-shaped at all — same reasoning as the MiniMax/GLM/Kimi branches.
        let k = if m.contains('/') { family_id } else { m.as_str() };
        let (context_window, max_output, supports_vision) = if k == "mimo-v2-flash" {
            (262_144, 65_536, false)
        } else if k == "mimo-v2-omni" {
            (262_144, 131_072, true)
        } else if k == "mimo-v2-pro" {
            (1_048_576, 131_072, false)
        } else if k == "mimo-v2.5" {
            (1_048_576, 131_072, true)
        } else {
            // "mimo-v2.5-pro", "mimo-v2.5-pro-ultraspeed", and any future id this exact-match table
            // doesn't recognize yet: most current ids share this 1M/131k, text-only shape.
            (1_048_576, 131_072, false)
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::DeepSeek,
        };
    }

    // Mistral: a native `KNOWN_PROVIDERS` gateway route (`crates/gateway/src/route.rs`) whose 30-id
    // catalogue (`packages/ai/src/providers/mistral.models.ts`) has no shared prefix and enough per-id
    // variance (bare "mistral-large-2411" is 131_072/16_384 while "mistral-large-2512" two generations
    // later is 262_144/262_144) that a family-level default would still misreport several ids by a wide
    // margin — so this is ported id-for-id, not bucketed, matched via `is_mistral_id` above. This
    // codebase has no bespoke Mistral wire client (pi's own `mistral-conversations.ts` speaks a
    // different, non-OpenAI-shaped API); every Mistral id is routed through the generic OpenAI Chat
    // Completions dialect instead — see `dialect::openai::build_body`'s Mistral-specific tool-call-id
    // reshaping and `reasoning_wire_override` below for the two wire-shape adjustments that gap needs.
    if is_mistral_id(&m) {
        let (context_window, max_output, reasoning_effort, supports_vision) = match m.as_str() {
            "codestral-latest" => (256_000, 4_096, false, false),
            "devstral-2512" => (262_144, 262_144, false, false),
            "devstral-latest" => (262_144, 262_144, false, false),
            "devstral-medium-2507" => (128_000, 128_000, false, false),
            "devstral-medium-latest" => (262_144, 262_144, false, false),
            "devstral-small-2505" => (128_000, 128_000, false, false),
            "devstral-small-2507" => (128_000, 128_000, false, false),
            "labs-devstral-small-2512" => (256_000, 256_000, false, true),
            "magistral-medium-latest" => (128_000, 16_384, true, false),
            "magistral-small" => (128_000, 128_000, true, false),
            "ministral-3b-latest" => (128_000, 128_000, false, false),
            "ministral-8b-latest" => (128_000, 128_000, false, false),
            "mistral-large-2411" => (131_072, 16_384, false, false),
            "mistral-large-2512" => (262_144, 262_144, false, true),
            "mistral-large-latest" => (262_144, 262_144, false, true),
            "mistral-medium-2505" => (131_072, 131_072, false, true),
            "mistral-medium-2508" => (262_144, 262_144, false, true),
            "mistral-medium-2604" => (262_144, 262_144, true, true),
            "mistral-medium-3.5" => (262_144, 262_144, true, true),
            "mistral-medium-latest" => (262_144, 262_144, false, true),
            "mistral-nemo" => (128_000, 128_000, false, false),
            "mistral-small-2506" => (128_000, 16_384, false, true),
            "mistral-small-2603" => (256_000, 256_000, true, true),
            "mistral-small-latest" => (256_000, 256_000, true, true),
            "open-mistral-7b" => (8_000, 8_000, false, false),
            "open-mistral-nemo" => (128_000, 128_000, false, false),
            "open-mixtral-8x22b" => (64_000, 64_000, false, false),
            "open-mixtral-8x7b" => (32_000, 32_000, false, false),
            "pixtral-12b" => (128_000, 128_000, false, true),
            "pixtral-large-latest" => (128_000, 128_000, false, true),
            // A future/uncatalogued Mistral id this exact-match table doesn't recognize yet: a
            // reasonable family-wide default (128k/128k, no reasoning, no vision) rather than falling
            // all the way through to `unknown()`'s flatter 4096-token ceiling.
            _ => (128_000, 128_000, false, false),
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            // Not in pi's Together/Cloudflare/NVIDIA/Ant-Ling `supportsLongCacheRetention` denylist.
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort,
            // No Mistral id nulls "off" in pi's catalogue (none carry a `thinkingLevelMap` at all) —
            // every reasoning-capable id can be told to turn reasoning off.
            reasoning_disableable: reasoning_effort,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            // No Mistral id defines an "xhigh" wire value at all.
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh", // unread: Mistral never uses Adaptive shape.
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }

    // Z.ai/GLM: `compat.thinkingFormat: "zai"` — a `thinking:{enabled/disabled}` toggle sent
    // unconditionally, with a `reasoning_effort` string only from GLM-5.2 onward
    // (`supportsReasoningEffort` is `false` on every earlier id in pi's catalogue). `glm-5v*` is the
    // one vision-capable id; every other current id is text-only. Also matches a vendor-slug id (e.g.
    // Together's/HuggingFace's `"zai-org/GLM-5.2"`) via `family_id` — see its own doc comment above.
    if m.starts_with("glm") || family_id.starts_with("glm") {
        // Keyed on whether `m` is slug-shaped at all (not on which disjunct above matched) — same
        // reasoning as the MiniMax branch below, applied uniformly even though no current GLM org
        // slug happens to collide with the literal prefix "glm" the way MiniMax's does.
        let g = if m.contains('/') { family_id } else { m.as_str() };
        let (context_window, max_output, reasoning_effort) = if g.starts_with("glm-5.2") {
            (1_000_000, 131_072, true)
        } else if g.starts_with("glm-4.5-air") {
            (131_072, 98_304, false)
        } else if g.starts_with("glm-4.7") {
            (204_800, 131_072, false)
        } else {
            (200_000, 131_072, false)
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: g.starts_with("glm-5v"),
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            // Every current GLM id sets pi's `compat.zaiToolStream: true` except "glm-4.5-air", the
            // one id both `zai.models.ts` and `zai-coding-cn.models.ts` leave it off for (see
            // `ModelCaps::supports_tool_stream`'s own doc comment).
            supports_tool_stream: !g.starts_with("glm-4.5-air"),
            api: ApiKind::ChatCompletions,
            // glm-5.2+ nulls "minimal" out of its `thinkingLevelMap` (excluded entirely, not just
            // remapped) — every earlier GLM id has no reasoning vocabulary at all, so this floor is
            // unread for them regardless.
            min_reasoning_effort: if g.starts_with("glm-5.2") {
                RE::Low
            } else {
                RE::Minimal
            },
            supports_xhigh_reasoning: reasoning_effort,
            adaptive_xhigh_effort_wire: "max",
            openai_reasoning_format: OpenAiReasoningFormat::Zai,
        };
    }

    // Moonshot/Kimi: pi tags this family with the identical `"deepseek"` thinkingFormat (the same
    // `thinking:{enabled/disabled}` toggle), but `supportsReasoningEffort` is `false` on every current
    // id — no `reasoning_effort` string ever, just the toggle. Only the non-"thinking" ids
    // (0711/0905/turbo-preview previews) have `reasoning: false` in pi's catalogue, so they get
    // `OpenAiReasoningFormat::Standard` instead (with `reasoning_effort: false`, that emits nothing —
    // matching pi's own `model.reasoning` gate). `kimi-k2.7-code*` uniquely has no "off" wire value
    // (`thinkingLevelMap: {"off": null}`); every other reasoning-capable id can be disabled. Also
    // matches a vendor-slug id (e.g. Together's/HuggingFace's `"moonshotai/Kimi-K2.6"`) via
    // `family_id` — see its own doc comment above.
    if m.starts_with("kimi") || family_id.starts_with("kimi") {
        // Keyed on whether `m` is slug-shaped at all — same reasoning as the MiniMax/GLM branches.
        let k = if m.contains('/') { family_id } else { m.as_str() };
        let non_reasoning = k.starts_with("kimi-k2-0711")
            || k.starts_with("kimi-k2-0905")
            || k.starts_with("kimi-k2-turbo-preview");
        let (context_window, max_output) = if k.starts_with("kimi-k2-0711") {
            (131_072, 16_384)
        } else {
            (262_144, 262_144)
        };
        let supports_vision =
            k.starts_with("kimi-k2.5") || k.starts_with("kimi-k2.6") || k.starts_with("kimi-k2.7");
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: !non_reasoning && !k.starts_with("kimi-k2.7-code"),
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: if non_reasoning {
                OpenAiReasoningFormat::Standard
            } else {
                OpenAiReasoningFormat::DeepSeek
            },
        };
    }

    // xAI/Grok: pi's `detectCompat` marks every xAI id `supportsReasoningEffort: false`
    // unconditionally — the reasoning models in this family (grok-4.2x-reasoning, grok-4.3,
    // grok-build) reason on their own, with no client-steerable toggle at all, so `Standard` format
    // with `reasoning_effort: false` correctly emits nothing (matching pi exactly) rather than
    // guessing at a shape xAI doesn't accept. `maxTokensField` auto-detects to
    // `max_completion_tokens` (xAI isn't in pi's `useMaxTokens` allowlist).
    if m.starts_with("grok") {
        let (context_window, max_output, supports_vision) = if m == "grok-3" || m == "grok-3-fast"
        {
            (131_072, 8_192, false)
        } else if m.starts_with("grok-code-fast") {
            (32_768, 8_192, false)
        } else if m.starts_with("grok-build") {
            (256_000, 256_000, true)
        } else {
            (1_000_000, 30_000, true)
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }

    // MiniMax: pi's own catalogue serves this family over the *Anthropic* wire
    // (`api: "anthropic-messages"`, `packages/ai/src/providers/minimax.models.ts`) — but this crate's
    // `Dialect::for_model` (`dialect/mod.rs`) only recognizes a `claude`/`anthropic`-named id as
    // Anthropic, so a `minimax-*` request is routed through this Chat Completions dialect regardless,
    // where its real Anthropic-shaped message format doesn't apply at all. That routing gap is out of
    // this table's scope to fix (`dialect/mod.rs` isn't touched here); this entry at least closes the
    // truncation gap (context/max output) `Agent::new` reads regardless of which dialect ends up
    // serving the request, and deliberately leaves `reasoning_effort`/`openai_reasoning_format` at
    // their inert defaults rather than emit an OpenAI-shaped reasoning toggle a real MiniMax endpoint
    // wouldn't understand. Also matches a vendor-slug id via `family_id` — see its own doc comment
    // above; this one matters even for a *bare* MiniMax-family match, not just a missed one:
    // HuggingFace/Together's real id is `"MiniMaxAI/MiniMax-M3"`, whose lowercased org slug
    // (`"minimaxai"`) itself starts with the literal string `"minimax"` one character early, so
    // matching against the raw `m` here would already (mis)fire this whole branch — just against the
    // wrong per-id sub-case inside it (the flat "else" shape below, not `minimax-m3`'s real one) —
    // without `family_id` correctly isolating the actual suffix first.
    if m.starts_with("minimax") || family_id.starts_with("minimax") {
        // Deliberately keyed on whether `m` is slug-shaped at all (`m.contains('/')`), not on which
        // disjunct above matched: `m.starts_with("minimax")` is *already* (coincidentally) true for
        // "minimaxai/minimax-m3" itself, so selecting on that would silently keep matching the raw,
        // slug-prefixed `m` below instead of the real suffix `family_id` isolated it to.
        let mm = if m.contains('/') { family_id } else { m.as_str() };
        let (context_window, max_output, supports_vision) = if mm.starts_with("minimax-m3") {
            (1_000_000, 128_000, true)
        } else {
            (204_800, 131_072, false)
        };
        return ModelCaps {
            context_window,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            supports_vision,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }

    // Qwen: pi's real behavior differs by host. Hosted on Groq, the one current id
    // (`qwen/qwen3-32b`) takes a bare `reasoning_effort` (no `compat` override ⇒ pi's `detectCompat`
    // default of the plain OpenAI shape). Hosted on Together — the more common route for this family
    // in pi's catalogue — every entry is `compat.thinkingFormat: "together"` instead. Matched by exact
    // id first so the one Groq case doesn't get swallowed by the generic Together-shaped default below.
    if m == "qwen/qwen3-32b" {
        return ModelCaps {
            context_window: 131_072,
            max_output: 40_960,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::High,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }
    // In practice every Together/HuggingFace-hosted Qwen id's own org slug is literally "qwen/…", so
    // `m.starts_with("qwen")` already matches these without needing `family_id` — but check it too
    // (harmless: `family_id` reduces to the same suffix either way) so a future host whose org slug
    // *doesn't* happen to start with "qwen" isn't silently missed the way Kimi/GLM/MiniMax/DeepSeek's
    // vendor-slug ids used to be.
    if m.starts_with("qwen") || family_id.starts_with("qwen") {
        return ModelCaps {
            context_window: 200_000,
            max_output: 40_960,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: false,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Together,
        };
    }

    // Cerebras-native ids (no vendor-slug prefix — distinct from Groq's own "openai/gpt-oss-*",
    // matched below): all three current ids share the same 131k/40960 shape and take a bare
    // `reasoning_effort` (Cerebras isn't excluded from pi's `supportsReasoningEffort` default).
    if m == "gpt-oss-120b"
        || m == "gpt-oss-20b"
        || m == "gpt-oss-safeguard-20b"
        || m == "gemma-4-31b"
        || m == "zai-glm-4.7"
    {
        return ModelCaps {
            context_window: 131_072,
            max_output: 40_960,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: m == "gemma-4-31b",
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }

    // Groq-hosted open models under the vendor's own OpenAI-compatible endpoint. `openai/gpt-oss-*`
    // (Groq's id, vendor-prefixed — distinct from Cerebras's un-prefixed "gpt-oss-*" above) is
    // reasoning-capable; every Llama id on Groq is not.
    if m.contains("gpt-oss") {
        return ModelCaps {
            context_window: 131_072,
            max_output: 65_536,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }
    if m.starts_with("llama") || m.contains("/llama") {
        return ModelCaps {
            context_window: 131_072,
            max_output: 32_768,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: m.contains("llama-4"),
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::Standard,
        };
    }

    // OpenRouter — and the generic fallback for any other vendor/model-shaped id (`vendor/model`) this
    // table doesn't otherwise recognize (most commonly an uncatalogued Together/Groq/Cerebras
    // addition). pi tags virtually every OpenRouter catalogue entry `compat.thinkingFormat:
    // "openrouter"` with `supportsReasoningEffort` left at its permissive default (`true`); a nested
    // `reasoning:{effort}` object is a reasonably safe shape to send even to a provider that doesn't
    // recognize it (an extra top-level field a permissive JSON API typically ignores, rather than a
    // field substitution that could 400 outright). Deliberately last, so a recognizable family
    // (deepseek/glm/kimi/grok/qwen/gpt-oss/llama) always gets its own real numbers first even when
    // reached with a vendor-slug id.
    if m.contains('/') {
        return ModelCaps {
            context_window: 128_000,
            max_output: 32_000,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: false,
            supports_temperature: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            supports_tool_stream: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: RE::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
            openai_reasoning_format: OpenAiReasoningFormat::OpenRouter,
        };
    }

    tracing::warn!(
        model,
        "unrecognized model id; falling back to conservative capabilities"
    );
    ModelCaps::unknown()
}

/// Per-model reasoning-effort wire-string remap for the OpenAI Chat-Completions dialect's third-party
/// reasoning-toggle shapes (`dialect::openai::apply_reasoning_wire`) — mirrors pi's per-model
/// `thinkingLevelMap` (`packages/ai/src/providers/*.models.ts`), which several providers use to spell a
/// given portable [`crate::transport::ReasoningEffort`] rung differently on the wire than that rung's
/// own name. [`ModelCaps::adaptive_xhigh_effort_wire`] already exists for the identical need on the
/// Anthropic adaptive-thinking shape, but it's a single xhigh-only string — insufficient here, where a
/// model can remap several rungs at once (GLM-5.2 remaps three: low, medium, and xhigh) — so this is a
/// standalone lookup instead of a new `ModelCaps` field, which would need touching every one of this
/// file's ~25 existing `ModelCaps` struct-literal construction sites for a table only four families
/// actually use.
///
/// `effort` is the *already-clamped* [`crate::transport::ReasoningEffort`] ([`clamp_reasoning_effort`]'s
/// output) — this function only remaps how an already-legal level is spelled on the wire, never decides
/// whether a level is legal in the first place (that's still `min_reasoning_effort`/
/// `supports_xhigh_reasoning`). `None` — the overwhelming common case — leaves the caller to fall back
/// to the clamped effort's own literal name, unchanged from before this function existed.
pub fn reasoning_wire_override(
    model: &str,
    effort: crate::transport::ReasoningEffort,
) -> Option<&'static str> {
    use crate::transport::ReasoningEffort as RE;
    let m = model.to_ascii_lowercase();
    // Matched the same way `capabilities` matches its own family branches — see `family_id`'s doc
    // comment there — so a vendor-slug id (`"zai-org/glm-5.2"`, Together/HuggingFace) remaps
    // identically to the bare id it's slug-prefixed with, instead of silently keeping the clamped
    // effort's own literal name just because the full id doesn't start with the family's prefix.
    let family_id: &str = match m.rfind('/') {
        Some(idx) => &m[idx + 1..],
        None => &m,
    };
    // DeepSeek: `thinkingLevelMap` nulls minimal/low/medium (already reflected in
    // `min_reasoning_effort: High`, so only High/XHigh can ever reach here); xhigh alone remaps, to
    // "max".
    if m.starts_with("deepseek") || family_id.starts_with("deepseek") {
        return (effort == RE::XHigh).then_some("max");
    }
    // GLM-5.2+: low/medium/high all collapse to the literal "high", and xhigh remaps to "max". Every
    // earlier GLM id has no reasoning_effort vocabulary at all (`caps.reasoning_effort == false`), so
    // `apply_reasoning_wire` never calls this for them regardless.
    if m.starts_with("glm-5.2") || family_id.starts_with("glm-5.2") {
        return match effort {
            RE::Low | RE::Medium | RE::High => Some("high"),
            RE::XHigh => Some("max"),
            // Excluded by `min_reasoning_effort: Low` above — never actually reaches here.
            RE::Minimal => None,
        };
    }
    // Groq's one qwen id: the only level it ever accepts (`min_reasoning_effort: High`,
    // `supports_xhigh_reasoning: false` clamp everything else away first) remaps from "high" to
    // "default".
    if m == "qwen/qwen3-32b" {
        return (effort == RE::High).then_some("default");
    }
    // Mistral reasoning-capable ids: pi's own client (`mistral-conversations.ts::mapReasoningEffort`)
    // has no per-level vocabulary for any Mistral id at all — none carry a `thinkingLevelMap` — so
    // every active level falls back to that function's own hardcoded default, "high", literally
    // regardless of which portable level was requested; only "off" ever omits the field. Mirrored here
    // rather than sending "minimal"/"low"/etc, which Mistral's real `reasoning_effort` wire vocabulary
    // (a bare `"none" | "high"` enum) doesn't recognize at all. Only ever reached when
    // `caps.reasoning_effort` is already true for the id (the caller's own gate), so no non-reasoning
    // Mistral id (e.g. "mistral-large-latest") ever hits this branch.
    if is_mistral_id(&m) {
        return Some("high");
    }
    None
}

/// A portable thinking-depth level, independent of which wire mechanism the active model actually uses
/// — an Anthropic token budget (`ThinkingShape::Budget`), an Anthropic adaptive effort
/// (`ThinkingShape::Adaptive`), or an OpenAI `reasoning_effort` parameter. A client (or
/// `cycle_thinking_level`) can move through these six rungs and land at a comparable depth no matter
/// which model is currently active — see [`thinking_for_level`] for the translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingLevel {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

/// Ladder order `ThinkingLevel::next` cycles through, wrapping from `XHigh` back to `Off`.
const THINKING_LEVEL_LADDER: [ThinkingLevel; 6] = [
    ThinkingLevel::Off,
    ThinkingLevel::Minimal,
    ThinkingLevel::Low,
    ThinkingLevel::Medium,
    ThinkingLevel::High,
    ThinkingLevel::XHigh,
];

impl ThinkingLevel {
    /// The wire string this level round-trips as: `"off"`, plus `ReasoningEffort::as_str`'s vocabulary
    /// for the other five.
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingLevel::Off => "off",
            ThinkingLevel::Minimal => "minimal",
            ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
            ThinkingLevel::XHigh => "xhigh",
        }
    }

    /// Parse [`Self::as_str`]'s vocabulary (case-sensitive, matching `ReasoningEffort`'s own wire
    /// strings). `None` for anything else.
    pub fn parse(s: &str) -> Option<Self> {
        THINKING_LEVEL_LADDER
            .iter()
            .copied()
            .find(|l| l.as_str() == s)
    }

    /// The next rung, wrapping from `XHigh` back to `Off`.
    pub fn next(self) -> Self {
        let idx = THINKING_LEVEL_LADDER
            .iter()
            .position(|&l| l == self)
            .unwrap_or(0);
        THINKING_LEVEL_LADDER[(idx + 1) % THINKING_LEVEL_LADDER.len()]
    }

    /// The `ReasoningEffort` this level carries, or `None` for `Off`.
    pub fn reasoning_effort(self) -> Option<crate::transport::ReasoningEffort> {
        use crate::transport::ReasoningEffort as RE;
        match self {
            ThinkingLevel::Off => None,
            ThinkingLevel::Minimal => Some(RE::Minimal),
            ThinkingLevel::Low => Some(RE::Low),
            ThinkingLevel::Medium => Some(RE::Medium),
            ThinkingLevel::High => Some(RE::High),
            ThinkingLevel::XHigh => Some(RE::XHigh),
        }
    }
}

impl From<crate::transport::ReasoningEffort> for ThinkingLevel {
    fn from(effort: crate::transport::ReasoningEffort) -> Self {
        use crate::transport::ReasoningEffort as RE;
        match effort {
            RE::Minimal => ThinkingLevel::Minimal,
            RE::Low => ThinkingLevel::Low,
            RE::Medium => ThinkingLevel::Medium,
            RE::High => ThinkingLevel::High,
            RE::XHigh => ThinkingLevel::XHigh,
        }
    }
}

/// The built-in effort→token-budget ladder, before any caller override or the `max_output` clamp.
/// Factored out of [`budget_for_effort`]/[`budget_for_effort_with_override`] so there's exactly one
/// literal source of truth for the fixed numbers.
fn default_budget_for_effort(effort: crate::transport::ReasoningEffort) -> u32 {
    use crate::transport::ReasoningEffort as RE;
    match effort {
        RE::Minimal => 1_024,
        RE::Low => 2_048,
        RE::Medium => 8_192,
        // pi: `packages/ai/src/api/simple-options.ts`'s `defaultBudgets.high` is 16384, not 24000.
        RE::High => 16_384,
        RE::XHigh => 32_000,
    }
}

/// A fixed effort→token-budget ladder for [`ThinkingShape::Budget`]-shape models, clamped below
/// `max_output` (a thinking budget must leave room for the turn's actual output). For
/// [`ThinkingShape::Adaptive`]-shape models this value is a pure on/off gate on the wire — only its
/// `Some`-ness is checked, the actual depth comes from `output_config.effort` (see
/// `dialect::anthropic::build_body`) — so its exact magnitude doesn't matter there, but reusing the same
/// ladder keeps one code path instead of two.
///
/// [`thinking_for_level`] (the only internal caller) always uses the built-in ladder — equivalent to
/// calling [`budget_for_effort_with_override`] with `overrides: None`. A caller that has an
/// operator-supplied override table should call that sibling directly instead.
fn budget_for_effort(effort: crate::transport::ReasoningEffort, max_output: u32) -> u32 {
    budget_for_effort_with_override(effort, max_output, None)
}

/// Same as [`budget_for_effort`], but consults `overrides` first — the extension point a settings/CLI
/// layer can wire an operator-supplied thinking-token budget through (e.g. `--thinking-budget
/// high=40000`), without this crate needing any opinion on where that table comes from. An effort level
/// missing from `overrides` (or `overrides: None` entirely) falls back to
/// [`default_budget_for_effort`]'s fixed ladder unchanged. The `max_output` clamp applies either way —
/// an override still can't leave a turn with zero room for its actual output.
pub fn budget_for_effort_with_override(
    effort: crate::transport::ReasoningEffort,
    max_output: u32,
    overrides: Option<&std::collections::HashMap<crate::transport::ReasoningEffort, u32>>,
) -> u32 {
    let budget = overrides
        .and_then(|table| table.get(&effort))
        .copied()
        .unwrap_or_else(|| default_budget_for_effort(effort));
    budget.min(max_output.saturating_sub(1).max(1))
}

/// Clamp a requested [`ReasoningEffort`](crate::transport::ReasoningEffort) to what `caps`'s model
/// actually accepts on the wire — pi's `clampThinkingLevel`, specialized to the two edges every current
/// model's exclusions actually trim (never a gap in the middle): a floor (`min_reasoning_effort`, e.g.
/// `gpt-5.5-pro` rejects `minimal`/`low`) and an `xhigh` ceiling (`supports_xhigh_reasoning`, e.g.
/// o-series/bare-gpt-5/gpt-5.1\* and Anthropic sonnet-4-6/sonnet-5 top out at `high`). Called by every
/// dialect at the point it's about to put an effort string on the wire, rather than by
/// [`thinking_for_level`], so a raw `with_reasoning_effort`/`--reasoning-effort` call (which never goes
/// through `thinking_for_level`) is clamped too.
pub fn clamp_reasoning_effort(
    caps: &ModelCaps,
    effort: crate::transport::ReasoningEffort,
) -> crate::transport::ReasoningEffort {
    use crate::transport::ReasoningEffort as RE;
    let mut e = effort.max(caps.min_reasoning_effort);
    if e == RE::XHigh && !caps.supports_xhigh_reasoning {
        e = RE::High;
    }
    e
}

/// Whether `caps`'s model has *any* client-steerable thinking/reasoning mechanism at all — an
/// Anthropic `thinking` shape, a bare OpenAI `reasoning_effort`, or one of the third-party
/// [`OpenAiReasoningFormat`] toggle shapes (DeepSeek/Zai/Together/OpenRouter). The third arm matters
/// because several of those families (Kimi, GLM below 5.2) set `reasoning_effort: false` — they have no
/// graduated effort vocabulary — while still exposing a real on/off toggle
/// (`dialect::openai::build_body`'s family-specific branches send it regardless of `reasoning_effort`);
/// without this arm such a model would be misreported as having no mechanism at all, `Off`-locked, even
/// though `--thinking high` genuinely turns its reasoning on. Shared by [`available_thinking_levels`],
/// [`clamp_thinking_level`], and [`thinking_for_level`] so the three stay consistent with each other.
///
/// `pub` (not module-private): `crates/agent`'s own `default_reasoning_effort_for_model`/`model_info`
/// call this directly instead of maintaining their own narrower copy of the same check. A prior version
/// of both did exactly that — checking only `reasoning_effort`/`thinking` — which silently missed the
/// third arm above: Kimi-thinking and pre-5.2 GLM models (`openai_reasoning_format` toggle shapes with
/// `reasoning_effort: false` and `thinking: ThinkingShape::None`) read as "no reasoning mechanism at
/// all" under that narrower check, so a bare invocation picked no default reasoning effort for them at
/// all, even though pi's own reference defaults every reasoning-capable model — this family included —
/// to medium effort.
pub fn has_reasoning_mechanism(caps: &ModelCaps) -> bool {
    caps.reasoning_effort
        || caps.thinking != ThinkingShape::None
        || caps.openai_reasoning_format != OpenAiReasoningFormat::Standard
}

/// The [`ThinkingLevel`] rungs `caps`'s model actually accepts, in ladder order — `Off` excluded when
/// the model has a thinking/reasoning mechanism it can't explicitly disable
/// (`reasoning_disableable == false`), `XHigh` excluded when `!supports_xhigh_reasoning`, and any rung
/// below `min_reasoning_effort` excluded. A model with no thinking/reasoning mechanism at all
/// (see [`has_reasoning_mechanism`]) has exactly one available level, `Off` — nothing else could ever
/// mean anything to it. Used by [`clamp_thinking_level`] (via containment) and
/// [`next_available_thinking_level`] (to cycle only through what's real for this model, instead of the
/// raw 6-rung ladder).
pub fn available_thinking_levels(caps: &ModelCaps) -> Vec<ThinkingLevel> {
    if !has_reasoning_mechanism(caps) {
        return vec![ThinkingLevel::Off];
    }
    THINKING_LEVEL_LADDER
        .iter()
        .copied()
        .filter(|&level| match level {
            ThinkingLevel::Off => caps.reasoning_disableable,
            ThinkingLevel::XHigh => caps.supports_xhigh_reasoning,
            _ => level
                .reasoning_effort()
                .is_some_and(|effort| effort >= caps.min_reasoning_effort),
        })
        .collect()
}

/// Clamp a portable [`ThinkingLevel`] to the nearest rung `caps`'s model actually accepts — the
/// ladder-wide counterpart to [`clamp_reasoning_effort`], additionally handling `Off`: a model that
/// can't explicitly disable a reasoning mechanism it has (`reasoning_disableable == false`) has no
/// legal `Off` state at all — pi's `thinkingLevelMap.off === null` — so requesting it bumps up to the
/// model's own floor (`min_reasoning_effort`) instead of silently omitting the reasoning field and
/// leaving the provider to apply its own hidden default. Mirrors pi's `clampThinkingLevel`/
/// `getSupportedThinkingLevels`, called on every model switch (`sdk.ts:241`) and every level change
/// (`agent-session.ts` `setThinkingLevel`/`cycleThinkingLevel`). Every non-`Off` rung reuses
/// [`clamp_reasoning_effort`]'s existing floor/`xhigh`-ceiling behavior unchanged.
///
/// Callers: every point that sets a session's active thinking level for a given model — the initial
/// level at startup, `set_model`/`cycle_model` (re-clamp the *existing* level against the *new*
/// model), `set_reasoning_effort`, and branch-switch model/level restoration. `cycle_thinking_level`
/// uses [`next_available_thinking_level`] instead (see its own doc comment for why a plain
/// `next().then(clamp)` gets stuck for models missing `xhigh`).
pub fn clamp_thinking_level(caps: &ModelCaps, level: ThinkingLevel) -> ThinkingLevel {
    match level.reasoning_effort() {
        None => {
            // `level == ThinkingLevel::Off` — the only variant with no `ReasoningEffort`.
            if has_reasoning_mechanism(caps) && !caps.reasoning_disableable {
                ThinkingLevel::from(caps.min_reasoning_effort)
            } else {
                ThinkingLevel::Off
            }
        }
        Some(effort) => ThinkingLevel::from(clamp_reasoning_effort(caps, effort)),
    }
}

/// The rung `cycle_thinking_level` should land on next, for `caps`'s model, from `level`. Advances
/// through [`available_thinking_levels`] — not the raw 6-rung ladder via [`ThinkingLevel::next`] — so
/// cycling always reaches every level the model actually supports and wraps cleanly at its own ends,
/// rather than getting stuck: a naive `level.next()` then [`clamp_thinking_level`] would bounce forever
/// between `High` and a re-clamped `XHigh` for any model lacking `xhigh` support, since `XHigh` always
/// clamps back down to the same `High` it started from and the ladder never advances past it. If
/// `level` itself isn't currently available (e.g. state predates a model switch that narrowed the set),
/// lands on its clamp first rather than jumping an extra rung ahead.
pub fn next_available_thinking_level(caps: &ModelCaps, level: ThinkingLevel) -> ThinkingLevel {
    let available = available_thinking_levels(caps);
    match available.iter().position(|&l| l == level) {
        Some(i) => available[(i + 1) % available.len()],
        None => clamp_thinking_level(caps, level),
    }
}

/// The Anthropic adaptive-thinking `output_config.effort` wire string for a (clamped) reasoning
/// effort, mirroring pi's `mapThinkingLevelToEffort`: Anthropic's adaptive shape has no `minimal` tier
/// at all (always sent as `"low"`), and `xhigh` is model-specific (`adaptive_xhigh_effort_wire`) —
/// everything else is the effort's own name. Only meaningful for `ThinkingShape::Adaptive`; callers
/// should clamp with [`clamp_reasoning_effort`] first so a model that doesn't support `xhigh` never
/// reaches this function still holding it.
pub fn anthropic_adaptive_effort_wire(
    caps: &ModelCaps,
    effort: crate::transport::ReasoningEffort,
) -> &'static str {
    use crate::transport::ReasoningEffort as RE;
    match clamp_reasoning_effort(caps, effort) {
        RE::Minimal | RE::Low => "low",
        RE::Medium => "medium",
        RE::High => "high",
        RE::XHigh => caps.adaptive_xhigh_effort_wire,
    }
}

/// Translate a portable `level` into the `(thinking_budget, reasoning_effort)` pair
/// [`crate::Agent::with_thinking`]/[`crate::Agent::with_reasoning_effort`] need for a model with
/// capabilities `caps` — whichever combination its `thinking`/`reasoning_effort` capabilities actually
/// call for:
/// - `ThinkingShape::None` and no OpenAI `reasoning_effort` support: neither field is ever set — the
///   model has no thinking/reasoning mechanism at all.
/// - `ThinkingShape::Budget` (Claude 3.7/4.x): a token budget scaled from `level`, clamped below
///   `max_output`; `reasoning_effort` stays unset (that dialect arm never reads it).
/// - `ThinkingShape::Adaptive` (gen-6+ Claude/Fable) and OpenAI's `reasoning_effort: true` models both
///   want `reasoning_effort` set — for `Adaptive`, `thinking` is *also* set (any nonzero budget; it's
///   purely the on/off gate for that shape, see `budget_for_effort`'s doc comment) since Anthropic's
///   `output_config.effort` sibling field only takes effect when thinking itself is enabled.
pub fn thinking_for_level(
    caps: &ModelCaps,
    level: ThinkingLevel,
) -> (Option<u32>, Option<crate::transport::ReasoningEffort>) {
    let Some(effort) = level.reasoning_effort() else {
        return (None, None);
    };
    let thinking = matches!(
        caps.thinking,
        ThinkingShape::Budget | ThinkingShape::Adaptive
    )
    .then(|| budget_for_effort(effort, caps.max_output));
    // Populated whenever the model has *any* steerable mechanism (see `has_reasoning_mechanism`) —
    // not just `reasoning_effort`/`Adaptive` — so a toggle-only third-party family (Kimi, GLM below
    // 5.2: `reasoning_effort: false` but a real `openai_reasoning_format` toggle) still gets
    // `req.reasoning_effort` set, letting `dialect::openai::build_body`'s family-specific branch turn
    // the toggle on even though it never puts a graduated effort string on the wire for these ids.
    let reasoning_effort_applies = caps.reasoning_effort
        || caps.thinking == ThinkingShape::Adaptive
        || caps.openai_reasoning_format != OpenAiReasoningFormat::Standard;
    let reasoning_effort = reasoning_effort_applies.then_some(effort);
    (thinking, reasoning_effort)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_gen6_plus_is_adaptive_with_1m_context() {
        // Our own default model: adaptive thinking, 1M context, 128k output.
        let c = capabilities("claude-opus-4-8");
        assert_eq!(c.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(c.supports_long_cache);
        assert_eq!(c.thinking, ThinkingShape::Adaptive);
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(c.max_output, 128_000);
        assert!(!c.reasoning_effort);

        for id in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-sonnet-5",
            "claude-fable-5",
        ] {
            let c = capabilities(id);
            assert_eq!(
                c.thinking,
                ThinkingShape::Adaptive,
                "{id} should be adaptive"
            );
            assert_eq!(c.context_window, 1_000_000, "{id} should have 1M context");
        }
        // sonnet-4-6 is adaptive but caps output at 64k, not 128k.
        let c = capabilities("claude-sonnet-4-6");
        assert_eq!(c.thinking, ThinkingShape::Adaptive);
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(c.max_output, 64_000);
    }

    #[test]
    fn only_opus_4_7_and_opus_4_8_reject_temperature() {
        // pi: `compat.supportsTemperature: false` is set on exactly these two ids in
        // `anthropic.models.ts`; every other current id (including the rest of gen6+, legacy Budget-
        // shape Claude, and every OpenAI-wire model) defaults to supporting it.
        for id in ["claude-opus-4-7", "claude-opus-4-8"] {
            assert!(
                !capabilities(id).supports_temperature,
                "{id} should reject temperature"
            );
        }
        for id in [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-3-5-sonnet-20241022",
            "gpt-4o",
            "o3",
        ] {
            assert!(
                capabilities(id).supports_temperature,
                "{id} should still support temperature"
            );
        }
    }

    #[test]
    fn claude_pre_gen6_stays_budget_shaped() {
        // opus-4-5 and sonnet-4-5 predate the adaptive requirement — still `Budget`, 200k context.
        for id in ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"] {
            let c = capabilities(id);
            assert_eq!(
                c.thinking,
                ThinkingShape::Budget,
                "{id} should stay Budget-shaped"
            );
            assert_eq!(
                c.context_window, 200_000,
                "{id} should stay at 200k context"
            );
        }
        assert_eq!(capabilities("claude-opus-4-5").max_output, 64_000);
        assert_eq!(capabilities("claude-sonnet-4-5").max_output, 64_000);
        assert_eq!(capabilities("claude-haiku-4-5").max_output, 64_000);
        assert_eq!(capabilities("claude-opus-4-1").max_output, 32_000);
    }

    #[test]
    fn claude_gen3_legacy_has_no_thinking() {
        for id in [
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
        ] {
            let c = capabilities(id);
            assert_eq!(
                c.thinking,
                ThinkingShape::None,
                "{id} predates extended thinking"
            );
            // No thinking support doesn't mean no long-cache support — prompt caching (including the
            // 1-hour TTL) is a separate, orthogonal capability that every current Claude 3.x id has.
            assert!(c.supports_long_cache, "{id} does support the 1h cache TTL");
        }
        // 3.7-sonnet is the exception: it *does* support extended thinking (Budget shape).
        assert_eq!(
            capabilities("claude-3-7-sonnet-20250219").thinking,
            ThinkingShape::Budget
        );
    }

    #[test]
    fn openai_reasoning_models_use_completion_tokens_field() {
        for id in ["o1", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"] {
            let c = capabilities(id);
            assert_eq!(
                c.max_tokens_field,
                MaxTokensField::MaxCompletionTokens,
                "{id} should use max_completion_tokens"
            );
            assert!(c.reasoning_effort, "{id} should take reasoning_effort");
            assert!(
                c.supports_long_cache,
                "{id} should default to supporting prompt_cache_retention (pi's denylist default)"
            );
            assert_eq!(c.thinking, ThinkingShape::None);
        }
    }

    #[test]
    fn openai_context_windows_match_family() {
        assert_eq!(capabilities("o1").context_window, 200_000);
        assert_eq!(capabilities("o1").max_output, 100_000);
        assert_eq!(capabilities("gpt-5").context_window, 400_000);
        assert_eq!(capabilities("gpt-5-mini").context_window, 400_000);
        assert_eq!(capabilities("gpt-5.4").context_window, 272_000);
        assert_eq!(capabilities("gpt-5.4-mini").context_window, 400_000);
        assert_eq!(capabilities("gpt-5.4-pro").context_window, 1_050_000);
        assert_eq!(
            capabilities("gpt-5-chat-latest").thinking,
            ThinkingShape::None
        );
        assert!(!capabilities("gpt-5-chat-latest").reasoning_effort);
    }

    #[test]
    fn o1_mini_is_no_longer_special_cased() {
        // LOW pi-parity gap (fixed): o1-mini used to be excluded from `supports_vision` alongside
        // o3-mini, but pi's live catalogue has fully retired the id — nothing upstream still serves
        // it, so it now falls through like any other (still-live) o-series id.
        assert!(capabilities("o1-mini").supports_vision);
    }

    #[test]
    fn o3_mini_is_text_only() {
        // The one o-series id that isn't vision-capable (pi: `input: ["text"]`); every other
        // o-series id is vision-capable.
        assert!(!capabilities("o3-mini").supports_vision);
        assert!(capabilities("o1").supports_vision);
        assert!(capabilities("o4-mini").supports_vision);
    }

    #[test]
    fn gpt5_pro_context_window_only_1_05m_for_5_4_and_5_5() {
        // Only the 5.4/5.5 generation of "-pro" gets the larger window; earlier -pro ids share the
        // rest of the family's 400k.
        assert_eq!(capabilities("gpt-5-pro").context_window, 400_000);
        assert_eq!(capabilities("gpt-5.2-pro").context_window, 400_000);
        assert_eq!(capabilities("gpt-5.4-pro").context_window, 1_050_000);
        assert_eq!(capabilities("gpt-5.5-pro").context_window, 1_050_000);
    }

    #[test]
    fn gpt_5_3_codex_spark_is_narrower_than_the_rest_of_the_family() {
        let c = capabilities("gpt-5.3-codex-spark");
        assert_eq!(c.context_window, 128_000);
        assert_eq!(c.max_output, 32_000);
    }

    #[test]
    fn chat_latest_reasoning_effort_is_id_specific() {
        // 5.1/5.2-chat-latest are still reasoning_effort-driven per pi's catalogue; the plain and
        // 5.3 chat-latest ids are not.
        assert!(capabilities("gpt-5.1-chat-latest").reasoning_effort);
        assert!(capabilities("gpt-5.2-chat-latest").reasoning_effort);
        assert!(!capabilities("gpt-5-chat-latest").reasoning_effort);
        assert!(!capabilities("gpt-5.3-chat-latest").reasoning_effort);
    }

    #[test]
    fn gpt4o_2024_05_13_pinned_snapshot_caps_output_at_4096() {
        // This one dated snapshot caps output tighter (4096) than every other 4o-family id (16384).
        assert_eq!(capabilities("gpt-4o-2024-05-13").max_output, 4_096);
        assert_eq!(capabilities("gpt-4o").max_output, 16_384);
    }

    #[test]
    fn reasoning_disableable_is_id_specific_not_family_wide() {
        // Anthropic: every gen6+ id can be told to disable thinking explicitly, except claude-fable-5.
        assert!(capabilities("claude-opus-4-8").reasoning_disableable);
        assert!(capabilities("claude-sonnet-5").reasoning_disableable);
        assert!(!capabilities("claude-fable-5").reasoning_disableable);
        assert!(!capabilities("fable-5").reasoning_disableable);
        // Budget-shaped (pre-gen6, post-gen3) Claude ids are disable-capable too.
        assert!(capabilities("claude-opus-4-5").reasoning_disableable);
        // Gen-3 legacy ids have no thinking support at all — nothing to disable.
        assert!(!capabilities("claude-3-5-sonnet-20241022").reasoning_disableable);

        // OpenAI o-series: disable-capable by default.
        assert!(capabilities("o3-mini").reasoning_disableable);
        assert!(capabilities("o1").reasoning_disableable);

        // OpenAI gpt-5: only the exact-match allowlist is disable-capable.
        assert!(capabilities("gpt-5.1").reasoning_disableable);
        assert!(capabilities("gpt-5.2").reasoning_disableable);
        assert!(capabilities("gpt-5.3-codex").reasoning_disableable);
        assert!(capabilities("gpt-5.4").reasoning_disableable);
        assert!(capabilities("gpt-5.4-mini").reasoning_disableable);
        assert!(capabilities("gpt-5.4-nano").reasoning_disableable);
        assert!(capabilities("gpt-5.5").reasoning_disableable);
        assert!(
            !capabilities("gpt-5").reasoning_disableable,
            "bare gpt-5 isn't in the allowlist"
        );
        assert!(
            !capabilities("gpt-5.1-chat-latest").reasoning_disableable,
            "chat-latest ids have no off signal even though they take reasoning_effort"
        );
        assert!(!capabilities("gpt-5.3-codex-spark").reasoning_disableable);

        // OpenAI gpt-4 family never takes reasoning_effort — nothing to disable.
        assert!(!capabilities("gpt-4o").reasoning_disableable);
        assert!(!capabilities("gpt-4.1").reasoning_disableable);
    }

    #[test]
    fn supports_eager_tool_streaming_is_anthropic_only() {
        // Every current Anthropic id (across all three generational branches) supports the per-tool
        // eager-input-streaming shape.
        for id in [
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-5",
        ] {
            assert!(
                capabilities(id).supports_eager_tool_streaming,
                "{id} should support eager tool streaming"
            );
        }
        // No current OpenAI id does — it's an Anthropic-wire-only concept.
        for id in [
            "o1",
            "gpt-5",
            "gpt-5-chat-latest",
            "gpt-5.3-codex-spark",
            "gpt-4o",
            "gpt-4.1",
        ] {
            assert!(
                !capabilities(id).supports_eager_tool_streaming,
                "{id} should not support eager tool streaming"
            );
        }
    }

    #[test]
    fn gpt4_uses_plain_max_tokens() {
        let c = capabilities("gpt-4o");
        assert_eq!(c.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(!c.reasoning_effort);
        assert!(c.supports_vision);
        assert_eq!(c.context_window, 128_000);
        assert!(c.supports_long_cache);

        // 4.1 gets the ~1M-token window; bare "gpt-4" stays at the original 8k.
        assert_eq!(capabilities("gpt-4.1").context_window, 1_047_576);
        assert_eq!(capabilities("gpt-4.1-mini").context_window, 1_047_576);
        assert_eq!(capabilities("gpt-4").context_window, 8_192);
    }

    #[test]
    fn bare_gpt4_predates_vision_support_unlike_every_other_gpt4_id() {
        // Bare "gpt-4" (pi's catalogue: `input: ["text"]`) is the original, text-only model — vision
        // shipped later as "gpt-4-vision-preview" and was folded into "gpt-4-turbo"/4o onward. Getting
        // this wrong would let the loop attach an image to a request the model can't accept at all.
        assert!(
            !capabilities("gpt-4").supports_vision,
            "bare gpt-4 must not be marked vision-capable"
        );
        for id in ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.1"] {
            assert!(
                capabilities(id).supports_vision,
                "{id} should still be vision-capable"
            );
        }
    }

    #[test]
    fn unknown_model_is_conservative() {
        let c = capabilities("some-future-model-x");
        assert_eq!(c.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(!c.supports_long_cache);
        assert_eq!(c.thinking, ThinkingShape::None);
    }

    #[test]
    fn matching_is_case_insensitive() {
        assert_eq!(
            capabilities("Claude-Opus-4-8").thinking,
            ThinkingShape::Adaptive
        );
    }

    #[test]
    fn native_openai_ids_speak_responses_api() {
        // Every native OpenAI id (o-series, gpt-5 family incl. chat-latest, gpt-4 family) routes
        // through the Responses API per pi's live catalogue — no exceptions among current ids.
        for id in [
            "o1",
            "o3-mini",
            "gpt-5",
            "gpt-5-chat-latest",
            "gpt-5.4-pro",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4-turbo",
            "gpt-4",
        ] {
            assert_eq!(
                capabilities(id).api,
                ApiKind::Responses,
                "{id} should speak the Responses API"
            );
        }
    }

    #[test]
    fn claude_and_unknown_ids_default_to_chat_completions_api() {
        // Anthropic ids ignore `api` (they always speak Dialect::Anthropic), but the field must still
        // resolve to a harmless default rather than accidentally inheriting `Responses`.
        assert_eq!(
            capabilities("claude-opus-4-8").api,
            ApiKind::ChatCompletions
        );
        // Third-party OpenAI-compatible ids (and anything unrecognized) stay on Chat Completions —
        // only native OpenAI ids get the Responses API.
        assert_eq!(capabilities("llama-3.1-70b").api, ApiKind::ChatCompletions);
        assert_eq!(
            capabilities("some-future-model-x").api,
            ApiKind::ChatCompletions
        );
    }

    #[test]
    fn thinking_level_next_cycles_through_all_six_rungs_and_wraps() {
        let mut level = ThinkingLevel::Off;
        let mut seen = vec![level];
        for _ in 0..5 {
            level = level.next();
            seen.push(level);
        }
        assert_eq!(
            seen,
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh,
            ]
        );
        assert_eq!(
            level.next(),
            ThinkingLevel::Off,
            "the ladder must wrap from XHigh back to Off"
        );
    }

    #[test]
    fn thinking_level_as_str_and_parse_round_trip() {
        for level in [
            ThinkingLevel::Off,
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::XHigh,
        ] {
            assert_eq!(ThinkingLevel::parse(level.as_str()), Some(level));
        }
        assert_eq!(ThinkingLevel::parse("nonsense"), None);
    }

    #[test]
    fn thinking_for_level_is_off_for_a_model_with_no_thinking_mechanism_at_all() {
        // Legacy gen-3 Claude ids: `ThinkingShape::None`, `reasoning_effort: false`.
        let caps = capabilities("claude-3-5-sonnet-20241022");
        assert_eq!(caps.thinking, ThinkingShape::None);
        for level in [
            ThinkingLevel::Minimal,
            ThinkingLevel::High,
            ThinkingLevel::XHigh,
        ] {
            assert_eq!(
                thinking_for_level(&caps, level),
                (None, None),
                "a model with no thinking mechanism must never get either field set"
            );
        }
    }

    #[test]
    fn thinking_for_level_off_always_clears_both_fields_regardless_of_shape() {
        for id in ["claude-opus-4-5", "claude-opus-4-8", "o3", "gpt-4o"] {
            assert_eq!(
                thinking_for_level(&capabilities(id), ThinkingLevel::Off),
                (None, None),
                "{id}: Off must clear both fields"
            );
        }
    }

    #[test]
    fn thinking_for_level_sets_only_a_budget_for_budget_shape_models() {
        // "claude-opus-4-5" is pre-gen6: `ThinkingShape::Budget`, no OpenAI-style `reasoning_effort`.
        let caps = capabilities("claude-opus-4-5");
        assert_eq!(caps.thinking, ThinkingShape::Budget);
        let (thinking, effort) = thinking_for_level(&caps, ThinkingLevel::High);
        assert_eq!(thinking, Some(16_384), "pi's defaultBudgets.high is 16384, not 24000");
        assert_eq!(
            effort, None,
            "a Budget-shape model's dialect never reads reasoning_effort"
        );
    }

    #[test]
    fn thinking_for_level_clamps_the_budget_below_max_output() {
        // A small `max_output` (e.g. a legacy id) must clamp XHigh's 32_000 rungs down to something
        // that still leaves room for the turn's own output.
        let caps = ModelCaps {
            max_output: 4_096,
            ..capabilities("claude-opus-4-5")
        };
        let (thinking, _) = thinking_for_level(&caps, ThinkingLevel::XHigh);
        assert_eq!(thinking, Some(4_095), "must clamp to max_output - 1");
    }

    #[test]
    fn thinking_for_level_sets_both_fields_for_adaptive_shape_models() {
        // Gen6+ Claude (e.g. the default "claude-opus-4-8"): `ThinkingShape::Adaptive`. Both `thinking`
        // (the on/off gate) and `reasoning_effort` (the actual depth) must be set together.
        let caps = capabilities("claude-opus-4-8");
        assert_eq!(caps.thinking, ThinkingShape::Adaptive);
        let (thinking, effort) = thinking_for_level(&caps, ThinkingLevel::Medium);
        assert!(thinking.is_some(), "Adaptive needs a nonzero gate value");
        assert_eq!(effort, Some(crate::transport::ReasoningEffort::Medium));
    }

    #[test]
    fn thinking_for_level_sets_only_reasoning_effort_for_openai_style_models() {
        // OpenAI reasoning models (o-series): `ThinkingShape::None` + `reasoning_effort: true` — no
        // Anthropic-style budget field at all, only the named effort.
        let caps = capabilities("o3");
        assert_eq!(caps.thinking, ThinkingShape::None);
        assert!(caps.reasoning_effort);
        let (thinking, effort) = thinking_for_level(&caps, ThinkingLevel::Low);
        assert_eq!(
            thinking, None,
            "OpenAI reasoning models take no budget field"
        );
        assert_eq!(effort, Some(crate::transport::ReasoningEffort::Low));
    }

    #[test]
    fn clamp_reasoning_effort_drops_unsupported_xhigh_to_high() {
        use crate::transport::ReasoningEffort as RE;
        // o-series, bare/early gpt-5 ids, and sonnet-4-6/sonnet-5 carry no `xhigh` wire value at all in
        // pi's live catalogue — requesting it must clamp down to `high`, not send an invalid value.
        for id in [
            "o3",
            "o1",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-codex",
            "gpt-5-pro",
            "gpt-5.1",
            "gpt-5.1-codex",
            "gpt-5.1-chat-latest",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
        ] {
            let caps = capabilities(id);
            assert_eq!(
                clamp_reasoning_effort(&caps, RE::XHigh),
                RE::High,
                "{id} should clamp xhigh down to high"
            );
        }
        // gpt-5.2+ and opus-4-6/4-7/4-8/fable-5 do support xhigh — must pass through unclamped.
        for id in [
            "gpt-5.2",
            "gpt-5.2-codex",
            "gpt-5.3-codex",
            "gpt-5.3-codex-spark",
            "gpt-5.4",
            "gpt-5.5",
            "gpt-5.5-pro",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-fable-5",
        ] {
            let caps = capabilities(id);
            assert_eq!(
                clamp_reasoning_effort(&caps, RE::XHigh),
                RE::XHigh,
                "{id} should support xhigh"
            );
        }
    }

    #[test]
    fn clamp_reasoning_effort_raises_below_a_models_floor() {
        use crate::transport::ReasoningEffort as RE;
        // gpt-5.5 excludes "minimal"; gpt-5.5-pro excludes "minimal" and "low" too.
        let gpt55 = capabilities("gpt-5.5");
        assert_eq!(clamp_reasoning_effort(&gpt55, RE::Minimal), RE::Low);
        assert_eq!(clamp_reasoning_effort(&gpt55, RE::Low), RE::Low);
        assert_eq!(clamp_reasoning_effort(&gpt55, RE::Medium), RE::Medium);

        let gpt55_pro = capabilities("gpt-5.5-pro");
        assert_eq!(clamp_reasoning_effort(&gpt55_pro, RE::Minimal), RE::Medium);
        assert_eq!(clamp_reasoning_effort(&gpt55_pro, RE::Low), RE::Medium);
        assert_eq!(clamp_reasoning_effort(&gpt55_pro, RE::Medium), RE::Medium);

        // Every other reasoning model accepts minimal unclamped.
        for id in ["o3", "gpt-5", "gpt-5.2", "claude-opus-4-8"] {
            assert_eq!(
                clamp_reasoning_effort(&capabilities(id), RE::Minimal),
                RE::Minimal,
                "{id} should accept minimal unclamped"
            );
        }
    }

    #[test]
    fn anthropic_adaptive_effort_wire_matches_pi_map_thinking_level_to_effort() {
        use crate::transport::ReasoningEffort as RE;
        // Anthropic's adaptive shape has no "minimal" tier at all — always sent as "low", on every
        // adaptive model, not just the ones with an explicit override.
        for id in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-sonnet-5",
            "claude-fable-5",
        ] {
            let caps = capabilities(id);
            assert_eq!(anthropic_adaptive_effort_wire(&caps, RE::Minimal), "low");
            assert_eq!(anthropic_adaptive_effort_wire(&caps, RE::Low), "low");
            assert_eq!(anthropic_adaptive_effort_wire(&caps, RE::Medium), "medium");
            assert_eq!(anthropic_adaptive_effort_wire(&caps, RE::High), "high");
        }
        // xhigh is model-specific: opus-4-6 uniquely sends "max"; opus-4-7/4-8/fable-5 send "xhigh"
        // literally; sonnet-4-6/sonnet-5 have already been clamped to High and so never reach "xhigh"
        // as an input in practice, but the function must still degrade gracefully if it ever did.
        assert_eq!(
            anthropic_adaptive_effort_wire(&capabilities("claude-opus-4-6"), RE::XHigh),
            "max"
        );
        for id in ["claude-opus-4-7", "claude-opus-4-8", "claude-fable-5"] {
            assert_eq!(
                anthropic_adaptive_effort_wire(&capabilities(id), RE::XHigh),
                "xhigh",
                "{id} should send xhigh literally"
            );
        }
        for id in ["claude-sonnet-4-6", "claude-sonnet-5"] {
            assert_eq!(
                anthropic_adaptive_effort_wire(&capabilities(id), RE::XHigh),
                "high",
                "{id} has no xhigh wire value; must degrade to high"
            );
        }
    }

    #[test]
    fn clamp_thinking_level_bumps_off_to_the_floor_for_non_disableable_reasoning_models() {
        // The CRITICAL gap this fix closes: a model that has a reasoning mechanism but can't
        // explicitly disable it must never be left at the stored level `Off` — that silently omits
        // the reasoning field entirely and lets the provider apply its own hidden default effort.
        for id in [
            "gpt-5",
            "gpt-5-codex",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-pro",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
            "gpt-5.1-chat-latest",
            "gpt-5.2-chat-latest",
            "gpt-5.2-codex",
            "gpt-5.2-pro",
            "gpt-5.3-codex-spark",
            "gpt-5.4-pro",
            "claude-fable-5",
        ] {
            let caps = capabilities(id);
            assert!(
                !caps.reasoning_disableable,
                "{id} must be a non-disableable model for this test to be meaningful"
            );
            assert_ne!(
                clamp_thinking_level(&caps, ThinkingLevel::Off),
                ThinkingLevel::Off,
                "{id}: Off must not survive the clamp for a non-disableable reasoning model"
            );
        }
        // Ordinary floor: bumps to Minimal.
        assert_eq!(
            clamp_thinking_level(&capabilities("gpt-5-codex"), ThinkingLevel::Off),
            ThinkingLevel::Minimal
        );
        // A model-specific floor above Minimal must be honored too, not just a hardcoded "Minimal".
        assert_eq!(
            clamp_thinking_level(&capabilities("gpt-5.5-pro"), ThinkingLevel::Off),
            ThinkingLevel::Medium,
            "gpt-5.5-pro's floor is Medium, not the generic Minimal"
        );
    }

    #[test]
    fn clamp_thinking_level_leaves_off_alone_when_legal() {
        // Disable-capable reasoning models: Off is a real, legal state.
        for id in ["claude-opus-4-8", "claude-sonnet-5", "o3", "o1", "gpt-5.1"] {
            assert_eq!(
                clamp_thinking_level(&capabilities(id), ThinkingLevel::Off),
                ThinkingLevel::Off,
                "{id}: Off is legal and must not be bumped"
            );
        }
        // Models with no thinking/reasoning mechanism at all: Off is the only sensible state.
        for id in ["gpt-4o", "gpt-4.1", "claude-3-5-sonnet-20241022"] {
            assert_eq!(
                clamp_thinking_level(&capabilities(id), ThinkingLevel::Off),
                ThinkingLevel::Off,
                "{id}: has no mechanism to disable; Off must pass through"
            );
        }
    }

    #[test]
    fn clamp_thinking_level_still_clamps_non_off_rungs_like_clamp_reasoning_effort() {
        use crate::transport::ReasoningEffort as RE;
        // xhigh ceiling: delegates to the same behavior as `clamp_reasoning_effort`.
        assert_eq!(
            clamp_thinking_level(&capabilities("o3"), ThinkingLevel::XHigh),
            ThinkingLevel::from(RE::High)
        );
        assert_eq!(
            clamp_thinking_level(&capabilities("claude-opus-4-8"), ThinkingLevel::XHigh),
            ThinkingLevel::XHigh
        );
        // floor: gpt-5.5-pro excludes Minimal/Low.
        assert_eq!(
            clamp_thinking_level(&capabilities("gpt-5.5-pro"), ThinkingLevel::Minimal),
            ThinkingLevel::Medium
        );
    }

    #[test]
    fn available_thinking_levels_excludes_off_for_non_disableable_models() {
        assert_eq!(
            available_thinking_levels(&capabilities("gpt-5-codex")),
            vec![
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
            ],
            "gpt-5-codex has no xhigh and no off"
        );
    }

    #[test]
    fn available_thinking_levels_is_the_full_ladder_for_a_fully_capable_model() {
        assert_eq!(
            available_thinking_levels(&capabilities("claude-opus-4-8")),
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh,
            ]
        );
    }

    #[test]
    fn available_thinking_levels_is_just_off_for_a_model_with_no_mechanism() {
        assert_eq!(
            available_thinking_levels(&capabilities("gpt-4o")),
            vec![ThinkingLevel::Off]
        );
    }

    #[test]
    fn budget_shape_models_do_not_offer_xhigh_but_adaptive_shape_gen6_plus_does() {
        // Regression for the pi-parity gap: pi's `thinkingLevelMap.xhigh` is only defined for the four
        // gen6+ Adaptive-shape ids — classic Budget-shape Claude models (sonnet-4-5, opus-4-5,
        // sonnet-3-7, haiku-4-5, opus-4-0/4-1, etc.) must not advertise or accept the `xhigh` rung, even
        // though `thinking_for_level`/`budget_for_effort` would happily turn it into a plain numeric
        // `budget_tokens` value the Anthropic wire accepts fine — the bug is purely that we shouldn't
        // *offer* it for these models in the first place.
        for id in [
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-3-7-sonnet-20250219",
            "claude-haiku-4-5",
        ] {
            let caps = capabilities(id);
            assert_eq!(
                caps.thinking,
                ThinkingShape::Budget,
                "{id} should be Budget-shape"
            );
            assert!(
                !caps.supports_xhigh_reasoning,
                "{id}: Budget-shape models must not support xhigh"
            );
            assert!(
                !available_thinking_levels(&caps).contains(&ThinkingLevel::XHigh),
                "{id}: xhigh must not be an available thinking level"
            );
            assert_eq!(
                clamp_thinking_level(&caps, ThinkingLevel::XHigh),
                ThinkingLevel::High,
                "{id}: requesting xhigh must clamp down to high"
            );
        }

        // Adaptive-shape gen6+ models are unaffected: xhigh is still offered where pi's catalogue
        // actually maps it (opus-4-6/4-7/4-8, fable-5 — sonnet-4-6/sonnet-5 are the two adaptive
        // exceptions covered by other tests in this module).
        let caps = capabilities("claude-opus-4-8");
        assert_eq!(caps.thinking, ThinkingShape::Adaptive);
        assert!(caps.supports_xhigh_reasoning);
        assert!(available_thinking_levels(&caps).contains(&ThinkingLevel::XHigh));
        assert_eq!(
            clamp_thinking_level(&caps, ThinkingLevel::XHigh),
            ThinkingLevel::XHigh
        );
    }

    #[test]
    fn next_available_thinking_level_wraps_within_the_models_own_ladder_without_getting_stuck() {
        // The regression this specifically guards: a naive `level.next()` + `clamp_thinking_level`
        // bounces forever between High and a re-clamped XHigh for a model lacking xhigh support,
        // since XHigh always clamps back down to the very High it started from.
        let caps = capabilities("gpt-5-codex");
        let mut level = ThinkingLevel::Minimal;
        let mut seen = vec![level];
        for _ in 0..3 {
            level = next_available_thinking_level(&caps, level);
            seen.push(level);
        }
        assert_eq!(
            seen,
            vec![
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
            ]
        );
        // Cycling past the top must wrap back to Minimal (the model's actual floor), not get stuck.
        assert_eq!(
            next_available_thinking_level(&caps, ThinkingLevel::High),
            ThinkingLevel::Minimal
        );
    }

    #[test]
    fn next_available_thinking_level_cycles_the_full_ladder_for_a_fully_capable_model() {
        let caps = capabilities("claude-opus-4-8");
        let mut level = ThinkingLevel::Off;
        let mut seen = vec![level];
        for _ in 0..5 {
            level = next_available_thinking_level(&caps, level);
            seen.push(level);
        }
        assert_eq!(
            seen,
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh,
            ]
        );
        assert_eq!(
            next_available_thinking_level(&caps, ThinkingLevel::XHigh),
            ThinkingLevel::Off
        );
    }

    #[test]
    fn next_available_thinking_level_stays_at_off_for_a_model_with_no_mechanism() {
        let caps = capabilities("gpt-4o");
        assert_eq!(
            next_available_thinking_level(&caps, ThinkingLevel::Off),
            ThinkingLevel::Off
        );
    }

    #[test]
    fn next_available_thinking_level_reclamps_a_stale_unavailable_level_instead_of_skipping() {
        // If the stored level isn't in the model's available set at all (e.g. carried over from a
        // model switch that should have re-clamped it, but didn't), landing on its clamp first is
        // safer than jumping straight to the next rung after it.
        let caps = capabilities("gpt-5-codex"); // Off is not available here.
        assert_eq!(
            next_available_thinking_level(&caps, ThinkingLevel::Off),
            clamp_thinking_level(&caps, ThinkingLevel::Off),
        );
    }

    // ---- Third-party provider coverage (pi-parity: these used to fall through to `unknown()`) ----

    #[test]
    fn deepseek_gets_its_real_context_window_and_max_output_not_the_unknown_default() {
        // The CRITICAL bug this closes: before this table had a DeepSeek entry, `deepseek-v4-pro`
        // resolved to `ModelCaps::unknown()` — 128k context, a 4096-token output ceiling regardless of
        // the model's real 384k one (`Agent::new` seeds `max_tokens` from `max_output`).
        for id in ["deepseek-v4-pro", "deepseek-v4-flash"] {
            let c = capabilities(id);
            assert_eq!(c.context_window, 1_000_000, "{id}");
            assert_eq!(c.max_output, 384_000, "{id}");
            assert_eq!(c.max_tokens_field, MaxTokensField::MaxCompletionTokens, "{id}");
            assert!(c.reasoning_effort, "{id}");
            assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek, "{id}");
            // thinkingLevelMap nulls minimal/low/medium — floor is High, xhigh wired as "max".
            assert_eq!(c.min_reasoning_effort, crate::transport::ReasoningEffort::High, "{id}");
            assert_eq!(c.adaptive_xhigh_effort_wire, "max", "{id}");
        }
    }

    #[test]
    fn deepseek_thinking_param_emits_the_deepseek_toggle_and_reasoning_effort() {
        use crate::transport::ReasoningEffort as RE;
        let caps = capabilities("deepseek-v4-pro");
        // A requested level turns on both the toggle (checked in dialect::openai's own tests) and the
        // portable (thinking_budget, reasoning_effort) pair `Agent::with_thinking`/`with_reasoning_effort`
        // are built from.
        let (thinking, effort) = thinking_for_level(&caps, ThinkingLevel::High);
        assert_eq!(thinking, None, "DeepSeek has no Anthropic-style budget field");
        assert_eq!(effort, Some(RE::High));
        // Off clears it.
        assert_eq!(thinking_for_level(&caps, ThinkingLevel::Off), (None, None));
    }

    #[test]
    fn kimi_thinking_models_have_a_mechanism_despite_no_reasoning_effort_string() {
        // Moonshot/Kimi: `supportsReasoningEffort: false` on every id (no graduated effort vocabulary
        // at all), but the "thinking"-suffixed ids still have a real client-steerable on/off toggle
        // (`OpenAiReasoningFormat::DeepSeek`, the same shape pi tags them with). Regression guard for
        // `has_reasoning_mechanism`: without its third arm, this model would incorrectly report having
        // no mechanism at all (`available_thinking_levels` collapsing to `[Off]`) even though
        // `--thinking high` genuinely turns its reasoning on.
        let caps = capabilities("kimi-k2-thinking");
        assert!(!caps.reasoning_effort, "Kimi has no effort vocabulary");
        assert_eq!(caps.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);
        assert_ne!(
            available_thinking_levels(&caps),
            vec![ThinkingLevel::Off],
            "a toggle-only mechanism must still offer more than just Off"
        );
        // thinking_for_level must still populate `reasoning_effort` (the toggle gate the dialect reads)
        // even though the model never gets a graduated wire string for it.
        let (_, effort) = thinking_for_level(&caps, ThinkingLevel::Medium);
        assert!(effort.is_some(), "the toggle gate must be populated");

        // The non-"thinking" preview ids (reasoning: false in pi's catalogue) get no mechanism at all.
        let non_reasoning = capabilities("kimi-k2-0711-preview");
        assert_eq!(non_reasoning.openai_reasoning_format, OpenAiReasoningFormat::Standard);
        assert!(!non_reasoning.reasoning_effort);
        assert_eq!(
            available_thinking_levels(&non_reasoning),
            vec![ThinkingLevel::Off]
        );

        // kimi-k2.7-code has no "off" wire value at all — Off must bump up to the floor, same as
        // claude-fable-5's identical shape.
        let code = capabilities("kimi-k2.7-code");
        assert!(!code.reasoning_disableable);
        assert_ne!(clamp_thinking_level(&code, ThinkingLevel::Off), ThinkingLevel::Off);
    }

    #[test]
    fn glm_5_2_gets_reasoning_effort_but_earlier_glm_ids_do_not() {
        let old = capabilities("glm-4.7");
        assert!(!old.reasoning_effort, "pre-5.2 GLM has no effort vocabulary");
        assert_eq!(old.openai_reasoning_format, OpenAiReasoningFormat::Zai);
        assert_eq!(old.context_window, 204_800);
        assert_eq!(old.max_output, 131_072);

        let new = capabilities("glm-5.2");
        assert!(new.reasoning_effort, "glm-5.2 gains a real effort vocabulary");
        assert_eq!(new.context_window, 1_000_000);
        assert_eq!(new.openai_reasoning_format, OpenAiReasoningFormat::Zai);

        // glm-5v-turbo is the one vision-capable id in this family.
        assert!(capabilities("glm-5v-turbo").supports_vision);
        assert!(!capabilities("glm-4.7").supports_vision);
    }

    #[test]
    fn xai_never_emits_a_reasoning_field_matching_pis_unconditional_opt_out() {
        // pi's `detectCompat` marks every xAI id `supportsReasoningEffort: false` unconditionally, even
        // for the reasoning-capable ids (grok-4.3, grok-4.20-*-reasoning) — they reason on their own
        // with no client-steerable toggle at all.
        for id in ["grok-3", "grok-4.3", "grok-4.20-0309-reasoning", "grok-build-0.1"] {
            let c = capabilities(id);
            assert!(!c.reasoning_effort, "{id}");
            assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::Standard, "{id}");
            assert_eq!(c.max_tokens_field, MaxTokensField::MaxCompletionTokens, "{id}");
            // No mechanism at all — Off is the only available level.
            assert_eq!(available_thinking_levels(&c), vec![ThinkingLevel::Off], "{id}");
        }
        assert_eq!(capabilities("grok-3").context_window, 131_072);
        assert_eq!(capabilities("grok-4.3").context_window, 1_000_000);
    }

    #[test]
    fn together_hosted_qwen_gets_the_together_wire_format() {
        let c = capabilities("Qwen/Qwen3.6-Plus".to_ascii_lowercase().as_str());
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::Together);
        assert!(!c.reasoning_effort);

        // The one Groq-hosted exception is matched by its exact id first and keeps the plain shape.
        let groq = capabilities("qwen/qwen3-32b");
        assert_eq!(groq.openai_reasoning_format, OpenAiReasoningFormat::Standard);
        assert!(groq.reasoning_effort);
    }

    #[test]
    fn generic_vendor_slug_fallback_beats_the_flat_unknown_default() {
        // An uncatalogued vendor/model-shaped id (most commonly OpenRouter) still gets meaningfully
        // better numbers than `ModelCaps::unknown()`'s flat 128k/4096.
        let c = capabilities("some-vendor/some-new-model");
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::OpenRouter);
        assert_eq!(c.max_output, 32_000);
        assert!(c.max_output > ModelCaps::unknown().max_output);

        // A bare (no-slash) unrecognized id still falls all the way through to `unknown()`.
        let bare = capabilities("some-future-model-x");
        assert_eq!(bare.openai_reasoning_format, OpenAiReasoningFormat::Standard);
        assert_eq!(bare.max_output, 4_096);
    }

    // ---- Vendor-slug family matching + MiMo (pi-parity remediation) ----

    #[test]
    fn together_hosted_vendor_slug_ids_hit_their_real_family_not_the_generic_fallback() {
        // Before `family_id`, a vendor-slug id whose org prefix doesn't happen to start with the
        // family name (`"moonshotai/Kimi-K2.6"` doesn't start with "kimi") fell all the way through to
        // the generic 128k/32k `ApiKind::ChatCompletions`/`OpenRouter`-shaped fallback further below —
        // wrong by roughly 2-8x depending on the field. Numbers ported from
        // `packages/ai/src/providers/together.models.ts`.
        let kimi = capabilities("moonshotai/Kimi-K2.6");
        assert_eq!(kimi.context_window, 262_144);
        assert!(kimi.supports_vision, "kimi-k2.6 is vision-capable");
        assert_eq!(kimi.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);
        assert_ne!(
            kimi.openai_reasoning_format,
            OpenAiReasoningFormat::OpenRouter,
            "must not land on the generic vendor-slug fallback"
        );

        let glm = capabilities("zai-org/GLM-5.2");
        assert_eq!(glm.context_window, 1_000_000);
        assert_eq!(glm.max_output, 131_072);
        assert!(glm.reasoning_effort, "glm-5.2 has a real effort vocabulary");
        assert_eq!(glm.openai_reasoning_format, OpenAiReasoningFormat::Zai);
    }

    #[test]
    fn huggingface_hosted_vendor_slug_ids_also_hit_their_real_family() {
        // A different aggregator, same org-slug id shape (`packages/ai/src/providers/
        // huggingface.models.ts`) — confirms the fix isn't Together-specific.
        let kimi = capabilities("moonshotai/Kimi-K2-Thinking");
        assert_eq!(kimi.context_window, 262_144);
        assert_eq!(kimi.max_output, 262_144);
        assert_eq!(kimi.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);

        let glm = capabilities("zai-org/GLM-4.7");
        assert_eq!(glm.context_window, 204_800);
        assert_eq!(glm.max_output, 131_072);
        assert_eq!(glm.openai_reasoning_format, OpenAiReasoningFormat::Zai);
    }

    #[test]
    fn vendor_slug_minimax_m3_isnt_shadowed_by_its_own_org_slug_prefix_collision() {
        // "MiniMaxAI/MiniMax-M3" lowercases to "minimaxai/minimax-m3" — the org slug ("minimaxai")
        // itself begins with the literal string "minimax" one character early. Selecting the match
        // target by "did the raw id already start with the family name" (rather than by whether the
        // id is slug-shaped at all) would silently land in the right family branch but match the
        // wrong per-id sub-case inside it: the flat, smaller "else" shape below instead of
        // "minimax-m3"'s real 1M-context/vision-capable one.
        let c = capabilities("MiniMaxAI/MiniMax-M3");
        assert_eq!(c.context_window, 1_000_000);
        assert_eq!(c.max_output, 128_000);
        assert!(c.supports_vision);
    }

    #[test]
    fn vendor_slug_deepseek_and_groq_qwen_ids_are_unaffected_by_family_id() {
        // These two already matched correctly before `family_id` existed (DeepSeek's org slug
        // "deepseek-ai" itself starts with "deepseek"; Groq's real qwen id already carries its own
        // slash). Regression guard that the refactor didn't change either.
        let ds = capabilities("deepseek-ai/DeepSeek-V4-Pro");
        assert_eq!(ds.context_window, 1_000_000);
        assert_eq!(ds.max_output, 384_000);

        let groq = capabilities("qwen/qwen3-32b");
        assert_eq!(groq.openai_reasoning_format, OpenAiReasoningFormat::Standard);
        assert!(groq.reasoning_effort);
    }

    #[test]
    fn mimo_gets_its_real_capabilities_not_the_unknown_default() {
        // The CRITICAL bug this closes: no "mimo"/"xiaomi" branch existed at all, so every bare
        // "mimo-*" id (`xiaomi.models.ts`'s ids aren't vendor-slug prefixed) fell all the way through
        // to `ModelCaps::unknown()` — a 4096-token output ceiling regardless of the model's real
        // 65536-384000 one, and no reasoning support at all.
        let flash = capabilities("mimo-v2-flash");
        assert_eq!(flash.context_window, 262_144);
        assert_eq!(flash.max_output, 65_536);
        assert!(!flash.supports_vision);

        let omni = capabilities("mimo-v2-omni");
        assert_eq!(omni.max_output, 131_072);
        assert!(omni.supports_vision);

        let v25 = capabilities("mimo-v2.5");
        assert_eq!(v25.context_window, 1_048_576);
        assert!(v25.supports_vision, "bare mimo-v2.5 is vision-capable");

        // "mimo-v2.5-pro" shares its prefix with "mimo-v2.5" but is NOT vision-capable — regression
        // guard that vision is matched by exact id, not by prefix.
        let v25_pro = capabilities("mimo-v2.5-pro");
        assert_eq!(v25_pro.context_window, 1_048_576);
        assert!(!v25_pro.supports_vision);

        for id in [
            "mimo-v2-flash",
            "mimo-v2-omni",
            "mimo-v2-pro",
            "mimo-v2.5",
            "mimo-v2.5-pro",
            "mimo-v2.5-pro-ultraspeed",
        ] {
            let c = capabilities(id);
            assert!(c.reasoning_effort, "{id}");
            assert!(c.reasoning_disableable, "{id}");
            assert_eq!(c.max_tokens_field, MaxTokensField::MaxCompletionTokens, "{id}");
            assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek, "{id}");
            assert!(
                c.max_output > ModelCaps::unknown().max_output,
                "{id} should beat unknown()'s 4096 ceiling"
            );
        }
    }

    #[test]
    fn mimo_vendor_slug_id_also_matches_the_family() {
        // HuggingFace hosts this family vendor-slug-prefixed: "XiaomiMiMo/MiMo-V2.5-Pro".
        let c = capabilities("XiaomiMiMo/MiMo-V2.5-Pro");
        assert_eq!(c.context_window, 1_048_576);
        assert_eq!(c.max_output, 131_072);
        assert!(!c.supports_vision);
        assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::DeepSeek);
    }

    #[test]
    fn supports_tool_stream_is_true_for_every_glm_id_except_glm_4_5_air() {
        for id in ["glm-4.7", "glm-5-turbo", "glm-5.1", "glm-5.2", "glm-5v-turbo"] {
            assert!(capabilities(id).supports_tool_stream, "{id} should set tool_stream");
        }
        assert!(
            !capabilities("glm-4.5-air").supports_tool_stream,
            "glm-4.5-air is the one id both zai.models.ts and zai-coding-cn.models.ts leave it off for"
        );
        // Vendor-slug GLM ids inherit the flag through the same family branch.
        assert!(capabilities("zai-org/GLM-5.2").supports_tool_stream);
        // No other family sets it.
        for id in [
            "deepseek-v4-pro",
            "kimi-k2-thinking",
            "gpt-4o",
            "claude-opus-4-8",
            "mimo-v2.5",
            "some-vendor/some-model",
        ] {
            assert!(!capabilities(id).supports_tool_stream, "{id}");
        }
    }

    #[test]
    fn budget_for_effort_with_override_prefers_the_override_table_but_still_clamps_to_max_output() {
        use crate::transport::ReasoningEffort as RE;
        let mut overrides = std::collections::HashMap::new();
        overrides.insert(RE::High, 40_000);
        // The overridden level uses the table's value...
        assert_eq!(
            budget_for_effort_with_override(RE::High, 100_000, Some(&overrides)),
            40_000
        );
        // ...a level missing from the table still falls back to the built-in ladder...
        assert_eq!(
            budget_for_effort_with_override(RE::Low, 100_000, Some(&overrides)),
            default_budget_for_effort(RE::Low)
        );
        // ...and the max_output clamp still applies even to an overridden value.
        assert_eq!(
            budget_for_effort_with_override(RE::High, 4_096, Some(&overrides)),
            4_095
        );
        // No table at all behaves exactly like `budget_for_effort`.
        assert_eq!(
            budget_for_effort_with_override(RE::Medium, 100_000, None),
            budget_for_effort(RE::Medium, 100_000)
        );
    }

    // ---- Mistral (pi-parity pass 15: this whole family used to fall through to `unknown()`) ----

    #[test]
    fn mistral_ids_get_real_capabilities_not_the_unknown_default() {
        // The CRITICAL bug this closes: every Mistral model id used to resolve to
        // `ModelCaps::unknown()` — 128k context, a 4096-token output ceiling regardless of the
        // model's real (often much larger) one, vision silently disabled on every vision-capable id,
        // and no reasoning/thinking wire support at all.
        let devstral = capabilities("devstral-2512");
        assert_eq!(devstral.context_window, 262_144);
        assert_eq!(devstral.max_output, 262_144);
        assert!(!devstral.reasoning_effort);
        assert!(!devstral.supports_vision);

        let large_2512 = capabilities("mistral-large-2512");
        assert_eq!(large_2512.context_window, 262_144);
        assert!(large_2512.supports_vision, "mistral-large-2512 gained vision");
        // The "-2411" predecessor is text-only and much smaller — a family-level default would have
        // misreported one of these by a wide margin.
        let large_2411 = capabilities("mistral-large-2411");
        assert_eq!(large_2411.context_window, 131_072);
        assert_eq!(large_2411.max_output, 16_384);
        assert!(!large_2411.supports_vision);

        assert!(capabilities("pixtral-12b").supports_vision);
        assert!(capabilities("pixtral-large-latest").supports_vision);
        assert!(!capabilities("open-mistral-7b").supports_vision);
        assert_eq!(capabilities("open-mistral-7b").context_window, 8_000);
    }

    #[test]
    fn mistral_reasoning_capable_ids_get_a_thinking_mechanism() {
        // pixtral/mistral-medium-*/mistral-small-2603+ used to silently have no thinking mechanism at
        // all (`reasoning_effort: false`, `openai_reasoning_format: Standard` with nothing to drive
        // it) — `--thinking high` was a silent no-op.
        for id in [
            "magistral-medium-latest",
            "magistral-small",
            "mistral-medium-2604",
            "mistral-medium-3.5",
            "mistral-small-2603",
            "mistral-small-latest",
        ] {
            let c = capabilities(id);
            assert!(c.reasoning_effort, "{id} should gain a thinking mechanism");
            assert_eq!(c.openai_reasoning_format, OpenAiReasoningFormat::Standard, "{id}");
            assert!(c.reasoning_disableable, "{id}: off is always legal (no id nulls it)");
            assert!(!c.supports_xhigh_reasoning, "{id}: no Mistral id defines xhigh");
        }
        // "mistral-medium-latest" is (per pi's live catalogue) an alias still pointing at a
        // non-reasoning snapshot, unlike its "-2604"/"-3.5" siblings.
        assert!(!capabilities("mistral-medium-latest").reasoning_effort);
        assert!(!capabilities("mistral-large-latest").reasoning_effort);
    }

    #[test]
    fn is_mistral_model_matches_every_current_family_prefix() {
        for id in [
            "codestral-latest",
            "devstral-2512",
            "ministral-3b-latest",
            "magistral-small",
            "mistral-large-latest",
            "pixtral-12b",
            "open-mistral-7b",
            "open-mixtral-8x22b",
            "labs-devstral-small-2512",
            "MISTRAL-SMALL-LATEST", // case-insensitive
        ] {
            assert!(is_mistral_model(id), "{id} should be recognized as Mistral");
        }
        for id in ["gpt-4o", "claude-opus-4-8", "deepseek-v4-pro", "some-vendor/model"] {
            assert!(!is_mistral_model(id), "{id} should not be recognized as Mistral");
        }
    }

    #[test]
    fn is_deepseek_model_is_narrower_than_the_shared_deepseek_wire_shape() {
        assert!(is_deepseek_model("deepseek-v4-pro"));
        assert!(is_deepseek_model("DeepSeek-V4-Flash"));
        // Kimi/Moonshot shares `OpenAiReasoningFormat::DeepSeek`'s wire *shape* but is a different
        // provider — `is_deepseek_model` must not conflate the two.
        assert!(!is_deepseek_model("kimi-k2-thinking"));
        assert!(!is_deepseek_model("gpt-4o"));
    }

    // ---- Per-model reasoning-effort wire remap (pi-parity pass 15) ----

    #[test]
    fn reasoning_wire_override_remaps_deepseek_xhigh_to_max() {
        use crate::transport::ReasoningEffort as RE;
        assert_eq!(reasoning_wire_override("deepseek-v4-pro", RE::XHigh), Some("max"));
        // High is a literal passthrough — no override.
        assert_eq!(reasoning_wire_override("deepseek-v4-pro", RE::High), None);
    }

    #[test]
    fn reasoning_wire_override_remaps_glm_5_2_low_medium_and_xhigh() {
        use crate::transport::ReasoningEffort as RE;
        for (effort, wire) in [(RE::Low, "high"), (RE::Medium, "high"), (RE::High, "high")] {
            assert_eq!(reasoning_wire_override("glm-5.2", effort), Some(wire), "{effort:?}");
        }
        assert_eq!(reasoning_wire_override("glm-5.2", RE::XHigh), Some("max"));
        // Minimal is excluded by the model's own floor (`min_reasoning_effort: Low`) before this
        // function would ever see it in practice; confirm that floor directly.
        assert_eq!(capabilities("glm-5.2").min_reasoning_effort, RE::Low);
        assert_eq!(capabilities("glm-4.7").min_reasoning_effort, RE::Minimal);
    }

    #[test]
    fn reasoning_wire_override_remaps_vendor_slug_glm_5_2_same_as_the_bare_id() {
        use crate::transport::ReasoningEffort as RE;
        // "zai-org/glm-5.2" (Together/HuggingFace) must remap identically to the bare "glm-5.2" it's
        // slug-prefixed with, not silently fall through to `None` because the full id doesn't start
        // with "glm-5.2".
        assert_eq!(reasoning_wire_override("zai-org/glm-5.2", RE::Low), Some("high"));
        assert_eq!(reasoning_wire_override("zai-org/glm-5.2", RE::XHigh), Some("max"));
    }

    #[test]
    fn reasoning_wire_override_remaps_groq_qwen_high_to_default() {
        use crate::transport::ReasoningEffort as RE;
        assert_eq!(reasoning_wire_override("qwen/qwen3-32b", RE::High), Some("default"));
        // A different qwen host (Together-shaped) isn't this one exact Groq id — no override.
        assert_eq!(reasoning_wire_override("qwen/qwen3.6-plus", RE::High), None);
    }

    #[test]
    fn reasoning_wire_override_sends_high_for_every_mistral_reasoning_id() {
        use crate::transport::ReasoningEffort as RE;
        // Mistral's real reasoning_effort vocabulary is a bare "none"|"high" enum with no per-level
        // map at all — every active level collapses to "high" on the wire, matching
        // `mapReasoningEffort`'s literal fallback in pi's own Mistral client.
        for effort in [RE::Minimal, RE::Low, RE::Medium, RE::High, RE::XHigh] {
            assert_eq!(
                reasoning_wire_override("mistral-small-2603", effort),
                Some("high"),
                "{effort:?}"
            );
        }
        // A non-reasoning Mistral id is never routed here by the caller, but the function itself is
        // still harmless if it somehow were.
        assert_eq!(
            reasoning_wire_override("mistral-large-latest", RE::High),
            Some("high")
        );
    }

    #[test]
    fn reasoning_wire_override_is_none_for_every_other_family() {
        use crate::transport::ReasoningEffort as RE;
        for id in ["o3", "gpt-5.2", "claude-opus-4-8", "grok-4.3", "kimi-k2-thinking"] {
            assert_eq!(reasoning_wire_override(id, RE::High), None, "{id}");
        }
    }

    #[test]
    fn is_non_standard_store_provider_matches_pis_denylist_where_the_id_alone_can_tell() {
        for id in [
            "deepseek-v4-pro",
            "glm-5.2",
            "kimi-k2-thinking",
            "grok-4.3",
            "qwen/qwen3.6-plus",
            "gpt-oss-120b",
            "zai-glm-4.7",
        ] {
            assert!(is_non_standard_store_provider(id), "{id} should be non-standard");
        }
        // The one Groq-hosted qwen exception is standard (Groq isn't in pi's denylist).
        assert!(!is_non_standard_store_provider("qwen/qwen3-32b"));
        for id in ["gpt-4o", "claude-opus-4-8", "mistral-large-latest", "some-vendor/model"] {
            assert!(!is_non_standard_store_provider(id), "{id} should be standard");
        }
    }
}
