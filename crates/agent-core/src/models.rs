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
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            api: ApiKind::ChatCompletions,
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
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

/// Resolve a model id to its [`ModelCaps`]. Matching is by id prefix (most-specific first); unknown
/// ids fall back to [`ModelCaps::unknown`] (logged, since a silent conservative fallback can otherwise
/// mask a model we should have taught this table about).
///
/// The per-family numbers below are ported from the reference agent's live model catalogue
/// (`packages/ai/src/providers/{anthropic,openai}.models.ts` in `badlogic/pi-mono`) rather than
/// invented — re-check that catalogue when adding a new model family, since it's regenerated upstream
/// and may have moved on by the time you read this.
pub fn capabilities(model: &str) -> ModelCaps {
    let m = model.to_ascii_lowercase();

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
            return ModelCaps {
                context_window: 1_000_000,
                max_output,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                thinking: ThinkingShape::Adaptive,
                reasoning_effort: false,
                reasoning_disableable,
                supports_eager_tool_streaming: true,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning,
                adaptive_xhigh_effort_wire,
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
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                // No thinking support at all here — nothing to explicitly disable, so the `thinking`
                // field stays omitted entirely rather than sending a `{"type":"disabled"}` a model
                // that never supported thinking might reject.
                reasoning_disableable: false,
                supports_eager_tool_streaming: true,
                api: ApiKind::ChatCompletions,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
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
            thinking: ThinkingShape::Budget,
            reasoning_effort: false,
            reasoning_disableable: true,
            supports_eager_tool_streaming: true,
            api: ApiKind::ChatCompletions,
            // Budget-shape thinking sends a numeric `budget_tokens`, never a named effort string, so
            // neither field below is ever read for this shape — left at their harmless defaults.
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
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
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            // o3-mini is text-only (pi's catalogue: `input: ["text"]`) — the one o-series id that
            // isn't vision-capable, unlike o1-mini's exclusion above for a different reason.
            supports_vision: !m.starts_with("o1-mini") && !m.starts_with("o3-mini"),
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            // o-series ids are disable-capable by default in pi's catalogue (no override).
            reasoning_disableable: true,
            supports_eager_tool_streaming: false,
            api: ApiKind::Responses,
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            // No o-series id carries a `thinkingLevelMap` at all in pi's catalogue, and an unmapped
            // "xhigh" is excluded from `getSupportedThinkingLevels` — none of them accept it.
            supports_xhigh_reasoning: false,
            adaptive_xhigh_effort_wire: "xhigh", // unread: o-series never uses Adaptive shape.
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
                thinking: ThinkingShape::None,
                reasoning_effort,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                api: ApiKind::Responses,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: gpt5_supports_xhigh(&m),
                adaptive_xhigh_effort_wire: "xhigh", // unread: this bucket never uses Adaptive shape.
            };
        }
        // "gpt-5.3-codex-spark" is a narrower model than the rest of the family — 128k context, 32k
        // output — not the generic 400k/128k every other gpt-5 id gets below. Not in pi's
        // disable-capable allowlist (that's `gpt-5.3-codex`, a different id).
        if m == "gpt-5.3-codex-spark" {
            return ModelCaps {
                context_window: 128_000,
                max_output: 32_000,
                max_tokens_field: MaxTokensField::MaxCompletionTokens,
                supports_long_cache: true,
                supports_vision: true,
                thinking: ThinkingShape::None,
                reasoning_effort: true,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                api: ApiKind::Responses,
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
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
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            reasoning_disableable,
            supports_eager_tool_streaming: false,
            api: ApiKind::Responses,
            min_reasoning_effort,
            supports_xhigh_reasoning: gpt5_supports_xhigh(&m),
            adaptive_xhigh_effort_wire: "xhigh", // unread: this bucket never uses Adaptive shape.
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
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                api: ApiKind::Responses,
                // The gpt-4 family never takes `reasoning_effort` at all — these three are unread.
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
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
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                reasoning_disableable: false,
                supports_eager_tool_streaming: false,
                api: ApiKind::Responses,
                // The gpt-4 family never takes `reasoning_effort` at all — these three are unread.
                min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
                supports_xhigh_reasoning: true,
                adaptive_xhigh_effort_wire: "xhigh",
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
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            reasoning_disableable: false,
            supports_eager_tool_streaming: false,
            api: ApiKind::Responses,
            min_reasoning_effort: crate::transport::ReasoningEffort::Minimal,
            supports_xhigh_reasoning: true,
            adaptive_xhigh_effort_wire: "xhigh",
        };
    }

    tracing::warn!(
        model,
        "unrecognized model id; falling back to conservative capabilities"
    );
    ModelCaps::unknown()
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

/// A fixed effort→token-budget ladder for [`ThinkingShape::Budget`]-shape models, clamped below
/// `max_output` (a thinking budget must leave room for the turn's actual output). For
/// [`ThinkingShape::Adaptive`]-shape models this value is a pure on/off gate on the wire — only its
/// `Some`-ness is checked, the actual depth comes from `output_config.effort` (see
/// `dialect::anthropic::build_body`) — so its exact magnitude doesn't matter there, but reusing the same
/// ladder keeps one code path instead of two.
fn budget_for_effort(effort: crate::transport::ReasoningEffort, max_output: u32) -> u32 {
    use crate::transport::ReasoningEffort as RE;
    let budget = match effort {
        RE::Minimal => 1_024,
        RE::Low => 2_048,
        RE::Medium => 8_192,
        RE::High => 24_000,
        RE::XHigh => 32_000,
    };
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
    let reasoning_effort_applies =
        caps.reasoning_effort || caps.thinking == ThinkingShape::Adaptive;
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
    fn o3_mini_is_text_only() {
        // The one o-series id that isn't vision-capable (pi: `input: ["text"]`), unlike o1-mini
        // (excluded above for a different reason) and every other o-series id.
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
        assert_eq!(thinking, Some(24_000));
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
}
