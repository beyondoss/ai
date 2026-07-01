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
    /// Which OpenAI-wire API surface the model speaks (ignored for Anthropic ids). See [`ApiKind`].
    pub api: ApiKind,
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
            api: ApiKind::ChatCompletions,
        }
    }
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
            return ModelCaps {
                context_window: 1_000_000,
                max_output,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                thinking: ThinkingShape::Adaptive,
                reasoning_effort: false,
                api: ApiKind::ChatCompletions,
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
                supports_long_cache: false,
                supports_vision: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                api: ApiKind::ChatCompletions,
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
            api: ApiKind::ChatCompletions,
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
            supports_vision: !m.starts_with("o1-mini"),
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            api: ApiKind::Responses,
        };
    }

    // ---- OpenAI GPT-5 family (reasoning) ----
    if m.starts_with("gpt-5") {
        // The narrower "-chat-latest" variants share the family name but cap at the older
        // chat-completions ceiling (128k/16384), not the reasoning-model one, and aren't uniformly
        // `reasoning_effort`-driven — treat them like a non-reasoning chat model.
        if m.contains("chat-latest") {
            return ModelCaps {
                context_window: 128_000,
                max_output: 16_384,
                max_tokens_field: MaxTokensField::MaxTokens,
                supports_long_cache: true,
                supports_vision: true,
                thinking: ThinkingShape::None,
                reasoning_effort: false,
                api: ApiKind::Responses,
            };
        }
        // "-pro" variants ship a much larger 1.05M context. The bare "gpt-5.4"/"gpt-5.5" release (no
        // mini/nano/pro suffix) runs a smaller 272k window; every other family member is 400k.
        let context_window = if m.contains("pro") {
            1_050_000
        } else if m == "gpt-5.4" || m == "gpt-5.5" {
            272_000
        } else {
            400_000
        };
        return ModelCaps {
            context_window,
            max_output: 128_000,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: true,
            supports_vision: true,
            thinking: ThinkingShape::None,
            reasoning_effort: true,
            api: ApiKind::Responses,
        };
    }

    // ---- OpenAI GPT-4 family (bare gpt-4 / 4-turbo / 4o / 4.1) ----
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
                api: ApiKind::Responses,
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
            supports_vision: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
            api: ApiKind::Responses,
        };
    }

    tracing::warn!(
        model,
        "unrecognized model id; falling back to conservative capabilities"
    );
    ModelCaps::unknown()
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
            assert!(!c.supports_long_cache);
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
}
