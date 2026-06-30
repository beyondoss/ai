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

/// The extended-thinking request shape a model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingShape {
    /// No Anthropic-style thinking control — omit the `thinking` field (OpenAI reasoning models use
    /// `reasoning_effort` instead; see [`ModelCaps::reasoning_effort`]).
    None,
    /// Anthropic `{type:"enabled", budget_tokens}` — Claude 3.7 / 4.x extended thinking.
    Budget,
    /// Anthropic `{type:"adaptive", display, output_config.effort}` — the newer shape for models that
    /// require it. (We only mark a model `Adaptive` when its `Budget` shape is known *not* to work;
    /// every current Claude we ship against accepts `Budget`, which the live smoke test exercises.)
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
        }
    }
}

/// Resolve a model id to its [`ModelCaps`]. Matching is by id prefix (most-specific first); unknown
/// ids fall back to [`ModelCaps::unknown`].
pub fn capabilities(model: &str) -> ModelCaps {
    let m = model.to_ascii_lowercase();

    // ---- Anthropic Claude (+ Fable, which speaks the Anthropic wire) ----
    if m.starts_with("claude") || m.starts_with("fable") {
        // Modern Claude / Fable all honor the 1h cache TTL and take vision input. Thinking is the
        // `Budget` (`enabled`) shape — live-validated for `claude-opus-4-8`, our default model.
        let max_output = if m.contains("sonnet") {
            64_000
        } else if m.contains("haiku") {
            8_192
        } else {
            32_000 // opus / fable / generic claude
        };
        return ModelCaps {
            context_window: 200_000,
            max_output,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: true,
            supports_vision: true,
            thinking: ThinkingShape::Budget,
            reasoning_effort: false,
        };
    }

    // ---- OpenAI reasoning models: o-series + gpt-5 family ----
    // These reject `max_tokens` (require `max_completion_tokens`) and are driven by `reasoning_effort`
    // rather than an Anthropic-style thinking block.
    let is_o_series = m.starts_with("o1") || m.starts_with("o3") || m.starts_with("o4");
    if is_o_series || m.starts_with("gpt-5") {
        return ModelCaps {
            context_window: 200_000,
            max_output: 32_000,
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
            supports_long_cache: false,
            supports_vision: !m.starts_with("o1-mini"),
            thinking: ThinkingShape::None,
            reasoning_effort: true,
        };
    }

    // ---- OpenAI GPT-4 family (4o / 4.1 / 4-turbo) ----
    if m.starts_with("gpt-4") {
        return ModelCaps {
            context_window: 128_000,
            max_output: 16_384,
            max_tokens_field: MaxTokensField::MaxTokens,
            supports_long_cache: false,
            supports_vision: true,
            thinking: ThinkingShape::None,
            reasoning_effort: false,
        };
    }

    ModelCaps::unknown()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_defaults_are_anthropic_shaped() {
        let c = capabilities("claude-opus-4-8");
        assert_eq!(c.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(c.supports_long_cache);
        assert_eq!(c.thinking, ThinkingShape::Budget);
        assert_eq!(c.context_window, 200_000);
        assert!(!c.reasoning_effort);
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
            assert!(!c.supports_long_cache);
            assert_eq!(c.thinking, ThinkingShape::None);
        }
    }

    #[test]
    fn gpt4_uses_plain_max_tokens() {
        let c = capabilities("gpt-4o");
        assert_eq!(c.max_tokens_field, MaxTokensField::MaxTokens);
        assert!(!c.reasoning_effort);
        assert!(c.supports_vision);
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
            ThinkingShape::Budget
        );
    }
}
