//! The model-transport seam.
//!
//! The agent loop depends only on [`ModelTransport`] — never on `reqwest` or a dialect. Two things
//! implement it (in later milestones): the real HTTP client that POSTs OpenAI/Anthropic wire to the
//! Beyond gateway, and a `MockTransport` that replays scripted events so the loop is testable with
//! no network. Keeping the trait here, free of any client, is what makes that possible.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::Result;
use crate::message::{Message, StreamEvent, ToolDef};

/// Extended-thinking request config. Presence asks the model to think before answering; `budget_tokens`
/// caps the thinking spend (and must be below `max_tokens`).
#[derive(Debug, Clone, Copy)]
pub struct ThinkingConfig {
    pub budget_tokens: u32,
    /// How much of the model's reasoning to surface in the response (Anthropic dialect only — see
    /// [`ThinkingDisplay`]). Defaults to [`ThinkingDisplay::Summarized`] via [`ModelRequest::with_thinking`];
    /// override with [`ModelRequest::with_thinking_display`].
    pub display: ThinkingDisplay,
}

/// How much of a model's reasoning Anthropic's `thinking.display` should surface in the response.
/// Only meaningful alongside [`ThinkingConfig`] on the Anthropic dialect; every OpenAI-shaped dialect
/// ignores it (a reasoning model there returns only an opaque `encrypted_content` blob for replay,
/// never thinking text at all, so there is nothing for a display mode to control).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingDisplay {
    /// Thinking blocks contain a readable summary of the model's reasoning. Matches pi's own
    /// cross-model default (`anthropic-messages.ts`) — kept as this crate's default too, even though
    /// it differs from Anthropic's *own* undocumented API default on 4.7+/Fable-generation models
    /// (which is `Omitted`), so behavior stays consistent across the whole Claude 4 family regardless
    /// of which specific id a request targets.
    #[default]
    Summarized,
    /// Thinking blocks stream back with an empty `thinking` text field; the signature still travels
    /// back for multi-turn replay continuity. Lower latency to first visible text token — pi: "Use for
    /// faster time-to-first-text-token when your UI does not surface thinking."
    Omitted,
}

impl ThinkingDisplay {
    /// The wire string Anthropic's `thinking.display` field expects.
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingDisplay::Summarized => "summarized",
            ThinkingDisplay::Omitted => "omitted",
        }
    }
}

/// A provider-neutral reasoning effort level. Maps to OpenAI's `reasoning_effort` parameter on
/// reasoning models (o-series, gpt-5), and to the `output_config.effort` field of Anthropic's
/// adaptive-thinking shape on models that take it.
///
/// `PartialOrd`/`Ord` follow declaration order (`Minimal` lowest, `XHigh` highest) — depended on by
/// [`crate::models::clamp_reasoning_effort`] to clamp a requested level up/down to whatever a specific
/// model actually accepts on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
}

impl ReasoningEffort {
    /// The wire string both providers use.
    pub fn as_str(self) -> &'static str {
        match self {
            ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::XHigh => "xhigh",
        }
    }
}

/// How verbose a reasoning model's visible thinking summary should be — OpenAI Responses'
/// `reasoning.summary` field. Only meaningful alongside an explicit [`ReasoningEffort`] (a model with
/// no effort set gets no summary either, regardless of this). `None` on [`ModelRequest`] defaults to
/// `Auto` on the wire, matching pi's own `reasoningSummary` default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSummary {
    /// The provider picks a level of detail automatically.
    Auto,
    /// A fuller, more detailed summary.
    Detailed,
    /// A shorter summary.
    Concise,
}

impl ReasoningSummary {
    /// The wire string OpenAI's Responses API uses.
    pub fn as_str(self) -> &'static str {
        match self {
            ReasoningSummary::Auto => "auto",
            ReasoningSummary::Detailed => "detailed",
            ReasoningSummary::Concise => "concise",
        }
    }
}

/// OpenAI Responses' `service_tier` — a queueing/latency class, not purely a pricing knob: `Flex` and
/// `Priority` change how quickly a request is served (cheaper-and-slower vs. pricier-and-faster),
/// independent of the model or effort chosen. `None` on [`ModelRequest`] omits the field, leaving
/// OpenAI's own default tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceTier {
    /// The provider picks a tier automatically.
    Auto,
    /// The account's configured default tier.
    Default,
    /// Cheaper, slower, best-effort queueing.
    Flex,
    /// Pricier, faster queueing.
    Priority,
}

impl ServiceTier {
    /// The wire string OpenAI's Responses API uses.
    pub fn as_str(self) -> &'static str {
        match self {
            ServiceTier::Auto => "auto",
            ServiceTier::Default => "default",
            ServiceTier::Flex => "flex",
            ServiceTier::Priority => "priority",
        }
    }
}

/// How the model may use the advertised tools this turn. A request's [`ModelRequest::tool_choice`] is
/// an `Option<ToolChoice>`: `None` (the default) emits **nothing** on the wire, leaving the provider's
/// own default (auto when tools are present). The variants force a choice, and each dialect maps them
/// to its own vocabulary (see the dialect `build_body`s):
/// - `Auto` — the model decides whether to call a tool (Anthropic `{type:"auto"}`, OpenAI `"auto"`).
/// - `None` — forbid tool calls this turn (Anthropic `{type:"none"}`, OpenAI `"none"`).
/// - `Required` — the model *must* call some tool (Anthropic `{type:"any"}`, OpenAI `"required"`).
/// - `Tool(name)` — pin the call to one named tool (Anthropic `{type:"tool", name}`, OpenAI the
///   `{type:"function", function:{name}}` shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model decides whether to call a tool.
    Auto,
    /// No tool may be called this turn.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call this specific tool.
    Tool(String),
}

/// One model request: the conversation so far plus the tools the model may call this turn.
#[derive(Debug, Clone)]
pub struct ModelRequest {
    /// Model identifier (e.g. `claude-opus-4-8`); the client maps it to a dialect + gateway path.
    pub model: String,
    /// Optional system prompt (kept separate from `messages` — both wire dialects treat it specially).
    pub system: Option<String>,
    /// Conversation history. An `Arc<Vec<…>>` shared with the `Session` it came from: building a
    /// request clones the pointer, not the (growing) message list.
    pub messages: Arc<Vec<Message>>,
    /// Tools advertised to the model this turn. An `Arc<[…]>` so building a request clones the pointer,
    /// not the (schema-bearing) definitions. Fixed for any one *request*, but not necessarily for the
    /// whole *run* it belongs to: `Agent::run_events_steered` normally hands the same `Arc` to every
    /// turn, but a mid-run `Steering::request_tool_set` (Task #13) swaps it for a different one starting
    /// the very next turn.
    pub tools: Arc<[ToolDef]>,
    /// Output token ceiling for the turn.
    pub max_tokens: u32,
    /// Extended thinking, when requested. `None` leaves it off (the default).
    pub thinking: Option<ThinkingConfig>,
    /// Reasoning effort, for models driven by an effort level rather than a token budget (OpenAI
    /// reasoning models; Anthropic adaptive thinking). `None` leaves the provider default.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// How the model may use tools this turn (auto / none / required / a specific tool). `None` leaves
    /// the provider default (auto when tools are present) and emits nothing on the wire — so the
    /// default request shape is unchanged. See [`ToolChoice`].
    pub tool_choice: Option<ToolChoice>,
    /// A stable per-conversation cache key, for prompt-cache affinity. OpenAI routes cache hits by
    /// `prompt_cache_key`; supplying a consistent value keeps a session pinned to a warm cache node.
    pub cache_key: Option<String>,
    /// Use the 1-hour prompt-cache TTL instead of the default 5 minutes. Worth it for a long-running
    /// agent that may pause more than 5 minutes between turns (a slow tool, a thinking user), at a
    /// higher cache-write price.
    pub cache_long: bool,
    /// Skip prompt-cache breakpoint stamping entirely (Anthropic's `cache_control`; equivalently
    /// OpenAI's `prompt_cache_key`/`prompt_cache_retention`). A cache write costs ~1.25x the input-token
    /// price up front; that only pays off if a later turn reads it back. A genuinely one-off,
    /// non-conversational request (no follow-up turn to amortize the write against) should opt out
    /// rather than eating that premium for a cache entry nothing will ever read.
    pub no_cache: bool,
    /// An opaque, caller-chosen identifier for the end user on whose behalf this request runs — sent as
    /// Anthropic's `metadata.user_id` (should be a hashed/anonymized id, per Anthropic's own guidance,
    /// never a raw name/email), which Anthropic uses for abuse detection and rate limiting. `None`
    /// (the default) omits the field entirely — matches pi's own `CompletionOptions.metadata`, a generic
    /// passthrough its library layer exposes but its own CLI never populates; the dialects that don't
    /// understand `metadata.user_id` (OpenAI-shaped ones) simply ignore it, same as pi's provider
    /// implementations only extracting the fields they recognize.
    pub user_id: Option<String>,
    /// How verbose a visible reasoning summary should be (OpenAI Responses only; ignored by every
    /// other dialect). `None` defaults to [`ReasoningSummary::Auto`] on the wire.
    pub reasoning_summary: Option<ReasoningSummary>,
    /// Queueing/latency class for the request (OpenAI Responses only; ignored by every other
    /// dialect). `None` omits the field, leaving OpenAI's own default tier.
    pub service_tier: Option<ServiceTier>,
    /// Sampling temperature. `None` leaves the provider default. On the Anthropic dialect this is
    /// silently omitted whenever `thinking` is set — Anthropic's API requires `temperature: 1` (or the
    /// field entirely absent) while extended thinking is enabled, matching pi's own
    /// `!options?.thinkingEnabled` gate (`anthropic-messages.ts`) — and also omitted unconditionally
    /// for a model whose capability table marks `supports_temperature: false`
    /// (`claude-opus-4-7`/`claude-opus-4-8`, which reject the field outright regardless of thinking
    /// state; see `crate::models::ModelCaps::supports_temperature`). Both OpenAI-shaped dialects send
    /// it unconditionally when set, matching pi's own unconditional `openai-completions.ts`/
    /// `openai-responses.ts` — a caller asking for a fixed temperature on a reasoning model that
    /// rejects it is a caller error, same as pi's.
    pub temperature: Option<f64>,
}

impl ModelRequest {
    /// A request with no system prompt and no tools — the minimal shape.
    pub fn new(
        model: impl Into<String>,
        messages: impl Into<Arc<Vec<Message>>>,
        max_tokens: u32,
    ) -> Self {
        Self {
            model: model.into(),
            system: None,
            messages: messages.into(),
            tools: Vec::new().into(),
            max_tokens,
            thinking: None,
            reasoning_effort: None,
            tool_choice: None,
            cache_key: None,
            cache_long: false,
            no_cache: false,
            user_id: None,
            reasoning_summary: None,
            service_tier: None,
            temperature: None,
        }
    }

    /// Builder-style: attach the tools advertised for this turn. Accepts anything convertible into
    /// `Arc<[ToolDef]>` (a `Vec<ToolDef>`, or an `Arc<[ToolDef]>` cloned from a cache).
    pub fn with_tools(mut self, tools: impl Into<Arc<[ToolDef]>>) -> Self {
        self.tools = tools.into();
        self
    }

    /// Builder-style: attach a system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Builder-style: request extended thinking with the given token budget. Display defaults to
    /// [`ThinkingDisplay::Summarized`]; override with [`with_thinking_display`](Self::with_thinking_display).
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig {
            budget_tokens,
            display: ThinkingDisplay::default(),
        });
        self
    }

    /// Builder-style: override the thinking display mode (see [`ThinkingDisplay`]) on a previously-set
    /// `thinking` config. A no-op if `thinking` hasn't been set yet — call
    /// [`with_thinking`](Self::with_thinking) first.
    pub fn with_thinking_display(mut self, display: ThinkingDisplay) -> Self {
        if let Some(thinking) = &mut self.thinking {
            thinking.display = display;
        }
        self
    }

    /// Builder-style: set the reasoning effort level.
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }

    /// Builder-style: constrain how the model may use tools this turn (see [`ToolChoice`]). Leaving
    /// it unset emits nothing on the wire (the provider default).
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Builder-style: set the prompt-cache affinity key (a stable per-conversation id).
    pub fn with_cache_key(mut self, key: impl Into<String>) -> Self {
        self.cache_key = Some(key.into());
        self
    }

    /// Builder-style: opt into the 1-hour prompt-cache TTL.
    pub fn with_cache_long(mut self, long: bool) -> Self {
        self.cache_long = long;
        self
    }

    /// Builder-style: skip prompt-cache breakpoint stamping entirely, for a one-off request with no
    /// follow-up turn to amortize the cache-write premium against.
    pub fn with_no_cache(mut self, no_cache: bool) -> Self {
        self.no_cache = no_cache;
        self
    }

    /// Builder-style: set the end-user identifier sent as Anthropic's `metadata.user_id`. See the
    /// field's own doc comment for the hashed/anonymized-id expectation.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Builder-style: set how verbose a visible reasoning summary should be (OpenAI Responses only).
    pub fn with_reasoning_summary(mut self, summary: ReasoningSummary) -> Self {
        self.reasoning_summary = Some(summary);
        self
    }

    /// Builder-style: set the request's queueing/latency tier (OpenAI Responses only).
    pub fn with_service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    /// Builder-style: set the sampling temperature. See the field's own doc comment for per-dialect
    /// gating (Anthropic omits it while thinking is enabled).
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

/// Streamed events from a single model turn. `'static` so the stream can outlive the call (it's
/// driven to completion by the loop, not the transport).
pub type EventStream = BoxStream<'static, Result<StreamEvent>>;

/// The boundary between the agent loop and the network. Implementors turn a [`ModelRequest`] into a
/// stream of normalized [`StreamEvent`]s.
#[async_trait]
pub trait ModelTransport: Send + Sync {
    /// Issue the request and return its event stream. Errors here are connection/setup failures;
    /// per-event failures surface as `Err` items within the stream.
    async fn stream(&self, req: ModelRequest) -> Result<EventStream>;
}
