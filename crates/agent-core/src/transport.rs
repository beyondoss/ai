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
}

/// A provider-neutral reasoning effort level. Maps to OpenAI's `reasoning_effort` parameter on
/// reasoning models (o-series, gpt-5), and to the `output_config.effort` field of Anthropic's
/// adaptive-thinking shape on models that take it.
///
/// `PartialOrd`/`Ord` follow declaration order (`Minimal` lowest, `XHigh` highest) — depended on by
/// [`crate::models::clamp_reasoning_effort`] to clamp a requested level up/down to whatever a specific
/// model actually accepts on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    /// Tools advertised to the model this turn. An `Arc<[…]>` because the set is fixed for the run:
    /// the agent computes it once and hands the same slice to every turn's request by cloning the
    /// pointer, not the (schema-bearing) definitions.
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

    /// Builder-style: request extended thinking with the given token budget.
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig { budget_tokens });
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
