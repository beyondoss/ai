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
