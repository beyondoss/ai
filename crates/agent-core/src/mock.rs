//! A scripted [`ModelTransport`] for tests.
//!
//! Replaying a fixed sequence of `StreamEvent` "turns" lets the agent loop — and the binaries that
//! drive it — be exercised end to end with no network and no model. Each call to `stream` pops the
//! next turn and records the request, so a test can assert exactly what the loop sent back.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::error::{Error, Result};
use crate::message::StreamEvent;
use crate::transport::{EventStream, ModelRequest, ModelTransport};

/// A transport that yields pre-scripted turns. Construct with one `Vec<StreamEvent>` per model turn.
pub struct MockTransport {
    turns: Mutex<VecDeque<Vec<StreamEvent>>>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl MockTransport {
    /// Script the transport with the turns it will return, in order.
    pub fn new(turns: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            turns: Mutex::new(turns.into()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// The requests the loop has sent so far, in order (for asserting what the loop fed back).
    pub fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// How many turns the loop has consumed.
    pub fn calls(&self) -> usize {
        self.requests.lock().map(|r| r.len()).unwrap_or(0)
    }
}

#[async_trait]
impl ModelTransport for MockTransport {
    async fn stream(&self, req: ModelRequest) -> Result<EventStream> {
        if let Ok(mut reqs) = self.requests.lock() {
            reqs.push(req);
        }
        let turn = self
            .turns
            .lock()
            .ok()
            .and_then(|mut t| t.pop_front())
            .ok_or_else(|| Error::Transport("MockTransport: no more scripted turns".into()))?;
        Ok(Box::pin(futures::stream::iter(turn.into_iter().map(Ok))))
    }
}

/// Convenience builders for the events that make up a scripted turn.
pub mod turn {
    use crate::message::{StopReason, StreamEvent};

    /// A turn that emits one text block and ends the conversation.
    pub fn text(s: &str) -> Vec<StreamEvent> {
        vec![
            StreamEvent::MessageStart,
            StreamEvent::TextDelta {
                text: s.to_string(),
            },
            StreamEvent::ContentBlockStop,
            StreamEvent::MessageStop {
                stop_reason: StopReason::EndTurn,
            },
        ]
    }

    /// A turn that calls one tool with the given (whole) JSON argument string.
    pub fn tool_call(id: &str, name: &str, args_json: &str) -> Vec<StreamEvent> {
        vec![
            StreamEvent::MessageStart,
            StreamEvent::ToolUseStart {
                id: id.to_string(),
                name: name.to_string(),
            },
            StreamEvent::InputJsonDelta {
                partial_json: args_json.to_string(),
            },
            StreamEvent::ContentBlockStop,
            StreamEvent::MessageStop {
                stop_reason: StopReason::ToolUse,
            },
        ]
    }
}
