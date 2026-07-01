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

/// A transport that yields pre-scripted turns. Construct with one `Vec<StreamEvent>` per model turn
/// ([`new`](Self::new)), or — to script a turn that fails partway through, e.g. to exercise the loop's
/// mid-stream retry — one `Vec<Result<StreamEvent, Error>>` per turn ([`scripted`](Self::scripted)).
pub struct MockTransport {
    turns: Mutex<VecDeque<Vec<Result<StreamEvent>>>>,
    requests: Mutex<Vec<ModelRequest>>,
}

impl MockTransport {
    /// Script the transport with the turns it will return, in order. Every event succeeds; for a turn
    /// that fails partway through, use [`scripted`](Self::scripted) instead.
    pub fn new(turns: Vec<Vec<StreamEvent>>) -> Self {
        Self::scripted(
            turns
                .into_iter()
                .map(|t| t.into_iter().map(Ok).collect())
                .collect(),
        )
    }

    /// Script the transport with turns whose individual events may themselves be errors — a stream
    /// that starts fine and then dies partway through (a truncated connection, an in-band
    /// `overloaded_error`), the shape `run_turn`'s mid-stream retry needs to exercise.
    pub fn scripted(turns: Vec<Vec<Result<StreamEvent>>>) -> Self {
        Self {
            turns: Mutex::new(turns.into()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// The requests the loop has sent so far, in order (for asserting what the loop fed back).
    pub fn requests(&self) -> Vec<ModelRequest> {
        // Recover the data on a poisoned lock (a panicked test thread) rather than silently
        // returning an empty vec, which would mask the real failure behind a confusing assertion.
        self.requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// How many turns the loop has consumed.
    pub fn calls(&self) -> usize {
        self.requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }
}

#[async_trait]
impl ModelTransport for MockTransport {
    async fn stream(&self, req: ModelRequest) -> Result<EventStream> {
        self.requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(req);
        let turn = self
            .turns
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
            .ok_or_else(|| Error::Transport("MockTransport: no more scripted turns".into()))?;
        Ok(Box::pin(futures::stream::iter(turn)))
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

    /// A turn that emits a refusal explanation and ends with `StopReason::Refusal` — a distinct
    /// terminal condition from a normal end-of-turn (see `Agent::run_events_steered`).
    pub fn refusal(s: &str) -> Vec<StreamEvent> {
        vec![
            StreamEvent::MessageStart,
            StreamEvent::TextDelta {
                text: s.to_string(),
            },
            StreamEvent::ContentBlockStop,
            StreamEvent::MessageStop {
                stop_reason: StopReason::Refusal,
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
