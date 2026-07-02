//! Beyond agent harness — core.
//!
//! A runtime-agnostic, network-agnostic agent loop, modeled on Pi (`pi-agent-core` + the dialect
//! half of `pi-ai`). The pieces here are deliberately free of HTTP, providers, and any executor so
//! they unit-test without a network or a live model — the same split the gateway uses to keep its
//! load-bearing logic testable without Pingora.
//!
//! Two seams keep everything above the wire testable with mocks:
//! - [`Tool`] — capabilities are values in a [`ToolRegistry`]; tests register a mock tool.
//! - [`ModelTransport`] — the loop talks to this, never to `reqwest`. The real HTTP-to-gateway
//!   client and the test `MockTransport` both implement it (added in later milestones).
//!
//! **The Beyond twist:** the model layer never manages provider keys or endpoints. It speaks
//! OpenAI/Anthropic *wire* to the Beyond gateway, which owns routing, auth, and metering. So the
//! only part of `pi-ai` worth porting is the dialect-agnostic message model in [`message`].

// Production code in this crate is panic-free (see `[workspace.lints.clippy]`); a unit test's whole
// job is to assert a precondition holds, so `.unwrap()` *is* the assertion — allow it under `test`.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod agent;
pub mod branch_summary;
pub mod client;
pub mod compaction;
pub mod dialect;
pub mod error;
pub mod hooks;
pub mod message;
pub mod mock;
pub mod models;
pub mod session;
pub mod steering;
pub mod tool;
pub mod transport;
pub mod validation;
pub mod write_lock;

pub use agent::{Agent, AgentEvent};
pub use branch_summary::{
    BRANCH_SUMMARY_INSTRUCTION, BRANCH_SUMMARY_MARKER, branch_summary_request,
};
pub use client::GatewayClient;
pub use compaction::{CompactionConfig, CompactionProvenance, CompactionReason};
pub use error::{Error, Result, ToolError};
pub use hooks::{AgentHooks, CheckpointHook, NoCheckpoint, NoHooks};
pub use message::{
    ContentBlock, ImageSource, Message, Role, StopReason, StreamEvent, TokenUsage, ToolDef,
};
pub use mock::MockTransport;
pub use models::{
    ModelCaps, ThinkingLevel, capabilities, clamp_thinking_level, next_available_thinking_level,
    thinking_for_level,
};
pub use session::Session;
pub use steering::{QueueMode, Steering, SteeringMessage};
pub use tool::{Tool, ToolOutput, ToolProgress, ToolRegistry, ToolUpdate};
pub use transport::{
    ModelRequest, ModelTransport, ReasoningEffort, ReasoningSummary, ServiceTier, ToolChoice,
};
pub use write_lock::WriteLockRegistry;
// Re-exported because it appears in `Agent::run_events_cancellable`'s signature — callers shouldn't
// need a direct `tokio-util` dependency to drive a cancellable run.
pub use tokio_util::sync::CancellationToken;
