//! Beyond agent harness — library surface.
//!
//! The coding/platform tools and the headless `serve` protocol live here (rather than inline in
//! `main.rs`) so integration tests and benches can drive them directly. The binary (`main.rs`) is a
//! thin CLI shim over this crate. Model traffic flows through the gateway; see `agent_core`.

// Production code stays panic-free (workspace lints); a unit test asserting a precondition with
// `.unwrap()` *is* the assertion — allow it under `test`, matching the binary and agent-core roots.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod agents;
pub mod auth_credential_source;
pub mod auth_store;
pub mod export;
pub mod gateway_credential;
pub mod mcp_auth_store;
pub mod oauth;
pub(crate) mod path_utils;
pub mod policy;
pub mod prompts;
pub mod resources;
pub mod retry;
pub mod serve;
pub mod serve_ws;
pub mod session_store;
pub mod settings;
pub mod skills;
pub mod timing;
pub mod tools;
#[cfg(test)]
pub(crate) mod tracing_test;
pub mod trust_store;
pub mod worktree;
