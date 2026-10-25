//! Beyond agent harness — library surface.
//!
//! The coding/platform tools and the headless `serve` protocol live here (rather than inline in
//! `main.rs`) so integration tests and benches can drive them directly. The binary (`main.rs`) is a
//! thin CLI shim over this crate. Model traffic flows through the gateway; see `agent_core`.

// Production code stays panic-free (workspace lints); a unit test asserting a precondition with
// `.unwrap()` *is* the assertion — allow it under `test`, matching the binary and agent-core roots.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

/// The installed command's name — what every error, warning and hint tells a user to type
/// (`beyond-ai-agent login anthropic`). One place, so a hint can't name a command that doesn't
/// exist; `program_name_tests` fails on a string that spells another name.
pub const PROGRAM: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod program_name_tests;

pub mod agents;
pub mod approval;
pub mod auth_credential_source;
pub mod auth_store;
pub mod bundle;
pub mod conformance;
pub mod exec_endpoint;
pub mod export;
pub mod file_lock;
pub mod gateway_credential;
pub mod grant;
pub mod lifecycle;
pub mod managed_mcp_token;
pub mod mcp_auth_store;
pub mod mcp_json;
pub mod mcp_resume;
pub mod memory;
pub mod metrics;
pub mod mode;
pub mod notify_token;
pub mod oauth;
pub(crate) mod path_utils;
pub mod policy;
pub mod prompts;
pub mod resources;
pub mod retry;
pub mod serve;
pub mod serve_ws;
pub mod service;
pub mod session_store;
pub mod settings;
pub mod shard_mount;
pub mod skills;
pub(crate) mod store_file;
pub mod subreaper;
#[cfg(test)]
pub(crate) mod test_support;
pub mod timing;
pub mod tools;
#[cfg(test)]
pub(crate) mod tracing_test;
pub mod trust_store;
pub mod worktree;
