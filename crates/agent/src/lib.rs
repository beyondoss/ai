//! Beyond agent harness — library surface.
//!
//! The coding/platform tools and the headless `serve` protocol live here (rather than inline in
//! `main.rs`) so integration tests and benches can drive them directly. The binary (`main.rs`) is a
//! thin CLI shim over this crate. Model traffic flows through the gateway; see `agent_core`.

// Production code stays panic-free (workspace lints); a unit test asserting a precondition with
// `.unwrap()` *is* the assertion — allow it under `test`, matching the binary and agent-core roots.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod serve;
pub mod tools;
