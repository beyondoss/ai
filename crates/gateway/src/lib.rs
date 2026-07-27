//! Beyond AI gateway library.
//!
//! `src/main.rs` wires these modules into a Pingora `ProxyHttp` service. The load-bearing logic
//! (virtual-key verification, deny-set, usage parsing, routing, request peek) lives in modules
//! free of Pingora/IO so it is unit-tested without a running proxy or live providers.

// Lint gates (`unsafe_code = "forbid"`, `unused_must_use = "deny"`) live in `[workspace.lints]` so
// they apply to *both* crate roots — this lib and the `main.rs` binary — not just whichever unit
// carries a crate-level `#![deny]`. A dropped `Result` (e.g. an unchecked `write_response_*`) is
// therefore a hard error, and `unsafe` is forbidden, everywhere in the crate.
//
// `unwrap_used`/`expect_used`/`panic` are denied in production code (see `[workspace.lints.clippy]`)
// but a unit test's whole job is to assert a precondition holds — `.unwrap()` *is* the assertion — so
// allow them in `#[cfg(test)]` modules.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod admin;
pub mod capture;
pub mod capture_sink;
pub mod circuit_breaker;
pub mod config;
pub mod control;
pub mod deny;
pub mod doctor;
pub mod error;
pub mod key;
pub mod metrics;
pub mod peek;
pub mod proxy;
pub mod ratelimit;
pub mod route;
pub mod secret;
pub mod state;
pub mod store_watch;
pub mod usage;
