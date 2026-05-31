//! Beyond AI gateway library.
//!
//! `src/main.rs` wires these modules into a Pingora `ProxyHttp` service. The load-bearing logic
//! (virtual-key verification, deny-set, usage parsing, routing, request peek) lives in modules
//! free of Pingora/IO so it is unit-tested without a running proxy or live providers.

// Application crate: no `unsafe` is needed, so forbid it outright. `unused_must_use` is denied so
// a dropped `Result` (e.g. an unchecked `write_response_*`) is a hard error, not a silent swallow.
#![deny(unsafe_code)]
#![deny(unused_must_use)]

pub mod config;
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
