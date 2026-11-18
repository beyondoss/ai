#![doc = include_str!("../README.md")]
// `.unwrap()` *is* the assertion in a unit test.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod config;
pub mod error;
pub mod http;
pub mod server;

pub use config::Config;
pub use server::SandboxMcp;

/// The longest JSON-RPC line either end of the stdio pipe reads: rmcp's own default is 16 MiB, and a
/// line past it is dropped without an answer. An `exec` result carries its output twice (the text
/// and the structured copy, each base64), so 7 MB of output is already past 16 MiB.
pub const MAX_LINE: usize = 256 * 1024 * 1024;
