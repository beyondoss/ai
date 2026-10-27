#![doc = include_str!("../README.md")]
// `.unwrap()` *is* the assertion in a unit test.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod config;
pub mod error;
pub mod http;
pub mod server;

pub use config::Config;
pub use server::SandboxMcp;
