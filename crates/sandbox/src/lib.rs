#![doc = include_str!("../README.md")]
// `.unwrap()` *is* the assertion in a unit test.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod exec;
pub mod exec_endpoint;
pub mod fs;
pub mod procgroup;
mod sandbox;
#[cfg(test)]
mod testing;
mod tls;
mod token;

pub use exec::{CommandRunner as Exec, ExecResult};
pub use exec_endpoint::{HttpExecRunner as HttpExec, TemplateRunner};
pub use fs::FsBackend;
pub use fs::shell::ShellFs as Files;
pub use sandbox::{Command, Jobs, Sandbox};
pub use token::TokenSource;
