#![doc = include_str!("../README.md")]
// `.unwrap()` *is* the assertion in a unit test.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod bulk;
pub mod error;
pub mod exec;
pub mod exec_endpoint;
pub mod fs;
pub mod procgroup;
mod sandbox;
#[cfg(test)]
mod testing;
mod tls;
mod token;

pub use bulk::{BulkError, Sources, Transfer};
pub use error::{ErrorKind, SandboxError};
pub use exec::{CommandRunner as Exec, ExecOutput, ExecResult};
pub use exec_endpoint::HttpExecRunner as HttpExec;
pub use fs::FsBackend;
pub use fs::shell::ShellFs as Files;
pub use sandbox::{Command, Jobs, Sandbox, StartJob};
pub use token::TokenSource;
