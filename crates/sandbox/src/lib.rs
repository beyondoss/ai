//! `beyond-sandbox`: use a sandbox over the exec protocol.
//!
//! Three layers, each usable alone:
//!
//! 1. **Exec** ([`exec::CommandRunner`], also [`Exec`]): run a command somewhere, one-shot or
//!    streamed. [`HttpExecRunner`] (protocol v1.1 to v1.4) is the default implementation;
//!    [`TemplateRunner`] drives `ssh`/`docker exec`; anything else (a vendor SDK, a test double) is
//!    one trait implementation away.
//! 2. **Helpers**, built only on layer 1: [`Files`] (read, write, edit, list, search, glob) and
//!    [`Jobs`] (long-running commands that outlive the call).
//! 3. **Surface**: [`Sandbox`], one object that connects, owns the three, and always exposes the raw
//!    [`Exec`].
//!
//! The wire protocol is specified in the module doc of [`exec_endpoint`]; the file helpers' exact
//! commands in `sdk/conformance/FILES-OVER-EXEC.md`.

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
