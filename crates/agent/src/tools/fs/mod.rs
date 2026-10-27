//! The filesystem backend seam: [`FsBackend`], its types and its shared result policy live in the
//! `beyond-sandbox` crate (with [`shell::ShellFs`], the backend that works through any command
//! runner) and are re-exported here. [`local::LocalFs`], the host's own filesystem, is the agent's.

pub mod local;

pub use beyond_sandbox::fs::*;

/// A backend failure as the tool error the model reads: a malformed query is the caller's mistake
/// (`InvalidInput`), anything else the backend's (`Execution`). The message is the same text
/// whichever backend produced it.
pub fn tool_error(e: FsError) -> agent_core::ToolError {
    match e {
        FsError::InvalidQuery(m) => agent_core::ToolError::InvalidInput(m),
        FsError::Backend(m) => agent_core::ToolError::Execution(m),
    }
}
