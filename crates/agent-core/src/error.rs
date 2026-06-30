//! Error types for the agent core. Loop/transport failures are [`Error`]; a tool's own failure is
//! [`ToolError`], which the loop converts into a `tool_result` with `is_error = true` rather than
//! aborting the run — a failed tool is a value the model can react to, not a dead turn.

use thiserror::Error;

/// A failure inside a [`crate::Tool`].
#[derive(Debug, Error)]
pub enum ToolError {
    /// The arguments object didn't match the tool's schema / expectations.
    #[error("invalid tool input: {0}")]
    InvalidInput(String),
    /// The tool ran but failed (command non-zero, file missing, etc.).
    #[error("tool execution failed: {0}")]
    Execution(String),
    /// An underlying IO error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// A failure in the agent loop or its transport.
#[derive(Debug, Error)]
pub enum Error {
    /// The model transport failed (network, decode, gateway error).
    #[error("transport error: {0}")]
    Transport(String),
    // Note: an unknown tool is *not* an `Error`. The loop turns it into an error `tool_result`
    // (`is_error: true`) the model can react to next turn, never aborting the run — so there is no
    // fatal `UnknownTool` variant by design.
    /// The loop hit its step ceiling without the model ending its turn.
    #[error("reached max steps ({0}) without completion")]
    MaxSteps(u32),
    /// A streamed tool call carried malformed JSON arguments.
    #[error("malformed tool-call arguments: {0}")]
    MalformedToolInput(String),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, Error>;
