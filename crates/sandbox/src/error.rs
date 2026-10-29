//! The error vocabulary of FILES-OVER-EXEC.md section 3.1 (plus `not_offered` for a
//! transport or surface that lacks a capability, matching the TypeScript client).
//!
//! Layer-1 calls still return [`std::io::Error`]: that is the Rust idiom for I/O, and the
//! [`Exec`](crate::Exec) trait stays transport-shaped. Callers that want the spec's kinds
//! convert with [`SandboxError::from`] / [`SandboxError::classify`]. The MCP server and any
//! surface that talks to a conformance driver should prefer this type.

use std::fmt;

use crate::bulk::BulkError;
use crate::fs::FsError;

/// One kind from FILES-OVER-EXEC.md §3.1, plus [`Self::NotOffered`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    NotFound,
    IsDirectory,
    NotADirectory,
    PermissionDenied,
    Io,
    InvalidQuery,
    Conflict,
    TooLarge,
    Auth,
    Timeout,
    Transport,
    Protocol,
    Endpoint,
    InvalidRequest,
    /// The transport or surface does not offer this operation (no jobs, no stdin, …).
    NotOffered,
}

impl ErrorKind {
    /// The wire / JSON name (`not_found`, `auth`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotFound => "not_found",
            Self::IsDirectory => "is_directory",
            Self::NotADirectory => "not_a_directory",
            Self::PermissionDenied => "permission_denied",
            Self::Io => "io",
            Self::InvalidQuery => "invalid_query",
            Self::Conflict => "conflict",
            Self::TooLarge => "too_large",
            Self::Auth => "auth",
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::Protocol => "protocol",
            Self::Endpoint => "endpoint",
            Self::InvalidRequest => "invalid_request",
            Self::NotOffered => "not_offered",
        }
    }

    /// Parse a kind name; unknown names are [`None`] (callers that must accept any string
    /// fall back to [`Self::Io`] themselves).
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "not_found" => Self::NotFound,
            "is_directory" => Self::IsDirectory,
            "not_a_directory" => Self::NotADirectory,
            "permission_denied" => Self::PermissionDenied,
            "io" => Self::Io,
            "invalid_query" => Self::InvalidQuery,
            "conflict" => Self::Conflict,
            "too_large" => Self::TooLarge,
            "auth" => Self::Auth,
            "timeout" => Self::Timeout,
            "transport" => Self::Transport,
            "protocol" => Self::Protocol,
            "endpoint" => Self::Endpoint,
            "invalid_request" => Self::InvalidRequest,
            "not_offered" => Self::NotOffered,
            _ => return None,
        })
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A failure carrying the spec's [`ErrorKind`] and a free-text detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxError {
    pub kind: ErrorKind,
    pub detail: String,
}

impl SandboxError {
    pub fn new(kind: ErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    pub fn invalid_request(detail: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidRequest, detail)
    }

    pub fn not_offered(detail: impl Into<String>) -> Self {
        Self::new(ErrorKind::NotOffered, detail)
    }

    pub fn protocol(detail: impl Into<String>) -> Self {
        Self::new(ErrorKind::Protocol, detail)
    }

    pub fn kind_str(&self) -> &'static str {
        self.kind.as_str()
    }

    /// Classify a transport or helper failure from its message and optional [`std::io::ErrorKind`].
    /// Same rules the TypeScript/Python clients and the MCP server use (FILES-OVER-EXEC §3.2–3.3).
    pub fn classify(msg: &str, io_kind: Option<std::io::ErrorKind>) -> Self {
        let err = |k: ErrorKind| Self::new(k, msg);
        if io_kind == Some(std::io::ErrorKind::TimedOut) || msg.contains("timed out") {
            return err(ErrorKind::Timeout);
        }
        if io_kind == Some(std::io::ErrorKind::Unsupported) {
            return err(ErrorKind::NotOffered);
        }
        if msg.contains("returned 401")
            || msg.contains("returned 403")
            || msg.contains("token source failed")
        {
            return err(ErrorKind::Auth);
        }
        if msg.contains("exceeded the") && msg.contains("-byte cap") {
            return err(ErrorKind::TooLarge);
        }
        if msg.contains("returned 422") && msg.contains("cap") {
            return err(ErrorKind::TooLarge);
        }
        if msg.contains("unreachable") {
            return err(ErrorKind::Transport);
        }
        if msg.contains("unparseable JSON")
            || msg.contains("not a stream")
            || msg.contains("unparseable frame")
            || msg.contains("unparseable job frame")
            || msg.contains("is not standard base64")
            || msg.contains("without a newline")
            || msg.contains("unreadable job")
        {
            return err(ErrorKind::Protocol);
        }
        if msg.contains("exec endpoint") && msg.contains(" returned ") {
            return err(ErrorKind::Endpoint);
        }
        if msg.contains("exec endpoint") || msg.contains("job stream") {
            return err(ErrorKind::Transport);
        }
        err(classify_command_text(msg))
    }

    /// FILES-OVER-EXEC §3.3: classify a command that ran and failed, from its stderr text alone.
    pub fn from_command_stderr(detail: &str) -> Self {
        Self::new(classify_command_text(detail), detail)
    }
}

fn classify_command_text(msg: &str) -> ErrorKind {
    let m = msg.to_ascii_lowercase();
    if m.contains("no such file or directory")
        || m.contains("directory nonexistent")
        || m.contains("nonexistent directory")
    {
        ErrorKind::NotFound
    } else if m.contains("is a directory") {
        ErrorKind::IsDirectory
    } else if m.contains("not a directory") {
        ErrorKind::NotADirectory
    } else if m.contains("permission denied") {
        ErrorKind::PermissionDenied
    } else {
        ErrorKind::Io
    }
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.detail.is_empty() {
            write!(f, "{}", self.kind)
        } else {
            write!(f, "{}: {}", self.kind, self.detail)
        }
    }
}

impl std::error::Error for SandboxError {}

impl From<&std::io::Error> for SandboxError {
    fn from(e: &std::io::Error) -> Self {
        Self::classify(&e.to_string(), Some(e.kind()))
    }
}

impl From<std::io::Error> for SandboxError {
    fn from(e: std::io::Error) -> Self {
        Self::from(&e)
    }
}

impl From<SandboxError> for std::io::Error {
    fn from(e: SandboxError) -> Self {
        let kind = match e.kind {
            ErrorKind::NotFound => std::io::ErrorKind::NotFound,
            ErrorKind::PermissionDenied => std::io::ErrorKind::PermissionDenied,
            ErrorKind::Timeout => std::io::ErrorKind::TimedOut,
            ErrorKind::TooLarge => std::io::ErrorKind::FileTooLarge,
            ErrorKind::InvalidRequest | ErrorKind::InvalidQuery => std::io::ErrorKind::InvalidInput,
            ErrorKind::NotOffered => std::io::ErrorKind::Unsupported,
            ErrorKind::IsDirectory
            | ErrorKind::NotADirectory
            | ErrorKind::Io
            | ErrorKind::Conflict
            | ErrorKind::Auth
            | ErrorKind::Transport
            | ErrorKind::Protocol
            | ErrorKind::Endpoint => std::io::ErrorKind::Other,
        };
        std::io::Error::new(kind, e)
    }
}

impl From<FsError> for SandboxError {
    fn from(e: FsError) -> Self {
        match e {
            FsError::InvalidQuery(m) => Self::new(ErrorKind::InvalidQuery, m),
            FsError::Backend(m) => Self::classify(&m, None),
        }
    }
}

impl From<BulkError> for SandboxError {
    fn from(e: BulkError) -> Self {
        match e {
            BulkError::InvalidRequest(m) => Self::invalid_request(m),
            BulkError::Archive(m) => Self::protocol(m),
            BulkError::Command(m) => Self::classify(&m, None),
            BulkError::Io(e) => Self::from(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_follow_the_text() {
        let k = |m: &str| SandboxError::classify(m, None).kind;
        assert_eq!(
            k("read /a: No such file or directory (os error 2)"),
            ErrorKind::NotFound
        );
        assert_eq!(k("exec endpoint returned 401"), ErrorKind::Auth);
        assert_eq!(k("exec endpoint returned 500: boom"), ErrorKind::Endpoint);
        assert_eq!(
            k("exec endpoint unreachable: refused"),
            ErrorKind::Transport
        );
        assert_eq!(
            k("unreadable job answer: the endpoint did not report the job's start: nope"),
            ErrorKind::Protocol
        );
        assert_eq!(
            k(
                "exec endpoint http://x answered a stream request with something that is not a stream"
            ),
            ErrorKind::Protocol
        );
        assert_eq!(
            k("exec endpoint http://x: unparseable frame (expected value): nope"),
            ErrorKind::Protocol
        );
        assert_eq!(
            k("job stream: output is not standard base64 (bad)"),
            ErrorKind::Protocol
        );
        assert_eq!(
            k("exec endpoint x: the stream ended before the command's final frame"),
            ErrorKind::Transport
        );
        assert_eq!(k("write: Permission denied"), ErrorKind::PermissionDenied);
        assert_eq!(k("something else"), ErrorKind::Io);
        assert_eq!(
            SandboxError::classify("x", Some(std::io::ErrorKind::Unsupported)).kind,
            ErrorKind::NotOffered
        );
    }

    #[test]
    fn round_trips_through_io_error() {
        let e = SandboxError::new(ErrorKind::Auth, "returned 401");
        let io = std::io::Error::from(e.clone());
        let back = SandboxError::from(io);
        assert_eq!(back.kind, ErrorKind::Auth);
        assert!(back.detail.contains("401"));
    }
}
