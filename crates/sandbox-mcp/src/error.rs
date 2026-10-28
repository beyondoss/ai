//! The error vocabulary of the tools. A tool that fails answers with a result (`isError: true`)
//! whose structured content is `{"error": {"kind", "detail"}}`, never a protocol error, so the
//! agent can read it and react. `kind` is one of the kinds of `sdk/conformance/DRIVER.md`
//! section 4, so a conformance driver can pass it through unchanged.

use beyond_sandbox::fs::FsError;
use rmcp::model::CallToolResult;
use serde_json::{Value, json};

/// A failed tool call: the spec's error kind, and text for the model.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolError {
    pub kind: &'static str,
    pub detail: String,
}

impl ToolError {
    pub fn new(kind: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    pub fn invalid(detail: impl Into<String>) -> Self {
        Self::new("invalid_request", detail)
    }

    /// The error as a tool result.
    pub fn into_result(self) -> CallToolResult {
        CallToolResult::structured_error(json!({
            "error": { "kind": self.kind, "detail": self.detail }
        }))
    }
}

impl From<&std::io::Error> for ToolError {
    fn from(e: &std::io::Error) -> Self {
        classify(&e.to_string(), Some(e.kind()))
    }
}

impl From<std::io::Error> for ToolError {
    fn from(e: std::io::Error) -> Self {
        Self::from(&e)
    }
}

impl From<beyond_sandbox::BulkError> for ToolError {
    fn from(e: beyond_sandbox::BulkError) -> Self {
        use beyond_sandbox::BulkError;
        match e {
            BulkError::InvalidRequest(m) => Self::invalid(m),
            BulkError::Archive(m) => Self::new("protocol", m),
            BulkError::Command(m) => classify(&m, None),
            BulkError::Io(e) => Self::from(e),
        }
    }
}

impl From<FsError> for ToolError {
    fn from(e: FsError) -> Self {
        match e {
            FsError::InvalidQuery(m) => Self::new("invalid_query", m),
            FsError::Backend(m) => classify(&m, None),
        }
    }
}

/// A tool's outcome as the result the client sees.
pub fn reply(r: Result<Value, ToolError>) -> CallToolResult {
    match r {
        Ok(v) => CallToolResult::structured(v),
        Err(e) => e.into_result(),
    }
}

/// The kind of an error, from the text the client library produces (FILES-OVER-EXEC.md section 3.3
/// and the exec client's own messages).
pub fn classify(msg: &str, kind: Option<std::io::ErrorKind>) -> ToolError {
    let err = |k: &'static str| ToolError::new(k, msg);
    if kind == Some(std::io::ErrorKind::TimedOut) || msg.contains("timed out") {
        return err("timeout");
    }
    if kind == Some(std::io::ErrorKind::Unsupported) {
        return err("not_offered");
    }
    if msg.contains("returned 401") || msg.contains("returned 403") {
        return err("auth");
    }
    if msg.contains("exceeded the") && msg.contains("-byte cap") {
        return err("too_large");
    }
    if msg.contains("returned 422") && msg.contains("cap") {
        return err("too_large");
    }
    if msg.contains("unreachable") {
        return err("transport");
    }
    if msg.contains("unparseable JSON")
        || msg.contains("unparseable frame")
        || msg.contains("unparseable job frame")
        || msg.contains("is not standard base64")
        || msg.contains("without a newline")
    {
        return err("protocol");
    }
    if msg.contains("exec endpoint") && msg.contains(" returned ") {
        return err("endpoint");
    }
    if msg.contains("exec endpoint") || msg.contains("job stream") {
        return err("transport");
    }
    // A command that ran and failed: classify by what it said.
    let m = msg.to_ascii_lowercase();
    err(
        if m.contains("no such file or directory")
            || m.contains("directory nonexistent")
            || m.contains("nonexistent directory")
        {
            "not_found"
        } else if m.contains("is a directory") {
            "is_directory"
        } else if m.contains("not a directory") {
            "not_a_directory"
        } else if m.contains("permission denied") {
            "permission_denied"
        } else {
            "io"
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_follow_the_text() {
        let k = |m: &str| classify(m, None).kind;
        assert_eq!(
            k("read /a: No such file or directory (os error 2)"),
            "not_found"
        );
        assert_eq!(k("exec endpoint returned 401"), "auth");
        assert_eq!(k("exec endpoint returned 500: boom"), "endpoint");
        assert_eq!(k("exec endpoint unreachable: refused"), "transport");
        assert_eq!(
            k("exec endpoint http://x: unparseable frame (expected value): nope"),
            "protocol"
        );
        assert_eq!(
            k("job stream: output is not standard base64 (bad)"),
            "protocol"
        );
        assert_eq!(
            k("exec endpoint x: the stream ended before the command's final frame"),
            "transport"
        );
        assert_eq!(k("write: Permission denied"), "permission_denied");
        assert_eq!(k("something else"), "io");
        assert_eq!(
            classify("x", Some(std::io::ErrorKind::Unsupported)).kind,
            "not_offered"
        );
    }

    #[test]
    fn an_error_is_a_result_not_a_protocol_error() {
        let r = ToolError::new("conflict", "changed").into_result();
        assert_eq!(r.is_error, Some(true));
        assert_eq!(
            r.structured_content.unwrap()["error"]["kind"],
            serde_json::json!("conflict")
        );
    }
}
