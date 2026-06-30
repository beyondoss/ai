//! `read` — read a file, optionally a line range, with line numbers.

use agent_core::ToolError;
use agent_core::tool::Tool;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Default cap on returned lines when no `limit` is given (keeps large files from flooding context).
const DEFAULT_LIMIT: usize = 2000;

pub struct Read;

#[async_trait]
impl Tool for Read {
    fn name(&self) -> &str {
        "read"
    }
    fn description(&self) -> &str {
        "Read a text file. Optionally start at a 1-based line `offset` and return up to `limit` \
         lines. Output is line-numbered."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path to read." },
                "offset": { "type": "integer", "description": "1-based first line to read." },
                "limit": { "type": "integer", "description": "Max lines to return." }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, input: Value) -> Result<String, ToolError> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `path`".into()))?;
        let content = std::fs::read_to_string(path)
            .map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
        if content.is_empty() {
            return Ok("(empty file)".into());
        }

        let offset = input
            .get("offset")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .max(1) as usize;
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let mut out = String::new();
        let mut shown = 0usize;
        for (i, line) in content.lines().enumerate() {
            let lineno = i + 1;
            if lineno < offset {
                continue;
            }
            if shown >= limit {
                out.push_str(&format!(
                    "… (truncated at {limit} lines; pass a larger `limit` to see more)\n"
                ));
                break;
            }
            out.push_str(&format!("{lineno:>6}\t{line}\n"));
            shown += 1;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_file(contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        f
    }

    #[tokio::test]
    async fn reads_with_line_numbers() {
        let f = tmp_file("alpha\nbeta\ngamma\n");
        let out = Read
            .run(json!({ "path": f.path().to_str().unwrap() }))
            .await
            .unwrap();
        assert!(out.contains("     1\talpha"));
        assert!(out.contains("     3\tgamma"));
    }

    #[tokio::test]
    async fn honors_offset_and_limit() {
        let f = tmp_file("a\nb\nc\nd\ne\n");
        let out = Read
            .run(json!({ "path": f.path().to_str().unwrap(), "offset": 2, "limit": 2 }))
            .await
            .unwrap();
        assert!(out.contains("     2\tb"));
        assert!(out.contains("     3\tc"));
        assert!(!out.contains("\td\n"));
    }

    #[tokio::test]
    async fn missing_file_is_execution_error() {
        let err = Read
            .run(json!({ "path": "/no/such/file/xyz" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
    }
}
