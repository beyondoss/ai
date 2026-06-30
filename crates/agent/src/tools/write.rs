//! `write` — create or overwrite a file (creating parent directories).

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

pub struct Write;

#[async_trait]
impl Tool for Write {
    fn name(&self) -> &str {
        "write"
    }
    fn description(&self) -> &str {
        "Create or overwrite a file with the given contents. Parent directories are created."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File path to write." },
                "content": { "type": "string", "description": "Full file contents." }
            },
            "required": ["path", "content"]
        })
    }

    fn write_target(&self, input: &Value) -> Option<String> {
        input
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_string)
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let path = input
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `path`".into()))?;
        let content = input
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `content`".into()))?;
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    ToolError::Execution(format!("mkdir {}: {e}", parent.display()))
                })?;
            }
        }
        // Atomic temp-file + rename: an overwrite killed mid-write must not leave a half-written
        // file — the same guarantee `edit` makes (and which `serve` reattach depends on for the
        // session file). `create_dir_all` above ensures the sibling temp's directory exists.
        super::write_atomic(path, content.as_bytes())
            .map_err(|e| ToolError::Execution(format!("write {path}: {e}")))?;
        Ok(format!("wrote {} bytes to {path}", content.len()).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn writes_and_creates_parents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/dir/file.txt");
        let out = Write
            .run(json!({ "path": path.to_str().unwrap(), "content": "hello" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("5 bytes"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[tokio::test]
    async fn missing_content_is_invalid_input() {
        let err = Write.run(json!({ "path": "/tmp/x" })).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn overwrites_existing_file_and_leaves_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, "old contents").unwrap();
        Write
            .run(json!({ "path": path.to_str().unwrap(), "content": "new" }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        // The atomic write must not leave its sibling temp behind.
        let temps: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(temps.is_empty(), "atomic write left a temp file behind");
    }
}
