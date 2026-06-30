//! `write` — create or overwrite a file (creating parent directories).

use agent_core::ToolError;
use agent_core::tool::Tool;
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
        input.get("path").and_then(Value::as_str).map(str::to_string)
    }

    async fn run(&self, input: Value) -> Result<String, ToolError> {
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
        std::fs::write(path, content)
            .map_err(|e| ToolError::Execution(format!("write {path}: {e}")))?;
        Ok(format!("wrote {} bytes to {path}", content.len()))
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
            .unwrap();
        assert!(out.contains("5 bytes"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
    }

    #[tokio::test]
    async fn missing_content_is_invalid_input() {
        let err = Write.run(json!({ "path": "/tmp/x" })).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }
}
