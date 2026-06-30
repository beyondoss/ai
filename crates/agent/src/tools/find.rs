//! `find` — locate files by glob, gitignore-aware (ripgrep's `ignore` + `globset`).

use agent_core::ToolError;
use agent_core::tool::Tool;
use async_trait::async_trait;
use globset::Glob;
use ignore::WalkBuilder;
use serde_json::{Value, json};

/// Default cap on reported paths.
const DEFAULT_LIMIT: usize = 1000;

pub struct Find;

#[async_trait]
impl Tool for Find {
    fn name(&self) -> &str {
        "find"
    }
    fn description(&self) -> &str {
        "Find files by glob pattern (e.g. \"*.rs\", \"src/**/*.test.ts\"), honoring .gitignore. A \
         pattern without \"/\" matches the file name; with \"/\" it matches the full path."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern to match." },
                "path": { "type": "string", "description": "Directory to search (default \".\")." },
                "limit": { "type": "integer", "description": "Max results (default 1000)." }
            },
            "required": ["pattern"]
        })
    }

    async fn run(&self, input: Value) -> Result<String, ToolError> {
        let pattern = input
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("missing `pattern`".into()))?;
        let root = input.get("path").and_then(Value::as_str).unwrap_or(".");
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LIMIT);

        // A pattern with no path separator matches the basename; otherwise match the whole path
        // (prepend `**/` so `src/**/*.rs`-style anchored patterns still match nested roots).
        let basename_only = !pattern.contains('/');
        let glob_src = if basename_only || pattern.starts_with("**/") || pattern.starts_with('/') {
            pattern.to_string()
        } else {
            format!("**/{pattern}")
        };
        let matcher = Glob::new(&glob_src)
            .map_err(|e| ToolError::InvalidInput(format!("bad glob: {e}")))?
            .compile_matcher();

        let mut out = String::new();
        let mut hits = 0usize;
        for entry in WalkBuilder::new(root).hidden(false).build() {
            if hits >= limit {
                out.push_str(&format!(
                    "… (result limit {limit} reached; raise `limit` for more)\n"
                ));
                break;
            }
            let Ok(entry) = entry else { continue };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            let candidate = if basename_only {
                path.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            } else {
                path.to_string_lossy().into_owned()
            };
            if matcher.is_match(candidate.as_str()) {
                out.push_str(&format!("{}\n", path.display()));
                hits += 1;
            }
        }
        if out.is_empty() {
            return Ok(format!("no files matching {pattern:?}"));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn finds_by_basename_glob() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "").unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "").unwrap();
        std::fs::write(dir.path().join("README.md"), "").unwrap();

        let out = Find
            .run(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap();
        assert!(out.contains("main.rs"));
        assert!(out.contains("lib.rs"));
        assert!(!out.contains("README.md"));
    }

    #[tokio::test]
    async fn no_match_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        let out = Find
            .run(json!({ "pattern": "*.zzz", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap();
        assert!(out.contains("no files matching"));
    }
}
