//! `grep` — regex search across files, gitignore-aware (ripgrep's `ignore` + `regex` crates).

use agent_core::ToolError;
use agent_core::tool::Tool;
use async_trait::async_trait;
use globset::Glob;
use ignore::WalkBuilder;
use regex::RegexBuilder;
use serde_json::{Value, json};

/// Default cap on reported matches.
const DEFAULT_LIMIT: usize = 100;
/// Long match lines are clipped to keep output readable.
const MAX_LINE: usize = 500;

pub struct Grep;

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str {
        "grep"
    }
    fn description(&self) -> &str {
        "Search file contents by regular expression, honoring .gitignore. Optionally restrict to a \
         `path`, a `glob` (e.g. \"*.rs\"), case-insensitive with `ignore_case`."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regular expression to search for." },
                "path": { "type": "string", "description": "Directory or file to search (default \".\")." },
                "glob": { "type": "string", "description": "Only search files matching this glob." },
                "ignore_case": { "type": "boolean", "description": "Case-insensitive (default false)." },
                "limit": { "type": "integer", "description": "Max matches to report (default 100)." }
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
        let ignore_case = input
            .get("ignore_case")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let re = RegexBuilder::new(pattern)
            .case_insensitive(ignore_case)
            .build()
            .map_err(|e| ToolError::InvalidInput(format!("bad regex: {e}")))?;

        let glob = match input.get("glob").and_then(Value::as_str) {
            Some(g) => Some(
                Glob::new(g)
                    .map_err(|e| ToolError::InvalidInput(format!("bad glob: {e}")))?
                    .compile_matcher(),
            ),
            None => None,
        };

        let mut out = String::new();
        let mut hits = 0usize;
        // `hidden(false)` includes dotfiles (like ripgrep --hidden); .gitignore is respected by default.
        for entry in WalkBuilder::new(root).hidden(false).build() {
            if hits >= limit {
                out.push_str(&format!(
                    "… (match limit {limit} reached; narrow the pattern or raise `limit`)\n"
                ));
                break;
            }
            let Ok(entry) = entry else { continue };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            if let Some(g) = &glob {
                if !g.is_match(path) {
                    continue;
                }
            }
            // Skip non-UTF8 / binary files silently.
            let Ok(content) = std::fs::read_to_string(path) else {
                continue;
            };
            for (i, line) in content.lines().enumerate() {
                if hits >= limit {
                    break;
                }
                if re.is_match(line) {
                    let shown = if line.len() > MAX_LINE {
                        format!("{}… [truncated]", &line[..MAX_LINE])
                    } else {
                        line.to_string()
                    };
                    out.push_str(&format!("{}:{}: {}\n", path.display(), i + 1, shown));
                    hits += 1;
                }
            }
        }
        if out.is_empty() {
            return Ok(format!("no matches for {pattern:?}"));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn finds_matches_with_path_and_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\nworld\nhello again\n").unwrap();
        std::fs::write(dir.path().join("b.log"), "nothing here\n").unwrap();

        let out = Grep
            .run(json!({ "pattern": "hello", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap();
        assert!(out.contains("a.txt:1: hello"));
        assert!(out.contains("a.txt:3: hello again"));
        assert!(!out.contains("b.log"));
    }

    #[tokio::test]
    async fn glob_filters_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("keep.rs"), "fn target() {}\n").unwrap();
        std::fs::write(dir.path().join("skip.txt"), "fn target() {}\n").unwrap();
        let out = Grep
            .run(json!({ "pattern": "target", "path": dir.path().to_str().unwrap(), "glob": "*.rs" }))
            .await
            .unwrap();
        assert!(out.contains("keep.rs"));
        assert!(!out.contains("skip.txt"));
    }

    #[tokio::test]
    async fn no_matches_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "abc\n").unwrap();
        let out = Grep
            .run(json!({ "pattern": "zzz", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap();
        assert!(out.contains("no matches"));
    }
}
