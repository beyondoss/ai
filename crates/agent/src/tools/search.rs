//! `search` — one tool in place of `grep`, `find` and `ls` (`--tool-tuning search-tool=merged`).
//!
//! An A/B candidate, not a new capability: it is a thin router over the three existing tools, so
//! every result, truncation marker and error is exactly what that tool would have returned. With a
//! `pattern` it searches file contents (`grep`); with only a `glob` it lists matching files
//! recursively (`find`); with neither it lists one directory (`ls`). The point is the advertised
//! surface: one schema instead of three (about 215 o200k tokens against 601) and one choice instead
//! of three. Claude Code makes the same move for names (one Glob tool, no `find`/`ls`).

use std::sync::Arc;

use agent_core::tool::{Tool, ToolProgress};
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Map, Value, json};

/// The registered name.
pub const NAME: &str = "search";

pub struct Search {
    grep: Arc<dyn Tool>,
    find: Arc<dyn Tool>,
    ls: Arc<dyn Tool>,
}

impl Search {
    /// Route over these three tools — the ones the registry would otherwise advertise, so they share
    /// its root and filesystem backend.
    pub fn new(grep: Arc<dyn Tool>, find: Arc<dyn Tool>, ls: Arc<dyn Tool>) -> Self {
        Self { grep, find, ls }
    }

    /// Which tool a call goes to, and the input that tool takes.
    fn route(&self, input: &Value) -> Result<(&Arc<dyn Tool>, Value), ToolError> {
        let obj = input.as_object().cloned().unwrap_or_default();
        let pick = |keys: &[&str]| -> Value {
            let mut m = Map::new();
            for k in keys {
                if let Some(v) = obj.get(*k).filter(|v| !v.is_null()) {
                    m.insert((*k).to_string(), v.clone());
                }
            }
            Value::Object(m)
        };
        let nonempty = |k: &str| {
            obj.get(k)
                .and_then(Value::as_str)
                .is_some_and(|s| !s.is_empty())
        };
        if nonempty("pattern") {
            return Ok((
                &self.grep,
                pick(&[
                    "pattern",
                    "path",
                    "glob",
                    "ignore_case",
                    "literal",
                    "context",
                    "limit",
                ]),
            ));
        }
        if nonempty("glob") {
            let mut v = pick(&["path", "limit"]);
            v["pattern"] = obj["glob"].clone();
            return Ok((&self.find, v));
        }
        for k in ["ignore_case", "literal", "context"] {
            if obj
                .get(k)
                .is_some_and(|v| !v.is_null() && v != &json!(false) && v != &json!(0))
            {
                return Err(ToolError::InvalidInput(format!(
                    "`{k}` applies to a content search; give a `pattern`"
                )));
            }
        }
        Ok((&self.ls, pick(&["path", "limit"])))
    }
}

#[async_trait]
impl Tool for Search {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        "Search the working tree, honoring .gitignore. With `pattern`: search file contents by regular \
         expression (optionally only files matching `glob`, with `ignore_case`, `literal`, `context` \
         lines). With only `glob` (e.g. \"**/*.rs\"): list matching files recursively. With neither: \
         list the directory at `path`. Results are sorted and truncated to `limit`."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regex to search file contents for. Omit to list files." },
                "glob": { "type": "string", "description": "With `pattern`, only search files matching it (\"!\" prefix excludes). Without, list files matching it." },
                "path": { "type": "string", "description": "Directory or file to search (default \".\")." },
                "ignore_case": { "type": "boolean", "description": "Case-insensitive content search." },
                "literal": { "type": "boolean", "description": "Treat `pattern` as a literal string." },
                "context": { "type": "integer", "description": "Lines of context around each content match." },
                "limit": { "type": "integer", "description": "Max results." }
            }
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let (tool, input) = self.route(&input)?;
        tool.run(input).await
    }

    async fn run_streaming(
        &self,
        input: Value,
        progress: &ToolProgress,
    ) -> Result<ToolOutput, ToolError> {
        let (tool, input) = self.route(&input)?;
        tool.run_streaming(input, progress).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search(root: &std::path::Path) -> Search {
        Search::new(
            Arc::new(super::super::grep::Grep::new(root)),
            Arc::new(super::super::find::Find::new(root)),
            Arc::new(super::super::ls::Ls::new(root)),
        )
    }

    fn tree() -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join("src")).unwrap();
        std::fs::write(d.path().join("src/a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        std::fs::write(d.path().join("src/b.py"), "def alpha():\n    pass\n").unwrap();
        std::fs::write(d.path().join("README.md"), "Alpha project\n").unwrap();
        d
    }

    fn text(out: ToolOutput) -> String {
        out.text
    }

    /// Each mode returns byte-for-byte what the tool it routes to returns.
    #[tokio::test]
    async fn each_mode_is_exactly_the_tool_it_routes_to() {
        let d = tree();
        let s = search(d.path());
        let grep = super::super::grep::Grep::new(d.path());
        let find = super::super::find::Find::new(d.path());
        let ls = super::super::ls::Ls::new(d.path());

        let got = text(
            s.run(json!({ "pattern": "alpha", "glob": "*.rs" }))
                .await
                .unwrap(),
        );
        let want = text(
            grep.run(json!({ "pattern": "alpha", "glob": "*.rs" }))
                .await
                .unwrap(),
        );
        assert_eq!(got, want);
        assert!(got.contains("a.rs") && !got.contains("b.py"), "{got}");

        let got = text(s.run(json!({ "glob": "**/*.py" })).await.unwrap());
        let want = text(find.run(json!({ "pattern": "**/*.py" })).await.unwrap());
        assert_eq!(got, want);
        assert!(got.contains("b.py"), "{got}");

        let got = text(s.run(json!({ "path": "src" })).await.unwrap());
        let want = text(ls.run(json!({ "path": "src" })).await.unwrap());
        assert_eq!(got, want);
    }

    #[tokio::test]
    async fn content_options_without_a_pattern_are_rejected_not_ignored() {
        let d = tree();
        let err = search(d.path())
            .run(json!({ "ignore_case": true }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(m) if m.contains("pattern")));
    }

    #[tokio::test]
    async fn an_empty_pattern_lists_rather_than_matching_everything() {
        let d = tree();
        let s = search(d.path());
        let got = text(
            s.run(json!({ "pattern": "", "path": "src" }))
                .await
                .unwrap(),
        );
        let want = text(
            super::super::ls::Ls::new(d.path())
                .run(json!({ "path": "src" }))
                .await
                .unwrap(),
        );
        assert_eq!(got, want);
    }
}
