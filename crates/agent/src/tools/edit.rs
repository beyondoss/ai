//! `edit` — exact string replacement in a file (unique match unless `replace_all`).

use agent_core::ToolError;
use agent_core::tool::Tool;
use async_trait::async_trait;
use serde_json::{Value, json};

pub struct Edit;

#[async_trait]
impl Tool for Edit {
    fn name(&self) -> &str {
        "edit"
    }
    fn description(&self) -> &str {
        "Apply exact-match replacements to a file. Pass an `edits` array of {old_string, new_string} \
         (each `old_string` must match exactly once), or a single old_string/new_string. With \
         `replace_all`, a single edit replaces every occurrence."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "File to edit." },
                "edits": {
                    "type": "array",
                    "description": "Ordered replacements; each old_string must match exactly once.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_string": { "type": "string" },
                            "new_string": { "type": "string" }
                        },
                        "required": ["old_string", "new_string"]
                    }
                },
                "old_string": { "type": "string", "description": "Single-edit form: exact text to replace." },
                "new_string": { "type": "string", "description": "Single-edit form: replacement text." },
                "replace_all": { "type": "boolean", "description": "Single-edit form: replace every occurrence." }
            },
            "required": ["path"]
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
        let edits = parse_edits(&input)?;
        let replace_all = input
            .get("replace_all")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let mut content = std::fs::read_to_string(path)
            .map_err(|e| ToolError::Execution(format!("read {path}: {e}")))?;
        let mut applied = 0usize;
        for (old, new) in &edits {
            let count = content.matches(old.as_str()).count();
            if count == 0 {
                return Err(ToolError::InvalidInput(format!(
                    "`old_string` not found in {path}: {old:?}"
                )));
            }
            if count > 1 && !(replace_all && edits.len() == 1) {
                return Err(ToolError::InvalidInput(format!(
                    "`old_string` is not unique in {path} ({count} matches): {old:?}; add surrounding context"
                )));
            }
            // Replace once per edit (or all, only for the single-edit replace_all form).
            content = if replace_all && edits.len() == 1 {
                applied += count;
                content.replace(old.as_str(), new)
            } else {
                applied += 1;
                content.replacen(old.as_str(), new, 1)
            };
        }
        write_atomic(path, &content)
            .map_err(|e| ToolError::Execution(format!("write {path}: {e}")))?;
        Ok(format!(
            "edited {path} ({applied} replacement{})",
            if applied == 1 { "" } else { "s" }
        ))
    }
}

/// Overwrite `path` atomically: write a sibling temp file, then `rename` it over the target.
/// `rename(2)` is atomic on a single filesystem, so a concurrent reader — or a crash mid-write —
/// sees either the original file or the fully-edited one, never a half-written source file. A bare
/// `std::fs::write` truncates in place and would leave a partial file if the process died between
/// truncation and the final byte. The temp file is a sibling (same directory) so the rename stays
/// within one filesystem.
fn write_atomic(path: &str, content: &str) -> std::io::Result<()> {
    let p = std::path::Path::new(path);
    let tmp = match p.file_name() {
        Some(name) => p.with_file_name(format!(".{}.tmp", name.to_string_lossy())),
        None => return Err(std::io::Error::other(format!("invalid path: {path}"))),
    };
    std::fs::write(&tmp, content)?;
    match std::fs::rename(&tmp, p) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp); // don't leave the temp behind on failure
            Err(e)
        }
    }
}

/// Accept either the `edits` array form (pi-style) or the single old_string/new_string form.
fn parse_edits(input: &Value) -> Result<Vec<(String, String)>, ToolError> {
    if let Some(arr) = input.get("edits").and_then(Value::as_array) {
        if arr.is_empty() {
            return Err(ToolError::InvalidInput("`edits` is empty".into()));
        }
        return arr
            .iter()
            .map(|e| {
                let old = e.get("old_string").and_then(Value::as_str);
                let new = e.get("new_string").and_then(Value::as_str);
                match (old, new) {
                    (Some(o), Some(n)) => Ok((o.to_string(), n.to_string())),
                    _ => Err(ToolError::InvalidInput(
                        "each edit needs old_string and new_string".into(),
                    )),
                }
            })
            .collect();
    }
    let old = input.get("old_string").and_then(Value::as_str);
    let new = input.get("new_string").and_then(Value::as_str);
    match (old, new) {
        (Some(o), Some(n)) => Ok(vec![(o.to_string(), n.to_string())]),
        _ => Err(ToolError::InvalidInput(
            "provide `edits`, or `old_string` and `new_string`".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(contents: &str) -> tempfile::NamedTempFile {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        f
    }

    #[tokio::test]
    async fn replaces_unique_match() {
        let f = write_tmp("the quick brown fox");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "quick", "new_string": "slow" }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "the slow brown fox");
    }

    #[tokio::test]
    async fn rejects_ambiguous_match() {
        let f = write_tmp("a a a");
        let err = Edit
            .run(
                json!({ "path": f.path().to_str().unwrap(), "old_string": "a", "new_string": "b" }),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn replace_all_replaces_every_occurrence() {
        let f = write_tmp("a a a");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({ "path": p, "old_string": "a", "new_string": "b", "replace_all": true }))
            .await
            .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "b b b");
    }

    #[tokio::test]
    async fn applies_edits_array_in_order() {
        let f = write_tmp("foo and bar");
        let p = f.path().to_str().unwrap();
        Edit.run(json!({
            "path": p,
            "edits": [
                { "old_string": "foo", "new_string": "baz" },
                { "old_string": "bar", "new_string": "qux" }
            ]
        }))
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(p).unwrap(), "baz and qux");
    }

    #[tokio::test]
    async fn missing_string_is_invalid_input() {
        let f = write_tmp("hello");
        let err = Edit
            .run(json!({ "path": f.path().to_str().unwrap(), "old_string": "zzz", "new_string": "x" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }
}
