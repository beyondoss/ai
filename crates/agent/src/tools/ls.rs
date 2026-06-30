//! `ls` — list a directory's entries (directories suffixed with `/`).

use agent_core::ToolError;
use agent_core::tool::Tool;
use async_trait::async_trait;
use serde_json::{Value, json};

/// Max entries returned before truncating, to protect the model's context from a `node_modules`-sized
/// directory. The model can narrow with a more specific path or `find`/`grep`.
const MAX_ENTRIES: usize = 500;

pub struct Ls;

#[async_trait]
impl Tool for Ls {
    fn name(&self) -> &str {
        "ls"
    }
    fn description(&self) -> &str {
        "List the entries of a directory. Directories are suffixed with `/`. Hidden entries are \
         shown only when `all` is true."
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to list (default \".\")." },
                "all": { "type": "boolean", "description": "Include dot-files (default false)." }
            }
        })
    }

    async fn run(&self, input: Value) -> Result<String, ToolError> {
        let path = input.get("path").and_then(Value::as_str).unwrap_or(".");
        let all = input.get("all").and_then(Value::as_bool).unwrap_or(false);

        let mut entries: Vec<String> = Vec::new();
        let dir =
            std::fs::read_dir(path).map_err(|e| ToolError::Execution(format!("ls {path}: {e}")))?;
        for entry in dir {
            let entry = entry.map_err(|e| ToolError::Execution(e.to_string()))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !all && name.starts_with('.') {
                continue;
            }
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            entries.push(if is_dir { format!("{name}/") } else { name });
        }
        // Directories first, then alphabetical — stable, predictable output for the model.
        entries.sort_by(|a, b| {
            let (ad, bd) = (a.ends_with('/'), b.ends_with('/'));
            bd.cmp(&ad).then_with(|| a.cmp(b))
        });
        if entries.is_empty() {
            return Ok("(empty directory)".into());
        }
        // Cap the listing so a huge directory can't flood the model's context.
        let total = entries.len();
        if total > MAX_ENTRIES {
            entries.truncate(MAX_ENTRIES);
            let mut out = entries.join("\n");
            out.push_str(&format!(
                "\n… ({} more entries; {total} total — narrow with a subpath or use `find`/`grep`)",
                total - MAX_ENTRIES
            ));
            return Ok(out);
        }
        Ok(entries.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn lists_dirs_first_and_hides_dotfiles() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        std::fs::write(dir.path().join("file.txt"), "x").unwrap();
        std::fs::write(dir.path().join(".hidden"), "x").unwrap();

        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap();
        assert_eq!(out, "subdir/\nfile.txt");

        let all = Ls
            .run(json!({ "path": dir.path().to_str().unwrap(), "all": true }))
            .await
            .unwrap();
        assert!(all.contains(".hidden"));
    }

    #[tokio::test]
    async fn caps_a_huge_directory() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..(MAX_ENTRIES + 50) {
            std::fs::write(dir.path().join(format!("f{i:04}")), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap();
        assert!(out.contains("more entries"));
        // The body is capped to MAX_ENTRIES lines (plus the truncation note).
        assert_eq!(out.lines().count(), MAX_ENTRIES + 1);
    }
}
