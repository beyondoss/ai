//! `ls` — list a directory's entries (directories suffixed with `/`).
//!
//! Two deliberate divergences from the reference agent, kept rather than "fixed" to match: dotfiles are
//! hidden by default (`all: true` opts back in — cuts real noise like `.git`/editor swapfiles without
//! losing access), and directories are always sorted before files (a stable UX improvement independent
//! of parity with anything else).

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

/// Default cap on entries returned before truncating, to protect the model's context from a
/// `node_modules`-sized directory. Overridable per call via the `limit` argument. The model can also
/// narrow with a more specific path or `find`/`grep`.
const DEFAULT_LIMIT: usize = 500;

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
                "all": { "type": "boolean", "description": "Include dot-files (default false)." },
                "limit": { "type": "integer", "description": "Max entries to list before truncating (default 500)." }
            }
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let path = input.get("path").and_then(Value::as_str).unwrap_or(".");
        let path = &super::normalize_path(path);
        let all = input.get("all").and_then(Value::as_bool).unwrap_or(false);
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let mut entries: Vec<String> = Vec::new();
        let dir =
            std::fs::read_dir(path).map_err(|e| ToolError::Execution(format!("ls {path}: {e}")))?;
        for entry in dir {
            let entry = entry.map_err(|e| ToolError::Execution(e.to_string()))?;
            let fname = entry.file_name();
            let name = fname.to_string_lossy();
            if !all && name.starts_with('.') {
                continue;
            }
            let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
            // Build the display name once, with room for the trailing `/`, so a directory entry doesn't
            // allocate a second String (the old `format!("{name}/")` dropped the first).
            let mut display = String::with_capacity(name.len() + usize::from(is_dir));
            display.push_str(&name);
            if is_dir {
                display.push('/');
            }
            entries.push(display);
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
        let count_truncated = total > limit;
        if count_truncated {
            entries.truncate(limit);
        }
        let mut out = entries.join("\n");
        if count_truncated {
            out.push('\n');
        }
        // The byte cap is checked *before* the count marker, and takes priority when both would
        // otherwise fire — see `cap_listing_bytes`'s doc comment.
        let byte_capped = super::output::cap_listing_bytes(
            &mut out,
            "narrow with a subpath or use `find`/`grep` to see more",
        );
        if !byte_capped && count_truncated {
            out.push_str(&super::output::marker(format_args!(
                "{} more entries; {total} total — narrow with a subpath or use `find`/`grep`",
                total - limit
            )));
        }
        Ok(out.into())
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
            .unwrap()
            .text;
        assert_eq!(out, "subdir/\nfile.txt");

        let all = Ls
            .run(json!({ "path": dir.path().to_str().unwrap(), "all": true }))
            .await
            .unwrap()
            .text;
        assert!(all.contains(".hidden"));
    }

    #[tokio::test]
    async fn run_normalizes_the_path_argument() {
        // Proves `run` actually calls `super::normalize_path`, via its `@`-prefix-strip behavior
        // (needs no `$HOME` mutation — see `expand_tilde`'s own direct unit tests for that half).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("file.txt"), "x").unwrap();
        let at_prefixed = format!("@{}", dir.path().to_str().unwrap());
        let out = Ls.run(json!({ "path": at_prefixed })).await.unwrap().text;
        assert!(out.contains("file.txt"));
    }

    #[tokio::test]
    async fn caps_a_huge_directory() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..(DEFAULT_LIMIT + 50) {
            std::fs::write(dir.path().join(format!("f{i:04}")), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("more entries"));
        // The body is capped to DEFAULT_LIMIT lines (plus the truncation note).
        assert_eq!(out.lines().count(), DEFAULT_LIMIT + 1);
    }

    #[tokio::test]
    async fn limit_param_overrides_the_default_cap() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(dir.path().join(format!("f{i:02}")), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap(), "limit": 3 }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("more entries"),
            "a small limit must truncate: {out}"
        );
        assert!(out.contains("[7 more entries; 10 total"));
        // Three entry lines plus the truncation note.
        assert_eq!(out.lines().count(), 4);
    }

    #[tokio::test]
    async fn output_byte_cap_truncates_even_under_the_entry_limit() {
        // 300 entries with long names: well under the 500-entry default `limit` (so the entry-count
        // marker never fires), but the aggregate listing still blows past the 50KB output cap on its
        // own.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..300 {
            let name = format!("{i:04}-{}", "x".repeat(200));
            std::fs::write(dir.path().join(name), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.len() <= super::super::output::MAX_LISTING_BYTES + 256,
            "output should be capped near MAX_LISTING_BYTES, got {} bytes",
            out.len()
        );
        assert!(
            out.contains("[output truncated at 50.0KB"),
            "byte-cap marker missing: {}",
            &out[out.len().saturating_sub(200)..]
        );
        assert!(
            !out.contains("more entries;"),
            "entry-count marker must not co-fire when count never exceeded the limit"
        );
    }

    #[tokio::test]
    async fn output_byte_cap_takes_priority_over_the_entry_count_marker() {
        // 600 long-named entries against the default 500-entry `limit`: both the entry-count marker
        // (600 > 500) and the byte cap would fire on the same rendered text. The byte cap must win
        // outright rather than leaving a marker sliced in half.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..600 {
            let name = format!("{i:04}-{}", "x".repeat(200));
            std::fs::write(dir.path().join(name), "x").unwrap();
        }
        let out = Ls
            .run(json!({ "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("[output truncated at 50.0KB"),
            "byte-cap marker missing: {out}"
        );
        assert!(
            !out.contains("more entries;"),
            "the byte-cap marker must win cleanly, not leave a mangled count marker behind"
        );
        assert!(
            out.len() <= super::super::output::MAX_LISTING_BYTES + 256,
            "output should be capped near MAX_LISTING_BYTES, got {} bytes",
            out.len()
        );
    }
}
