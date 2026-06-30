//! `grep` — regex search across files, gitignore-aware (ripgrep's `ignore` + `regex` crates).
//!
//! The walk runs in parallel (ripgrep's `WalkParallel`): worker threads read and scan files
//! concurrently — the difference between grep and ripgrep on a large tree. Matches are collected,
//! sorted by `(path, line)`, then truncated to `limit`, so output is deterministic regardless of
//! thread scheduling and truncation keeps the lexicographically-smallest matches. A hard cap bounds
//! memory on pathological patterns; if it trips, which matches survive truncation may vary (flagged
//! in the output). The blocking walk runs on `spawn_blocking` so it never stalls the async runtime.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_core::ToolError;
use agent_core::tool::Tool;
use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use ignore::{WalkBuilder, WalkState};
use regex::{Regex, RegexBuilder};
use serde_json::{Value, json};

/// Default cap on reported matches.
const DEFAULT_LIMIT: usize = 100;
/// Long match lines are clipped to keep output readable.
const MAX_LINE: usize = 500;
/// Hard ceiling on matches collected before the walk bails — an OOM guard for pathological patterns
/// (matching nearly every line of a huge tree). Far above any sane `limit`; when it trips, the
/// surviving subset can depend on scheduling, which the output flags.
const HARD_CAP: usize = 10_000;

pub struct Grep;

/// A prepared grep: compiled regex, optional file glob, search root, and the report cap.
pub struct GrepJob {
    re: Regex,
    glob: Option<GlobMatcher>,
    root: PathBuf,
    limit: usize,
}

impl GrepJob {
    /// Assemble a job from already-compiled matchers.
    pub fn new(re: Regex, glob: Option<GlobMatcher>, root: PathBuf, limit: usize) -> Self {
        Self {
            re,
            glob,
            root,
            limit,
        }
    }
}

/// Run the parallel search. `threads == 0` lets `ignore` choose (≈ CPU count); the bench passes 1 to
/// measure the single-threaded baseline. Returns matches sorted by `(path, line)` and whether the
/// result was truncated (by `limit` or the hard cap).
pub fn search(job: &GrepJob, threads: usize) -> (Vec<(PathBuf, usize, String)>, bool) {
    let collected: Arc<Mutex<Vec<(PathBuf, usize, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let total = Arc::new(AtomicUsize::new(0));

    // `hidden(false)` includes dotfiles (like ripgrep --hidden); .gitignore is respected by default.
    WalkBuilder::new(&job.root)
        .hidden(false)
        .threads(threads)
        .build_parallel()
        .run(|| {
            let collected = Arc::clone(&collected);
            let total = Arc::clone(&total);
            let re = &job.re;
            let glob = &job.glob;
            Box::new(move |entry| {
                if total.load(Ordering::Relaxed) >= HARD_CAP {
                    return WalkState::Quit;
                }
                let Ok(entry) = entry else {
                    return WalkState::Continue;
                };
                if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    return WalkState::Continue;
                }
                let path = entry.path();
                if let Some(g) = glob {
                    if !g.is_match(path) {
                        return WalkState::Continue;
                    }
                }
                // Skip non-UTF8 / binary files silently.
                let Ok(content) = std::fs::read_to_string(path) else {
                    return WalkState::Continue;
                };
                // Scan the whole file into a local Vec, then push once — one lock per matching file,
                // not per matching line.
                let mut local = Vec::new();
                for (i, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        local.push((path.to_path_buf(), i + 1, clip(line)));
                    }
                }
                if !local.is_empty() {
                    let added = local.len();
                    if let Ok(mut guard) = collected.lock() {
                        guard.extend(local);
                    }
                    if total.fetch_add(added, Ordering::Relaxed) + added >= HARD_CAP {
                        return WalkState::Quit;
                    }
                }
                WalkState::Continue
            })
        });

    let mut matches = Arc::try_unwrap(collected)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .unwrap_or_default();
    matches.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let hard_cap_hit = total.load(Ordering::Relaxed) >= HARD_CAP;
    let truncated = matches.len() > job.limit || hard_cap_hit;
    matches.truncate(job.limit);
    (matches, truncated)
}

/// Clip a long line to `MAX_LINE` bytes at a UTF-8 char boundary (never panics mid-codepoint).
fn clip(line: &str) -> String {
    if line.len() <= MAX_LINE {
        return line.to_string();
    }
    let mut end = MAX_LINE;
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… [truncated]", &line[..end])
}

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str {
        "grep"
    }
    fn description(&self) -> &str {
        "Search file contents by regular expression, honoring .gitignore. Optionally restrict to a \
         `path`, a `glob` (e.g. \"*.rs\"), case-insensitive with `ignore_case`. Results are sorted by \
         path; when more than `limit` match, the lexicographically-smallest are reported."
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

        let no_match = format!("no matches for {pattern:?}");
        let job = GrepJob::new(re, glob, PathBuf::from(root), limit);
        // The walk blocks (its own thread pool, synchronous reads); keep it off the async runtime.
        let (matches, truncated) = tokio::task::spawn_blocking(move || search(&job, 0))
            .await
            .map_err(|e| ToolError::Execution(format!("grep task failed: {e}")))?;

        if matches.is_empty() {
            return Ok(no_match);
        }
        let mut out = String::new();
        for (path, line, text) in &matches {
            out.push_str(&format!("{}:{}: {}\n", path.display(), line, text));
        }
        if truncated {
            out.push_str(&format!(
                "… (match limit {limit} reached; narrow the pattern or raise `limit`)\n"
            ));
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

    #[tokio::test]
    async fn output_is_path_sorted_and_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        // Created out of order; the report must come back path-sorted regardless of walk/thread order.
        for name in ["z.txt", "a.txt", "m.txt"] {
            std::fs::write(dir.path().join(name), "needle\n").unwrap();
        }
        let path = dir.path().to_str().unwrap();
        let run_once = || async {
            Grep.run(json!({ "pattern": "needle", "path": path }))
                .await
                .unwrap()
        };
        let out = run_once().await;
        let order: Vec<&str> = out
            .lines()
            .filter_map(|l| l.split(':').next())
            .filter_map(|p| p.rsplit('/').next())
            .collect();
        assert_eq!(order, vec!["a.txt", "m.txt", "z.txt"]);
        // Stable across runs.
        assert_eq!(out, run_once().await);
    }

    #[tokio::test]
    async fn limit_truncates_to_smallest_paths() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(dir.path().join(format!("f{i:02}.txt")), "needle\n").unwrap();
        }
        let out = Grep
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap(), "limit": 3 }))
            .await
            .unwrap();
        assert!(out.contains("match limit 3 reached"));
        // The three lexicographically-smallest files, deterministically.
        assert!(out.contains("f00.txt") && out.contains("f01.txt") && out.contains("f02.txt"));
        assert!(!out.contains("f03.txt"));
    }
}
