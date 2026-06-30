//! `grep` — regex search across files, gitignore-aware (ripgrep's `ignore` + `regex` crates).
//!
//! The walk runs in parallel (ripgrep's `WalkParallel`): worker threads read and scan files
//! concurrently — the difference between grep and ripgrep on a large tree. Matches are collected,
//! sorted by `(path, line)`, then truncated to `limit`, so output is deterministic regardless of
//! thread scheduling and truncation keeps the lexicographically-smallest matches. A hard cap bounds
//! memory on pathological patterns; if it trips, which matches survive truncation may vary (flagged
//! in the output). The blocking walk runs on `spawn_blocking` so it never stalls the async runtime.

use std::collections::VecDeque;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
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
/// Ceiling on the `before`/`after` context window. Context lines aren't bounded by `HARD_CAP` (which
/// counts matches), so clamp them to keep a huge `after` on a match-dense file from ballooning memory.
const MAX_CONTEXT: usize = 100;

pub struct Grep;

/// One reported line: its path, line number, text, and whether it is a match (vs a context line).
type Hit = (PathBuf, usize, String, bool);

/// A prepared grep: compiled regex, optional file glob, search root, the report cap, and how many
/// lines of context to show around each match (`before`/`after`, like ripgrep's `-B`/`-A`/`-C`).
pub struct GrepJob {
    re: Regex,
    glob: Option<GlobMatcher>,
    root: PathBuf,
    limit: usize,
    before: usize,
    after: usize,
}

impl GrepJob {
    /// Assemble a job from already-compiled matchers, with no surrounding context.
    pub fn new(re: Regex, glob: Option<GlobMatcher>, root: PathBuf, limit: usize) -> Self {
        Self {
            re,
            glob,
            root,
            limit,
            before: 0,
            after: 0,
        }
    }

    /// Show `before` lines before and `after` lines after each match (clamped to [`MAX_CONTEXT`]).
    pub fn with_context(mut self, before: usize, after: usize) -> Self {
        self.before = before.min(MAX_CONTEXT);
        self.after = after.min(MAX_CONTEXT);
        self
    }
}

/// Run the parallel search. `threads == 0` lets `ignore` choose (≈ CPU count); the bench passes 1 to
/// measure the single-threaded baseline. Returns hits sorted by `(path, line)` — each flagged as a
/// match or a context line — and whether the result was truncated (by `limit` or the hard cap).
/// `limit` and the hard cap count *matches*, not context lines.
pub fn search(job: &GrepJob, threads: usize) -> (Vec<Hit>, bool) {
    let collected: Arc<Mutex<Vec<Hit>>> = Arc::new(Mutex::new(Vec::new()));
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
            let before = job.before;
            let after = job.after;
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
                // Stream the file line-by-line rather than slurping it whole: a large committed file
                // (fixtures, lockfiles, vendored sources) shouldn't be fully buffered, and the walk
                // runs ≈CPU-count of these workers at once. Hold at most one line plus the matches.
                let Ok(file) = std::fs::File::open(path) else {
                    return WalkState::Continue;
                };
                let mut reader = std::io::BufReader::new(file);
                // Scan into a local Vec, then push once — one lock per matching file, not per line.
                // Context tracking: `ring` holds the last `before` not-yet-emitted lines (candidate
                // before-context for the next match); `after_remaining` counts down the after-context
                // owed to the most recent match. Every line is emitted at most once, so windows never
                // duplicate a line shared by two nearby matches.
                let mut local: Vec<Hit> = Vec::new();
                let mut ring: VecDeque<(usize, String)> = VecDeque::new();
                let mut after_remaining = 0usize;
                let mut match_count = 0usize;
                let mut lineno = 0usize;
                let mut buf = String::new();
                loop {
                    buf.clear();
                    // A read error mid-file (invalid UTF-8 / binary) skips the whole file, dropping
                    // any matches gathered so far — same outcome as the prior whole-file read.
                    match reader.read_line(&mut buf) {
                        Ok(0) => break, // EOF
                        Ok(_) => {}
                        Err(_) => return WalkState::Continue,
                    }
                    lineno += 1;
                    let line = buf.strip_suffix('\n').unwrap_or(&buf);
                    let line = line.strip_suffix('\r').unwrap_or(line);
                    if re.is_match(line) {
                        // Flush the buffered before-context, then the match itself.
                        for (ln, txt) in ring.drain(..) {
                            local.push((path.to_path_buf(), ln, txt, false));
                        }
                        local.push((path.to_path_buf(), lineno, clip(line), true));
                        match_count += 1;
                        after_remaining = after;
                    } else if after_remaining > 0 {
                        // After-context owed to the preceding match.
                        local.push((path.to_path_buf(), lineno, clip(line), false));
                        after_remaining -= 1;
                    } else if before > 0 {
                        // A candidate before-context line for a later match — keep only the last `before`.
                        if ring.len() == before {
                            ring.pop_front();
                        }
                        ring.push_back((lineno, clip(line)));
                    }
                }
                if !local.is_empty() {
                    if let Ok(mut guard) = collected.lock() {
                        guard.extend(local);
                    }
                    // Only matches count toward the hard cap (context is bounded per match).
                    if total.fetch_add(match_count, Ordering::Relaxed) + match_count >= HARD_CAP {
                        return WalkState::Quit;
                    }
                }
                WalkState::Continue
            })
        });

    let mut hits = Arc::try_unwrap(collected)
        .ok()
        .and_then(|m| m.into_inner().ok())
        .unwrap_or_default();
    hits.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let hard_cap_hit = total.load(Ordering::Relaxed) >= HARD_CAP;
    let match_total = hits.iter().filter(|h| h.3).count();
    let truncated = match_total > job.limit || hard_cap_hit;
    // Keep at most `limit` matched lines (context lines don't count toward it). Walk counting matches,
    // cut at the match that would exceed the limit, then drop any now-dangling trailing context (the
    // before-context of the dropped match that trailed into the kept prefix). With no context
    // requested every hit is a match, so this reduces to a plain `truncate(limit)`.
    if match_total > job.limit {
        let mut seen = 0usize;
        let mut cut = hits.len();
        for (i, h) in hits.iter().enumerate() {
            if h.3 {
                seen += 1;
                if seen > job.limit {
                    cut = i;
                    break;
                }
            }
        }
        hits.truncate(cut);
        while hits.last().is_some_and(|h| !h.3) {
            hits.pop();
        }
    }
    (hits, truncated)
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
                "literal": { "type": "boolean", "description": "Treat `pattern` as a literal string, not a regex (default false)." },
                "limit": { "type": "integer", "description": "Max matches to report (default 100)." },
                "context": { "type": "integer", "description": "Lines of context to show before and after each match (like ripgrep -C). Default 0." },
                "before": { "type": "integer", "description": "Lines of context before each match (overrides `context`)." },
                "after": { "type": "integer", "description": "Lines of context after each match (overrides `context`)." }
            },
            "required": ["pattern"]
        })
    }

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
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

        // Context lines around each match. `context` sets both sides; `before`/`after` override it per
        // side. Absent everywhere → 0 (the prior behavior, just the matching lines).
        let usize_arg = |key: &str| input.get(key).and_then(Value::as_u64).map(|n| n as usize);
        let context = usize_arg("context");
        let before = usize_arg("before").or(context).unwrap_or(0);
        let after = usize_arg("after").or(context).unwrap_or(0);

        // `literal` searches for the pattern verbatim (regex-escaped), so a model can grep for
        // `a.b(c)` or `Vec<T>` without escaping the regex metacharacters itself.
        let literal = input
            .get("literal")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let escaped;
        let effective = if literal {
            escaped = regex::escape(pattern);
            escaped.as_str()
        } else {
            pattern
        };
        let re = RegexBuilder::new(effective)
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
        let job = GrepJob::new(re, glob, PathBuf::from(root), limit).with_context(before, after);
        // The walk blocks (its own thread pool, synchronous reads); keep it off the async runtime.
        let (matches, truncated) = tokio::task::spawn_blocking(move || search(&job, 0))
            .await
            .map_err(|e| ToolError::Execution(format!("grep task failed: {e}")))?;

        if matches.is_empty() {
            return Ok(no_match.into());
        }
        let mut out = String::new();
        for (path, line, text, is_match) in &matches {
            // Match lines use `path:line:` (ripgrep's separator); context lines use `path-line-` so a
            // reader (and the model) can tell a hit from its surrounding context at a glance.
            if *is_match {
                out.push_str(&format!("{}:{}: {}\n", path.display(), line, text));
            } else {
                out.push_str(&format!("{}-{}- {}\n", path.display(), line, text));
            }
        }
        if truncated {
            out.push_str(&format!(
                "… (match limit {limit} reached; narrow the pattern or raise `limit`)\n"
            ));
        }
        Ok(out.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn literal_mode_searches_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "call a.b(c) here\nand axbc too\n").unwrap();
        // As a regex `a.b(c)` matches `axbc`; literal mode must match only the verbatim text.
        let out = Grep
            .run(json!({
                "pattern": "a.b(c)",
                "literal": true,
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("call a.b(c)"));
        assert!(
            !out.contains("axbc"),
            "literal mode must not regex-match: {out}"
        );
    }

    #[tokio::test]
    async fn finds_matches_with_path_and_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\nworld\nhello again\n").unwrap();
        std::fs::write(dir.path().join("b.log"), "nothing here\n").unwrap();

        let out = Grep
            .run(json!({ "pattern": "hello", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
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
            .unwrap()
            .text;
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
            .unwrap()
            .text;
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
                .text
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
    async fn context_lines_surround_each_match() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "one\ntwo\nNEEDLE\nfour\nfive\n").unwrap();
        let path = dir.path().to_str().unwrap();

        // `context: 1` shows one line on each side. The match keeps the `:` separator; context lines
        // use the `-` separator.
        let out = Grep
            .run(json!({ "pattern": "NEEDLE", "path": path, "context": 1 }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("a.txt:3: NEEDLE"), "match line: {out}");
        assert!(out.contains("a.txt-2- two"), "before-context: {out}");
        assert!(out.contains("a.txt-4- four"), "after-context: {out}");
        assert!(
            !out.contains("one"),
            "context window must not reach line 1: {out}"
        );
        assert!(
            !out.contains("five"),
            "context window must not reach line 5: {out}"
        );

        // `before`/`after` set each side independently (here: only after).
        let out = Grep
            .run(json!({ "pattern": "NEEDLE", "path": path, "before": 0, "after": 2 }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("a.txt:3: NEEDLE"));
        assert!(
            !out.contains("two"),
            "before=0 must show no preceding context: {out}"
        );
        assert!(out.contains("a.txt-4- four") && out.contains("a.txt-5- five"));
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
            .unwrap()
            .text;
        assert!(out.contains("match limit 3 reached"));
        // The three lexicographically-smallest files, deterministically.
        assert!(out.contains("f00.txt") && out.contains("f01.txt") && out.contains("f02.txt"));
        assert!(!out.contains("f03.txt"));
    }
}
