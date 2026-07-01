//! `find` — locate files by glob, gitignore-aware (ripgrep's `ignore` + `globset`).
//!
//! The walk is **sequential**: unlike `grep`, find's per-file work is a single glob test, so the
//! traversal — not per-file CPU — is the cost, and the benchmark showed `ignore`'s parallel walker
//! adds more thread-coordination overhead than it saves on realistic trees (it ran ~2× slower). What
//! we keep from the parallel design is the part that's a win regardless: results are collected and
//! sorted by path, so output is deterministic and `limit` truncation keeps the lexicographically-
//! smallest paths. A hard cap bounds memory; the blocking walk runs on `spawn_blocking`.

use std::fmt::Write as _;
use std::path::PathBuf;

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use ignore::WalkBuilder;
use serde_json::{Value, json};

use super::output::format_path;

/// Default cap on reported paths.
const DEFAULT_LIMIT: usize = 1000;
/// Hard ceiling on paths collected before the walk bails — an OOM guard for huge trees. Far above any
/// sane `limit`; when it trips, which paths survive truncation can depend on walk order (flagged).
const HARD_CAP: usize = 10_000;

pub struct Find;

/// A prepared find: compiled glob, whether it matches the basename only (vs. the full path), search
/// root, and the report cap.
pub struct FindJob {
    matcher: GlobMatcher,
    basename_only: bool,
    root: PathBuf,
    limit: usize,
}

impl FindJob {
    /// Assemble a job from an already-compiled glob matcher.
    pub fn new(matcher: GlobMatcher, basename_only: bool, root: PathBuf, limit: usize) -> Self {
        Self {
            matcher,
            basename_only,
            root,
            limit,
        }
    }
}

/// Walk the tree and collect matching paths, sorted lexicographically. Returns the paths and whether
/// the result was truncated (by `limit` or the hard cap).
pub fn search(job: &FindJob) -> (Vec<PathBuf>, bool) {
    let mut paths: Vec<PathBuf> = Vec::new();
    // `hidden(false)` includes dotfiles (like ripgrep --hidden); .gitignore is respected by default.
    // `require_git(false)` keeps that respect even outside an actual git repository (e.g. a plain
    // checkout with no `.git`, or a tree copied without its VCS metadata) — the ignore crate's default
    // otherwise silently stops honoring `.gitignore` the moment there's no repo to find.
    for entry in WalkBuilder::new(&job.root)
        .hidden(false)
        .require_git(false)
        .build()
    {
        if paths.len() >= HARD_CAP {
            break;
        }
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        // Skip the search root itself, but match both files *and* directories — `find "node_modules"`
        // or any directory-name pattern would otherwise return nothing.
        if path == job.root {
            continue;
        }
        let candidate = if job.basename_only {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            path.to_string_lossy().into_owned()
        };
        if job.matcher.is_match(candidate.as_str()) {
            paths.push(path.to_path_buf());
        }
    }

    paths.sort();
    let truncated = paths.len() > job.limit || paths.len() >= HARD_CAP;
    paths.truncate(job.limit);
    (paths, truncated)
}

#[async_trait]
impl Tool for Find {
    fn name(&self) -> &str {
        "find"
    }
    fn description(&self) -> &str {
        "Find files by glob pattern (e.g. \"*.rs\", \"src/**/*.test.ts\"), honoring .gitignore. A \
         pattern without \"/\" matches the file name; with \"/\" it matches the full path. Results are \
         sorted by path; when more than `limit` match, the lexicographically-smallest are reported."
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

    async fn run(&self, input: Value) -> Result<ToolOutput, ToolError> {
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

        let no_match = format!("no files matching {pattern:?}");
        let root = PathBuf::from(root);
        let job = FindJob::new(matcher, basename_only, root.clone(), limit);
        // The walk blocks (synchronous directory reads); keep it off the async runtime.
        let (paths, truncated) = tokio::task::spawn_blocking(move || search(&job))
            .await
            .map_err(|e| ToolError::Execution(format!("find task failed: {e}")))?;

        if paths.is_empty() {
            return Ok(no_match.into());
        }
        // Write straight into `out` instead of allocating a `format!` temp String per path — same fix
        // as `read`/`ls`/`grep`'s formatting loops. `writeln!` into a `String` can't fail, so the
        // `Result` is discarded. Paths are reported relative to the search root (matching pi), not the
        // full path straight from the walk entry — see `format_path`'s doc comment.
        let mut out = String::new();
        for path in &paths {
            let _ = writeln!(out, "{}", format_path(path, &root));
        }
        // The byte cap is checked *before* the result-count marker, and takes priority when both would
        // otherwise fire — see `cap_listing_bytes`'s doc comment.
        let byte_capped =
            super::output::cap_listing_bytes(&mut out, "narrow the pattern or path to see more");
        if !byte_capped && truncated {
            let _ = writeln!(
                out,
                "{}",
                super::output::marker(format_args!(
                    "result limit {limit} reached; raise `limit` for more"
                ))
            );
        }
        Ok(out.into())
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
            .unwrap()
            .text;
        assert!(out.contains("main.rs"));
        assert!(out.contains("lib.rs"));
        assert!(!out.contains("README.md"));
    }

    #[tokio::test]
    async fn output_paths_are_relative_to_the_search_root_not_absolute() {
        // Same fix as grep's: the prior behavior reported the full path straight from the walk entry
        // (often absolute), costing the model extra tokens for a prefix it already knows and diverging
        // from pi's documented "relative to search directory" contract.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "").unwrap();

        let out = Find
            .run(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim_end(), "src/main.rs", "got: {out}");
        assert!(
            !out.contains(dir.path().to_str().unwrap()),
            "the search root's own absolute prefix must not appear in the output: {out}"
        );
    }

    #[tokio::test]
    async fn matches_directories_too() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules")).unwrap();
        std::fs::write(dir.path().join("node_modules/pkg.json"), "").unwrap();
        let out = Find
            .run(json!({ "pattern": "node_modules", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("node_modules"),
            "directory-name searches must return the directory: {out}"
        );
    }

    #[tokio::test]
    async fn no_match_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        let out = Find
            .run(json!({ "pattern": "*.zzz", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("no files matching"));
    }

    #[tokio::test]
    async fn output_is_path_sorted_and_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["z.rs", "a.rs", "m.rs"] {
            std::fs::write(dir.path().join(name), "").unwrap();
        }
        let path = dir.path().to_str().unwrap();
        let run_once = || async {
            Find.run(json!({ "pattern": "*.rs", "path": path }))
                .await
                .unwrap()
                .text
        };
        let out = run_once().await;
        let order: Vec<&str> = out.lines().filter_map(|l| l.rsplit('/').next()).collect();
        assert_eq!(order, vec!["a.rs", "m.rs", "z.rs"]);
        assert_eq!(out, run_once().await);
    }

    #[tokio::test]
    async fn limit_truncates_to_smallest_paths() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(dir.path().join(format!("f{i:02}.rs")), "").unwrap();
        }
        let out = Find
            .run(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap(), "limit": 3 }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("[result limit 3 reached; raise `limit` for more]"));
        assert!(out.contains("f00.rs") && out.contains("f01.rs") && out.contains("f02.rs"));
        assert!(!out.contains("f03.rs"));
    }

    #[tokio::test]
    async fn gitignore_is_honored_even_outside_a_git_repository() {
        // `tempfile::tempdir()` lives under the system temp dir, not inside any git repository, so
        // this exercises `require_git(false)`: without it, the `ignore` crate's default behavior is
        // to stop respecting `.gitignore` files entirely once there's no `.git` ancestor to find.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.rs\n").unwrap();
        std::fs::write(dir.path().join("ignored.rs"), "").unwrap();
        std::fs::write(dir.path().join("kept.rs"), "").unwrap();
        let out = Find
            .run(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("kept.rs"), "got: {out}");
        assert!(
            !out.contains("ignored.rs"),
            ".gitignore should be honored even without a .git directory: {out}"
        );
    }

    #[tokio::test]
    async fn output_byte_cap_truncates_even_under_the_result_limit() {
        // 300 long-named matches: well under the 1000-result default `limit` (so the result-count
        // marker never fires), but the aggregate listing of paths still blows past the 50KB output
        // cap on its own.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..300 {
            let name = format!("{i:04}-{}.rs", "x".repeat(200));
            std::fs::write(dir.path().join(name), "").unwrap();
        }
        let out = Find
            .run(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap() }))
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
            !out.contains("result limit"),
            "result-count marker must not co-fire when count never exceeded the limit"
        );
    }

    #[tokio::test]
    async fn output_byte_cap_takes_priority_over_the_result_count_marker() {
        // 1200 long-named matches against the default 1000-result `limit`: both the result-count
        // marker (1200 > 1000) and the byte cap would fire on the same rendered text. The byte cap
        // must win outright rather than leaving a marker sliced in half.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..1200 {
            let name = format!("{i:04}-{}.rs", "x".repeat(200));
            std::fs::write(dir.path().join(name), "").unwrap();
        }
        let out = Find
            .run(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("[output truncated at 50.0KB"),
            "byte-cap marker missing: {out}"
        );
        assert!(
            !out.contains("result limit"),
            "the byte-cap marker must win cleanly, not leave a mangled count marker behind"
        );
        assert!(
            out.len() <= super::super::output::MAX_LISTING_BYTES + 256,
            "output should be capped near MAX_LISTING_BYTES, got {} bytes",
            out.len()
        );
    }
}
