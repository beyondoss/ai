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
use globset::{GlobBuilder, GlobMatcher};
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

/// Walk the tree and collect matching paths, sorted lexicographically. Returns the paths, whether the
/// result was truncated (by `limit` or the hard cap), and, when the walk hit at least one unreadable
/// path (permission denied, a broken symlink, …), the first such error's message — matching real `fd`,
/// which exits non-zero and reports the real error text the moment it can't read *any* path in the
/// tree, rather than silently treating it as "not found here."
pub fn search(job: &FindJob) -> (Vec<PathBuf>, bool, Option<String>) {
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut first_error: Option<String> = None;
    // `hidden(false)` includes dotfiles (like ripgrep --hidden); .gitignore is respected by default.
    // `require_git(false)` keeps that respect even outside an actual git repository (e.g. a plain
    // checkout with no `.git`, or a tree copied without its VCS metadata) — the ignore crate's default
    // otherwise silently stops honoring `.gitignore` the moment there's no repo to find. Only applied
    // outside a real repo, though — matching pi's own `fd` invocation (`--no-require-git` only when the
    // root isn't inside a git repo): inside one, the default git-aware walk keeps a nested repo's own
    // `.gitignore` from leaking parent rules across its boundary.
    let mut builder = WalkBuilder::new(&job.root);
    builder.hidden(false);
    if !super::root_is_inside_git_repo(&job.root) {
        builder.require_git(false);
    }
    for entry in builder.build() {
        if paths.len() >= HARD_CAP {
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            // Recorded (not just skipped, the prior behavior), so an otherwise-empty result can say
            // "couldn't fully search" instead of a confidently wrong "no files matching".
            Err(e) => {
                first_error.get_or_insert_with(|| e.to_string());
                continue;
            }
        };
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
    (paths, truncated, first_error)
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
        let root = &super::normalize_path(root);
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
        // Smart-case, matching pi's own `find` (which never passes `--ignore-case`/`-s`, so `fd`'s own
        // smart-case default applies): a pattern with no uppercase letter matches case-insensitively
        // (an everyday `*.md` still finds `README.MD`), one with any uppercase letter matches exactly
        // as typed. Without this, `find` was always case-sensitive outright — `*.rs` never matched
        // `File.RS` — silently missing files pi's own `find` would return.
        let case_insensitive = !pattern.chars().any(|c| c.is_uppercase());
        let matcher = GlobBuilder::new(&glob_src)
            .case_insensitive(case_insensitive)
            .build()
            .map_err(|e| ToolError::InvalidInput(format!("bad glob: {e}")))?
            .compile_matcher();

        let no_match = format!("no files matching {pattern:?}");
        let root = PathBuf::from(root);
        let job = FindJob::new(matcher, basename_only, root.clone(), limit);
        // The walk blocks (synchronous directory reads); keep it off the async runtime.
        let (paths, truncated, walk_error) = tokio::task::spawn_blocking(move || search(&job))
            .await
            .map_err(|e| ToolError::Execution(format!("find task failed: {e}")))?;

        if paths.is_empty() {
            // A genuine zero-match walk and a walk that hit an unreadable path along the way both
            // land here with an empty result — but they aren't the same thing, and reporting them
            // identically ("no files matching") tells the model a confident falsehood in the second
            // case. Matches real `fd`, which exits non-zero and reports the real error the moment it
            // can't read *any* path in the tree.
            if let Some(err) = walk_error {
                return Err(ToolError::Execution(format!(
                    "search was incomplete, so \"no files matching\" may not be accurate: {err}"
                )));
            }
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
        if !byte_capped {
            let mut notices = Vec::new();
            if truncated {
                notices.push(format!(
                    "result limit {limit} reached; raise `limit` for more"
                ));
            }
            if let Some(err) = walk_error {
                notices.push(format!(
                    "search was incomplete — some paths couldn't be read, so there may be more \
                     matches than shown: {err}"
                ));
            }
            if !notices.is_empty() {
                let _ = writeln!(out, "{}", super::output::marker(notices.join(". ")));
            }
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
    async fn a_lowercase_pattern_matches_case_insensitively_smart_case() {
        // Pi-parity audit H70: pi's own `find` never passes `--ignore-case`/`-s`, so `fd`'s own
        // smart-case default applies (verified against real `fd`: `--glob '*.rs'` matches `File.RS`).
        // This crate's `find` was always case-sensitive outright — an everyday lowercase pattern like
        // `*.md` silently missed mixed/upper-case-named files pi's own `find` would return.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("File.RS"), "").unwrap();
        std::fs::write(dir.path().join("lower.rs"), "").unwrap();
        std::fs::write(dir.path().join("Mixed.Rs"), "").unwrap();

        let out = Find
            .run(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("File.RS"), "got: {out}");
        assert!(out.contains("lower.rs"), "got: {out}");
        assert!(out.contains("Mixed.Rs"), "got: {out}");
    }

    #[tokio::test]
    async fn a_pattern_containing_an_uppercase_letter_matches_case_sensitively() {
        // Smart-case's other half: the moment a pattern has *any* uppercase letter, it must match
        // exactly as typed — verified against real `fd`: `--glob '*.RS'` does not match `lower.rs`.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("File.RS"), "").unwrap();
        std::fs::write(dir.path().join("lower.rs"), "").unwrap();

        let out = Find
            .run(json!({ "pattern": "*.RS", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("File.RS"), "got: {out}");
        assert!(
            !out.contains("lower.rs"),
            "an uppercase-containing pattern must not match a lowercase file: {out}"
        );
    }

    #[tokio::test]
    async fn run_normalizes_the_path_argument() {
        // Proves `run` actually calls `super::normalize_path`, via its `@`-prefix-strip behavior
        // (needs no `$HOME` mutation — see `expand_tilde`'s own direct unit tests for that half).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("main.rs"), "").unwrap();
        let at_prefixed = format!("@{}", dir.path().to_str().unwrap());
        let out = Find
            .run(json!({ "pattern": "*.rs", "path": at_prefixed }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("main.rs"));
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
    async fn a_malformed_glob_pattern_is_a_clear_invalid_input_error() {
        // pi: tools.test.ts, "should surface fd glob parse errors" — coverage exists for the happy
        // path only; an unterminated character class must produce a clear tool error, not a panic or a
        // confusing "no files matching" result.
        let dir = tempfile::tempdir().unwrap();
        let err = Find
            .run(json!({ "pattern": "[", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidInput(msg) => assert!(msg.contains("bad glob"), "got: {msg}"),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
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

    /// Skip a permission-bit test under a runtime that doesn't actually enforce them (root, some
    /// sandboxes) — matches the same guard `grep.rs`'s identical tests use.
    #[cfg(unix)]
    fn mode_actually_blocks_reads(path: &std::path::Path) -> bool {
        std::fs::read_dir(path).is_err()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_search_root_reports_an_error_not_a_false_no_matches() {
        // Pi-parity audit H69: real `fd` exits non-zero the moment it can't read a path in the tree,
        // root included — this crate's own walk previously swallowed that into a skipped entry, so an
        // unreadable root silently reported "no files matching" instead of surfacing the real problem.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "").unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o000)).unwrap();
        let blocks = mode_actually_blocks_reads(dir.path());

        let result = Find
            .run(json!({ "pattern": "*.txt", "path": dir.path().to_str().unwrap() }))
            .await;
        let _ = std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755));

        if !blocks {
            return;
        }
        let err =
            result.expect_err("an unreadable root must surface as an error, not empty output");
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error, got {err:?}")
        };
        assert!(
            msg.contains("incomplete") && msg.contains("Permission denied"),
            "must surface the real reason, not a bare confident-empty result: {msg}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_subdirectory_alongside_real_matches_still_returns_them_with_a_notice() {
        // A real match elsewhere in the tree must not be discarded just because some other, unrelated
        // path couldn't be read — matches `grep.rs`'s identical, deliberate divergence from pi's own
        // all-or-nothing behavior.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("readable.rs"), "").unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::write(locked.join("secret.rs"), "").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let blocks = mode_actually_blocks_reads(&locked);

        let result = Find
            .run(json!({ "pattern": "*.rs", "path": dir.path().to_str().unwrap() }))
            .await;
        let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755));

        if !blocks {
            return;
        }
        let out = result.unwrap().text;
        assert!(
            out.contains("readable.rs"),
            "a real match elsewhere must still be reported: {out}"
        );
        assert!(
            out.contains("incomplete"),
            "the result must note it may be incomplete: {out}"
        );
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
    async fn gitignore_is_still_honored_inside_a_real_git_repository() {
        // pi-parity fix (L7): `require_git(false)` is now conditional on the search root *not* being
        // inside a real git repo (matching pi's own `fd --no-require-git` scoping) — this guards
        // against the conditional accidentally breaking the far more common case, a search root that
        // *is* inside a real repo, which must still honor `.gitignore` exactly as before.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
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
            ".gitignore must still be honored inside a real git repository: {out}"
        );
    }

    #[tokio::test]
    async fn dotfiles_are_included_by_default() {
        // pi: tools.test.ts, "should include hidden files that are not gitignored". `hidden(false)`
        // (see `search`'s doc comment, just above) is the only thing standing between this and a
        // dotfile silently vanishing from every `find` result by default, the way it would under
        // `ignore`'s (and ripgrep's) own default — pin it with a real dotfile, not just the doc comment.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".secret")).unwrap();
        std::fs::write(dir.path().join(".secret/hidden.txt"), "hidden").unwrap();
        std::fs::write(dir.path().join("visible.txt"), "visible").unwrap();

        let out = Find
            .run(json!({ "pattern": "**/*.txt", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        let lines: Vec<&str> = out
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        assert!(lines.contains(&"visible.txt"), "got: {out}");
        assert!(lines.contains(&".secret/hidden.txt"), "got: {out}");
    }

    #[tokio::test]
    async fn regression_3303_a_nested_gitignore_only_scopes_its_own_subtree_flat_siblings() {
        // Regression for pi's #3303: the prior fd-backed implementation collected every `.gitignore`
        // under the search path and passed them to fd via a single `--ignore-file` list, which fd
        // treats as one *global* ignore source — so `a/.gitignore`'s rules leaked into sibling `b/`.
        // Ours never collected `.gitignore` files by hand at all (`WalkBuilder` walks and applies each
        // one in its own scope natively, see `search`'s doc comment on `require_git(false)`), so this
        // is expected to already hold; proving it, not fixing a suspected bug here.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("a")).unwrap();
        std::fs::create_dir_all(dir.path().join("b")).unwrap();
        std::fs::write(dir.path().join("a/.gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("a/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("b/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("b/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("root.txt"), "").unwrap();

        let out = Find
            .run(json!({ "pattern": "**/*.txt", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        let mut lines: Vec<&str> = out
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        lines.sort_unstable();
        assert_eq!(
            lines,
            vec!["a/kept.txt", "b/ignored.txt", "b/kept.txt", "root.txt"],
            "got: {out}"
        );
    }

    #[tokio::test]
    async fn regression_3303_each_nested_gitignore_scopes_only_its_own_subtree() {
        // The deeper case from the same pi regression: `a/.gitignore` ignores `ignored.txt` within
        // both `a/` and `a/deep/` (rules apply to the whole subtree rooted where the file lives), while
        // `a/deep/.gitignore`'s *additional* `secret.txt` rule applies only within `a/deep/` — and `b/`
        // is untouched by either.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("a/deep")).unwrap();
        std::fs::create_dir_all(dir.path().join("b")).unwrap();
        std::fs::write(dir.path().join("a/.gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("a/deep/.gitignore"), "secret.txt\n").unwrap();
        std::fs::write(dir.path().join("a/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/deep/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/deep/secret.txt"), "").unwrap();
        std::fs::write(dir.path().join("a/deep/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("b/ignored.txt"), "").unwrap();
        std::fs::write(dir.path().join("b/kept.txt"), "").unwrap();
        std::fs::write(dir.path().join("root.txt"), "").unwrap();

        let out = Find
            .run(json!({ "pattern": "**/*.txt", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        let mut lines: Vec<&str> = out
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        lines.sort_unstable();
        assert_eq!(
            lines,
            vec![
                "a/deep/kept.txt",
                "a/kept.txt",
                "b/ignored.txt",
                "b/kept.txt",
                "root.txt"
            ],
            "got: {out}"
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

    /// The tree pi's own regression fixture (`3302-find-path-glob.test.ts`) builds: a `some/parent/
    /// child/` subtree with two files, and a separate `src/foo/bar/` subtree with one `.spec.ts` file.
    fn regression_3302_tree() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("some/parent/child")).unwrap();
        std::fs::create_dir_all(dir.path().join("src/foo/bar")).unwrap();
        std::fs::write(dir.path().join("some/parent/child/file.ext"), "").unwrap();
        std::fs::write(dir.path().join("some/parent/child/test.spec.ts"), "").unwrap();
        std::fs::write(dir.path().join("src/foo/bar/example.spec.ts"), "").unwrap();
        dir
    }

    // pi issue #3302: `find` advertises path-glob patterns like `src/**/*.spec.ts`, but the original
    // fd-backed implementation matched only against the basename whenever the pattern contained a `/`,
    // so every one of the four scenarios below silently returned no matches. Ours never shelled out to
    // fd (see the module doc comment), but the same anchoring bug class — a `/`-containing pattern not
    // actually being matched against the full path — is exactly what `basename_only`/the `**/` prefix
    // in `Find::run` (just above `search`) exists to avoid; this pins pi's own regression cases against
    // it directly, not just against the general "matches nested paths" happy path already covered above.

    #[tokio::test]
    async fn regression_3302_basename_pattern_still_matches() {
        // "regression-safe": a `/`-free pattern must keep matching by basename exactly as before, even
        // once path-glob support (below) is layered on top of it.
        let dir = regression_3302_tree();
        let out = Find
            .run(json!({ "pattern": "*.spec.ts", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("some/parent/child/test.spec.ts"), "got: {out}");
        assert!(out.contains("src/foo/bar/example.spec.ts"), "got: {out}");
    }

    #[tokio::test]
    async fn regression_3302_directory_prefixed_pattern_with_trailing_double_star_matches_the_subtree()
     {
        let dir = regression_3302_tree();
        let out = Find
            .run(json!({ "pattern": "some/parent/child/**", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("some/parent/child/file.ext"), "got: {out}");
        assert!(out.contains("some/parent/child/test.spec.ts"), "got: {out}");
    }

    #[tokio::test]
    async fn regression_3302_leading_double_star_wildcard_with_path_segments_matches() {
        let dir = regression_3302_tree();
        let out = Find
            .run(json!({ "pattern": "**/parent/child/*", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("some/parent/child/file.ext"), "got: {out}");
        assert!(out.contains("some/parent/child/test.spec.ts"), "got: {out}");
    }

    #[tokio::test]
    async fn regression_3302_nested_src_glob_matches_the_nested_spec_file() {
        let dir = regression_3302_tree();
        let out = Find
            .run(json!({ "pattern": "src/**/*.spec.ts", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim_end(), "src/foo/bar/example.spec.ts", "got: {out}");
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
