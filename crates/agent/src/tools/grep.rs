//! `grep` — regex search across files, gitignore-aware.
//!
//! This module is **parse and render only**. Building the query from the model's JSON and formatting
//! the hits back into `path:line: text` lives here; actually walking a tree and matching lines lives
//! behind [`FsBackend::search`](super::fs::FsBackend::search), because that is the one part of this
//! tool that has to change when the files aren't on the host.
//!
//! The split is drawn where it is on purpose: everything a reader would call "grep's behavior" —
//! the `!`-negated glob, the `limit` floor, `context`/`before`/`after`, the `path:line:` vs
//! `path-line-` distinction, the truncation notices — is on this side of the seam and therefore
//! cannot vary by backend. See [`super::fs`] for why line clipping and limit trimming are shared too.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

use agent_core::tool::Tool;
use agent_core::{ToolError, ToolOutput};
use async_trait::async_trait;
use serde_json::{Value, json};

use super::fs::local::LocalFs;
use super::fs::{FsBackend, LINE_TRUNCATED_SUFFIX, MAX_CONTEXT, MAX_LINE, SearchQuery};
use super::output::format_path;

/// Default cap on reported matches.
const DEFAULT_LIMIT: usize = 100;

pub struct Grep {
    /// A relative search `path` (including its `"."` default) resolves against this. Empty = the
    /// process cwd. See [`super::resolve_against`].
    root: PathBuf,
    /// Where the search actually runs. Defaults to the host filesystem, so every existing caller and
    /// test gets exactly the behavior it had before this seam existed.
    backend: Arc<dyn FsBackend>,
}

impl Default for Grep {
    fn default() -> Self {
        Self::new(PathBuf::new())
    }
}

impl Grep {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            backend: Arc::new(LocalFs::new()),
        }
    }

    /// Search somewhere other than the host filesystem. The `Arc` is shared rather than owned because
    /// one backend serves every tool bound to the same target, and the tool registry is rebuilt on
    /// every model switch — a backend that reconnected per rebuild would be a bug, not a cost.
    pub fn with_backend(mut self, backend: Arc<dyn FsBackend>) -> Self {
        self.backend = backend;
        self
    }
}

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str {
        "grep"
    }
    fn description(&self) -> &str {
        // Pi-parity fix (task 52): built via `format!` referencing the real constants — like
        // `bash.rs`'s `describe()` — instead of a hand-typed literal safety-netted only by a unit test
        // that has to be kept in sync manually. Unlike `bash`'s, this description depends on no
        // per-instance config (only on compile-time constants), so it's rendered once into a `OnceLock`
        // rather than stored on every `Grep` the way `Bash` stores its timeout-dependent one.
        static DESC: OnceLock<String> = OnceLock::new();
        DESC.get_or_init(|| {
            format!(
                "Search file contents by regular expression, honoring .gitignore. Optionally restrict \
                 to a `path`, a `glob` (e.g. \"*.rs\"), case-insensitive with `ignore_case`. Results \
                 are sorted by path and truncated to at most {DEFAULT_LIMIT} matches or {}, whichever \
                 is hit first, with individual lines truncated to {MAX_LINE} characters; when \
                 truncated, the lexicographically-smallest matches are reported.",
                super::output::format_size(super::output::MAX_LISTING_BYTES as u64)
            )
        })
    }
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regular expression to search for." },
                "path": { "type": "string", "description": "Directory or file to search (default \".\")." },
                "glob": { "type": "string", "description": "Only search files matching this glob. Prefix with \"!\" to exclude files matching it instead." },
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
        let root = &super::resolve_against(&self.root, root);
        let ignore_case = input
            .get("ignore_case")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        // Floored at 1 — matches pi's own `grep.ts` (`Math.max(1, limit ?? DEFAULT_LIMIT)`). Without
        // this, `limit: 0` makes `stop_at` (in `search`) zero, so the walk's very first "have we hit
        // the threshold?" check is already true before a single file is scanned — a confidently wrong
        // "no matches" even when real matches exist. `find`'s own `limit` deliberately has no such
        // floor (pi's `find.ts` doesn't apply one either — a genuine asymmetry between the two tools
        // upstream, not an oversight to "fix" into false symmetry here).
        let limit = input
            .get("limit")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LIMIT)
            .max(1);

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
        // A leading "!" negates the glob (ripgrep-CLI-style: `rg --glob '!*.test.rs'` excludes rather
        // than restricts). The negation is stripped *here* rather than in a backend, so every backend
        // receives the same already-decided `(pattern, negate)` pair and none of them can disagree
        // about what a leading "!" means. `globset::Glob` has no negation syntax of its own — a
        // leading "!" would otherwise be matched literally, so `!*.test.rs` would confidently match
        // nothing at all and the whole query would report zero results instead of applying the
        // exclusion.
        let glob = input
            .get("glob")
            .and_then(Value::as_str)
            .map(|g| match g.strip_prefix('!') {
                Some(rest) => (rest.to_string(), true),
                None => (g.to_string(), false),
            });

        let no_match = format!("no matches for {pattern:?}");
        let root = PathBuf::from(root);
        let query = SearchQuery {
            pattern: pattern.to_string(),
            literal,
            ignore_case,
            glob,
            root: root.clone(),
            limit,
            // Clamped here so the query handed to any backend is already valid, rather than each
            // backend being trusted to clamp a window it didn't choose.
            before: before.min(MAX_CONTEXT),
            after: after.min(MAX_CONTEXT),
        };
        // A malformed regex or glob surfaces from the backend now (it compiles them), but as the same
        // `ToolError::InvalidInput` with the same message text it always produced.
        let outcome = self.backend.search(&query).await?;
        let (matches, truncated, walk_error) =
            (outcome.hits, outcome.truncated, outcome.first_error);

        if matches.is_empty() {
            // A genuine zero-match walk and a walk that hit an unreadable path along the way both
            // land here with an empty result — but they aren't the same thing, and reporting them
            // identically ("no matches") tells the model a confident falsehood in the second case.
            // Matches real ripgrep, which exits non-zero and reports the real error the moment it
            // can't read *any* path in the tree.
            if let Some(err) = walk_error {
                return Err(ToolError::Execution(format!(
                    "search was incomplete, so \"no matches\" may not be accurate: {err}"
                )));
            }
            return Ok(no_match.into());
        }
        // A rough per-line estimate (path + line number + up to `MAX_LINE` chars of text), capped by
        // the same byte ceiling the loop below enforces anyway — avoids paying for several
        // `String` grow-and-copy steps on a large match set without ever over-allocating past what
        // the output could actually reach.
        let mut out = String::with_capacity(
            (matches.len().saturating_mul(64)).min(super::output::MAX_LISTING_BYTES),
        );
        let mut lines_truncated = false;
        for hit in &matches {
            let (path, line, text, is_match) = (&hit.path, hit.line, &hit.text, hit.is_match);
            // Write straight into `out` instead of allocating a `format!` temp String per line — same
            // fix as `read`/`ls`'s formatting loops. Match lines use `path:line:` (ripgrep's
            // separator); context lines use `path-line-` so a reader (and the model) can tell a hit
            // from its surrounding context at a glance. `writeln!` into a `String` can't fail, so the
            // `Result` is discarded.
            if text.ends_with(LINE_TRUNCATED_SUFFIX) {
                lines_truncated = true;
            }
            let display_path = format_path(path, &root);
            let _ = if is_match {
                writeln!(out, "{display_path}:{line}: {text}")
            } else {
                writeln!(out, "{display_path}-{line}- {text}")
            };
        }
        // The byte cap is checked *before* any count/line marker is appended, and takes priority when
        // any would otherwise fire — see `cap_listing_bytes`'s doc comment. The byte-cap marker's own
        // "narrow the pattern, path, or context" guidance already covers both cases, so dropping the
        // more specific markers in favor of it loses nothing actionable.
        let byte_capped = super::output::cap_listing_bytes(
            &mut out,
            "narrow the pattern, path, or context to see more",
        );
        if !byte_capped {
            let mut notices = Vec::new();
            if truncated {
                // pi-parity fix: pi's own `grep.ts` truncation message names a concrete next value to
                // try ("${effectiveLimit} matches limit reached. Use limit=${effectiveLimit * 2} for
                // more, or refine pattern") rather than just gesturing at "raise `limit`" with no number
                // — the model had to guess how much higher to go. `saturating_mul`, not plain `*`: a
                // model-supplied `limit` has no upper bound here (only a floor of 1), and this crate
                // runs with `overflow-checks = true` in release — an adversarial/huge `limit` must
                // degrade to `usize::MAX`, never panic.
                notices.push(format!(
                    "match limit {limit} reached; narrow the pattern or use limit={} for more",
                    limit.saturating_mul(2)
                ));
            }
            if lines_truncated {
                notices.push("some lines truncated; use the read tool to see them in full".into());
            }
            if let Some(err) = &walk_error {
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
    async fn literal_mode_searches_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "call a.b(c) here\nand axbc too\n").unwrap();
        // As a regex `a.b(c)` matches `axbc`; literal mode must match only the verbatim text.
        let out = Grep::default()
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
    async fn output_paths_are_relative_to_the_search_root_not_absolute() {
        // A prior bug reported the full path straight from the walk entry (often absolute, since the
        // search root itself usually is) — costing the model extra tokens per line for a prefix it
        // already knows, and diverging from pi's documented "relative to search directory" contract.
        // Exact-match (not `.contains`) so a leftover root prefix can't slip through undetected.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "needle\n").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub").join("b.txt"), "needle\n").unwrap();

        let out = Grep::default()
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        let lines: Vec<&str> = out.lines().collect();
        assert!(
            lines.contains(&"a.txt:1: needle"),
            "expected a root-relative bare filename, got: {out}"
        );
        assert!(
            lines.contains(&"sub/b.txt:1: needle"),
            "expected a root-relative nested path, got: {out}"
        );
        assert!(
            !out.contains(dir.path().to_str().unwrap()),
            "the search root's own absolute prefix must not appear in the output: {out}"
        );
    }

    #[tokio::test]
    async fn searching_a_single_file_reports_just_its_basename() {
        // Mirrors pi: when `path` names a single file (not a directory), there's nothing meaningful to
        // show beyond its own name — the "relative to root" and "the file itself" cases coincide.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "needle\n").unwrap();

        let out = Grep::default()
            .run(json!({ "pattern": "needle", "path": file.to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert_eq!(out.trim_end(), "a.txt:1: needle", "got: {out}");
    }

    #[tokio::test]
    async fn finds_matches_with_path_and_line() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\nworld\nhello again\n").unwrap();
        std::fs::write(dir.path().join("b.log"), "nothing here\n").unwrap();

        let out = Grep::default()
            .run(json!({ "pattern": "hello", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("a.txt:1: hello"));
        assert!(out.contains("a.txt:3: hello again"));
        assert!(!out.contains("b.log"));
    }

    #[tokio::test]
    async fn run_normalizes_the_path_argument() {
        // Proves `run` actually calls `super::normalize_path`, via its `@`-prefix-strip behavior
        // (needs no `$HOME` mutation — see `expand_tilde`'s own direct unit tests for that half).
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello\n").unwrap();
        let at_prefixed = format!("@{}", dir.path().to_str().unwrap());
        let out = Grep::default()
            .run(json!({ "pattern": "hello", "path": at_prefixed }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("a.txt:1: hello"));
    }

    #[tokio::test]
    async fn glob_filters_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("keep.rs"), "fn target() {}\n").unwrap();
        std::fs::write(dir.path().join("skip.txt"), "fn target() {}\n").unwrap();
        let out = Grep::default()
            .run(json!({ "pattern": "target", "path": dir.path().to_str().unwrap(), "glob": "*.rs" }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("keep.rs"));
        assert!(!out.contains("skip.txt"));
    }

    #[tokio::test]
    async fn a_leading_bang_in_the_glob_negates_it() {
        // pi-parity fix (task 16): `globset::Glob` has no negation syntax, so a leading "!" was
        // matched literally and `!*.test.rs` confidently matched nothing at all — the entire query
        // returned zero results instead of applying the exclusion, unlike real `rg --glob`.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("plain.rs"), "fn target() {}\n").unwrap();
        std::fs::write(dir.path().join("thing.test.rs"), "fn target() {}\n").unwrap();
        let out = Grep::default()
            .run(json!({
                "pattern": "target",
                "path": dir.path().to_str().unwrap(),
                "glob": "!*.test.rs"
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("plain.rs"),
            "the non-excluded file must still be found: {out}"
        );
        assert!(
            !out.contains("thing.test.rs"),
            "the excluded file must not be reported: {out}"
        );
    }

    #[tokio::test]
    async fn gitignore_is_honored_even_outside_a_git_repository() {
        // `tempfile::tempdir()` lives under the system temp dir, not inside any git repository, so
        // this exercises `require_git(false)`: without it, the `ignore` crate's default behavior is
        // to stop respecting `.gitignore` files entirely once there's no `.git` ancestor to find.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "needle\n").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "needle\n").unwrap();
        let out = Grep::default()
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("kept.txt"), "got: {out}");
        assert!(
            !out.contains("ignored.txt"),
            ".gitignore should be honored even without a .git directory: {out}"
        );
    }

    #[tokio::test]
    async fn gitignore_is_still_honored_inside_a_real_git_repository() {
        // pi-parity fix (L7): `require_git(false)` is now conditional on the search root *not* being
        // inside a real git repo (a deliberate divergence from pi's own `rg` invocation, which has no
        // such conditional at all — see this module's doc comment on `require_git(false)`) — this
        // guards against the conditional accidentally breaking the far more common case, a search root
        // that *is* inside a real repo, which must still honor `.gitignore` exactly as before.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(dir.path().join("ignored.txt"), "needle\n").unwrap();
        std::fs::write(dir.path().join("kept.txt"), "needle\n").unwrap();
        let out = Grep::default()
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("kept.txt"), "got: {out}");
        assert!(
            !out.contains("ignored.txt"),
            ".gitignore must still be honored inside a real git repository: {out}"
        );
    }

    #[tokio::test]
    async fn no_matches_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "abc\n").unwrap();
        let out = Grep::default()
            .run(json!({ "pattern": "zzz", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("no matches"));
    }

    /// Skip a permission-bit test under a runtime that doesn't actually enforce them (root, some
    /// sandboxes) — matches the same guard `resources.rs`'s unreadable-file tests use.
    #[cfg(unix)]
    fn mode_actually_blocks_reads(path: &std::path::Path) -> bool {
        std::fs::read_dir(path).is_err()
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_search_root_reports_an_error_not_a_false_no_matches() {
        // Pi-parity audit H69: real ripgrep exits non-zero the moment it can't read a path in the
        // tree, root included — this crate's own walk previously swallowed that into `WalkState::
        // Continue`, so an unreadable root silently reported "no matches" instead of surfacing the
        // real problem.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "needle\n").unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o000)).unwrap();
        let blocks = mode_actually_blocks_reads(dir.path());

        let result = Grep::default()
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
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
    async fn an_unreadable_subdirectory_with_no_matches_elsewhere_reports_an_error() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("readable.txt"), "no match here\n").unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::write(locked.join("secret.txt"), "needle\n").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let blocks = mode_actually_blocks_reads(&locked);

        let result = Grep::default()
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await;
        let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755));

        if !blocks {
            return;
        }
        let err = result.expect_err(
            "an unreadable subdirectory with no matches found elsewhere must surface as an error, \
             not a false \"no matches\"",
        );
        let ToolError::Execution(msg) = err else {
            panic!("expected Execution error, got {err:?}")
        };
        assert!(
            !msg.contains("no matches for"),
            "must not phrase this as a confident empty result: {msg}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_subdirectory_alongside_real_matches_still_returns_them_with_a_notice() {
        // A real match elsewhere in the tree must not be discarded just because some other, unrelated
        // path couldn't be read — a deliberate, documented divergence from pi's own all-or-nothing
        // behavior (discarding every match the moment `rg` exits non-zero): silently losing real,
        // already-found matches over an unrelated permission hiccup would make the tool much less
        // useful in practice. The incompleteness is still surfaced, just as a notice, not a failure.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("readable.txt"), "needle here\n").unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::write(locked.join("secret.txt"), "needle too\n").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let blocks = mode_actually_blocks_reads(&locked);

        let result = Grep::default()
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap() }))
            .await;
        let _ = std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755));

        if !blocks {
            return;
        }
        let out = result.unwrap().text;
        assert!(
            out.contains("readable.txt"),
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
        // Created out of order; the report must come back path-sorted regardless of walk/thread order.
        for name in ["z.txt", "a.txt", "m.txt"] {
            std::fs::write(dir.path().join(name), "needle\n").unwrap();
        }
        let path = dir.path().to_str().unwrap();
        let run_once = || async {
            Grep::default()
                .run(json!({ "pattern": "needle", "path": path }))
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
        let out = Grep::default()
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
        let out = Grep::default()
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
    async fn output_byte_cap_truncates_even_under_the_match_limit() {
        let dir = tempfile::tempdir().unwrap();
        // 100 long matching lines with generous context: well under the 100-match `limit` (so the
        // match-count marker never fires), but the rendered text — line + surrounding context — blows
        // past the 50KB output cap on its own. Each line is also individually clipped by `MAX_LINE`, so
        // this exercises the per-line "[… truncated]" marker and the whole-output byte-cap marker
        // together in one realistic large-context scenario.
        let long_line = format!("needle {}", "x".repeat(600));
        let body = std::iter::repeat_n(long_line, 100)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("a.txt"), &body).unwrap();
        let out = Grep::default()
            .run(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
                "limit": 1000,
                "context": 5,
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.len() <= super::super::output::MAX_LISTING_BYTES + 256,
            "output should be capped near MAX_LISTING_BYTES, got {} bytes",
            out.len()
        );
        assert!(
            out.contains("… [truncated]"),
            "per-line clip marker missing"
        );
        assert!(
            out.contains("[output truncated at 50.0KB"),
            "byte-cap marker missing: {}",
            &out[out.len().saturating_sub(200)..]
        );
        assert!(
            !out.contains("match limit"),
            "match-count marker must not co-fire when count never exceeded the limit"
        );
        assert!(
            !out.contains("some lines truncated"),
            "the aggregate line-truncation notice must not survive the byte cap either — same \
             precedence as the match-count marker, and for the same reason (it could be sliced \
             through by the truncation it would be describing)"
        );
    }

    #[tokio::test]
    async fn aggregate_notice_reports_when_lines_were_individually_clipped() {
        let dir = tempfile::tempdir().unwrap();
        // One over-long matching line among otherwise-short ones: neither the match-count marker nor
        // the byte-cap marker fires (both stay well clear of their thresholds), isolating the
        // aggregate "some lines truncated" notice as the only truncation signal in the output.
        let long_line = format!("needle {}", "x".repeat(600));
        let body = format!("{long_line}\nneedle short\n");
        std::fs::write(dir.path().join("a.txt"), &body).unwrap();
        let out = Grep::default()
            .run(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains(LINE_TRUNCATED_SUFFIX),
            "per-line clip missing: {out}"
        );
        assert!(
            out.contains("[some lines truncated; use the read tool to see them in full]"),
            "aggregate notice missing: {out}"
        );
        assert!(
            !out.contains("match limit"),
            "match-count marker must not co-fire: {out}"
        );
    }

    #[tokio::test]
    async fn aggregate_notice_joins_match_limit_and_line_truncation_when_both_fire() {
        let dir = tempfile::tempdir().unwrap();
        // First matching line is over-long (individually clipped); many more short matches follow so
        // the match-count marker also fires. A single file keeps in-file line order deterministic
        // (unlike a multi-file walk), so the over-long first line is guaranteed to survive the
        // `limit` truncation and both notices are guaranteed to co-fire.
        let long_line = format!("needle {}", "x".repeat(600));
        let mut lines = vec![long_line];
        lines.extend(std::iter::repeat_n("needle short".to_string(), 20));
        let body = lines.join("\n");
        std::fs::write(dir.path().join("a.txt"), &body).unwrap();
        let out = Grep::default()
            .run(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
                "limit": 5,
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains(
                "[match limit 5 reached; narrow the pattern or use limit=10 for more. some lines \
                 truncated; use the read tool to see them in full]"
            ),
            "joined notice missing or in the wrong order: {out}"
        );
    }

    #[tokio::test]
    async fn output_byte_cap_takes_priority_over_the_match_count_marker() {
        let dir = tempfile::tempdir().unwrap();
        // 200 long matching lines, default `limit` of 100: both the match-count marker (200 > 100) and
        // the byte cap would fire on the same rendered text. Appending the count marker and *then*
        // truncating to the byte cap would risk slicing straight through (and corrupting) that just-
        // appended marker — the byte-cap check runs on the body first and, when it trips, wins outright:
        // only its own marker (whose "narrow the pattern, path, or context" advice already subsumes the
        // count case) appears, never a mangled partial marker.
        let long_line = format!("needle {}", "x".repeat(600));
        let body = std::iter::repeat_n(long_line, 200)
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("a.txt"), &body).unwrap();
        let out = Grep::default()
            .run(json!({
                "pattern": "needle",
                "path": dir.path().to_str().unwrap(),
                "context": 5,
            }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("[output truncated at 50.0KB"),
            "byte-cap marker missing: {out}"
        );
        assert!(
            !out.contains("match limit"),
            "the byte-cap marker must win cleanly, not leave a mangled match-count marker behind"
        );
        assert!(
            out.len() <= super::super::output::MAX_LISTING_BYTES + 256,
            "output should be capped near MAX_LISTING_BYTES, got {} bytes",
            out.len()
        );
    }

    #[tokio::test]
    async fn limit_truncates_to_at_most_limit_matches() {
        // The walk now quits as soon as `limit` matches are found (see `search`'s doc comment on why),
        // so which 3 of the 10 files survive can vary with thread scheduling across a *parallel* walk —
        // unlike the untruncated case (`output_is_path_sorted_and_deterministic`), which always finds
        // everything and so is always fully deterministic. This test asserts the contract that's still
        // guaranteed: truncation actually happened, and no more than `limit` matches came back.
        let dir = tempfile::tempdir().unwrap();
        for i in 0..10 {
            std::fs::write(dir.path().join(format!("f{i:02}.txt")), "needle\n").unwrap();
        }
        let out = Grep::default()
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap(), "limit": 3 }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("[match limit 3 reached; narrow the pattern or use limit=6 for more]")
        );
        let match_lines = out.lines().filter(|l| l.contains("needle")).count();
        assert_eq!(
            match_lines, 3,
            "at most `limit` matches must be reported: {out}"
        );
    }

    #[tokio::test]
    async fn a_limit_of_zero_is_floored_to_one_not_a_confident_no_matches() {
        // Pi-parity audit M10: matches pi's own `grep.ts` (`Math.max(1, limit ?? DEFAULT_LIMIT)`).
        // Without the floor, `limit: 0` made `stop_at` zero, so `search`'s very first "have we hit the
        // threshold?" check was already true before a single file was scanned — a confidently wrong
        // "no matches" even though a real match exists.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "needle\n").unwrap();
        let out = Grep::default()
            .run(json!({ "pattern": "needle", "path": dir.path().to_str().unwrap(), "limit": 0 }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("needle"),
            "limit: 0 must still find the real match (floored to 1), not report false emptiness: {out}"
        );
    }

    #[tokio::test]
    async fn an_untruncated_result_is_still_fully_deterministic() {
        // Fewer matches than `limit` — the walk never quits early, so every match is found regardless
        // of thread scheduling, and the sorted output is exactly reproducible run to run (unlike the
        // truncated case, which the early-exit optimization deliberately no longer guarantees).
        let dir = tempfile::tempdir().unwrap();
        for name in ["z.txt", "a.txt", "m.txt"] {
            std::fs::write(dir.path().join(name), "needle\n").unwrap();
        }
        let path = dir.path().to_str().unwrap();
        let run_once = || async {
            Grep::default()
                .run(json!({ "pattern": "needle", "path": path, "limit": 100 }))
                .await
                .unwrap()
                .text
        };
        let first = run_once().await;
        assert!(first.contains("a.txt") && first.contains("m.txt") && first.contains("z.txt"));
        for _ in 0..5 {
            assert_eq!(
                first,
                run_once().await,
                "an untruncated result must be stable across runs"
            );
        }
    }

    #[tokio::test]
    async fn searches_non_utf8_files() {
        // A file with an invalid UTF-8 byte (0xE9, latin-1 'é') around an ASCII match. The old
        // `BufReader::read_line` loop errored on the bad byte and silently dropped the whole file; the
        // byte-oriented ripgrep engine finds the match and decodes the line lossily. This is the
        // correctness regression vs pi's `rg` that the rewrite fixes.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("latin1.txt"), b"caf\xe9 NEEDLE here\n").unwrap();
        let out = Grep::default()
            .run(json!({ "pattern": "NEEDLE", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(
            out.contains("NEEDLE"),
            "must match inside a non-UTF-8 file: {out}"
        );
        assert!(out.contains("latin1.txt"));
    }

    #[tokio::test]
    async fn skips_binary_files() {
        // A NUL byte marks a file as binary; the searcher quits on it instead of emitting garbage
        // (ripgrep's default). The adjacent text file still matches.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("blob.bin"), b"\x00\x00 NEEDLE \x00binary").unwrap();
        std::fs::write(dir.path().join("text.txt"), b"NEEDLE in text\n").unwrap();
        let out = Grep::default()
            .run(json!({ "pattern": "NEEDLE", "path": dir.path().to_str().unwrap() }))
            .await
            .unwrap()
            .text;
        assert!(out.contains("text.txt"), "the text file must match: {out}");
        assert!(
            !out.contains("blob.bin"),
            "the binary file must be skipped, not reported: {out}"
        );
    }

    #[test]
    fn description_documents_the_truncation_budgets() {
        // Pi-parity fix: unlike `bash`'s description (which states both its default timeout and its
        // output truncation budget numerically), `grep`'s description only stated the match-count
        // default in the `limit` schema field, never the overall byte-truncation cap it shares with
        // `find`/`ls`, nor the per-line character clip.
        let desc = Grep::default().description().to_string();
        assert!(
            desc.contains(&DEFAULT_LIMIT.to_string())
                && desc.contains(&super::super::output::format_size(
                    super::super::output::MAX_LISTING_BYTES as u64
                ))
                && desc.contains(&MAX_LINE.to_string()),
            "description should state the match-count, byte, and per-line truncation budgets \
             numerically, got: {desc}"
        );
    }
}
