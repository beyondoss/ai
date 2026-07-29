//! Tool-call permission policy — a concrete [`agent_core::AgentHooks`] implementation.
//!
//! `agent_core::hooks::AgentHooks` is the seam a permission/approval system hangs off of, but the
//! crate that defines it ships zero implementations (it has no concrete `Tool`s to gate, and no
//! opinion on what a host's policy should look like). This is that implementation for `beyond-ai-agent`
//! itself: a static, config-time deny-list — the smallest useful policy, and the one every richer
//! policy (an approval round-trip to a client, a rules engine) would still need as its base case.
//!
//! Before this existed, `Agent::with_hooks` had exactly one call site in the whole workspace — a unit
//! test in `agent_core::hooks` — so every real `run`/`serve` process built its `Agent` with the no-op
//! default `NoHooks`, meaning `bash`/`write`/`edit` ran completely unconstrained: the only thing this
//! codebase's separate trust system gates is whether *project-local files are read* at startup, never
//! whether a *tool call executes*.

use agent_core::{AgentHooks, CancellationToken};
use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use serde_json::Value;
use std::collections::HashSet;

/// A static tool-call gate: block a call outright by tool name, by (for `bash` specifically) a
/// substring match against the command text, or by (for `write`/`edit` specifically) a glob match
/// against the target path. Case-insensitive on the bash-pattern side — a policy author writing
/// `rm -rf` shouldn't have to also list `RM -RF`.
#[derive(Debug, Default, Clone)]
pub struct ToolPolicy {
    denied_tools: HashSet<String>,
    /// Lowercased once at construction time so `before_tool_call` never re-lowercases the pattern list
    /// on every single call — only the (much shorter-lived) command string gets lowercased per call.
    denied_bash_patterns: Vec<String>,
    /// Compiled once at construction (same reasoning as `denied_bash_patterns`'s one-time lowercasing)
    /// rather than re-parsing the same glob source on every `write`/`edit` call.
    denied_paths: Vec<GlobMatcher>,
    /// The same root the gated agent's filesystem tools resolve relative paths against (see
    /// [`crate::tools::resolve_against`]). Empty = the process cwd, which is every non-subagent case.
    ///
    /// This must track the tools' root or the policy checks a *different path than the one that gets
    /// written*: a subagent under `isolation: worktree` resolves `edit path=notes.md` to
    /// `<worktree>/notes.md`, so a policy still keying on `<process cwd>/notes.md` would compare the
    /// wrong string. Relative globs (`**/secrets.env`) then behave identically inside a worktree.
    ///
    /// An **absolute** deny glob (`/home/me/repo/secrets.env`) still cannot match a worktree path, and
    /// this field does not change that. What closes that hole is the merge-back step, which re-checks
    /// every path in a child's patch against these same `denied_paths` — mapped back to the main tree —
    /// before `git apply` runs. `git apply` is not a tool call and never reaches `before_tool_call`, so
    /// merge-back is the only place that check can live.
    root: std::path::PathBuf,
    /// The target's `$HOME` when the gated agent's filesystem tools act somewhere other than this
    /// host, or `None` for the ordinary local case.
    ///
    /// Presence of a value is what selects lexical rather than `canonicalize`-based path resolution —
    /// see [`crate::tools::write_key`]. That distinction is load-bearing *here specifically*: a deny
    /// glob evaluated against a host-`canonicalize`d spelling of a path that lives on another machine
    /// is checking a different file than the one about to be written, which is a bypass rather than a
    /// cosmetic bug.
    remote_home: Option<String>,
    /// Whether the gated tools are remote at all. Separate from [`ToolPolicy::remote_home`] because a
    /// target whose home we don't know is still remote, and collapsing the two would silently drop
    /// such a policy back onto host resolution.
    remote: bool,
}

impl ToolPolicy {
    /// An empty policy — every call allowed. Prefer [`ToolPolicy::is_empty`] over constructing this
    /// and calling `.with_hooks` unconditionally: an empty policy is behaviorally identical to
    /// `NoHooks`, just paying an extra vtable indirection per call for no reason.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from `--deny-tool`/`--deny-bash-pattern`/`--deny-path`-shaped lists — shared by `run` and
    /// `serve` so the three flags mean exactly the same thing regardless of which subcommand installs
    /// them.
    pub fn from_lists(
        deny_tool: &[String],
        deny_bash_pattern: &[String],
        deny_path: &[String],
    ) -> Self {
        let mut policy = Self::new();
        for name in deny_tool {
            policy = policy.deny_tool(name.clone());
        }
        for pattern in deny_bash_pattern {
            policy = policy.deny_bash_pattern(pattern.clone());
        }
        for pattern in deny_path {
            policy = policy.deny_path(pattern);
        }
        policy
    }

    /// Builder-style: resolve `write`/`edit` paths against `root` rather than the process cwd — see the
    /// [`ToolPolicy::root`] field. Must be given the same root the gated agent's tool registry was built
    /// with.
    pub fn with_root(mut self, root: impl Into<std::path::PathBuf>) -> Self {
        self.root = root.into();
        self
    }

    /// Builder-style: police paths on another filesystem rather than this host. Must be set whenever
    /// the gated agent's tool registry was built with a non-local [`crate::tools::fs::FsBackend`], or
    /// the deny-list evaluates a host-resolved spelling of a path that isn't on this host.
    pub fn with_remote(mut self, home: Option<String>) -> Self {
        self.remote = true;
        self.remote_home = home;
        self
    }

    /// The world this policy resolves paths in — the same one the gated tools use.
    pub fn world(&self) -> crate::tools::fs::PathWorld {
        if self.remote {
            crate::tools::fs::PathWorld::Remote {
                home: self.remote_home.clone(),
            }
        } else {
            crate::tools::fs::PathWorld::Local
        }
    }

    /// The compiled deny-path globs, for the merge-back re-check described on [`ToolPolicy::root`].
    pub fn denied_paths(&self) -> &[GlobMatcher] {
        &self.denied_paths
    }

    /// The filesystem root this policy resolves relative paths against — see [`ToolPolicy::root`].
    /// [`crate::approval`] needs it for the same reason this hook does: a remembered "always allow this
    /// path" decision must key on the path that will actually be written, not a spelling of it.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    /// Deny every call to a tool named `name`, regardless of arguments.
    pub fn deny_tool(mut self, name: impl Into<String>) -> Self {
        self.denied_tools.insert(name.into());
        self
    }

    /// Deny a `bash` call whenever its `command` string contains `pattern` (case-insensitive substring
    /// match — deliberately simple, not a regex/glob engine: a config-time deny-list is meant to be
    /// read and audited by a human, and a plain substring is the easiest shape to reason about
    /// correctly under that constraint).
    pub fn deny_bash_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.denied_bash_patterns
            .push(pattern.into().to_ascii_lowercase());
        self
    }

    /// Deny a `write`/`edit` call whenever its `path` argument matches `pattern` — the same glob
    /// engine `grep`'s `--glob`/`find`'s own pattern matching already use (`globset::Glob`), so a
    /// policy author reaches for one familiar syntax across this whole crate. An unparseable pattern
    /// is reported and otherwise ignored by this low-level builder, so a single bad glob can't panic a
    /// long-running `serve` process mid-`build_agent`-rebuild. This is *not* a safe default for
    /// CLI/config-facing callers, though: silently dropping the pattern here means the resulting
    /// policy just has one fewer rule, which for a security control is a fail-open — a typo'd
    /// `--deny-path` would otherwise leave an operator believing a path is blocked when it plainly
    /// isn't. Callers building a policy from user-supplied input must call
    /// [`Self::validate_deny_path_patterns`] first and fail the whole process on `Err`, exactly as
    /// `main.rs` does for both `run` and `serve` before ever reaching this builder.
    pub fn deny_path(mut self, pattern: impl AsRef<str>) -> Self {
        let pattern = pattern.as_ref();
        match Glob::new(pattern) {
            Ok(glob) => self.denied_paths.push(glob.compile_matcher()),
            Err(e) => eprintln!("policy: ignoring invalid --deny-path glob {pattern:?}: {e}"),
        }
        self
    }

    /// Validate every `--deny-path` glob up front, before any policy is installed or any hook
    /// decision is made from it. Returns the first unparseable pattern's error so the caller can
    /// reject the whole invocation rather than silently continuing with that rule dropped — see
    /// [`Self::deny_path`]'s doc comment for why leniency at this boundary is the wrong default.
    pub fn validate_deny_path_patterns(deny_path: &[String]) -> Result<(), String> {
        for pattern in deny_path {
            Glob::new(pattern).map_err(|e| format!("invalid --deny-path glob {pattern:?}: {e}"))?;
        }
        Ok(())
    }

    /// Whether this policy would ever actually block anything — a caller builds one unconditionally
    /// from CLI flags/env, so this is the one-time check that decides whether installing the hook is
    /// worth the per-call overhead at all.
    pub fn is_empty(&self) -> bool {
        self.denied_tools.is_empty()
            && self.denied_bash_patterns.is_empty()
            && self.denied_paths.is_empty()
    }
}

#[async_trait]
impl AgentHooks for ToolPolicy {
    async fn before_tool_call(
        &self,
        name: &str,
        input: &Value,
        _session: &agent_core::Session,
        _cancel: &CancellationToken,
    ) -> Option<String> {
        if self.denied_tools.contains(name) {
            return Some(format!("tool '{name}' is denied by policy"));
        }
        if name == "bash" && !self.denied_bash_patterns.is_empty() {
            let command = input.get("command").and_then(Value::as_str)?;
            let lower = command.to_ascii_lowercase();
            if let Some(pattern) = self
                .denied_bash_patterns
                .iter()
                .find(|p| lower.contains(p.as_str()))
            {
                return Some(format!(
                    "bash command blocked by policy (matches denied pattern {pattern:?})"
                ));
            }
        }
        if (name == "write" || name == "edit") && !self.denied_paths.is_empty() {
            let path = input.get("path").and_then(Value::as_str)?;
            // The same canonical form `WriteLockRegistry` keys on, and — crucially — the same one the
            // `write`/`edit` tools themselves compute: resolved against *this policy's root* (the tools'
            // root; the process cwd when empty) and symlink-followed when the target already exists, so
            // a pattern like `--deny-path '/etc/**'` can't be sidestepped with a `./`/`..`-laden or
            // relative spelling of the same path. Resolving against a different root than the tools use
            // would check a path that is not the one about to be written.
            let target = crate::tools::write_key(&self.root, path, &self.world());
            if let Some(m) = self.denied_paths.iter().find(|m| m.is_match(&target)) {
                return Some(format!(
                    "path '{target}' is denied by policy (matches {:?})",
                    m.glob().glob()
                ));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_policy_reports_empty() {
        assert!(ToolPolicy::new().is_empty());
        assert!(!ToolPolicy::new().deny_tool("bash").is_empty());
        assert!(!ToolPolicy::new().deny_bash_pattern("rm -rf").is_empty());
    }

    #[tokio::test]
    async fn denies_a_tool_by_name_regardless_of_input() {
        let policy = ToolPolicy::new().deny_tool("bash");
        let session = agent_core::Session::new();
        let cancel = CancellationToken::new();
        let reason = policy
            .before_tool_call("bash", &json!({"command": "echo hi"}), &session, &cancel)
            .await;
        assert_eq!(reason, Some("tool 'bash' is denied by policy".to_string()));
        assert!(
            policy
                .before_tool_call("read", &json!({}), &session, &cancel)
                .await
                .is_none(),
            "a name-based deny must not affect other tools"
        );
    }

    #[tokio::test]
    async fn denies_a_bash_command_matching_a_pattern_case_insensitively() {
        let policy = ToolPolicy::new().deny_bash_pattern("rm -rf");
        let session = agent_core::Session::new();
        let cancel = CancellationToken::new();
        assert!(
            policy
                .before_tool_call("bash", &json!({"command": "RM -RF /"}), &session, &cancel)
                .await
                .is_some(),
            "pattern matching must be case-insensitive"
        );
        assert!(
            policy
                .before_tool_call("bash", &json!({"command": "echo hi"}), &session, &cancel)
                .await
                .is_none(),
            "an unrelated command must not be blocked"
        );
    }

    #[tokio::test]
    async fn bash_patterns_do_not_affect_other_tools_even_with_a_command_like_field() {
        let policy = ToolPolicy::new().deny_bash_pattern("secret");
        let session = agent_core::Session::new();
        let cancel = CancellationToken::new();
        assert!(
            policy
                .before_tool_call("write", &json!({"command": "secret"}), &session, &cancel)
                .await
                .is_none(),
            "bash-pattern deny-listing must only ever apply to the `bash` tool itself"
        );
    }

    #[tokio::test]
    async fn denies_a_write_or_edit_call_whose_path_matches_a_denied_glob() {
        let dir = tempfile::tempdir().unwrap();
        let secret = dir.path().join("secrets.env");
        let policy = ToolPolicy::new().deny_path("*.env");
        let session = agent_core::Session::new();
        let cancel = CancellationToken::new();

        let reason = policy
            .before_tool_call(
                "write",
                &json!({"path": secret.to_str().unwrap(), "content": "x"}),
                &session,
                &cancel,
            )
            .await;
        assert!(
            reason.is_some_and(|r| r.contains("denied by policy")),
            "a `write` whose path matches a denied glob must be blocked"
        );

        // `edit` is gated the same way as `write`.
        assert!(
            policy
                .before_tool_call(
                    "edit",
                    &json!({"path": secret.to_str().unwrap(), "old_string": "a", "new_string": "b"}),
                    &session,
                    &cancel,
                )
                .await
                .is_some(),
            "`edit` must be gated by `deny_path` the same way `write` is"
        );

        // An unrelated path is untouched.
        let other = dir.path().join("notes.md");
        assert!(
            policy
                .before_tool_call(
                    "write",
                    &json!({"path": other.to_str().unwrap(), "content": "x"}),
                    &session,
                    &cancel,
                )
                .await
                .is_none(),
            "a path that doesn't match the glob must not be blocked"
        );

        // A tool other than `write`/`edit` is never gated by `deny_path`, even with a matching `path`.
        assert!(
            policy
                .before_tool_call(
                    "read",
                    &json!({"path": secret.to_str().unwrap()}),
                    &session,
                    &cancel,
                )
                .await
                .is_none(),
            "deny_path must only ever apply to write/edit"
        );
    }

    #[tokio::test]
    async fn an_invalid_deny_path_glob_is_ignored_rather_than_denying_everything() {
        // A malformed `--deny-path` pattern must not silently deny every `write`/`edit` call, nor
        // crash the process — see `deny_path`'s own doc comment. (The CLI boundary is expected to
        // reject this input outright before it ever reaches this low-level builder — see
        // `validate_deny_path_patterns_rejects_an_unparseable_glob` below.)
        let policy = ToolPolicy::new().deny_path("[");
        assert!(
            policy.is_empty(),
            "an unparseable glob must be dropped, not silently kept as a standing deny-all"
        );
    }

    #[tokio::test]
    async fn a_rooted_policy_resolves_a_relative_path_against_its_root_not_the_process_cwd() {
        // A subagent under `isolation: worktree` gets tools rooted at its worktree, so `edit
        // path=secrets.env` writes `<worktree>/secrets.env`. The policy must check *that* path — a
        // policy still keying on `<process cwd>/secrets.env` would be comparing a string that no tool
        // is ever going to write, and the deny would silently stop firing.
        let wt = tempfile::tempdir().unwrap();
        let policy = ToolPolicy::new()
            .deny_path("**/secrets.env")
            .with_root(wt.path());
        let session = agent_core::Session::new();
        let cancel = CancellationToken::new();

        let reason = policy
            .before_tool_call(
                "edit",
                &json!({"path": "secrets.env", "old_string": "a", "new_string": "b"}),
                &session,
                &cancel,
            )
            .await
            .expect("a relative path resolving into a denied glob must be blocked");
        assert!(reason.contains("denied by policy"), "{reason}");
        assert!(
            reason.contains(&wt.path().join("secrets.env").display().to_string()),
            "the message must name the path actually about to be written: {reason}"
        );
    }

    #[tokio::test]
    async fn an_unrooted_policy_is_unchanged_by_the_root_field() {
        // Every non-subagent caller leaves `root` empty; behavior must be bit-identical to before it
        // existed — an absolute path is matched exactly as it always was.
        let dir = tempfile::tempdir().unwrap();
        let secret = dir.path().join("secrets.env");
        let policy = ToolPolicy::new().deny_path("*.env");
        let session = agent_core::Session::new();
        let cancel = CancellationToken::new();
        let reason = policy
            .before_tool_call(
                "write",
                &json!({"path": secret.to_str().unwrap(), "content": "x"}),
                &session,
                &cancel,
            )
            .await;
        assert!(reason.is_some_and(|r| r.contains("denied by policy")));
    }

    #[test]
    fn validate_deny_path_patterns_rejects_an_unparseable_glob() {
        let err = ToolPolicy::validate_deny_path_patterns(&["[".to_string()])
            .expect_err("an unparseable glob must be rejected, not silently accepted");
        assert!(
            err.contains("--deny-path"),
            "error should name the flag: {err}"
        );
    }

    #[test]
    fn validate_deny_path_patterns_accepts_valid_globs_and_an_empty_list() {
        assert!(ToolPolicy::validate_deny_path_patterns(&[]).is_ok());
        assert!(
            ToolPolicy::validate_deny_path_patterns(&[
                "*.env".to_string(),
                "**/secrets/**".to_string()
            ])
            .is_ok()
        );
    }

    #[tokio::test]
    async fn a_bash_call_with_no_command_field_is_not_blocked_by_a_pattern_policy() {
        // Malformed input reaching a hook (rather than the tool's own schema validation) shouldn't
        // panic or false-positive-deny — it should simply not match any pattern.
        let policy = ToolPolicy::new().deny_bash_pattern("rm -rf");
        let session = agent_core::Session::new();
        let cancel = CancellationToken::new();
        assert!(
            policy
                .before_tool_call("bash", &json!({}), &session, &cancel)
                .await
                .is_none()
        );
    }
}
