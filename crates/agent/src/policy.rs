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
use globset::{Glob, GlobBuilder, GlobMatcher};
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
    /// The `--deny-path` patterns as written. Kept because a relative pattern's meaning depends on
    /// [`ToolPolicy::root`] and the world, which builders set *after* the patterns (`from_lists`
    /// then `with_root`/`in_world`); each of those recompiles [`ToolPolicy::denied_paths`] from here.
    deny_path_patterns: Vec<String>,
    /// Compiled from `deny_path_patterns` against the current root and world (same reasoning as
    /// `denied_bash_patterns`'s one-time lowercasing) rather than re-parsing the same glob source on
    /// every `write`/`edit` call. See [`DenyRule`] and [`anchored_globs`].
    denied_paths: Vec<DenyRule>,
    /// The same root the gated agent's filesystem tools resolve relative paths against (see
    /// [`crate::tools::resolve_against`]). Empty = the process cwd, which is every non-subagent case.
    ///
    /// This must track the tools' root or the policy checks a *different path than the one that gets
    /// written*: a subagent under `isolation: worktree` resolves `edit path=notes.md` to
    /// `<worktree>/notes.md`, so a policy still keying on `<process cwd>/notes.md` would compare the
    /// wrong string. It is also what an anchored `--deny-path` (`secrets/*`) is anchored to — see
    /// [`anchored_globs`] — so relative patterns behave identically inside a worktree.
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

/// One compiled glob derived from a `--deny-path` pattern (see [`anchored_globs`]).
///
/// An absolute glob whose fixed prefix runs through a symlink (macOS's `/tmp -> /private/tmp`) gets
/// a second rule with that prefix resolved, because targets are keyed on their resolved path and
/// would otherwise never match the pattern as typed. Those twins are derived from this host's
/// filesystem, so a remote policy never compiles them.
#[derive(Debug, Clone)]
struct DenyRule {
    exact: GlobMatcher,
    /// The same glob matched ignoring case — used when the target sits on a case-insensitive volume,
    /// where `SECRETS/x` *is* `secrets/x` and a case-sensitive match would be a bypass.
    folded: GlobMatcher,
    /// The `--deny-path` pattern as the operator wrote it, for the denial message.
    source: String,
}

impl DenyRule {
    fn compile(glob: &str, source: &str) -> Result<Self, globset::Error> {
        Ok(Self {
            exact: Glob::new(glob)?.compile_matcher(),
            folded: GlobBuilder::new(glob)
                .case_insensitive(true)
                .build()?
                .compile_matcher(),
            source: source.to_string(),
        })
    }

    fn matches(&self, target: &str, case_insensitive: bool) -> bool {
        if case_insensitive {
            self.folded.is_match(target)
        } else {
            self.exact.is_match(target)
        }
    }
}

/// The globs one `--deny-path` pattern compiles to, with gitignore semantics:
///
/// - **No `/`** other than a trailing one (`.env`, `*.pem`): the name at any depth — `**/.env`.
/// - **A `/` inside** (`secrets/*`, `./config/prod.yml`): anchored to `root`, the policy's root.
/// - **Absolute** (`/etc/**`): as written.
/// - **Starting with `*`** (`**/secrets/**`, `*/x`): as written. globset's `*` crosses `/`, so these
///   already match at any depth, and anchoring would only narrow a deny-list.
/// - Every pattern also covers everything beneath a directory it names (`<glob>/**`), as gitignore
///   excluding a directory excludes its contents. So a **trailing `/`** (`secrets/`) means the
///   directory and everything beneath it; it is accepted and otherwise changes nothing.
///
/// `root` is `None` only for a remote policy with no root, whose targets stay relative too
/// ([`crate::tools::lexical_write_target`]); an anchored pattern then stays relative to match them.
fn anchored_globs(pattern: &str, root: Option<&std::path::Path>) -> Vec<String> {
    let trimmed = pattern.trim_end_matches('/');
    let base = if trimmed.is_empty() {
        // `/`: the filesystem root, i.e. everything.
        String::from("/")
    } else if trimmed.starts_with('/') || trimmed.starts_with('*') {
        trimmed.to_string()
    } else if !trimmed.contains('/') {
        format!("**/{trimmed}")
    } else {
        let rel = trimmed.trim_start_matches("./");
        match root {
            Some(root) => {
                let root = globset::escape(&root.to_string_lossy());
                format!("{}/{rel}", root.trim_end_matches('/'))
            }
            None => rel.to_string(),
        }
    };
    if base == "**" || base.ends_with("/**") {
        return vec![base];
    }
    let beneath = format!("{}/**", base.trim_end_matches('/'));
    vec![base, beneath]
}

/// An absolute deny pattern with its fixed (glob-free) leading components resolved against this
/// host — or `None` when resolving changes nothing, or the pattern is relative (`**/secrets/**`
/// matches anywhere and needs no prefix).
///
/// Targets are keyed on their resolved path ([`crate::tools::write_key`]), so `--deny-path
/// '/tmp/proj/secrets/**'` on macOS, where every target under it keys as `/private/tmp/...`, would
/// otherwise never fire.
fn host_resolved_pattern(pattern: &str) -> Option<String> {
    let path = std::path::Path::new(pattern);
    if !path.is_absolute() {
        return None;
    }
    // Split at the first component carrying glob syntax; everything before it is a literal path.
    let mut prefix = std::path::PathBuf::new();
    let mut rest: Vec<&str> = Vec::new();
    for component in path.components() {
        let s = component.as_os_str().to_str()?;
        if rest.is_empty() && !s.contains(['*', '?', '[', ']', '{', '}', '\\']) {
            prefix.push(component.as_os_str());
        } else {
            rest.push(s);
        }
    }
    let real = crate::path_utils::resolve_nearest(&prefix);
    if real == prefix {
        return None;
    }
    let mut out = globset::escape(real.to_str()?);
    for part in rest {
        if !out.ends_with('/') {
            out.push('/');
        }
        out.push_str(part);
    }
    Some(out)
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
        self.denied_paths = self.compile_deny_paths(&self.root);
        self
    }

    /// Builder-style: police paths on another filesystem rather than this host. Must be set whenever
    /// the gated agent's tool registry was built with a non-local [`crate::tools::fs::FsBackend`], or
    /// the deny-list evaluates a host-resolved spelling of a path that isn't on this host.
    pub fn with_remote(mut self, home: Option<String>) -> Self {
        self.remote = true;
        self.remote_home = home;
        self.denied_paths = self.compile_deny_paths(&self.root);
        self
    }

    /// Builder-style: take the world straight from the backend the gated tools were built with.
    ///
    /// Preferred over calling [`with_remote`](Self::with_remote) from a branch the caller writes
    /// itself: *the backend* is the authority on which filesystem a tool call lands on, and a policy
    /// that decides separately can disagree with it — which is a silently non-firing deny-list, not a
    /// visible error. Every policy construction site that has a backend on hand uses this.
    pub fn in_world(self, world: crate::tools::fs::PathWorld) -> Self {
        match world {
            crate::tools::fs::PathWorld::Local => self,
            crate::tools::fs::PathWorld::Remote { home } => self.with_remote(home),
        }
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

    /// The compiled deny-path globs — each pattern anchored at this policy's root (see
    /// [`anchored_globs`]), plus any host-resolved twin (see `host_resolved_pattern`).
    pub fn denied_paths(&self) -> Vec<GlobMatcher> {
        self.denied_paths.iter().map(|r| r.exact.clone()).collect()
    }

    /// [`denied_paths`](Self::denied_paths) with every anchored pattern anchored at `root` instead,
    /// for the merge-back re-check described on [`ToolPolicy::root`]: a worktree child's policy is
    /// rooted at its worktree, but its patch lands in the main tree, where `secrets/*` must mean
    /// `<main tree>/secrets/*`.
    pub fn denied_paths_at(&self, root: &std::path::Path) -> Vec<GlobMatcher> {
        self.compile_deny_paths(root)
            .into_iter()
            .map(|r| r.exact)
            .collect()
    }

    /// Compile every `--deny-path` pattern against `root` in this policy's world.
    fn compile_deny_paths(&self, root: &std::path::Path) -> Vec<DenyRule> {
        // An empty root is the process cwd: a true answer on this host, a fabricated one anywhere
        // else (the same rule `lexical_write_target` applies to the targets themselves).
        let cwd;
        let anchor = if !root.as_os_str().is_empty() {
            Some(root)
        } else if self.remote {
            None
        } else {
            cwd = std::env::current_dir().unwrap_or_default();
            Some(cwd.as_path())
        };
        let mut rules = Vec::new();
        for pattern in &self.deny_path_patterns {
            for glob in anchored_globs(pattern, anchor) {
                match DenyRule::compile(&glob, pattern) {
                    Ok(rule) => rules.push(rule),
                    Err(e) => {
                        eprintln!("policy: ignoring invalid --deny-path glob {pattern:?}: {e}");
                        continue;
                    }
                }
                // Targets key on their resolved path on this host, so a pattern whose literal
                // prefix runs through a symlink gets a resolved twin. Meaningless remotely.
                if !self.remote
                    && let Some(resolved) = host_resolved_pattern(&glob)
                    && let Ok(rule) = DenyRule::compile(&resolved, pattern)
                {
                    rules.push(rule);
                }
            }
        }
        rules
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
    ///
    /// A relative pattern follows gitignore semantics, anchored to the policy's root — see
    /// [`anchored_globs`].
    pub fn deny_path(mut self, pattern: impl AsRef<str>) -> Self {
        let pattern = pattern.as_ref();
        if let Err(e) = Glob::new(pattern) {
            eprintln!("policy: ignoring invalid --deny-path glob {pattern:?}: {e}");
            return self;
        }
        self.deny_path_patterns.push(pattern.to_string());
        self.denied_paths = self.compile_deny_paths(&self.root);
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
            // Every file the call writes — a multi-file `edit` names one per entry, and checking only
            // the top-level `path` would let an entry write a denied file.
            for path in agent_core::tool::written_paths(name, input) {
                if let Some(reason) = self.denied_path_reason(&path) {
                    return Some(reason);
                }
            }
        }
        None
    }
}

impl ToolPolicy {
    fn denied_path_reason(&self, path: &str) -> Option<String> {
        // The same canonical form `WriteLockRegistry` keys on, and — crucially — the same one the
        // `write`/`edit` tools themselves compute: resolved against *this policy's root* (the tools'
        // root; the process cwd when empty) and symlink-followed when the target already exists, so
        // a pattern like `--deny-path '/etc/**'` can't be sidestepped with a `./`/`..`-laden or
        // relative spelling of the same path. Resolving against a different root than the tools use
        // would check a path that is not the one about to be written.
        //
        // A target that doesn't exist yet still has its nearest existing ancestor resolved, so a
        // symlinked parent (`sl/new.env`, `sl -> secrets`) can't hide it; on a case-insensitive
        // volume the key is case-folded and matched case-insensitively, so `SECRETS/new.env` can't
        // either. The typed spelling is checked too: denying a path because of how it was *named*
        // is the fail-safe direction for a deny-list.
        let world = self.world();
        let (target, ci) = crate::tools::write_key_ci(&self.root, path, &world);
        let typed = crate::tools::lexical_write_target(
            &self.root,
            &crate::tools::resolve_against_in(&self.root, path, &world),
            &world,
        );
        let rule = self
            .denied_paths
            .iter()
            .find(|r| r.matches(&target, ci) || r.matches(&typed, ci))?;
        Some(format!(
            "path '{target}' is denied by policy (matches {:?})",
            rule.source
        ))
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
    async fn denies_a_multi_file_edit_when_any_entry_names_a_denied_path() {
        // The top-level `path` is fine; the second entry writes the secret. Checking only the
        // top-level path would let a per-entry `path` walk straight past `--deny-path`, including in
        // the JSON-string `edits` form the tool also accepts.
        let dir = tempfile::tempdir().unwrap();
        let policy = ToolPolicy::new().deny_path("*.env");
        let session = agent_core::Session::new();
        let cancel = CancellationToken::new();
        let ok = dir.path().join("notes.md");
        let secret = dir.path().join("secrets.env");
        let entries = json!([
            { "old_string": "a", "new_string": "b" },
            { "path": secret.to_str().unwrap(), "old_string": "a", "new_string": "b" }
        ]);
        for edits in [entries.clone(), json!(entries.to_string())] {
            let reason = policy
                .before_tool_call(
                    "edit",
                    &json!({ "path": ok.to_str().unwrap(), "edits": edits }),
                    &session,
                    &cancel,
                )
                .await;
            assert!(
                reason.as_deref().is_some_and(|r| r.contains("secrets.env")),
                "got {reason:?}"
            );
        }
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

    async fn deny_reason(policy: &ToolPolicy, path: &std::path::Path) -> Option<String> {
        policy
            .before_tool_call(
                "write",
                &json!({"path": path.to_str().unwrap(), "content": "x"}),
                &agent_core::Session::new(),
                &CancellationToken::new(),
            )
            .await
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_new_file_under_a_symlinked_directory_cannot_dodge_a_deny_glob() {
        // `sl -> secrets`; `sl/new.env` doesn't exist yet, so canonicalize failed and the
        // typed spelling `.../sl/new.env` was matched — never `**/secrets/**`.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("secrets")).unwrap();
        std::os::unix::fs::symlink(dir.path().join("secrets"), dir.path().join("sl")).unwrap();
        let policy = ToolPolicy::new().deny_path("**/secrets/**");

        let reason = deny_reason(&policy, &dir.path().join("sl/new.env")).await;
        assert!(
            reason
                .as_deref()
                .is_some_and(|r| r.contains("denied by policy")),
            "got {reason:?}"
        );
        // Also through a not-yet-created subdirectory, and through `..` after the link.
        assert!(
            deny_reason(&policy, &dir.path().join("sl/deeper/new.env"))
                .await
                .is_some()
        );
        assert!(
            deny_reason(&policy, &dir.path().join("sl/../secrets/x.env"))
                .await
                .is_some()
        );
        // A sibling that really is outside stays allowed.
        assert!(
            deny_reason(&policy, &dir.path().join("public/new.env"))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn an_absolute_deny_pattern_through_a_symlinked_prefix_still_fires() {
        // macOS: `--deny-path /tmp/proj/secrets/**` while every target keys as `/private/tmp/...`.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().canonicalize().unwrap().join("private-tmp");
        std::fs::create_dir_all(real.join("proj/secrets")).unwrap();
        let link = dir.path().join("tmp");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let pattern = format!("{}/proj/secrets/**", link.display());
        let policy = ToolPolicy::new().deny_path(&pattern);

        for target in [
            real.join("proj/secrets/a.env"),
            link.join("proj/secrets/b.env"),
        ] {
            assert!(
                deny_reason(&policy, &target).await.is_some(),
                "{} must be denied by {pattern}",
                target.display()
            );
        }
        // Remote policing never uses a host-resolved twin.
        let remote = ToolPolicy::new()
            .deny_path(&pattern)
            .with_remote(None)
            .denied_paths();
        assert_eq!(remote.len(), 1);
        assert_eq!(remote[0].glob().glob(), pattern);
    }

    #[tokio::test]
    async fn a_case_variant_cannot_dodge_a_deny_glob_on_a_case_insensitive_volume() {
        // `SECRETS/new2.env` *is* `secrets/new2.env` on APFS/NTFS. Only meaningful where the
        // volume really ignores case, so it's skipped elsewhere; the comparison itself is unit-tested
        // below regardless.
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("secrets")).unwrap();
        if !dir.path().join("SECRETS").exists() {
            eprintln!("skipping: {} is case-sensitive", dir.path().display());
            return;
        }
        let policy = ToolPolicy::new().deny_path("**/secrets/**");
        assert!(
            deny_reason(&policy, &dir.path().join("SECRETS/new2.env"))
                .await
                .is_some()
        );
    }

    #[test]
    fn a_deny_rule_matches_case_insensitively_only_when_the_volume_does() {
        let rule = DenyRule::compile("**/Secrets/**", "Secrets/").unwrap();
        assert!(rule.matches("/repo/secrets/new2.env", true));
        assert!(rule.matches("/repo/SECRETS/new2.env", true));
        assert!(!rule.matches("/repo/secrets/new2.env", false));
        assert!(rule.matches("/repo/Secrets/new2.env", false));
    }

    /// Whether a `write` to `path` (as the model would name it) is denied.
    async fn denies(policy: &ToolPolicy, path: &str) -> bool {
        policy
            .before_tool_call(
                "write",
                &json!({"path": path, "content": "x"}),
                &agent_core::Session::new(),
                &CancellationToken::new(),
            )
            .await
            .is_some()
    }

    fn rooted(patterns: &[&str], root: &std::path::Path) -> ToolPolicy {
        let patterns: Vec<String> = patterns.iter().map(|p| p.to_string()).collect();
        ToolPolicy::from_lists(&[], &[], &patterns).with_root(root)
    }

    #[tokio::test]
    async fn a_bare_name_matches_that_name_at_any_depth() {
        let dir = tempfile::tempdir().unwrap();
        let policy = rooted(&[".env"], dir.path());
        assert!(denies(&policy, ".env").await);
        assert!(denies(&policy, "app/config/.env").await);
        let abs = dir.path().join("sub/.env");
        assert!(denies(&policy, abs.to_str().unwrap()).await);
        assert!(!denies(&policy, ".env.example").await);
        assert!(!denies(&policy, "app/my.env").await);
    }

    #[tokio::test]
    async fn a_bare_name_also_covers_everything_beneath_a_directory_of_that_name() {
        let dir = tempfile::tempdir().unwrap();
        let policy = rooted(&["secrets"], dir.path());
        assert!(denies(&policy, "secrets").await);
        assert!(denies(&policy, "secrets/token").await);
        assert!(denies(&policy, "deep/secrets/a/b").await);
        assert!(!denies(&policy, "secretsx/token").await);
    }

    #[tokio::test]
    async fn a_pattern_with_a_slash_is_anchored_to_the_policy_root() {
        let dir = tempfile::tempdir().unwrap();
        let policy = rooted(&["secrets/*"], dir.path());
        assert!(denies(&policy, "secrets/token").await);
        assert!(denies(&policy, "./secrets/token").await);
        let abs = dir.path().join("secrets/token");
        assert!(denies(&policy, abs.to_str().unwrap()).await);
        assert!(
            !denies(&policy, "nested/secrets/token").await,
            "an anchored pattern matches only under the root"
        );
        // A leading `./` anchors the same way.
        let policy = rooted(&["./config/prod.yml"], dir.path());
        assert!(denies(&policy, "config/prod.yml").await);
        assert!(!denies(&policy, "x/config/prod.yml").await);
    }

    #[tokio::test]
    async fn a_trailing_slash_means_the_directory_and_everything_beneath_it() {
        let dir = tempfile::tempdir().unwrap();
        let policy = rooted(&["secrets/"], dir.path());
        assert!(denies(&policy, "secrets/token").await);
        assert!(denies(&policy, "secrets/a/b/c").await);
        assert!(
            denies(&policy, "deep/secrets/token").await,
            "no slash but the trailing one: any depth"
        );
        assert!(!denies(&policy, "secrets.txt").await);
        let policy = rooted(&["config/keys/"], dir.path());
        assert!(denies(&policy, "config/keys/a/b").await);
        assert!(!denies(&policy, "x/config/keys/a").await);
    }

    #[tokio::test]
    async fn an_absolute_pattern_stays_absolute_and_globs_starting_with_a_star_match_anywhere() {
        let dir = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let abs = format!("{}/locked/*", other.path().display());
        let policy = rooted(&[&abs, "**/vault/**", "*.pem"], dir.path());
        assert!(denies(&policy, other.path().join("locked/a").to_str().unwrap()).await);
        assert!(!denies(&policy, "locked/a").await);
        assert!(denies(&policy, other.path().join("x/vault/y").to_str().unwrap()).await);
        assert!(denies(&policy, "deep/key.pem").await);
    }

    #[tokio::test]
    async fn a_root_set_after_the_patterns_still_anchors_them() {
        // `from_lists` runs before `with_root` at every construction site.
        let dir = tempfile::tempdir().unwrap();
        let policy = ToolPolicy::from_lists(&[], &[], &["secrets/*".into()]).with_root(dir.path());
        assert!(denies(&policy, "secrets/a").await);
        // And a pattern added after the root is anchored to it too.
        let policy = ToolPolicy::new().with_root(dir.path()).deny_path("keys/*");
        assert!(denies(&policy, "keys/a").await);
    }

    #[tokio::test]
    async fn an_unrooted_policy_anchors_to_the_process_cwd() {
        let policy = ToolPolicy::from_lists(&[], &[], &["no-such-dir-xyz/*".into(), ".env".into()]);
        assert!(denies(&policy, "no-such-dir-xyz/a").await);
        assert!(denies(&policy, "no-such-dir-abc/.env").await);
        assert!(!denies(&policy, "elsewhere/no-such-dir-xyz/a").await);
    }

    #[tokio::test]
    async fn a_remote_policy_anchors_lexically_to_its_root() {
        let policy = ToolPolicy::from_lists(&[], &[], &["secrets/*".into(), ".env".into()])
            .with_root("/workspace")
            .with_remote(Some("/home/agent".into()));
        assert!(denies(&policy, "secrets/a").await);
        assert!(denies(&policy, "/workspace/secrets/a").await);
        assert!(denies(&policy, "/workspace/app/.env").await);
        assert!(!denies(&policy, "/elsewhere/secrets/a").await);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn an_anchored_pattern_fires_through_a_symlinked_root() {
        // The root is spelled through a symlink; targets key on the resolved path. The anchored
        // pattern gets a resolved twin just as an absolute one does.
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().canonicalize().unwrap().join("real");
        std::fs::create_dir_all(real.join("secrets")).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let policy = rooted(&["secrets/*"], &link);
        assert!(denies(&policy, "secrets/a").await);
        assert!(denies(&policy, real.join("secrets/b").to_str().unwrap()).await);
    }

    #[test]
    fn the_merge_back_matchers_anchor_relative_patterns_at_the_tree_they_are_given() {
        // A worktree child's policy is rooted at its worktree; merge-back re-checks the patch in
        // the main tree, so the anchored patterns must be re-anchored there.
        let wt = tempfile::tempdir().unwrap();
        let main = tempfile::tempdir().unwrap();
        let policy = rooted(&["secrets/*", ".env"], wt.path());
        let matchers = policy.denied_paths_at(main.path());
        let hit = |p: &str| matchers.iter().any(|m| m.is_match(p));
        assert!(hit(&main.path().join("secrets/a").display().to_string()));
        assert!(hit(&main.path().join("app/.env").display().to_string()));
        assert!(hit("app/.env"));
        assert!(!hit(&main.path().join("x/secrets/a").display().to_string()));
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
