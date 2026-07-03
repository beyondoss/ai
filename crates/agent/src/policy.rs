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
use serde_json::Value;
use std::collections::HashSet;

/// A static tool-call gate: block a call outright by tool name, or (for `bash` specifically) by a
/// substring match against the command text. Case-insensitive on the bash-pattern side — a policy
/// author writing `rm -rf` shouldn't have to also list `RM -RF`.
#[derive(Debug, Default, Clone)]
pub struct ToolPolicy {
    denied_tools: HashSet<String>,
    /// Lowercased once at construction time so `before_tool_call` never re-lowercases the pattern list
    /// on every single call — only the (much shorter-lived) command string gets lowercased per call.
    denied_bash_patterns: Vec<String>,
}

impl ToolPolicy {
    /// An empty policy — every call allowed. Prefer [`ToolPolicy::is_empty`] over constructing this
    /// and calling `.with_hooks` unconditionally: an empty policy is behaviorally identical to
    /// `NoHooks`, just paying an extra vtable indirection per call for no reason.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from `--deny-tool`/`--deny-bash-pattern`-shaped lists — shared by `run` and `serve` so the
    /// two flags mean exactly the same thing regardless of which subcommand installs them.
    pub fn from_lists(deny_tool: &[String], deny_bash_pattern: &[String]) -> Self {
        let mut policy = Self::new();
        for name in deny_tool {
            policy = policy.deny_tool(name.clone());
        }
        for pattern in deny_bash_pattern {
            policy = policy.deny_bash_pattern(pattern.clone());
        }
        policy
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

    /// Whether this policy would ever actually block anything — a caller builds one unconditionally
    /// from CLI flags/env, so this is the one-time check that decides whether installing the hook is
    /// worth the per-call overhead at all.
    pub fn is_empty(&self) -> bool {
        self.denied_tools.is_empty() && self.denied_bash_patterns.is_empty()
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
