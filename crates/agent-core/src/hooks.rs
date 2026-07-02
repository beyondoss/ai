//! Agent hooks — the loop's interception seam.
//!
//! Hooks let a host gate and rewrite tool calls without forking the loop: [`before_tool_call`] can
//! deny a call (a permission system), and [`after_tool_call`] can rewrite its result. Both default to
//! no-ops, so an agent without hooks behaves exactly as before. This is the minimal version of pi's
//! richer hook surface — the pieces a headless, gateway-fronted server actually needs.
//!
//! **Deliberately narrower than pi's `agent-loop.ts` config in two specific ways**, both confirmed —
//! not just assumed — during a pi-parity test-coverage pass:
//! - pi's `beforeToolCall` can *mutate* `args` in place before dispatch (`agent-loop.test.ts`,
//!   "should execute mutated beforeToolCall args without revalidation"). [`before_tool_call`] here is
//!   allow/deny-only, `&Value` not `&mut Value` — no caller in this codebase (a gateway-fronted
//!   headless server, not an embeddable SDK with third-party tool-authoring) has ever needed a hook to
//!   rewrite a model's own tool-call arguments; a permission gate only ever needs to see them.
//! - pi separately lets a *tool itself* define `prepareArguments` (a per-tool legacy-shape normalizer,
//!   `agent-loop.test.ts`'s "should prepare tool arguments for validation"), and lets `afterToolCall`
//!   force a whole batch to terminate the run (`agent-loop.test.ts`'s "should allow afterToolCall to
//!   mark a tool batch as terminating"). Neither is a global-hook concern here: this codebase's tools
//!   already do their own legacy-input normalization internally (see `crates/agent/src/tools/edit.rs`'s
//!   `parse_edits` folding a legacy `old_string`/`new_string` pair into its `edits` array — the same
//!   shape-normalization job pi's `prepareArguments` does, just localized to the one tool that needs it
//!   instead of exposed as a cross-cutting seam), and run-termination is a *tool's own result*
//!   ([`crate::tool::ToolOutput::terminate`]) reduced across a batch by the loop itself
//!   (`Agent::run_events_steered`'s `terminate &= wants_terminate`), not a separate hook layer bolted
//!   on afterward.
//!
//! [`before_tool_call`]: AgentHooks::before_tool_call
//! [`after_tool_call`]: AgentHooks::after_tool_call

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

/// Interception points the agent loop calls around each tool invocation. All methods default to
/// no-ops; implement only what you need.
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Called before a tool runs. Return `Some(reason)` to **block** the call — the loop feeds the
    /// reason back to the model as an error `tool_result` instead of running the tool. Return `None`
    /// to allow it. This is the seam a permission/approval system hangs off.
    ///
    /// `cancel` is the run's cancellation token: a hook doing its own (possibly slow) I/O — an
    /// external permission check, say — can check or await it directly to bail out promptly, rather
    /// than relying solely on being cut off by the loop dropping its future mid-await.
    async fn before_tool_call(
        &self,
        _name: &str,
        _input: &Value,
        _cancel: &CancellationToken,
    ) -> Option<String> {
        None
    }

    /// Called after a tool produced `(output, is_error)`. Return the (possibly rewritten) result to
    /// feed back to the model — e.g. redact secrets, cap size, or reclassify success/failure.
    ///
    /// See [`before_tool_call`](Self::before_tool_call) for what `cancel` is for.
    async fn after_tool_call(
        &self,
        _name: &str,
        _input: &Value,
        output: String,
        is_error: bool,
        _cancel: &CancellationToken,
    ) -> (String, bool) {
        (output, is_error)
    }
}

/// The default: every hook is a no-op. Used when no hooks are configured.
pub struct NoHooks;

#[async_trait]
impl AgentHooks for NoHooks {}

/// Called when the session reaches a durable, resumable checkpoint mid-run — after a tool round-trip's
/// results are recorded, or a steered/follow-up message is injected — points a host can persist from
/// without ever writing a message half of a `tool_use`/`tool_result` pair (see the call sites in
/// [`crate::agent::Agent::run_events_steered`] for exactly which points those are). Without this, a
/// multi-step run (several tool round-trips) is only ever durable once the *entire* run finishes: a
/// crash, OOM-kill, or panic mid-run loses everything back to the turn's start, including the user's
/// own prompt.
///
/// Async (unlike [`AgentHooks`]'s tool-interception methods, which stay on the hot per-call path) so a
/// host can perform its own blocking I/O — appending to a session file — off of whatever executor it
/// runs on (e.g. via `tokio::task::spawn_blocking`) without this crate depending on a specific one.
/// Defaults to a no-op; implement only if incremental persistence matters to you.
#[async_trait]
pub trait CheckpointHook: Send + Sync {
    async fn checkpoint(&self, _session: &crate::session::Session) {}
}

/// The default: checkpoints are no-ops. A caller happy with "only ever persisted once the run
/// completes" (or one with no persistence at all) never needs to configure anything else.
pub struct NoCheckpoint;

#[async_trait]
impl CheckpointHook for NoCheckpoint {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct DenyBash;
    #[async_trait]
    impl AgentHooks for DenyBash {
        async fn before_tool_call(
            &self,
            name: &str,
            _input: &Value,
            _cancel: &CancellationToken,
        ) -> Option<String> {
            (name == "bash").then(|| "bash is not allowed in this context".to_string())
        }
    }

    #[tokio::test]
    async fn before_tool_call_can_block() {
        let hooks = DenyBash;
        let cancel = CancellationToken::new();
        assert!(
            hooks
                .before_tool_call("bash", &json!({}), &cancel)
                .await
                .is_some()
        );
        assert!(
            hooks
                .before_tool_call("read", &json!({}), &cancel)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn default_hooks_allow_and_passthrough() {
        let h = NoHooks;
        let cancel = CancellationToken::new();
        assert!(
            h.before_tool_call("bash", &json!({}), &cancel)
                .await
                .is_none()
        );
        let (out, err) = h
            .after_tool_call("x", &json!({}), "ok".into(), false, &cancel)
            .await;
        assert_eq!((out.as_str(), err), ("ok", false));
    }

    struct CancelAwareHook;
    #[async_trait]
    impl AgentHooks for CancelAwareHook {
        async fn before_tool_call(
            &self,
            _name: &str,
            _input: &Value,
            cancel: &CancellationToken,
        ) -> Option<String> {
            cancel
                .is_cancelled()
                .then(|| "run was cancelled before this tool started".to_string())
        }
    }

    #[tokio::test]
    async fn before_tool_call_hook_can_observe_cancellation() {
        let hooks = CancelAwareHook;
        let cancel = CancellationToken::new();
        assert!(
            hooks
                .before_tool_call("anything", &json!({}), &cancel)
                .await
                .is_none(),
            "not cancelled yet — the hook should allow the call"
        );
        cancel.cancel();
        assert_eq!(
            hooks
                .before_tool_call("anything", &json!({}), &cancel)
                .await,
            Some("run was cancelled before this tool started".to_string()),
            "a hook can preemptively bail once it observes cancellation"
        );
    }
}
