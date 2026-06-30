//! Agent hooks — the loop's interception seam.
//!
//! Hooks let a host gate and rewrite tool calls without forking the loop: [`before_tool_call`] can
//! deny a call (a permission system), and [`after_tool_call`] can rewrite its result. Both default to
//! no-ops, so an agent without hooks behaves exactly as before. This is the minimal version of pi's
//! richer hook surface — the pieces a headless, gateway-fronted server actually needs.
//!
//! [`before_tool_call`]: AgentHooks::before_tool_call
//! [`after_tool_call`]: AgentHooks::after_tool_call

use async_trait::async_trait;
use serde_json::Value;

/// Interception points the agent loop calls around each tool invocation. All methods default to
/// no-ops; implement only what you need.
#[async_trait]
pub trait AgentHooks: Send + Sync {
    /// Called before a tool runs. Return `Some(reason)` to **block** the call — the loop feeds the
    /// reason back to the model as an error `tool_result` instead of running the tool. Return `None`
    /// to allow it. This is the seam a permission/approval system hangs off.
    async fn before_tool_call(&self, _name: &str, _input: &Value) -> Option<String> {
        None
    }

    /// Called after a tool produced `(output, is_error)`. Return the (possibly rewritten) result to
    /// feed back to the model — e.g. redact secrets, cap size, or reclassify success/failure.
    async fn after_tool_call(
        &self,
        _name: &str,
        _input: &Value,
        output: String,
        is_error: bool,
    ) -> (String, bool) {
        (output, is_error)
    }
}

/// The default: every hook is a no-op. Used when no hooks are configured.
pub struct NoHooks;

#[async_trait]
impl AgentHooks for NoHooks {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct DenyBash;
    #[async_trait]
    impl AgentHooks for DenyBash {
        async fn before_tool_call(&self, name: &str, _input: &Value) -> Option<String> {
            (name == "bash").then(|| "bash is not allowed in this context".to_string())
        }
    }

    #[tokio::test]
    async fn before_tool_call_can_block() {
        let hooks = DenyBash;
        assert!(hooks.before_tool_call("bash", &json!({})).await.is_some());
        assert!(hooks.before_tool_call("read", &json!({})).await.is_none());
    }

    #[tokio::test]
    async fn default_hooks_allow_and_passthrough() {
        let h = NoHooks;
        assert!(h.before_tool_call("bash", &json!({})).await.is_none());
        let (out, err) = h.after_tool_call("x", &json!({}), "ok".into(), false).await;
        assert_eq!((out.as_str(), err), ("ok", false));
    }
}
