//! Agent hooks — the loop's interception seam.
//!
//! Hooks let a host gate and rewrite tool calls without forking the loop: [`before_tool_call`] can
//! deny a call (a permission system), and [`after_tool_call`] can rewrite its result.
//! [`on_assistant_message`] does the same for the model's own generated content — pi's `message_end`
//! extension event, narrowed to the assistant-authored case. All default to no-ops, so an agent without
//! hooks behaves exactly as before. This is the minimal version of pi's richer hook surface — the
//! pieces a headless, gateway-fronted server actually needs.
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
//! [`on_assistant_message`]: AgentHooks::on_assistant_message

use crate::message::ImageSource;
use crate::session::Session;
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
    /// `session` is the live conversation as of the instant this call was dispatched — by the time
    /// this fires, the requesting assistant turn (the `tool_use` block this call came from) is already
    /// `session.messages.last()` (see `Agent::run_events_steered`'s pre-dispatch checkpoint, which
    /// establishes the same invariant), and the rest of the history is visible too — pi's
    /// `BeforeToolCallContext` carries the same (`assistantMessage`/`context`), so a policy can condition
    /// its decision on the surrounding conversation, not just this one call's name/args in isolation
    /// (e.g. "block `bash` only if the assistant didn't explain why in its own message", or "flag a
    /// repeated identical call"). Read-only: a hook can't mutate history from here, only gate the call.
    ///
    /// `cancel` is the run's cancellation token: a hook doing its own (possibly slow) I/O — an
    /// external permission check, say — can check or await it directly to bail out promptly, rather
    /// than relying solely on being cut off by the loop dropping its future mid-await.
    async fn before_tool_call(
        &self,
        _name: &str,
        _input: &Value,
        _session: &Session,
        _cancel: &CancellationToken,
    ) -> Option<String> {
        None
    }

    /// Called after a tool produced `(output, images, is_error)`. Return the (possibly rewritten)
    /// result to feed back to the model — e.g. redact secrets, cap size, reclassify success/failure, or
    /// strip/replace an image the tool returned (a screenshot a `read` call pulled off disk, say).
    /// `images` is empty for the overwhelming majority of tool calls (only `read` on an image file
    /// populates it today), so a hook that only cares about text can ignore the parameter entirely and
    /// pass it straight through, same as the default implementation does.
    ///
    /// See [`before_tool_call`](Self::before_tool_call) for what `session`/`cancel` are for.
    // 8 independent inputs (name/input/output/images/is_error/session/cancel), each one a caller
    // already has in scope and a hook implementation may need to inspect on its own — bundling them
    // into a params struct would just be a second place those same fields live, not a reduction in
    // what the hook needs to know, and every existing override here only touches 2-3 of them anyway
    // (the rest stay `_`-prefixed and unused).
    #[allow(clippy::too_many_arguments)]
    async fn after_tool_call(
        &self,
        _name: &str,
        _input: &Value,
        output: String,
        images: Vec<ImageSource>,
        is_error: bool,
        _session: &Session,
        _cancel: &CancellationToken,
    ) -> (String, Vec<ImageSource>, bool) {
        (output, images, is_error)
    }

    /// Called at every turn boundary the loop would otherwise continue past — right after a turn's
    /// tool calls (if any) finished and their results were committed, but before the next model call
    /// would start. Return `true` to end the run here, gracefully (the same shape a `false` from
    /// [`crate::steering::Steering::request_stop`] produces — this is an *additional*, content-aware
    /// way to request the same stop, not a replacement for it; the loop honors either). A beyond-only
    /// capability, not a literal port: pi's own `shouldStopAfterTurn` config field (`agent-loop.ts`,
    /// `types.ts`) is real, but it's dropped at the `Agent` wrapper class pi's real shipped product
    /// actually constructs (`packages/coding-agent/src/core/sdk.ts`'s `new Agent({...})`) — that
    /// class's `AgentOptions`/`createLoopConfig` never forwards it to `runAgentLoop`, so no post-turn
    /// stop hook is reachable from pi's real product at all. The only place it's ever wired end-to-end
    /// is the unused `packages/agent/src/harness/agent-harness.ts` (zero references from
    /// `packages/coding-agent`), which is not what pi ships.
    ///
    /// `message` is the assistant turn that just completed (its blocks are already committed to
    /// `session`); `tool_results` is that turn's own tool_result blocks, empty when the turn made no
    /// tool calls. `session` is the full conversation as of this instant, for a decision that needs
    /// more context than just this one turn (e.g. "stop once the last 3 turns all touched the same
    /// file"). Defaults to `false` (never request a stop) — a caller happy with only the external,
    /// `Steering`-driven stop mechanism never needs to implement this.
    async fn should_stop_after_turn(
        &self,
        _message: &crate::message::Message,
        _tool_results: &[crate::message::ContentBlock],
        _session: &Session,
        _cancel: &CancellationToken,
    ) -> bool {
        false
    }

    /// Called with the assistant's own finalized message — after the model finishes streaming, before
    /// it's committed to `session.messages`, checkpointed, or returned to the caller in
    /// [`crate::agent::AgentEvent::TurnEnd`]. Return a rewritten message to redact or otherwise modify
    /// its content (e.g. scrubbing a secret/credential the model echoed back verbatim); return it
    /// unchanged (the default) to leave the turn as-is. Mirrors pi's `message_end` extension event
    /// narrowed to the assistant-authored case — [`before_tool_call`](Self::before_tool_call) gates a
    /// pending call and [`after_tool_call`](Self::after_tool_call) rewrites a tool's own result; this is
    /// the analogous seam for the model's own generated text.
    ///
    /// The returned message's role is expected to stay [`crate::message::Role::Assistant`] — matching
    /// pi's own "the replacement must keep the original message role" contract — a caller returning any
    /// other role is a caller bug; the loop discards a role-mismatched replacement and keeps the
    /// original rather than risk splicing a wrong-role message into the transcript. A panicking hook
    /// also falls back to the original message rather than losing the turn's content or crashing the
    /// run — same "fails open" convention [`should_stop_after_turn`](Self::should_stop_after_turn) uses.
    async fn on_assistant_message(
        &self,
        message: crate::message::Message,
        _session: &Session,
        _cancel: &CancellationToken,
    ) -> crate::message::Message {
        message
    }

    /// Called immediately before a provider request is sent, once per attempt (including every
    /// mid-stream retry — see `Agent::run_turn`), with the fully-built request. The seam a host uses to
    /// inject a per-request header's *logical* equivalent at this layer (a stop sequence, a sampling
    /// tweak — an actual wire header belongs in [`crate::client::GatewayClient::with_extra_headers`],
    /// one layer below where a `ModelRequest` even exists), or otherwise patch what's about to be sent
    /// before `client.rs` builds the dialect-specific wire body from it. A beyond-only capability, not
    /// a literal port: pi's real shipped product (`packages/coding-agent/src/core/sdk.ts:332-338`) has
    /// exactly one pre-send hook, `onPayload` — wired to the extension event confusingly also named
    /// `"before_provider_request"` — and it only ever sees the literal dialect-specific wire JSON (the
    /// same layer [`before_provider_payload`](Self::before_provider_payload) below operates at), never
    /// an earlier, abstract, dialect-agnostic request. There is no real-product equivalent of this
    /// method's `ModelRequest`-level seam; the only place one existed was the unused
    /// `packages/agent/src/harness/agent-harness.ts:251-320`, never wired into anything
    /// `packages/coding-agent` actually ships. Mutates `req` in place; defaults to a
    /// no-op. A panicking hook has its (possibly partial) mutation discarded — the loop falls back to
    /// the request exactly as it was before this call, the same "fails open" convention
    /// [`on_assistant_message`](Self::on_assistant_message) uses.
    async fn before_provider_request(&self, _req: &mut crate::transport::ModelRequest) {}

    /// Called once the dialect-specific wire body is fully built from the (possibly
    /// [`before_provider_request`](Self::before_provider_request)-patched) [`crate::transport::ModelRequest`],
    /// immediately before it's sent — [`crate::client::GatewayClient::stream`], right after
    /// `dialect.build_body(..)` runs, once per attempt (same per-attempt/per-retry cadence as
    /// `before_provider_request`). **Unlike `before_provider_request`'s abstract, dialect-agnostic
    /// `ModelRequest`, `payload` here is the literal wire-format JSON about to cross the wire** —
    /// Anthropic's `{"model": ..., "messages": [...], "tools": [...]}` shape, OpenAI's
    /// Responses/Chat-Completions shape, whichever dialect this turn's model resolved to. This is the
    /// seam a host uses when it needs to see or rewrite the actual bytes on the wire — a dialect-specific
    /// field `ModelRequest` has no abstraction for — rather than the higher-level request
    /// `before_provider_request` exposes one layer up.
    ///
    /// Mirrors pi's real (single-layer) version of this seam: `packages/coding-agent/src/core/sdk.ts`'s
    /// `onPayload`, passed straight to its HTTP-streaming function and wired to the extension event
    /// `"before_provider_request"` (`extensions/runner.ts`'s `emitBeforeProviderRequest`) — despite the
    /// name collision with [`before_provider_request`](Self::before_provider_request) above, `onPayload`
    /// operates on the literal wire JSON, which is what this method mirrors. The unused
    /// `packages/agent/src/harness/agent-harness.ts` additionally exposes its own, differently-shaped
    /// `beforeProviderPayload` (`emitBeforeProviderPayload`, `BeforeProviderPayloadEvent`/
    /// `BeforeProviderPayloadResult` in `types.ts`, which can replace the payload object wholesale) —
    /// but that harness is dead code (zero references from `packages/coding-agent`), not a second layer
    /// pi's real product actually has. This mutates `payload` in place instead of returning a
    /// replacement — matching `before_provider_request`'s own convention rather than giving this trait
    /// two different "rewrite a request" shapes.
    ///
    /// Defaults to a no-op. A panicking hook has its (possibly partial) mutation discarded —
    /// [`crate::client::GatewayClient::stream`] falls back to the payload exactly as `build_body` produced
    /// it, the same "fails open" convention [`before_provider_request`](Self::before_provider_request)
    /// uses.
    async fn before_provider_payload(&self, _payload: &mut Value) {}

    /// Called once a provider response's raw HTTP status and headers are known, before its body starts
    /// streaming — pi's real `onResponse` (`packages/coding-agent/src/core/sdk.ts:340-346`), wired to
    /// the extension event `"after_provider_response"`. Read-only observability
    /// (a host logging rate-limit headers, say): the response is already normalized into `StreamEvent`s
    /// the loop consumes directly, so this isn't a rewrite seam the way [`after_tool_call`]
    /// (Self::after_tool_call) is. `headers` is `(name, value)` pairs in wire order, a plain,
    /// transport-agnostic shape rather than a `reqwest`-specific type — this hook stays implementable
    /// without any implementor depending on `reqwest`, matching
    /// [`crate::transport::ModelTransport`]'s own "the loop never depends on reqwest" contract. Called
    /// from [`crate::client::GatewayClient::stream`] specifically (the one transport that actually talks
    /// HTTP) rather than from the loop itself — a `MockTransport`-driven test never fires this, since
    /// there is no real response to report. Defaults to a no-op.
    async fn after_provider_response(&self, _status: u16, _headers: &[(String, String)]) {}
}

/// The default: every hook is a no-op. Used when no hooks are configured.
pub struct NoHooks;

#[async_trait]
impl AgentHooks for NoHooks {}

/// Called every time the session reaches a durable, resumable point — before the very first request of
/// a fresh run (so the user's own just-submitted prompt is persisted before the model is ever called),
/// after a tool round-trip's results are recorded, after a tool-less assistant reply, or when a steered/
/// follow-up message is injected — points a host can persist from without ever writing a message half
/// of a `tool_use`/`tool_result` pair (see the call sites in [`crate::agent::Agent::run_events_steered`]
/// for exactly which points those are). Without this, a run is only ever durable once the *entire* run
/// finishes: a crash, OOM-kill, or panic anywhere in between loses everything back to the last
/// checkpoint, which used to include the pathological case of losing the user's own prompt outright
/// (a crash before the first tool round-trip, or during a run that never calls a tool at all) — this
/// call fires often enough that neither gap exists anymore.
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
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> Option<String> {
            (name == "bash").then(|| "bash is not allowed in this context".to_string())
        }
    }

    #[tokio::test]
    async fn before_tool_call_can_block() {
        let hooks = DenyBash;
        let session = Session::new();
        let cancel = CancellationToken::new();
        assert!(
            hooks
                .before_tool_call("bash", &json!({}), &session, &cancel)
                .await
                .is_some()
        );
        assert!(
            hooks
                .before_tool_call("read", &json!({}), &session, &cancel)
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn default_hooks_allow_and_passthrough() {
        let h = NoHooks;
        let session = Session::new();
        let cancel = CancellationToken::new();
        assert!(
            h.before_tool_call("bash", &json!({}), &session, &cancel)
                .await
                .is_none()
        );
        let (out, images, err) = h
            .after_tool_call(
                "x",
                &json!({}),
                "ok".into(),
                Vec::new(),
                false,
                &session,
                &cancel,
            )
            .await;
        assert_eq!((out.as_str(), images.len(), err), ("ok", 0, false));

        let msg = crate::message::Message::assistant(vec![crate::message::ContentBlock::text(
            "secret-token-123",
        )]);
        let unchanged = h.on_assistant_message(msg.clone(), &session, &cancel).await;
        assert_eq!(unchanged, msg);

        // Task #14 (pi-parity): the default provider-request/response pair must be true no-ops too —
        // `before_provider_request` leaves `req` untouched, `after_provider_response` does nothing
        // observable (it has no return value to assert on; just confirm it doesn't panic/block).
        let mut req = crate::transport::ModelRequest::new("claude-test", Vec::new(), 100);
        let before = req.clone();
        h.before_provider_request(&mut req).await;
        assert_eq!(req.model, before.model);
        assert_eq!(req.system, before.system);
        h.after_provider_response(200, &[("x-test".to_string(), "1".to_string())])
            .await;

        // pi-parity gap: no seam previously exposed the literal wire-format JSON payload (only the
        // abstract `ModelRequest`) — the default must be a true no-op here too.
        let mut payload = json!({"model": "claude-test", "messages": []});
        let before = payload.clone();
        h.before_provider_payload(&mut payload).await;
        assert_eq!(payload, before);
    }

    struct InjectsPayloadMarker;
    #[async_trait]
    impl AgentHooks for InjectsPayloadMarker {
        async fn before_provider_payload(&self, payload: &mut Value) {
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("_injected_marker".to_string(), json!(true));
            }
        }
    }

    #[tokio::test]
    async fn before_provider_payload_hook_can_rewrite_the_literal_wire_body() {
        // pi-parity fix: unlike `before_provider_request` (which only ever sees the abstract
        // `ModelRequest`), this hook must see and be able to mutate the fully dialect-built JSON body —
        // pi's `onPayload`/`beforeProviderPayload` seam.
        let hooks = InjectsPayloadMarker;
        let mut payload = json!({"model": "claude-test", "messages": []});
        hooks.before_provider_payload(&mut payload).await;
        assert_eq!(payload["_injected_marker"], json!(true));
        // The rest of the payload must survive untouched.
        assert_eq!(payload["model"], json!("claude-test"));
    }

    struct InjectsHeaderNote;
    #[async_trait]
    impl AgentHooks for InjectsHeaderNote {
        async fn before_provider_request(&self, req: &mut crate::transport::ModelRequest) {
            req.system = Some(format!(
                "{} [patched]",
                req.system.clone().unwrap_or_default()
            ));
        }
    }

    #[tokio::test]
    async fn before_provider_request_hook_can_patch_the_outgoing_request() {
        let hooks = InjectsHeaderNote;
        let mut req = crate::transport::ModelRequest::new("claude-test", Vec::new(), 100)
            .with_system("base prompt");
        hooks.before_provider_request(&mut req).await;
        assert_eq!(req.system.as_deref(), Some("base prompt [patched]"));
    }

    type SeenResponse = (u16, Vec<(String, String)>);
    struct RecordsResponseMeta {
        seen: std::sync::Mutex<Option<SeenResponse>>,
    }
    #[async_trait]
    impl AgentHooks for RecordsResponseMeta {
        async fn after_provider_response(&self, status: u16, headers: &[(String, String)]) {
            *self.seen.lock().unwrap() = Some((status, headers.to_vec()));
        }
    }

    #[tokio::test]
    async fn after_provider_response_hook_observes_status_and_headers() {
        let hooks = RecordsResponseMeta {
            seen: std::sync::Mutex::new(None),
        };
        hooks
            .after_provider_response(429, &[("retry-after".to_string(), "2".to_string())])
            .await;
        let seen = hooks.seen.lock().unwrap().clone().expect("must be set");
        assert_eq!(seen.0, 429);
        assert_eq!(seen.1, vec![("retry-after".to_string(), "2".to_string())]);
    }

    struct RedactSecrets;
    #[async_trait]
    impl AgentHooks for RedactSecrets {
        async fn on_assistant_message(
            &self,
            mut message: crate::message::Message,
            _session: &Session,
            _cancel: &CancellationToken,
        ) -> crate::message::Message {
            for block in &mut message.content {
                if let crate::message::ContentBlock::Text { text, .. } = block {
                    *text = text.replace("secret-token-123", "[REDACTED]");
                }
            }
            message
        }
    }

    #[tokio::test]
    async fn on_assistant_message_hook_can_rewrite_the_models_own_content() {
        let hooks = RedactSecrets;
        let session = Session::new();
        let cancel = CancellationToken::new();
        let msg = crate::message::Message::assistant(vec![crate::message::ContentBlock::text(
            "here is the secret-token-123 you asked about",
        )]);
        let rewritten = hooks.on_assistant_message(msg, &session, &cancel).await;
        let crate::message::ContentBlock::Text { text, .. } = &rewritten.content[0] else {
            panic!("expected a text block");
        };
        assert_eq!(text, "here is the [REDACTED] you asked about");
        assert_eq!(rewritten.role, crate::message::Role::Assistant);
    }

    struct CancelAwareHook;
    #[async_trait]
    impl AgentHooks for CancelAwareHook {
        async fn before_tool_call(
            &self,
            _name: &str,
            _input: &Value,
            _session: &Session,
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
        let session = Session::new();
        let cancel = CancellationToken::new();
        assert!(
            hooks
                .before_tool_call("anything", &json!({}), &session, &cancel)
                .await
                .is_none(),
            "not cancelled yet — the hook should allow the call"
        );
        cancel.cancel();
        assert_eq!(
            hooks
                .before_tool_call("anything", &json!({}), &session, &cancel)
                .await,
            Some("run was cancelled before this tool started".to_string()),
            "a hook can preemptively bail once it observes cancellation"
        );
    }

    #[tokio::test]
    async fn before_tool_call_hook_can_see_the_requesting_assistant_message() {
        // pi-parity gap: pi's `BeforeToolCallContext` carries the requesting `assistantMessage` and the
        // full `AgentContext`, so a policy can condition on the surrounding conversation, not just this
        // one call's name/args in isolation. Confirms the seam actually exposes it.
        struct InspectsHistory {
            saw_expected_last_message: std::sync::atomic::AtomicBool,
        }
        #[async_trait]
        impl AgentHooks for InspectsHistory {
            async fn before_tool_call(
                &self,
                _name: &str,
                _input: &Value,
                session: &Session,
                _cancel: &CancellationToken,
            ) -> Option<String> {
                let sees_it = session
                    .messages
                    .last()
                    .is_some_and(|m| matches!(&m.content[0], crate::message::ContentBlock::ToolUse { name, .. } if name == "bash"));
                self.saw_expected_last_message
                    .store(sees_it, std::sync::atomic::Ordering::SeqCst);
                None
            }
        }
        let hooks = InspectsHistory {
            saw_expected_last_message: std::sync::atomic::AtomicBool::new(false),
        };
        let mut session = Session::new();
        session.user("run something");
        session.push(crate::message::Message::assistant(vec![
            crate::message::ContentBlock::tool_use("t1", "bash", json!({"command": "echo hi"})),
        ]));
        let cancel = CancellationToken::new();
        hooks
            .before_tool_call("bash", &json!({"command": "echo hi"}), &session, &cancel)
            .await;
        assert!(
            hooks
                .saw_expected_last_message
                .load(std::sync::atomic::Ordering::SeqCst),
            "the hook must see the requesting assistant's tool_use message as session.messages.last()"
        );
    }
}
