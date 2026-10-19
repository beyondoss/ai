//! The questions a session asks its client and waits on: tool approval, MCP elicitation and sampling, and the
//! login callbacks, each with the pending map the answering command resolves against.

use crate::serve::*;

/// The manual-code-paste channel for an in-flight `login`: `ServeLoginCallbacks::prompt_text` parks a
/// `oneshot::Sender` here and awaits its receiver; the `"submit_code"` dispatch arm (running
/// concurrently in the same process's main idle loop — see the `"login"` arm) takes it and sends the
/// pasted code, waking the parked `login`.
pub(super) type PendingCodeSlot =
    Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<String>>>>;

/// Tracks the one `login` allowed in flight at a time — see `pending_login`'s own declaration.
///
/// `generation` is a monotonic tag stamped when this login claimed the slot. It's what lets the
/// [`PendingLoginGuard`] tell "the slot still holds *my* login" from "a newer login has since taken it"
/// on drop, so a task winding down late can't wipe a successor's slot — the race that made a subsequent
/// `abort_login` silently no-op (the login it should have cancelled then never resolves). See
/// [`PendingLoginGuard`].
pub(super) struct PendingLogin {
    pub(super) generation: u64,
    pub(super) cancel: agent_core::CancellationToken,
    pub(super) pending_code: PendingCodeSlot,
}

/// Resets `pending_login` back to `None` when the detached login task's future ends, for any
/// reason — including a panic unwinding through it. The login task's own JoinHandle is discarded
/// (nothing awaits it), so without this guard a panic anywhere in `crate::oauth::login` or the
/// credential save would skip the task's normal end-of-body reset and leave `pending_login`
/// permanently `Some`: every subsequent `login` call would return "busy: a login is already in
/// flight" forever, with no way to recover short of restarting the whole `serve` process (even
/// `abort_login` can't help — it just cancels a token nothing is left alive to observe). A `Drop`
/// impl runs during unwinding, unlike a plain line of cleanup code placed after the panicking call,
/// so this holds regardless of where in the task body things go wrong.
///
/// **Clears only its own generation.** `abort_login` clears the slot *synchronously* (so the next
/// `login` is accepted the instant the abort is acknowledged, not whenever this detached task happens to
/// wind down), which means by the time this guard runs a *newer* login may already own the slot. Wiping
/// it unconditionally would drop that newer login's marker — the next `abort_login` would then find an
/// empty slot and no-op, leaving the newer login uncancellable and its client hung waiting for a terminal
/// response that never comes. Comparing generations makes the guard a no-op unless the slot is still this
/// login's.
pub(super) struct PendingLoginGuard {
    pub(super) slot: Arc<std::sync::Mutex<Option<PendingLogin>>>,
    pub(super) generation: u64,
}

impl Drop for PendingLoginGuard {
    fn drop(&mut self) {
        let mut slot = lock_ignoring_poison(&self.slot);
        if slot
            .as_ref()
            .is_some_and(|p| p.generation == self.generation)
        {
            *slot = None;
        }
    }
}

/// Outstanding approval questions, keyed by request id.
///
/// A **map**, not the single slot `pending_login` gets away with: the loop dispatches tool groups
/// `buffer_unordered`, so several `before_tool_call` hooks can await concurrently, and a `subagent`
/// fan-out multiplies that again. One `oneshot` per question.
pub(super) type PendingApprovals =
    Arc<std::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<ApprovalDecision>>>>;

/// Clears a pending approval's slot when its `request` call returns — normally, on timeout, on abort, or
/// through a panic. The `PendingLoginGuard` idea, keyed.
pub(super) struct PendingApprovalGuard {
    pub(super) pending: PendingApprovals,
    pub(super) id: String,
}

impl Drop for PendingApprovalGuard {
    fn drop(&mut self) {
        lock_ignoring_poison(&self.pending).remove(&self.id);
    }
}

/// `serve`'s [`ApprovalGate`]: broadcast the question to every attached client, park a `oneshot`, and
/// let whichever client answers first resolve it — the same ack-now/respond-later shape
/// [`ServeLoginCallbacks`] uses for a pasted OAuth code, generalized to N concurrent questions and N
/// concurrent answerers.
///
/// Broadcasts through `out` (the fan-out) rather than the session's `out_tx` writer channel, because the
/// gate is constructed before that channel exists — and because `broadcast` *is* the multi-attach
/// primitive. A control frame may therefore interleave with streamed `event` frames; that is fine, and
/// the client correlates by `request_id` regardless.
pub(crate) struct ServeApprovalGate {
    pub(super) out: SharedOutConn,
    pub(super) pending: PendingApprovals,
    /// Width-1: the user is asked **one question at a time**, even when eight tool calls (or eight
    /// subagent children sharing this gate) block at once.
    ///
    /// Deliberately not `ToolExecutionMode::Sequential`: one sequential-requesting call reroutes the
    /// loop's *entire* batch through its interleaved dispatch path, which is a cross-cutting change to
    /// execution semantics in service of a UI concern. Approved calls still run concurrently afterward.
    pub(super) serialize: tokio::sync::Mutex<()>,
    /// `None` waits forever. A finite default exists because `running` stays `true` for the whole
    /// prompt, so an unanswered question on a session whose clients all detached mid-wait would
    /// otherwise pin it indefinitely (an `abort` still frees it either way).
    pub(super) timeout: Option<std::time::Duration>,
    pub(super) seq: AtomicU64,
    pub(super) session_id: String,
}

impl ServeApprovalGate {
    pub(crate) fn new(
        out: SharedOutConn,
        timeout: Option<std::time::Duration>,
        session_id: impl Into<String>,
    ) -> (Arc<Self>, PendingApprovals) {
        let pending: PendingApprovals = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let gate = Arc::new(Self {
            out,
            pending: pending.clone(),
            serialize: tokio::sync::Mutex::new(()),
            timeout,
            seq: AtomicU64::new(0),
            session_id: session_id.into(),
        });
        (gate, pending)
    }

    pub(super) fn broadcast(&self, frame: OutFrame) {
        lock_ignoring_poison(&self.out).broadcast(frame);
    }

    pub(super) fn no_client(&self) -> bool {
        lock_ignoring_poison(&self.out).is_empty()
    }
}

/// Resolve a parked approval. Returns `false` when the id is unknown or already answered — which is how
/// the *losing* client in a multi-attach race learns its answer arrived too late.
pub(crate) fn resolve_approval(
    pending: &PendingApprovals,
    request_id: &str,
    decision: ApprovalDecision,
) -> bool {
    let sender = lock_ignoring_poison(pending).remove(request_id);
    match sender {
        Some(tx) => tx.send(decision).is_ok(),
        None => false,
    }
}

#[async_trait::async_trait]
impl crate::approval::ApprovalGate for ServeApprovalGate {
    async fn request(
        &self,
        req: crate::approval::ApprovalRequest,
        cancel: &CancellationToken,
    ) -> Result<ApprovalDecision, ApprovalError> {
        // One question at a time. An abort while queued behind another question must not wait for it.
        let _one_at_a_time = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(ApprovalError::Cancelled),
            guard = self.serialize.lock() => guard,
        };
        if cancel.is_cancelled() {
            return Err(ApprovalError::Cancelled);
        }
        // Fail closed rather than hang: a gate that silently executes privileged operations with nobody
        // watching is not a gate, and a detached background session has nobody watching.
        if self.no_client() {
            return Err(ApprovalError::NoClient);
        }

        let request_id = format!(
            "{}:{}",
            self.session_id,
            self.seq.fetch_add(1, Ordering::Relaxed)
        );
        let (tx, rx) = tokio::sync::oneshot::channel();
        lock_ignoring_poison(&self.pending).insert(request_id.clone(), tx);
        let _slot = PendingApprovalGuard {
            pending: self.pending.clone(),
            id: request_id.clone(),
        };

        self.broadcast(approval_request_frame(&request_id, &req));

        // `biased`, so a photo-finish resolves the only way that is defensible in both directions: an
        // abort outranks a human's answer (never run a tool after the run was cancelled), and a human's
        // answer outranks the timeout (they *did* answer, and `resolve_approval` already told them it was
        // accepted). Left unbiased, `select!` picks at random and a client could be told `accepted:true`
        // for a call the timeout then denied.
        let outcome = tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(ApprovalError::Cancelled),
            answer = rx => answer.map_err(|_| ApprovalError::Cancelled),
            _ = sleep_opt(self.timeout) => Err(ApprovalError::TimedOut),
        };
        // Every other attached client is still showing this question. Tell them it's answered.
        self.broadcast(approval_resolved_frame(&request_id, outcome));
        outcome
    }
}

/// Marks a session busy — its `running` flag, which the daemon's idle reaper and drain read — for as
/// long as a host `bash` command or a manual `compact` is in flight, and clears it on drop.
///
/// A `prompt` manages the flag itself (it spans retries and keeps `agent_runs_in_flight` in step).
/// These two commands are not runs, so the gauge is untouched, but their work is just as lost if
/// the session is reaped or drained out from under them: a client that starts a long host `bash`
/// and detaches, or an HTTP POST past its reply timeout, has nothing else keeping the session alive.
/// `None` when the flag was already set, so an enclosing owner's clear is the one that counts.
pub(super) struct BusyGuard<'a>(&'a std::sync::atomic::AtomicBool);

impl<'a> BusyGuard<'a> {
    pub(super) fn mark(flag: &'a std::sync::atomic::AtomicBool) -> Option<Self> {
        (!flag.swap(true, Ordering::Relaxed)).then_some(Self(flag))
    }
}

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

/// `sleep(d)`, or a future that never resolves when `d` is `None`.
pub(super) async fn sleep_opt(duration: Option<std::time::Duration>) {
    match duration {
        Some(d) => tokio::time::sleep(d).await,
        None => std::future::pending().await,
    }
}

/// `{type:"approval_request", …}` — a tool call waiting on a human. Broadcast to every attached client.
pub(super) fn approval_request_frame(
    request_id: &str,
    req: &crate::approval::ApprovalRequest,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("approval_request"));
    m.insert("request_id".into(), json!(request_id));
    m.insert("tool".into(), json!(req.tool));
    m.insert("summary".into(), req.summary.clone());
    m.insert("scope_key".into(), json!(req.scope_key));
    m.insert("origin".into(), req.origin.to_json());
    if let Some(context) = &req.context {
        m.insert("mcp_skill".into(), context.clone());
    }
    m.insert(
        "options".into(),
        json!(["allow_once", "allow_session", "deny_once", "deny_session"]),
    );
    Value::Object(m).into()
}

/// `{type:"approval_resolved", …}` — this question is settled, however it settled.
///
/// No OAuth analogue: a login is a single-client RPC, but an approval fans out to every attached client
/// and only one of them answers. Without this the others keep a dead prompt on screen forever.
pub(super) fn approval_resolved_frame(
    request_id: &str,
    outcome: Result<ApprovalDecision, ApprovalError>,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("approval_resolved"));
    m.insert("request_id".into(), json!(request_id));
    let (decision, reason) = match outcome {
        Ok(d) if d.allow => ("allow", None),
        Ok(_) => ("deny", None),
        Err(ApprovalError::TimedOut) => ("deny", Some("timed_out")),
        Err(ApprovalError::Cancelled) => ("deny", Some("cancelled")),
        Err(ApprovalError::NoClient) => ("deny", Some("no_client")),
    };
    m.insert("decision".into(), json!(decision));
    if let Ok(d) = outcome {
        m.insert(
            "scope".into(),
            json!(match d.scope {
                ApprovalScope::Once => "once",
                ApprovalScope::Session => "session",
            }),
        );
    }
    if let Some(reason) = reason {
        m.insert("reason".into(), json!(reason));
    }
    Value::Object(m).into()
}

/// The `approve` command, shared by the busy and idle dispatch loops.
///
/// `accepted` is `false` when the `request_id` names no outstanding question — it was already answered
/// by another attached client, it timed out, or the run was aborted. That is not an error: it is the
/// answer a client races and loses.
///
/// `tool_gate` is whether this session asks about tool calls (`--approve`). Without one, the only
/// questions it can raise are MCP skills' — so an answer that matches none of them is refused with a
/// pointer to `--approve` rather than acknowledged, since a client sending it is most likely expecting
/// tool approvals this session will never ask for.
pub(super) fn handle_approve(
    id: Option<String>,
    cmd: &Value,
    pending: &PendingApprovals,
    tool_gate: bool,
) -> OutFrame {
    let Some(request_id) = cmd.get("request_id").and_then(Value::as_str) else {
        return response(id, "approve", false, None, Some("missing `request_id`"));
    };
    let decision = match parse_approval_decision(cmd) {
        Ok(d) => d,
        Err(e) => return response(id, "approve", false, None, Some(&e)),
    };
    let accepted = resolve_approval(pending, request_id, decision);
    if !accepted && !tool_gate {
        return response(
            id,
            "approve",
            false,
            None,
            Some(
                "no such question: this session asks only about MCP skills; tool approvals need \
                 `--approve`",
            ),
        );
    }
    response(
        id,
        "approve",
        true,
        Some(json!({ "accepted": accepted })),
        None,
    )
}

pub(super) type PendingElicitations =
    Arc<std::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<rmcp::model::ElicitResult>>>>;

pub(super) struct PendingElicitGuard {
    pub(super) pending: PendingElicitations,
    pub(super) id: String,
}

impl Drop for PendingElicitGuard {
    fn drop(&mut self) {
        lock_ignoring_poison(&self.pending).remove(&self.id);
    }
}

/// `serve`'s MCP elicitation gate — same ack-now/respond-later shape as [`ServeApprovalGate`].
///
/// Holds a [`std::sync::Weak`] to the session out-fanout (not a strong [`SharedOutConn`]) so installing
/// this gate into an MCP host hub — this session's own, or the process-wide one — cannot keep a
/// session's connection graph alive or create lock-order surprises against the live writer.
pub(super) struct ServeElicitationGate {
    pub(super) out: std::sync::Weak<std::sync::Mutex<OutFanout>>,
    pub(super) pending: PendingElicitations,
    pub(super) timeout: Option<std::time::Duration>,
    pub(super) seq: AtomicU64,
    pub(super) session_id: String,
}

impl ServeElicitationGate {
    pub(super) fn new(
        out: SharedOutConn,
        timeout: Option<std::time::Duration>,
        session_id: impl Into<String>,
    ) -> (Arc<Self>, PendingElicitations) {
        let pending: PendingElicitations = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let gate = Arc::new(Self {
            out: Arc::downgrade(&out),
            pending: pending.clone(),
            timeout,
            seq: AtomicU64::new(0),
            session_id: session_id.into(),
        });
        (gate, pending)
    }

    pub(super) fn broadcast(&self, frame: OutFrame) {
        if let Some(out) = self.out.upgrade() {
            lock_ignoring_poison(&out).broadcast(frame);
        }
    }

    pub(super) fn no_client(&self) -> bool {
        self.out
            .upgrade()
            .map(|out| lock_ignoring_poison(&out).is_empty())
            .unwrap_or(true)
    }
}

#[async_trait::async_trait]
impl crate::tools::mcp_host::ElicitationGate for ServeElicitationGate {
    async fn elicit(
        &self,
        ask: crate::tools::mcp_host::ElicitationAsk,
    ) -> Result<rmcp::model::ElicitResult, crate::tools::mcp_host::ElicitationError> {
        if self.no_client() {
            return Err(crate::tools::mcp_host::ElicitationError::NoClient);
        }
        let request_id = format!(
            "elicit:{}:{}",
            self.session_id,
            self.seq.fetch_add(1, Ordering::Relaxed)
        );
        let (tx, rx) = tokio::sync::oneshot::channel();
        lock_ignoring_poison(&self.pending).insert(request_id.clone(), tx);
        let _slot = PendingElicitGuard {
            pending: self.pending.clone(),
            id: request_id.clone(),
        };

        self.broadcast(elicitation_request_frame(&request_id, &ask));

        // Never wait forever: an attached client that never answers must not pin the session.
        let wait = self.timeout.or(Some(std::time::Duration::from_secs(120)));
        let outcome = tokio::select! {
            biased;
            answer = rx => answer.map_err(|_| crate::tools::mcp_host::ElicitationError::Cancelled),
            _ = sleep_opt(wait) => Err(crate::tools::mcp_host::ElicitationError::TimedOut),
        };
        self.broadcast(elicitation_resolved_frame(&request_id, &outcome));
        outcome
    }
}

pub(super) fn elicitation_request_frame(
    request_id: &str,
    ask: &crate::tools::mcp_host::ElicitationAsk,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("elicitation_request"));
    m.insert("request_id".into(), json!(request_id));
    m.insert("server".into(), json!(ask.server));
    m.insert(
        "params".into(),
        crate::tools::mcp_host::elicitation_params_json(&ask.params),
    );
    m.insert("options".into(), json!(["accept", "decline", "cancel"]));
    Value::Object(m).into()
}

pub(super) fn elicitation_resolved_frame(
    request_id: &str,
    outcome: &Result<rmcp::model::ElicitResult, crate::tools::mcp_host::ElicitationError>,
) -> OutFrame {
    let mut m = Map::new();
    m.insert("type".into(), json!("elicitation_resolved"));
    m.insert("request_id".into(), json!(request_id));
    match outcome {
        Ok(result) => {
            m.insert(
                "action".into(),
                json!(match result.action {
                    rmcp::model::ElicitationAction::Accept => "accept",
                    rmcp::model::ElicitationAction::Decline => "decline",
                    rmcp::model::ElicitationAction::Cancel => "cancel",
                    _ => "decline",
                }),
            );
            if let Some(content) = &result.content {
                m.insert("content".into(), content.clone());
            }
        }
        Err(crate::tools::mcp_host::ElicitationError::TimedOut) => {
            m.insert("action".into(), json!("decline"));
            m.insert("reason".into(), json!("timed_out"));
        }
        Err(crate::tools::mcp_host::ElicitationError::Cancelled) => {
            m.insert("action".into(), json!("decline"));
            m.insert("reason".into(), json!("cancelled"));
        }
        Err(crate::tools::mcp_host::ElicitationError::NoClient) => {
            m.insert("action".into(), json!("decline"));
            m.insert("reason".into(), json!("no_client"));
        }
    }
    Value::Object(m).into()
}

pub(super) fn handle_elicit(
    id: Option<String>,
    cmd: &Value,
    pending: Option<&PendingElicitations>,
) -> OutFrame {
    let Some(request_id) = cmd.get("request_id").and_then(Value::as_str) else {
        return response(id, "elicit", false, None, Some("missing `request_id`"));
    };
    let action = match cmd.get("action").and_then(Value::as_str) {
        Some("accept") => rmcp::model::ElicitationAction::Accept,
        Some("decline") => rmcp::model::ElicitationAction::Decline,
        Some("cancel") => rmcp::model::ElicitationAction::Cancel,
        _ => {
            return response(
                id,
                "elicit",
                false,
                None,
                Some("`action` must be \"accept\", \"decline\", or \"cancel\""),
            );
        }
    };
    let mut result = rmcp::model::ElicitResult::new(action);
    if let Some(content) = cmd.get("content") {
        result = result.with_content(content.clone());
    }
    let Some(pending) = pending else {
        return response(
            id,
            "elicit",
            false,
            None,
            Some("this session has no elicitation gate"),
        );
    };
    let accepted = lock_ignoring_poison(pending)
        .remove(request_id)
        .and_then(|tx| tx.send(result).ok())
        .is_some();
    response(
        id,
        "elicit",
        true,
        Some(json!({ "accepted": accepted })),
        None,
    )
}

/// MCP sampling answered by the attached host (see [`serve_sampling::ServeSamplingGate`]).
pub(super) mod serve_sampling {
    #![expect(deprecated)] // Sampling: SEP-2577-deprecated but still on the wire.
    use super::*;

    /// Outstanding `sampling_request`s: request id → where the host's `sample` answer goes.
    pub(in crate::serve) type PendingSamplings = Arc<
        std::sync::Mutex<
            HashMap<
                String,
                tokio::sync::oneshot::Sender<Result<rmcp::model::CreateMessageResult, String>>,
            >,
        >,
    >;

    /// Removes a request's slot however its wait ends, including the gate's future being dropped
    /// (an aborted call), so an abandoned request never lingers in `pending`.
    struct PendingSlot {
        pending: PendingSamplings,
        id: String,
    }

    impl Drop for PendingSlot {
        fn drop(&mut self) {
            lock_ignoring_poison(&self.pending).remove(&self.id);
        }
    }

    /// `serve`'s MCP sampling gate: the same ask-the-host shape as [`ServeElicitationGate`]. A
    /// server's `sampling/createMessage` (nested, MRTR, or an in-task `inputRequests` entry, which
    /// the spec says must be treated exactly like the standalone request) becomes a
    /// `sampling_request` frame naming the asking `server`, and the host answers with a `sample`
    /// command carrying the `CreateMessageResult`, or declines. Every attached view then sees a
    /// `sampling_resolved` frame, so a second client stops offering an answer. The host decides
    /// what runs the completion; the agent never spends model tokens on a server's behalf unasked.
    /// No attached client, a decline, or the timeout refuses the request.
    pub(in crate::serve) struct ServeSamplingGate {
        out: std::sync::Weak<std::sync::Mutex<OutFanout>>,
        pending: PendingSamplings,
        timeout: Option<std::time::Duration>,
        seq: AtomicU64,
        session_id: String,
    }

    impl ServeSamplingGate {
        pub(in crate::serve) fn new(
            out: SharedOutConn,
            timeout: Option<std::time::Duration>,
            session_id: impl Into<String>,
        ) -> (Arc<Self>, PendingSamplings) {
            let pending: PendingSamplings = Arc::new(std::sync::Mutex::new(HashMap::new()));
            let gate = Arc::new(Self {
                out: Arc::downgrade(&out),
                pending: pending.clone(),
                timeout,
                seq: AtomicU64::new(0),
                session_id: session_id.into(),
            });
            (gate, pending)
        }

        fn broadcast(&self, frame: Value) {
            if let Some(out) = self.out.upgrade() {
                lock_ignoring_poison(&out).broadcast(frame.into());
            }
        }
    }

    #[async_trait::async_trait]
    impl crate::tools::mcp_host::SamplingGate for ServeSamplingGate {
        async fn create_message(
            &self,
            server: &str,
            params: rmcp::model::CreateMessageRequestParams,
        ) -> Result<rmcp::model::CreateMessageResult, rmcp::ErrorData> {
            let refuse =
                |why: &str| rmcp::ErrorData::new(rmcp::model::ErrorCode(-1), why.to_owned(), None);
            let attached = self
                .out
                .upgrade()
                .is_some_and(|out| !lock_ignoring_poison(&out).is_empty());
            if !attached {
                return Err(refuse("no client attached to answer sampling"));
            }
            let request_id = format!(
                "sample:{}:{}",
                self.session_id,
                self.seq.fetch_add(1, Ordering::Relaxed)
            );
            let (tx, rx) = tokio::sync::oneshot::channel();
            lock_ignoring_poison(&self.pending).insert(request_id.clone(), tx);
            let _slot = PendingSlot {
                pending: self.pending.clone(),
                id: request_id.clone(),
            };
            self.broadcast(json!({
                "type": "sampling_request",
                "request_id": request_id,
                "server": server,
                "params": params,
            }));
            let wait = self.timeout.or(Some(std::time::Duration::from_secs(120)));
            let outcome = tokio::select! {
                biased;
                answer = rx => answer.unwrap_or_else(|_| Err("cancelled".to_owned())),
                _ = sleep_opt(wait) => Err("timed out".to_owned()),
            };
            let mut resolved = json!({ "type": "sampling_resolved", "request_id": request_id });
            match &outcome {
                Ok(_) => resolved["action"] = json!("accept"),
                Err(why) => {
                    resolved["action"] = json!("decline");
                    resolved["reason"] = json!(why);
                }
            }
            self.broadcast(resolved);
            outcome.map_err(|why| refuse(&format!("sampling request refused: {why}")))
        }
    }

    /// `{type:"sample", request_id, result}` answers a `sampling_request` with a
    /// `CreateMessageResult`; `{type:"sample", request_id, action:"decline"}` refuses it.
    pub(in crate::serve) fn handle_sample(
        id: Option<String>,
        cmd: &Value,
        pending: &PendingSamplings,
    ) -> OutFrame {
        let Some(request_id) = cmd.get("request_id").and_then(Value::as_str) else {
            return response(id, "sample", false, None, Some("missing `request_id`"));
        };
        let answer = match (cmd.get("result"), cmd.get("action").and_then(Value::as_str)) {
            (Some(result), _) => match serde_json::from_value(result.clone()) {
                Ok(result) => Ok(result),
                Err(e) => {
                    let why = format!("invalid `result` (CreateMessageResult): {e}");
                    return response(id, "sample", false, None, Some(&why));
                }
            },
            (None, Some("decline")) => Err("declined by the host".to_owned()),
            _ => {
                let why = "`sample` needs a `result` or `action: \"decline\"`";
                return response(id, "sample", false, None, Some(why));
            }
        };
        let accepted = lock_ignoring_poison(pending)
            .remove(request_id)
            .and_then(|tx| tx.send(answer).ok())
            .is_some();
        response(
            id,
            "sample",
            true,
            Some(json!({ "accepted": accepted })),
            None,
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::tools::mcp_host::SamplingGate;

        /// F7: a sampling request whose call is abandoned (the gate's future dropped mid-wait)
        /// must not leave its slot behind in `pending`.
        #[tokio::test]
        async fn an_abandoned_sampling_request_leaves_no_pending_slot() {
            let out: SharedOutConn = Arc::new(std::sync::Mutex::new(OutFanout::default()));
            let (tx, _rx) = mpsc::unbounded_channel();
            lock_ignoring_poison(&out).add(OutSink::Unbounded(tx));
            let (gate, pending) = ServeSamplingGate::new(out.clone(), None, "session-under-test");
            let params: rmcp::model::CreateMessageRequestParams = serde_json::from_value(json!({
                "messages": [{ "role": "user", "content": { "type": "text", "text": "hi" } }],
                "maxTokens": 8,
            }))
            .unwrap();
            let mut ask = Box::pin(gate.create_message("srv", params));
            // Poll until the request is outstanding, then abandon it.
            for _ in 0..10 {
                if !lock_ignoring_poison(&pending).is_empty() {
                    break;
                }
                let _ = futures::poll!(ask.as_mut());
            }
            assert_eq!(
                lock_ignoring_poison(&pending).len(),
                1,
                "the request is outstanding"
            );
            drop(ask);
            assert!(
                lock_ignoring_poison(&pending).is_empty(),
                "an abandoned request must free its slot"
            );
        }
    }
}
pub(super) use serve_sampling::{ServeSamplingGate, handle_sample};

/// Parse an inbound `approve` command's `decision`/`scope` fields.
pub(super) fn parse_approval_decision(cmd: &Value) -> Result<ApprovalDecision, String> {
    let allow = match cmd.get("decision").and_then(Value::as_str) {
        Some("allow") => true,
        Some("deny") => false,
        _ => return Err("`decision` must be \"allow\" or \"deny\"".to_string()),
    };
    let scope = match cmd.get("scope").and_then(Value::as_str) {
        None | Some("once") => ApprovalScope::Once,
        Some("session") => ApprovalScope::Session,
        Some(other) => {
            return Err(format!(
                "`scope` must be \"once\" or \"session\" (got {other:?})"
            ));
        }
    };
    Ok(ApprovalDecision { allow, scope })
}

/// Drives an in-flight `login` RPC command: pushes `login_progress` frames for whatever the flow
/// needs to show (a URL, a device code, narration), and for the manual-code-paste path, parks on
/// `pending_code` until a `"submit_code"` command (processed by the same main loop, concurrently —
/// see the `"login"` dispatch arm) wakes it with the pasted value.
pub(super) struct ServeLoginCallbacks {
    pub(super) out_tx: OutTx,
    pub(super) id: Option<String>,
    pub(super) provider: crate::oauth::OAuthProviderId,
    pub(super) pending_code: PendingCodeSlot,
    /// This login's own token, so the wait below can lose a race against it. Not merely decorative:
    /// two provider flows (`oauth::github_copilot`'s Enterprise-domain prompt, `oauth::openai_codex`'s
    /// port-already-bound fallback) `await` `prompt_text` with *no other* cancellation observer in the
    /// select, so if this wait doesn't watch the token itself, nothing does — `abort_login` becomes a
    /// silent no-op and the task parks forever.
    pub(super) cancel: agent_core::CancellationToken,
}

/// How long an unanswered `manual_code` prompt waits before the login is abandoned.
///
/// The wait is on a human: read a browser page, copy a code, paste it back. Generous enough that a
/// distracted user doesn't lose a legitimate flow, finite because the alternative is a task parked for
/// the process's lifetime — and a parked login task is not cheap. It holds a clone of the session's
/// `out_tx`, and the session's writer ends *only* when every sender drops, so a login nobody ever
/// answers wedges `serve_session` itself: the OS thread, its runtime, the `Agent`, the `GatewayClient`
/// and its pool, the whole message history, and the OAuth callback server's bound (and fixed) port all
/// stay live until the daemon dies. Failing closed on silence is what keeps an abandoned browser tab
/// from costing a session.
pub(super) const LOGIN_PROMPT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

#[async_trait::async_trait]
impl crate::oauth::LoginCallbacks for ServeLoginCallbacks {
    async fn show_auth_url(&self, url: &str, _instructions: Option<&str>) {
        let _ = self.out_tx.send(login_progress_frame(
            self.id.clone(),
            &self.provider.to_string(),
            "browser",
            Some(url),
            None,
            None,
            None,
            None,
        ));
    }

    async fn show_device_code(&self, info: &crate::oauth::DeviceCodeInfo) {
        let _ = self.out_tx.send(login_progress_frame(
            self.id.clone(),
            &self.provider.to_string(),
            "device_code",
            None,
            Some(&info.user_code),
            Some(&info.verification_uri),
            info.expires_in.map(|d| d.as_secs()),
            None,
        ));
    }

    async fn progress(&self, message: &str) {
        let _ = self.out_tx.send(login_progress_frame(
            self.id.clone(),
            &self.provider.to_string(),
            "progress",
            None,
            None,
            None,
            None,
            Some(message),
        ));
    }

    async fn prompt_text(
        &self,
        _prompt: &crate::oauth::TextPrompt<'_>,
    ) -> std::result::Result<String, crate::oauth::OAuthError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        *lock_ignoring_poison(&self.pending_code) = Some(tx);
        let _ = self.out_tx.send(login_progress_frame(
            self.id.clone(),
            &self.provider.to_string(),
            "manual_code",
            None,
            None,
            None,
            None,
            None,
        ));
        // The same shape `ServeApprovalGate::request` uses, and for the same reason: this is the second
        // instance of the one human-in-the-loop seam, so it races the token and a timeout and **fails
        // closed** — no answer, for any reason, abandons the login. A bare `rx.await` here (what this
        // used to be) has no way out at all: the sender lives inside these callbacks, which the login
        // future owns, so nothing external can even drop it to wake the wait.
        //
        // `biased` so a cancellation already signalled wins over a code that happened to land in the
        // same poll — an aborted login must not quietly succeed on the strength of a late paste.
        tokio::select! {
            biased;
            _ = self.cancel.cancelled() => Err(crate::oauth::OAuthError::LoginCancelled),
            answer = rx => answer.map_err(|_| crate::oauth::OAuthError::LoginCancelled),
            _ = tokio::time::sleep(LOGIN_PROMPT_TIMEOUT) => {
                tracing::debug!(
                    provider = %self.provider,
                    "login code prompt went unanswered; abandoning the login"
                );
                Err(crate::oauth::OAuthError::LoginCancelled)
            }
        }
    }

    async fn select(
        &self,
        prompt: &crate::oauth::SelectPrompt<'_>,
    ) -> std::result::Result<Option<String>, crate::oauth::OAuthError> {
        // Headless: always take the first (recommended) option rather than interactively prompting —
        // matches this trait's own documented contract for a non-interactive caller.
        Ok(prompt.options.first().map(|o| o.id.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_login_guard_resets_the_slot_even_when_the_task_panics() {
        // Regression: the detached `login` task's `JoinHandle` is discarded (see the `"login"`
        // dispatch arm), so nothing else can react to a panic inside it — only a `Drop`-based guard
        // can guarantee `pending_login` gets reset, closing the "every subsequent login hangs behind
        // 'busy' forever" failure mode. Exercises `PendingLoginGuard` directly rather than the real
        // `oauth::login` flow, since forcing a genuine panic through that path would need a fake
        // provider seam this module doesn't otherwise need.
        let pending_login: Arc<std::sync::Mutex<Option<PendingLogin>>> =
            Arc::new(std::sync::Mutex::new(Some(PendingLogin {
                generation: 1,
                cancel: CancellationToken::new(),
                pending_code: Arc::new(std::sync::Mutex::new(None)),
            })));
        let slot = pending_login.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = PendingLoginGuard {
                slot: slot.clone(),
                generation: 1,
            };
            panic!("simulated failure inside the login task");
        }));
        assert!(
            result.is_err(),
            "the simulated panic should propagate out of catch_unwind"
        );
        assert!(
            lock_ignoring_poison(&pending_login).is_none(),
            "pending_login must be cleared even though the guarded scope panicked"
        );
    }

    #[test]
    fn pending_login_guard_only_clears_its_own_generation_not_a_successor() {
        // The race behind the flaky `serve_auth` abort-login hang: `abort_login` clears the slot
        // synchronously, so a *newer* login can claim it before the cancelled task's guard runs. That
        // guard must then be a no-op — wiping the successor's marker would make the next `abort_login`
        // find an empty slot and silently no-op, leaving the newer login uncancellable (its client hangs).
        let slot: Arc<std::sync::Mutex<Option<PendingLogin>>> =
            Arc::new(std::sync::Mutex::new(None));
        // A newer login (generation 2) currently owns the slot.
        let newer_cancel = CancellationToken::new();
        *lock_ignoring_poison(&slot) = Some(PendingLogin {
            generation: 2,
            cancel: newer_cancel.clone(),
            pending_code: Arc::new(std::sync::Mutex::new(None)),
        });
        // An older login's guard (generation 1) winds down now.
        drop(PendingLoginGuard {
            slot: slot.clone(),
            generation: 1,
        });
        // The successor's marker must survive, and still be the cancel `abort_login` would reach.
        let held = lock_ignoring_poison(&slot);
        assert!(
            held.as_ref().is_some_and(|p| p.generation == 2),
            "the older guard must not clear the newer login's slot"
        );
        drop(held);
        // And a guard whose generation *does* match clears it (the normal, own-slot case).
        drop(PendingLoginGuard {
            slot: slot.clone(),
            generation: 2,
        });
        assert!(lock_ignoring_poison(&slot).is_none());
    }
}
