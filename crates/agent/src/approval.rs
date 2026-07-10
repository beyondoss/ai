//! Interactive tool approval — the human-in-the-loop gate.
//!
//! [`AgentHooks::before_tool_call`] is documented as "the seam a permission/approval system hangs off,"
//! and it carries the run's `CancellationToken` precisely so a hook can do slow external I/O (like
//! asking a person) and still bail out promptly. Until now only [`ToolPolicy`]'s static deny-lists used
//! it. This module adds the other half: a call the operator has to say yes to before it runs.
//!
//! The shape is the one this codebase already uses for its only other human-in-the-loop flow, OAuth's
//! manual-code paste: an **abstract callback trait** ([`ApprovalGate`], the sibling of
//! `oauth::LoginCallbacks`) with one implementation per host. `serve` broadcasts a frame and parks a
//! `oneshot` behind a drop guard; a future CLI implementation would prompt on a TTY. Nothing in
//! `agent-core` changes.
//!
//! # Fail closed
//!
//! Every way of *not* getting an answer — no client attached, a timeout, the run cancelled — denies the
//! call. A security control that fails open on absence provides no security, and denying is also what
//! keeps a detached background session from hanging forever on a question nobody will ever see: the
//! model gets an error `tool_result` and the run carries on.
//!
//! # Remembering a decision
//!
//! "Always allow" is keyed on `(tool, scope_key)`, where [`scope_key`] is deliberately narrow:
//!
//! - `write`/`edit` → the **resolved absolute path**, computed with the same canonicalization the policy
//!   and the tools themselves use, so `../` and symlink spellings can't evade a remembered decision.
//! - `bash` → the **verbatim command string**. Explicitly *not* a command prefix: `git status` versus
//!   `git push` is not fixable with prefixes (is `git` safe? `git p`? `git push --force`?). Exact
//!   matching fails safe — the only cost is re-asking when a command varies by a byte, which is the
//!   right direction for a security control.
//! - everything else → the bare tool name.
//!
//! [`AgentHooks::before_tool_call`]: agent_core::AgentHooks::before_tool_call

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use agent_core::{CancellationToken, Session};
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::policy::ToolPolicy;

/// Tools that only ever read. `GatedSet::All` still skips them: asking a human to approve a `grep` is
/// how an approval prompt gets trained away into a reflex.
const READ_ONLY_TOOLS: [&str; 7] = [
    "read",
    "grep",
    "find",
    "ls",
    "todo",
    "logs",
    "structured_output",
];

/// Tools that mutate the filesystem through a path they name. `bash` is deliberately absent: it mutates
/// anything reachable from its cwd, which is why it is gated by [`GatedSet::All`] rather than `Writes`.
const WRITE_TOOLS: [&str; 2] = ["write", "edit"];

/// How long a `write`/`bash` argument may be before it is truncated in the request sent to clients. The
/// point is to show a human *what* is about to happen, not to fan a megabyte file body out to every
/// attached phone.
const SUMMARY_MAX_CHARS: usize = 400;

/// Which agent is asking. A parallel fan-out can have several children blocked at once, and "something
/// wants to run `rm -rf`" is not an answerable question.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApprovalOrigin {
    /// The session's own agent.
    Main,
    /// A `subagent` child: which persona, and which spawn of it.
    Subagent { agent: String, spawn_id: String },
}

impl ApprovalOrigin {
    pub fn to_json(&self) -> Value {
        match self {
            Self::Main => json!("main"),
            Self::Subagent { agent, spawn_id } => json!({ "agent": agent, "spawn_id": spawn_id }),
        }
    }
}

/// One pending question. The transport assigns the request id; this is just what is being asked.
#[derive(Clone, Debug)]
pub struct ApprovalRequest {
    pub tool: String,
    /// A truncated preview of the call's arguments, for a human to read.
    pub summary: Value,
    /// What an "always" decision would be remembered against. Shipped to the client so the user can see
    /// exactly what they are agreeing to for the rest of the session.
    pub scope_key: String,
    pub origin: ApprovalOrigin,
}

/// Whether a decision applies to this call only, or is remembered for the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalScope {
    Once,
    Session,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApprovalDecision {
    pub allow: bool,
    pub scope: ApprovalScope,
}

/// Every way of failing to get an answer. All of them deny — see the module doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalError {
    /// The run was aborted while the question was outstanding.
    Cancelled,
    /// Nobody answered in time.
    TimedOut,
    /// There is no one to ask: every client has detached.
    NoClient,
}

/// The human-in-the-loop seam. One implementation per host.
#[async_trait]
pub trait ApprovalGate: Send + Sync {
    async fn request(
        &self,
        req: ApprovalRequest,
        cancel: &CancellationToken,
    ) -> Result<ApprovalDecision, ApprovalError>;
}

/// Which tools require approval.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum GatedSet {
    /// No gate at all — the default, and what every existing invocation gets.
    #[default]
    Off,
    /// The filesystem-mutating tools that name their target: `write`, `edit`.
    Writes,
    /// An explicit list.
    Tools(HashSet<String>),
    /// Everything except [`READ_ONLY_TOOLS`].
    All,
}

impl GatedSet {
    pub fn is_gated(&self, tool: &str) -> bool {
        match self {
            Self::Off => false,
            Self::Writes => WRITE_TOOLS.contains(&tool),
            Self::Tools(names) => names.contains(tool),
            Self::All => !READ_ONLY_TOOLS.contains(&tool),
        }
    }

    pub fn is_off(&self) -> bool {
        matches!(self, Self::Off)
    }

    /// Parse `off` / `writes` / `all` / `tools:a,b`.
    ///
    /// The list form needs its `tools:` prefix. Without it a typo (`--approve wrties`) would parse as a
    /// list naming one tool that does not exist, which gates *nothing* — a security control that fails
    /// open on a misspelling. An unrecognized bare word is an error instead.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let spec = spec.trim();
        if let Some(list) = spec.strip_prefix("tools:") {
            let names: HashSet<String> = list
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
            if names.is_empty() {
                return Err("--approve tools: names no tools".to_string());
            }
            return Ok(Self::Tools(names));
        }
        match spec {
            "off" | "" => Ok(Self::Off),
            "writes" => Ok(Self::Writes),
            "all" => Ok(Self::All),
            other => Err(format!(
                "--approve: expected `off`, `writes`, `all`, or `tools:<name>,<name>` (got {other:?})"
            )),
        }
    }
}

/// Session-scoped memory of "always allow"/"always deny" decisions.
///
/// Shared by `Arc` with every subagent child, so a decision the user already made isn't asked again per
/// child. **That widens privilege across the subagent tree**: an "always allow `bash: cargo test`"
/// granted to one child applies to its siblings and to the parent for the rest of the session. That is
/// the intended "don't re-ask" behavior, but it is a real property, not an accident.
#[derive(Default)]
pub struct SessionMemory {
    decisions: Mutex<HashMap<(String, String), bool>>,
}

impl SessionMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lookup(&self, tool: &str, scope_key: &str) -> Option<bool> {
        self.lock()
            .get(&(tool.to_string(), scope_key.to_string()))
            .copied()
    }

    pub fn remember(&self, tool: &str, scope_key: &str, allow: bool) {
        self.lock()
            .insert((tool.to_string(), scope_key.to_string()), allow);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<(String, String), bool>> {
        self.decisions.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Everything the gate needs, cloned into every subagent child so the whole tree shares one gate, one
/// memory, and (crucially) one prompt-serializing lock.
#[derive(Clone)]
pub struct ApprovalRuntime {
    pub gate: Arc<dyn ApprovalGate>,
    pub gated: GatedSet,
    pub memory: Arc<SessionMemory>,
}

impl ApprovalRuntime {
    pub fn new(gate: Arc<dyn ApprovalGate>, gated: GatedSet) -> Self {
        Self {
            gate,
            gated,
            memory: Arc::new(SessionMemory::new()),
        }
    }
}

/// What an "always" decision is remembered against. See the module doc for why each is what it is.
pub fn scope_key(tool: &str, input: &Value, root: &std::path::Path) -> String {
    match tool {
        "bash" => match input.get("command").and_then(Value::as_str) {
            // Verbatim, never a prefix.
            Some(command) => format!("cmd:{command}"),
            None => "cmd:<none>".to_string(),
        },
        "write" | "edit" => match input.get("path").and_then(Value::as_str) {
            Some(path) => {
                // The same canonical form the tools themselves write to, so `./x`, `../dir/x` and a
                // symlink to `x` all resolve to one remembered decision rather than three.
                let target = crate::tools::canonical_write_target(
                    root,
                    &crate::tools::resolve_against(root, path),
                );
                format!("path:{target}")
            }
            None => "path:<none>".to_string(),
        },
        other => format!("tool:{other}"),
    }
}

/// A truncated, human-readable preview of a call's arguments.
pub fn summarize(input: &Value) -> Value {
    match input {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(k, _)| *k != agent_core::tool::MODEL_SUPPORTS_VISION_KEY)
                .map(|(k, v)| (k.clone(), truncate_value(v)))
                .collect(),
        ),
        other => truncate_value(other),
    }
}

fn truncate_value(value: &Value) -> Value {
    match value.as_str() {
        Some(s) if s.chars().count() > SUMMARY_MAX_CHARS => {
            let kept: String = s.chars().take(SUMMARY_MAX_CHARS).collect();
            let dropped = s.chars().count() - SUMMARY_MAX_CHARS;
            Value::String(format!("{kept}… [{dropped} more characters]"))
        }
        _ => value.clone(),
    }
}

/// The composed `before_tool_call` body: static policy, then the gate.
///
/// The order is load-bearing:
/// 1. A **static deny wins outright**, with no round trip. Asking a human to approve something the
///    operator already forbade is both pointless and a way to social-engineer past the deny-list.
/// 2. A tool outside the gated set runs unimpeded.
/// 3. A remembered decision answers without asking.
/// 4. Otherwise, ask — and remember the answer if it was a "session" decision.
/// 5. **Every failure to get an answer denies** (see the module doc).
pub async fn gated_before_tool_call(
    policy: &ToolPolicy,
    runtime: Option<&ApprovalRuntime>,
    origin: &ApprovalOrigin,
    name: &str,
    input: &Value,
    session: &Session,
    cancel: &CancellationToken,
) -> Option<String> {
    use agent_core::AgentHooks as _;

    if let Some(reason) = policy.before_tool_call(name, input, session, cancel).await {
        return Some(reason);
    }
    let runtime = runtime?;
    if !runtime.gated.is_gated(name) {
        return None;
    }

    let key = scope_key(name, input, policy.root());
    match runtime.memory.lookup(name, &key) {
        Some(true) => return None,
        Some(false) => {
            return Some(format!(
                "'{name}' was denied for the rest of this session by an earlier decision"
            ));
        }
        None => {}
    }

    let request = ApprovalRequest {
        tool: name.to_string(),
        summary: summarize(input),
        scope_key: key.clone(),
        origin: origin.clone(),
    };
    match runtime.gate.request(request, cancel).await {
        Ok(decision) => {
            if decision.scope == ApprovalScope::Session {
                runtime.memory.remember(name, &key, decision.allow);
            }
            if decision.allow {
                None
            } else {
                Some(format!("'{name}' was denied by the user"))
            }
        }
        Err(ApprovalError::NoClient) => Some(format!(
            "'{name}' requires approval, but no client is attached to give it"
        )),
        Err(ApprovalError::TimedOut) => Some(format!(
            "'{name}' was denied: the approval request timed out"
        )),
        Err(ApprovalError::Cancelled) => {
            Some(format!("'{name}' was denied: the run was cancelled"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn cwd() -> std::path::PathBuf {
        std::path::PathBuf::new()
    }

    // ---- GatedSet ---------------------------------------------------------------------------------

    #[test]
    fn off_gates_nothing() {
        assert!(!GatedSet::Off.is_gated("bash"));
        assert!(!GatedSet::Off.is_gated("write"));
    }

    #[test]
    fn writes_gates_only_the_tools_that_name_their_target() {
        let g = GatedSet::Writes;
        assert!(g.is_gated("write"));
        assert!(g.is_gated("edit"));
        // `bash` can write anywhere; it belongs to `all`, not to a set defined by a `path` argument.
        assert!(!g.is_gated("bash"));
        assert!(!g.is_gated("read"));
    }

    #[test]
    fn all_gates_everything_except_the_read_only_tools() {
        let g = GatedSet::All;
        assert!(g.is_gated("bash"));
        assert!(g.is_gated("write"));
        assert!(g.is_gated("subagent"));
        for name in READ_ONLY_TOOLS {
            assert!(!g.is_gated(name), "{name} must not require approval");
        }
    }

    #[test]
    fn an_explicit_list_gates_exactly_what_it_names() {
        let g = GatedSet::parse("tools:bash,subagent").unwrap();
        assert!(g.is_gated("bash"));
        assert!(g.is_gated("subagent"));
        assert!(!g.is_gated("write"));
    }

    #[test]
    fn gated_set_parses_its_three_keywords_and_rejects_nonsense() {
        assert_eq!(GatedSet::parse("off").unwrap(), GatedSet::Off);
        assert_eq!(GatedSet::parse("").unwrap(), GatedSet::Off);
        assert_eq!(GatedSet::parse("writes").unwrap(), GatedSet::Writes);
        assert_eq!(GatedSet::parse("all").unwrap(), GatedSet::All);
        assert!(GatedSet::parse("tools:").is_err());
        assert!(GatedSet::parse("tools:,,").is_err());
    }

    #[test]
    fn a_misspelled_keyword_is_an_error_rather_than_a_silently_empty_gate() {
        // Without the `tools:` prefix, `--approve wrties` would parse as a list naming one tool that
        // does not exist — a security control that fails open on a typo.
        let err = GatedSet::parse("wrties").unwrap_err();
        assert!(err.contains("expected `off`, `writes`, `all`"), "{err}");
        assert!(GatedSet::parse("bash,write").is_err());
    }

    // ---- scope_key --------------------------------------------------------------------------------

    #[test]
    fn a_bash_scope_key_is_the_verbatim_command_not_a_prefix() {
        let key = |c: &str| scope_key("bash", &json!({ "command": c }), &cwd());
        assert_eq!(key("git status"), "cmd:git status");
        assert_ne!(
            key("git status"),
            key("git push"),
            "a remembered `git status` must never authorize `git push`"
        );
        assert_ne!(key("git push"), key("git push --force"));
    }

    #[test]
    fn a_write_scope_key_is_the_resolved_path_so_spellings_cannot_evade_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.md"), "x").unwrap();
        let root = dir.path();

        let direct = scope_key("write", &json!({ "path": "notes.md" }), root);
        let dotted = scope_key("write", &json!({ "path": "./notes.md" }), root);
        let dotdot = scope_key("edit", &json!({ "path": "sub/../notes.md" }), root);
        assert_eq!(direct, dotted);
        assert_eq!(
            direct, dotdot,
            "an `edit` and a `write` agree on the target"
        );
        assert!(direct.starts_with("path:/"), "must be absolute: {direct}");

        // A different file is a different decision.
        assert_ne!(
            direct,
            scope_key("write", &json!({ "path": "other.md" }), root)
        );
    }

    #[test]
    fn any_other_tool_keys_on_its_name() {
        assert_eq!(
            scope_key("subagent", &json!({ "agent": "x" }), &cwd()),
            "tool:subagent"
        );
    }

    #[test]
    fn a_missing_argument_still_produces_a_stable_key_rather_than_panicking() {
        assert_eq!(scope_key("bash", &json!({}), &cwd()), "cmd:<none>");
        assert_eq!(scope_key("write", &json!({}), &cwd()), "path:<none>");
    }

    // ---- summarize --------------------------------------------------------------------------------

    #[test]
    fn a_summary_truncates_a_huge_argument_and_drops_the_internal_vision_flag() {
        let body = "x".repeat(SUMMARY_MAX_CHARS + 50);
        let input = json!({
            "path": "a.rs",
            "content": body,
            agent_core::tool::MODEL_SUPPORTS_VISION_KEY: true,
        });
        let s = summarize(&input);
        assert_eq!(s["path"], "a.rs");
        assert!(
            s.get(agent_core::tool::MODEL_SUPPORTS_VISION_KEY).is_none(),
            "the loop's internal key is noise to a human: {s}"
        );
        let content = s["content"].as_str().unwrap();
        assert!(content.contains("[50 more characters]"), "{content}");
        assert!(content.chars().count() < body.chars().count());
    }

    // ---- SessionMemory ----------------------------------------------------------------------------

    #[test]
    fn session_memory_is_keyed_by_tool_and_scope() {
        let m = SessionMemory::new();
        assert_eq!(m.lookup("bash", "cmd:ls"), None);
        m.remember("bash", "cmd:ls", true);
        assert_eq!(m.lookup("bash", "cmd:ls"), Some(true));
        assert_eq!(
            m.lookup("bash", "cmd:rm"),
            None,
            "a different command is a different decision"
        );
        assert_eq!(
            m.lookup("write", "cmd:ls"),
            None,
            "a different tool is a different decision"
        );

        m.remember("bash", "cmd:rm", false);
        assert_eq!(m.lookup("bash", "cmd:rm"), Some(false));
    }

    // ---- the decision table -----------------------------------------------------------------------

    /// A gate that answers with a fixed decision (or error) and counts how often it was asked.
    struct FakeGate {
        answer: Result<ApprovalDecision, ApprovalError>,
        asked: AtomicUsize,
    }

    impl FakeGate {
        fn new(answer: Result<ApprovalDecision, ApprovalError>) -> Arc<Self> {
            Arc::new(Self {
                answer,
                asked: AtomicUsize::new(0),
            })
        }
        fn asked(&self) -> usize {
            self.asked.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl ApprovalGate for FakeGate {
        async fn request(
            &self,
            _req: ApprovalRequest,
            _cancel: &CancellationToken,
        ) -> Result<ApprovalDecision, ApprovalError> {
            self.asked.fetch_add(1, Ordering::Relaxed);
            self.answer
        }
    }

    fn allow(scope: ApprovalScope) -> Result<ApprovalDecision, ApprovalError> {
        Ok(ApprovalDecision { allow: true, scope })
    }
    fn deny(scope: ApprovalScope) -> Result<ApprovalDecision, ApprovalError> {
        Ok(ApprovalDecision {
            allow: false,
            scope,
        })
    }

    async fn check(
        policy: &ToolPolicy,
        rt: Option<&ApprovalRuntime>,
        name: &str,
        input: Value,
    ) -> Option<String> {
        gated_before_tool_call(
            policy,
            rt,
            &ApprovalOrigin::Main,
            name,
            &input,
            &Session::new(),
            &CancellationToken::new(),
        )
        .await
    }

    #[tokio::test]
    async fn a_static_deny_wins_without_ever_asking_a_human() {
        // Asking someone to approve what the operator already forbade is both pointless and a way to
        // social-engineer past the deny-list.
        let gate = FakeGate::new(allow(ApprovalScope::Once));
        let rt = ApprovalRuntime::new(gate.clone(), GatedSet::All);
        let policy = ToolPolicy::new().deny_tool("bash");

        let reason = check(&policy, Some(&rt), "bash", json!({ "command": "ls" })).await;
        assert!(reason.unwrap().contains("denied by policy"));
        assert_eq!(gate.asked(), 0, "the gate must never have been consulted");
    }

    #[tokio::test]
    async fn an_ungated_tool_is_never_asked_about() {
        let gate = FakeGate::new(deny(ApprovalScope::Once));
        let rt = ApprovalRuntime::new(gate.clone(), GatedSet::Writes);
        assert!(
            check(
                &ToolPolicy::new(),
                Some(&rt),
                "read",
                json!({ "path": "a" })
            )
            .await
            .is_none()
        );
        assert_eq!(gate.asked(), 0);
    }

    #[tokio::test]
    async fn no_runtime_at_all_leaves_the_static_policy_in_charge() {
        let policy = ToolPolicy::new().deny_tool("bash");
        assert!(
            check(&policy, None, "bash", json!({ "command": "ls" }))
                .await
                .is_some()
        );
        assert!(
            check(&policy, None, "write", json!({ "path": "a" }))
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn an_allow_lets_the_call_through() {
        let gate = FakeGate::new(allow(ApprovalScope::Once));
        let rt = ApprovalRuntime::new(gate.clone(), GatedSet::Writes);
        assert!(
            check(
                &ToolPolicy::new(),
                Some(&rt),
                "write",
                json!({ "path": "a" })
            )
            .await
            .is_none()
        );
        assert_eq!(gate.asked(), 1);
    }

    #[tokio::test]
    async fn a_deny_blocks_the_call_with_a_reason_the_model_can_read() {
        let gate = FakeGate::new(deny(ApprovalScope::Once));
        let rt = ApprovalRuntime::new(gate, GatedSet::Writes);
        let reason = check(
            &ToolPolicy::new(),
            Some(&rt),
            "write",
            json!({ "path": "a" }),
        )
        .await;
        assert!(reason.unwrap().contains("denied by the user"));
    }

    #[tokio::test]
    async fn a_once_decision_is_not_remembered_but_a_session_decision_is() {
        let gate = FakeGate::new(allow(ApprovalScope::Once));
        let rt = ApprovalRuntime::new(gate.clone(), GatedSet::All);
        let call = json!({ "command": "cargo test" });
        check(&ToolPolicy::new(), Some(&rt), "bash", call.clone()).await;
        check(&ToolPolicy::new(), Some(&rt), "bash", call.clone()).await;
        assert_eq!(gate.asked(), 2, "a one-shot allow must be asked again");

        let gate = FakeGate::new(allow(ApprovalScope::Session));
        let rt = ApprovalRuntime::new(gate.clone(), GatedSet::All);
        assert!(
            check(&ToolPolicy::new(), Some(&rt), "bash", call.clone())
                .await
                .is_none()
        );
        assert!(
            check(&ToolPolicy::new(), Some(&rt), "bash", call)
                .await
                .is_none()
        );
        assert_eq!(gate.asked(), 1, "a session allow must be remembered");
    }

    #[tokio::test]
    async fn a_session_allow_does_not_authorize_a_different_command() {
        let gate = FakeGate::new(allow(ApprovalScope::Session));
        let rt = ApprovalRuntime::new(gate.clone(), GatedSet::All);
        check(
            &ToolPolicy::new(),
            Some(&rt),
            "bash",
            json!({ "command": "git status" }),
        )
        .await;
        check(
            &ToolPolicy::new(),
            Some(&rt),
            "bash",
            json!({ "command": "git push" }),
        )
        .await;
        assert_eq!(
            gate.asked(),
            2,
            "`git status` must never authorize `git push`"
        );
    }

    #[tokio::test]
    async fn a_session_deny_is_remembered_too() {
        let gate = FakeGate::new(deny(ApprovalScope::Session));
        let rt = ApprovalRuntime::new(gate.clone(), GatedSet::All);
        let call = json!({ "command": "rm -rf /" });
        assert!(
            check(&ToolPolicy::new(), Some(&rt), "bash", call.clone())
                .await
                .is_some()
        );
        let second = check(&ToolPolicy::new(), Some(&rt), "bash", call).await;
        assert!(second.unwrap().contains("earlier decision"));
        assert_eq!(gate.asked(), 1);
    }

    #[tokio::test]
    async fn every_way_of_not_getting_an_answer_denies() {
        for (err, needle) in [
            (ApprovalError::NoClient, "no client is attached"),
            (ApprovalError::TimedOut, "timed out"),
            (ApprovalError::Cancelled, "run was cancelled"),
        ] {
            let rt = ApprovalRuntime::new(FakeGate::new(Err(err)), GatedSet::All);
            let reason = check(
                &ToolPolicy::new(),
                Some(&rt),
                "bash",
                json!({ "command": "ls" }),
            )
            .await
            .unwrap_or_else(|| panic!("{err:?} must deny"));
            assert!(reason.contains(needle), "{err:?} -> {reason}");
        }
    }

    #[tokio::test]
    async fn a_failure_to_answer_is_never_remembered() {
        // Otherwise one transient "no client attached" would poison the tool for the whole session.
        let gate = FakeGate::new(Err(ApprovalError::NoClient));
        let rt = ApprovalRuntime::new(gate.clone(), GatedSet::All);
        let call = json!({ "command": "ls" });
        check(&ToolPolicy::new(), Some(&rt), "bash", call.clone()).await;
        check(&ToolPolicy::new(), Some(&rt), "bash", call).await;
        assert_eq!(gate.asked(), 2);
        assert_eq!(rt.memory.lookup("bash", "cmd:ls"), None);
    }

    #[tokio::test]
    async fn the_origin_reaches_the_gate_so_a_ui_can_say_which_child_is_asking() {
        struct Capture(Mutex<Option<ApprovalOrigin>>);
        #[async_trait]
        impl ApprovalGate for Capture {
            async fn request(
                &self,
                req: ApprovalRequest,
                _c: &CancellationToken,
            ) -> Result<ApprovalDecision, ApprovalError> {
                *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(req.origin);
                allow(ApprovalScope::Once)
            }
        }
        let gate = Arc::new(Capture(Mutex::new(None)));
        let rt = ApprovalRuntime::new(gate.clone(), GatedSet::All);
        let origin = ApprovalOrigin::Subagent {
            agent: "reviewer".into(),
            spawn_id: "7".into(),
        };
        gated_before_tool_call(
            &ToolPolicy::new(),
            Some(&rt),
            &origin,
            "bash",
            &json!({ "command": "ls" }),
            &Session::new(),
            &CancellationToken::new(),
        )
        .await;
        assert_eq!(
            gate.0.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            Some(origin)
        );
    }

    #[test]
    fn an_origin_serializes_for_the_wire() {
        assert_eq!(ApprovalOrigin::Main.to_json(), json!("main"));
        assert_eq!(
            ApprovalOrigin::Subagent {
                agent: "reviewer".into(),
                spawn_id: "7".into()
            }
            .to_json(),
            json!({ "agent": "reviewer", "spawn_id": "7" })
        );
    }
}
