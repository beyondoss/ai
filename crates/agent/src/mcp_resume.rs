//! Durable SEP-2663 tasks across a `serve` restart.
//!
//! The spec asks a client to persist task ids so polling can resume after a crash or restart. The
//! pieces:
//!
//! - **Journal.** When a task is created (and whenever its durable state changes) the MCP tool
//!   emits its [`McpTaskRecord`] in a `tool_progress`. `serve` appends it to the session as an
//!   [`TASK_ENTRY_KIND`] custom entry, stamped with the owning session id and the `tool_use` id.
//!   When a resumed task resolves, its result is appended as a [`RESULT_ENTRY_KIND`] entry.
//! - **Pending.** A journaled task is pending when its record belongs to *this* session (a fork
//!   inherits the transcript, not the right to answer the parent's task), its `tool_use` is on the
//!   active path, and nothing on the path, journal included, answers it.
//! - **Resumer.** Started when the session is loaded, it polls every pending task in the background
//!   (in-task input reaches the session's host as usual), streams `tool_progress` for each, and
//!   journals each result the moment it lands. It never cancels a task.
//! - **Prompt policy.** A prompt that arrives while a resumed task is outstanding waits for it,
//!   visibly (a `tool_start` and a "waiting" `tool_progress` per task, then the task's own
//!   progress), and abortably: an abort ends the prompt, not the task, which keeps polling and is
//!   waited on again by the next prompt. Every resolved result is then spliced in front of the
//!   first user turn after the assistant turn holding its `tool_use` (see [`splice`]), so role
//!   alternation holds wherever the call sits on the path, and a `tool_end` is emitted.
//!
//! Custom entries contribute nothing to the materialized messages, so the journal never reaches the
//! model; and since a splice into an older turn is not an append, it is reapplied from the journal
//! at every prompt rather than rewritten into the file.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use agent_core::{AgentEvent, CancellationToken, ContentBlock, ImageSource, Message, Role};
use agent_core::{ToolProgress, ToolUpdate};
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};

use crate::tools::mcp::{MCP_TASK_DETAILS_KEY, McpCatalog, McpTaskRecord, with_session_host};
use crate::tools::mcp_host::McpHost;

/// Custom-entry kind of a journaled task record (the latest per `toolUseId` wins).
pub const TASK_ENTRY_KIND: &str = crate::tools::mcp::MCP_TASK_ENTRY_KIND;
/// Custom-entry kind of a resolved task's result.
pub const RESULT_ENTRY_KIND: &str = "mcp_task_result";

/// Where journal writes go: `(kind, data)`, `data` carrying `sessionId` so a write that outlives a
/// session switch is dropped instead of landing in the wrong session's file.
pub type JournalTx = mpsc::UnboundedSender<(String, Value)>;

/// How a resumer reports events (`tool_progress` frames) to the session's clients.
pub type EmitEvent = Arc<dyn Fn(AgentEvent) + Send + Sync>;

/// One journaled task whose `tool_use` nothing answers yet.
#[derive(Clone, Debug)]
pub struct PendingTask {
    pub tool_use_id: String,
    /// The registered tool name the model called (`mcp__<server>__<tool>`).
    pub name: String,
    pub record: McpTaskRecord,
}

/// A resolved task: what the model is told.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Resolved {
    pub tool_use_id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageSource>,
}

/// The journal data for a task record emitted by tool `tool_use_id` in session `session_id`.
pub fn task_entry(record: &Value, tool_use_id: &str, session_id: &str) -> Value {
    let mut data = record.clone();
    data["toolUseId"] = json!(tool_use_id);
    data["sessionId"] = json!(session_id);
    data
}

/// Every `tool_use` id answered on the path: by a `tool_result`, or by a journaled result.
fn answered_ids(messages: &[Message], results: &[Resolved]) -> HashSet<String> {
    let mut ids: HashSet<String> = results.iter().map(|r| r.tool_use_id.clone()).collect();
    for message in messages {
        for block in &message.content {
            if let ContentBlock::ToolResult { tool_use_id, .. } = block {
                ids.insert(tool_use_id.clone());
            }
        }
    }
    ids
}

/// Journaled results on the active path, oldest first.
pub fn results(journal: &[Value]) -> Vec<Resolved> {
    journal
        .iter()
        .filter_map(|entry| serde_json::from_value(entry.clone()).ok())
        .collect()
}

/// The journaled tasks of `session_id` that nothing on the path answers yet.
pub fn pending(
    messages: &[Message],
    tasks: &[Value],
    results: &[Resolved],
    session_id: &str,
) -> Vec<PendingTask> {
    let answered = answered_ids(messages, results);
    let calls: HashMap<&str, &str> = messages
        .iter()
        .filter(|m| m.role == Role::Assistant)
        .flat_map(|m| m.content.iter())
        .filter_map(|block| match block {
            ContentBlock::ToolUse { id, name, .. } => Some((id.as_str(), name.as_str())),
            _ => None,
        })
        .collect();
    // Latest record per tool_use wins (a later one carries answered keys or a moved TTL).
    let mut latest: Vec<(String, Value)> = Vec::new();
    for entry in tasks {
        if entry["sessionId"].as_str() != Some(session_id) {
            continue;
        }
        let Some(id) = entry["toolUseId"].as_str() else {
            continue;
        };
        match latest.iter_mut().find(|(seen, _)| seen == id) {
            Some(slot) => slot.1 = entry.clone(),
            None => latest.push((id.to_owned(), entry.clone())),
        }
    }
    latest
        .into_iter()
        .filter(|(id, _)| !answered.contains(id))
        .filter_map(|(id, entry)| {
            let name = calls.get(id.as_str())?;
            Some(PendingTask {
                name: (*name).to_owned(),
                record: serde_json::from_value(entry).ok()?,
                tool_use_id: id,
            })
        })
        .collect()
}

/// Put each result's `tool_result` at the front of the first user turn after the assistant turn
/// holding its `tool_use`: exactly where the model would have seen it, with no turn added, so role
/// alternation holds whether the call is at the tip (the new prompt's turn) or further back (an
/// aborted prompt's turn). A result already answered on the path, or with no user turn after its
/// call yet, is skipped. Returns the results spliced.
pub fn splice(messages: &mut [Message], results: &[Resolved]) -> Vec<Resolved> {
    let answered = answered_ids(messages, &[]);
    let mut spliced = Vec::new();
    for result in results {
        if answered.contains(&result.tool_use_id) {
            continue;
        }
        let Some(call) = messages.iter().position(|m| {
            m.role == Role::Assistant
                && m.content.iter().any(
                    |b| matches!(b, ContentBlock::ToolUse { id, .. } if *id == result.tool_use_id),
                )
        }) else {
            continue;
        };
        let Some(turn) = messages[call + 1..]
            .iter_mut()
            .find(|m| m.role == Role::User)
        else {
            continue;
        };
        turn.content.insert(
            0,
            ContentBlock::ToolResult {
                tool_use_id: result.tool_use_id.clone(),
                content: result.content.clone().into(),
                is_error: result.is_error,
                images: result.images.clone(),
            },
        );
        spliced.push(result.clone());
    }
    spliced
}

#[derive(Default)]
struct State {
    outstanding: Vec<PendingTask>,
    done: Vec<Resolved>,
}

/// Polls one session's pending tasks in the background. Dropping it stops the polling (a session
/// switch, or the session ending) without cancelling any task.
pub struct Resumer {
    session_id: String,
    live: Arc<std::sync::atomic::AtomicBool>,
    state: Arc<Mutex<State>>,
    notify: Arc<Notify>,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl Resumer {
    /// A resumer with nothing to do, for a session without pending tasks.
    pub fn idle(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            live: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            state: Arc::default(),
            notify: Arc::new(Notify::new()),
            handles: Vec::new(),
        }
    }

    pub fn start(
        session_id: impl Into<String>,
        pending: Vec<PendingTask>,
        catalog: McpCatalog,
        host: Arc<McpHost>,
        emit: EmitEvent,
        journal: JournalTx,
    ) -> Self {
        let mut resumer = Self::idle(session_id);
        // Once this resumer is dropped (a session switch), nothing it still has in flight may
        // reach the clients, which by then belong to another session.
        let emit: EmitEvent = {
            let live = resumer.live.clone();
            Arc::new(move |ev| {
                if live.load(std::sync::atomic::Ordering::Acquire) {
                    emit(ev);
                }
            })
        };
        if let Ok(mut state) = resumer.state.lock() {
            state.outstanding = pending.clone();
        }
        for task in pending {
            let state = resumer.state.clone();
            let notify = resumer.notify.clone();
            let catalog = catalog.clone();
            let task_host = host.clone();
            let emit = emit.clone();
            let journal = journal.clone();
            let session_id = resumer.session_id.clone();
            let work = async move {
                let resolved =
                    resolve(&task, &catalog, &task_host, &emit, &journal, &session_id).await;
                let mut data = serde_json::to_value(&resolved).unwrap_or_default();
                data["sessionId"] = json!(session_id);
                let _ = journal.send((RESULT_ENTRY_KIND.to_owned(), data));
                if let Ok(mut state) = state.lock() {
                    state
                        .outstanding
                        .retain(|t| t.tool_use_id != task.tool_use_id);
                    state.done.push(resolved);
                }
                notify.notify_waiters();
            };
            let scope = host.clone();
            resumer
                .handles
                .push(tokio::spawn(with_session_host(scope, work)));
        }
        resumer
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The tasks still being polled.
    pub fn outstanding(&self) -> Vec<PendingTask> {
        self.state
            .lock()
            .map(|s| s.outstanding.clone())
            .unwrap_or_default()
    }

    /// Wait until no task is outstanding, then return every result resolved so far.
    pub async fn wait(&self) -> Vec<Resolved> {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Ok(state) = self.state.lock()
                && state.outstanding.is_empty()
            {
                return state.done.clone();
            }
            notified.await;
        }
    }
}

impl Drop for Resumer {
    fn drop(&mut self) {
        self.live.store(false, std::sync::atomic::Ordering::Release);
        for handle in &self.handles {
            handle.abort();
        }
    }
}

/// Poll one task to its result, forwarding its progress and journaling its record updates.
async fn resolve(
    task: &PendingTask,
    catalog: &McpCatalog,
    host: &McpHost,
    emit: &EmitEvent,
    journal: &JournalTx,
    session_id: &str,
) -> Resolved {
    // A visible start, which is not a journal record (the task's record is already journaled).
    emit(AgentEvent::ToolProgress {
        id: task.tool_use_id.clone(),
        name: task.name.clone(),
        snapshot: format!(
            "resuming MCP task `{}` after a restart",
            task.record.task_id
        ),
        details: Some(json!({ "taskId": task.record.task_id, "status": "resuming" })),
    });
    let (tx, mut rx) = futures::channel::mpsc::unbounded();
    let progress = ToolProgress::new(
        tx,
        task.tool_use_id.clone(),
        task.name.clone(),
        CancellationToken::new(),
    );
    let forward = async {
        while let Some(update) = futures::StreamExt::next(&mut rx).await {
            if let ToolUpdate::Progress {
                id,
                name,
                snapshot,
                details,
            } = update
            {
                if let Some(record) = details.as_ref().and_then(|d| d.get(MCP_TASK_DETAILS_KEY)) {
                    let _ = journal.send((
                        TASK_ENTRY_KIND.to_owned(),
                        task_entry(record, &id, session_id),
                    ));
                }
                emit(AgentEvent::ToolProgress {
                    id,
                    name,
                    snapshot,
                    details,
                });
            }
        }
    };
    let work = async move {
        let result = catalog
            .resume_task(task.record.clone(), host, Some(&progress))
            .await;
        drop(progress);
        result
    };
    let (result, ()) = tokio::join!(work, forward);
    let (content, is_error, images) = match result {
        Ok(output) => (output.text, false, output.images),
        Err(e) => (e.to_string(), true, Vec::new()),
    };
    Resolved {
        tool_use_id: task.tool_use_id.clone(),
        name: task.name.clone(),
        content,
        is_error,
        images,
    }
}

/// The prompt side of the policy (see the module doc): wait, visibly and abortably, for every task
/// the resumer is still polling, then splice every resolved result into the transcript and report
/// each through the run's own event sink (a `tool_start` and a `tool_end`, so the session's live
/// stats and lifecycle observer see the call finish). An abort returns early with nothing spliced;
/// the tasks keep polling and the next prompt waits again.
pub async fn resume_into_turn<F: FnMut(AgentEvent)>(
    session: &mut agent_core::Session,
    resumer: &Resumer,
    journaled: &[Resolved],
    sink: &mut F,
    cancel: &CancellationToken,
) {
    let input_of = |messages: &[Message], id: &str| {
        messages
            .iter()
            .flat_map(|m| m.content.iter())
            .find_map(|b| match b {
                ContentBlock::ToolUse {
                    id: use_id, input, ..
                } if use_id == id => Some(input.clone()),
                _ => None,
            })
            .unwrap_or(Value::Null)
    };
    let outstanding = resumer.outstanding();
    for task in &outstanding {
        sink(AgentEvent::ToolStart {
            id: task.tool_use_id.clone(),
            name: task.name.clone(),
            input: input_of(&session.messages, &task.tool_use_id),
        });
        sink(AgentEvent::ToolProgress {
            id: task.tool_use_id.clone(),
            name: task.name.clone(),
            snapshot: format!(
                "waiting for MCP task `{}` (resumed after a restart) before this turn; abort to skip",
                task.record.task_id
            ),
            details: Some(json!({ "taskId": task.record.task_id, "status": "waiting" })),
        });
    }
    let done = tokio::select! {
        biased;
        () = cancel.cancelled() => return,
        done = resumer.wait() => done,
    };
    let mut results: Vec<Resolved> = journaled.to_vec();
    for result in done {
        if !results.iter().any(|r| r.tool_use_id == result.tool_use_id) {
            results.push(result);
        }
    }
    // Only touch the transcript when a result has somewhere to go: `make_mut` copies a shared
    // history, and an untouched turn must leave `session.messages` exactly as it was.
    let answered = answered_ids(&session.messages, &[]);
    let calls: HashSet<&str> = session
        .messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|b| match b {
            ContentBlock::ToolUse { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    results
        .retain(|r| calls.contains(r.tool_use_id.as_str()) && !answered.contains(&r.tool_use_id));
    if results.is_empty() {
        return;
    }
    let messages: &mut Vec<Message> = Arc::make_mut(&mut session.messages);
    let spliced = splice(messages, &results);
    for result in spliced {
        if !outstanding
            .iter()
            .any(|t| t.tool_use_id == result.tool_use_id)
        {
            sink(AgentEvent::ToolStart {
                id: result.tool_use_id.clone(),
                name: result.name.clone(),
                input: input_of(&session.messages, &result.tool_use_id),
            });
        }
        sink(AgentEvent::ToolEnd {
            id: result.tool_use_id,
            name: result.name,
            result: result.content,
            is_error: result.is_error,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(task_id: &str) -> Value {
        json!({
            "server": "s", "tool": "job", "taskId": task_id, "createdAtMs": 0, "ttlMs": null,
        })
    }

    fn call(id: &str) -> Message {
        Message::assistant(vec![ContentBlock::tool_use(id, "mcp__s__job", json!({}))])
    }

    fn resolved(id: &str) -> Resolved {
        Resolved {
            tool_use_id: id.into(),
            name: "mcp__s__job".into(),
            content: "done".into(),
            is_error: false,
            images: Vec::new(),
        }
    }

    /// F9: only the session that journaled a task may resume it; a copy of the journal under
    /// another session id (a fork, a copied file) leaves the call to the generic repair.
    #[test]
    fn pending_ignores_tasks_journaled_by_another_session() {
        let messages = vec![Message::user("go"), call("tu1")];
        let journal = vec![task_entry(&record("t1"), "tu1", "parent")];
        assert!(pending(&messages, &journal, &[], "fork").is_empty());
        assert_eq!(pending(&messages, &journal, &[], "parent").len(), 1);
    }

    /// F1: a journaled call is pending wherever it sits on the path, not only at the tip (an
    /// aborted prompt leaves its turn after it), and not once anything answers it.
    #[test]
    fn pending_finds_an_unanswered_call_anywhere_on_the_path() {
        let mut messages = vec![
            Message::user("go"),
            call("tu1"),
            Message::user("p1"),
            Message::assistant(Vec::new()),
        ];
        let journal = vec![task_entry(&record("t1"), "tu1", "s")];
        let found = pending(&messages, &journal, &[], "s");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].record.task_id, "t1");
        assert!(pending(&messages, &journal, &[resolved("tu1")], "s").is_empty());
        messages[2].content.insert(
            0,
            ContentBlock::ToolResult {
                tool_use_id: "tu1".into(),
                content: "x".into(),
                is_error: false,
                images: Vec::new(),
            },
        );
        assert!(pending(&messages, &journal, &[], "s").is_empty());
    }

    /// F10: the latest record for a call wins (it carries the keys answered since creation).
    #[test]
    fn pending_uses_the_latest_record_for_a_call() {
        let messages = vec![Message::user("go"), call("tu1")];
        let mut later = record("t1");
        later["answered"] = json!(["name"]);
        let journal = vec![
            task_entry(&record("t1"), "tu1", "s"),
            task_entry(&later, "tu1", "s"),
        ];
        let found = pending(&messages, &journal, &[], "s");
        assert_eq!(found[0].record.answered, vec!["name".to_owned()]);
    }

    /// F1/F3: a result for a call behind an aborted turn goes to the front of the first user turn
    /// after the call, adding no turn: roles still alternate.
    #[test]
    fn splice_answers_a_call_behind_an_aborted_turn_keeping_roles_alternating() {
        let mut messages = vec![
            Message::user("go"),
            call("tu1"),
            Message::user("p1"),
            Message::assistant(Vec::new()),
            Message::user("p2"),
        ];
        let spliced = splice(&mut messages, &[resolved("tu1")]);
        assert_eq!(spliced.len(), 1);
        assert_eq!(messages.len(), 5, "no turn added");
        assert!(matches!(
            &messages[2].content[0],
            ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "tu1"
        ));
        for pair in messages.windows(2) {
            assert_ne!(pair[0].role, pair[1].role);
        }
        // Idempotent: already answered, so a second splice adds nothing.
        assert!(splice(&mut messages, &[resolved("tu1")]).is_empty());
    }
}
