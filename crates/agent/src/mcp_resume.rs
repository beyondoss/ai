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

/// Custom-entry kind marking a `tool_use` whose `tool_result` on the path is a "result pending"
/// placeholder (see [`pending_placeholder`]), not an answer. A placeholder is known by this entry,
/// never by its content, so a real result that happens to read like one is still an answer.
pub const PLACEHOLDER_ENTRY_KIND: &str = "mcp_task_placeholder";

/// The journal kinds [`JournalAuth`] seals and checks.
fn is_journal_kind(kind: &str) -> bool {
    kind == TASK_ENTRY_KIND || kind == RESULT_ENTRY_KIND || kind == PLACEHOLDER_ENTRY_KIND
}

/// How one session's journal entries are authenticated on replay. Held by the session's store,
/// which seals what it journals ([`seal`](Self::seal)) and replays only what passes
/// ([`accepts`](Self::accepts)).
///
/// - **Storage**: the store already authenticates every line (service mode's segments, sealed under
///   the tenant's key, which is strictly stronger than a MAC). No MAC, so any replica holding the
///   tenant key reads the journal.
/// - **Keyed**: a key that lives *with the session*, in a `0600` sidecar ([`SIDECAR`](Self::SIDECAR))
///   that moves, trashes and restores with it, so the journal resumes wherever the session goes:
///   another machine, a fresh `$HOME`, an upgrade. It is made by the session's first journal write
///   (a session that never journals gets no file). Each entry carries a `mac` (HMAC-SHA256 over its
///   kind and content). `legacy` holds the MACs of the entries the session already had when its key
///   was made (written before per-session keys): accepted, so a session that resumed before still
///   does.
/// - **Unkeyed**: no key yet (the session has not journaled since per-session keys, or its key
///   could not be written): entries are accepted as they were before journals were sealed, and the
///   next journal write makes the key.
///
/// What a key does and does not stop, honestly: a session's `.jsonl` is ordinary file content, and
/// a model with the ungated `write`/`edit` tools can append lines to it. A line it plants without
/// the key carries no valid `mac` and is ignored on replay, so it neither causes a poll nor is
/// delivered as a tool result. But the key is a file beside the session that the agent's own user
/// can read, so a model that can also read files and goes looking can forge an entry. A model with
/// `write` can also delete the key or overwrite it with anything that does not parse: the session
/// is then unkeyed (what is there is accepted) until its next journal write makes a new key, which
/// accepts every line already in the file, planted ones included; and the transcript itself
/// is unauthenticated, so a model that can write the file can always plant an ordinary
/// `tool_result`. Locally this raises the bar from "append a line" to a deliberate read-then-forge,
/// not more. In service mode the store is out of the tools' reach (they run in the tenant's
/// sandbox), and forgery is closed.
#[derive(Clone, Debug)]
pub enum JournalAuth {
    Storage,
    Keyed {
        key: [u8; 32],
        legacy: Arc<HashSet<String>>,
    },
    Unkeyed,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl JournalAuth {
    /// The sidecar's name: a suffix on a single-file session's whole file name
    /// (`<session>.jsonl.mcp-task-journal.json`), or this name inside a segmented session's directory
    /// name inside a segmented session's directory (one without a tenant codec).
    pub const SIDECAR: &'static str = "mcp-task-journal.json";

    /// The auth a sidecar's bytes describe; `None` when they describe none.
    pub fn from_sidecar(bytes: &[u8]) -> Option<Self> {
        let v: Value = serde_json::from_slice(bytes).ok()?;
        let text = v.get("key")?.as_str()?;
        if text.len() != 64 {
            return None;
        }
        let mut key = [0u8; 32];
        for (i, byte) in key.iter_mut().enumerate() {
            *byte = u8::from_str_radix(text.get(i * 2..i * 2 + 2)?, 16).ok()?;
        }
        let legacy = v
            .get("legacy")
            .and_then(Value::as_array)
            .map(|macs| {
                macs.iter()
                    .filter_map(|m| m.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        Some(Self::Keyed {
            key,
            legacy: Arc::new(legacy),
        })
    }

    /// A new key for a session whose entries so far are `existing` (`(kind, data)` of every custom
    /// entry it holds, on any branch): its journal entries among them stay accepted. Returns the
    /// auth and the sidecar bytes to write; `None` if no randomness is available.
    pub fn fresh<'a>(
        existing: impl IntoIterator<Item = (&'a str, &'a Value)>,
    ) -> Option<(Self, Vec<u8>)> {
        let mut key = [0u8; 32];
        getrandom::fill(&mut key).ok()?;
        let legacy: HashSet<String> = existing
            .into_iter()
            .filter(|(kind, _)| is_journal_kind(kind))
            .filter_map(|(kind, data)| mac_of(&key, kind, data))
            .collect();
        let mut sorted: Vec<&String> = legacy.iter().collect();
        sorted.sort();
        let bytes = serde_json::to_vec(&json!({ "key": hex(&key), "legacy": sorted })).ok()?;
        Some((
            Self::Keyed {
                key,
                legacy: Arc::new(legacy),
            },
            bytes,
        ))
    }

    /// Bind a journal entry to its session before it is written (a `mac`, when keyed).
    pub fn seal(&self, kind: &str, data: &mut Value) {
        if let Self::Keyed { key, .. } = self
            && let Some(mac) = mac_of(key, kind, data)
            && let Value::Object(map) = data
        {
            map.insert("mac".into(), json!(mac));
        }
    }

    /// Whether a journal entry read back may be replayed: planted or tampered entries are not.
    pub fn accepts(&self, kind: &str, data: &Value) -> bool {
        let Self::Keyed { key, legacy } = self else {
            return true;
        };
        let Some(want) = mac_of(key, kind, data) else {
            return false;
        };
        let sealed = data.get("mac").and_then(Value::as_str).is_some_and(|got| {
            got.len() == want.len()
                && got
                    .bytes()
                    .zip(want.bytes())
                    .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                    == 0
        });
        sealed || legacy.contains(&want)
    }
}

fn mac_of(key: &[u8; 32], kind: &str, data: &Value) -> Option<String> {
    use hmac::{Hmac, Mac};
    let mut body = data.clone();
    if let Value::Object(map) = &mut body {
        map.remove("mac");
    }
    let mut mac = Hmac::<sha2::Sha256>::new_from_slice(key).ok()?;
    mac.update(kind.as_bytes());
    mac.update(b"\n");
    mac.update(serde_json::to_string(&body).ok()?.as_bytes());
    Some(hex(&mac.finalize().into_bytes()))
}

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
    /// Not an answer: the server could not be reached, so the call is told its result is pending
    /// (see [`pending_placeholder`]). Never journaled as a result; the task stays pending and is
    /// tried again.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pending: bool,
}

/// How a "result pending" placeholder `tool_result` reads, for the model. Only the text: what makes
/// a `tool_result` a placeholder is a [`PLACEHOLDER_ENTRY_KIND`] entry for its call, never this.
pub const PENDING_PREFIX: &str = "[MCP task result pending] ";

/// The placeholder answer for a task whose server could not be reached. Whoever puts it on the path
/// journals [`placeholder_entry`] for it, so the call stays unanswered and the real result replaces
/// it when it lands.
pub fn pending_placeholder(task: &PendingTask, why: &str) -> Resolved {
    Resolved {
        tool_use_id: task.tool_use_id.clone(),
        name: task.name.clone(),
        content: format!(
            "{PENDING_PREFIX}{why}. Task `{}` may still be running on the server; its result will \
             be delivered on a later turn once the server is reachable.",
            task.record.task_id
        ),
        is_error: true,
        images: Vec::new(),
        pending: true,
    }
}

/// The journal data marking `tool_use_id`'s `tool_result` as a placeholder.
pub fn placeholder_entry(tool_use_id: &str, session_id: &str) -> Value {
    json!({ "toolUseId": tool_use_id, "sessionId": session_id })
}

/// The `tool_use` ids whose `tool_result` on the path is a placeholder, from the journal's
/// [`PLACEHOLDER_ENTRY_KIND`] entries.
pub fn placeholder_ids(journal: &[Value]) -> HashSet<String> {
    journal
        .iter()
        .filter_map(|entry| entry["toolUseId"].as_str().map(str::to_owned))
        .collect()
}

/// The journal data for a task record emitted by tool `tool_use_id` in session `session_id`.
pub fn task_entry(record: &Value, tool_use_id: &str, session_id: &str) -> Value {
    let mut data = record.clone();
    data["toolUseId"] = json!(tool_use_id);
    data["sessionId"] = json!(session_id);
    data
}

/// Every `tool_use` id answered on the path: by a `tool_result` that is not a placeholder, or by a
/// journaled result.
fn answered_ids(
    messages: &[Message],
    results: &[Resolved],
    placeholders: &HashSet<String>,
) -> HashSet<String> {
    let mut ids: HashSet<String> = results.iter().map(|r| r.tool_use_id.clone()).collect();
    for message in messages {
        for block in &message.content {
            if let ContentBlock::ToolResult { tool_use_id, .. } = block
                && !placeholders.contains(tool_use_id)
            {
                ids.insert(tool_use_id.clone());
            }
        }
    }
    ids
}

/// Journaled results on the active path, oldest first. `journal` holds only entries the session's
/// store accepted (see [`JournalAuth`]).
pub fn results(journal: &[Value]) -> Vec<Resolved> {
    journal
        .iter()
        .filter_map(|entry| serde_json::from_value(entry.clone()).ok())
        .collect()
}

/// The journaled tasks of `session_id` that nothing on the path answers yet. `tasks` holds only
/// entries the session's store accepted (see [`JournalAuth`]).
pub fn pending(
    messages: &[Message],
    tasks: &[Value],
    results: &[Resolved],
    placeholders: &HashSet<String>,
    session_id: &str,
) -> Vec<PendingTask> {
    let answered = answered_ids(messages, results, placeholders);
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
            let record: McpTaskRecord = serde_json::from_value(entry).ok()?;
            // The record must describe the call it claims to answer: the same server and tool.
            if *name != crate::tools::mcp::registered_name(&record.server, &record.tool) {
                return None;
            }
            Some(PendingTask {
                name: (*name).to_owned(),
                record,
                tool_use_id: id,
            })
        })
        .collect()
}

/// Put each result's `tool_result` at the front of the first user turn after the assistant turn
/// holding its `tool_use`: exactly where the model would have seen it, with no turn added, so role
/// alternation holds whether the call is at the tip (the new prompt's turn) or further back (an
/// aborted prompt's turn). A result already answered on the path, or with no user turn after its
/// call yet, is skipped. `placeholders` names the calls whose `tool_result` is a placeholder (see
/// [`placeholder_ids`]): the real result replaces it where it is. Returns the results spliced.
pub fn splice(
    messages: &mut [Message],
    results: &[Resolved],
    placeholders: &HashSet<String>,
) -> Vec<Resolved> {
    let answered = answered_ids(messages, &[], placeholders);
    let mut spliced = Vec::new();
    for result in results {
        if answered.contains(&result.tool_use_id) {
            continue;
        }
        // A placeholder already in place: the real result replaces it where it is (once: a block
        // already holding this result is left alone); a second placeholder adds nothing.
        let placeholder = messages
            .iter_mut()
            .flat_map(|m| m.content.iter_mut())
            .find(|b| {
                matches!(b, ContentBlock::ToolResult { tool_use_id, .. }
                if *tool_use_id == result.tool_use_id)
            });
        if let Some(block) = placeholder {
            let applied = matches!(block, ContentBlock::ToolResult { content, is_error, .. }
                if *is_error == result.is_error && &**content == result.content.as_str());
            if !result.pending && !applied {
                *block = ContentBlock::ToolResult {
                    tool_use_id: result.tool_use_id.clone(),
                    content: result.content.clone().into(),
                    is_error: result.is_error,
                    images: result.images.clone(),
                };
                spliced.push(result.clone());
            }
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
                // A pending placeholder is not an answer: the task stays journaled as pending, and
                // the placeholder is marked as one wherever it lands on the path.
                if resolved.pending {
                    let _ = journal.send((
                        PLACEHOLDER_ENTRY_KIND.to_owned(),
                        placeholder_entry(&resolved.tool_use_id, &session_id),
                    ));
                } else {
                    let mut data = serde_json::to_value(&resolved).unwrap_or_default();
                    data["sessionId"] = json!(session_id);
                    let _ = journal.send((RESULT_ENTRY_KIND.to_owned(), data));
                }
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

    /// Whether this resumer has finished and some task's server was unreachable, so a fresh
    /// resumer should try again (at the next prompt).
    pub fn needs_retry(&self) -> bool {
        self.state
            .lock()
            .is_ok_and(|s| s.outstanding.is_empty() && s.done.iter().any(|r| r.pending))
    }

    /// The definitive results this resumer resolved (journaled, or about to be).
    pub fn finals(&self) -> Vec<Resolved> {
        self.state
            .lock()
            .map(|s| s.done.iter().filter(|r| !r.pending).cloned().collect())
            .unwrap_or_default()
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
    match result {
        Ok(output) => Resolved {
            tool_use_id: task.tool_use_id.clone(),
            name: task.name.clone(),
            content: output.text,
            is_error: false,
            images: output.images,
            pending: false,
        },
        Err(crate::tools::mcp::ResumeError::Unreachable(why)) => pending_placeholder(task, &why),
        Err(crate::tools::mcp::ResumeError::Final(e)) => Resolved {
            tool_use_id: task.tool_use_id.clone(),
            name: task.name.clone(),
            content: e.to_string(),
            is_error: true,
            images: Vec::new(),
            pending: false,
        },
    }
}

/// The prompt side of the policy (see the module doc): wait, visibly and abortably, for every task
/// the resumer is still polling, then splice every resolved result into the transcript and report
/// each through the run's own event sink (a `tool_start` and a `tool_end`, so the session's live
/// stats and lifecycle observer see the call finish). An abort returns early with nothing spliced;
/// the tasks keep polling and the next prompt waits again. `placeholders` are the journal's
/// placeholder marks (see [`placeholder_ids`]); this resumer's own are added.
pub async fn resume_into_turn<F: FnMut(AgentEvent)>(
    session: &mut agent_core::Session,
    resumer: &Resumer,
    journaled: &[Resolved],
    placeholders: &HashSet<String>,
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
    let mut placeholders = placeholders.clone();
    placeholders.extend(
        done.iter()
            .filter(|r| r.pending)
            .map(|r| r.tool_use_id.clone()),
    );
    let mut results: Vec<Resolved> = journaled.to_vec();
    for result in done {
        if !results.iter().any(|r| r.tool_use_id == result.tool_use_id) {
            results.push(result);
        }
    }
    // Only touch the transcript when a result has somewhere to go: `make_mut` copies a shared
    // history, and an untouched turn must leave `session.messages` exactly as it was.
    let answered = answered_ids(&session.messages, &[], &placeholders);
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
    let spliced = splice(messages, &results, &placeholders);
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

    fn none() -> HashSet<String> {
        HashSet::new()
    }

    fn keyed() -> JournalAuth {
        JournalAuth::fresh([]).unwrap().0
    }

    /// Under a per-session key, a journal line planted without it (or altered after sealing) is
    /// refused; an entry sealed under another session's key is too.
    #[test]
    fn a_keyed_journal_refuses_unsealed_tampered_and_foreign_entries() {
        let auth = keyed();
        let planted = task_entry(&record("t1"), "tu1", "s");
        assert!(!auth.accepts(TASK_ENTRY_KIND, &planted));
        let mut good = planted.clone();
        auth.seal(TASK_ENTRY_KIND, &mut good);
        assert!(auth.accepts(TASK_ENTRY_KIND, &good));
        // The MAC covers the kind too: a sealed record is not a sealed result.
        assert!(!auth.accepts(RESULT_ENTRY_KIND, &good));
        let mut tampered = good.clone();
        tampered["taskId"] = json!("someone-elses");
        assert!(!auth.accepts(TASK_ENTRY_KIND, &tampered));
        assert!(!keyed().accepts(TASK_ENTRY_KIND, &good));
    }

    /// The key travels as the sidecar's bytes; the entries a pre-key session already held when its
    /// key was made stay accepted (once made, the legacy set never grows).
    #[test]
    fn a_fresh_key_round_trips_through_its_sidecar_and_keeps_existing_entries() {
        let old = task_entry(&record("t1"), "tu1", "s");
        let custom = json!({ "anything": 1 });
        let (auth, bytes) =
            JournalAuth::fresh([(TASK_ENTRY_KIND, &old), ("someone_elses_kind", &custom)]).unwrap();
        assert!(auth.accepts(TASK_ENTRY_KIND, &old));
        let JournalAuth::Keyed { legacy, .. } = &auth else {
            panic!("keyed")
        };
        assert_eq!(legacy.len(), 1, "only journal kinds are carried over");
        let reread = JournalAuth::from_sidecar(&bytes).unwrap();
        assert!(reread.accepts(TASK_ENTRY_KIND, &old));
        let mut sealed = task_entry(&record("t2"), "tu2", "s");
        auth.seal(TASK_ENTRY_KIND, &mut sealed);
        assert!(reread.accepts(TASK_ENTRY_KIND, &sealed));
        assert!(!reread.accepts(TASK_ENTRY_KIND, &task_entry(&record("t3"), "tu3", "s")));
        assert!(JournalAuth::from_sidecar(b"{\"key\":\"short\"}").is_none());
    }

    /// Sealed storage authenticates every line itself: nothing is sealed or refused on top.
    #[test]
    fn storage_and_unkeyed_journals_accept_entries_as_they_are() {
        for auth in [JournalAuth::Storage, JournalAuth::Unkeyed] {
            let mut entry = task_entry(&record("t1"), "tu1", "s");
            auth.seal(TASK_ENTRY_KIND, &mut entry);
            assert!(entry.get("mac").is_none());
            assert!(auth.accepts(TASK_ENTRY_KIND, &entry));
        }
    }

    /// A placeholder is known by its journal mark, never by its text: a real result that happens to
    /// begin like one answers its call, and a marked one does not.
    #[test]
    fn a_placeholder_is_known_by_its_mark_not_its_content() {
        let lookalike = format!("{PENDING_PREFIX}but this is the real answer");
        let messages = vec![
            Message::user("go"),
            call("tu1"),
            Message::tool_result("tu1", lookalike.as_str(), false),
        ];
        let journal = vec![task_entry(&record("t1"), "tu1", "s")];
        assert!(pending(&messages, &journal, &[], &none(), "s").is_empty());
        let marked: HashSet<String> = placeholder_ids(&[placeholder_entry("tu1", "s")]);
        assert_eq!(pending(&messages, &journal, &[], &marked, "s").len(), 1);

        // The real result replaces a marked placeholder where it is, once; an unmarked look-alike
        // is an answer and stays.
        let mut spliced = messages.clone();
        assert!(splice(&mut spliced, &[resolved("tu1")], &none()).is_empty());
        assert_eq!(spliced, messages);
        assert_eq!(splice(&mut spliced, &[resolved("tu1")], &marked).len(), 1);
        assert!(matches!(
            &spliced[2].content[0],
            ContentBlock::ToolResult { content, .. } if &**content == "done"
        ));
        assert_eq!(spliced[2].content.len(), 1);
        assert!(splice(&mut spliced, &[resolved("tu1")], &marked).is_empty());
    }

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
            pending: false,
        }
    }

    /// A record that does not describe the call it names (another server or tool) is not
    /// resumed: whoever wrote it, it cannot make the resumer poll something the call never was.
    #[test]
    fn pending_ignores_a_record_that_does_not_match_its_call() {
        let messages = vec![Message::user("go"), call("tu1")];
        let mut other = record("t1");
        other["server"] = json!("elsewhere");
        let journal = vec![(task_entry(&other, "tu1", "s"))];
        assert!(pending(&messages, &journal, &[], &none(), "s").is_empty());
        let journal = vec![(task_entry(&record("t1"), "tu1", "s"))];
        assert_eq!(pending(&messages, &journal, &[], &none(), "s").len(), 1);
    }

    /// F9: only the session that journaled a task may resume it; a copy of the journal under
    /// another session id (a fork, a copied file) leaves the call to the generic repair.
    #[test]
    fn pending_ignores_tasks_journaled_by_another_session() {
        let messages = vec![Message::user("go"), call("tu1")];
        let journal = vec![(task_entry(&record("t1"), "tu1", "parent"))];
        assert!(pending(&messages, &journal, &[], &none(), "fork").is_empty());
        assert_eq!(
            pending(&messages, &journal, &[], &none(), "parent").len(),
            1
        );
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
        let journal = vec![(task_entry(&record("t1"), "tu1", "s"))];
        let found = pending(&messages, &journal, &[], &none(), "s");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].record.task_id, "t1");
        assert!(pending(&messages, &journal, &[resolved("tu1")], &none(), "s").is_empty());
        messages[2].content.insert(
            0,
            ContentBlock::ToolResult {
                tool_use_id: "tu1".into(),
                content: "x".into(),
                is_error: false,
                images: Vec::new(),
            },
        );
        assert!(pending(&messages, &journal, &[], &none(), "s").is_empty());
    }

    /// F10: the latest record for a call wins (it carries the keys answered since creation).
    #[test]
    fn pending_uses_the_latest_record_for_a_call() {
        let messages = vec![Message::user("go"), call("tu1")];
        let mut later = record("t1");
        later["answered"] = json!(["name"]);
        let journal = vec![
            (task_entry(&record("t1"), "tu1", "s")),
            (task_entry(&later, "tu1", "s")),
        ];
        let found = pending(&messages, &journal, &[], &none(), "s");
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
        let spliced = splice(&mut messages, &[resolved("tu1")], &none());
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
        assert!(splice(&mut messages, &[resolved("tu1")], &none()).is_empty());
    }
}
