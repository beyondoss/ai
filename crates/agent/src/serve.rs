//! Headless `serve` — a newline-delimited JSON control protocol over stdio.
//!
//! The server is the source of truth; any client (a TUI, an editor, or an `ssh` pipe) drives it by
//! writing one JSON command per line to stdin and reading one JSON frame per line from stdout. The
//! shape mirrors pi's `rpc` mode and opencode's session server: commands get a `response` frame,
//! and a `prompt` streams `event` frames (the agent's `AgentEvent`s) before its response.
//!
//! Sessions persist as append-only JSONL: `--session-file` for one session, or `--session-dir` for a
//! [`SessionRepo`](crate::session_store::SessionRepo) of many. A turn appends only its new messages
//! (compaction rewrites atomically). A reattaching client sees a **stable** session id and metadata.
//!
//! Commands (stdin): `{id?, type, …}`
//!   - `{type:"prompt", message, streaming_behavior?}` run a turn: an immediate lightweight `ack`
//!     frame (the turn is queued and starting), then `event` frames, then a `response` whose `data`
//!     includes `refused: bool` — whether the run's last turn ended in a refusal rather than an
//!     ordinary stop (a refusal doesn't drain queued `steer`/`follow_up` messages; see `agent-core`).
//!     Sent while another `prompt` is already in flight, it's rejected as busy *unless*
//!     `streaming_behavior` is `"steer"` or `"follow_up"`, in which case `message` is queued through
//!     the same `Steering` lane an explicit `steer`/`follow_up` command would use.
//!   - `{type:"abort"}`                  cancel the in-flight `prompt` (if any), else a no-op ack
//!   - `{type:"stop_after_turn"}`        request a graceful stop at the next turn boundary — the
//!     current turn's tool calls (if any) still finish and their results are committed, unlike
//!     `abort`. A no-op ack when no `prompt` is in flight (never affects a *future* prompt).
//!   - `{type:"get_state"}`              → `data: {session_id, model, steps, message_count, title, …}`
//!   - `{type:"get_messages"}`           → `data: {messages: [...]}` (each tagged with its tree `id`
//!     when persistence is configured, so a client can fork from any point via `switch_branch`)
//!   - `{type:"new_session"}`            start a fresh session → `data: {session_id}`
//!   - `{type:"list_sessions"}`          (repo mode) → `data: {sessions: [SessionMeta…]}`, this
//!     project's sessions only (matched by the default per-cwd directory, or whatever `--session-dir`
//!     points at)
//!   - `{type:"list_all_sessions"}`      (repo mode) → `data: {sessions: [SessionMeta…]}` across every
//!     project's own session directory, not just this one's — each entry's own `cwd` field says which
//!     project it belongs to (pi's cross-project `listAll`)
//!   - `{type:"switch_session", session_id}` (repo mode) load another session
//!   - `{type:"fork", upto?}`            (repo mode) copy the prefix into a new session, switch to it
//!   - `{type:"get_fork_messages", session_id?, upto?}` (repo mode) preview what `fork` would produce —
//!     no new session, no switch
//!   - `{type:"set_session_name", title}` set the session's title
//!   - `{type:"compact"}`                summarize the prefix now → `data: {compacted: bool}`
//!   - `{type:"get_last_assistant_text"}` → `data: {text}` (the latest assistant reply)
//!   - `{type:"get_session_stats"}`      → token/step accounting
//!   - `{type:"get_commands"}`           → discoverable skills + prompt templates
//!   - `{type:"reload"}`                 re-run project-instruction/skill/prompt-template discovery and
//!     re-check trust, refreshing the static half of the system prompt (the cheap date/cwd footer is
//!     already refreshed every turn regardless)
//!   - `{type:"set_model", model}`       switch the model for subsequent prompts → `data: {model}`
//!   - `{type:"set_thinking", budget}`   set/clear the thinking budget (integer, or `null` to disable)
//!   - `{type:"cycle_model"}`            advance through `get_available_models`'s list, wrapping
//!   - `{type:"cycle_thinking_level"}`   advance through a fixed Off/Low/Medium/High budget ladder
//!   - `{type:"set_auto_compaction", enabled}` toggle threshold-triggered compaction (manual `compact`
//!     is unaffected either way)
//!   - `{type:"get_available_models"}`   → `data: {models: […]}` (a known, non-exhaustive id list)
//!   - `{type:"list_branches"}`          → `data: {branches: [BranchInfo…]}` (the session's *leaves*)
//!   - `{type:"get_tree"}`               → `data: {nodes: [TreeNode…]}` (every message on every
//!     branch, not just the leaves `list_branches` reports)
//!   - `{type:"switch_branch", target_id, summarize?}` navigate to another point in the tree,
//!     summarizing the abandoned branch's activity first unless `summarize:false`
//!   - `{type:"bash", command, cwd?, timeout_ms?}` run a shell command directly — independent of the
//!     model's own tool-call loop, and of any conversation/session state — streaming `tool_progress`/
//!     `tool_end` events exactly like a model-invoked `bash` call. Rejected if `bash` isn't registered
//!     for this process (`--exclude-tools bash` / `--no-tools`). While it runs, only `abort_bash`/
//!     `abort` (cancel it) are accepted; everything else is rejected as busy.
//!   - `{type:"abort_bash"}`             cancel an in-flight host `bash` command, else a no-op ack
//!
//! While a `prompt` runs, the loop keeps reading stdin so an `abort` can cancel it, or `steer` /
//! `follow_up` (with a `message`) can queue input: a `steer` is injected mid-run at the next tool
//! turn (to redirect a busy agent), a `follow_up` waits for the model to next stop. A handful of
//! read-only commands that don't need the run's exclusively-borrowed session — `get_state` (with
//! `message_count: null`, the one field that genuinely needs it), `get_session_stats` (from a live
//! mirror of the session's own counters, updated as the run streams), `get_commands`, `list_branches`,
//! `get_tree`, `list_sessions`, `list_all_sessions`, `get_available_models` — are answered live too,
//! rather than rejected, so a client can poll for progress during a long tool-heavy turn. Everything
//! else issued during a run is rejected as busy. `steer`/`follow_up` are also accepted while idle (no
//! `prompt` in flight) — they
//! queue against a persistent handle and are picked up by whichever `prompt` runs next;
//! `new_session`/`switch_session`/`fork`/`switch_branch` clear that queue so a message meant for one
//! session's next turn can't leak into another.
//!
//! Frames (stdout) are `{type:"ack", id?, command}`, `{type:"response", id?, command, success, data?,
//! error?}`, or `{type:"event", event: <AgentEvent>}`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use agent_core::{
    Agent, AgentEvent, CancellationToken, GatewayClient, Session, StopReason, StreamEvent,
    ToolUpdate,
};
use futures::StreamExt;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::session_store::{
    BranchInfo, BranchSummaryDetails, CompactionMeta, SessionMeta, SessionRepo, SessionStore,
};
use crate::tools;

/// Options for the headless server (mirrors `run`, plus persistence).
pub struct ServeConfig {
    pub gateway: String,
    pub key: String,
    pub model: String,
    pub max_steps: u32,
    /// Base system prompt (agent identity). Project instructions, skills, and env are layered on top.
    pub system: String,
    /// Extra instructions appended after the base prompt (`--append-system-prompt`).
    pub append_system: Option<String>,
    /// Whether to discover and inject `AGENTS.md`/`CLAUDE.md` project-instruction files.
    pub context_files: bool,
    /// Persist a single session to this JSONL file (append-per-turn). Mutually exclusive with
    /// `session_dir`.
    pub session_file: Option<String>,
    /// Persist many sessions under this directory (a [`SessionRepo`]). Enables the multi-session
    /// commands (`list_sessions`, `switch_session`, `fork`, `set_session_name`).
    pub session_dir: Option<String>,
    /// Skip persistence entirely, even though neither `session_file` nor `session_dir` was set —
    /// without this, `Persistence::open` defaults to a per-cwd directory under
    /// `~/.claude/sessions/<encoded-cwd>/` rather than silently running in-memory-only (an operator
    /// who simply forgot the flag would otherwise lose all history on restart with no indication why).
    /// An explicit opt-out for the cases that genuinely want pure in-memory operation (e.g. a
    /// short-lived test harness).
    pub no_session_persistence: bool,
    /// The model's context window in tokens — the budget the loop compacts against to stay below.
    /// `None` defers to the model's own capability-table window (`Agent::new`'s default), so switching
    /// to a model with a different real window (via `set_model`) gets that model's true budget instead
    /// of a stale operator-supplied number. `Some` pins a fixed budget that survives model switches.
    pub context_window: Option<u32>,
    /// Use the 1-hour prompt-cache TTL (vs the default 5 minutes) — useful when turns are spaced out.
    pub cache_long: bool,
    /// Extended-thinking token budget, when enabled (`None` leaves thinking off). Must be below the
    /// per-turn `max_tokens`.
    pub thinking: Option<u32>,
    /// Reasoning effort for models driven by an effort level rather than a token budget (OpenAI
    /// reasoning models, Anthropic adaptive-thinking models). `None` leaves the provider default.
    /// Fixed for the process — unlike `thinking`, there's no `set_reasoning_effort` RPC command.
    pub reasoning_effort: Option<agent_core::ReasoningEffort>,
    /// Trust the working directory for this run only, so a project-local `.claude/SYSTEM.md` is
    /// honored even if the directory isn't in the persisted allowlist (`agent trust <path>`). See
    /// `crate::trust_store`.
    pub trust_project: bool,
    /// Compaction headroom (tokens) reserved below the context window. `None` keeps
    /// `CompactionConfig::default()`'s value.
    pub compaction_reserve_tokens: Option<u32>,
    /// Roughly how many tokens of recent conversation compaction keeps verbatim. `None` keeps
    /// `CompactionConfig::default()`'s value.
    pub compaction_keep_recent_tokens: Option<u32>,
    /// How many times to retry a gateway request that fails before the first response byte arrives.
    /// `None` keeps the client's built-in default.
    pub retry_max_retries: Option<u32>,
    /// Base of the exponential backoff between those retries. `None` keeps the client's built-in
    /// default.
    pub retry_base_delay_ms: Option<std::time::Duration>,
    /// Default `bash` command timeout when the model omits `timeout_ms`. `None` keeps the tool's
    /// built-in default.
    pub bash_timeout_ms: Option<u64>,
    /// Restrict the tool set to exactly these names, dropping everything else. Combine with
    /// `exclude_tools` to carve one back out of the allow-list. Fixed for the process — like `system`,
    /// there's no runtime RPC to change it, but it does survive a `set_model`/`set_thinking` rebuild
    /// (`build_agent` reapplies it every time).
    pub tools: Option<Vec<String>>,
    /// Drop these tools from the default set — e.g. `["bash", "write"]` for a read-only reviewer.
    pub exclude_tools: Option<Vec<String>>,
    /// Register no tools at all. Wins over `tools`/`exclude_tools`.
    pub no_tools: bool,
}

/// Waits for an OS shutdown request (SIGTERM, or SIGINT/Ctrl-C) so `serve` can drain in-flight work
/// and persist before exiting — the same graceful-shutdown path stdin closing already takes (cancel
/// the run, persist, break), just with the process's terminate signal as a second trigger. Without
/// this, a `systemctl restart`/`docker stop`/pod eviction mid-turn hits Rust's default disposition —
/// immediate termination, no destructors run — losing that turn's unpersisted messages and orphaning
/// any backgrounded child process `exec`'s `kill_on_drop` cleanup depends on `Drop` running to reap.
struct ShutdownSignal {
    #[cfg(unix)]
    sigterm: tokio::signal::unix::Signal,
}

impl ShutdownSignal {
    fn new() -> std::io::Result<Self> {
        #[cfg(unix)]
        {
            Ok(Self {
                sigterm: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
            })
        }
        #[cfg(not(unix))]
        {
            Ok(Self {})
        }
    }

    /// Resolves once a shutdown signal arrives. Safe to call fresh on every loop iteration — both
    /// `Signal::recv` and `tokio::signal::ctrl_c` are re-armable, not one-shot.
    async fn wait(&mut self) {
        #[cfg(unix)]
        {
            tokio::select! {
                _ = self.sigterm.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}

/// Runs [`Persistence::persist`]'s blocking file I/O (`sync_all`, directory `fsync`) on tokio's
/// blocking thread pool instead of `serve`'s single async control-loop task. `persist` runs after
/// every turn (and every manual `compact`), so on a slow or network-backed session directory, a
/// multi-ms-to-100ms `sync_all` would otherwise stall that task directly — delaying its stdin
/// `select!` loop, and so `abort`/`steer` responsiveness, for the duration. `persistence` and `session`
/// are moved in and handed back so the caller can keep using them (`Session` is cheap to clone: its
/// `messages` field is an `Arc`, so this is a pointer clone, not a deep copy of the transcript).
async fn persist_blocking(
    mut persistence: Persistence,
    session: Session,
    tokens_before: Option<u32>,
) -> Persistence {
    // `persist` never panics itself (it handles its own I/O errors internally), and this task is
    // never cancelled (always awaited directly, never `.abort()`ed) — so a `JoinError` here can only
    // mean the closure panicked. Re-raise that panic rather than `.expect()`ing (denied by the
    // workspace's panic-surface lints) on what would otherwise look like an ordinary recoverable
    // error.
    match tokio::task::spawn_blocking(move || {
        persistence.persist(&session, tokens_before);
        persistence
    })
    .await
    {
        Ok(persistence) => persistence,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

/// Like [`persist_blocking`], but for one incremental mid-run checkpoint (see [`ChannelCheckpoint`]):
/// takes just the message snapshot a checkpoint carries rather than a whole `Session`, and always
/// appends (never a compacted rewrite — see [`Persistence::persist_messages`]).
async fn persist_messages_blocking(
    mut persistence: Persistence,
    messages: Arc<Vec<agent_core::Message>>,
) -> Persistence {
    match tokio::task::spawn_blocking(move || {
        persistence.persist_messages(&messages);
        persistence
    })
    .await
    {
        Ok(persistence) => persistence,
        Err(e) => std::panic::resume_unwind(e.into_panic()),
    }
}

/// A [`agent_core::CheckpointHook`] that forwards each mid-run checkpoint's message snapshot through an
/// unbounded channel rather than persisting it directly — sending is instant (never blocks on I/O), so
/// it's safe to call from deep inside `Agent::run_events_steered`'s hot loop without stalling it. The
/// `"prompt"` arm's busy-loop (the only place a checkpoint can fire from) drains the receiving half and
/// does the actual (blocking) append.
struct ChannelCheckpoint(mpsc::UnboundedSender<Arc<Vec<agent_core::Message>>>);

#[async_trait::async_trait]
impl agent_core::CheckpointHook for ChannelCheckpoint {
    async fn checkpoint(&self, session: &Session) {
        // Best-effort: if the receiving end is gone (a prior run already dropped it — shouldn't happen
        // while a run is in flight, since the receiver outlives every `"prompt"` arm) there's nothing to
        // recover, and the run itself must not fail just because incremental persistence couldn't.
        let _ = self.0.send(session.messages.clone());
    }
}

/// Where the server persists sessions: a multi-session [`SessionRepo`] (`--session-dir`), a single
/// JSONL file (`--session-file`), or nowhere (in-memory). It always carries the current session's
/// [`SessionMeta`] so the session id is stable across reattaches.
struct Persistence {
    repo: Option<SessionRepo>,
    store: Option<SessionStore>,
    meta: SessionMeta,
}

impl Persistence {
    /// Open persistence and restore (or create) the active session. In repo mode, reopens the most
    /// recent session or creates a fresh one; in file mode, opens the file or creates it.
    fn open(cfg: &ServeConfig) -> std::io::Result<(Self, Session)> {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Some(dir) = &cfg.session_dir {
            return Self::open_repo(dir, &cwd, &cfg.model);
        }
        if let Some(path) = &cfg.session_file {
            let path = std::path::PathBuf::from(path);
            let (store, session) = if path.exists() {
                SessionStore::open(path)?
            } else {
                let store = SessionStore::create(path, SessionMeta::new(&cwd, &cfg.model))?;
                (store, Session::new())
            };
            let meta = store.meta().clone();
            return Ok((
                Self {
                    repo: None,
                    store: Some(store),
                    meta,
                },
                session,
            ));
        }
        if cfg.no_session_persistence {
            return Ok((
                Self {
                    repo: None,
                    store: None,
                    meta: SessionMeta::new(&cwd, &cfg.model),
                },
                Session::new(),
            ));
        }
        // Neither flag was given and persistence wasn't explicitly opted out: default to a per-cwd
        // repo directory rather than silently running in-memory-only, so an operator who simply
        // forgot the flag doesn't lose all history on the next restart with no indication why.
        Self::open_repo(default_session_dir(&cwd), &cwd, &cfg.model)
    }

    /// Open (creating if needed) a multi-session repo at `dir` and reattach to the most recent session
    /// whose recorded cwd matches `cwd` — not just the globally newest one, so a shared `--session-dir`
    /// spanning multiple projects (or the shared default directory before cwd-encoding existed) doesn't
    /// resume a stranger's unrelated session. No match (a fresh directory, or one with no session for
    /// this cwd yet) creates a new one.
    fn open_repo(
        dir: impl Into<std::path::PathBuf>,
        cwd: &str,
        model: &str,
    ) -> std::io::Result<(Self, Session)> {
        let repo = SessionRepo::open(dir)?;
        let (store, session) = match repo.list()?.into_iter().find(|m| m.cwd == cwd) {
            Some(meta) => repo.open_id(&meta.id)?,
            None => {
                let store = repo.create(SessionMeta::new(cwd, model))?;
                (store, Session::new())
            }
        };
        let meta = store.meta().clone();
        Ok((
            Self {
                repo: Some(repo),
                store: Some(store),
                meta,
            },
            session,
        ))
    }

    fn session_id(&self) -> &str {
        &self.meta.id
    }

    /// Persist the session after a turn: non-destructively rewrite the transcript (see
    /// `SessionStore::rewrite_compacted`) if compaction fired this round, otherwise append just the
    /// new messages. `tokens_before` is `Some` (carrying `AgentEvent::Compacted`'s own value) exactly
    /// when a compaction fired.
    fn persist(&mut self, session: &Session, tokens_before: Option<u32>) {
        if let Some(store) = &mut self.store {
            let r = match tokens_before {
                Some(tokens_before) => {
                    store.rewrite_compacted(&session.messages, CompactionMeta { tokens_before })
                }
                None => store.append_new(&session.messages),
            };
            if let Err(e) = r {
                eprintln!("serve: failed to persist session: {e}");
            }
        }
    }

    /// Incremental persist for a mid-run checkpoint (see [`ChannelCheckpoint`]) — always a plain
    /// append, never a compacted rewrite: compaction only ever runs at a turn's *start*, strictly
    /// before that turn's own checkpoint(s), so the session a checkpoint carries is never mid-compaction.
    fn persist_messages(&mut self, messages: &[agent_core::Message]) {
        if let Some(store) = &mut self.store {
            if let Err(e) = store.append_new(messages) {
                eprintln!("serve: failed to persist checkpoint: {e}");
            }
        }
    }

    /// The working directory, for new session metadata.
    fn cwd() -> String {
        std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// Start a fresh session. In repo mode this creates a new file (new id); in single-file mode it
    /// resets the existing file (keeping its id); in-memory it just mints new metadata.
    fn new_session(&mut self, model: &str) -> Session {
        let cwd = Self::cwd();
        if let Some(repo) = &self.repo {
            match repo.create(SessionMeta::new(&cwd, model)) {
                Ok(store) => {
                    self.meta = store.meta().clone();
                    self.store = Some(store);
                }
                Err(e) => eprintln!("serve: failed to create session: {e}"),
            }
        } else if let Some(store) = &mut self.store {
            if let Err(e) = store.rewrite(&[]) {
                eprintln!("serve: failed to reset session: {e}");
            }
        } else {
            self.meta = SessionMeta::new(&cwd, model);
        }
        Session::new()
    }

    /// Switch to another session by id (repo mode only).
    fn switch(&mut self, id: &str) -> std::io::Result<Session> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let (store, session) = repo.open_id(id)?;
        self.meta = store.meta().clone();
        self.store = Some(store);
        Ok(session)
    }

    /// Fork the current session at `upto` messages into a new session and switch to it (repo mode).
    fn fork(&mut self, upto: usize) -> std::io::Result<Session> {
        let id = self.meta.id.clone();
        let (store, session) = self
            .repo
            .as_ref()
            .ok_or_else(not_in_repo_mode)?
            .fork(&id, upto)?;
        self.meta = store.meta().clone();
        self.store = Some(store);
        Ok(session)
    }

    /// Preview what forking `session_id` at `upto` messages would produce — the exact prefix `fork`
    /// would copy — without creating a new session file or switching to it (repo mode only, like
    /// `switch`/`fork`). A client browsing `list_sessions` uses this to preview a fork point before
    /// committing to it.
    fn fork_messages(
        &self,
        session_id: &str,
        upto: usize,
    ) -> std::io::Result<Vec<agent_core::Message>> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let (_, session) = repo.open_id(session_id)?;
        let upto = upto.min(session.messages.len());
        Ok(session.messages[..upto].to_vec())
    }

    /// All sessions' metadata, newest first (empty unless in repo mode).
    fn list(&self) -> Vec<SessionMeta> {
        self.repo
            .as_ref()
            .and_then(|r| match r.list() {
                Ok(sessions) => Some(sessions),
                Err(e) => {
                    eprintln!("serve: failed to list sessions: {e}");
                    None
                }
            })
            .unwrap_or_default()
    }

    /// Every session across every project's own repo directory (pi's cross-project `listAll`), not
    /// just this process's own — the parent of this repo's directory is treated as the shared sessions
    /// root, with one subdirectory per project (the convention [`default_session_dir`] follows).
    /// `Err` when not in repo mode, or the repo directory has no parent to scan siblings of.
    fn list_all(&self) -> std::io::Result<Vec<SessionMeta>> {
        let repo = self.repo.as_ref().ok_or_else(not_in_repo_mode)?;
        let root = repo.dir().parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "session directory has no parent to list other projects from",
            )
        })?;
        SessionRepo::list_all(root)
    }

    /// Set the current session's title.
    fn set_title(&mut self, title: &str, messages: &[agent_core::Message]) -> std::io::Result<()> {
        if let Some(store) = &mut self.store {
            store.set_title(title, messages)?;
            self.meta = store.meta().clone();
        }
        Ok(())
    }

    /// Every branch in the current session's tree (empty unless persistence is configured — a
    /// pure in-memory session, with no `--session-file`/`--session-dir`, has no tree to report).
    fn list_branches(&self) -> Vec<BranchInfo> {
        self.store
            .as_ref()
            .map(SessionStore::list_branches)
            .unwrap_or_default()
    }

    /// Every node in the current session's tree — every message on every branch, unlike
    /// `list_branches`' leaves-only view (empty unless persistence is configured).
    fn tree(&self) -> Vec<crate::session_store::TreeNode> {
        self.store
            .as_ref()
            .map(SessionStore::tree)
            .unwrap_or_default()
    }

    /// Ids of the active path's messages, root-first — parallel to the live `Session.messages` when
    /// persistence is configured; empty in pure in-memory mode (nothing to id). What `get_messages`
    /// tags each message with, and what a client names as `switch_branch`'s `target_id`.
    fn active_ids(&self) -> &[String] {
        self.store
            .as_ref()
            .map(SessionStore::active_ids)
            .unwrap_or(&[])
    }

    /// Switch the active branch to `target_id`. When `summarize` and the branch being left behind has
    /// unsummarized activity (see `SessionStore::abandoned_by_switch`), generates a summary via
    /// `agent` and persists it *before* switching — mirroring pi's `navigateTree`. A summarization
    /// failure (a network error, or the model returning nothing) is logged and the switch proceeds
    /// anyway: losing the recap is far better than being unable to navigate away from a branch at all.
    async fn switch_branch(
        &mut self,
        agent: &Agent,
        target_id: &str,
        summarize: bool,
    ) -> std::io::Result<Session> {
        let store = self.store.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no session persistence configured (start serve with --session-file or --session-dir)",
            )
        })?;

        // A summary, once generated, is applied via `switch_active_with_summary` — it both persists
        // the recap *and* installs it as the new active tip in one step, so it actually reaches the
        // model on the next turn. Anything else (nothing abandoned, `summarize` off, the model call
        // failed or returned nothing) falls through to a plain `switch_active`.
        let mut summary_to_apply: Option<(String, String, BranchSummaryDetails)> = None;
        if summarize {
            let abandoned = store.abandoned_by_switch(target_id);
            if !abandoned.is_empty() {
                if let Some(from_id) = store.active_ids().last().cloned() {
                    let (ids, messages): (Vec<String>, Vec<agent_core::Message>) =
                        abandoned.into_iter().unzip();
                    match agent
                        .summarize_branch(&messages, &CancellationToken::new())
                        .await
                    {
                        Ok(summary) if !summary.trim().is_empty() => {
                            let (mut read_files, mut modified_files) =
                                agent_core::compaction::extract_file_ops(&messages);
                            // Fold forward any nested branch summary's own file-tracking within this
                            // same abandoned range — otherwise a detour-off-a-detour loses the earlier
                            // summary's file awareness the moment only its prose survives to be scanned
                            // (see `SessionStore::branch_summary_details_within`).
                            let prior = store.branch_summary_details_within(&ids);
                            for f in prior.read_files {
                                if !read_files.contains(&f) {
                                    read_files.push(f);
                                }
                            }
                            for f in prior.modified_files {
                                if !modified_files.contains(&f) {
                                    modified_files.push(f);
                                }
                            }
                            let details = BranchSummaryDetails {
                                read_files,
                                modified_files,
                                summarized_messages: messages.len() as u64,
                            };
                            summary_to_apply = Some((summary, from_id, details));
                        }
                        Ok(_) => {} // empty summary — nothing worth recording
                        Err(e) => {
                            eprintln!("serve: branch summarization failed, switching anyway: {e}")
                        }
                    }
                }
            }
        }

        let messages = match summary_to_apply {
            Some((summary, from_id, details)) => {
                match store.switch_active_with_summary(target_id, summary, from_id, details) {
                    Ok(messages) => messages,
                    Err(e) => {
                        // Recording the summary failed (a disk error mid-rewrite) — the switch itself
                        // must still succeed rather than leaving the client stuck on the old branch.
                        eprintln!("serve: failed to persist branch summary, switching anyway: {e}");
                        store.switch_active(target_id)?
                    }
                }
            }
            None => store.switch_active(target_id)?,
        };
        self.meta = store.meta().clone();
        let mut session = Session::new();
        session.messages = Arc::new(messages);
        Ok(session)
    }
}

fn not_in_repo_mode() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "not in repo mode (start serve with --session-dir)",
    )
}

/// Encode `cwd` into a filesystem-safe directory-name component: every path separator becomes `-`, so
/// `/home/jared/ai` becomes `-home-jared-ai` — the same convention this repo's other per-project state
/// already uses (`trust_store.rs`'s `~/.claude/trusted-projects.json`, `prompts.rs`'s
/// `~/.claude/prompts`), extended here to give each project its own session subdirectory.
fn encode_cwd(cwd: &str) -> String {
    cwd.chars()
        .map(|c| if c == '/' || c == '\\' { '-' } else { c })
        .collect()
}

/// The default session directory when neither `--session-file` nor `--session-dir` is given (and
/// persistence wasn't explicitly opted out): `~/.claude/sessions/<encoded-cwd>/`, one subdirectory per
/// project so unrelated projects' sessions never mix in the same listing.
fn default_session_dir(cwd: &str) -> std::path::PathBuf {
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_default();
    home.join(".claude/sessions").join(encode_cwd(cwd))
}

/// Run the control loop until stdin closes.
pub async fn serve(cfg: ServeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let (mut persistence, mut session) = Persistence::open(&cfg)?;

    // Assemble the system prompt from the base identity + this repo's project instructions + skills +
    // environment, so the agent behaves like it belongs in the working directory. Split into a static
    // half (this discovery-based block — expensive, rebuilt only on `set_model`/`set_thinking`/`reload`)
    // and a cheap dynamic footer (current date/cwd, recomputed before every `prompt` via `full_system`)
    // so a long-running `serve` process doesn't re-walk the filesystem every turn just for the date.
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut project_trusted =
        cfg.trust_project || crate::trust_store::TrustStore::open_default().is_trusted(&cwd);
    let mut static_system =
        crate::resources::build_static_system_prompt(&crate::resources::PromptOptions {
            base: &cfg.system,
            append: cfg.append_system.as_deref(),
            cwd: &cwd,
            include_context_files: cfg.context_files,
            include_skills: project_trusted,
            project_trusted,
        });

    // Slash-command prompt templates (`/name args`) and discoverable skills, for `get_commands` and
    // for expanding a `/name` prompt before it reaches the model. Gated on trust: an untrusted repo's
    // `.claude/skills`/`.claude/prompts` are attacker-controlled instructions, so they're neither
    // advertised nor invocable until the directory is trusted — otherwise `/skill:name` or `/name`
    // would inject arbitrary content into context regardless of trust. The `_with_diagnostics` variant
    // also reports name collisions (the same `/name` or skill name shadowed across roots), surfaced via
    // `get_commands`'s `collisions` field rather than silently resolved with no way for a client to
    // notice.
    let (mut prompt_templates, mut prompt_collisions, mut skills, mut skill_collisions) =
        if project_trusted {
            let (templates, tc) = crate::prompts::discover_with_diagnostics(&cwd);
            let (skills, sc) = crate::skills::discover_with_diagnostics(&cwd);
            (templates, tc, skills, sc)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

    // Keep the transport in an `Arc` we can clone: `set_model`/`set_thinking` rebuild the `Agent` at
    // runtime (a new model id picks a new dialect), and each rebuild reuses this one HTTP client.
    let client = GatewayClient::new(cfg.gateway.clone(), cfg.key.clone())?.with_retry(
        cfg.retry_max_retries
            .unwrap_or(agent_core::client::MAX_RETRIES),
        cfg.retry_base_delay_ms
            .unwrap_or(agent_core::client::BASE_BACKOFF),
    );
    let client = Arc::new(client);

    // The model, thinking budget, and auto-compaction flag are runtime-switchable; everything else
    // (transport, tools, system prompt, loop bounds, cache settings) is fixed for the process.
    // `build_agent` folds the mutable trio into a fresh `Agent` whenever any of them changes.
    let mut current_model = cfg.model.clone();
    let mut current_thinking = cfg.thinking;
    let mut current_auto_compaction = true;
    // Shared across every `build_agent` rebuild for this process's lifetime, so file-mutation
    // exclusivity (same-path `edit`/`write` calls) survives a `set_model`/`set_thinking` rebuild.
    let write_locks = Arc::new(agent_core::WriteLockRegistry::new());
    // A multi-step run (several tool round-trips) is otherwise only ever durable once it *fully*
    // completes — a crash, OOM-kill, or panic mid-run loses everything back to the turn's start,
    // including the user's own prompt. `ChannelCheckpoint` streams a cheap `Arc` snapshot of
    // `session.messages` through this channel at each durable mid-run point (see
    // `agent_core::CheckpointHook`'s doc comment for exactly which points those are); the `"prompt"`
    // arm's own busy-loop below drains it and persists incrementally, off the async task via
    // `spawn_blocking` like every other write in this file. The channel — not a `Mutex` around
    // `Persistence` itself — is what lets the checkpoint hook (called deep inside `agent.
    // run_events_steered`, which holds `&mut session` for the run's whole duration) reach persistence
    // without ever needing to borrow `session` a second time.
    let (checkpoint_tx, mut checkpoint_rx) =
        mpsc::unbounded_channel::<Arc<Vec<agent_core::Message>>>();
    let checkpoint: Arc<dyn agent_core::CheckpointHook> =
        Arc::new(ChannelCheckpoint(checkpoint_tx));
    let mut agent = build_agent(
        client.clone(),
        &full_system(&static_system, &cwd),
        &cfg,
        &current_model,
        current_thinking,
        current_auto_compaction,
        persistence.session_id(),
        &write_locks,
        &checkpoint,
    );
    // Persistent across every `prompt` call (not just the one currently in flight), so `steer`/
    // `follow_up` sent while idle queue for the *next* `prompt` instead of being rejected as an unknown
    // command. Cleared on every session switch (`new_session`/`switch_session`/`fork`/`switch_branch`)
    // so a message meant for the old session's next turn can't leak into the newly switched-to one.
    let steering = agent_core::Steering::new();
    // The `bash` tool, looked up once from the same (possibly filtered) registry `build_agent` builds —
    // `None` when `bash` was excluded (`--exclude-tools bash` / `--no-tools`), which the `bash` RPC
    // command below reports as a clean error rather than a side door around that restriction. A direct
    // `Arc<dyn Tool>` handle, not routed through `agent`/the model loop: the host `bash` RPC command runs
    // independent of any conversation turn.
    let bash_tool = build_tools(&cfg).get("bash");
    // Distinguishes successive host `bash` calls in their `tool_start`/`tool_progress`/`tool_end` event
    // ids — only ever one in flight at a time (see the `bash` command arm), but a stable, incrementing
    // id per call still lets a client correlate a run's own three events without ambiguity.
    let mut host_bash_seq: u32 = 0;

    // One writer task owns stdout; every frame (events + responses) is serialized through it in FIFO
    // order, so output never interleaves.
    //
    // The channel is intentionally unbounded. The event `sink` (see `Agent::run_events`) is a
    // synchronous `FnMut`, so the producer cannot `.await` to apply backpressure; a bounded channel
    // would force `try_send`, which silently drops frames and corrupts the event stream — unacceptable
    // for a protocol. In practice the backlog is bounded by one in-flight turn's events (capped by
    // `max_steps`), drained concurrently by the writer as fast as stdout accepts. If a client stops
    // reading, stdout's write eventually fails and the writer tears down (below), surfacing the stall
    // rather than masking it.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Value>();
    let writer = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        while let Some(frame) = out_rx.recv().await {
            // A frame we built ourselves failing to serialize is a bug, not bad input — skip it
            // rather than tearing down the whole stream.
            let line = match serde_json::to_string(&frame) {
                Ok(line) => line,
                Err(e) => {
                    eprintln!("serve: failed to serialize output frame: {e}");
                    continue;
                }
            };
            // stdout is the only sink. If it breaks (client hung up, broken pipe) there is nothing
            // left to do but stop; dropping `out_rx` here makes every sender observe the closure and
            // halt the control loop instead of writing into a dead pipe forever.
            if let Err(e) = write_frame(&mut out, &line).await {
                eprintln!("serve: stdout write failed, shutting down writer: {e}");
                break;
            }
        }
    });

    // Sends a frame through the writer; if the writer has shut down (stdout closed), stop the control
    // loop — there is no way to deliver any further response, so continuing would only swallow output.
    macro_rules! emit {
        ($frame:expr) => {
            if out_tx.send($frame).is_err() {
                break;
            }
        };
    }

    // Announce readiness so a client can sync before issuing commands. If this already fails the
    // writer never started; there is nothing to serve.
    if out_tx
        .send(
            json!({ "type": "ready", "session_id": persistence.session_id(), "model": current_model }),
        )
        .is_err()
    {
        let _ = writer.await;
        return Ok(());
    }

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut shutdown = ShutdownSignal::new()?;
    loop {
        let line = tokio::select! {
            biased;
            // Idle between commands: nothing is in flight, so a shutdown request needs no drain —
            // just stop reading and fall out to the writer join below.
            () = shutdown.wait() => break,
            maybe_line = lines.next_line() => match maybe_line? {
                Some(l) => l,
                None => break,
            },
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cmd: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                emit!(response(
                    None,
                    "?",
                    false,
                    None,
                    Some(&format!("invalid JSON: {e}")),
                ));
                continue;
            }
        };
        let id = cmd.get("id").and_then(Value::as_str).map(str::to_string);
        let ctype = cmd
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        match ctype.as_str() {
            "prompt" => {
                // Refresh the cheap, time-varying part of the system prompt (the date) every turn —
                // the expensive discovery-based static half only changes on `set_model`/`set_thinking`/
                // `reload`, so it's cached rather than recomputed here.
                agent.set_system(full_system(&static_system, &cwd));
                // A `prompt` with no (or non-string) `message` is a malformed command: running an
                // empty user turn would spend a model call on nothing and still report success.
                let Some(message) = cmd.get("message").and_then(Value::as_str) else {
                    emit!(response(
                        id,
                        "prompt",
                        false,
                        None,
                        Some("missing `message`"),
                    ));
                    continue;
                };
                // Expand an explicit `/skill:name` invocation first (its own prefix, so it can't
                // collide with a `/name` prompt template), then fall through to prompt-template
                // expansion — a no-op on whichever message reaches it unmatched.
                let message = crate::skills::expand_if_skill_invocation(message, &skills);
                let message = crate::prompts::expand_if_slash(&message, &prompt_templates);
                // Optional image attachments: `images: [{media_type, data}]` (base64). Builds a
                // multimodal user turn; absent or empty → a plain text turn.
                let images = parse_images(cmd.get("images"));
                if images.is_empty() {
                    session.user(message);
                } else {
                    session.push(agent_core::Message::user_with_images(message, images));
                }
                // Acknowledge immediately — the turn is queued and about to start — rather than
                // leaving a client with no signal until the (possibly much later) terminal response.
                emit!(ack(id.clone(), "prompt"));
                let tx = out_tx.clone();
                let cancel = CancellationToken::new();
                // The sink sets this to the compaction's `tokens_before` when the loop compacts
                // mid-run, so we know to non-destructively rewrite (not append) the persisted
                // transcript afterwards. 0 doubles as "no compaction fired this run" — see
                // `Persistence::persist`; in practice a real compaction's `tokens_before` is never
                // legitimately 0 (`should_compact`/`compact` only ever fire once real usage has been
                // recorded).
                let tokens_before = Arc::new(AtomicU32::new(0));
                let tokens_before_sink = tokens_before.clone();
                // Whether the run's *last* turn ended in a refusal — set from every `TurnEnd`, so by
                // the time the run returns it reflects only the final one (any refusal earlier in a
                // multi-turn run is superseded once the model goes on to call tools or answer plainly).
                let refused = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let refused_sink = refused.clone();
                // Live token/step counters a `get_state`/`get_session_stats` sent while this run is in
                // flight can answer from (see the busy-loop's own arms for those types below) — seeded
                // from the session's current totals, then kept current from the same events a streaming
                // client already observes, so a poller gets real progress instead of a busy rejection.
                let live_stats = Arc::new(LiveStats::from_session(&session));
                let live_stats_sink = live_stats.clone();

                // Drive the run while staying responsive to stdin: `abort` cancels it, `steer` queues a
                // mid-run injection and `follow_up` a stop-boundary one; any other command is
                // rejected as busy (the session is borrowed by the in-flight run). If stdin closes
                // mid-run, cancel and drain. The block scopes the run's `&mut session` borrow so we can
                // persist after.
                let mut stdin_open = true;
                let result = {
                    let run = agent.run_events_steered(
                        &mut session,
                        move |ev| {
                            if let AgentEvent::Compacted { tokens_before, .. } = ev {
                                tokens_before_sink.store(tokens_before, Ordering::Relaxed);
                            }
                            if let AgentEvent::TurnEnd { stop_reason, step } = &ev {
                                refused_sink
                                    .store(*stop_reason == StopReason::Refusal, Ordering::Relaxed);
                                live_stats_sink.set_steps(*step);
                            }
                            if let AgentEvent::Stream(StreamEvent::Usage(usage)) = &ev {
                                live_stats_sink.record_usage(usage);
                            }
                            // Best-effort: a sync sink can't break the control loop. If the writer is
                            // gone the send fails here and the terminal response send below detects it
                            // via `emit!` and stops the loop. An unserializable event is skipped rather
                            // than emitted as a malformed frame (see `event_frame`).
                            if let Some(frame) = event_frame(ev) {
                                let _ = tx.send(frame);
                            }
                        },
                        cancel.clone(),
                        steering.clone(),
                    );
                    tokio::pin!(run);
                    loop {
                        tokio::select! {
                            biased;
                            r = &mut run => break r,
                            // A shutdown request mid-run gets the same treatment as stdin closing:
                            // cancel and let the run unwind so the code below can persist before we
                            // fall out of the outer loop (`!stdin_open` breaks it, further down).
                            () = shutdown.wait() => {
                                stdin_open = false;
                                cancel.cancel();
                            }
                            // Drain and persist each mid-run checkpoint as it arrives (see
                            // `ChannelCheckpoint`). `persistence` isn't touched by `run` itself (only
                            // `session` is borrowed there), so reassigning it here is safe exactly like
                            // `stdin_open`/`cancel` being mutated from a sibling branch above. Any
                            // checkpoint left undrained when the run ends is harmless — the unconditional
                            // full persist right after this loop (below) is a strict superset of it.
                            Some(messages) = checkpoint_rx.recv() => {
                                persistence = persist_messages_blocking(persistence, messages).await;
                            }
                            maybe_line = lines.next_line(), if stdin_open => match maybe_line {
                                Ok(Some(l)) => {
                                    let l = l.trim();
                                    if l.is_empty() {
                                        continue;
                                    }
                                    let c: Value = match serde_json::from_str(l) {
                                        Ok(v) => v,
                                        Err(e) => {
                                            let _ = out_tx.send(response(None, "?", false, None, Some(&format!("invalid JSON: {e}"))));
                                            continue;
                                        }
                                    };
                                    let cid = c.get("id").and_then(Value::as_str).map(str::to_string);
                                    match c.get("type").and_then(Value::as_str).unwrap_or("") {
                                        "abort" => {
                                            cancel.cancel();
                                            let _ = out_tx.send(response(cid, "abort", true, None, None));
                                        }
                                        "stop_after_turn" => {
                                            // Graceful, not a cancel: the current turn's tool calls (if
                                            // any) still finish and their results are committed; the run
                                            // just doesn't start another model call afterward. See
                                            // `agent_core::Steering::request_stop`.
                                            steering.request_stop();
                                            let _ = out_tx.send(response(cid, "stop_after_turn", true, None, None));
                                        }
                                        cmd @ ("steer" | "follow_up") => {
                                            match c.get("message").and_then(Value::as_str) {
                                                Some(m) => {
                                                    // `steer` redirects mid-run (injected at the next
                                                    // tool turn); `follow_up` waits for the stop
                                                    // boundary. Two separate lanes.
                                                    if cmd == "steer" {
                                                        steering.push_steer(m);
                                                    } else {
                                                        steering.push(m);
                                                    }
                                                    let _ = out_tx.send(response(cid, cmd, true, None, None));
                                                }
                                                None => {
                                                    let _ = out_tx.send(response(cid, cmd, false, None, Some("missing `message`")));
                                                }
                                            }
                                        }
                                        // A `prompt` sent while busy is rejected by default (the
                                        // session is borrowed by the in-flight run) — *unless* it
                                        // carries `streaming_behavior: "steer"|"follow_up"`, in which
                                        // case its `message` is routed through the same `Steering`
                                        // queue as an explicit `steer`/`follow_up` command, rather than
                                        // forcing the client to re-encode it as a different command type.
                                        "prompt" => {
                                            match (
                                                c.get("streaming_behavior").and_then(Value::as_str),
                                                c.get("message").and_then(Value::as_str),
                                            ) {
                                                (Some("steer"), Some(m)) => {
                                                    steering.push_steer(m);
                                                    let _ = out_tx.send(response(cid, "prompt", true, Some(json!({ "queued_as": "steer" })), None));
                                                }
                                                (Some("follow_up"), Some(m)) => {
                                                    steering.push(m);
                                                    let _ = out_tx.send(response(cid, "prompt", true, Some(json!({ "queued_as": "follow_up" })), None));
                                                }
                                                _ => {
                                                    let _ = out_tx.send(response(cid, "prompt", false, None, Some("busy: a prompt is running; only `abort`/`steer`/`follow_up`, or a `prompt` with `streaming_behavior: \"steer\"|\"follow_up\"`, are accepted")));
                                                }
                                            }
                                        }
                                        // Read-only commands that don't need the `&mut Session` this
                                        // run holds — answered live instead of rejected as busy, so a
                                        // client polling for progress (tokens/steps so far) during a
                                        // long tool-heavy turn isn't left with nothing until it ends.
                                        // `get_state`/`get_session_stats` read `live_stats` (mirrored
                                        // from this run's own events, see above) instead of `session`
                                        // directly; `message_count` is `null` here specifically — it's
                                        // the one `get_state` field that genuinely needs `&session` to
                                        // report exactly, and a stale/guessed number would be worse
                                        // than an honest "not available mid-run".
                                        "get_state" => {
                                            let mut data = live_stats.snapshot();
                                            if let Value::Object(m) = &mut data {
                                                m.insert("session_id".into(), json!(persistence.session_id()));
                                                m.insert("model".into(), json!(current_model));
                                                m.insert("message_count".into(), Value::Null);
                                                m.insert("title".into(), json!(persistence.meta.title));
                                            }
                                            let _ = out_tx.send(response(cid, "get_state", true, Some(data), None));
                                        }
                                        "get_session_stats" => {
                                            let _ = out_tx.send(response(cid, "get_session_stats", true, Some(live_stats.snapshot()), None));
                                        }
                                        "get_commands" => {
                                            let mut commands: Vec<Value> = skills.iter().map(|s| {
                                                json!({ "name": s.name, "source": "skill", "description": s.description })
                                            }).collect();
                                            commands.extend(prompt_templates.iter().map(|t| {
                                                json!({ "name": t.name, "source": "prompt", "description": t.argument_hint })
                                            }));
                                            let collisions: Vec<&str> = skill_collisions.iter().chain(prompt_collisions.iter()).map(String::as_str).collect();
                                            let _ = out_tx.send(response(cid, "get_commands", true, Some(json!({ "commands": commands, "collisions": collisions })), None));
                                        }
                                        "list_branches" => {
                                            let _ = out_tx.send(response(cid, "list_branches", true, Some(json!({ "branches": persistence.list_branches() })), None));
                                        }
                                        "get_tree" => {
                                            let _ = out_tx.send(response(cid, "get_tree", true, Some(json!({ "nodes": persistence.tree() })), None));
                                        }
                                        "list_sessions" => {
                                            let sessions = serde_json::to_value(persistence.list()).unwrap_or(Value::Null);
                                            let _ = out_tx.send(response(cid, "list_sessions", true, Some(json!({ "sessions": sessions })), None));
                                        }
                                        "list_all_sessions" => {
                                            match persistence.list_all() {
                                                Ok(sessions) => {
                                                    let sessions = serde_json::to_value(sessions).unwrap_or(Value::Null);
                                                    let _ = out_tx.send(response(cid, "list_all_sessions", true, Some(json!({ "sessions": sessions })), None));
                                                }
                                                Err(e) => {
                                                    let _ = out_tx.send(response(cid, "list_all_sessions", false, None, Some(&e.to_string())));
                                                }
                                            }
                                        }
                                        "get_available_models" => {
                                            let _ = out_tx.send(response(cid, "get_available_models", true, Some(json!({ "models": available_models() })), None));
                                        }
                                        other => {
                                            let _ = out_tx.send(response(cid, other, false, None, Some("busy: a prompt is running; only `abort`/`steer`/`follow_up`, or a handful of read-only commands (get_state/get_session_stats/get_commands/list_branches/get_tree/list_sessions/list_all_sessions/get_available_models), are accepted")));
                                        }
                                    }
                                }
                                // stdin closed (or errored) mid-run: cancel and let the run unwind, then
                                // we'll fall out of the outer loop below.
                                Ok(None) | Err(_) => {
                                    stdin_open = false;
                                    cancel.cancel();
                                }
                            }
                        }
                    }
                };

                let compacted_tokens_before = match tokens_before.load(Ordering::Relaxed) {
                    0 => None,
                    n => Some(n),
                };
                persistence =
                    persist_blocking(persistence, session.clone(), compacted_tokens_before).await;
                let frame = match result {
                    Ok(()) => {
                        let mut data = session_stats(&session);
                        if let Value::Object(m) = &mut data {
                            m.insert("refused".into(), json!(refused.load(Ordering::Relaxed)));
                        }
                        response(id.clone(), "prompt", true, Some(data), None)
                    }
                    Err(e) => response(id.clone(), "prompt", false, None, Some(&e.to_string())),
                };
                emit!(frame);
                if !stdin_open {
                    break;
                }
            }
            // Reachable while idle (a mid-run `steer`/`follow_up` is handled inside the `prompt` arm's
            // own busy-loop above, which shares this same persistent `steering` handle) — queues for
            // whichever `prompt` runs next, rather than being rejected as an unknown command just
            // because nothing is in flight *yet*.
            cmd_type @ ("steer" | "follow_up") => {
                match cmd.get("message").and_then(Value::as_str) {
                    Some(m) => {
                        if cmd_type == "steer" {
                            steering.push_steer(m);
                        } else {
                            steering.push(m);
                        }
                        emit!(response(id, cmd_type, true, None, None));
                    }
                    None => emit!(response(
                        id,
                        cmd_type,
                        false,
                        None,
                        Some("missing `message`")
                    )),
                }
            }
            "abort" => {
                // No run is in flight (a mid-run abort is handled inside the `prompt` arm above), so
                // there is nothing to cancel — acknowledge idempotently.
                emit!(response(id, "abort", true, None, None));
            }
            "stop_after_turn" => {
                // No run is in flight, so there is no turn boundary to stop at. Unlike `steer`/
                // `follow_up`, deliberately *not* forwarded to `steering.request_stop()` here: that
                // would silently cut the *next* `prompt` off after its first turn, surprising a client
                // that sent this expecting to affect a run that no longer exists. Acknowledge as a
                // no-op instead, matching `abort`'s idle behavior.
                emit!(response(id, "stop_after_turn", true, None, None));
            }
            "get_state" => {
                let mut data = session_stats(&session);
                if let Value::Object(m) = &mut data {
                    m.insert("session_id".into(), json!(persistence.session_id()));
                    m.insert("model".into(), json!(current_model));
                    m.insert("message_count".into(), json!(session.messages.len()));
                    m.insert("title".into(), json!(persistence.meta.title));
                }
                emit!(response(id, "get_state", true, Some(data), None));
            }
            "get_messages" => {
                // Tag each message with its tree id when persistence tracks one (parallel to
                // `session.messages` — see `Persistence::active_ids`), so a client can pick a
                // `target_id` for `switch_branch` from anywhere in the visible transcript, not only
                // from a branch's leaf (which is all `list_branches` alone can offer). A length
                // mismatch (in-memory mode, or a transient inconsistency) leaves messages untagged
                // rather than mistagging them.
                let msg_ids = persistence.active_ids();
                let mut messages =
                    serde_json::to_value(session.messages.as_ref()).unwrap_or(Value::Null);
                if let Value::Array(arr) = &mut messages {
                    if arr.len() == msg_ids.len() {
                        for (m, mid) in arr.iter_mut().zip(msg_ids) {
                            if let Value::Object(obj) = m {
                                obj.insert("id".into(), json!(mid));
                            }
                        }
                    }
                }
                emit!(response(
                    id,
                    "get_messages",
                    true,
                    Some(json!({ "messages": messages })),
                    None,
                ));
            }
            "new_session" => {
                session = persistence.new_session(&current_model);
                steering.clear();
                emit!(response(
                    id,
                    "new_session",
                    true,
                    Some(json!({ "session_id": persistence.session_id() })),
                    None,
                ));
            }
            "list_sessions" => {
                let sessions = serde_json::to_value(persistence.list()).unwrap_or(Value::Null);
                emit!(response(
                    id,
                    "list_sessions",
                    true,
                    Some(json!({ "sessions": sessions })),
                    None,
                ));
            }
            "list_all_sessions" => match persistence.list_all() {
                Ok(sessions) => {
                    let sessions = serde_json::to_value(sessions).unwrap_or(Value::Null);
                    emit!(response(
                        id,
                        "list_all_sessions",
                        true,
                        Some(json!({ "sessions": sessions })),
                        None,
                    ));
                }
                Err(e) => emit!(response(
                    id,
                    "list_all_sessions",
                    false,
                    None,
                    Some(&e.to_string())
                )),
            },
            "switch_session" => match cmd.get("session_id").and_then(Value::as_str) {
                Some(target) => match persistence.switch(target) {
                    Ok(s) => {
                        session = s;
                        steering.clear();
                        emit!(response(
                            id,
                            "switch_session",
                            true,
                            Some(json!({ "session_id": persistence.session_id() })),
                            None,
                        ));
                    }
                    Err(e) => emit!(response(
                        id,
                        "switch_session",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                },
                None => emit!(response(
                    id,
                    "switch_session",
                    false,
                    None,
                    Some("missing `session_id`")
                )),
            },
            "fork" => {
                // `upto` messages to copy into the new session; absent = clone the whole session.
                let upto = cmd
                    .get("upto")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize)
                    .unwrap_or(usize::MAX);
                match persistence.fork(upto) {
                    Ok(s) => {
                        session = s;
                        steering.clear();
                        emit!(response(
                            id,
                            "fork",
                            true,
                            Some(json!({ "session_id": persistence.session_id() })),
                            None,
                        ));
                    }
                    Err(e) => emit!(response(id, "fork", false, None, Some(&e.to_string()))),
                }
            }
            "get_fork_messages" => {
                // A read-only preview of what `fork` would produce for `session_id` (default: the
                // current session) at `upto` messages — no new session file, no switch. Lets a client
                // browsing `list_sessions` show a fork point before committing to it.
                let target_id = cmd
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| persistence.session_id().to_string());
                let upto = cmd
                    .get("upto")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize)
                    .unwrap_or(usize::MAX);
                match persistence.fork_messages(&target_id, upto) {
                    Ok(messages) => emit!(response(
                        id,
                        "get_fork_messages",
                        true,
                        Some(json!({ "messages": messages })),
                        None,
                    )),
                    Err(e) => emit!(response(
                        id,
                        "get_fork_messages",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                }
            }
            "set_session_name" => match cmd.get("title").and_then(Value::as_str) {
                Some(title) => match persistence.set_title(title, &session.messages) {
                    Ok(()) => emit!(response(id, "set_session_name", true, None, None)),
                    Err(e) => emit!(response(
                        id,
                        "set_session_name",
                        false,
                        None,
                        Some(&e.to_string())
                    )),
                },
                None => emit!(response(
                    id,
                    "set_session_name",
                    false,
                    None,
                    Some("missing `title`")
                )),
            },
            "compact" => {
                // Manual compaction (no run in flight here). Streams a `compacted` event if it cuts.
                let tx = out_tx.clone();
                let mut compacted_tokens_before: Option<u32> = None;
                let result = agent
                    .compact(
                        &mut session,
                        agent_core::CompactionReason::Manual,
                        &CancellationToken::new(),
                        &mut |ev| {
                            if let AgentEvent::Compacted { tokens_before, .. } = ev {
                                compacted_tokens_before = Some(tokens_before);
                            }
                            if let Some(frame) = event_frame(ev) {
                                let _ = tx.send(frame);
                            }
                        },
                    )
                    .await;
                match result {
                    Ok(did) => {
                        persistence =
                            persist_blocking(persistence, session.clone(), compacted_tokens_before)
                                .await;
                        emit!(response(
                            id,
                            "compact",
                            true,
                            Some(json!({ "compacted": did })),
                            None
                        ));
                    }
                    Err(e) => emit!(response(id, "compact", false, None, Some(&e.to_string()))),
                }
            }
            "get_last_assistant_text" => {
                let text = last_assistant_text(&session);
                emit!(response(
                    id,
                    "get_last_assistant_text",
                    true,
                    Some(json!({ "text": text })),
                    None,
                ));
            }
            "get_session_stats" => {
                emit!(response(
                    id,
                    "get_session_stats",
                    true,
                    Some(session_stats(&session)),
                    None
                ));
            }
            "get_commands" => {
                // Skills (read-on-demand) and prompt templates (`/name`), for client autocomplete.
                let mut commands: Vec<Value> = skills
                    .iter()
                    .map(|s| {
                        json!({ "name": s.name, "source": "skill", "description": s.description })
                    })
                    .collect();
                commands.extend(prompt_templates.iter().map(|t| {
                    json!({ "name": t.name, "source": "prompt", "description": t.argument_hint })
                }));
                // Every shadowed name (a skill or template defined at more than one path) is otherwise
                // silently resolved with no way for a client to notice — surfaced here instead.
                let collisions: Vec<&str> = skill_collisions
                    .iter()
                    .chain(prompt_collisions.iter())
                    .map(String::as_str)
                    .collect();
                emit!(response(
                    id,
                    "get_commands",
                    true,
                    Some(json!({ "commands": commands, "collisions": collisions })),
                    None,
                ));
            }
            "reload" => {
                // Re-run the full (expensive) discovery pi's `/reload` triggers: trust may have
                // changed (`agent trust`/`--trust-project` since startup), and project instructions,
                // `SYSTEM.md`, and skills/prompt templates may have changed on disk. The per-turn
                // `full_system` refresh alone only ever picks up the cheap date/cwd footer.
                project_trusted = cfg.trust_project
                    || crate::trust_store::TrustStore::open_default().is_trusted(&cwd);
                static_system = crate::resources::build_static_system_prompt(
                    &crate::resources::PromptOptions {
                        base: &cfg.system,
                        append: cfg.append_system.as_deref(),
                        cwd: &cwd,
                        include_context_files: cfg.context_files,
                        include_skills: project_trusted,
                        project_trusted,
                    },
                );
                (
                    prompt_templates,
                    prompt_collisions,
                    skills,
                    skill_collisions,
                ) = if project_trusted {
                    let (templates, tc) = crate::prompts::discover_with_diagnostics(&cwd);
                    let (discovered_skills, sc) = crate::skills::discover_with_diagnostics(&cwd);
                    (templates, tc, discovered_skills, sc)
                } else {
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                };
                agent.set_system(full_system(&static_system, &cwd));
                emit!(response(id, "reload", true, None, None));
            }
            "set_model" => match cmd.get("model").and_then(Value::as_str) {
                Some(model) => {
                    // Rebuild the agent so subsequent prompts use the new model (and, via the id, its
                    // dialect + capabilities). A signed thinking block is only valid for replay to the
                    // model that produced it, and a combined OpenAI-Responses tool-call id only means
                    // anything back to that same model — scrub both from any message not already
                    // stamped with the model we're switching to.
                    session.scrub_cross_model_state(model);
                    current_model = model.to_string();
                    agent = build_agent(
                        client.clone(),
                        &full_system(&static_system, &cwd),
                        &cfg,
                        &current_model,
                        current_thinking,
                        current_auto_compaction,
                        persistence.session_id(),
                        &write_locks,
                        &checkpoint,
                    );
                    emit!(response(
                        id,
                        "set_model",
                        true,
                        Some(json!({ "model": current_model })),
                        None,
                    ));
                }
                None => emit!(response(
                    id,
                    "set_model",
                    false,
                    None,
                    Some("missing `model`")
                )),
            },
            "set_thinking" => {
                // `budget` is a positive integer to enable extended thinking, or `null` to disable it.
                // A present-but-null value is the explicit "turn it off" signal; a missing key is an
                // error (so a typo can't silently no-op).
                match cmd.get("budget") {
                    Some(Value::Null) => {
                        current_thinking = None;
                        agent = build_agent(
                            client.clone(),
                            &full_system(&static_system, &cwd),
                            &cfg,
                            &current_model,
                            current_thinking,
                            current_auto_compaction,
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
                        );
                        emit!(response(
                            id,
                            "set_thinking",
                            true,
                            Some(json!({ "thinking": Value::Null })),
                            None,
                        ));
                    }
                    Some(v) if v.as_u64().is_some() => {
                        current_thinking = v.as_u64().map(|n| n as u32);
                        agent = build_agent(
                            client.clone(),
                            &full_system(&static_system, &cwd),
                            &cfg,
                            &current_model,
                            current_thinking,
                            current_auto_compaction,
                            persistence.session_id(),
                            &write_locks,
                            &checkpoint,
                        );
                        emit!(response(
                            id,
                            "set_thinking",
                            true,
                            Some(json!({ "thinking": current_thinking })),
                            None,
                        ));
                    }
                    _ => emit!(response(
                        id,
                        "set_thinking",
                        false,
                        None,
                        Some("`budget` must be a non-negative integer or null"),
                    )),
                }
            }
            "cycle_model" => {
                // Advance through `available_models()`, wrapping — a quick way for a client to step
                // through the known model list without needing its own copy of it. An id outside the
                // list (a custom `set_model` the client issued directly) wraps to the first entry,
                // same as "not found".
                let models = available_models();
                let next_idx = models
                    .iter()
                    .position(|m| *m == current_model)
                    .map_or(0, |i| (i + 1) % models.len());
                let next_model = models[next_idx];
                session.scrub_cross_model_state(next_model);
                current_model = next_model.to_string();
                agent = build_agent(
                    client.clone(),
                    &full_system(&static_system, &cwd),
                    &cfg,
                    &current_model,
                    current_thinking,
                    current_auto_compaction,
                    persistence.session_id(),
                    &write_locks,
                    &checkpoint,
                );
                emit!(response(
                    id,
                    "cycle_model",
                    true,
                    Some(json!({ "model": current_model })),
                    None,
                ));
            }
            "cycle_thinking_level" => {
                // Advance through a fixed Off/Low/Medium/High token-budget ladder, wrapping — the
                // current budget maps to its nearest rung first, so cycling from an arbitrary
                // `set_thinking` value still advances sensibly rather than jumping to a rung far from
                // where it started.
                current_thinking = next_thinking_level(
                    current_thinking,
                    agent_core::capabilities(&current_model).max_output,
                );
                agent = build_agent(
                    client.clone(),
                    &full_system(&static_system, &cwd),
                    &cfg,
                    &current_model,
                    current_thinking,
                    current_auto_compaction,
                    persistence.session_id(),
                    &write_locks,
                    &checkpoint,
                );
                emit!(response(
                    id,
                    "cycle_thinking_level",
                    true,
                    Some(json!({ "thinking": current_thinking })),
                    None,
                ));
            }
            "set_auto_compaction" => match cmd.get("enabled").and_then(Value::as_bool) {
                Some(enabled) => {
                    current_auto_compaction = enabled;
                    agent = build_agent(
                        client.clone(),
                        &full_system(&static_system, &cwd),
                        &cfg,
                        &current_model,
                        current_thinking,
                        current_auto_compaction,
                        persistence.session_id(),
                        &write_locks,
                        &checkpoint,
                    );
                    emit!(response(
                        id,
                        "set_auto_compaction",
                        true,
                        Some(json!({ "auto_compaction": current_auto_compaction })),
                        None,
                    ));
                }
                None => emit!(response(
                    id,
                    "set_auto_compaction",
                    false,
                    None,
                    Some("missing boolean `enabled`")
                )),
            },
            "get_available_models" => {
                emit!(response(
                    id,
                    "get_available_models",
                    true,
                    Some(json!({ "models": available_models() })),
                    None,
                ));
            }
            "list_branches" => {
                emit!(response(
                    id,
                    "list_branches",
                    true,
                    Some(json!({ "branches": persistence.list_branches() })),
                    None,
                ));
            }
            "get_tree" => {
                emit!(response(
                    id,
                    "get_tree",
                    true,
                    Some(json!({ "nodes": persistence.tree() })),
                    None,
                ));
            }
            "switch_branch" => match cmd.get("target_id").and_then(Value::as_str) {
                Some(target_id) => {
                    // Defaults to summarizing the abandoned branch's activity (mirroring pi's
                    // `navigateTree`); a client can pass `summarize:false` for a quick, cheap switch.
                    let summarize = cmd
                        .get("summarize")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
                    let target_id = target_id.to_string();
                    match persistence
                        .switch_branch(&agent, &target_id, summarize)
                        .await
                    {
                        Ok(s) => {
                            session = s;
                            steering.clear();
                            emit!(response(
                                id,
                                "switch_branch",
                                true,
                                Some(json!({ "target_id": target_id })),
                                None,
                            ));
                        }
                        Err(e) => emit!(response(
                            id,
                            "switch_branch",
                            false,
                            None,
                            Some(&e.to_string())
                        )),
                    }
                }
                None => emit!(response(
                    id,
                    "switch_branch",
                    false,
                    None,
                    Some("missing `target_id`")
                )),
            },
            "bash" => {
                let Some(command) = cmd.get("command").and_then(Value::as_str) else {
                    emit!(response(id, "bash", false, None, Some("missing `command`")));
                    continue;
                };
                let Some(tool) = bash_tool.clone() else {
                    emit!(response(
                        id,
                        "bash",
                        false,
                        None,
                        Some(
                            "the `bash` tool is not registered for this process \
                             (excluded via --exclude-tools/--no-tools)"
                        ),
                    ));
                    continue;
                };
                let mut input = json!({ "command": command });
                if let Some(cwd) = cmd.get("cwd").and_then(Value::as_str) {
                    input["cwd"] = json!(cwd);
                }
                if let Some(timeout_ms) = cmd.get("timeout_ms").and_then(Value::as_u64) {
                    input["timeout_ms"] = json!(timeout_ms);
                }
                host_bash_seq += 1;
                let call_id = format!("host-bash-{host_bash_seq}");
                if let Some(frame) = event_frame(AgentEvent::ToolStart {
                    id: call_id.clone(),
                    name: "bash".to_string(),
                    input: input.clone(),
                }) {
                    emit!(frame);
                }

                // Independent of the model loop and the session — runs directly against the
                // registered `bash` tool, reusing its own streaming/truncation/cancellation
                // machinery. Races the command against continued stdin reads so `abort_bash`/`abort`
                // (or shutdown) can cancel it; any other command is rejected as busy, the same shape
                // as the `prompt` arm's own busy-loop, simplified with no session/steering involved.
                let cancel = CancellationToken::new();
                let (prog_tx, mut prog_rx) = futures::channel::mpsc::unbounded::<ToolUpdate>();
                let progress = agent_core::ToolProgress::new(
                    prog_tx,
                    call_id.clone(),
                    "bash".to_string(),
                    cancel.clone(),
                );
                let mut stdin_open = true;
                let outcome = {
                    let run = tool.run_streaming(input, &progress);
                    tokio::pin!(run);
                    loop {
                        tokio::select! {
                            biased;
                            r = &mut run => break r,
                            // `cancel.cancel()` alone (set by `abort_bash`/`abort` below, or by
                            // shutdown) doesn't interrupt an in-flight `run` — `bash` has no
                            // cooperative cancellation check of its own; it relies on the loop
                            // *dropping* its future, same as the model-loop's own tool dispatch does.
                            // Breaking here (rather than continuing to await `run`) drops the pinned
                            // future when this block ends, killing the subprocess via its
                            // `kill_on_drop`/process-group guard.
                            () = cancel.cancelled() => {
                                break Err(agent_core::ToolError::Execution("cancelled".to_string()));
                            }
                            () = shutdown.wait() => {
                                stdin_open = false;
                                cancel.cancel();
                            }
                            update = prog_rx.next() => {
                                if let Some(ToolUpdate::Progress { id, name, snapshot, details }) = update {
                                    if let Some(frame) = event_frame(AgentEvent::ToolProgress { id, name, snapshot, details }) {
                                        let _ = out_tx.send(frame);
                                    }
                                }
                            }
                            maybe_line = lines.next_line(), if stdin_open => match maybe_line {
                                Ok(Some(l)) => {
                                    let l = l.trim();
                                    if l.is_empty() {
                                        continue;
                                    }
                                    let c: Value = match serde_json::from_str(l) {
                                        Ok(v) => v,
                                        Err(e) => {
                                            let _ = out_tx.send(response(None, "?", false, None, Some(&format!("invalid JSON: {e}"))));
                                            continue;
                                        }
                                    };
                                    let cid = c.get("id").and_then(Value::as_str).map(str::to_string);
                                    match c.get("type").and_then(Value::as_str).unwrap_or("") {
                                        "abort_bash" | "abort" => {
                                            cancel.cancel();
                                            let _ = out_tx.send(response(cid, "abort_bash", true, None, None));
                                        }
                                        other => {
                                            let _ = out_tx.send(response(cid, other, false, None, Some("busy: a host bash command is running; only `abort_bash`/`abort` are accepted")));
                                        }
                                    }
                                }
                                Ok(None) | Err(_) => {
                                    stdin_open = false;
                                    cancel.cancel();
                                }
                            }
                        }
                    }
                };
                // Flush anything buffered between the final poll above and the tool's own return.
                while let Ok(update) = prog_rx.try_recv() {
                    if let ToolUpdate::Progress {
                        id,
                        name,
                        snapshot,
                        details,
                    } = update
                    {
                        if let Some(frame) = event_frame(AgentEvent::ToolProgress {
                            id,
                            name,
                            snapshot,
                            details,
                        }) {
                            let _ = out_tx.send(frame);
                        }
                    }
                }

                let (result_text, is_error) = match outcome {
                    Ok(output) => (output.text, false),
                    Err(e) => (e.to_string(), true),
                };
                if let Some(frame) = event_frame(AgentEvent::ToolEnd {
                    id: call_id,
                    name: "bash".to_string(),
                    result: result_text.clone(),
                    is_error,
                }) {
                    emit!(frame);
                }
                emit!(response(
                    id,
                    "bash",
                    true,
                    Some(json!({ "result": result_text, "is_error": is_error })),
                    None,
                ));
                if !stdin_open {
                    break;
                }
            }
            "abort_bash" => {
                // No host bash is in flight (a running one is handled inside the `bash` arm above), so
                // there is nothing to cancel — acknowledge idempotently, matching `abort`'s idle shape.
                emit!(response(id, "abort_bash", true, None, None));
            }
            other => {
                emit!(response(id, other, false, None, Some("unknown command")));
            }
        }
    }

    drop(out_tx);
    let _ = writer.await;
    Ok(())
}

/// Join the cached static system prompt with a freshly-computed dynamic footer (current date/cwd) —
/// the full prompt text an `Agent` should carry. Cheap: `static_system` is already-computed text and
/// `dynamic_footer` does no filesystem discovery, so this is safe to call every turn (see the `prompt`
/// arm's per-turn refresh) as well as at every `build_agent` rebuild.
fn full_system(static_system: &str, cwd: &std::path::Path) -> String {
    format!("{static_system}{}", crate::resources::dynamic_footer(cwd))
}

/// Build the [`Agent`] for the current model + thinking budget + auto-compaction flag. Called once at
/// startup and again on every `set_model`/`set_thinking`/`cycle_model`/`cycle_thinking_level`/
/// `set_auto_compaction`, so a client can re-tune the run without restarting `serve`. The transport,
/// tools, system prompt, loop bounds, and cache settings are the same each time; only the model id,
/// thinking budget, and auto-compaction flag vary. Compaction's `context_window` defaults to `model`'s
/// own capabilities; an explicit `--context-window` overrides that and stays pinned across a model
/// switch (the operator's compaction budget, not the dialect's) — left unset, each switch picks up the
/// *new* model's real window instead of a stale operator number. `reserve_tokens`/`keep_recent_tokens`
/// default to `CompactionConfig::default()`, overridable independently of `context_window`.
// 8 arguments, all independent inputs every call site already has on hand from `cfg`/local
// runtime-switchable state — bundling them into a struct would just be a second place those same
// fields live, not a reduction in what the function needs to know (see `client.rs::send_with_retry`
// for the same tradeoff). Private, single-purpose helper, not a public API shape.
#[allow(clippy::too_many_arguments)]
fn build_agent(
    transport: Arc<GatewayClient>,
    system: &str,
    cfg: &ServeConfig,
    model: &str,
    thinking: Option<u32>,
    auto_compaction: bool,
    cache_key: &str,
    write_locks: &Arc<agent_core::WriteLockRegistry>,
    checkpoint: &Arc<dyn agent_core::CheckpointHook>,
) -> Agent {
    let mut compaction = agent_core::CompactionConfig {
        context_window: cfg
            .context_window
            .unwrap_or_else(|| agent_core::capabilities(model).context_window),
        enabled: auto_compaction,
        ..agent_core::CompactionConfig::default()
    };
    if let Some(reserve) = cfg.compaction_reserve_tokens {
        compaction.reserve_tokens = reserve;
    }
    if let Some(keep_recent) = cfg.compaction_keep_recent_tokens {
        compaction.keep_recent_tokens = keep_recent;
    }

    let mut agent = Agent::new(transport, model.to_string())
        .with_tools(build_tools(cfg))
        .with_system(system.to_string())
        .with_max_steps(cfg.max_steps)
        .with_compaction(compaction)
        // Pin this session to a warm prompt-cache node via its stable id.
        .with_cache_key(cache_key.to_string())
        .with_cache_long(cfg.cache_long)
        // Shared across every `build_agent` call for this process (including `set_model`/
        // `set_thinking` rebuilds), so file-mutation exclusivity survives a model switch and extends
        // to any other session sharing this same registry.
        .with_write_locks(write_locks.clone())
        // Streams a snapshot of `session.messages` through `checkpoint`'s channel at each durable
        // mid-run point (see `ChannelCheckpoint`), so a long multi-step turn is persisted incrementally
        // instead of only once it fully completes.
        .with_checkpoint_hook(checkpoint.clone());
    if let Some(budget) = thinking {
        agent = agent.with_thinking(budget);
    }
    if let Some(effort) = cfg.reasoning_effort {
        agent = agent.with_reasoning_effort(effort);
    }
    agent
}

/// The tool registry after `--tools`/`--exclude-tools`/`--no-tools` filtering — shared by every
/// `build_agent` rebuild and by the host-level `bash` RPC command (see [`serve`]), so excluding `bash`
/// from the model's own tool set also disables the host command rather than leaving a side door open
/// around an operator's explicit restriction.
fn build_tools(cfg: &ServeConfig) -> agent_core::ToolRegistry {
    let mut registry = tools::default_registry_with(cfg.bash_timeout_ms);
    tools::apply_filter(
        &mut registry,
        cfg.tools.as_deref(),
        cfg.exclude_tools.as_deref(),
        cfg.no_tools,
    );
    registry
}

/// The fixed Off/Low/Medium/High thinking-budget ladder `cycle_thinking_level` steps through.
const THINKING_LEVELS: [Option<u32>; 4] = [None, Some(2_048), Some(8_192), Some(24_000)];

/// The next rung on [`THINKING_LEVELS`] after `current`, wrapping — clamped below `max_output` (a
/// thinking budget must leave room for the turn's actual output). `current` first maps to its *nearest*
/// rung (not necessarily the one it came from, if `set_thinking` set an arbitrary value), so cycling
/// always advances sensibly rather than jumping to a rung far from where the budget actually was.
fn next_thinking_level(current: Option<u32>, max_output: u32) -> Option<u32> {
    let nearest = match current {
        None => 0,
        Some(budget) => THINKING_LEVELS
            .iter()
            .enumerate()
            .min_by_key(|(_, lvl)| lvl.map_or(u32::MAX, |b| b.abs_diff(budget)))
            .map(|(i, _)| i)
            .unwrap_or(0),
    };
    let next = THINKING_LEVELS[(nearest + 1) % THINKING_LEVELS.len()];
    next.map(|budget| budget.min(max_output.saturating_sub(1).max(1)))
}

/// A small, non-exhaustive list of model ids the [`capabilities`](agent_core::capabilities) table
/// recognizes, for a client's model picker. The gateway forwards any id verbatim, so this is a
/// convenience hint — not an allowlist; `set_model` accepts ids outside this list.
fn available_models() -> &'static [&'static str] {
    &[
        "claude-opus-4-8",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "gpt-5",
        "gpt-5-mini",
        "gpt-4o",
        "gpt-4.1",
        "o3",
        "o4-mini",
    ]
}

/// The concatenated text of the most recent assistant message, for scripting clients that just want
/// the answer. Empty if there's no assistant turn yet.
fn last_assistant_text(session: &Session) -> String {
    session
        .messages
        .iter()
        .rev()
        .find(|m| m.role == agent_core::Role::Assistant)
        .map(|m| {
            m.content
                .iter()
                .filter_map(|b| match b {
                    agent_core::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>()
        })
        .unwrap_or_default()
}

/// Parse a `prompt`'s optional `images` array into base64 image sources. Each entry is
/// `{media_type, data}`; malformed entries are skipped.
fn parse_images(images: Option<&Value>) -> Vec<agent_core::ImageSource> {
    images
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|img| {
                    let media_type = img.get("media_type").and_then(Value::as_str)?;
                    let data = img.get("data").and_then(Value::as_str)?;
                    Some(agent_core::ImageSource::base64(media_type, data))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Token + step accounting for a session, including the prompt-cache reads/writes that make the
/// Anthropic cache breakpoints observable. Shared by the `prompt` response and `get_state`.
fn session_stats(session: &Session) -> Value {
    json!({
        "steps": session.steps,
        "input_tokens": session.input_tokens,
        "output_tokens": session.output_tokens,
        "cache_read_tokens": session.cache_read_tokens,
        "cache_write_tokens": session.cache_write_tokens,
        "cache_write_1h_tokens": session.cache_write_1h_tokens,
        "reasoning_tokens": session.reasoning_tokens,
        "last_input_tokens": session.last_input_tokens,
    })
}

/// A live mirror of [`Session`]'s token/step counters, updated from the event sink as a `prompt` runs —
/// so `get_state`/`get_session_stats` can answer with real progress from *inside* the busy-loop
/// (see the `prompt` arm), without the `&mut Session` the run itself holds exclusively for the turn's
/// duration. Seeded from the session's own values right before the run starts, then kept current by
/// mirroring exactly the events a client watching the stream already sees: `record_usage` matches
/// `Session::record_usage`'s accumulation, and `set_steps` takes `AgentEvent::TurnEnd`'s `step` field,
/// which is already `session.steps` post-increment at the moment that event fires. Lock-free (plain
/// atomics): the sink runs synchronously inside the run's own task, and a command handler reads it from
/// the control loop's task — never contended enough to need more than `Relaxed` ordering, since these
/// are independent counters with no cross-field invariant a reader depends on.
#[derive(Default)]
struct LiveStats {
    steps: AtomicU32,
    input_tokens: std::sync::atomic::AtomicU64,
    output_tokens: std::sync::atomic::AtomicU64,
    cache_read_tokens: std::sync::atomic::AtomicU64,
    cache_write_tokens: std::sync::atomic::AtomicU64,
    cache_write_1h_tokens: std::sync::atomic::AtomicU64,
    reasoning_tokens: std::sync::atomic::AtomicU64,
    last_input_tokens: AtomicU32,
}

impl LiveStats {
    /// Seed from a session's current cumulative totals, so a `get_state`/`get_session_stats` answered
    /// one event into a brand-new turn still reflects everything before it, not a reset-to-zero count.
    fn from_session(session: &Session) -> Self {
        Self {
            steps: AtomicU32::new(session.steps),
            input_tokens: session.input_tokens.into(),
            output_tokens: session.output_tokens.into(),
            cache_read_tokens: session.cache_read_tokens.into(),
            cache_write_tokens: session.cache_write_tokens.into(),
            cache_write_1h_tokens: session.cache_write_1h_tokens.into(),
            reasoning_tokens: session.reasoning_tokens.into(),
            last_input_tokens: AtomicU32::new(session.last_input_tokens),
        }
    }

    /// Fold one turn's usage into the running totals — the same arithmetic as
    /// [`Session::record_usage`], applied to atomics instead of plain fields.
    fn record_usage(&self, usage: &agent_core::TokenUsage) {
        self.input_tokens
            .fetch_add(usage.input_tokens.into(), Ordering::Relaxed);
        self.output_tokens
            .fetch_add(usage.output_tokens.into(), Ordering::Relaxed);
        self.cache_read_tokens
            .fetch_add(usage.cache_read_tokens.into(), Ordering::Relaxed);
        self.cache_write_tokens
            .fetch_add(usage.cache_write_tokens.into(), Ordering::Relaxed);
        self.cache_write_1h_tokens
            .fetch_add(usage.cache_write_1h_tokens.into(), Ordering::Relaxed);
        self.reasoning_tokens
            .fetch_add(usage.reasoning_tokens.into(), Ordering::Relaxed);
        let last = usage
            .input_tokens
            .saturating_add(usage.cache_read_tokens)
            .saturating_add(usage.cache_write_tokens);
        self.last_input_tokens.store(last, Ordering::Relaxed);
    }

    fn set_steps(&self, step: u32) {
        self.steps.store(step, Ordering::Relaxed);
    }

    /// A `session_stats`-shaped snapshot of the current values.
    fn snapshot(&self) -> Value {
        json!({
            "steps": self.steps.load(Ordering::Relaxed),
            "input_tokens": self.input_tokens.load(Ordering::Relaxed),
            "output_tokens": self.output_tokens.load(Ordering::Relaxed),
            "cache_read_tokens": self.cache_read_tokens.load(Ordering::Relaxed),
            "cache_write_tokens": self.cache_write_tokens.load(Ordering::Relaxed),
            "cache_write_1h_tokens": self.cache_write_1h_tokens.load(Ordering::Relaxed),
            "reasoning_tokens": self.reasoning_tokens.load(Ordering::Relaxed),
            "last_input_tokens": self.last_input_tokens.load(Ordering::Relaxed),
        })
    }
}

/// Build a `response` frame.
fn response(
    id: Option<String>,
    command: &str,
    success: bool,
    data: Option<Value>,
    error: Option<&str>,
) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("response"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    m.insert("success".into(), json!(success));
    if let Some(d) = data {
        m.insert("data".into(), d);
    }
    if let Some(e) = error {
        m.insert("error".into(), json!(e));
    }
    Value::Object(m)
}

/// Wrap an `AgentEvent` in an `event` frame, or `None` if it can't be serialized. Returning `None`
/// (and skipping the frame) rather than emitting `{type:"event"}` with no `event` field keeps a
/// serialization bug from putting a malformed frame on the wire that a client would silently mis-read.
/// A lightweight acknowledgement frame, emitted the moment a `prompt` is queued — before the model
/// turn(s) actually run — so a client can distinguish "received and starting" from the eventual
/// terminal `response` (which may be seconds away on a long tool-heavy run).
fn ack(id: Option<String>, command: &str) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("ack"));
    if let Some(id) = id {
        m.insert("id".into(), json!(id));
    }
    m.insert("command".into(), json!(command));
    Value::Object(m)
}

fn event_frame(ev: AgentEvent) -> Option<Value> {
    let event = serde_json::to_value(&ev)
        .inspect_err(|e| eprintln!("serve: failed to serialize agent event: {e}"))
        .ok()?;
    let mut m = Map::new();
    m.insert("type".into(), json!("event"));
    m.insert("event".into(), event);
    Some(Value::Object(m))
}

/// Write one newline-delimited frame to stdout and flush it.
async fn write_frame(out: &mut tokio::io::Stdout, line: &str) -> std::io::Result<()> {
    out.write_all(line.as_bytes()).await?;
    out.write_all(b"\n").await?;
    out.flush().await
}
