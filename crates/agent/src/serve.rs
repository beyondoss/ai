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
//!   - `{type:"prompt", message}`        run a turn; streams `event` frames, then a `response`
//!   - `{type:"abort"}`                  cancel the in-flight `prompt` (if any), else a no-op ack
//!   - `{type:"get_state"}`              → `data: {session_id, model, steps, message_count, title, …}`
//!   - `{type:"get_messages"}`           → `data: {messages: [...]}`
//!   - `{type:"new_session"}`            start a fresh session → `data: {session_id}`
//!   - `{type:"list_sessions"}`          (repo mode) → `data: {sessions: [SessionMeta…]}`
//!   - `{type:"switch_session", session_id}` (repo mode) load another session
//!   - `{type:"fork", upto?}`            (repo mode) copy the prefix into a new session, switch to it
//!   - `{type:"set_session_name", title}` set the session's title
//!   - `{type:"compact"}`                summarize the prefix now → `data: {compacted: bool}`
//!   - `{type:"get_last_assistant_text"}` → `data: {text}` (the latest assistant reply)
//!   - `{type:"get_session_stats"}`      → token/step accounting
//!   - `{type:"get_commands"}`           → discoverable skills + prompt templates
//!
//! While a `prompt` runs, the loop keeps reading stdin so an `abort` can cancel it, or `steer`/
//! `follow_up` (with a `message`) can queue input to inject when the model next stops; any other
//! command issued during a run is rejected as busy (the session is borrowed by the run).
//!
//! Frames (stdout) are either `{type:"response", id?, command, success, data?, error?}` or
//! `{type:"event", event: <AgentEvent>}`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use agent_core::{Agent, AgentEvent, CancellationToken, GatewayClient, Session};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::session_store::{SessionMeta, SessionRepo, SessionStore};
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
    /// The model's context window in tokens — the budget the loop compacts against to stay below.
    pub context_window: u32,
    /// Use the 1-hour prompt-cache TTL (vs the default 5 minutes) — useful when turns are spaced out.
    pub cache_long: bool,
    /// Extended-thinking token budget, when enabled (`None` leaves thinking off). Must be below the
    /// per-turn `max_tokens`.
    pub thinking: Option<u32>,
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
            let repo = SessionRepo::open(dir)?;
            let (store, session) = match repo.list()?.first() {
                Some(meta) => repo.open_id(&meta.id)?,
                None => {
                    let store = repo.create(SessionMeta::new(&cwd, &cfg.model))?;
                    (store, Session::new())
                }
            };
            let meta = store.meta().clone();
            return Ok((
                Self {
                    repo: Some(repo),
                    store: Some(store),
                    meta,
                },
                session,
            ));
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
        Ok((
            Self {
                repo: None,
                store: None,
                meta: SessionMeta::new(&cwd, &cfg.model),
            },
            Session::new(),
        ))
    }

    fn session_id(&self) -> &str {
        &self.meta.id
    }

    /// Persist the session after a turn: rewrite the whole file if compaction rewrote the transcript,
    /// otherwise append just the new messages.
    fn persist(&mut self, session: &Session, compacted: bool) {
        if let Some(store) = &mut self.store {
            let r = if compacted {
                store.rewrite(&session.messages)
            } else {
                store.append_new(&session.messages)
            };
            if let Err(e) = r {
                eprintln!("serve: failed to persist session: {e}");
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

    /// All sessions' metadata, newest first (empty unless in repo mode).
    fn list(&self) -> Vec<SessionMeta> {
        self.repo
            .as_ref()
            .and_then(|r| r.list().ok())
            .unwrap_or_default()
    }

    /// Set the current session's title.
    fn set_title(&mut self, title: &str, messages: &[agent_core::Message]) -> std::io::Result<()> {
        if let Some(store) = &mut self.store {
            store.set_title(title, messages)?;
            self.meta = store.meta().clone();
        }
        Ok(())
    }
}

fn not_in_repo_mode() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "not in repo mode (start serve with --session-dir)",
    )
}

/// Run the control loop until stdin closes.
pub async fn serve(cfg: ServeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let (mut persistence, mut session) = Persistence::open(&cfg)?;

    // Assemble the system prompt from the base identity + this repo's project instructions + skills +
    // environment, so the agent behaves like it belongs in the working directory.
    let cwd = std::env::current_dir().unwrap_or_default();
    let system = crate::resources::build_system_prompt(&crate::resources::PromptOptions {
        base: &cfg.system,
        append: cfg.append_system.as_deref(),
        cwd: &cwd,
        include_context_files: cfg.context_files,
        include_skills: true,
    });

    // Slash-command prompt templates (`/name args`) and discoverable skills, for `get_commands` and
    // for expanding a `/name` prompt before it reaches the model.
    let prompt_templates = crate::prompts::discover(&cwd);
    let skills = crate::skills::discover(&cwd);

    let client = GatewayClient::new(cfg.gateway, cfg.key)?;
    let mut agent = Agent::new(Arc::new(client), cfg.model.clone())
        .with_tools(tools::default_registry())
        .with_system(system)
        .with_max_steps(cfg.max_steps)
        .with_context_window(cfg.context_window)
        // Pin this session to a warm prompt-cache node via its stable id.
        .with_cache_key(persistence.session_id().to_string())
        .with_cache_long(cfg.cache_long);
    if let Some(budget) = cfg.thinking {
        agent = agent.with_thinking(budget);
    }

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
            json!({ "type": "ready", "session_id": persistence.session_id(), "model": cfg.model }),
        )
        .is_err()
    {
        let _ = writer.await;
        return Ok(());
    }

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
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
                // Expand a `/name args` slash command from a prompt template (no-op otherwise).
                let message = crate::prompts::expand_if_slash(message, &prompt_templates);
                // Optional image attachments: `images: [{media_type, data}]` (base64). Builds a
                // multimodal user turn; absent or empty → a plain text turn.
                let images = parse_images(cmd.get("images"));
                if images.is_empty() {
                    session.user(message);
                } else {
                    session.push(agent_core::Message::user_with_images(message, images));
                }
                let tx = out_tx.clone();
                let cancel = CancellationToken::new();
                // The sink sets this when the loop compacts mid-run, so we know to rewrite (not append)
                // the persisted transcript afterwards.
                let compacted = Arc::new(AtomicBool::new(false));
                let compacted_sink = compacted.clone();
                // Queue for `steer`/`follow_up` messages a client sends while the run is in flight.
                let steering = agent_core::Steering::new();

                // Drive the run while staying responsive to stdin: `abort` cancels it, `steer`/
                // `follow_up` queue a message to inject at the next stop boundary; any other command is
                // rejected as busy (the session is borrowed by the in-flight run). If stdin closes
                // mid-run, cancel and drain. The block scopes the run's `&mut session` borrow so we can
                // persist after.
                let mut stdin_open = true;
                let result = {
                    let run = agent.run_events_steered(
                        &mut session,
                        move |ev| {
                            if matches!(ev, AgentEvent::Compacted { .. }) {
                                compacted_sink.store(true, Ordering::Relaxed);
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
                                        cmd @ ("steer" | "follow_up") => {
                                            match c.get("message").and_then(Value::as_str) {
                                                Some(m) => {
                                                    steering.push(m);
                                                    let _ = out_tx.send(response(cid, cmd, true, None, None));
                                                }
                                                None => {
                                                    let _ = out_tx.send(response(cid, cmd, false, None, Some("missing `message`")));
                                                }
                                            }
                                        }
                                        other => {
                                            let _ = out_tx.send(response(cid, other, false, None, Some("busy: a prompt is running; only `abort`/`steer`/`follow_up` are accepted")));
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

                persistence.persist(&session, compacted.load(Ordering::Relaxed));
                let frame = match result {
                    Ok(()) => response(
                        id.clone(),
                        "prompt",
                        true,
                        Some(session_stats(&session)),
                        None,
                    ),
                    Err(e) => response(id.clone(), "prompt", false, None, Some(&e.to_string())),
                };
                emit!(frame);
                if !stdin_open {
                    break;
                }
            }
            "abort" => {
                // No run is in flight (a mid-run abort is handled inside the `prompt` arm above), so
                // there is nothing to cancel — acknowledge idempotently.
                emit!(response(id, "abort", true, None, None));
            }
            "get_state" => {
                let mut data = session_stats(&session);
                if let Value::Object(m) = &mut data {
                    m.insert("session_id".into(), json!(persistence.session_id()));
                    m.insert("model".into(), json!(cfg.model));
                    m.insert("message_count".into(), json!(session.messages.len()));
                    m.insert("title".into(), json!(persistence.meta.title));
                }
                emit!(response(id, "get_state", true, Some(data), None));
            }
            "get_messages" => {
                let messages =
                    serde_json::to_value(session.messages.as_ref()).unwrap_or(Value::Null);
                emit!(response(
                    id,
                    "get_messages",
                    true,
                    Some(json!({ "messages": messages })),
                    None,
                ));
            }
            "new_session" => {
                session = persistence.new_session(&cfg.model);
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
            "switch_session" => match cmd.get("session_id").and_then(Value::as_str) {
                Some(target) => match persistence.switch(target) {
                    Ok(s) => {
                        session = s;
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
                let result = agent
                    .compact(&mut session, &CancellationToken::new(), &mut |ev| {
                        if let Some(frame) = event_frame(ev) {
                            let _ = tx.send(frame);
                        }
                    })
                    .await;
                match result {
                    Ok(did) => {
                        persistence.persist(&session, did);
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
                emit!(response(
                    id,
                    "get_commands",
                    true,
                    Some(json!({ "commands": commands })),
                    None,
                ));
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
        "reasoning_tokens": session.reasoning_tokens,
        "last_input_tokens": session.last_input_tokens,
    })
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
