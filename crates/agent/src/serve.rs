//! Headless `serve` — a newline-delimited JSON control protocol over stdio.
//!
//! The server is the source of truth; any client (a TUI, an editor, or an `ssh` pipe) drives it by
//! writing one JSON command per line to stdin and reading one JSON frame per line from stdout. The
//! shape mirrors pi's `rpc` mode and opencode's session server: commands get a `response` frame,
//! and a `prompt` streams `event` frames (the agent's `AgentEvent`s) before its response.
//!
//! Session state is `serde`-persisted to `--session-file` after each turn, so a client can detach
//! and a later `serve` over the same file reattaches with the full transcript (`get_messages`).
//!
//! Commands (stdin): `{id?, type, …}`
//!   - `{type:"prompt", message}`        run a turn; streams `event` frames, then a `response`
//!   - `{type:"get_state"}`              → `data: {session_id, model, steps, message_count, …}`
//!   - `{type:"get_messages"}`           → `data: {messages: [...]}`
//!   - `{type:"new_session"}`            reset transcript → `data: {session_id}`
//!
//! Frames (stdout) are either `{type:"response", id?, command, success, data?, error?}` or
//! `{type:"event", event: <AgentEvent>}`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{Agent, AgentEvent, GatewayClient, Session};
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use crate::tools;

/// Options for the headless server (mirrors `run`, plus a session file).
pub struct ServeConfig {
    pub gateway: String,
    pub key: String,
    pub model: String,
    pub max_steps: u32,
    pub system: String,
    pub session_file: Option<String>,
}

/// Run the control loop until stdin closes.
pub async fn serve(cfg: ServeConfig) -> Result<(), Box<dyn std::error::Error>> {
    let client = GatewayClient::new(cfg.gateway, cfg.key)?;
    let agent = Agent::new(Arc::new(client), cfg.model.clone())
        .with_tools(tools::default_registry())
        .with_system(cfg.system)
        .with_max_steps(cfg.max_steps);

    let mut session = match &cfg.session_file {
        Some(path) => load_session(path),
        None => Session::new(),
    };
    let mut session_id = make_id();

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
        .send(json!({ "type": "ready", "session_id": session_id, "model": cfg.model }))
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
                session.user(message);
                let tx = out_tx.clone();
                let result = agent
                    .run_events(&mut session, move |ev| {
                        // Best-effort: a sync sink can't break the control loop. If the writer is
                        // gone the send fails here and the terminal response send below detects it
                        // via `emit!` and stops the loop. An unserializable event is skipped rather
                        // than emitted as a malformed frame (see `event_frame`).
                        if let Some(frame) = event_frame(ev) {
                            let _ = tx.send(frame);
                        }
                    })
                    .await;

                if let Some(path) = &cfg.session_file {
                    if let Err(e) = save_session(path, &session) {
                        eprintln!("serve: failed to persist session to {path}: {e}");
                    }
                }
                let frame = match result {
                    Ok(()) => response(
                        id.clone(),
                        "prompt",
                        true,
                        Some(
                            json!({ "steps": session.steps, "input_tokens": session.input_tokens, "output_tokens": session.output_tokens }),
                        ),
                        None,
                    ),
                    Err(e) => response(id.clone(), "prompt", false, None, Some(&e.to_string())),
                };
                emit!(frame);
            }
            "get_state" => {
                let data = json!({
                    "session_id": session_id,
                    "model": cfg.model,
                    "steps": session.steps,
                    "message_count": session.messages.len(),
                    "input_tokens": session.input_tokens,
                    "output_tokens": session.output_tokens,
                });
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
                session = Session::new();
                session_id = make_id();
                if let Some(path) = &cfg.session_file {
                    if let Err(e) = save_session(path, &session) {
                        eprintln!("serve: failed to persist session to {path}: {e}");
                    }
                }
                emit!(response(
                    id,
                    "new_session",
                    true,
                    Some(json!({ "session_id": session_id })),
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

fn load_session(path: &str) -> Session {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Persist the session to `path` atomically. Returns the error so the caller can surface a failed
/// write (read-only path, full disk) instead of silently losing the transcript.
///
/// Writes to a sibling temp file then `rename`s it over `path`: `rename(2)` is atomic on a single
/// filesystem, so a reader (or a later reattach) ever sees either the old transcript or the new one,
/// never a half-written file. A bare `std::fs::write` truncates in place — a kill mid-write would
/// leave corrupt JSON, which `load_session` then silently discards as an empty session, losing the
/// whole transcript the session file exists to preserve.
fn save_session(path: &str, session: &Session) -> std::io::Result<()> {
    let s = serde_json::to_string(session)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, s)?;
    std::fs::rename(&tmp, path)
}

/// A short session id, unique within a process. The nanosecond timestamp orders ids roughly by
/// creation, and the process-local counter guarantees uniqueness even for ids minted in the same
/// nanosecond (or if the clock is unavailable and `nanos` falls back to 0).
fn make_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("sess_{nanos:x}_{seq:x}")
}
