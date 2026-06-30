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
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Value>();
    let writer = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        while let Some(frame) = out_rx.recv().await {
            if let Ok(line) = serde_json::to_string(&frame) {
                let _ = out.write_all(line.as_bytes()).await;
                let _ = out.write_all(b"\n").await;
                let _ = out.flush().await;
            }
        }
    });

    // Announce readiness so a client can sync before issuing commands.
    let _ = out_tx.send(json!({ "type": "ready", "session_id": session_id, "model": cfg.model }));

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cmd: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                let _ = out_tx.send(response(
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
                let message = cmd
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                session.user(message);
                let tx = out_tx.clone();
                let result = agent
                    .run_events(&mut session, move |ev| {
                        let _ = tx.send(event_frame(ev));
                    })
                    .await;

                if let Some(path) = &cfg.session_file {
                    save_session(path, &session);
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
                let _ = out_tx.send(frame);
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
                let _ = out_tx.send(response(id, "get_state", true, Some(data), None));
            }
            "get_messages" => {
                let messages = serde_json::to_value(&session.messages).unwrap_or(Value::Null);
                let _ = out_tx.send(response(
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
                    save_session(path, &session);
                }
                let _ = out_tx.send(response(
                    id,
                    "new_session",
                    true,
                    Some(json!({ "session_id": session_id })),
                    None,
                ));
            }
            other => {
                let _ = out_tx.send(response(id, other, false, None, Some("unknown command")));
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

/// Wrap an `AgentEvent` in an `event` frame.
fn event_frame(ev: AgentEvent) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), json!("event"));
    if let Ok(v) = serde_json::to_value(&ev) {
        m.insert("event".into(), v);
    }
    Value::Object(m)
}

fn load_session(path: &str) -> Session {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_session(path: &str, session: &Session) {
    if let Ok(s) = serde_json::to_string(session) {
        let _ = std::fs::write(path, s);
    }
}

/// A short, monotonic-ish session id (no extra deps; uniqueness across a process is sufficient).
fn make_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("sess_{nanos:x}")
}
