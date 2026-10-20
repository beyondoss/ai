//! The demo's client: one WebSocket connection to a service-mode agent, one conversation step.
//!
//! It prints what the user would watch to stderr, appends every frame it sees to an events file
//! (one JSON line each, wall-clock stamped) and prints one JSON summary to stdout for `demo.sh` to
//! turn into a check. The grant comes from `DEMO_GRANT`, never argv.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::{SinkExt as _, StreamExt as _};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
use tokio_tungstenite::tungstenite::http::HeaderValue;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub struct Args {
    pub host: String,
    pub port: u16,
    pub session: String,
    pub prompt: Option<String>,
    /// A raw protocol command to send instead of a prompt; its `response` is the summary.
    pub command: Option<String>,
    /// Keep listening this long for a turn a background job's end wakes. After the prompt's own
    /// turn, or, with no prompt and no command, from the moment of connecting.
    pub wait_wake_secs: u64,
    pub events: Option<PathBuf>,
    pub timeout_secs: u64,
    /// Run `true` in the sandbox once connected. A replica reaches a session's sandbox lazily, and a
    /// reopened session learns what its jobs did when it does; this is the touch that makes it.
    pub touch_sandbox: bool,
}

/// Whether `frame` is the agent's `ack` (or an early `response`) to the prompt sent with `id`: the
/// first frame of the connection that is about that prompt rather than an earlier run's replay.
fn answers_prompt(frame: &Value, id: &str) -> bool {
    matches!(frame["type"].as_str(), Some("ack" | "response"))
        && frame["command"] == "prompt"
        && frame["id"] == id
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

struct Log(Option<std::fs::File>);

impl Log {
    fn frame(&mut self, session: &str, frame: &Value) {
        if let Some(f) = &mut self.0 {
            // One write per line: two chats append to the same file at once (a connection held open
            // while another asks), and a line written in pieces would interleave.
            let line = format!(
                "{}\n",
                json!({"ts_ms": now_ms(), "src": "client", "session": session, "frame": frame})
            );
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// Connect and run one step. A refused connection is a result, not an error: `{"refused":true,
/// "status":401}`.
pub async fn run(a: Args) -> Res<Value> {
    let grant = std::env::var("DEMO_GRANT").map_err(|_| "DEMO_GRANT is not set")?;
    let url = format!(
        "ws://{}:{}/_beyond/agent?session_id={}",
        a.host, a.port, a.session
    );
    let mut req = url.into_client_request()?;
    req.headers_mut()
        .insert("x-beyond-grant", grant.parse::<HeaderValue>()?);
    let t0 = Instant::now();
    let mut ws = match tokio_tungstenite::connect_async(req).await {
        Ok((ws, _)) => ws,
        Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => {
            return Ok(json!({"refused": true, "status": resp.status().as_u16()}));
        }
        Err(e) => return Err(e.into()),
    };
    let connected_ms = t0.elapsed().as_millis() as u64;
    let connected_at_ms = now_ms();
    let mut log = Log(match &a.events {
        Some(p) => Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)?,
        ),
        None => None,
    });

    let (mut sent_at, mut sent_at_ms, mut first_token_ms) = (Instant::now(), now_ms(), None::<u64>);
    // Unique per connection, so a replayed run's own ack for an earlier prompt cannot be mistaken for ours.
    let prompt_id = format!("p{}", now_ms());
    if let Some(p) = &a.prompt {
        eprintln!("  you      > {p}");
        ws.send(Message::text(
            json!({"type": "prompt", "id": prompt_id, "message": p}).to_string(),
        ))
        .await?;
        sent_at = Instant::now();
        sent_at_ms = now_ms();
        log.frame(
            &a.session,
            &json!({"type": "demo_prompt_sent", "message": p}),
        );
    } else if let Some(c) = &a.command {
        ws.send(Message::text(c.clone())).await?;
    }

    if a.touch_sandbox {
        ws.send(Message::text(
            json!({"type": "bash", "id": "touch", "command": "true"}).to_string(),
        ))
        .await?;
    }
    let mut deadline = Instant::now() + Duration::from_secs(a.timeout_secs);
    // With no prompt of our own, the only turn there is to wait for is the one a job's end wakes.
    let mut wake_wait = a.prompt.is_none() && a.command.is_none() && a.wait_wake_secs > 0;
    if wake_wait {
        deadline = Instant::now() + Duration::from_secs(a.wait_wake_secs);
    }
    let (mut text, mut cur_line) = (String::new(), String::new());
    let (mut tools, mut background) = (Vec::<Value>::new(), Vec::<Value>::new());
    let mut responses = Vec::<Value>::new();
    let (mut turns_done, mut woke_at_ms, mut wake_done_ms) = (0usize, None::<u64>, None::<u64>);

    // Attaching to a session whose run is still in flight replays that run's frames ahead of
    // everything else (`catchup`, then the turn so far). They belong to an earlier prompt, so with a
    // prompt of our own nothing counts, or is logged, until the agent acknowledges it.
    let mut replaying = a.prompt.is_some();
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        let msg = match tokio::time::timeout(left, ws.next()).await {
            Err(_) | Ok(None) => break,
            Ok(Some(m)) => m?,
        };
        let Message::Text(raw) = msg else { continue };
        let Ok(frame) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        if replaying {
            replaying = !answers_prompt(&frame, &prompt_id);
            if replaying {
                continue;
            }
        }
        log.frame(&a.session, &frame);
        match frame["type"].as_str().unwrap_or("") {
            "event" => {
                let ev = &frame["event"];
                match ev["kind"].as_str().unwrap_or("") {
                    "stream" if ev["type"] == "text_delta" => {
                        let d = ev["text"].as_str().unwrap_or("");
                        if !d.is_empty() && first_token_ms.is_none() {
                            first_token_ms = Some(sent_at.elapsed().as_millis() as u64);
                        }
                        text.push_str(d);
                        cur_line.push_str(d);
                    }
                    // Each model call is its own message: keep them apart in the text.
                    "turn_start" => {
                        flush(&mut cur_line);
                        if !text.is_empty() && !text.ends_with('\n') {
                            text.push('\n');
                        }
                    }
                    "tool_start" => {
                        flush(&mut cur_line);
                        eprintln!(
                            "  tool     > {} {}",
                            ev["name"].as_str().unwrap_or(""),
                            short(&ev["input"].to_string(), 120)
                        );
                        tools.push(
                            json!({"name": ev["name"], "input": ev["input"], "t_ms": now_ms()}),
                        );
                    }
                    "tool_end" => {
                        eprintln!(
                            "  result   > {}",
                            short(ev["result"].as_str().unwrap_or(""), 160)
                        );
                        if let Some(t) = tools.last_mut() {
                            t["output"] = ev["result"].clone();
                            t["end_ms"] = json!(now_ms());
                        }
                    }
                    "background" => {
                        flush(&mut cur_line);
                        eprintln!(
                            "  notice   > {}",
                            short(ev["text"].as_str().unwrap_or(""), 160)
                        );
                        background.push(json!({"reason": ev["reason"], "text": ev["text"],
                            "wake": ev["wake"], "t_ms": now_ms()}));
                        if ev["wake"] == json!(true) && woke_at_ms.is_none() {
                            woke_at_ms = Some(now_ms());
                        }
                    }
                    _ => {}
                }
            }
            "response" => {
                responses.push(frame.clone());
                if frame["command"] == "prompt" {
                    flush(&mut cur_line);
                    turns_done += 1;
                    if wake_wait {
                        wake_done_ms = Some(now_ms());
                        break;
                    } else if a.wait_wake_secs > 0 {
                        // The prompt's own turn is over; now wait for the job's end to wake the session.
                        wake_wait = true;
                        deadline = Instant::now() + Duration::from_secs(a.wait_wake_secs);
                    } else {
                        break;
                    }
                } else if a.command.is_some() {
                    break;
                }
            }
            _ => {}
        }
    }
    flush(&mut cur_line);
    let _ = ws.close(None).await;
    let prompt_response = responses
        .iter()
        .find(|r| r["command"] == "prompt")
        .cloned()
        .unwrap_or(Value::Null);
    let wake_response = responses
        .iter()
        .filter(|r| r["command"] == "prompt")
        .nth(usize::from(a.prompt.is_some()))
        .cloned()
        .unwrap_or(Value::Null);
    Ok(json!({
        "refused": false,
        "session": a.session,
        "connected_ms": connected_ms,
        "connected_at_ms": connected_at_ms,
        "sent_at_ms": sent_at_ms,
        "done_at_ms": now_ms(),
        "first_token_ms": first_token_ms,
        "total_ms": sent_at.elapsed().as_millis() as u64,
        "turns": turns_done,
        "text": text,
        "tools": tools,
        "background": background,
        "woke_at_ms": woke_at_ms,
        "wake_done_ms": wake_done_ms,
        "prompt_response": prompt_response,
        "wake_response": wake_response,
        "responses": responses,
    }))
}

fn flush(line: &mut String) {
    if !line.is_empty() {
        eprintln!("  agent    > {}", line.trim());
        line.clear();
    }
}

fn short(s: &str, n: usize) -> String {
    let one: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one.chars().count() <= n {
        one
    } else {
        format!("{}…", one.chars().take(n).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_prompts_own_ack_ends_the_replay() {
        let tool = json!({"type": "event", "event": {"kind": "tool_start"}});
        assert!(!answers_prompt(&tool, "p2"));
        // The replayed run answers its own prompt, under another id.
        let earlier = json!({"type": "response", "command": "prompt", "id": "p1"});
        assert!(!answers_prompt(&earlier, "p2"));
        let ack = json!({"type": "ack", "command": "prompt", "id": "p2"});
        assert!(answers_prompt(&ack, "p2"));
    }
}
