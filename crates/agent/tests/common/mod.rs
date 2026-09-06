//! Shared test helpers: a mock model server speaking Anthropic SSE, port helpers, and a locator for
//! the gateway binary.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, dead_code)]

use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};

/// Deterministic dev signing public key (standard base64), for a gateway `[signing_keys] 1 = …`.
pub const DEV_PUBKEY_B64: &str = "6kpsY+KcUgq+9VB7Ey7F+ZVHdq6+vnuSQh7qaRRG0iw=";
/// The matching dev `bai_v1` token (tenant 1 / vpc 1, kid 1).
pub const DEV_TOKEN: &str = "bai_v1.1.AQAAAAAAAAABAAAAAAAAAA.WrWcPbklu91PS-4WuR6GnBNF3h4nROpH0EQQlfJf06f7_lEnlQOCSBimhH2JMwXFJgw40BniTB7-yIdFnpldDw";

/// A `HOME` that deliberately doesn't exist on disk. `serve`/`run` read `~/.claude/skills` and
/// `~/.claude/trusted-projects.json` unconditionally (skill discovery is no longer gated on project
/// trust — an untrusted project must not blank out the user's own global skills), and every codepath
/// that reads under `HOME` already treats a missing file/directory as "nothing there" rather than an
/// error, so this keeps a test hermetic (never sees, and can't pollute, the actual developer's real
/// `~/.claude/`) without needing a `TempDir` guard kept alive for the spawned process's lifetime. A test
/// that specifically wants real HOME-relative behavior (trust store writes, seeded skills) overrides
/// this afterward via its own `.env("HOME", ...)`, which simply wins — `Command::env` is last-write.
pub const ISOLATED_HOME: &str = "/nonexistent-beyond-ai-agent-test-home";

/// A spawned child process that is killed and reaped when it goes out of scope — **including when the
/// enclosing test panics**.
///
/// `std::process::Child` deliberately does *not* kill on drop, so the usual test shape
///
/// ```ignore
/// let mut child = serve_cmd(..).spawn().unwrap();
/// assert_eq!(thing, other);   // <-- panics here
/// let _ = child.kill();       // <-- never runs
/// ```
///
/// orphans a real `serve` daemon on every failing assertion. Those orphans are reparented to init and
/// survive the whole `cargo test` run: they hold their listening port, their session directory (long
/// after the `TempDir` is gone), and tens of MB of RSS each. `--session-idle-timeout 0` tests pin
/// sessions for the daemon's lifetime by design, so those leaks never self-reap at all.
///
/// The compounding is what makes this worth a guard rather than more `kill()` calls: one genuine
/// failure leaks a daemon, the leak starves the *next* concurrent test of ports/memory, and that one
/// fails too — turning a single real bug into a cascade of unrelated red tests, which is exactly the
/// shape that makes a flaky suite impossible to read. Killing on drop makes a failure cost exactly one
/// test.
///
/// Derefs to [`Child`], so `child.stdin.take()`, `child.kill()`, and `child.wait()` all keep working
/// unchanged; an explicit `wait()` before the drop is fine (the drop's `kill` on an already-reaped pid
/// fails harmlessly and is ignored).
pub struct ChildGuard(Option<std::process::Child>);

impl ChildGuard {
    /// Spawn `cmd` under the guard. Panics with the command's name on failure, matching the
    /// `.spawn().unwrap()` this replaces.
    pub fn spawn(cmd: &mut Command) -> Self {
        Self(Some(cmd.spawn().unwrap_or_else(|e| {
            panic!("failed to spawn {:?}: {e}", cmd.get_program())
        })))
    }

    /// [`Child::wait_with_output`], which consumes the child and so can't come through `Deref`. Taking
    /// the child out disarms the guard — safe precisely because this call reaps the process itself.
    pub fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        self.0.take().expect("child taken").wait_with_output()
    }
}

impl std::ops::Deref for ChildGuard {
    type Target = std::process::Child;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("child taken")
    }
}

impl std::ops::DerefMut for ChildGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().expect("child taken")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// `cmd.spawn_guarded()` — [`ChildGuard::spawn`] as a method, so an existing
/// `Command`-building chain only has to swap its trailing `.spawn().unwrap()`.
pub trait SpawnGuarded {
    fn spawn_guarded(&mut self) -> ChildGuard;
}

impl SpawnGuarded for Command {
    fn spawn_guarded(&mut self) -> ChildGuard {
        ChildGuard::spawn(self)
    }
}

/// Locate the gateway binary (built beside the agent binary); build it on demand if absent.
pub fn gateway_bin() -> PathBuf {
    let agent = PathBuf::from(env!("CARGO_BIN_EXE_beyond-ai-agent"));
    let dir = agent.parent().unwrap();
    let gw = dir.join("beyond-ai");
    if !gw.exists() {
        let mut args = vec!["build", "-q", "-p", "beyond-ai", "--bin", "beyond-ai"];
        if agent.to_string_lossy().contains("/release/") {
            args.push("--release");
        }
        let status = Command::new(env!("CARGO"))
            .args(&args)
            .status()
            .expect("build gateway");
        assert!(status.success(), "failed to build the gateway binary");
    }
    gw
}

/// An Anthropic SSE turn that calls one tool with the given JSON-argument string.
pub fn turn_tool_use(id: &str, name: &str, args_json: &str) -> String {
    sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": args_json } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
        json!({ "type": "message_stop" }),
    ])
}

/// An Anthropic SSE turn that emits text and ends.
pub fn turn_text(text: &str) -> String {
    sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 12, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 6 } }),
        json!({ "type": "message_stop" }),
    ])
}

/// An OpenAI **Responses** dialect SSE turn that emits text and completes — the `gpt-4o` counterpart
/// to `turn_text`'s Anthropic shape, for tests that need to inspect a Responses-dialect-only wire
/// field (e.g. `prompt_cache_key`, which Anthropic's dialect never sends).
pub fn turn_text_responses(text: &str) -> String {
    sse(&[
        json!({ "type": "response.output_item.added", "output_index": 0, "item": { "type": "message", "id": "msg_1" } }),
        json!({ "type": "response.output_text.delta", "output_index": 0, "delta": text }),
        json!({ "type": "response.output_item.done", "output_index": 0, "item": { "type": "message", "id": "msg_1", "phase": "final_answer", "content": [{ "type": "output_text", "text": text }] } }),
        json!({ "type": "response.completed", "response": { "status": "completed", "usage": { "input_tokens": 1, "output_tokens": 1 } } }),
    ])
}

/// An Anthropic SSE turn that ends with `stop_reason: "refusal"` — a distinct terminal condition from
/// a normal end-of-turn (see `agent_core::agent::Agent::run_events_steered`).
pub fn turn_refusal(text: &str) -> String {
    sse(&[
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 12, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "refusal" }, "usage": { "output_tokens": 6 } }),
        json!({ "type": "message_stop" }),
    ])
}

/// Frame each event as its own `data: ...\n\n` SSE block — used directly by callers that build a turn
/// out of the standard shape (`turn_text`/`turn_refusal`/`turn_tool_use`) as well as ones assembling a
/// bespoke event sequence (e.g. chunked tool-call argument streaming).
pub fn sse(events: &[Value]) -> String {
    events.iter().map(|e| format!("data: {e}\n\n")).collect()
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
            let len = headers
                .lines()
                .find_map(|l| {
                    l.strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            // Keep reading until the full body has arrived, then return the WHOLE raw request
            // (headers + body) so callers can assert on both (e.g. a swapped-in pool key).
            let need = pos + 4 + len;
            while buf.len() < need {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            return String::from_utf8_lossy(&buf).into_owned();
        }
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            return String::from_utf8_lossy(&buf).into_owned();
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

/// Spawn a model server answering `responses` in order, recording each full raw request (headers +
/// body). Returns the base URL and the shared record of requests.
pub fn spawn_model_server(responses: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorder = requests.clone();
    thread::spawn(move || {
        for resp in responses {
            if let Ok((mut stream, _)) = listener.accept() {
                let req = read_http_request(&mut stream);
                recorder.lock().unwrap().push(req);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{resp}"
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            }
        }
    });
    (format!("http://{addr}"), requests)
}

/// A model server that picks its reply by matching a substring against each request's body, rather than
/// answering in arrival order. This is what parallel `subagent` tests need: several children hit the
/// server concurrently in a nondeterministic order, so [`spawn_model_server`]'s strict FIFO queue would
/// hand child A's reply to child B. Each `routes` entry is `(needle, sse_response)`; the first entry
/// whose `needle` appears anywhere in the raw request wins. An unmatched request gets `fallback`.
///
/// A route may be used any number of times (a retry, or two children with the same marker), so this
/// serves an unbounded number of requests until the listener is dropped. Records every raw request.
pub fn spawn_model_server_routed(
    routes: Vec<(String, String)>,
    fallback: String,
) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorder = requests.clone();
    thread::spawn(move || {
        // Each accepted connection is served on its own thread so concurrent children don't serialize
        // behind one another (a slow child must not block a fast sibling's reply).
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let routes = routes.clone();
            let fallback = fallback.clone();
            let recorder = recorder.clone();
            thread::spawn(move || {
                let req = read_http_request(&mut stream);
                let resp = routes
                    .iter()
                    .find(|(needle, _)| req.contains(needle.as_str()))
                    .map(|(_, r)| r.clone())
                    .unwrap_or(fallback);
                recorder.lock().unwrap().push(req);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{resp}"
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            });
        }
    });
    (format!("http://{addr}"), requests)
}

/// A model server whose first `fast.len()` requests get an instant response, and whose next request
/// (e.g. the summarization call a `switch_branch{summarize:true}` triggers, or a plain `prompt`'s own
/// model call) sends only a partial SSE body — proving the request genuinely reached the server and
/// started streaming — then stalls for `stall` before completing, giving a test a reliable window to
/// `abort` (or, for pi-parity Task 4's busy-then-self-abort commands, send `compact`/`switch_session`/
/// `fork`/`clone`/`new_session` mid-run) a provably in-flight call instead of racing a near-instant
/// local round trip. Every request in `after` then gets its own instant response too, in order — for a
/// test whose busy-time command itself makes a further model call once it resumes idle (e.g. `compact`,
/// when the session has enough content to attempt a real summarization rather than short-circuiting as
/// too small).
pub fn spawn_model_server_with_stalled_response(
    fast: Vec<String>,
    stall: std::time::Duration,
    after: Vec<String>,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || {
        for resp in fast {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{resp}"
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            }
        }
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let preamble = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n\
                data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n";
            let _ = stream.write_all(preamble.as_bytes());
            let _ = stream.flush();
            thread::sleep(stall);
            // Finishes the turn normally as a fallback safety net in case a test using this doesn't
            // interrupt it before `stall` elapses — a silently-hanging server would fail such a test
            // far more confusingly than a completed-but-too-late response would.
            let rest = "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
                data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"recap\"}}\n\n\
                data: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
                data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n\
                data: {\"type\":\"message_stop\"}\n\n";
            let _ = stream.write_all(rest.as_bytes());
            let _ = stream.flush();
        }
        for resp in after {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{resp}"
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            }
        }
    });
    format!("http://{addr}")
}

/// A free localhost port (bind `:0`, read it back, release). A subprocess must bind it promptly;
/// there's a small TOCTOU window, acceptable for tests.
pub fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Block until `port` accepts a TCP connection, or panic after ~5s.
pub fn wait_for_port(port: u16) {
    for _ in 0..500 {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("port {port} never came up");
}

/// Read stdout frames from a `serve` child until the `response` frame for `command` arrives; return
/// all frames seen (including any `event`/progress frames along the way).
pub fn read_until_response(reader: &mut impl BufRead, command: &str) -> Vec<Value> {
    let mut frames = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        let done = v.get("type").and_then(Value::as_str) == Some("response")
            && v.get("command").and_then(Value::as_str) == Some(command);
        frames.push(v);
        if done {
            break;
        }
    }
    frames
}

/// Strip ambient provider-routing env so a mock `--gateway-url` actually wins.
///
/// Eval hosts (this one included) export `AI_DIRECT=1` / `AI_PROVIDER=openrouter` /
/// `OPENROUTER_API_KEY` for Harbor runs. `AI_DIRECT=1` makes the binary ignore `--gateway-url`
/// and dial OpenRouter; a hermetic test that inherited that would bill a live provider and fail
/// on `claude-test`. Tests that *want* those vars (the ignored live Code Mode smoke) re-set them
/// after [`run_cmd`] — `Command::env` is last-write.
fn isolate_provider_env(cmd: &mut Command) {
    for key in [
        "AI_DIRECT",
        "AI_PROVIDER",
        "AI_BASE_URL",
        "AI_API_KEY",
        "AI_GATEWAY_URL",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
    ] {
        cmd.env_remove(key);
    }
}

/// A `serve` child bound to a single session file, talking to the mock gateway at `base`.
pub fn serve_cmd(bin: &str, base: &str, session_file: &str) -> Command {
    let mut c = Command::new(bin);
    isolate_provider_env(&mut c);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-file",
        session_file,
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

/// Like `serve_cmd`, but bound to a session *directory* (`--session-dir`) rather than a single file —
/// exercises the multi-session-per-process repo mode (`list_sessions`, `switch`, fork).
pub fn serve_dir_cmd(bin: &str, base: &str, session_dir: &str) -> Command {
    let mut c = Command::new(bin);
    isolate_provider_env(&mut c);
    c.args([
        "serve",
        "--gateway-url",
        base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
        "--session-dir",
        session_dir,
    ])
    .env("HOME", ISOLATED_HOME)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::null());
    c
}

/// [`Command::new`] for the `run` binary, pre-isolated from the real machine's `HOME` — see
/// [`ISOLATED_HOME`]. A test that wants real HOME-relative behavior overrides it via its own
/// `.env("HOME", ...)`, which simply wins (`Command::env` is last-write).
pub fn run_cmd(bin: &str) -> Command {
    let mut c = Command::new(bin);
    isolate_provider_env(&mut c);
    c.env("HOME", ISOLATED_HOME);
    c
}

/// The fixed WebSocket path `serve --listen` accepts (see `serve_ws`).
pub const WS_PATH: &str = "/_beyond/agent";

/// A connected test WebSocket client (over plain `ws://`, so `MaybeTlsStream` is always the plain arm).
pub type TestWs =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Connect a WebSocket client to a `serve --listen` port, optionally naming a session via the URL.
pub async fn ws_connect(port: u16, session_id: Option<&str>) -> TestWs {
    let url = match session_id {
        Some(id) => format!("ws://127.0.0.1:{port}{WS_PATH}?session_id={id}"),
        None => format!("ws://127.0.0.1:{port}{WS_PATH}"),
    };
    let (ws, _resp) = tokio_tungstenite::connect_async(url)
        .await
        .expect("websocket connect");
    ws
}

/// A test WebSocket client running over a Unix-domain socket (the `--listen-uds` transport). The
/// generic stream arm differs from [`TestWs`]'s TCP one, so UDS helpers take this type; the frame
/// helpers ([`ws_send`], [`ws_next_frame`], [`ws_read_until_response`]) are generic over the stream.
pub type TestWsUds = tokio_tungstenite::WebSocketStream<tokio::net::UnixStream>;

/// Dial a `serve --listen-uds` socket at `path`, optionally naming a session via the URL. The HTTP
/// `Host` is ignored over a UDS, so it's a synthetic `localhost`; only the path + `?session_id=` query
/// matter (the handshake validates the path and parses the session id).
pub async fn ws_connect_uds(path: &std::path::Path, session_id: Option<&str>) -> TestWsUds {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let url = match session_id {
        Some(id) => format!("ws://localhost{WS_PATH}?session_id={id}"),
        None => format!("ws://localhost{WS_PATH}"),
    };
    let request = url.into_client_request().expect("build ws request");
    let stream = tokio::net::UnixStream::connect(path)
        .await
        .expect("connect unix socket");
    let (ws, _resp) = tokio_tungstenite::client_async(request, stream)
        .await
        .expect("websocket handshake over uds");
    ws
}

/// Send one command object as a single WS text message. Generic over the underlying transport so the
/// same helper drives both the TCP ([`TestWs`]) and UDS ([`TestWsUds`]) clients.
pub async fn ws_send<T>(ws: &mut tokio_tungstenite::WebSocketStream<T>, v: Value)
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    use futures::SinkExt as _;
    ws.send(tokio_tungstenite::tungstenite::Message::Text(
        v.to_string().into(),
    ))
    .await
    .expect("websocket send");
}

/// Read the next JSON frame (skipping ping/pong/binary), or `None` if the socket closed.
pub async fn ws_next_frame<T>(ws: &mut tokio_tungstenite::WebSocketStream<T>) -> Option<Value>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    use futures::StreamExt as _;
    use tokio_tungstenite::tungstenite::Message;
    while let Some(msg) = ws.next().await {
        match msg.expect("websocket recv") {
            Message::Text(t) => {
                if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                    return Some(v);
                }
            }
            Message::Close(_) => return None,
            _ => {}
        }
    }
    None
}

/// Collect WS frames until the `response` frame for `command` arrives (or the socket closes).
pub async fn ws_read_until_response<T>(
    ws: &mut tokio_tungstenite::WebSocketStream<T>,
    command: &str,
) -> Vec<Value>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut frames = Vec::new();
    while let Some(v) = ws_next_frame(ws).await {
        let done = v.get("type").and_then(Value::as_str) == Some("response")
            && v.get("command").and_then(Value::as_str) == Some(command);
        frames.push(v);
        if done {
            break;
        }
    }
    frames
}

/// The JSON body of a raw recorded request (as `spawn_model_server` records it: headers + body).
pub fn body_json(raw_request: &str) -> Value {
    let body = raw_request
        .split_once("\r\n\r\n")
        .expect("request must have a body")
        .1;
    serde_json::from_str(body).expect("request body must be JSON")
}

/// The names of the tools a request *advertised* to the model.
///
/// Not a substring search on the raw request: the body also carries the conversation history, whose
/// `tool_use`/`tool_result` blocks name every tool the model has ever called. A turn that no longer
/// offers a tool still mentions it, so only the `tools` array answers "what may the model call now?".
pub fn advertised_tools(raw_request: &str) -> Vec<String> {
    body_json(raw_request)["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| t["name"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Every persisted `message` entry's id, in order, read straight off a session JSONL file.
pub fn message_ids(session_file: &str) -> Vec<String> {
    std::fs::read_to_string(session_file)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v["type"] == "message")
        .filter_map(|v| v["id"].as_str().map(str::to_string))
        .collect()
}
