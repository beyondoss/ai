//! Shared harness for the `mcp_events_*` suites: the `mcp_fixture_events_server` binary's control
//! API, settings writing, and a bounded frame reader over a `serve` child's stdout.
//!
//! Every wait here is bounded — a missing frame fails the test with what *was* seen, instead of
//! hanging the suite.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::{ChildGuard, SpawnGuarded};

/// The events fixture binary.
pub const FIXTURE: &str = env!("CARGO_BIN_EXE_mcp_fixture_events_server");

/// One blocking HTTP/1.1 request to the fixture's control API. Plain bytes, no client library.
pub fn control(base: &str, method: &str, path: &str, body: Option<&Value>) -> Value {
    let hostport = base.strip_prefix("http://").expect("http control url");
    let mut stream = TcpStream::connect(hostport).expect("connect to the fixture control API");
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let body = body.map(|b| b.to_string()).unwrap_or_default();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).unwrap();
    let mut buf = String::new();
    stream.read_to_string(&mut buf).unwrap();
    let (_, body) = buf.split_once("\r\n\r\n").unwrap_or(("", ""));
    serde_json::from_str(body).unwrap_or(Value::Null)
}

pub fn state(base: &str) -> Value {
    control(base, "GET", "/control/state", None)
}

pub fn emit(base: &str, body: Value) -> Value {
    control(base, "POST", "/control/emit", Some(&body))
}

/// Poll `f` until it returns `Some`, or panic after `timeout` with `what`.
pub fn eventually<T>(timeout: Duration, what: &str, mut f: impl FnMut() -> Option<T>) -> T {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(v) = f() {
            return v;
        }
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Wait for the fixture to write its control address to `path`.
pub fn wait_control_file(path: &Path) -> String {
    eventually(
        Duration::from_secs(20),
        "the fixture's control file",
        || std::fs::read_to_string(path).ok().filter(|s| !s.is_empty()),
    )
}

/// A streamable-HTTP fixture the test owns. Returns the guard, the MCP URL and the control URL.
pub fn spawn_http_fixture(envs: &[(&str, &str)]) -> (ChildGuard, String, String) {
    let mut cmd = Command::new(FIXTURE);
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn_guarded();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let v: Value = serde_json::from_str(line.trim()).expect("fixture banner");
    (
        child,
        v["mcp"].as_str().unwrap().to_owned(),
        v["control"].as_str().unwrap().to_owned(),
    )
}

/// Write `$HOME/.claude/settings.json` with exactly these `mcp_servers`.
pub fn write_settings(home: &Path, mcp_servers: Value) {
    let dir = home.join(".claude");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("settings.json"),
        serde_json::to_string_pretty(&json!({ "mcp_servers": mcp_servers })).unwrap(),
    )
    .unwrap();
}

/// A stdio fixture server entry for `mcp_servers`, with its control file at `control_file`.
pub fn stdio_server(name: &str, control_file: &Path, extra_env: Value, events: Value) -> Value {
    let mut env = json!({ "MCP_FIXTURE_CONTROL_FILE": control_file.to_string_lossy() });
    if let (Value::Object(e), Value::Object(x)) = (&mut env, extra_env) {
        e.extend(x);
    }
    json!({
        "name": name,
        "transport": "stdio",
        "command": FIXTURE,
        "args": ["--stdio"],
        "env": env,
        "events": events,
    })
}

/// Environment that makes the client side fast and deterministic for tests.
pub fn fast_knobs(cmd: &mut Command) -> &mut Command {
    cmd.env("BEYOND_AI_AGENT_MCP_EVENTS_POLL_FLOOR_MS", "100")
        .env("BEYOND_AI_AGENT_MCP_EVENTS_COALESCE_MS", "300")
        .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
}

/// Frames from a `serve` child's stdout, read on a thread so every wait can be bounded.
pub struct Frames {
    rx: mpsc::Receiver<Value>,
    /// Everything received so far, in order.
    pub seen: Vec<Value>,
    stderr: Option<PathBuf>,
}

impl Frames {
    pub fn new(stdout: ChildStdout, stderr: Option<PathBuf>) -> Self {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<Value>(line.trim())
                    && tx.send(v).is_err()
                {
                    break;
                }
            }
        });
        Self {
            rx,
            seen: Vec::new(),
            stderr,
        }
    }

    fn diagnostics(&self) -> String {
        let stderr = self
            .stderr
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default();
        format!(
            "frames seen: {:#?}\nserve stderr:\n{stderr}",
            self.seen.iter().rev().take(30).collect::<Vec<_>>()
        )
    }

    /// The first frame (from now on) matching `pred`.
    pub fn wait(&mut self, timeout: Duration, what: &str, pred: impl Fn(&Value) -> bool) -> Value {
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.rx.recv_timeout(left) {
                Ok(v) => {
                    self.seen.push(v.clone());
                    if pred(&v) {
                        return v;
                    }
                }
                Err(_) => panic!("timed out waiting for {what}\n{}", self.diagnostics()),
            }
        }
    }

    /// Collect every frame for `window`, returning those matching `pred`.
    pub fn collect(&mut self, window: Duration, pred: impl Fn(&Value) -> bool) -> Vec<Value> {
        let deadline = Instant::now() + window;
        let mut out = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return out;
            }
            if let Ok(v) = self.rx.recv_timeout(left) {
                self.seen.push(v.clone());
                if pred(&v) {
                    out.push(v);
                }
            }
        }
    }

    /// The `response` frame correlated to `id`.
    pub fn response(&mut self, id: &str) -> Value {
        let id = id.to_owned();
        self.wait(
            Duration::from_secs(30),
            &format!("response {id}"),
            move |f| f["type"] == "response" && f["id"] == id.as_str(),
        )
    }
}

/// Send one command line to a stdio `serve`.
pub fn send(stdin: &mut impl Write, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

/// Ask `mcp_events_list` until every expected subscription reports `active`.
pub fn wait_active(stdin: &mut impl Write, frames: &mut Frames, expected: usize) -> Value {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut n = 0;
    loop {
        n += 1;
        let id = format!("list-{n}");
        send(stdin, json!({ "type": "mcp_events_list", "id": id }));
        let r = frames.response(&id);
        let subs = r["data"]["subscriptions"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if subs.len() >= expected && subs.iter().all(|s| s["state"] == "active") {
            return r;
        }
        assert!(
            Instant::now() < deadline,
            "subscriptions never became active: {r:#}\n{}",
            frames.diagnostics()
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Every agent-run model request (one that advertises tools — not the session-title utility call
/// `serve` makes after a session's first run) whose body mentions `needle`.
pub fn model_requests_with(bodies: &std::sync::Mutex<Vec<String>>, needle: &str) -> Vec<String> {
    bodies
        .lock()
        .unwrap()
        .iter()
        .filter(|b| b.contains(needle) && b.contains("\"tools\":["))
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Daemon (`serve --listen`) helpers
// ---------------------------------------------------------------------------------------------

/// The daemon session that owns configured subscriptions by default.
pub const EVENTS_SESSION: &str = "mcp-events";

/// A listening `serve` daemon with `--mcp-events-callback-url` pointing back at it, and its port. The
/// callback URL names the port before the daemon starts, so the port is a [`super::HeldPort`].
pub fn spawn_daemon(home: &Path, base: &str, extra: &[&str]) -> (ChildGuard, u16) {
    spawn_daemon_env(home, base, extra, &[])
}

/// [`spawn_daemon`] with extra environment.
pub fn spawn_daemon_env(
    home: &Path,
    base: &str,
    extra: &[&str],
    env: &[(&str, &str)],
) -> (ChildGuard, u16) {
    let held = super::HeldPort::bind();
    // `held` drops on return: the daemon's is the only copy, so if it dies the port is refused.
    (spawn_daemon_on(home, base, &held, extra, env), held.port())
}

/// [`spawn_daemon_env`] on a port the test holds — to restart a daemon where its callback URL, and
/// so every subscription's, still points.
pub fn spawn_daemon_on(
    home: &Path,
    base: &str,
    held: &super::HeldPort,
    extra: &[&str],
    env: &[(&str, &str)],
) -> ChildGuard {
    let port = held.port();
    let mut cmd = super::serve_dir_cmd(super::BIN, base, &home.join("sessions").to_string_lossy());
    cmd.envs(env.iter().copied());
    cmd.args([
        "--mcp-events-callback-url",
        &format!("http://127.0.0.1:{port}"),
    ])
    .args(extra)
    .env("HOME", home)
    .env("BEYOND_AI_AGENT_MCP_EVENTS_COALESCE_MS", "300")
    .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
    let stderr = std::fs::File::create(home.join(format!("serve-{port}.stderr"))).unwrap();
    held.spawn(&cmd, stderr)
}

/// The next WebSocket frame matching `pred`, or a panic naming `what` after `timeout`.
pub async fn ws_next(
    ws: &mut super::TestWs,
    timeout: Duration,
    what: &str,
    pred: impl Fn(&Value) -> bool,
) -> Value {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, super::ws_next_frame(ws)).await {
            Ok(Some(f)) if pred(&f) => return f,
            Ok(Some(_)) => {}
            Ok(None) => panic!("socket closed waiting for {what}"),
            Err(_) => panic!("timed out waiting for {what}"),
        }
    }
}

/// Every WebSocket frame matching `pred` within `window`.
pub async fn ws_collect(
    ws: &mut super::TestWs,
    window: Duration,
    pred: impl Fn(&Value) -> bool,
) -> Vec<Value> {
    let deadline = Instant::now() + window;
    let mut out = Vec::new();
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return out;
        }
        if let Ok(Some(f)) = tokio::time::timeout(left, super::ws_next_frame(ws)).await
            && pred(&f)
        {
            out.push(f);
        }
    }
}

/// `mcp_events_list` over a WebSocket.
pub async fn ws_list(ws: &mut super::TestWs, id: &str) -> Value {
    super::ws_send(ws, json!({ "type": "mcp_events_list", "id": id })).await;
    ws_next(ws, Duration::from_secs(20), id, |f| {
        f["type"] == "response" && f["id"] == id
    })
    .await
}

/// Wait until the session's subscriptions are all `active`.
pub async fn ws_wait_active(ws: &mut super::TestWs) -> Value {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut n = 0;
    loop {
        n += 1;
        let l = ws_list(ws, &format!("wa{n}")).await;
        let subs = l["data"]["subscriptions"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !subs.is_empty() && subs.iter().all(|s| s["state"] == "active") {
            return subs[0].clone();
        }
        assert!(Instant::now() < deadline, "never active: {l:#}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// `list_daemon_sessions`, on a fresh connection: `id → live`.
pub async fn daemon_sessions(port: u16) -> std::collections::HashMap<String, bool> {
    let mut ws = super::ws_connect(port, None).await;
    super::ws_send(
        &mut ws,
        json!({ "type": "list_daemon_sessions", "id": "lds" }),
    )
    .await;
    let r = ws_next(
        &mut ws,
        Duration::from_secs(20),
        "list_daemon_sessions",
        |f| f["type"] == "response" && f["command"] == "list_daemon_sessions",
    )
    .await;
    r["data"]["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| (s["id"].as_str().unwrap_or("").to_owned(), s["live"] == true))
        .collect()
}

/// SIGTERM a child and wait (bounded) for it to exit.
pub fn sigterm_and_wait(child: &mut ChildGuard) {
    Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while child.try_wait().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "serve did not exit after SIGTERM"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// One raw HTTP request to the daemon's listener; returns the status code (0 if no answer came
/// within `timeout`).
pub fn raw_request(port: u16, head: &str, body: &[u8], timeout: Duration) -> u16 {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(timeout)).unwrap();
    s.write_all(head.as_bytes()).unwrap();
    let _ = s.write_all(body);
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        match s.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
        }
    }
    String::from_utf8_lossy(&buf)
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0)
}

/// How many agent runs `needle` *triggered*: model requests (that advertise tools — not the title
/// call) whose **last** user turn mentions it. Later runs carry it in their history; they don't
/// count.
pub fn runs_for_event(bodies: &std::sync::Mutex<Vec<String>>, needle: &str) -> usize {
    bodies
        .lock()
        .unwrap()
        .iter()
        .filter(|b| b.contains("\"tools\":["))
        .filter(|b| {
            let body = b.split_once("\r\n\r\n").map(|(_, body)| body).unwrap_or(b);
            let Ok(v) = serde_json::from_str::<Value>(body) else {
                return false;
            };
            v["messages"]
                .as_array()
                .and_then(|m| m.iter().rev().find(|m| m["role"] == "user"))
                .is_some_and(|m| m.to_string().contains(needle))
        })
        .count()
}

/// The undelivered events a session's events state holds on disk: its pending log
/// (`<session>.mcp-events.log`, beside the snapshot at `state_json`) replayed — `add` records
/// minus `done` ones, oldest first. A torn line is skipped, as the agent itself does.
pub fn pending_on_disk(state_json: &Path) -> Vec<Value> {
    let log = std::fs::read(state_json.with_extension("log")).unwrap_or_default();
    let mut live: std::collections::BTreeMap<u64, Value> = std::collections::BTreeMap::new();
    for line in log.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        let Ok(op) = serde_json::from_slice::<Value>(line) else {
            continue;
        };
        if let Some(add) = op.get("add") {
            live.insert(add["seq"].as_u64().unwrap_or(0), add.clone());
        } else if let Some(done) = op.get("done").and_then(Value::as_array) {
            for s in done {
                live.remove(&s.as_u64().unwrap_or(0));
            }
        }
    }
    live.into_values().collect()
}
