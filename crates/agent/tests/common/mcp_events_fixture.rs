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
