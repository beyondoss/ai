//! Shared harness for the SEP-2663 task e2e files (`mcp_tasks_durable.rs`, `mcp_tasks_resume.rs`,
//! `mcp_tasks_daemon.rs`): the strict `mcp_fixture_tasks_server` as a stdio server the agent spawns
//! or a standalone HTTP server the test owns (so it, and its tasks, outlive a killed `serve`), its
//! wire log, and deadline-bounded frame readers.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::common::{self, ChildGuard, SpawnGuarded, serve_cmd};

pub const FIXTURE: &str = env!("CARGO_BIN_EXE_mcp_fixture_tasks_server");

pub struct Env {
    pub dir: tempfile::TempDir,
    pub home: PathBuf,
    pub log: PathBuf,
    pub gate: PathBuf,
    pub state: PathBuf,
}

impl Env {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(home.join(".claude")).unwrap();
        let log = dir.path().join("wire.jsonl");
        let gate = dir.path().join("gate");
        let state = dir.path().join("fixture-state");
        Self {
            dir,
            home,
            log,
            gate,
            state,
        }
    }

    pub fn settings(&self, servers: Value) {
        std::fs::write(
            self.home.join(".claude/settings.json"),
            json!({ "mcp_servers": servers }).to_string(),
        )
        .unwrap();
    }

    /// The fixture as a stdio server the agent spawns, with extra env on top of the log and gate.
    pub fn stdio_with(&self, extra_env: Value) -> Value {
        let mut env = json!({
            "MCP_TASKS_FIXTURE_LOG": self.log.to_string_lossy(),
            "MCP_TASKS_FIXTURE_GATE": self.gate.to_string_lossy(),
        });
        for (k, v) in extra_env.as_object().unwrap() {
            env[k] = v.clone();
        }
        json!({ "name": "t", "transport": "stdio", "command": FIXTURE, "args": [], "env": env })
    }

    pub fn stdio(&self) {
        self.settings(json!([self.stdio_with(json!({}))]));
    }

    /// Start the fixture as a standalone HTTP server, on a port it picks; returns it and its settings
    /// entry. Gated tasks survive a restart (`MCP_TASKS_FIXTURE_KEEP`); a restarted server keeps its
    /// URL behind a [`Relay`].
    pub fn http_server(&self) -> (ChildGuard, Value) {
        let port_file = self.dir.path().join("port");
        let _ = std::fs::remove_file(&port_file);
        let child = Command::new(FIXTURE)
            .env("MCP_TASKS_FIXTURE_HTTP_PORT_FILE", &port_file)
            .env("MCP_TASKS_FIXTURE_LOG", &self.log)
            .env("MCP_TASKS_FIXTURE_GATE", &self.gate)
            .env("MCP_TASKS_FIXTURE_KEEP", self.dir.path().join("kept"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn_guarded();
        let deadline = Instant::now() + Duration::from_secs(20);
        let port = loop {
            if let Ok(p) = std::fs::read_to_string(&port_file) {
                break p;
            }
            assert!(Instant::now() < deadline, "fixture never wrote its port");
            std::thread::sleep(Duration::from_millis(20));
        };
        let entry = json!({
            "name": "t",
            "transport": "http",
            "url": format!("http://127.0.0.1:{}/mcp", port.trim()),
            "headers": {},
        });
        (child, entry)
    }

    /// The fixture as a standalone HTTP server, configured as the only MCP server.
    pub fn http(&self) -> ChildGuard {
        let (child, entry) = self.http_server();
        self.settings(json!([entry]));
        child
    }

    pub fn session_file(&self) -> String {
        self.dir
            .path()
            .join("s.jsonl")
            .to_string_lossy()
            .into_owned()
    }

    pub fn session_text(&self) -> String {
        std::fs::read_to_string(self.session_file()).unwrap_or_default()
    }

    pub fn serve(&self, base: &str) -> ChildGuard {
        let mut cmd = serve_cmd(common::BIN, base, &self.session_file());
        cmd.env("HOME", &self.home)
            .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0");
        cmd.spawn_guarded()
    }

    pub fn wire(&self) -> Vec<Value> {
        std::fs::read_to_string(&self.log)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    pub fn methods(&self, method: &str) -> Vec<Value> {
        self.wire()
            .into_iter()
            .filter(|m| m["method"] == method)
            .collect()
    }
}

pub fn send(stdin: &mut impl Write, v: Value) {
    writeln!(stdin, "{v}").unwrap();
    stdin.flush().unwrap();
}

pub fn prompt(stdin: &mut impl Write, message: &str) {
    send(stdin, json!({ "type": "prompt", "message": message }));
}

/// Read frames until one satisfies `pred`, failing at `deadline` or on any frame `forbid` flags;
/// returns everything read, that frame last.
pub fn read_until_or_fail(
    reader: &mut impl BufRead,
    what: &str,
    timeout: Duration,
    forbid: impl Fn(&Value) -> bool,
    pred: impl Fn(&Value) -> bool,
) -> Vec<Value> {
    // A reader thread would be needed to time out a blocked read; the fixtures here always produce
    // frames (progress, polls) or end, so checking between frames bounds every test in practice.
    let deadline = Instant::now() + timeout;
    let mut frames = Vec::new();
    let mut line = String::new();
    loop {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what}: {frames:?}"
        );
        line.clear();
        assert!(
            reader.read_line(&mut line).unwrap() > 0,
            "stream ended waiting for {what}: {frames:?}"
        );
        let Ok(frame) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        assert!(
            !forbid(&frame),
            "forbidden frame while waiting for {what}: {frame}"
        );
        let done = pred(&frame);
        frames.push(frame);
        if done {
            return frames;
        }
    }
}

pub fn read_until(
    reader: &mut impl BufRead,
    what: &str,
    pred: impl Fn(&Value) -> bool,
) -> Vec<Value> {
    read_until_or_fail(reader, what, Duration::from_secs(30), |_| false, pred)
}

pub fn read_until_prompt_done(reader: &mut impl BufRead) -> Vec<Value> {
    read_until(reader, "the prompt response", |f| {
        f["type"] == "response" && f["command"] == "prompt"
    })
}

pub fn wait_for(what: &str, cond: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !cond() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

pub fn is_event(frame: &Value, kind: &str, id: &str) -> bool {
    frame["type"] == "event" && frame["event"]["kind"] == kind && frame["event"]["id"] == id
}

pub fn tool_end<'a>(frames: &'a [Value], id: &str) -> &'a Value {
    frames
        .iter()
        .find(|f| is_event(f, "tool_end", id))
        .map(|f| &f["event"])
        .unwrap_or_else(|| panic!("no tool_end for {id}: {frames:?}"))
}

/// The roles of a recorded model request's messages, in order.
pub fn roles(request: &str) -> Vec<String> {
    common::body_json(request)["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap().to_owned())
        .collect()
}

/// Assert a request alternates user/assistant, starting with user (what every wire requires).
pub fn assert_alternates(request: &str) {
    let roles = roles(request);
    for (i, pair) in roles.windows(2).enumerate() {
        assert_ne!(pair[0], pair[1], "roles must alternate (at {i}): {roles:?}");
    }
    assert_eq!(roles.first().map(String::as_str), Some("user"), "{roles:?}");
}

/// The `tool_result` sent for `tool_use_id` in a recorded model request.
pub fn tool_result_sent(request: &str, tool_use_id: &str) -> Value {
    let body = common::body_json(request);
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["content"].as_array().cloned().unwrap_or_default())
        .find(|b| b["type"] == "tool_result" && b["tool_use_id"] == tool_use_id)
        .unwrap_or_else(|| panic!("no tool_result for {tool_use_id} in {body}"))
}

/// A URL that outlives the HTTP fixture behind it: a loopback listener the test binds and holds for
/// its whole life, relaying each connection to whichever fixture is up, and closing it at once while
/// none is. A server "restarted on the same URL" without releasing a port for anything else to take.
pub struct Relay {
    pub port: u16,
    backend: std::sync::Arc<std::sync::Mutex<Option<u16>>>,
}

impl Relay {
    pub fn start() -> Self {
        let listener = beyond_ai_test_support::ports::listener();
        let port = listener.local_addr().unwrap().port();
        let backend = std::sync::Arc::new(std::sync::Mutex::new(None::<u16>));
        let routes = backend.clone();
        std::thread::spawn(move || {
            for client in listener.incoming() {
                let Ok(client) = client else { continue };
                let target = *routes.lock().unwrap();
                let Some(server) =
                    target.and_then(|p| std::net::TcpStream::connect(("127.0.0.1", p)).ok())
                else {
                    continue; // nothing up: the connection closes as it drops
                };
                for (mut from, mut to) in [
                    (client.try_clone().unwrap(), server.try_clone().unwrap()),
                    (server, client),
                ] {
                    std::thread::spawn(move || {
                        let _ = std::io::copy(&mut from, &mut to);
                        let _ = to.shutdown(std::net::Shutdown::Write);
                    });
                }
            }
        });
        Self { port, backend }
    }

    /// Relay to the fixture `entry` (from [`Env::http_server`]) from now on, or to nothing.
    pub fn route_to(&self, entry: Option<&Value>) {
        let port = entry.map(|e| {
            let url = e["url"].as_str().unwrap();
            let port = url.rsplit(':').next().unwrap();
            port.trim_end_matches("/mcp").parse::<u16>().unwrap()
        });
        *self.backend.lock().unwrap() = port;
    }

    /// `entry` with its URL pointed at the relay.
    pub fn entry_for(&self, entry: &Value) -> Value {
        let mut entry = entry.clone();
        entry["url"] = json!(format!("http://127.0.0.1:{}/mcp", self.port));
        entry
    }
}
