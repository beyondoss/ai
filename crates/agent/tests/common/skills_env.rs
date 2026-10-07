//! Shared harness for the MCP Skills (SEP-2640) suites (`tests/mcp_skills*.rs`): an isolated `$HOME`
//! with `mcp_skills_fixture_server` configured as the global `docs` server, and readers for what the
//! mock model was sent.

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::{ChildGuard, SpawnGuarded, body_json, run_cmd, serve_cmd};

pub const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");
pub const FIXTURE: &str = env!("CARGO_BIN_EXE_mcp_skills_fixture_server");
pub const SKILL_TOOL: &str = "mcp__docs__skill__read";

/// One isolated `$HOME` with the skills fixture configured as the global `docs` server.
pub struct Env {
    pub dir: tempfile::TempDir,
    pub home: PathBuf,
    pub cwd: PathBuf,
    pub log: PathBuf,
}

/// What a `run` produced.
pub struct RunOutput {
    pub ok: bool,
    pub stdout: String,
    pub stderr: String,
    pub bodies: Vec<String>,
}

impl Env {
    pub fn new(extra_env: Value) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let cwd = dir.path().join("work");
        std::fs::create_dir_all(home.join(".claude")).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        let log = dir.path().join("fixture.log");
        let env = Self {
            dir,
            home,
            cwd,
            log,
        };
        env.configure(extra_env);
        env
    }

    /// (Re)write the `docs` server's config, with `extra_env` on top of the request log.
    pub fn configure(&self, extra_env: Value) {
        let mut server_env = json!({ "MCP_SKILLS_FIXTURE_LOG": self.log });
        for (k, v) in extra_env.as_object().unwrap() {
            server_env[k] = v.clone();
        }
        std::fs::write(
            self.home.join(".claude/settings.json"),
            serde_json::to_string_pretty(&json!({ "mcp_servers": [{
                "name": "docs",
                "transport": "stdio",
                "command": FIXTURE,
                "args": [],
                "env": server_env,
            }] }))
            .unwrap(),
        )
        .unwrap();
    }

    /// What reached the fixture, one `<method> <uri>` per line (`start -` per process start).
    pub fn log(&self) -> Vec<String> {
        std::fs::read_to_string(&self.log)
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// The log lines starting with `prefix`.
    pub fn log_of(&self, prefix: &str) -> Vec<String> {
        self.log()
            .into_iter()
            .filter(|l| l.starts_with(prefix))
            .collect()
    }

    pub fn clear_log(&self) {
        let _ = std::fs::remove_file(&self.log);
    }

    /// Run the real `run` binary with every MCP skill approved in advance
    /// (`--approve-mcp-skills`) — the suites that are not about approval; return the model request
    /// bodies.
    pub fn run(&self, task: &str, responses: Vec<String>) -> Vec<String> {
        let out = self.run_with(task, responses, &["--approve-mcp-skills"]);
        assert!(out.ok, "run failed: {}", out.stderr);
        out.bodies
    }

    /// Run the real `run` binary with `extra` flags, persisting nothing.
    pub fn run_with(&self, task: &str, responses: Vec<String>, extra: &[&str]) -> RunOutput {
        let mut args = vec!["--no-session-persistence"];
        args.extend_from_slice(extra);
        self.run_raw(task, responses, &args)
    }

    /// Run the real `run` binary with exactly `extra` beyond the model flags (so a test can persist
    /// and resume a session).
    pub fn run_raw(&self, task: &str, responses: Vec<String>, extra: &[&str]) -> RunOutput {
        let (base, bodies) = super::spawn_model_server(responses);
        let output = run_cmd(BIN)
            .env("HOME", &self.home)
            .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
            .args([
                "run",
                task,
                "--gateway-url",
                &base,
                "--key",
                "bai_v1.test",
                "--model",
                "claude-test",
                "--max-steps",
                "8",
            ])
            .args(extra)
            .current_dir(&self.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn_guarded()
            .wait_with_output()
            .unwrap();
        RunOutput {
            ok: output.status.success(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            bodies: bodies.lock().unwrap().clone(),
        }
    }

    /// Start the real `serve` binary (stdin/stdout JSON lines) against `base`, on this env's
    /// session file.
    pub fn serve(&self, base: &str, extra: &[&str]) -> Serve {
        let session_file = self.dir.path().join("s.jsonl");
        self.serve_on(base, &session_file, extra)
    }

    /// [`Self::serve`] on a given session file (a restart reopens the same one).
    pub fn serve_on(&self, base: &str, session_file: &std::path::Path, extra: &[&str]) -> Serve {
        let mut cmd = serve_cmd(BIN, base, session_file.to_str().unwrap());
        cmd.env("HOME", &self.home)
            .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
            .args(extra)
            .current_dir(&self.cwd);
        Serve::spawn(&mut cmd)
    }

    /// Start `serve` in repo mode (`--session-dir`), where sessions can be switched, forked and
    /// cloned.
    pub fn serve_dir(&self, base: &str, extra: &[&str]) -> Serve {
        let sessions = self.dir.path().join("sessions");
        let mut cmd = super::serve_dir_cmd(BIN, base, sessions.to_str().unwrap());
        cmd.env("HOME", &self.home)
            .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
            .args(extra)
            .current_dir(&self.cwd);
        Serve::spawn(&mut cmd)
    }

    /// Configure a second server, `other`, beside `docs` — the same fixture under another label.
    pub fn configure_two(&self) {
        std::fs::write(
            self.home.join(".claude/settings.json"),
            serde_json::to_string_pretty(&json!({ "mcp_servers": [
                { "name": "docs", "transport": "stdio", "command": FIXTURE, "args": [],
                  "env": { "MCP_SKILLS_FIXTURE_LOG": self.log } },
                { "name": "other", "transport": "stdio", "command": FIXTURE, "args": [],
                  "env": { "MCP_SKILLS_FIXTURE_LOG": self.dir.path().join("other.log") } },
            ] }))
            .unwrap(),
        )
        .unwrap();
    }
}

/// The recorded raw model requests of a mock model server.
pub type Bodies = std::sync::Arc<std::sync::Mutex<Vec<String>>>;

/// How long any one wait on `serve` may take before a test fails — so a regression fails fast with
/// what was seen, instead of hanging the suite.
pub const DEADLINE: Duration = super::FRAME_DEADLINE;

/// A running `serve` over stdin/stdout. Its stdout is read on a thread of its own, so every wait has
/// a deadline.
pub struct Serve {
    pub child: ChildGuard,
    pub stdin: ChildStdin,
    frames: super::Frames,
}

impl Serve {
    pub fn spawn(cmd: &mut Command) -> Self {
        let mut child = cmd.spawn_guarded();
        let stdin = child.stdin.take().unwrap();
        let frames = super::serve_frames(child.stdout.take().unwrap());
        Self {
            child,
            stdin,
            frames,
        }
    }

    pub fn send(&mut self, cmd: Value) {
        writeln!(self.stdin, "{cmd}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// Send `cmd` and read to its response; return every frame seen.
    pub fn call(&mut self, cmd: Value, command: &str) -> Vec<Value> {
        self.send(cmd);
        let command = command.to_string();
        self.read_until(|f| is_response(f, &command))
    }

    /// Read frames until one satisfies `stop` (returned last). Panics, with every frame seen, if the
    /// stream ends or [`DEADLINE`] passes first.
    pub fn read_until(&mut self, stop: impl FnMut(&Value) -> bool) -> Vec<Value> {
        self.read_until_within(DEADLINE, stop)
    }

    /// [`Self::read_until`] with its own deadline.
    pub fn read_until_within(
        &mut self,
        limit: Duration,
        mut stop: impl FnMut(&Value) -> bool,
    ) -> Vec<Value> {
        let deadline = Instant::now() + limit;
        let mut frames = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.frames.next_frame(left) {
                Ok(v) => {
                    let done = stop(&v);
                    frames.push(v);
                    if done {
                        return frames;
                    }
                }
                Err(super::NoFrame::TimedOut) => {
                    panic!("serve did not answer within {limit:?}; saw: {frames:#?}")
                }
                Err(super::NoFrame::Closed) => {
                    panic!("serve's stdout closed; saw: {frames:#?}")
                }
            }
        }
    }

    /// Answer an `approval_request`.
    pub fn approve(&mut self, request: &Value, decision: &str, scope: &str) {
        self.send(json!({
            "type": "approve",
            "request_id": request["request_id"],
            "decision": decision,
            "scope": scope,
        }));
    }

    pub fn finish(mut self) {
        drop(self.stdin);
        let _ = self.child.wait();
    }
}

/// Is `frame` the terminal response to `command`?
pub fn is_response(frame: &Value, command: &str) -> bool {
    frame["type"] == "response" && frame["command"] == command
}

/// Every text the model saw in `v` (a bare string, or the `text`/`content` strings of its blocks),
/// newline-joined — so assertions read the text rather than its JSON escaping.
pub fn all_text(v: &Value) -> String {
    fn walk(v: &Value, out: &mut String) {
        match v {
            Value::String(s) => {
                out.push_str(s);
                out.push('\n');
            }
            Value::Array(items) => items.iter().for_each(|i| walk(i, out)),
            Value::Object(map) => map
                .iter()
                .filter(|(k, _)| matches!(k.as_str(), "text" | "content"))
                .for_each(|(_, i)| walk(i, out)),
            _ => {}
        }
    }
    let mut out = String::new();
    walk(v, &mut out);
    out
}

pub fn system_text(raw: &str) -> String {
    all_text(&body_json(raw)["system"])
}

/// The text of the request's last message (a tool result, or the user's turn).
pub fn last_message_text(raw: &str) -> String {
    let body = body_json(raw);
    all_text(body["messages"].as_array().unwrap().last().unwrap())
}

/// The model's turn calling the skill tool on `uri`.
pub fn read_skill(id: &str, uri: &str) -> String {
    super::turn_tool_use(id, SKILL_TOOL, &json!({ "uri": uri }).to_string())
}

/// The model's turn running `command` in `bash`.
pub fn bash(id: &str, command: &str) -> String {
    super::turn_tool_use(id, "bash", &json!({ "command": command }).to_string())
}

/// Bodies sent for real turns, without the session-title request a first `serve` prompt triggers.
pub fn turn_bodies(bodies: &[String]) -> Vec<String> {
    bodies
        .iter()
        .filter(|b| !system_text(b).starts_with("You write short titles"))
        .cloned()
        .collect()
}
