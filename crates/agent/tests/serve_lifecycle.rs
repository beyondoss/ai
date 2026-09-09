//! `serve`/`run` e2e: host-emitted run lifecycle against a real binary and a mock HTTP collector.
//!
//! The unit-level latch (one started, one terminal, progress after terminal dropped, latest-wins)
//! lives in `src/lifecycle.rs`. This file is the wiring proof: a `serve` prompt and a `run`
//! invocation POST the same JSON the collector would see in production, without holding a session
//! connection, and an unconfigured process does not POST at all.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use common::{
    SpawnGuarded, read_until_response, run_cmd, serve_cmd, spawn_model_server, turn_refusal,
    turn_text, turn_tool_use,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");
const WAIT: Duration = Duration::from_secs(5);

struct Collector {
    url: String,
    posts: Arc<Mutex<Vec<(String, String)>>>,
}

impl Collector {
    fn spawn() -> Self {
        Self::spawn_with_delay(Duration::ZERO)
    }

    fn spawn_with_delay(delay: Duration) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let posts = Arc::new(Mutex::new(Vec::new()));
        let recorder = posts.clone();
        thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(stream) = conn else { break };
                let recorder = recorder.clone();
                thread::spawn(move || handle_post(stream, delay, recorder));
            }
        });
        Self {
            url: format!("http://{addr}/lifecycle"),
            posts,
        }
    }

    fn bodies(&self) -> Vec<Value> {
        self.posts
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(_, b)| serde_json::from_str(b).ok())
            .collect()
    }

    fn wait_until(&self, pred: impl Fn(&[Value]) -> bool) -> Vec<Value> {
        let start = Instant::now();
        loop {
            let bodies = self.bodies();
            if pred(&bodies) {
                return bodies;
            }
            if start.elapsed() > WAIT {
                panic!("lifecycle collector timed out after {WAIT:?}: {bodies:#?}");
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn headers_joined(&self) -> String {
        self.posts
            .lock()
            .unwrap()
            .iter()
            .map(|(h, _)| h.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn handle_post(mut stream: TcpStream, delay: Duration, posts: Arc<Mutex<Vec<(String, String)>>>) {
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
            let need = pos + 4 + len;
            while buf.len() < need {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            break;
        }
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(buf.len());
    let headers = String::from_utf8_lossy(&buf[..split]).into_owned();
    let body = String::from_utf8_lossy(&buf[split.min(buf.len())..]).into_owned();
    posts.lock().unwrap().push((headers, body));
    if !delay.is_zero() {
        thread::sleep(delay);
    }
    let _ =
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
    let _ = stream.flush();
}

fn of_type<'a>(bodies: &'a [Value], kind: &str) -> Vec<&'a Value> {
    bodies.iter().filter(|v| v["type"] == kind).collect()
}

fn todo_item(content: &str, status: &str) -> Value {
    json!({ "content": content, "activeForm": format!("{content}ing"), "status": status })
}

fn send(stdin: &mut impl Write, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

fn serve_life(base: &str, session_file: &str, url: &str) -> std::process::Command {
    let mut cmd = serve_cmd(BIN, base, session_file);
    cmd.args(["--lifecycle-url", url]);
    cmd
}

#[test]
fn serve_prompt_posts_started_then_succeeded_with_command_id_and_summary() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let collector = Collector::spawn();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hello from the run")]);

    let mut child = serve_life(&base, &session_file, &collector.url)
        .args(["--lifecycle-header", "X-Tenant: acme"])
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "say hi" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let bodies = collector
        .wait_until(|b| of_type(b, "started").len() == 1 && of_type(b, "succeeded").len() == 1);
    let started = of_type(&bodies, "started")[0];
    let done = of_type(&bodies, "succeeded")[0];
    assert_eq!(started["run_id"], done["run_id"], "{bodies:#?}");
    assert_eq!(started["command_id"], "p1");
    assert_eq!(done["command_id"], "p1");
    assert!(
        started["session_id"]
            .as_str()
            .is_some_and(|s| !s.is_empty())
    );
    assert_eq!(started["session_id"], done["session_id"]);
    assert_eq!(started["model"], "claude-test");
    assert_eq!(done["summary"], "hello from the run");
    assert!(done.get("refused").is_none() || done["refused"] == false);
    let headers = collector.headers_joined().to_ascii_lowercase();
    assert!(
        headers.contains("x-tenant: acme"),
        "auth header must reach the collector: {headers}"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_abort_posts_started_then_aborted() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let collector = Collector::spawn();
    let turn1 = turn_tool_use(
        "toolu_b",
        "bash",
        &json!({ "command": "sleep 30" }).to_string(),
    );
    let (base, _bodies) = spawn_model_server(vec![turn1]);

    let mut child = serve_life(&base, &session_file, &collector.url).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "abort-me", "message": "run a long sleep" }),
    );
    thread::sleep(Duration::from_millis(500));
    send(&mut stdin, json!({ "type": "abort", "id": "a1" }));

    let start = Instant::now();
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        start.elapsed() < Duration::from_secs(15),
        "abort must return promptly, took {:?}",
        start.elapsed()
    );
    assert_eq!(frames.last().unwrap()["success"], false, "{frames:#?}");

    let bodies = collector
        .wait_until(|b| of_type(b, "started").len() == 1 && of_type(b, "aborted").len() == 1);
    assert_eq!(of_type(&bodies, "succeeded").len(), 0, "{bodies:#?}");
    assert_eq!(of_type(&bodies, "failed").len(), 0, "{bodies:#?}");
    let started = of_type(&bodies, "started")[0];
    let aborted = of_type(&bodies, "aborted")[0];
    assert_eq!(started["run_id"], aborted["run_id"]);
    assert_eq!(started["command_id"], "abort-me");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_todo_and_tool_progress_carries_tools_and_todos() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let collector = Collector::spawn();
    let todos = json!([
        todo_item("Wire the retry loop", "in_progress"),
        todo_item("Add tests", "pending"),
    ]);
    let (base, _bodies) = spawn_model_server(vec![
        turn_tool_use("t-todo", "todo", &json!({ "todos": todos }).to_string()),
        turn_tool_use(
            "t-bash",
            "bash",
            &json!({ "command": "sleep 2" }).to_string(),
        ),
        turn_text("all done"),
    ]);

    let mut child = serve_life(&base, &session_file, &collector.url).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "plan", "message": "do the work" }),
    );
    let bodies = collector.wait_until(|b| {
        of_type(b, "progress").iter().any(|p| {
            let tools = p["tools"].as_array().cloned().unwrap_or_default();
            let has_bash = tools.iter().any(|t| t == "bash");
            let has_todos = p["todos"].as_array().is_some_and(|a| !a.is_empty());
            has_bash && has_todos
        })
    });
    let progress = of_type(&bodies, "progress");
    assert!(
        progress.iter().any(|p| {
            p["todos"]
                .as_array()
                .is_some_and(|a| a.iter().any(|t| t["content"] == "Wire the retry loop"))
        }),
        "todos must ride on progress: {bodies:#?}"
    );
    // Tool arguments must never appear — names only.
    let dumped = serde_json::to_string(&bodies).unwrap();
    assert!(
        !dumped.contains("sleep 2"),
        "tool arguments leaked onto the wire: {dumped}"
    );

    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    collector.wait_until(|b| of_type(b, "succeeded").len() == 1);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_two_prompts_mint_two_run_ids_each_with_one_terminal() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let collector = Collector::spawn();
    let (base, _bodies) = spawn_model_server(vec![turn_text("first"), turn_text("second")]);

    let mut child = serve_life(&base, &session_file, &collector.url).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p1", "message": "one" }),
    );
    read_until_response(&mut stdout, "prompt");
    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "p2", "message": "two" }),
    );
    read_until_response(&mut stdout, "prompt");

    let bodies = collector
        .wait_until(|b| of_type(b, "started").len() == 2 && of_type(b, "succeeded").len() == 2);
    let started = of_type(&bodies, "started");
    let done = of_type(&bodies, "succeeded");
    let id_a = started[0]["run_id"].as_str().unwrap();
    let id_b = started[1]["run_id"].as_str().unwrap();
    assert_ne!(id_a, id_b, "each prompt is its own run: {bodies:#?}");
    assert_eq!(started[0]["command_id"], "p1");
    assert_eq!(started[1]["command_id"], "p2");
    let terminals: Vec<&str> = done.iter().map(|v| v["run_id"].as_str().unwrap()).collect();
    assert!(
        terminals.contains(&id_a) && terminals.contains(&id_b),
        "{bodies:#?}"
    );
    assert_eq!(of_type(&bodies, "started").len(), 2);
    assert_eq!(of_type(&bodies, "succeeded").len(), 2);

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_unconfigured_posts_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let collector = Collector::spawn();
    let (base, _bodies) = spawn_model_server(vec![turn_text("hello")]);

    let mut child = serve_cmd(BIN, &base, &session_file).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(&mut stdin, json!({ "type": "prompt", "message": "hi" }));
    read_until_response(&mut stdout, "prompt");
    thread::sleep(Duration::from_millis(400));
    assert!(
        collector.bodies().is_empty(),
        "unconfigured serve must not POST: {:#?}",
        collector.bodies()
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_lifecycle_url_from_env_without_the_flag() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let collector = Collector::spawn();
    let (base, _bodies) = spawn_model_server(vec![turn_text("from env")]);

    let mut child = serve_cmd(BIN, &base, &session_file)
        .env("AI_AGENT_LIFECYCLE_URL", &collector.url)
        .spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "env-1", "message": "hi" }),
    );
    read_until_response(&mut stdout, "prompt");
    let bodies = collector.wait_until(|b| of_type(b, "succeeded").len() == 1);
    assert_eq!(of_type(&bodies, "started")[0]["command_id"], "env-1");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn serve_fails_fast_on_a_malformed_lifecycle_url() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _bodies) = spawn_model_server(vec![]);
    let mut cmd = serve_cmd(BIN, &base, &session_file);
    cmd.args(["--lifecycle-url", "file:///etc/passwd"]);
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn_guarded();
    drop(child.stdin.take());
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    let status = child.wait().unwrap();
    assert!(!status.success(), "stderr: {stderr}");
    assert_eq!(status.code(), Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains("lifecycle URL") || stderr.contains("http or https"),
        "stderr: {stderr}"
    );
}

#[test]
fn serve_refusal_is_succeeded_with_refused() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let collector = Collector::spawn();
    let (base, _bodies) = spawn_model_server(vec![turn_refusal("I can't help with that.")]);

    let mut child = serve_life(&base, &session_file, &collector.url).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "r1", "message": "no" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    let bodies = collector.wait_until(|b| of_type(b, "succeeded").len() == 1);
    let done = of_type(&bodies, "succeeded")[0];
    assert_eq!(done["refused"], true, "{bodies:#?}");
    assert_eq!(of_type(&bodies, "failed").len(), 0, "{bodies:#?}");

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn a_slow_lifecycle_consumer_does_not_delay_the_prompt_response() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let collector = Collector::spawn_with_delay(Duration::from_secs(2));
    // Stall the *model* as well so we can send the prompt and then measure only the control-plane
    // round trip — but a stalled lifecycle POST must not hold the response. Instant model, slow
    // collector: if `emit` awaited the network, the prompt response would take ~2s.
    let (base, _bodies) = spawn_model_server(vec![turn_text("quick")]);

    let mut child = serve_life(&base, &session_file, &collector.url).spawn_guarded();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "id": "fast", "message": "hi" }),
    );
    let start = Instant::now();
    let frames = read_until_response(&mut stdout, "prompt");
    let elapsed = start.elapsed();
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert!(
        elapsed < Duration::from_millis(1500),
        "prompt response waited on the lifecycle consumer ({elapsed:?})"
    );

    drop(stdin);
    child.wait().unwrap();
}

#[test]
fn run_posts_started_then_succeeded() {
    let dir = tempfile::tempdir().unwrap();
    let collector = Collector::spawn();
    let (base, _bodies) = spawn_model_server(vec![turn_text("all done")]);

    let output = run_cmd(BIN)
        .args([
            "run",
            "say hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
            "--lifecycle-url",
            &collector.url,
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = collector
        .wait_until(|b| of_type(b, "started").len() == 1 && of_type(b, "succeeded").len() == 1);
    let started = of_type(&bodies, "started")[0];
    let done = of_type(&bodies, "succeeded")[0];
    assert_eq!(started["run_id"], done["run_id"]);
    assert!(started.get("command_id").is_none() || started["command_id"].is_null());
    assert_eq!(done["summary"], "all done");
}

#[test]
fn run_text_mode_refusal_posts_failed_with_refused() {
    let dir = tempfile::tempdir().unwrap();
    let collector = Collector::spawn();
    let (base, _bodies) = spawn_model_server(vec![turn_refusal("I can't help with that.")]);

    let output = run_cmd(BIN)
        .args([
            "run",
            "do something the model refuses",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
            "--lifecycle-url",
            &collector.url,
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(!output.status.success(), "text-mode refusal must exit 1");

    let bodies = collector.wait_until(|b| of_type(b, "failed").len() == 1);
    let failed = of_type(&bodies, "failed")[0];
    assert_eq!(failed["refused"], true, "{bodies:#?}");
    assert_eq!(of_type(&bodies, "succeeded").len(), 0, "{bodies:#?}");
}

#[test]
fn run_json_mode_refusal_posts_succeeded_with_refused() {
    let dir = tempfile::tempdir().unwrap();
    let collector = Collector::spawn();
    let (base, _bodies) = spawn_model_server(vec![turn_refusal("I can't help with that.")]);

    let output = run_cmd(BIN)
        .args([
            "run",
            "do something the model refuses",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
            "--json",
            "--lifecycle-url",
            &collector.url,
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "json-mode refusal exits 0.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = collector.wait_until(|b| of_type(b, "succeeded").len() == 1);
    let done = of_type(&bodies, "succeeded")[0];
    assert_eq!(done["refused"], true, "{bodies:#?}");
    assert_eq!(of_type(&bodies, "failed").len(), 0, "{bodies:#?}");
}
