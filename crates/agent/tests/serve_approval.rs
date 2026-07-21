//! `serve` e2e: the interactive approval gate, driven through the real binary.
//!
//! The gate's whole value is what it does when things go *wrong*, so most of this file is the failure
//! modes: a static deny that must never reach a human, an abort while a question is outstanding, a
//! timeout, a detached client, a stale answer, and eight simultaneous questions that must not become
//! eight simultaneous prompts.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, Command, Stdio};
use std::time::Duration;

use common::{
    ChildGuard, ISOLATED_HOME, SpawnGuarded, free_port, read_until_response, spawn_model_server,
    turn_text, turn_tool_use, wait_for_port, ws_connect, ws_next_frame, ws_send,
};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

// ---- harness ---------------------------------------------------------------------------------------

/// A stdio `serve` child with an approval gate installed.
fn serve_approving(base: &str, session_file: &str, extra: &[&str]) -> ChildGuard {
    Command::new(BIN)
        .args([
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
        .args(extra)
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn_guarded()
}

fn send(stdin: &mut ChildStdin, cmd: Value) {
    writeln!(stdin, "{cmd}").unwrap();
    stdin.flush().unwrap();
}

/// Read frames until one of `type == ty` arrives. Returns everything seen, that frame included.
fn read_until_type(reader: &mut impl BufRead, ty: &str) -> Vec<Value> {
    let mut frames = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).unwrap() == 0 {
            panic!("stream closed before a `{ty}` frame arrived; saw: {frames:#?}");
        }
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        let done = v["type"] == ty;
        frames.push(v);
        if done {
            return frames;
        }
    }
}

fn last(frames: &[Value]) -> &Value {
    frames.last().expect("at least one frame")
}

fn approve(stdin: &mut ChildStdin, request_id: &str, decision: &str, scope: &str) {
    send(
        stdin,
        json!({ "type": "approve", "id": "a", "request_id": request_id, "decision": decision, "scope": scope }),
    );
}

fn kill(mut child: ChildGuard) {
    let _ = child.kill();
    let _ = child.wait();
}

fn write_turn(id: &str, path: &str) -> String {
    turn_tool_use(
        id,
        "write",
        &json!({ "path": path, "content": "written\n" }).to_string(),
    )
}

/// Every `tool_result` the parent's transcript recorded, as text.
fn tool_results(session_file: &str) -> Vec<String> {
    std::fs::read_to_string(session_file)
        .unwrap()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["type"] == "message")
        .flat_map(|v| {
            v["content"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
        })
        .filter(|b| b["type"] == "tool_result")
        .filter_map(|b| b["content"].as_str().map(str::to_string))
        .collect()
}

// ---- the happy path --------------------------------------------------------------------------------

#[test]
fn an_allowed_call_actually_runs() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let target = dir.path().join("out.txt");
    let (base, _b) = spawn_model_server(vec![
        write_turn("tu_1", target.to_str().unwrap()),
        turn_text("wrote it"),
    ]);

    let mut child = serve_approving(&base, &session_file, &["--approve", "writes"]);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "write the file" }),
    );
    let frames = read_until_type(&mut stdout, "approval_request");
    let req = last(&frames);
    assert_eq!(req["tool"], "write");
    assert_eq!(req["origin"], "main");
    assert_eq!(req["summary"]["content"], "written\n");
    assert!(req["scope_key"].as_str().unwrap().starts_with("path:/"));
    assert_eq!(
        req["options"],
        json!(["allow_once", "allow_session", "deny_once", "deny_session"])
    );
    assert!(
        !target.exists(),
        "the tool must not have run before the human answered"
    );

    approve(
        &mut stdin,
        req["request_id"].as_str().unwrap(),
        "allow",
        "once",
    );
    let frames = read_until_type(&mut stdout, "approval_resolved");
    assert_eq!(last(&frames)["decision"], "allow");
    assert_eq!(last(&frames)["scope"], "once");

    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(last(&frames)["success"], true);
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "written\n");
    kill(child);
}

#[test]
fn a_denied_call_does_not_run_and_the_model_is_told_why() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let target = dir.path().join("out.txt");
    let (base, _b) = spawn_model_server(vec![
        write_turn("tu_1", target.to_str().unwrap()),
        turn_text("understood, I won't"),
    ]);

    let mut child = serve_approving(&base, &session_file, &["--approve", "writes"]);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "write the file" }),
    );
    let req = last(&read_until_type(&mut stdout, "approval_request")).clone();
    approve(
        &mut stdin,
        req["request_id"].as_str().unwrap(),
        "deny",
        "once",
    );

    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(
        last(&frames)["success"],
        true,
        "a denial is not a run failure"
    );
    assert!(!target.exists(), "the tool must not have run");

    let results = tool_results(&session_file);
    assert!(
        results.iter().any(|r| r.contains("denied by the user")),
        "the model must be told: {results:?}"
    );
    kill(child);
}

#[test]
fn an_ungated_tool_is_never_asked_about() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let readable = dir.path().join("in.txt");
    std::fs::write(&readable, "SENTINEL\n").unwrap();

    let (base, _b) = spawn_model_server(vec![
        turn_tool_use(
            "tu_1",
            "read",
            &json!({ "path": readable.to_str().unwrap() }).to_string(),
        ),
        turn_text("read it"),
    ]);

    let mut child = serve_approving(&base, &session_file, &["--approve", "writes"]);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "read it" }),
    );
    let frames = read_until_response(&mut stdout, "prompt");
    assert_eq!(last(&frames)["success"], true);
    assert!(
        !frames.iter().any(|f| f["type"] == "approval_request"),
        "`read` is not a gated tool"
    );
    kill(child);
}

// ---- ordering: a static deny must never reach a human -----------------------------------------------

#[test]
fn a_static_deny_short_circuits_without_ever_asking() {
    // Asking someone to approve what the operator already forbade is both pointless and a way to
    // social-engineer past the deny-list.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _b) = spawn_model_server(vec![
        turn_tool_use("tu_1", "bash", &json!({ "command": "echo hi" }).to_string()),
        turn_text("blocked"),
    ]);

    let mut child = serve_approving(
        &base,
        &session_file,
        &["--approve", "all", "--deny-tool", "bash"],
    );
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(&mut stdin, json!({ "type": "prompt", "message": "run it" }));
    let frames = read_until_response(&mut stdout, "prompt");
    assert!(
        !frames.iter().any(|f| f["type"] == "approval_request"),
        "the human must never have been asked"
    );
    let results = tool_results(&session_file);
    assert!(
        results.iter().any(|r| r.contains("denied by policy")),
        "{results:?}"
    );
    kill(child);
}

// ---- remembering a decision --------------------------------------------------------------------------

/// Drive `n` prompts, each writing `paths[i]`, answering every approval request with `decision`/`scope`.
/// Returns how many times a question was asked.
fn count_questions(paths: &[&str], decision: &str, scope: &str, session_file: &str) -> usize {
    let mut turns = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        turns.push(write_turn(&format!("tu_{i}"), p));
        turns.push(turn_text("done"));
    }
    let (base, _b) = spawn_model_server(turns);
    let mut child = serve_approving(&base, session_file, &["--approve", "writes"]);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let mut asked = 0;
    for _ in paths {
        send(&mut stdin, json!({ "type": "prompt", "message": "write" }));
        // Read to the prompt's terminal response, answering any question along the way.
        let mut line = String::new();
        loop {
            line.clear();
            assert!(stdout.read_line(&mut line).unwrap() > 0, "stream closed");
            let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
                continue;
            };
            if v["type"] == "approval_request" {
                asked += 1;
                approve(
                    &mut stdin,
                    v["request_id"].as_str().unwrap(),
                    decision,
                    scope,
                );
            }
            if v["type"] == "response" && v["command"] == "prompt" {
                break;
            }
        }
    }
    kill(child);
    asked
}

#[test]
fn a_session_scoped_allow_is_remembered_but_a_once_scoped_one_is_not() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.txt").to_string_lossy().into_owned();
    let sf = dir.path().join("s1.jsonl").to_string_lossy().into_owned();
    assert_eq!(count_questions(&[&p, &p], "allow", "once", &sf), 2);

    let sf = dir.path().join("s2.jsonl").to_string_lossy().into_owned();
    assert_eq!(
        count_questions(&[&p, &p], "allow", "session", &sf),
        1,
        "the same path must not be asked about twice"
    );
}

#[test]
fn a_session_scoped_allow_does_not_authorize_a_different_path() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.txt").to_string_lossy().into_owned();
    let b = dir.path().join("b.txt").to_string_lossy().into_owned();
    let sf = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    assert_eq!(
        count_questions(&[&a, &b], "allow", "session", &sf),
        2,
        "a decision about `a.txt` says nothing about `b.txt`"
    );
}

#[test]
fn a_session_scoped_allow_survives_a_different_spelling_of_the_same_path() {
    // The scope key is the *resolved* path, so `./a.txt` cannot slip past a decision made about
    // `a.txt` — nor force a second question about the same file.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "seed").unwrap();
    let a = dir.path().join("a.txt").to_string_lossy().into_owned();
    let dotted = dir
        .path()
        .join("./sub/../a.txt")
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(dir.path().join("sub")).unwrap();
    let sf = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    assert_eq!(count_questions(&[&a, &dotted], "allow", "session", &sf), 1);
}

#[test]
fn a_session_scoped_deny_is_remembered_too() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("a.txt").to_string_lossy().into_owned();
    let sf = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    assert_eq!(count_questions(&[&p, &p], "deny", "session", &sf), 1);

    let results = tool_results(&sf);
    assert!(
        results.iter().any(|r| r.contains("earlier decision")),
        "the second call must be refused from memory: {results:?}"
    );
}

// ---- failing closed ---------------------------------------------------------------------------------

#[test]
fn an_abort_unblocks_a_pending_question_promptly() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let target = dir.path().join("out.txt");
    let (base, _b) = spawn_model_server(vec![write_turn("tu_1", target.to_str().unwrap())]);

    let mut child = serve_approving(
        &base,
        &session_file,
        // Wait forever: only the abort can free this.
        &["--approve", "writes", "--approval-timeout", "0"],
    );
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(&mut stdin, json!({ "type": "prompt", "message": "write" }));
    read_until_type(&mut stdout, "approval_request");

    send(&mut stdin, json!({ "type": "abort", "id": "x" }));
    let frames = read_until_type(&mut stdout, "approval_resolved");
    let resolved = last(&frames);
    assert_eq!(resolved["decision"], "deny");
    assert_eq!(resolved["reason"], "cancelled");
    assert!(!target.exists());
    kill(child);
}

#[test]
fn an_unanswered_question_times_out_and_denies() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let target = dir.path().join("out.txt");
    let (base, _b) = spawn_model_server(vec![
        write_turn("tu_1", target.to_str().unwrap()),
        turn_text("blocked"),
    ]);

    let mut child = serve_approving(
        &base,
        &session_file,
        &["--approve", "writes", "--approval-timeout", "1"],
    );
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(&mut stdin, json!({ "type": "prompt", "message": "write" }));
    read_until_type(&mut stdout, "approval_request");
    // Answer nothing.
    let frames = read_until_type(&mut stdout, "approval_resolved");
    assert_eq!(last(&frames)["decision"], "deny");
    assert_eq!(last(&frames)["reason"], "timed_out");

    read_until_response(&mut stdout, "prompt");
    assert!(!target.exists(), "a timed-out call must not run");
    let results = tool_results(&session_file);
    assert!(
        results.iter().any(|r| r.contains("timed out")),
        "{results:?}"
    );
    kill(child);
}

// ---- the `approve` command itself --------------------------------------------------------------------

#[test]
fn an_unknown_request_id_is_not_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _b) = spawn_model_server(vec![]);

    let mut child = serve_approving(&base, &session_file, &["--approve", "all"]);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    approve(&mut stdin, "no-such-request", "allow", "once");
    let frames = read_until_response(&mut stdout, "approve");
    let resp = last(&frames);
    assert_eq!(resp["success"], true, "a lost race is not an error");
    assert_eq!(resp["data"]["accepted"], false);
    kill(child);
}

#[test]
fn a_malformed_approve_is_rejected_and_a_session_without_a_gate_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let (base, _b) = spawn_model_server(vec![]);

    let mut child = serve_approving(&base, &session_file, &["--approve", "all"]);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "approve", "id": "a", "decision": "allow" }),
    );
    assert_eq!(
        last(&read_until_response(&mut stdout, "approve"))["success"],
        false
    );

    send(
        &mut stdin,
        json!({ "type": "approve", "id": "a", "request_id": "x", "decision": "maybe" }),
    );
    let resp = last(&read_until_response(&mut stdout, "approve")).clone();
    assert_eq!(resp["success"], false);
    assert!(resp["error"].as_str().unwrap().contains("allow"));
    kill(child);

    // And a session with no gate at all rejects the command outright rather than silently accepting it.
    let session_file = dir.path().join("s2.jsonl").to_string_lossy().into_owned();
    let (base, _b) = spawn_model_server(vec![]);
    let mut child = serve_approving(&base, &session_file, &[]);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    approve(&mut stdin, "x", "allow", "once");
    let resp = last(&read_until_response(&mut stdout, "approve")).clone();
    assert_eq!(resp["success"], false);
    assert!(resp["error"].as_str().unwrap().contains("--approve"));
    kill(child);
}

// ---- one question at a time --------------------------------------------------------------------------

/// One assistant turn dispatching two tool calls at once.
fn turn_two_writes(a: &str, b: &str) -> String {
    let mut events = vec![
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
    ];
    for (i, (id, path)) in [("tu_1", a), ("tu_2", b)].iter().enumerate() {
        let args = json!({ "path": path, "content": "written\n" }).to_string();
        events.push(json!({ "type": "content_block_start", "index": i, "content_block": { "type": "tool_use", "id": id, "name": "write", "input": {} } }));
        events.push(json!({ "type": "content_block_delta", "index": i, "delta": { "type": "input_json_delta", "partial_json": args } }));
        events.push(json!({ "type": "content_block_stop", "index": i }));
    }
    events.push(
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
    );
    events.push(json!({ "type": "message_stop" }));
    common::sse(&events)
}

#[test]
fn two_gated_calls_in_one_turn_are_asked_about_one_at_a_time() {
    // Eight concurrent `before_tool_call` hooks must not become eight simultaneous prompts. The gate
    // serializes questions; the approved calls still run concurrently afterward.
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl").to_string_lossy().into_owned();
    let a = dir.path().join("a.txt");
    let b = dir.path().join("b.txt");
    let (base, _bd) = spawn_model_server(vec![
        turn_two_writes(a.to_str().unwrap(), b.to_str().unwrap()),
        turn_text("wrote both"),
    ]);

    let mut child = serve_approving(&base, &session_file, &["--approve", "writes"]);
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    send(
        &mut stdin,
        json!({ "type": "prompt", "message": "write both" }),
    );

    let mut order: Vec<String> = Vec::new();
    let mut answered = 0;
    let mut line = String::new();
    loop {
        line.clear();
        assert!(stdout.read_line(&mut line).unwrap() > 0, "stream closed");
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        match v["type"].as_str() {
            Some("approval_request") => {
                order.push("request".into());
                approve(
                    &mut stdin,
                    v["request_id"].as_str().unwrap(),
                    "allow",
                    "once",
                );
                answered += 1;
            }
            Some("approval_resolved") => order.push("resolved".into()),
            Some("response") if v["command"] == "prompt" => break,
            _ => {}
        }
    }

    assert_eq!(answered, 2, "both calls must have been gated");
    // The second question can only be broadcast after the first one released the width-1 lock, which
    // happens after its `approval_resolved` goes out.
    assert_eq!(
        order,
        vec!["request", "resolved", "request", "resolved"],
        "questions must not overlap"
    );
    assert!(
        a.exists() && b.exists(),
        "both approved calls must have run"
    );
    kill(child);
}

// ---- multi-attach, over the real WebSocket transport ------------------------------------------------

fn serve_ws_approving(base: &str, session_dir: &str, port: u16, extra: &[&str]) -> ChildGuard {
    Command::new(BIN)
        .args([
            "serve",
            "--listen",
            &format!("127.0.0.1:{port}"),
            "--gateway-url",
            base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-dir",
            session_dir,
        ])
        .args(extra)
        .env("HOME", ISOLATED_HOME)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn_guarded()
}

/// A WebSocket client that *buffers* frames it wasn't looking for.
///
/// Frames a session broadcasts race each other — an `approval_resolved` and the `approve` command's own
/// `response` are produced by two different tasks — so a reader that discards whatever it isn't currently
/// waiting for will drop one of them and then block forever on it.
struct Client<T> {
    ws: tokio_tungstenite::WebSocketStream<T>,
    seen: Vec<Value>,
}

impl<T> Client<T>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    fn new(ws: tokio_tungstenite::WebSocketStream<T>) -> Self {
        Self {
            ws,
            seen: Vec::new(),
        }
    }

    async fn send(&mut self, v: Value) {
        ws_send(&mut self.ws, v).await;
    }

    /// The first buffered-or-incoming frame matching `pred`, removed from the buffer.
    async fn take(&mut self, what: &str, pred: impl Fn(&Value) -> bool) -> Value {
        if let Some(i) = self.seen.iter().position(&pred) {
            return self.seen.remove(i);
        }
        loop {
            let frame = tokio::time::timeout(Duration::from_secs(10), ws_next_frame(&mut self.ws))
                .await
                .unwrap_or_else(|_| {
                    panic!("timed out waiting for {what}; buffered: {:#?}", self.seen)
                })
                .expect("socket closed");
            if pred(&frame) {
                return frame;
            }
            self.seen.push(frame);
        }
    }

    async fn take_type(&mut self, ty: &str) -> Value {
        let want = ty.to_string();
        self.take(ty, move |f| f["type"] == want).await
    }

    async fn take_response(&mut self, command: &str) -> Value {
        let want = command.to_string();
        self.take(&format!("response to {command}"), move |f| {
            f["type"] == "response" && f["command"] == want
        })
        .await
    }

    /// A `response` correlated to the request `id` *this* client sent. Responses fan out to every
    /// attached client, so the other one's answer to the same command is on this socket too.
    async fn take_response_id(&mut self, command: &str, id: &str) -> Value {
        let (want, id) = (command.to_string(), id.to_string());
        self.take(&format!("response {id} to {command}"), move |f| {
            f["type"] == "response" && f["command"] == want && f["id"] == id
        })
        .await
    }
}

#[tokio::test]
async fn every_attached_client_sees_the_question_and_the_first_answer_wins() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("out.txt");
    let (base, _b) = spawn_model_server(vec![
        write_turn("tu_1", target.to_str().unwrap()),
        turn_text("wrote it"),
    ]);
    let port = free_port();
    let child = serve_ws_approving(
        &base,
        dir.path().to_str().unwrap(),
        port,
        &["--approve", "writes"],
    );
    wait_for_port(port);

    // Phone and TUI on the same session — they coexist, neither evicts the other.
    const SID: &str = "approvalshared1";
    let mut a = Client::new(ws_connect(port, Some(SID)).await);
    let mut b = Client::new(ws_connect(port, Some(SID)).await);

    a.send(json!({ "type": "prompt", "message": "write it" }))
        .await;

    // Both clients are asked the same question.
    let req_a = a.take_type("approval_request").await;
    let req_b = b.take_type("approval_request").await;
    assert_eq!(req_a["request_id"], req_b["request_id"]);
    let request_id = req_a["request_id"].as_str().unwrap().to_string();

    // B answers first.
    b.send(json!({ "type": "approve", "id": "b1", "request_id": request_id, "decision": "allow", "scope": "once" }))
        .await;
    assert_eq!(
        b.take_response_id("approve", "b1").await["data"]["accepted"],
        true
    );

    // Both are told it is settled, so A can dismiss its prompt.
    assert_eq!(a.take_type("approval_resolved").await["decision"], "allow");
    assert_eq!(b.take_type("approval_resolved").await["decision"], "allow");

    // A's late answer loses the race — and says so, rather than erroring.
    a.send(json!({ "type": "approve", "id": "a1", "request_id": request_id, "decision": "deny", "scope": "once" }))
        .await;
    let late = a.take_response_id("approve", "a1").await;
    assert_eq!(late["success"], true, "a lost race is not an error");
    assert_eq!(late["data"]["accepted"], false, "the first answer won");

    // And the call really ran.
    assert_eq!(a.take_response("prompt").await["success"], true);
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "written\n");
    kill(child);
}

#[tokio::test]
async fn a_question_with_no_client_left_to_answer_it_is_denied_rather_than_hung() {
    // A detached background session must not block forever on a question nobody will ever see. The
    // model gets a denial and the run finishes.
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("out.txt");
    let (base, _b) = spawn_model_server(vec![
        // A slow, ungated call gives the client time to disconnect before the gated one is reached.
        turn_tool_use("tu_0", "bash", &json!({ "command": "sleep 1" }).to_string()),
        write_turn("tu_1", target.to_str().unwrap()),
        turn_text("could not write"),
    ]);
    let port = free_port();
    let child = serve_ws_approving(
        &base,
        dir.path().to_str().unwrap(),
        port,
        // Wait forever: only the no-client check can free this.
        &["--approve", "writes", "--approval-timeout", "0"],
    );
    wait_for_port(port);

    const SID: &str = "approvaldetach1";
    let mut ws = Client::new(ws_connect(port, Some(SID)).await);

    ws.send(json!({ "type": "prompt", "message": "write it" }))
        .await;
    ws.take_type("ack").await;
    drop(ws); // the phone goes into a tunnel, mid-run

    // Wait out the `sleep 1` plus the run's tail.
    tokio::time::sleep(Duration::from_secs(4)).await;

    // Reattach: the write was denied, and the session is idle rather than stuck.
    let mut ws = Client::new(ws_connect(port, Some(SID)).await);
    ws.send(json!({ "type": "get_state" })).await;
    let state = ws.take_response("get_state").await;
    assert_eq!(
        state["data"]["is_streaming"], false,
        "the run must have finished"
    );
    assert!(!target.exists(), "the write must not have run unwatched");

    let session_file = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .expect("a session file");
    let results = tool_results(session_file.to_str().unwrap());
    assert!(
        results.iter().any(|r| r.contains("no client is attached")),
        "the model must be told why: {results:?}"
    );
    kill(child);
}
