//! End-to-end: the real `beyond-ai-agent run` binary against a mock model server.
//!
//! Scripts a two-turn exchange — the model calls the `read` tool, the loop runs it and feeds the
//! result back, the model replies and ends — and asserts the binary (a) performed the tool call and
//! (b) fed the file's contents back to the model on the second request. No gateway, no provider, no
//! network beyond loopback. This exercises the entire stack: CLI → GatewayClient → loop → tools.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::json;

/// Build an Anthropic SSE turn that calls one tool with the given JSON-argument string.
fn turn_tool_use(id: &str, name: &str, args_json: &str) -> String {
    let events = [
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": args_json } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
        json!({ "type": "message_stop" }),
    ];
    sse(&events)
}

/// Build an Anthropic SSE turn that calls one tool with its JSON arguments split across multiple
/// `input_json_delta` fragments, the way a real streaming response actually arrives (not the single
/// whole-string delta `turn_tool_use` uses).
fn turn_tool_use_chunked(id: &str, name: &str, arg_fragments: &[&str]) -> String {
    let mut events = vec![
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": id, "name": name, "input": {} } }),
    ];
    for fragment in arg_fragments {
        events.push(json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "input_json_delta", "partial_json": fragment }
        }));
    }
    events.push(json!({ "type": "content_block_stop", "index": 0 }));
    events.push(
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 8 } }),
    );
    events.push(json!({ "type": "message_stop" }));
    sse(&events)
}

/// Build an Anthropic SSE turn that emits text and ends.
fn turn_text(text: &str) -> String {
    let events = [
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 12, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 6 } }),
        json!({ "type": "message_stop" }),
    ];
    sse(&events)
}

fn sse(events: &[serde_json::Value]) -> String {
    events.iter().map(|e| format!("data: {e}\n\n")).collect()
}

/// Read a full HTTP/1.1 request (headers + Content-Length body) from a stream.
fn read_http_request(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 2048];
    loop {
        let header_end = buf.windows(4).position(|w| w == b"\r\n\r\n");
        if let Some(pos) = header_end {
            let headers = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
            let len = headers
                .lines()
                .find_map(|l| {
                    l.strip_prefix("content-length:")
                        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                })
                .unwrap_or(0);
            let mut body = buf[pos + 4..].to_vec();
            while body.len() < len {
                let n = stream.read(&mut tmp).unwrap_or(0);
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&tmp[..n]);
            }
            return String::from_utf8_lossy(&body).into_owned();
        }
        let n = stream.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            return String::from_utf8_lossy(&buf).into_owned();
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

/// Spawn a model server that answers `responses` in order, recording each request body.
fn spawn_model_server(responses: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let recorder = bodies.clone();
    thread::spawn(move || {
        for resp in responses {
            if let Ok((mut stream, _)) = listener.accept() {
                let body = read_http_request(&mut stream);
                recorder.lock().unwrap().push(body);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{resp}"
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            }
        }
    });
    (format!("http://{addr}"), bodies)
}

/// A `HOME` that deliberately doesn't exist on disk, so the spawned `run` binary's `~/.claude/skills`/
/// `~/.claude/trusted-projects.json` discovery (unconditional — skill discovery is no longer gated on
/// project trust) sees "nothing there" instead of the actual developer machine's real files. Every
/// codepath that reads under `HOME` already treats a missing file/directory as empty, not an error.
const ISOLATED_HOME: &str = "/nonexistent-beyond-ai-agent-test-home";

/// [`Command::new`] for the `run` binary, pre-isolated from the real machine's `HOME` — see
/// [`ISOLATED_HOME`]. A test that wants real HOME-relative behavior overrides it via its own
/// `.env("HOME", ...)`, which simply wins (`Command::env` is last-write).
fn run_cmd(bin: &str) -> Command {
    let mut c = Command::new(bin);
    c.env("HOME", ISOLATED_HOME);
    c
}

#[test]
fn run_binary_performs_tool_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "secret-marker-42\n").unwrap();
    let abs = dir.path().join("hello.txt").to_string_lossy().into_owned();

    let turn1 = turn_tool_use("toolu_1", "read", &json!({ "path": abs }).to_string());
    let turn2 = turn_text("I read the file.");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "read hello.txt and report it",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("[tool: read]"),
        "should show the tool call.\nstdout: {stdout}"
    );
    assert!(
        stdout.contains(&abs),
        "the streamed tool-call arguments (a live preview of the call, not just its name) should \
         appear in stdout.\nstdout: {stdout}"
    );
    assert!(
        stdout.contains("I read the file."),
        "should print the final turn.\nstdout: {stdout}"
    );

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "expected two model requests");
    assert!(
        bodies[1].contains("secret-marker-42"),
        "the 2nd request must feed the file contents back as a tool_result.\nbody: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_prints_a_live_preview_of_streamed_tool_arguments() {
    // A real streaming response delivers a tool call's JSON arguments as several fragments, not one
    // whole string. Assembled in order right after the `[tool: name]` marker, they must read as the
    // complete, valid argument JSON on stdout — proving the loop actually renders each
    // `InputJsonDelta` live rather than only ever seeing (and rendering) a single final fragment.
    let dir = tempfile::tempdir().unwrap();
    let turn1 = turn_tool_use_chunked(
        "toolu_1",
        "unknown_preview_tool",
        &[r#"{"comman"#, r#"d":"echo hi"}"#],
    );
    let (base, _bodies) = spawn_model_server(vec![turn1, turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "do the thing",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("[tool: unknown_preview_tool] {\"command\":\"echo hi\"}"),
        "the two fragments must render adjacently, in order, as the complete argument JSON: {stdout}"
    );
}

#[test]
fn run_binary_exports_the_transcript_when_asked() {
    let dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("all done")]);
    let export_path = dir.path().join("transcript.html");

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "say hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--export",
            export_path.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let html = std::fs::read_to_string(&export_path).expect("exported file must exist");
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("say hi"));
    assert!(html.contains("all done"));
}

#[test]
fn run_binary_sends_multiple_messages_as_sequential_turns() {
    // Two positional messages must run as two separate turns — the second only sent after the first
    // fully completes — not concatenated into one prompt.
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) =
        spawn_model_server(vec![turn_text("first reply"), turn_text("second reply")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "first message",
            "second message",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "binary failed.\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("first reply"));
    assert!(stdout.contains("second reply"));

    let bodies = bodies.lock().unwrap();
    assert_eq!(bodies.len(), 2, "each message must be its own model call");
    assert!(
        bodies[0].contains("first message") && !bodies[0].contains("second message"),
        "the first request must not already include the second message: {}",
        bodies[0]
    );
    assert!(
        bodies[1].contains("first message") && bodies[1].contains("second message"),
        "the second request must carry the accumulated history: {}",
        bodies[1]
    );
}

#[test]
fn run_binary_composes_an_at_file_reference_into_the_first_message() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), "the-file-marker-77").unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("got it")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "@notes.txt",
            "summarize this",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("the-file-marker-77") && bodies[0].contains("summarize this"),
        "the request must carry both the file contents and the message: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_reads_piped_stdin_into_the_first_message() {
    let dir = tempfile::tempdir().unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("noted")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let mut child = run_cmd(bin)
        .args([
            "run",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .current_dir(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn binary");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"piped-content-marker-99")
        .unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bodies = bodies.lock().unwrap();
    assert!(
        bodies[0].contains("piped-content-marker-99"),
        "piped stdin must reach the model as the message: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_errors_when_no_task_stdin_or_file_is_given() {
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "--gateway-url",
            "http://127.0.0.1:1",
            "--key",
            "bai_v1.test",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        !output.status.success(),
        "an empty invocation must fail, not silently prompt with nothing"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no task given"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_binary_list_models_prints_known_model_ids_with_no_gateway_or_key() {
    // A pure informational query — no `--gateway-url`/`--key` needed, matching `tools`'s own shape.
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args(["list-models"])
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("claude-opus-4-8"), "stdout: {stdout}");
    assert!(stdout.contains("gpt-5"), "stdout: {stdout}");
}

#[test]
fn run_binary_session_flag_persists_and_resumes_across_invocations() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");

    let (base1, bodies1) = spawn_model_server(vec![turn_text("first answer")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output1 = run_cmd(bin)
        .args([
            "run",
            "remember the marker: xyzzy-42",
            "--gateway-url",
            &base1,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output1.status.success(),
        "first run failed.\nstderr: {}",
        String::from_utf8_lossy(&output1.stderr)
    );
    drop(bodies1);
    assert!(session_file.exists(), "the session file must be created");

    let (base2, bodies2) = spawn_model_server(vec![turn_text("second answer")]);
    let output2 = run_cmd(bin)
        .args([
            "run",
            "what was the marker?",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session",
            session_file.to_str().unwrap(),
        ])
        .current_dir(dir.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(
        output2.status.success(),
        "second run failed.\nstderr: {}",
        String::from_utf8_lossy(&output2.stderr)
    );

    let bodies2 = bodies2.lock().unwrap();
    assert!(
        bodies2[0].contains("xyzzy-42"),
        "the second run must see the first run's history: {}",
        bodies2[0]
    );
    assert!(bodies2[0].contains("first answer"));
    assert!(bodies2[0].contains("what was the marker?"));
}
