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

use serde_json::{Value, json};

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

/// Build an Anthropic SSE turn that ends with `stop_reason: "refusal"` — a distinct terminal
/// condition from a normal end-of-turn (see `agent_core::message::StopReason::Refusal`).
fn turn_refusal(text: &str) -> String {
    let events = [
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 12, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": text } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "refusal" }, "usage": { "output_tokens": 6 } }),
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
fn run_binary_exits_nonzero_on_a_refusal_in_text_mode() {
    // Track L18: a refusal in text mode previously still exited 0, indistinguishable from a normal
    // completion — a script/CI caller has no other signal to key off of in that mode.
    let (base, _bodies) = spawn_model_server(vec![turn_refusal("I can't help with that.")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "do something the model refuses",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        !output.status.success(),
        "a refusal in text mode must exit non-zero.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn run_binary_a_normal_completion_in_text_mode_still_exits_zero() {
    // The exit-code check must not misfire on an ordinary end-of-turn — only an actual refusal.
    let (base, _bodies) = spawn_model_server(vec![turn_text("all done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn run_binary_refusal_exit_code_check_is_scoped_to_text_mode() {
    // `--json` mode already carries `stop_reason` on every `TurnEnd` event in its own output stream —
    // a caller there is expected to inspect it programmatically, not rely on the process exit code.
    let (base, _bodies) = spawn_model_server(vec![turn_refusal("I can't help with that.")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "do something the model refuses",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--json",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");

    assert!(
        output.status.success(),
        "--json mode must not apply the text-mode-only refusal exit code.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\"refusal\"") || stdout.contains("refusal"),
        "the refusal must still be observable in the JSON event stream itself: {stdout}"
    );
}

#[test]
fn run_binary_json_mode_streams_structured_agent_events_not_text() {
    // `--json` must emit a leading session header, then one `AgentEvent` object per line — the full
    // observation surface (tool_start/tool_end included, not just raw text deltas) — instead of the
    // human-readable `[tool: name]`/plain-text output the default text mode prints.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "secret-marker-42\n").unwrap();
    let abs = dir.path().join("hello.txt").to_string_lossy().into_owned();

    let turn1 = turn_tool_use("toolu_1", "read", &json!({ "path": abs }).to_string());
    let turn2 = turn_text("I read the file.");
    let (base, _bodies) = spawn_model_server(vec![turn1, turn2]);

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
            "--json",
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
        !stdout.contains("[tool:"),
        "json mode must not print the text-mode tool marker: {stdout}"
    );

    let lines: Vec<Value> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("non-JSON line {l:?}: {e}")))
        .collect();
    assert!(!lines.is_empty(), "must emit at least the header line");

    let header = &lines[0];
    assert_eq!(header["kind"], "session");
    assert!(header["id"].as_str().is_some_and(|s| !s.is_empty()));
    assert_eq!(header["model"], "claude-test");

    let kinds: Vec<&str> = lines[1..]
        .iter()
        .filter_map(|f| f["kind"].as_str())
        .collect();
    assert_eq!(
        kinds.first(),
        Some(&"agent_start"),
        "kinds: {kinds:?}\nstdout: {stdout}"
    );
    assert!(kinds.contains(&"tool_start"), "kinds: {kinds:?}");
    assert!(kinds.contains(&"tool_end"), "kinds: {kinds:?}");
    assert!(
        kinds.iter().filter(|k| **k == "turn_end").count() >= 2,
        "two turns ran (tool call, then final text): {kinds:?}"
    );
    assert_eq!(
        kinds.last(),
        Some(&"agent_end"),
        "kinds: {kinds:?}\nstdout: {stdout}"
    );

    // The final assistant text must still be present, just carried as a `stream`/`text_delta` event
    // rather than printed as bare text.
    let carries_final_text = lines.iter().any(|f| {
        f["kind"] == "stream"
            && f["type"] == "text_delta"
            && f["text"]
                .as_str()
                .is_some_and(|t| t.contains("I read the file."))
    });
    assert!(
        carries_final_text,
        "final assistant text must appear in a stream/text_delta event: {stdout}"
    );
}

#[test]
fn run_binary_session_id_flag_gives_a_deterministic_id_for_an_ephemeral_run() {
    // Track L6: `--session-id` (matching pi's own flag) lets a script/test harness pick a known,
    // predictable id instead of parsing a randomly-generated one back out of the run's own output —
    // for a plain ephemeral run (no `--session`/`--continue`), the case that previously had no way to
    // override the id at all.
    let dir = tempfile::tempdir().unwrap();
    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--session-id",
            "deterministic-test-id-42",
            "--json",
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
    let header: Value = serde_json::from_str(stdout.lines().next().unwrap()).unwrap();
    assert_eq!(header["kind"], "session");
    assert_eq!(
        header["id"], "deterministic-test-id-42",
        "the ephemeral run must report exactly the requested id, not a generated one: {header:#?}"
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
fn run_binary_prints_startup_timings_only_when_the_env_var_is_set() {
    // Track L10: `AI_AGENT_TIMING=1` (pi's own `PI_TIMING=1`) turns on a startup-timing breakdown to
    // stderr; unset, it must add nothing at all — and even enabled, it must never touch stdout, since
    // that's the streamed-text protocol surface.
    let (base, _bodies) = spawn_model_server(vec![turn_text("done")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let args = [
        "run",
        "hi",
        "--gateway-url",
        &base,
        "--key",
        "bai_v1.test",
        "--model",
        "claude-test",
    ];

    let off = run_cmd(bin)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(off.status.success());
    assert!(
        !String::from_utf8_lossy(&off.stderr).contains("Startup Timings"),
        "must print nothing when unset: {}",
        String::from_utf8_lossy(&off.stderr)
    );

    let (base2, _bodies2) = spawn_model_server(vec![turn_text("done")]);
    let on = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base2,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
        ])
        .env("AI_AGENT_TIMING", "1")
        .stdin(Stdio::null())
        .output()
        .expect("spawn binary");
    assert!(on.status.success());
    let stderr = String::from_utf8_lossy(&on.stderr);
    assert!(stderr.contains("Startup Timings"), "stderr: {stderr}");
    assert!(stderr.contains("TOTAL:"), "stderr: {stderr}");
    let stdout = String::from_utf8_lossy(&on.stdout);
    assert!(
        !stdout.contains("Startup Timings"),
        "timing output must never reach stdout: {stdout}"
    );
    assert!(stdout.contains("done"), "stdout: {stdout}");
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
fn run_binary_expands_a_skill_invocation_in_the_first_message() {
    // `serve` has always expanded `/skill:name`/`/name` invocations before sending them to the model;
    // `run` (this one-shot binary) previously did not — a message starting with either was sent as a
    // literal, unexpanded string. `--trust-project` is required for the project-local skill to be
    // discovered at all (skill discovery is trust-gated; the untrusted-by-default case is covered by
    // `serve_e2e.rs`'s own trust tests, not duplicated here).
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/greet");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: a test skill\n---\nSKILL-BODY-MARKER-123",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
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
        bodies[0].contains("SKILL-BODY-MARKER-123"),
        "the skill's body must be expanded into the first message: {}",
        bodies[0]
    );
    assert!(
        !bodies[0].contains("/skill:greet"),
        "the raw, unexpanded invocation must not reach the model: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_force_untrusted_overrides_trust_project() {
    // Track L8: `--force-untrusted` must win even when `--trust-project` is *also* given — the whole
    // point is a way to force the untrusted codepath for one run regardless of anything else asking
    // for trust.
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/greet");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: a test skill\n---\nSKILL-BODY-MARKER-123",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--force-untrusted",
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
        !bodies[0].contains("SKILL-BODY-MARKER-123"),
        "--force-untrusted must override --trust-project, so the skill must not expand: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_expands_a_prompt_template_in_the_first_message() {
    let dir = tempfile::tempdir().unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(
        prompt_dir.join("fix.md"),
        "Fix the bug in $1 — TEMPLATE-BODY-MARKER-456",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/fix foo.rs",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
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
        bodies[0].contains("TEMPLATE-BODY-MARKER-456") && bodies[0].contains("foo.rs"),
        "the template's body, with its argument substituted, must reach the model: {}",
        bodies[0]
    );
    assert!(
        !bodies[0].contains("/fix foo.rs"),
        "the raw, unexpanded invocation must not reach the model: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_no_skills_leaves_a_skill_invocation_unexpanded() {
    // Same fixture as `run_binary_expands_a_skill_invocation_in_the_first_message`, but with
    // `--no-skills` — the skill must not be discovered at all, so `/skill:greet` reaches the model
    // as a literal, unexpanded string instead of the skill's body.
    let dir = tempfile::tempdir().unwrap();
    let skill_dir = dir.path().join(".claude/skills/greet");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: greet\ndescription: a test skill\n---\nSKILL-BODY-MARKER-123",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/skill:greet",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-skills",
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
        !bodies[0].contains("SKILL-BODY-MARKER-123"),
        "--no-skills must prevent the skill from being discovered/expanded at all: {}",
        bodies[0]
    );
    assert!(
        bodies[0].contains("/skill:greet"),
        "the raw invocation must reach the model unexpanded: {}",
        bodies[0]
    );
}

#[test]
fn run_binary_no_prompt_templates_leaves_a_template_invocation_unexpanded() {
    // Same fixture as `run_binary_expands_a_prompt_template_in_the_first_message`, but with
    // `--no-prompt-templates` — the template must not be discovered at all, so `/fix foo.rs` reaches
    // the model as a literal, unexpanded string instead of the template's body.
    let dir = tempfile::tempdir().unwrap();
    let prompt_dir = dir.path().join(".claude/prompts");
    std::fs::create_dir_all(&prompt_dir).unwrap();
    std::fs::write(
        prompt_dir.join("fix.md"),
        "Fix the bug in $1 — TEMPLATE-BODY-MARKER-456",
    )
    .unwrap();
    let (base, bodies) = spawn_model_server(vec![turn_text("done")]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "/fix foo.rs",
            "--gateway-url",
            &base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--trust-project",
            "--no-prompt-templates",
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
        !bodies[0].contains("TEMPLATE-BODY-MARKER-456"),
        "--no-prompt-templates must prevent the template from being discovered/expanded at all: {}",
        bodies[0]
    );
    assert!(
        bodies[0].contains("/fix foo.rs"),
        "the raw invocation must reach the model unexpanded: {}",
        bodies[0]
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

#[test]
fn run_binary_initializes_an_existing_empty_session_file_instead_of_hard_failing() {
    // Track L8: `--session <path>` pointing at a zero-byte file (e.g. `touch`'d ahead of time by a
    // caller that wants the path to already exist) must initialize it in place, not hard-fail with
    // "session file has no header."
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");
    std::fs::write(&session_file, b"").unwrap(); // pre-create as an empty file
    assert_eq!(std::fs::metadata(&session_file).unwrap().len(), 0);

    let (base, _bodies) = spawn_model_server(vec![turn_text("ok")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = run_cmd(bin)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &base,
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
        output.status.success(),
        "binary failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        std::fs::metadata(&session_file).unwrap().len() > 0,
        "the session file must actually have a real header now"
    );
}

#[test]
fn export_subcommand_renders_an_existing_session_file_with_no_gateway_or_key() {
    let dir = tempfile::tempdir().unwrap();
    let session_file = dir.path().join("s.jsonl");

    // Create a session file the ordinary way, with a real (fake) model server — this part still
    // needs a gateway/key, exactly like any other `run`.
    let (base, _bodies) = spawn_model_server(vec![turn_text("the answer is 42")]);
    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let setup = run_cmd(bin)
        .args([
            "run",
            "what is the answer?",
            "--gateway-url",
            &base,
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
        setup.status.success(),
        "session setup run failed.\nstderr: {}",
        String::from_utf8_lossy(&setup.stderr)
    );
    assert!(session_file.exists());

    // Now export that already-persisted session file directly — no --gateway-url/--key/--model at
    // all, proving the export subcommand is pure offline rendering of what's on disk, unlike `run
    // --export` (which only exports after a live model run completes).
    let export_path = dir.path().join("transcript.html");
    let output = Command::new(bin)
        .args([
            "export",
            session_file.to_str().unwrap(),
            export_path.to_str().unwrap(),
        ])
        .env("HOME", dir.path())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "export subcommand failed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let html = std::fs::read_to_string(&export_path).expect("exported file must exist");
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("what is the answer?"));
    assert!(html.contains("the answer is 42"));
}
