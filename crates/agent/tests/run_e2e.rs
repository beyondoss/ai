//! End-to-end: the real `beyond-ai-agent run` binary against a mock model server.
//!
//! Scripts a two-turn exchange — the model calls the `read` tool, the loop runs it and feeds the
//! result back, the model replies and ends — and asserts the binary (a) performed the tool call and
//! (b) fed the file's contents back to the model on the second request. No gateway, no provider, no
//! network beyond loopback. This exercises the entire stack: CLI → GatewayClient → loop → tools.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
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

#[test]
fn run_binary_performs_tool_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("hello.txt"), "secret-marker-42\n").unwrap();
    let abs = dir.path().join("hello.txt").to_string_lossy().into_owned();

    let turn1 = turn_tool_use("toolu_1", "read", &json!({ "path": abs }).to_string());
    let turn2 = turn_text("I read the file.");
    let (base, bodies) = spawn_model_server(vec![turn1, turn2]);

    let bin = env!("CARGO_BIN_EXE_beyond-ai-agent");
    let output = Command::new(bin)
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
