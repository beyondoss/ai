//! Test doubles shared by more than one crate.
//!
//! These lived in `crates/agent/tests/common` until the fleet simulator needed them too, and a
//! simulator is a binary in its own crate — it cannot reach another crate's test module. Nothing
//! moved *changed*: `tests/common` re-exports this wholesale, so the 77 test files that use
//! `spawn_model_server` are untouched.
//!
//! The mock model server speaks Anthropic SSE over a hand-rolled HTTP/1.1 socket — deliberately, so
//! it has no async runtime and no framework of its own to agree with the thing under test.

pub mod grant;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};

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

fn read_more(stream: &mut TcpStream, buf: &mut Vec<u8>) -> bool {
    let mut tmp = [0u8; 2048];
    let n = stream.read(&mut tmp).unwrap_or(0);
    if n == 0 {
        return false;
    }
    buf.extend_from_slice(&tmp[..n]);
    true
}

/// Drain an HTTP/1.1 chunked body starting at `cursor`. The gateway's catalog walk (and OpenAI
/// `stream_options` injection) strips `Content-Length` and forwards `transfer-encoding: chunked`,
/// so a mock that only honors `Content-Length` returns on headers and never sees the body.
fn read_chunked_body(stream: &mut TcpStream, buf: &mut Vec<u8>, mut cursor: usize) {
    loop {
        let size_end = loop {
            if let Some(i) = buf[cursor..].windows(2).position(|w| w == b"\r\n") {
                break cursor + i;
            }
            if !read_more(stream, buf) {
                return;
            }
        };
        let size_line = std::str::from_utf8(&buf[cursor..size_end]).unwrap_or("");
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16).unwrap_or(0);
        if size == 0 {
            // Last chunk, then optional trailers, then a terminating blank line.
            loop {
                if buf[cursor..].windows(4).any(|w| w == b"\r\n\r\n") {
                    return;
                }
                if !read_more(stream, buf) {
                    return;
                }
            }
        }
        let chunk_end = size_end + 2 + size + 2; // CRLF after size, data, CRLF after data
        while buf.len() < chunk_end {
            if !read_more(stream, buf) {
                return;
            }
        }
        cursor = chunk_end;
    }
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
            // Keep reading until the full body has arrived, then return the WHOLE raw request
            // (headers + body) so callers can assert on both (e.g. a swapped-in pool key).
            if headers.lines().any(|l| {
                l.strip_prefix("transfer-encoding:")
                    .is_some_and(|v| v.split(',').any(|e| e.trim() == "chunked"))
            }) {
                read_chunked_body(stream, &mut buf, pos + 4);
            } else {
                let len = headers
                    .lines()
                    .find_map(|l| {
                        l.strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                let need = pos + 4 + len;
                while buf.len() < need && read_more(stream, &mut buf) {}
            }
            return String::from_utf8_lossy(&buf).into_owned();
        }
        if !read_more(stream, &mut buf) {
            return String::from_utf8_lossy(&buf).into_owned();
        }
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
                // Drain the *whole* request before answering. Closing a socket with unread data
                // in its receive buffer makes the kernel send an RST instead of a FIN, which
                // discards whatever the peer had not yet read — including the response just
                // written. That is how a perfectly good mock turns into "error sending request".
                let _ = read_http_request(&mut stream);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{resp}"
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            }
        }
        if let Ok((mut stream, _)) = listener.accept() {
            let _ = read_http_request(&mut stream);
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
                // Drain the *whole* request before answering. Closing a socket with unread data
                // in its receive buffer makes the kernel send an RST instead of a FIN, which
                // discards whatever the peer had not yet read — including the response just
                // written. That is how a perfectly good mock turns into "error sending request".
                let _ = read_http_request(&mut stream);
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
