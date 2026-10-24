//! The stand-in the load proof runs against: a scripted model and a bundle host in one process.
//!
//! - the **model** (`--model-listen`) answers instantly and deterministically, in Anthropic SSE: a prompt
//!   gets one `read` of the bundle's skill, the tool result gets a one-line answer. It is the only
//!   model the load run ever talks to, so the run spends nothing.
//! - the **bundle host** (`--listen`) serves the bundles the driver uploads (`PUT /b/<sha256>`, bearer
//!   token) to the replica (`GET /b/<sha256>`). Every GET is held for `--fetch-hold-ms` before it is
//!   answered, which makes concurrent fetches overlap so their peak is something a server can see;
//!   S3 gives the replica no such number. `GET /stats` reports, and `?reset=1` zeroes, the fetch and
//!   connector counters.
//! - **connector targets** (`/mcp/<n>`): any request is counted (connections, distinct paths, the most
//!   open at once) and answered 404 after `--mcp-hold-ms`. A `.mcp.json` with many servers shows up here
//!   as the number of dials the replica made.

// A stand-in for a test run: a poisoned counter lock is a bug to stop on, not to handle.
#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use beyond_ai_test_support::{turn_text, turn_tool_use};
use serde_json::{Value, json};

pub struct Args {
    pub listen: String,
    pub model_listen: String,
    pub token: String,
    pub fetch_hold_ms: u64,
    pub mcp_hold_ms: u64,
}

#[derive(Default)]
struct State {
    bundles: Mutex<HashMap<String, Arc<Vec<u8>>>>,
    fetch_now: AtomicUsize,
    fetch_peak: AtomicUsize,
    fetches: Mutex<BTreeMap<String, usize>>,
    mcp_now: AtomicUsize,
    mcp_peak: AtomicUsize,
    mcp_conns: AtomicUsize,
    mcp_paths: Mutex<HashSet<String>>,
}

impl State {
    fn stats(&self, reset: bool) -> Value {
        let mut fetches = self.fetches.lock().unwrap();
        let total: usize = fetches.values().sum();
        let refetched: usize = fetches.values().map(|n| n.saturating_sub(1)).sum();
        let mut paths = self.mcp_paths.lock().unwrap();
        let v = json!({
            "bundles_held": self.bundles.lock().unwrap().len(),
            "fetches": total,
            "distinct_bundles_fetched": fetches.len(),
            "refetches": refetched,
            "fetch_peak": self.fetch_peak.load(SeqCst),
            "mcp_connections": self.mcp_conns.load(SeqCst),
            "mcp_distinct_servers": paths.len(),
            "mcp_peak_open": self.mcp_peak.load(SeqCst),
        });
        if reset {
            fetches.clear();
            paths.clear();
            self.fetch_peak.store(self.fetch_now.load(SeqCst), SeqCst);
            self.mcp_peak.store(self.mcp_now.load(SeqCst), SeqCst);
            self.mcp_conns.store(0, SeqCst);
        }
        v
    }
}

static NEXT_CALL: AtomicUsize = AtomicUsize::new(0);

/// What the scripted model says to one request: read the skill, then answer.
fn answer(req: &str) -> String {
    let body = req.split_once("\r\n\r\n").map_or("", |(_, b)| b);
    let last_is_tool_result = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["messages"].as_array()?.last().cloned())
        .is_some_and(|m| m["content"].to_string().contains("tool_result"));
    if last_is_tool_result {
        turn_text("The house style is short, plain and signed. -- Juniper")
    } else {
        turn_tool_use(
            &format!("t{}", NEXT_CALL.fetch_add(1, SeqCst)),
            "read",
            r#"{"path":"skill:house-style"}"#,
        )
    }
}

struct Request {
    method: String,
    path: String,
    auth: String,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> Option<Request> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    r.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next()?.to_string(), parts.next()?.to_string());
    let (mut len, mut auth) = (0usize, String::new());
    loop {
        let mut h = String::new();
        r.read_line(&mut h).ok()?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            len = v.trim().parse().ok()?;
        } else if lower.starts_with("authorization:") {
            auth = h["authorization:".len()..].trim().to_string();
        }
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).ok()?;
    Some(Request {
        method,
        path,
        auth,
        body,
    })
}

fn respond(mut stream: TcpStream, status: &str, ctype: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

fn handle(stream: TcpStream, st: &State, args: &Args) {
    let Some(req) = read_request(&stream) else {
        return;
    };
    let path = req.path.split('?').next().unwrap_or("");
    let authed = req.auth == format!("Bearer {}", args.token);
    if let Some(sha) = path.strip_prefix("/b/") {
        if req.method == "PUT" {
            if !authed {
                return respond(stream, "401 Unauthorized", "text/plain", b"");
            }
            st.bundles
                .lock()
                .unwrap()
                .insert(sha.to_string(), Arc::new(req.body));
            return respond(stream, "200 OK", "text/plain", b"ok");
        }
        let bytes = st.bundles.lock().unwrap().get(sha).cloned();
        let Some(bytes) = bytes else {
            return respond(stream, "404 Not Found", "text/plain", b"");
        };
        let now = st.fetch_now.fetch_add(1, SeqCst) + 1;
        st.fetch_peak.fetch_max(now, SeqCst);
        *st.fetches
            .lock()
            .unwrap()
            .entry(sha.to_string())
            .or_default() += 1;
        std::thread::sleep(Duration::from_millis(args.fetch_hold_ms));
        st.fetch_now.fetch_sub(1, SeqCst);
        return respond(stream, "200 OK", "application/x-tar", &bytes);
    }
    if path.starts_with("/mcp/") {
        let now = st.mcp_now.fetch_add(1, SeqCst) + 1;
        st.mcp_peak.fetch_max(now, SeqCst);
        st.mcp_conns.fetch_add(1, SeqCst);
        st.mcp_paths.lock().unwrap().insert(path.to_string());
        std::thread::sleep(Duration::from_millis(args.mcp_hold_ms));
        st.mcp_now.fetch_sub(1, SeqCst);
        return respond(stream, "404 Not Found", "text/plain", b"");
    }
    if path == "/stats" {
        if !authed {
            return respond(stream, "401 Unauthorized", "text/plain", b"");
        }
        let reset = req.path.contains("reset=1");
        let body = st.stats(reset).to_string();
        return respond(stream, "200 OK", "application/json", body.as_bytes());
    }
    respond(stream, "404 Not Found", "text/plain", b"");
}

/// Serve until the process ends.
pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    beyond_ai_test_support::spawn_model_server_fn_on(&args.model_listen, answer);
    let listener = TcpListener::bind(&args.listen)?;
    eprintln!(
        "load-host: bundles on {}, model on {}",
        args.listen, args.model_listen
    );
    let (st, args) = (Arc::new(State::default()), Arc::new(args));
    for conn in listener.incoming() {
        let Ok(stream) = conn else { continue };
        let (st, args) = (st.clone(), args.clone());
        std::thread::spawn(move || handle(stream, &st, &args));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(last: Value) -> String {
        format!(
            "POST /v1/messages HTTP/1.1\r\n\r\n{}",
            json!({"messages": [last]})
        )
    }

    #[test]
    fn a_prompt_reads_the_skill_and_the_tool_result_gets_an_answer() {
        let prompt = answer(&req(
            json!({"role":"user","content":[{"type":"text","text":"hi"}]}),
        ));
        assert!(prompt.contains("skill:house-style"), "{prompt}");
        let result = answer(&req(
            json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"x"}]}),
        ));
        assert!(
            result.contains("Juniper") && !result.contains("tool_use"),
            "{result}"
        );
    }
}
