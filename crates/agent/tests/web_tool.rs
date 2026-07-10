//! `run` e2e: the `web` tool driven through the real binary. The model (scripted via the mock gateway)
//! calls `web`, which fetches a loopback **fixture** HTTP server. Two servers are in play: the mock
//! model gateway (`spawn_model_server`) and the fixture page server (`web_fixture` below).
//!
//! The tool refuses loopback by default, so every fetch here passes `--web-allow-host 127.0.0.1` — which
//! doubles as the test of that opt-in path. The SSRF *refusal* is asserted in the tool's own in-process
//! unit tests (`tools::web`); here we prove the whole path end to end through the real binary.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use common::{run_cmd, spawn_model_server, turn_text, turn_tool_use};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_beyond-ai-agent");

/// A persistent loopback HTTP/1.1 server that replies to every request with `raw_response` (a full
/// HTTP response including status line and headers). Records each raw request for assertions. Returns
/// `(base_url, requests)`; the thread lives for the test process.
fn web_fixture(raw_response: String) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorder = requests.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            recorder
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(&buf[..n]).into_owned());
            let _ = stream.write_all(raw_response.as_bytes());
            let _ = stream.flush();
        }
    });
    (format!("http://{addr}"), requests)
}

/// An HTTP/1.1 200 response whose `Content-Length` matches `body`.
fn html_ok(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// Drive `run --json` where the model makes one `web` call with `args`, then says "done". Returns the
/// parsed `AgentEvent` lines and the raw requests the fixture received.
fn drive(fixture_response: String, args: Value) -> (Vec<Value>, Vec<String>) {
    let (web_base, fixture_reqs) = web_fixture(fixture_response);
    // The model's `url` is the fixture base plus whatever path the args already carry.
    let mut args = args;
    if let Some(obj) = args.as_object_mut() {
        let path = obj
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or("/")
            .to_string();
        obj.insert("url".into(), json!(format!("{web_base}{path}")));
    }

    let (gw_base, _bodies) = spawn_model_server(vec![
        turn_tool_use("tu_1", "web", &args.to_string()),
        turn_text("done"),
    ]);

    let dir = tempfile::tempdir().unwrap();
    let output = run_cmd(BIN)
        .args([
            "run",
            "read the page",
            "--gateway-url",
            &gw_base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
            "--no-session-persistence",
            "--json",
            "--web-allow-host",
            "127.0.0.1",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let events: Vec<Value> = stdout
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let reqs = fixture_reqs.lock().unwrap().clone();
    (events, reqs)
}

/// The `web` tool_end result text.
fn web_result(events: &[Value]) -> String {
    events
        .iter()
        .find(|e| e["kind"] == "tool_end" && e["name"] == "web")
        .unwrap_or_else(|| panic!("no web tool_end in {events:#?}"))["result"]
        .as_str()
        .unwrap()
        .to_string()
}

fn web_is_error(events: &[Value]) -> bool {
    events
        .iter()
        .find(|e| e["kind"] == "tool_end" && e["name"] == "web")
        .map(|e| e["is_error"] == true)
        .unwrap_or(false)
}

const PAGE: &str = r#"<html><body>
    <h1>Docs</h1>
    <p>Some <a href="https://example.com">link</a>.</p>
    <div class="item"><h2>Alpha</h2><span class="price">150</span></div>
    <div class="item"><h2>Beta</h2><span class="price">50</span></div>
</body></html>"#;

#[test]
fn fetch_mode_returns_status_headers_and_body() {
    let (events, reqs) = drive(html_ok(PAGE), json!({ "mode": "fetch" }));
    let out = web_result(&events);
    assert!(out.contains("200 OK"), "{out}");
    assert!(out.contains("<h1>Docs</h1>"), "{out}");
    assert_eq!(reqs.len(), 1, "exactly one fetch");
    assert!(reqs[0].starts_with("GET /"), "{}", reqs[0]);
    assert!(
        reqs[0].contains("beyond-ai-agent/"),
        "sends our UA: {}",
        reqs[0]
    );
}

#[test]
fn markdown_mode_strips_html() {
    let (events, _) = drive(html_ok(PAGE), json!({ "mode": "markdown" }));
    let out = web_result(&events);
    assert!(out.contains("# Docs"), "{out}");
    assert!(out.contains("[link](https://example.com)"), "{out}");
    assert!(!out.contains("<h1>"), "raw HTML must be gone: {out}");
}

#[test]
fn extract_mode_pulls_keyed_rows_with_a_where_filter() {
    let (events, _) = drive(
        html_ok(PAGE),
        json!({ "mode": "extract", "selector": ".item", "fields": "name=h2, price=.price", "where": "price > 100" }),
    );
    let out = web_result(&events);
    assert!(out.contains("name\tprice"), "{out}");
    assert!(out.contains("Alpha\t150"), "{out}");
    assert!(!out.contains("Beta"), "Beta (50) filtered out: {out}");
}

#[test]
fn table_mode_reads_rows() {
    let table = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Ada</td><td>36</td></tr></table>";
    let (events, _) = drive(html_ok(table), json!({ "mode": "table" }));
    let out = web_result(&events);
    assert!(out.contains("Name\tAge"), "{out}");
    assert!(out.contains("Ada\t36"), "{out}");
}

#[test]
fn outline_mode_shows_repeating_structure() {
    let (events, _) = drive(html_ok(PAGE), json!({ "mode": "outline" }));
    let out = web_result(&events);
    assert!(out.contains("div.item"), "{out}");
}

#[test]
fn method_headers_and_body_are_sent_to_the_server() {
    let (_events, reqs) = drive(
        html_ok("<html><body>ok</body></html>"),
        json!({ "url": "/submit", "method": "POST", "headers": { "X-Marker": "z9" }, "body": "payload-42" }),
    );
    assert_eq!(reqs.len(), 1);
    let req = &reqs[0];
    assert!(req.starts_with("POST /submit"), "{req}");
    assert!(req.to_lowercase().contains("x-marker: z9"), "{req}");
    assert!(req.ends_with("payload-42"), "{req}");
}

#[test]
fn a_blocked_url_is_refused_end_to_end() {
    // Even with the fixture allow-listed, a *different* internal target (metadata) must be refused. Point
    // the model straight at a link-local address; the tool errors, the run survives.
    let (gw_base, _b) = spawn_model_server(vec![
        turn_tool_use(
            "tu_1",
            "web",
            &json!({ "url": "http://169.254.169.254/latest/meta-data/" }).to_string(),
        ),
        turn_text("understood"),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let output = run_cmd(BIN)
        .args([
            "run",
            "fetch it",
            "--gateway-url",
            &gw_base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--max-steps",
            "4",
            "--no-session-persistence",
            "--json",
            "--web-allow-host",
            "127.0.0.1",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");
    assert!(
        output.status.success(),
        "a refused fetch must not fail the run"
    );
    let events: Vec<Value> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    assert!(web_is_error(&events), "the metadata fetch must be an error");
    let out = web_result(&events);
    assert!(
        out.contains("blocked to prevent access") || out.contains("link-local"),
        "the model must be told why: {out}"
    );
}

#[test]
fn the_tool_and_its_schema_are_advertised_by_default() {
    let (gw_base, bodies) = spawn_model_server(vec![turn_text("nothing to fetch")]);
    let dir = tempfile::tempdir().unwrap();
    run_cmd(BIN)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &gw_base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");
    let body = &bodies.lock().unwrap()[0];
    let tools = common::advertised_tools(body);
    assert!(
        tools.iter().any(|t| t == "web"),
        "web must be advertised: {tools:?}"
    );
}

#[test]
fn excluding_the_tool_removes_it() {
    let (gw_base, bodies) = spawn_model_server(vec![turn_text("hi")]);
    let dir = tempfile::tempdir().unwrap();
    run_cmd(BIN)
        .args([
            "run",
            "hi",
            "--gateway-url",
            &gw_base,
            "--key",
            "bai_v1.test",
            "--model",
            "claude-test",
            "--no-session-persistence",
            "--exclude-tools",
            "web",
        ])
        .current_dir(dir.path())
        .output()
        .expect("spawn binary");
    let body = &bodies.lock().unwrap()[0];
    assert!(!common::advertised_tools(body).iter().any(|t| t == "web"));
}
