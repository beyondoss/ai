//! MCP Events (draft) over **webhook** — the mode ChatGPT ships — end to end: a real `serve --listen`
//! daemon receiving signed deliveries from a real streamable-HTTP events fixture on its own HTTP
//! listener (`/_beyond/mcp-events/<token>`), a session attached over WebSocket, and a mock model.
//!
//! Covers the verification challenge, delivery into an `mcp_event` frame and a model-visible
//! follow-up, `eventId` dedup under redelivery, a bad signature and a stale timestamp rejected,
//! the TTL refresh loop actually re-subscribing, secret rotation (previous secret accepted, older
//! refused), unsubscribe at shutdown (and 410 for anything after), `terminated` envelopes, an
//! unreachable callback failing the challenge, and the configuration guards.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::mcp_events_fixture::{
    control, emit, eventually, model_requests_with, spawn_http_fixture, state, write_settings,
};
use common::{
    BIN, ChildGuard, DeadPort, HeldPort, SpawnGuarded, TestWs, serve_dir_cmd, spawn_listening,
    spawn_model_server_routed, turn_text, ws_connect, ws_next_frame, ws_send,
};
use serde_json::{Value, json};

struct Daemon {
    child: ChildGuard,
    port: u16,
    bodies: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    home: tempfile::TempDir,
}

fn start_daemon(mcp_servers: Value) -> Daemon {
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), mcp_servers);
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let sessions = home.path().join("sessions");
    // The callback URL names the port before the daemon starts.
    let held = HeldPort::bind();
    let port = held.port();
    let stderr = home.path().join("serve.stderr");
    let mut cmd = serve_dir_cmd(BIN, &base, &sessions.to_string_lossy());
    cmd.args([
        "--mcp-events-callback-url",
        &format!("http://127.0.0.1:{port}"),
    ])
    .env("HOME", home.path())
    .env("BEYOND_AI_AGENT_MCP_EVENTS_COALESCE_MS", "300")
    .env("BEYOND_AI_AGENT_MCP_IDLE_SECS", "0")
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::from(std::fs::File::create(&stderr).unwrap()));
    held.hand_to(&mut cmd);
    let child = cmd.spawn_guarded();
    Daemon {
        child,
        port,
        bodies,
        home,
    }
}

fn webhook_server(mcp_url: &str, events: Value) -> Value {
    json!({
        "name": "hooks",
        "transport": "http",
        "url": mcp_url,
        "headers": { "Authorization": "Bearer principal-1" },
        "events": events,
    })
}

async fn next(
    ws: &mut TestWs,
    timeout: Duration,
    what: &str,
    pred: impl Fn(&Value) -> bool,
) -> Value {
    let deadline = Instant::now() + timeout;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(left, ws_next_frame(ws)).await {
            Ok(Some(f)) if pred(&f) => return f,
            Ok(Some(_)) => {}
            Ok(None) => panic!("socket closed waiting for {what}"),
            Err(_) => panic!("timed out waiting for {what}"),
        }
    }
}

/// Frames matching `pred` within `window`.
async fn quiet(ws: &mut TestWs, window: Duration, pred: impl Fn(&Value) -> bool) -> Vec<Value> {
    let deadline = Instant::now() + window;
    let mut out = Vec::new();
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return out;
        }
        if let Ok(Some(f)) = tokio::time::timeout(left, ws_next_frame(ws)).await
            && pred(&f)
        {
            out.push(f);
        }
    }
}

async fn list(ws: &mut TestWs, id: &str) -> Value {
    ws_send(ws, json!({ "type": "mcp_events_list", "id": id })).await;
    next(ws, Duration::from_secs(20), id, |f| {
        f["type"] == "response" && f["id"] == id
    })
    .await
}

async fn wait_active(ws: &mut TestWs) -> Value {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut n = 0;
    loop {
        n += 1;
        let l = list(ws, &format!("l{n}")).await;
        let subs = l["data"]["subscriptions"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if !subs.is_empty() && subs.iter().all(|s| s["state"] == "active") {
            return subs[0].clone();
        }
        assert!(Instant::now() < deadline, "never active: {l:#}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn hook(fixture: &str) -> Value {
    state(fixture)["hooks"][0].clone()
}

fn sigterm_and_wait(child: &mut ChildGuard) {
    let pid = child.id().to_string();
    Command::new("kill").args(["-TERM", &pid]).status().unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while child.try_wait().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "serve did not exit after SIGTERM"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn webhook_end_to_end_verify_deliver_dedup_reject_refresh_rotate_unsubscribe() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        // Grants are capped at 1.5 s, so the refresh loop has to run several times in this test.
        ("MCP_FIXTURE_MAX_TTL_MS", "1500"),
    ]);
    let mut d = start_daemon(json!([webhook_server(
        &mcp_url,
        json!([{
            "name": "ticket.updated",
            "arguments": { "project": "alpha" },
            "delivery": "webhook",
            "instructions": "Reply with the ticket id."
        }])
    )]));
    let mut ws = ws_connect(d.port, Some("mcp-events")).await;
    let sub = wait_active(&mut ws).await;
    assert_eq!(sub["delivery"], "webhook");
    assert!(sub["subscription_id"].as_str().unwrap().starts_with("sub_"));

    // The verification challenge was answered before the server activated delivery.
    let st = state(&fixture);
    assert_eq!(st["verifications"][0]["ok"], true, "{st:#}");
    let h = hook(&fixture);
    assert_eq!(h["principal"], "principal-1");
    let callback = h["url"].as_str().unwrap().to_owned();
    assert!(callback.starts_with(&format!("http://127.0.0.1:{}/_beyond/mcp-events/", d.port)));
    assert!(h["secrets"][0].as_str().unwrap().starts_with("whsec_"));

    // A delivery → an `mcp_event` frame and a follow-up the model sees.
    let r = emit(
        &fixture,
        json!({ "project": "alpha", "event_id": "gh-1", "data": { "ticket_id": "W-1", "summary": "webhook change" } }),
    );
    assert_eq!(r["deliveries"][0]["status"], 200, "{r:#}");
    let f = next(
        &mut ws,
        Duration::from_secs(20),
        "the webhook mcp_event",
        |f| f["type"] == "mcp_event",
    )
    .await;
    assert_eq!(f["delivery"], "webhook");
    assert_eq!(f["event"]["eventId"], "gh-1");
    assert_eq!(f["event"]["data"]["ticket_id"], "W-1");
    let resp = next(
        &mut ws,
        Duration::from_secs(30),
        "the injected prompt's response",
        |f| f["type"] == "response" && f["id"] == "mcp_events:1",
    )
    .await;
    assert_eq!(resp["success"], true, "{resp:#}");
    let runs = model_requests_with(&d.bodies, "webhook change");
    assert_eq!(runs.len(), 1);
    assert!(runs[0].contains("Reply with the ticket id."));

    // At-least-once redelivery of the same eventId: acknowledged, never surfaced twice.
    let r = control(
        &fixture,
        "POST",
        "/control/redeliver",
        Some(&json!({ "event_id": "gh-1" })),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 200,
        "a duplicate is still acked: {r:#}"
    );
    // A forged signature and a replayed (stale) timestamp are refused.
    let r = emit(
        &fixture,
        json!({ "project": "alpha", "event_id": "forged", "tamper": "signature", "data": {} }),
    );
    assert_eq!(r["deliveries"][0]["status"], 401, "{r:#}");
    let r = emit(
        &fixture,
        json!({ "project": "alpha", "event_id": "stale", "tamper": "stale", "data": {} }),
    );
    assert_eq!(r["deliveries"][0]["status"], 400, "{r:#}");
    let leaked = quiet(&mut ws, Duration::from_millis(800), |f| {
        f["type"] == "mcp_event"
    })
    .await;
    assert!(
        leaked.is_empty(),
        "no duplicate, forged or stale event surfaced: {leaked:#?}"
    );
    let l = list(&mut ws, "after-dupes").await;
    assert_eq!(l["data"]["subscriptions"][0]["duplicates"], 1, "{l:#}");
    assert_eq!(l["data"]["subscriptions"][0]["delivered"], 1);

    // The refresh loop re-subscribes before each 1.5 s grant lapses, rotating the secret each time.
    let h = eventually(Duration::from_secs(20), "two refreshes", || {
        let h = hook(&fixture);
        (h["refreshes"].as_u64().unwrap() >= 2).then_some(h)
    });
    assert_eq!(h["expired"], false, "refreshed before the grant lapsed");
    let secrets = h["secrets"].as_array().unwrap().len();
    assert!(secrets >= 3, "a fresh secret on every refresh: {h:#}");
    // Live after several TTLs: deliveries still land.
    let r = emit(
        &fixture,
        json!({ "project": "alpha", "event_id": "late", "data": { "ticket_id": "W-2" } }),
    );
    assert_eq!(r["deliveries"][0]["status"], 200, "{r:#}");
    next(
        &mut ws,
        Duration::from_secs(20),
        "the post-refresh event",
        |f| f["type"] == "mcp_event" && f["event"]["eventId"] == "late",
    )
    .await;

    // (Rotation grace is proven exactly in its own test below.)

    // Shutdown unsubscribes, and the callback answers 410 Gone afterwards.
    drop(ws);
    sigterm_and_wait(&mut d.child);
    let st = state(&fixture);
    let unsubs = st["unsubscribes"].as_array().unwrap();
    assert_eq!(unsubs.len(), 1, "{st:#}");
    assert_eq!(unsubs[0]["delivery"]["url"], callback.as_str());
    assert_eq!(unsubs[0]["arguments"], json!({ "project": "alpha" }));
    assert!(st["hooks"].as_array().unwrap().is_empty());
    let _ = d.home.path();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_terminated_envelope_ends_the_subscription_and_later_deliveries_get_410() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let d = start_daemon(json!([webhook_server(
        &mcp_url,
        json!([{ "name": "ticket.updated", "delivery": "webhook", "action": "notify" }])
    )]));
    let mut ws = ws_connect(d.port, Some("mcp-events")).await;
    wait_active(&mut ws).await;
    let callback = hook(&fixture)["url"].as_str().unwrap().to_owned();

    let r = control(&fixture, "POST", "/control/terminate", Some(&json!({})));
    assert_eq!(r["webhook_statuses"][0], 200, "{r:#}");
    let t = next(
        &mut ws,
        Duration::from_secs(20),
        "the terminated status",
        |f| f["type"] == "mcp_event_status" && f["kind"] == "terminated",
    )
    .await;
    assert_eq!(t["error"]["data"]["reason"], "Access revoked");
    let l = list(&mut ws, "after").await;
    assert_eq!(
        l["data"]["subscriptions"][0]["state"], "terminated",
        "{l:#}"
    );

    // The route is gone: anything posted to it is told not to retry.
    let path = callback
        .split_once(&format!("127.0.0.1:{}", d.port))
        .unwrap()
        .1
        .to_owned();
    let status = raw_post(d.port, &path, b"{}");
    assert_eq!(status, 410);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_callback_url_nothing_answers_fails_the_challenge_and_the_subscribe() {
    let (_fx, mcp_url, _fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), json!([webhook_server(&mcp_url, json!([]))]));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    // Points somewhere nothing listens (and, held, nothing can start to): the server's verification
    // POST cannot land.
    let dead = DeadPort::bind();
    let mut cmd = serve_dir_cmd(BIN, &base, &home.path().join("s").to_string_lossy());
    cmd.args([
        "--mcp-events-callback-url",
        &format!("http://127.0.0.1:{}", dead.port()),
    ])
    .env("HOME", home.path())
    .stdin(Stdio::null())
    .stdout(Stdio::null());
    let (_child, port) = spawn_listening(&mut cmd);
    let mut ws = ws_connect(port, Some("dead-callback")).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks", "name": "ticket.updated", "delivery": "webhook" }),
    )
    .await;
    let r = next(
        &mut ws,
        Duration::from_secs(30),
        "the subscribe response",
        |f| f["id"] == "s",
    )
    .await;
    assert_eq!(r["success"], false);
    let e = r["error"].as_str().unwrap();
    assert!(
        e.contains("CallbackEndpointError") && e.contains("-32015"),
        "{e}"
    );
}

#[test]
fn the_callback_url_flag_requires_a_listener() {
    let out = common::serve_cmd(BIN, "http://127.0.0.1:9", "/nonexistent/s.jsonl")
        .args(["--mcp-events-callback-url", "https://agent.example.com"])
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--mcp-events-callback-url requires --listen"),
        "{stderr}"
    );
}

/// One raw POST to the daemon's listener; returns the status code.
fn raw_post(port: u16, path: &str, body: &[u8]) -> u16 {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .unwrap();
    s.write_all(body).unwrap();
    let mut buf = String::new();
    let _ = s.read_to_string(&mut buf);
    buf.split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_server_can_verify_the_callback_through_the_receiver_document_instead_of_a_challenge() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_VERIFY_VIA_WELL_KNOWN", "1"),
    ]);
    let d = start_daemon(json!([webhook_server(
        &mcp_url,
        json!([{ "name": "ticket.updated", "delivery": "webhook", "action": "notify" }])
    )]));
    // The listener publishes which paths accept deliveries.
    let (status, body) = raw_get(d.port, "/.well-known/mcp-webhook-receiver.json");
    assert_eq!(status, 200);
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap(),
        json!({ "receivers": ["/_beyond/mcp-events/"] })
    );
    let mut ws = ws_connect(d.port, Some("mcp-events")).await;
    wait_active(&mut ws).await;
    let v = &state(&fixture)["verifications"][0];
    assert_eq!(v["via"], "well-known", "{v:#}");
    assert_eq!(v["ok"], true);
    let r = emit(&fixture, json!({ "event_id": "wk-1", "data": {} }));
    assert_eq!(r["deliveries"][0]["status"], 200, "{r:#}");
    next(&mut ws, Duration::from_secs(20), "wk-1", |f| {
        f["event"]["eventId"] == "wk-1"
    })
    .await;
}

/// One raw GET to the daemon's listener; returns the status code and body.
fn raw_get(port: u16, path: &str) -> (u16, String) {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(s, "GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let mut buf = String::new();
    let _ = s.read_to_string(&mut buf);
    let status = buf
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .unwrap_or(0);
    (
        status,
        buf.split_once("\r\n\r\n")
            .map(|(_, b)| b.to_owned())
            .unwrap_or_default(),
    )
}

/// Exact, with no race against the refresh loop: grants of 3 s mean a refresh lands every ~2.25 s,
/// and right after one has been observed there are ~2 s in which the receiver's secrets are known:
/// the current one, the previous one (accepted — rotation grace), and anything older (refused).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rotation_grace_accepts_exactly_the_previous_secret_and_refuses_older_ones() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_MAX_TTL_MS", "3000"),
    ]);
    let d = start_daemon(json!([webhook_server(
        &mcp_url,
        json!([{ "name": "ticket.updated", "delivery": "webhook", "action": "notify" }])
    )]));
    let mut ws = ws_connect(d.port, Some(common::mcp_events_fixture::EVENTS_SESSION)).await;
    wait_active(&mut ws).await;
    // Catch the moment the second refresh lands (three secrets seen: original + two rotations).
    let gens = eventually(Duration::from_secs(20), "the second refresh", || {
        let n = hook(&fixture)["secrets"].as_array().unwrap().len();
        (n >= 3).then_some(n)
    });
    let current = gens - 1;
    let r = emit(
        &fixture,
        json!({ "event_id": "cur-secret", "sign_with_generation": current, "data": {} }),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 200,
        "the current secret: {r:#}"
    );
    let r = emit(
        &fixture,
        json!({ "event_id": "prev-secret", "sign_with_generation": current - 1, "data": {} }),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 200,
        "the previous secret is still accepted: {r:#}"
    );
    let r = emit(
        &fixture,
        json!({ "event_id": "old-secret", "sign_with_generation": current - 2, "data": {} }),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 401,
        "two rotations old is refused: {r:#}"
    );
    assert_eq!(
        hook(&fixture)["secrets"].as_array().unwrap().len(),
        gens,
        "no refresh landed during the checks, so they were exact"
    );
    for id in ["cur-secret", "prev-secret"] {
        next(&mut ws, Duration::from_secs(10), id, |f| {
            f["event"]["eventId"] == id
        })
        .await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_future_dated_or_absurd_timestamp_is_refused_without_harm() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let d = start_daemon(json!([webhook_server(
        &mcp_url,
        json!([{ "name": "ticket.updated", "delivery": "webhook", "action": "notify" }])
    )]));
    let mut ws = ws_connect(d.port, Some(common::mcp_events_fixture::EVENTS_SESSION)).await;
    wait_active(&mut ws).await;
    // Ten minutes in the future, correctly signed: outside the window either way.
    let r = emit(
        &fixture,
        json!({ "event_id": "future", "tamper": "future", "data": {} }),
    );
    assert_eq!(r["deliveries"][0]["status"], 400, "{r:#}");
    // i64::MIN would overflow `now - ts`; the answer must still be a plain 400 (no panic, no reset).
    let path = callback_path(&fixture, d.port);
    let body = b"{}";
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
         webhook-id: x\r\nwebhook-timestamp: -9223372036854775808\r\nwebhook-signature: v1,AAAA\r\n\r\n",
        body.len()
    );
    let status =
        common::mcp_events_fixture::raw_request(d.port, &head, body, Duration::from_secs(5));
    assert_eq!(status, 400);
    // The daemon is fine afterwards.
    let r = emit(&fixture, json!({ "event_id": "after", "data": {} }));
    assert_eq!(r["deliveries"][0]["status"], 200, "{r:#}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_route_answers_unknown_and_oversized_deliveries_before_reading_the_body() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let d = start_daemon(json!([webhook_server(
        &mcp_url,
        json!([{ "name": "ticket.updated", "delivery": "webhook", "action": "notify" }])
    )]));
    let mut ws = ws_connect(d.port, Some(common::mcp_events_fixture::EVENTS_SESSION)).await;
    wait_active(&mut ws).await;
    // A 4 MB body is announced and never sent: the answer must come anyway, at once.
    let head = |path: &str, len: usize| {
        format!(
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len}\r\n\r\n"
        )
    };
    let started = Instant::now();
    let unknown = common::mcp_events_fixture::raw_request(
        d.port,
        &head("/_beyond/mcp-events/not-a-token", 4_000_000),
        b"",
        Duration::from_secs(3),
    );
    assert_eq!(
        unknown, 410,
        "unknown token: Gone, without reading the body"
    );
    let live = callback_path(&fixture, d.port);
    let oversized = common::mcp_events_fixture::raw_request(
        d.port,
        &head(&live, 2_000_000),
        b"",
        Duration::from_secs(3),
    );
    assert_eq!(
        oversized, 413,
        "over the 1 MiB cap: Payload Too Large, without reading the body"
    );
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "both answered without waiting for a body"
    );
}

fn callback_path(fixture: &str, port: u16) -> String {
    let url = hook(fixture)["url"].as_str().unwrap().to_owned();
    url.split_once(&format!("127.0.0.1:{port}"))
        .unwrap()
        .1
        .to_owned()
}

/// Service mode has no MCP Events, so it has no webhook route either — not even a `410`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_webhook_route_does_not_exist_in_service_mode() {
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let svc = common::service::Service::start(&base, &["s1"]).await;
    let status = common::mcp_events_fixture::raw_request(
        svc.port,
        "POST /_beyond/mcp-events/anything HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n",
        b"{}",
        Duration::from_secs(5),
    );
    assert_eq!(status, 404);
}
