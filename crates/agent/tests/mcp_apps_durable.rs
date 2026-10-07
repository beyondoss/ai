//! MCP Apps state that must outlive a process, a session switch, or a copy — and the limits that
//! must hold without trusting the server. Each against the real `serve` binary:
//! - view context and the view replay store survive a daemon restart (and, in service mode, are
//!   sealed with the tenant's key — never on disk in the clear);
//! - a fork and a clone keep the view context of the turns they copy;
//! - an unadvertised oversized view is refused at the transport, without being read whole — over
//!   stdio and over streamable HTTP;
//! - a question a server asks while calls from two sessions are in flight on one shared stdio
//!   connection is refused, not handed to either session.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::mcp_apps::{
    Ws, app_request, command, daemon, daemon_opts, declare_apps, home_with_fixture, of_type,
    open_view, prompt, request_with, strip_cache_control, tool_end_text, views_model,
};
use common::mcp_apps_http::spawn_http_apps_fixture;
use common::service::{Options, Service};
use common::{
    spawn_model_server_routed, turn_text, turn_tool_use, ws_next_frame, ws_read_until_response,
    ws_send,
};
use serde_json::{Value, json};

async fn frames_within<T>(
    ws: &mut tokio_tungstenite::WebSocketStream<T>,
    window: Duration,
) -> Vec<Value>
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut out = Vec::new();
    let deadline = tokio::time::Instant::now() + window;
    while let Ok(Some(f)) = tokio::time::timeout_at(deadline, ws_next_frame(ws)).await {
        out.push(f);
    }
    out
}

async fn cmd<T>(ws: &mut tokio_tungstenite::WebSocketStream<T>, c: Value) -> Value
where
    T: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let name = c["type"].as_str().unwrap().to_owned();
    ws_send(ws, c).await;
    let frames = tokio::time::timeout(Duration::from_secs(60), ws_read_until_response(ws, &name))
        .await
        .expect("response in time");
    frames.last().cloned().unwrap()
}

fn messages_of(bodies: &[String], needle: &str) -> Vec<Value> {
    strip_cache_control(request_with(bodies, needle)["messages"].clone())
        .as_array()
        .unwrap()
        .clone()
}

/// The user turn whose text is `needle`, from a request's messages.
fn turn_with<'a>(messages: &'a [Value], needle: &str) -> &'a Value {
    messages
        .iter()
        .find(|m| m["role"] == "user" && m.to_string().contains(needle))
        .unwrap_or_else(|| panic!("no turn `{needle}` in {messages:#?}"))
}

fn sidecars(dir: &std::path::Path, name: &str) -> Vec<std::path::PathBuf> {
    common::service::files_under(dir)
        .into_iter()
        .filter(|p| p.file_name().and_then(|n| n.to_str()) == Some(name))
        .collect()
}

#[tokio::test]
async fn in_service_mode_view_context_and_views_are_sealed_and_survive_a_replica_restart() {
    let fixture = spawn_http_apps_fixture();
    let (base, bodies) = spawn_model_server_routed(
        vec![
            ("Title:".into(), turn_text("title")),
            ("toolu_svc_x".into(), turn_text("done")),
            (
                "zqopenz".into(),
                turn_tool_use("toolu_svc_x", "mcp__wx__show_weather", "{}"),
            ),
        ],
        turn_text("ok"),
    );
    let mut svc = Service::start_with(
        &base,
        &["s1"],
        Options {
            extra_args: vec!["--mcp-allow-private".into()],
            ..Default::default()
        },
    )
    .await;
    let session = "s1.durable";
    let header = {
        let mut claims = svc.claims("tenant-a", session, &svc.shards[0].0);
        claims.mcp = [("wx".to_string(), fixture.url.clone())]
            .into_iter()
            .collect();
        let token = svc.minter.mint(&claims, &svc.secrets());
        svc.header(&token).map(|(k, v)| (k, v.to_owned()))
    };
    let connect = |port: u16| {
        let header = header.clone();
        async move {
            let headers: Vec<(&str, &str)> = header.iter().map(|(k, v)| (*k, v.as_str())).collect();
            common::ws_connect_with_headers(port, Some(session), &headers)
                .await
                .unwrap_or_else(|s| panic!("refused with HTTP {s}"))
        }
    };
    let mut ws = connect(svc.port).await;
    let r = cmd(
        &mut ws,
        json!({ "type": "set_mcp_apps", "id": "d", "enabled": true }),
    )
    .await;
    assert_eq!(r["success"], true, "{r}");
    let r = cmd(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "message": "zqopenz" }),
    )
    .await;
    assert_eq!(r["success"], true, "{r}");
    let r = cmd(
        &mut ws,
        json!({ "type": "mcp_app_request", "id": "c", "server": "wx", "app_id": "toolu_svc_x",
                "request": { "method": "ui/update-model-context",
                             "params": { "content": [{ "type": "text", "text": "sealed-ctx-text" }] } } }),
    )
    .await;
    assert_eq!(r["success"], true, "{r}");
    cmd(
        &mut ws,
        json!({ "type": "prompt", "id": "p2", "message": "zqbeforez" }),
    )
    .await;
    cmd(&mut ws, json!({ "type": "get_state", "id": "s" })).await;
    let before = messages_of(&bodies.lock().unwrap().clone(), "zqbeforez");
    assert!(
        turn_with(&before, "zqbeforez")
            .to_string()
            .contains("sealed-ctx-text")
    );

    // On disk: both sidecars, sealed — no plaintext — and they open only with the tenant's key.
    let sessions = svc.sessions_dir("s1", "tenant-a");
    let ctx = sidecars(&sessions, "mcp-app-context.json");
    let views = sidecars(&sessions, "mcp-app-views.json");
    assert_eq!(
        (ctx.len(), views.len()),
        (1, 1),
        "{:?}",
        common::service::files_under(&sessions)
    );
    let ctx_bytes = std::fs::read(&ctx[0]).unwrap();
    let views_bytes = std::fs::read(&views[0]).unwrap();
    for (bytes, secret) in [
        (&ctx_bytes, "sealed-ctx-text"),
        (&views_bytes, "service view"),
    ] {
        assert!(
            !String::from_utf8_lossy(bytes).contains(secret),
            "a service-mode sidecar must never hold plaintext"
        );
    }
    let codec = svc.codec("tenant-a");
    let opened = codec.open_sidecar(session, "context", &ctx_bytes).unwrap();
    assert!(String::from_utf8_lossy(&opened).contains("sealed-ctx-text"));
    assert!(
        codec.open_sidecar(session, "views", &ctx_bytes).is_err(),
        "bound to its kind"
    );
    assert!(
        codec
            .open_sidecar("s1.other", "context", &ctx_bytes)
            .is_err(),
        "bound to its session"
    );

    // Restart: a fresh replica on the same mounts.
    drop(ws);
    let _ = svc.child.kill();
    let _ = svc.child.wait();
    let peer = svc.start_peer(&["--mcp-allow-private"]);
    let mut ws = connect(peer.port).await;
    // The view comes back on attach…
    let replayed = frames_within(&mut ws, Duration::from_secs(2)).await;
    let open = of_type(&replayed, "mcp_app_open");
    assert_eq!(open.len(), 1, "{replayed:#?}");
    assert_eq!(open[0]["app_id"], "toolu_svc_x");
    assert_eq!(open[0]["replay"], true);
    // …and its context rides the same turn, byte-identically.
    cmd(
        &mut ws,
        json!({ "type": "prompt", "id": "p3", "message": "zqafterz" }),
    )
    .await;
    let after = messages_of(&bodies.lock().unwrap().clone(), "zqafterz");
    assert_eq!(
        after[..before.len()],
        before[..],
        "the restarted request must match"
    );
    svc.child = peer.child;
}

#[tokio::test]
async fn a_restarted_daemon_replays_views_and_their_bridge_still_binds() {
    let home = home_with_fixture(json!({}));
    let sessions = tempfile::tempdir().unwrap();
    let view = {
        let (base, _) = views_model(1, vec![]);
        let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
        let mut ws = daemon.connect("restart").await;
        declare_apps(&mut ws).await;
        let view = open_view(&mut ws, 1).await;
        command(&mut ws, json!({ "type": "get_state" })).await;
        view
    };
    let (base, _) = views_model(0, vec![]);
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("restart").await;
    let replayed = frames_within(&mut ws, Duration::from_secs(2)).await;
    let open = of_type(&replayed, "mcp_app_open");
    assert_eq!(
        open.len(),
        1,
        "the view must come back after a restart: {replayed:#?}"
    );
    assert_eq!(open[0]["app_id"], view.as_str());
    assert!(
        open[0]["resource"]["text"]
            .as_str()
            .unwrap()
            .contains("ui/initialize")
    );
    let result = of_type(&replayed, "mcp_app_result");
    assert_eq!(
        result[0]["result"]["structuredContent"]["secret"],
        "STRUCTURED-ONLY-Oslo"
    );
    // The renderer re-declares; the restored view's calls still bind to its server.
    declare_apps(&mut ws).await;
    let r = app_request(
        &mut ws,
        "wx",
        &view,
        "tools/call",
        json!({ "name": "refresh_weather", "arguments": { "city": "Oslo" } }),
    )
    .await;
    assert_eq!(
        r["data"]["result"]["content"][0]["text"], "refreshed:Oslo",
        "{r}"
    );
}

async fn context_turn(ws: &mut Ws, bodies: &std::sync::Mutex<Vec<String>>) -> Value {
    let view = open_view(ws, 1).await;
    let r = app_request(
        ws,
        "wx",
        &view,
        "ui/update-model-context",
        json!({ "content": [{ "type": "text", "text": "copied-ctx-text" }] }),
    )
    .await;
    assert_eq!(r["success"], true, "{r}");
    prompt(ws, "zqctxz").await;
    let messages = messages_of(&bodies.lock().unwrap().clone(), "zqctxz");
    let turn = turn_with(&messages, "zqctxz").clone();
    assert!(turn.to_string().contains("copied-ctx-text"));
    turn
}

#[tokio::test]
async fn a_fork_and_a_clone_keep_the_view_context_of_the_turns_they_copy() {
    let home = home_with_fixture(json!({}));
    let (base, bodies) = views_model(1, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("copies").await;
    declare_apps(&mut ws).await;
    let turn = context_turn(&mut ws, &bodies).await;

    // Fork at (and including) the context-carrying turn.
    let r = command(&mut ws, json!({ "type": "get_messages" })).await;
    let id = r["data"]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "user" && m.to_string().contains("zqctxz"))
        .and_then(|m| m["id"].as_str())
        .unwrap()
        .to_owned();
    let r = command(
        &mut ws,
        json!({ "type": "fork", "target_id": id, "before": false }),
    )
    .await;
    assert_eq!(r["success"], true, "{r}");
    prompt(&mut ws, "zqforkedz").await;
    let forked = messages_of(&bodies.lock().unwrap().clone(), "zqforkedz");
    assert_eq!(
        turn_with(&forked, "zqctxz"),
        &turn,
        "the fork lost the turn's view context"
    );

    // Clone the fork: still there.
    let r = command(&mut ws, json!({ "type": "clone" })).await;
    assert_eq!(r["success"], true, "{r}");
    prompt(&mut ws, "zqclonedz").await;
    let cloned = messages_of(&bodies.lock().unwrap().clone(), "zqclonedz");
    assert_eq!(
        turn_with(&cloned, "zqctxz"),
        &turn,
        "the clone lost the turn's view context"
    );
    // Never in the transcript.
    let r = command(&mut ws, json!({ "type": "get_messages" })).await;
    assert!(!r["data"].to_string().contains("copied-ctx-text"), "{r}");
}

#[tokio::test]
async fn an_unadvertised_oversized_view_is_refused_at_the_stdio_transport() {
    let home = home_with_fixture(json!({}));
    let (base, _) = views_model(
        0,
        vec![
            ("toolu_u_x".into(), turn_text("done")),
            (
                "zquz".into(),
                turn_tool_use("toolu_u_x", "mcp__wx__huge_unlisted_view", "{}"),
            ),
        ],
    );
    let sessions = tempfile::tempdir().unwrap();
    let log = sessions.path().join("daemon.log");
    let daemon = daemon_opts(
        &home.home,
        &base,
        sessions.path(),
        "0",
        &[],
        &[("RUST_LOG", "warn")],
        Some(&log),
    )
    .await;
    let mut ws = daemon.connect("stdio-cap").await;
    declare_apps(&mut ws).await;
    let frames = prompt(&mut ws, "zquz").await;
    assert_eq!(tool_end_text(&frames), "huge-unlisted-text");
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let logged = loop {
        let logged = std::fs::read_to_string(&log).unwrap_or_default();
        if logged.contains("refused an MCP server message over its size cap")
            || std::time::Instant::now() > deadline
        {
            break logged;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    // Refused by the transport, under the view cap — not read whole and refused after.
    assert!(
        logged.contains("refused an MCP server message over its size cap"),
        "{logged}"
    );
    assert!(
        logged.contains("4194304 bytes refused by the host"),
        "{logged}"
    );
    let late = frames_within(&mut ws, Duration::from_millis(500)).await;
    assert!(of_type(&late, "mcp_app_open").is_empty(), "{late:#?}");
}

#[tokio::test]
async fn an_unadvertised_oversized_view_is_refused_over_http_without_reading_its_body() {
    let fixture = spawn_http_apps_fixture();
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(home.join(".claude")).unwrap();
    std::fs::write(
        home.join(".claude/settings.json"),
        json!({ "mcp_servers": [{ "name": "wx", "transport": "http", "url": fixture.url, "headers": {} }] })
            .to_string(),
    )
    .unwrap();
    let (base, _) = spawn_model_server_routed(
        vec![
            ("Title:".into(), turn_text("title")),
            ("toolu_h_x".into(), turn_text("done")),
            (
                "zqhz".into(),
                turn_tool_use("toolu_h_x", "mcp__wx__huge_view", "{}"),
            ),
        ],
        turn_text("ok"),
    );
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("http-cap").await;
    declare_apps(&mut ws).await;
    let frames = prompt(&mut ws, "zqhz").await;
    assert_eq!(tool_end_text(&frames), "huge-view-text");
    // Give the view read (which follows the result) time to be refused and the server to stall.
    tokio::time::sleep(Duration::from_secs(5)).await;
    let late = frames_within(&mut ws, Duration::from_millis(300)).await;
    assert!(of_type(&late, "mcp_app_open").is_empty(), "{late:#?}");
    let sent = fixture.huge_bytes_sent();
    assert!(
        sent < common::mcp_apps_http::HUGE_VIEW_BYTES / 2,
        "the host read {sent} bytes of a 64 MiB view it should have refused from its Content-Length"
    );
}

#[tokio::test]
async fn a_question_raised_while_two_sessions_call_one_stdio_connection_is_refused() {
    let home = home_with_fixture(json!({}));
    let (base, _) = views_model(2, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut a = daemon.connect("caller-a").await;
    declare_apps(&mut a).await;
    let view_a = open_view(&mut a, 1).await;
    let mut b = daemon.connect("caller-b").await;
    declare_apps(&mut b).await;
    let view_b = open_view(&mut b, 2).await;

    // Session A's call is in flight on the shared apps connection…
    ws_send(
        &mut a,
        json!({ "type": "mcp_app_request", "id": "slow", "server": "wx", "app_id": view_a,
                "request": { "method": "tools/call", "params": { "name": "slow_view", "arguments": { "ms": 3000 } } } }),
    )
    .await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    // …when session B's call makes the server ask a question: nothing says whose it is.
    let r = app_request(
        &mut b,
        "wx",
        &view_b,
        "tools/call",
        json!({ "name": "ask_view" }),
    )
    .await;
    assert_eq!(r["success"], true, "{r}");
    let text = r["data"]["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        text.starts_with("asked:error"),
        "the question must be refused: {text}"
    );
    assert!(text.contains("refused"), "{text}");
    // Neither session was asked.
    let frames_b = frames_within(&mut b, Duration::from_millis(300)).await;
    assert!(
        of_type(&frames_b, "elicitation_request").is_empty(),
        "{frames_b:#?}"
    );
    let frames_a = frames_within(&mut a, Duration::from_secs(4)).await;
    assert!(
        of_type(&frames_a, "elicitation_request").is_empty(),
        "{frames_a:#?}"
    );
    let slow = frames_a
        .iter()
        .find(|f| f["type"] == "response" && f["id"] == "slow")
        .unwrap_or_else(|| panic!("A's call never finished: {frames_a:#?}"));
    assert_eq!(slow["success"], true, "{slow}");
}
