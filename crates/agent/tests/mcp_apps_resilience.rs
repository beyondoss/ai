//! MCP Apps when things go wrong or move — each against the real `serve` daemon:
//! - a failed apps dial falls back to the plain tools, is surfaced, and is retried until it lands;
//!   declaring never blocks the session's command loop on a dial;
//! - bridge requests are bounded per session and cancelled by `abort`, by withdrawing apps, by every
//!   client detaching, and by a deadline;
//! - views survive a reconnect even behind a turn replay longer than the connection's buffer;
//! - the model's result never waits for its view;
//! - a session switch (`new_session`/`switch_session`/`fork`/`clone`) tears views down and empties
//!   the replay store;
//! - service mode dials a session's own apps pool from its grant.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use common::mcp_apps::{
    Ws, app_request, command, daemon, daemon_opts, declare_apps, home_with_fixture, of_type,
    open_view, prompt, tool_end_text, view_id, views_model,
};
use common::mcp_apps_http::spawn_http_apps_fixture;
use common::service::{Options, Service};
use common::{
    spawn_model_server, spawn_model_server_routed, sse, turn_text, turn_tool_use, ws_next_frame,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

async fn next_frame(ws: &mut Ws) -> Value {
    tokio::time::timeout(Duration::from_secs(30), ws_next_frame(ws))
        .await
        .expect("a frame in time")
        .expect("socket open")
}

/// Frames that arrive within `window` (none is fine).
async fn frames_within(ws: &mut Ws, window: Duration) -> Vec<Value> {
    let mut out = Vec::new();
    let deadline = tokio::time::Instant::now() + window;
    while let Ok(Some(f)) = tokio::time::timeout_at(deadline, ws_next_frame(ws)).await {
        out.push(f);
    }
    out
}

fn slow_call(id: &str, app_id: &str, ms: u64) -> Value {
    json!({
        "type": "mcp_app_request", "id": id, "server": "wx", "app_id": app_id,
        "request": { "method": "tools/call", "params": { "name": "slow_view", "arguments": { "ms": ms } } },
    })
}

#[tokio::test]
async fn a_failed_apps_dial_falls_back_to_plain_tools_is_surfaced_and_is_retried() {
    let marker = tempfile::tempdir().unwrap();
    let marker = marker.path().join("failed-once");
    let home = home_with_fixture(json!({ "MCP_APPS_FIXTURE_FAIL_FIRST_UI": marker }));
    let (base, _) = views_model(2, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("retry").await;

    // The first apps dial dies: the response says which server and why, rather than only a log.
    let r = command(&mut ws, json!({ "type": "set_mcp_apps", "enabled": true })).await;
    assert_eq!(r["success"], true, "{r}");
    assert_eq!(r["data"]["servers"], json!([]), "{r}");
    assert_eq!(r["data"]["failed"][0]["server"], "wx", "{r}");
    assert!(r["data"]["failed"][0]["error"].is_string(), "{r}");
    assert!(marker.exists());

    // Meanwhile the server's plain tools still work — it did not vanish from the session.
    let frames = prompt(&mut ws, "zqopen1z").await;
    assert!(of_type(&frames, "mcp_app_open").is_empty(), "{frames:#?}");
    assert_eq!(tool_end_text(&frames), "weather-text:Oslo");

    // The retry (backoff starts at 1s) lands, and the next prompt uses the apps flavor.
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let r = command(&mut ws, json!({ "type": "get_mcp" })).await;
        if r["data"]["apps"]["servers"] == json!(["wx"]) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the apps dial was never retried: {r}"
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    open_view(&mut ws, 2).await;
}

#[tokio::test]
async fn declaring_apps_never_blocks_the_command_loop_on_the_dial() {
    let home = home_with_fixture(json!({ "MCP_APPS_FIXTURE_UI_DELAY_MS": "3000" }));
    let (base, _) = spawn_model_server(vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("slowdial").await;

    let started = Instant::now();
    ws_send(
        &mut ws,
        json!({ "type": "set_mcp_apps", "id": "d", "enabled": true }),
    )
    .await;
    ws_send(&mut ws, json!({ "type": "get_state", "id": "s" })).await;
    let mut order = Vec::new();
    while order.len() < 2 {
        let f = next_frame(&mut ws).await;
        if f["type"] == "response" {
            order.push((f["command"].as_str().unwrap().to_owned(), started.elapsed()));
        }
    }
    assert_eq!(order[0].0, "get_state", "{order:?}");
    assert!(order[0].1 < Duration::from_millis(2500), "{order:?}");
    assert_eq!(order[1].0, "set_mcp_apps", "{order:?}");
}

#[tokio::test]
async fn bridge_requests_are_bounded_and_cancelled_by_abort_detach_and_withdrawal() {
    let home = home_with_fixture(json!({}));
    let (base, _) = views_model(1, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("bound").await;
    declare_apps(&mut ws).await;
    let view = open_view(&mut ws, 1).await;

    // Eight in flight is the most; the ninth is refused at once.
    for n in 1..=9 {
        ws_send(&mut ws, slow_call(&format!("b{n}"), &view, 60_000)).await;
    }
    let refused = loop {
        let f = next_frame(&mut ws).await;
        if f["type"] == "response" && f["command"] == "mcp_app_request" {
            break f;
        }
    };
    assert_eq!(refused["success"], false, "{refused}");
    assert!(
        refused["error"].as_str().unwrap().contains("too many"),
        "{refused}"
    );

    // Every client detaches: the eight are cancelled, so a reconnecting client has all eight slots.
    drop(ws);
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let mut ws = daemon.connect("bound").await;
    for n in 1..=8 {
        ws_send(&mut ws, slow_call(&format!("c{n}"), &view, 60_000)).await;
    }
    let early = frames_within(&mut ws, Duration::from_millis(800)).await;
    assert!(
        !early
            .iter()
            .any(|f| f["command"] == "mcp_app_request" && f["success"] == false),
        "detached requests still held their slots: {early:#?}"
    );

    // `abort` cancels them all, promptly.
    let started = Instant::now();
    ws_send(&mut ws, json!({ "type": "abort", "id": "x" })).await;
    let mut cancelled = 0;
    while cancelled < 8 {
        let f = next_frame(&mut ws).await;
        if f["type"] == "response" && f["command"] == "mcp_app_request" {
            assert_eq!(f["success"], false, "{f}");
            assert!(f["error"].as_str().unwrap().contains("cancelled"), "{f}");
            cancelled += 1;
        }
    }
    assert!(started.elapsed() < Duration::from_secs(5));

    // Withdrawing apps cancels one in flight, too.
    ws_send(&mut ws, slow_call("w1", &view, 60_000)).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    ws_send(
        &mut ws,
        json!({ "type": "set_mcp_apps", "id": "off", "enabled": false }),
    )
    .await;
    loop {
        let f = next_frame(&mut ws).await;
        if f["type"] == "response" && f["command"] == "mcp_app_request" {
            assert!(f["error"].as_str().unwrap().contains("cancelled"), "{f}");
            break;
        }
    }
}

#[tokio::test]
async fn a_bridge_request_times_out() {
    let home = home_with_fixture(json!({}));
    let (base, _) = views_model(1, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon_opts(
        &home.home,
        &base,
        sessions.path(),
        "0",
        &[],
        &[("BEYOND_AI_AGENT_MCP_APP_REQUEST_TIMEOUT_MS", "500")],
        None,
    )
    .await;
    let mut ws = daemon.connect("deadline").await;
    declare_apps(&mut ws).await;
    let view = open_view(&mut ws, 1).await;
    let started = Instant::now();
    let r = command(&mut ws, slow_call("t", &view, 30_000)).await;
    assert_eq!(r["success"], false, "{r}");
    assert!(r["error"].as_str().unwrap().contains("timed out"), "{r}");
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn views_are_replayed_even_behind_a_turn_longer_than_the_connections_buffer() {
    let home = home_with_fixture(json!({}));
    // One assistant message: 2000 text deltas (each its own frame) and then a slow view.
    let mut events = vec![
        json!({ "type": "message_start", "message": { "usage": { "input_tokens": 1, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
    ];
    for _ in 0..2000 {
        events.push(json!({ "type": "content_block_delta", "index": 0,
                            "delta": { "type": "text_delta", "text": "w " } }));
    }
    events.extend([
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "content_block_start", "index": 1, "content_block":
                { "type": "tool_use", "id": "toolu_long_x", "name": "mcp__wx__slow_view", "input": {} } }),
        json!({ "type": "content_block_delta", "index": 1, "delta":
                { "type": "input_json_delta", "partial_json": "{\"ms\":4000}" } }),
        json!({ "type": "content_block_stop", "index": 1 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 1 } }),
        json!({ "type": "message_stop" }),
    ]);
    let (base, _) = spawn_model_server_routed(
        vec![
            ("Title:".into(), turn_text("title")),
            ("toolu_long_x".into(), turn_text("done")),
        ],
        sse(&events),
    );
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut a = daemon.connect("long").await;
    declare_apps(&mut a).await;
    ws_send(&mut a, json!({ "type": "prompt", "message": "go" })).await;
    // Not read: this client may well be pruned as a slow consumer by a 2000-frame burst. The view
    // opens within moments of the tool starting, and the tool runs for 4s.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    // A second client attaches mid-turn: the 2000+ turn frames overflow its buffer, but the view
    // was sent on capacity reserved before them — so the view is never the frame that gets dropped.
    let mut b = daemon.connect("long").await;
    let frames = frames_within(&mut b, Duration::from_secs(3)).await;
    let open = frames
        .iter()
        .position(|f| f["type"] == "mcp_app_open")
        .unwrap_or_else(|| panic!("the view was lost on attach ({} frames)", frames.len()));
    assert_eq!(frames[open]["replay"], true);
    assert_eq!(frames[open]["app_id"], "toolu_long_x");
    // In its place: after its own `tool_start`, if that frame survived the overflow at all.
    let tool_start = frames
        .iter()
        .position(|f| f["event"]["kind"] == "tool_start" && f["event"]["id"] == "toolu_long_x");
    assert!(
        tool_start.is_none_or(|t| t < open),
        "{tool_start:?} vs {open}"
    );
}

#[tokio::test]
async fn the_models_result_never_waits_for_its_view() {
    let home = home_with_fixture(json!({ "MCP_APPS_FIXTURE_VIEW_DELAY_MS": "2500" }));
    let (base, _) = views_model(1, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("late").await;
    declare_apps(&mut ws).await;

    let started = Instant::now();
    let mut frames = prompt(&mut ws, "zqopen1z").await;
    let ran_for = started.elapsed();
    assert!(
        ran_for < Duration::from_millis(2500),
        "the run waited for the view: {ran_for:?}"
    );
    assert_eq!(tool_end_text(&frames), "weather-text:Oslo");
    // The view follows the result: open, then its result.
    while of_type(&frames, "mcp_app_result").is_empty() {
        frames.push(next_frame(&mut ws).await);
    }
    let pos = |ty: &str| frames.iter().position(|f| f["type"] == ty).unwrap();
    let tool_end = frames
        .iter()
        .position(|f| f["event"]["kind"] == "tool_end")
        .unwrap();
    assert!(tool_end < pos("mcp_app_open"), "{frames:#?}");
    assert!(pos("mcp_app_open") < pos("mcp_app_result"));
    assert_eq!(frames[pos("mcp_app_open")]["app_id"], view_id(1));
}

#[tokio::test]
async fn a_session_switch_tears_views_down_and_empties_the_replay_store() {
    let home = home_with_fixture(json!({}));
    let (base, _) = views_model(1, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    // A session to switch to.
    let mut target = daemon.connect("target").await;
    prompt(&mut target, "hello").await;
    drop(target);

    for cmd in [
        json!({ "type": "new_session", "id": "c" }),
        json!({ "type": "fork", "id": "c" }),
        json!({ "type": "clone", "id": "c" }),
        json!({ "type": "switch_session", "id": "c", "session_id": "target" }),
    ] {
        let name = cmd["type"].as_str().unwrap().to_owned();
        let id = format!("switch-{name}");
        let mut ws = daemon.connect(&id).await;
        declare_apps(&mut ws).await;
        let view = open_view(&mut ws, 1).await;

        ws_send(&mut ws, cmd).await;
        let frames = tokio::time::timeout(
            Duration::from_secs(30),
            ws_read_until_response(&mut ws, &name),
        )
        .await
        .unwrap();
        assert_eq!(
            frames.last().unwrap()["success"],
            true,
            "{name}: {frames:#?}"
        );
        let teardown = of_type(&frames, "mcp_app_teardown");
        assert_eq!(teardown.len(), 1, "{name}: {frames:#?}");
        assert_eq!(teardown[0]["app_id"], view.as_str());
        // The apps state is gone with the session…
        let r = app_request(&mut ws, "wx", &view, "ping", json!({})).await;
        assert_eq!(r["success"], false, "{name}: {r}");
        let r = command(&mut ws, json!({ "type": "get_mcp" })).await;
        assert!(r["data"]["apps"].is_null(), "{name}: {r}");
        // …and so is the replay store: a client attaching now reopens nothing.
        drop(ws);
        let mut again = daemon.connect(&id).await;
        let replayed = frames_within(&mut again, Duration::from_millis(500)).await;
        assert!(
            of_type(&replayed, "mcp_app_open").is_empty(),
            "{name}: {replayed:#?}"
        );
    }
}

#[tokio::test]
async fn service_mode_dials_a_sessions_own_apps_pool_from_its_grant() {
    let fixture = spawn_http_apps_fixture();
    let (base, _) = spawn_model_server_routed(
        vec![
            ("Title:".into(), turn_text("title")),
            ("toolu_svc_x".into(), turn_text("done")),
        ],
        turn_tool_use("toolu_svc_x", "mcp__wx__show_weather", "{}"),
    );
    // The replica takes a TCP port picked by `free_port` — another test can grab it first, so retry
    // a start whose port turned out not to be ours.
    let mut attempt = 0;
    let (_svc, mut ws) = loop {
        attempt += 1;
        let svc = Service::start_with(
            &base,
            &["s1"],
            Options {
                extra_args: vec!["--mcp-allow-private".into()],
                ..Default::default()
            },
        )
        .await;
        let mut claims = svc.claims("tenant-a", "s1.apps", &svc.shards[0].0);
        claims.mcp = [("wx".to_string(), fixture.url.clone())]
            .into_iter()
            .collect();
        let token = svc.minter.mint(&claims, &svc.secrets());
        match tokio_tungstenite::connect_async(common::ws_request(
            svc.port,
            Some("s1.apps"),
            &svc.header(&token),
        ))
        .await
        {
            Ok((ws, _)) => break (svc, ws),
            Err(e) if attempt < 3 => eprintln!("replica start {attempt} unusable: {e}"),
            Err(e) => panic!("replica never accepted a session: {e}"),
        }
    };

    // The session's own connector is dialed again, with the extension advertised.
    assert_eq!(
        fixture.ui_requests(),
        0,
        "nothing advertised before the client declares"
    );
    ws_send(
        &mut ws,
        json!({ "type": "set_mcp_apps", "id": "d", "enabled": true }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "set_mcp_apps").await;
    let r = frames.last().unwrap();
    assert_eq!(r["success"], true, "{r}");
    assert_eq!(r["data"]["servers"], json!(["wx"]), "{r}");
    assert!(fixture.ui_requests() > 0);
    assert!(
        fixture.plain_requests() > 0,
        "the plain connection stays plain"
    );

    // And its view opens through it.
    ws_send(
        &mut ws,
        json!({ "type": "prompt", "id": "p", "message": "go" }),
    )
    .await;
    let frames = ws_read_until_response(&mut ws, "prompt").await;
    let open = of_type(&frames, "mcp_app_open");
    assert_eq!(open.len(), 1, "{frames:#?}");
    assert_eq!(open[0]["app_id"], "toolu_svc_x");
    assert_eq!(open[0]["resource"]["uri"], "ui://apps-http/weather");
    let result = of_type(&frames, "mcp_app_result");
    assert_eq!(result[0]["result"]["structuredContent"]["where"], "service");
}

#[tokio::test]
async fn a_withdrawn_declaration_never_answers_enabled_after_the_withdrawal() {
    let home = home_with_fixture(json!({ "MCP_APPS_FIXTURE_UI_DELAY_MS": "1500" }));
    let (base, _) = spawn_model_server(vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("race").await;

    // Declare (the slow dial holds its answer), then withdraw before it lands.
    ws_send(
        &mut ws,
        json!({ "type": "set_mcp_apps", "id": "on", "enabled": true }),
    )
    .await;
    ws_send(
        &mut ws,
        json!({ "type": "set_mcp_apps", "id": "off", "enabled": false }),
    )
    .await;
    let mut answers = Vec::new();
    while answers.len() < 2 {
        let f = next_frame(&mut ws).await;
        if f["type"] == "response" && f["command"] == "set_mcp_apps" {
            answers.push(f);
        }
    }
    let off = answers.iter().position(|a| a["id"] == "off").unwrap();
    let on = answers.iter().position(|a| a["id"] == "on").unwrap();
    assert_eq!(answers[off]["data"]["enabled"], false, "{answers:#?}");
    // The superseded declaration never claims to be enabled — wherever its answer lands.
    assert!(
        answers[on]["data"]["enabled"] != true,
        "a withdrawn declaration answered enabled: {answers:#?}"
    );
    assert_eq!(answers[on]["success"], false, "{answers:#?}");
    assert!(
        answers[on]["error"]
            .as_str()
            .unwrap()
            .contains("superseded")
    );
    let r = command(&mut ws, json!({ "type": "get_mcp" })).await;
    assert!(r["data"]["apps"].is_null(), "{r}");
}

#[tokio::test]
async fn abort_during_a_run_cancels_bridge_requests() {
    let home = home_with_fixture(json!({}));
    let (base, _) = views_model(
        1,
        vec![
            ("toolu_busy_x".into(), turn_text("done")),
            (
                "zqbusyz".into(),
                turn_tool_use(
                    "toolu_busy_x",
                    "mcp__wx__slow_view",
                    &json!({ "ms": 30_000 }).to_string(),
                ),
            ),
        ],
    );
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("busy").await;
    declare_apps(&mut ws).await;
    let view = open_view(&mut ws, 1).await;

    // A run is in flight (its tool sleeps), and a view's bridge request is pending too.
    ws_send(&mut ws, json!({ "type": "prompt", "message": "zqbusyz" })).await;
    loop {
        let f = next_frame(&mut ws).await;
        if f["event"]["kind"] == "tool_start" {
            break;
        }
    }
    ws_send(&mut ws, slow_call("pending", &view, 60_000)).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let started = Instant::now();
    ws_send(&mut ws, json!({ "type": "abort", "id": "a" })).await;
    let cancelled = loop {
        let f = next_frame(&mut ws).await;
        if f["type"] == "response" && f["command"] == "mcp_app_request" {
            break f;
        }
    };
    assert_eq!(cancelled["success"], false, "{cancelled}");
    assert!(
        cancelled["error"].as_str().unwrap().contains("cancelled"),
        "{cancelled}"
    );
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn an_oversized_view_is_refused_unread_when_advertised_and_never_cached_when_not() {
    let home = home_with_fixture(json!({}));
    let routes = vec![
        ("toolu_u2_x".into(), turn_text("done")),
        (
            "zqu2z".into(),
            turn_tool_use("toolu_u2_x", "mcp__wx__huge_unlisted_view", "{}"),
        ),
        ("toolu_u1_x".into(), turn_text("done")),
        (
            "zqu1z".into(),
            turn_tool_use("toolu_u1_x", "mcp__wx__huge_unlisted_view", "{}"),
        ),
        ("toolu_h_x".into(), turn_text("done")),
        (
            "zqhz".into(),
            turn_tool_use("toolu_h_x", "mcp__wx__huge_view", "{}"),
        ),
    ];
    let (base, _) = views_model(0, routes);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("huge").await;
    declare_apps(&mut ws).await;
    let settle = Duration::from_millis(1500);

    // Advertised at 5 MiB: refused without a single read.
    let frames = prompt(&mut ws, "zqhz").await;
    assert_eq!(tool_end_text(&frames), "huge-view-text");
    tokio::time::sleep(settle).await;
    assert_eq!(
        home.reads_of("ui://apps-fixture/huge"),
        0,
        "{:?}",
        home.requests()
    );

    // Unadvertised: read to find out, refused, and never cached — the second call reads again.
    for p in ["zqu1z", "zqu2z"] {
        let frames = prompt(&mut ws, p).await;
        assert_eq!(tool_end_text(&frames), "huge-unlisted-text");
        assert!(of_type(&frames, "mcp_app_open").is_empty());
        tokio::time::sleep(settle).await;
    }
    assert_eq!(home.reads_of("ui://apps-fixture/huge-unlisted"), 2);
}

#[tokio::test]
async fn redials_stop_once_no_session_wants_apps() {
    let home = home_with_fixture(json!({ "MCP_APPS_FIXTURE_FAIL_UI": "1" }));
    let (base, _) = spawn_model_server(vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("noretry").await;
    let attempts = |home: &common::mcp_apps::Home| {
        home.requests()
            .iter()
            .filter(|(m, ui)| *ui && (m == "server/discover" || m == "initialize"))
            .count()
    };

    let r = command(&mut ws, json!({ "type": "set_mcp_apps", "enabled": true })).await;
    assert_eq!(r["data"]["failed"][0]["server"], "wx", "{r}");
    // The first retry comes after 1s: the session still wants apps.
    let deadline = Instant::now() + Duration::from_secs(10);
    while attempts(&home) < 2 {
        assert!(Instant::now() < deadline, "never retried");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    // Withdrawn: nobody wants apps, so the next redial (due ~2s later) never happens.
    command(&mut ws, json!({ "type": "set_mcp_apps", "enabled": false })).await;
    let after_withdrawal = attempts(&home);
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(attempts(&home), after_withdrawal, "{:?}", home.requests());

    // Declaring again redials at once.
    let r = command(&mut ws, json!({ "type": "set_mcp_apps", "enabled": true })).await;
    assert_eq!(r["success"], true, "{r}");
    assert!(attempts(&home) > after_withdrawal);
}

#[tokio::test]
async fn a_restarted_daemon_hides_app_only_tools_from_the_cached_manifest() {
    let home = home_with_fixture(json!({}));
    let sessions = tempfile::tempdir().unwrap();
    {
        let (base, _) = spawn_model_server(vec![]);
        let _first = daemon(&home.home, &base, sessions.path(), "0").await;
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(home.home.join(".claude/mcp-manifest.json")).unwrap(),
    )
    .unwrap();
    assert!(
        manifest["wx"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["remote_name"] == "app_only_always"
                && t["ui"]["visibility"] == json!(["app"])),
        "{manifest:#}"
    );
    std::fs::remove_file(&home.log).unwrap();

    // Second boot: tools come from the plain manifest, no server starts, and the app-only tool
    // is still hidden from the model.
    let (base, bodies) = spawn_model_server(vec![turn_text("hi"), turn_text("title")]);
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("cached").await;
    prompt(&mut ws, "hello").await;
    assert!(home.requests().is_empty(), "{:?}", home.requests());
    let tools = common::advertised_tools(&bodies.lock().unwrap()[0]);
    assert!(
        tools.iter().any(|t| t == "mcp__wx__show_weather"),
        "{tools:?}"
    );
    assert!(
        !tools.iter().any(|t| t == "mcp__wx__app_only_always"),
        "{tools:?}"
    );
}

#[tokio::test]
async fn a_view_closed_while_loading_emits_nothing_after_its_teardown() {
    let home = home_with_fixture(json!({ "MCP_APPS_FIXTURE_VIEW_DELAY_MS": "2000" }));
    let (base, _) = views_model(1, vec![]);
    let sessions = tempfile::tempdir().unwrap();
    let daemon = daemon(&home.home, &base, sessions.path(), "0").await;
    let mut ws = daemon.connect("late-close").await;
    declare_apps(&mut ws).await;

    // The tool answers at once; its view is still loading (2s) when the run ends.
    let frames = prompt(&mut ws, "zqopen1z").await;
    assert_eq!(tool_end_text(&frames), "weather-text:Oslo");
    assert!(of_type(&frames, "mcp_app_open").is_empty(), "{frames:#?}");
    // The client withdraws before the view lands: nothing of that view may follow.
    let r = command(&mut ws, json!({ "type": "set_mcp_apps", "enabled": false })).await;
    assert_eq!(r["data"]["enabled"], false, "{r}");
    let later = frames_within(&mut ws, Duration::from_secs(4)).await;
    assert!(
        !later
            .iter()
            .any(|f| f["type"] == "mcp_app_open" || f["type"] == "mcp_app_result"),
        "a closed view emitted frames: {later:#?}"
    );
}
