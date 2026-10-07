//! MCP Events (draft) webhook receiving under pressure and failure, end to end: the pending queue
//! is bounded in bytes (a `503` past it), a delivery that cannot be made durable is not
//! acknowledged (`503`, and it lands on the server's retry), a body that never finishes does not
//! hold the connection, and the callback survives a crash so deliveries the server retries while
//! the daemon was down still land. Real `serve --listen`, a real fixture, a mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::{Duration, Instant};

use common::mcp_events_fixture::{
    EVENTS_SESSION, control, daemon_sessions, emit, eventually, raw_request, runs_for_event,
    spawn_daemon_env, spawn_daemon_on, spawn_http_fixture, state, write_settings, ws_next,
    ws_wait_active,
};
use common::{ChildGuard, HeldPort, spawn_model_server_routed, turn_text, ws_connect, ws_send};
use serde_json::{Value, json};

fn hooks(mcp_url: &str, action: &str) -> Value {
    json!([{
        "name": "hooks", "transport": "http", "url": mcp_url,
        "headers": { "Authorization": "Bearer principal-1" },
        "events": [{ "name": "ticket.updated", "delivery": "webhook", "action": action }],
    }])
}

/// A daemon on `held`, so a restart keeps the callback URL every subscription points at.
fn daemon(home: &std::path::Path, base: &str, held: &HeldPort, env: &[(&str, &str)]) -> ChildGuard {
    spawn_daemon_on(home, base, held, &[], env)
}

/// The events session's state snapshot, once it exists.
fn state_file(home: &std::path::Path) -> std::path::PathBuf {
    eventually(Duration::from_secs(20), "the events state file", || {
        std::fs::read_dir(home.join("sessions"))
            .ok()?
            .map(|e| e.unwrap().path())
            .find(|p| p.to_string_lossy().ends_with("mcp-events.mcp-events.json"))
    })
}

/// Keep the events session busy, so follow-ups stay pending.
async fn busy(ws: &mut common::TestWs, ms: u64) {
    ws_send(
        ws,
        json!({ "type": "prompt", "id": "long", "message": beyond_ai_test_support::stall_prompt(ms) }),
    )
    .await;
    ws_next(ws, Duration::from_secs(10), "the long run's ack", |f| {
        f["type"] == "ack" && f["id"] == "long"
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_pending_queue_is_bounded_in_bytes_and_answers_503_past_it() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "follow_up"));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon_env(
        home.path(),
        &base,
        &[],
        &[("BEYOND_AI_AGENT_MCP_EVENTS_MAX_PENDING_BYTES", "8000")],
    );
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    ws_wait_active(&mut ws).await;
    busy(&mut ws, 10_000).await;
    let statuses: Vec<Value> = (0..3)
        .map(|n| {
            let r = emit(
                &fixture,
                json!({ "event_id": format!("big-{n}"), "data": { "blob": "x".repeat(3000) } }),
            );
            r["deliveries"][0]["status"].clone()
        })
        .collect();
    assert_eq!(
        statuses,
        [json!(200), json!(200), json!(503)],
        "two ~3 KB events fit 8000 bytes; the third is refused (retryable), not stored"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delivery_that_cannot_be_stored_gets_503_and_lands_on_retry() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "follow_up"));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon_env(home.path(), &base, &[], &[]);
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    ws_wait_active(&mut ws).await;
    // The pending log cannot be written: its place is taken by a directory.
    let log = state_file(home.path()).with_extension("log");
    std::fs::create_dir(&log).unwrap();
    let r = emit(
        &fixture,
        json!({ "event_id": "unstorable-1", "data": { "summary": "stored on retry" } }),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 503,
        "not durable, not acknowledged: {r:#}"
    );
    std::fs::remove_dir(&log).unwrap();
    let r = control(
        &fixture,
        "POST",
        "/control/redeliver",
        Some(&json!({ "event_id": "unstorable-1" })),
    );
    assert_eq!(r["deliveries"][0]["status"], 200, "the retry lands: {r:#}");
    eventually(Duration::from_secs(20), "the model run", || {
        (runs_for_event(&bodies, "stored on retry") >= 1).then_some(())
    });
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(runs_for_event(&bodies, "stored on retry"), 1, "once");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delivery_body_that_never_finishes_is_timed_out() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "notify"));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon_env(
        home.path(),
        &base,
        &[],
        &[("BEYOND_AI_AGENT_MCP_EVENTS_BODY_TIMEOUT_MS", "500")],
    );
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    ws_wait_active(&mut ws).await;
    let url = eventually(Duration::from_secs(10), "the hook", || {
        state(&fixture)["hooks"][0]["url"]
            .as_str()
            .map(str::to_owned)
    });
    let path = url.split_once(&format!("127.0.0.1:{port}")).unwrap().1;
    let started = Instant::now();
    let status = raw_request(
        port,
        &format!("POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: 1000\r\n\r\n"),
        b"{\"partial\":",
        Duration::from_secs(10),
    );
    assert_eq!(status, 408, "a stalled body is cut off");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
}

/// A hard crash leaves the server's subscription in place, retrying deliveries to the callback.
/// The callback (token and secret) is kept across restarts, so once the daemon is back a retried
/// delivery finds its route and verifies — instead of `410`, which tells the server to give up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_callback_survives_a_crash_so_retried_deliveries_still_land() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "follow_up"));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let held = HeldPort::bind();
    let mut first = daemon(home.path(), &base, &held, &[]);
    let url = eventually(Duration::from_secs(20), "the subscription", || {
        state(&fixture)["hooks"][0]["url"]
            .as_str()
            .map(str::to_owned)
    });
    // The callback is on disk before anything depends on it.
    eventually(Duration::from_secs(10), "the persisted callback", || {
        let saved: Value =
            serde_json::from_slice(&std::fs::read(state_file(home.path())).ok()?).ok()?;
        saved["subscriptions"]
            .as_object()?
            .values()
            .any(|s| s["webhook"]["token"].is_string())
            .then_some(())
    });
    first.kill().unwrap();
    let _ = first.wait();
    // Down, as a crashed daemon is — but still held, so the restart gets the same port.
    let down = held.down();
    // While it is down the server's delivery fails.
    let r = emit(
        &fixture,
        json!({ "event_id": "crash-1", "data": { "summary": "retried after the crash" } }),
    );
    assert_ne!(r["deliveries"][0]["status"], 200, "{r:#}");

    drop(down);
    let _second = daemon(home.path(), &base, &held, &[]);
    eventually(Duration::from_secs(20), "the resubscribe", || {
        let hooks = state(&fixture)["hooks"].as_array().unwrap().clone();
        (hooks.len() == 1 && hooks[0]["refreshes"].as_u64().unwrap_or(0) >= 1
            || state(&fixture)["methods"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|m| *m == "events/subscribe")
                .count()
                >= 2)
            .then_some(())
    });
    assert_eq!(
        state(&fixture)["hooks"][0]["url"].as_str().unwrap(),
        url,
        "the same callback after the restart"
    );
    let r = control(
        &fixture,
        "POST",
        "/control/redeliver",
        Some(&json!({ "event_id": "crash-1" })),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 200,
        "the retried delivery lands: {r:#}"
    );
    eventually(Duration::from_secs(20), "the model run", || {
        (runs_for_event(&bodies, "retried after the crash") >= 1).then_some(())
    });
}

/// After a restart, a delivery the server retries can arrive before the subscription has
/// re-registered its route (here discovery is slow), and even before the daemon has read the state
/// that names its callback (held off on demand by a debug seam; a loaded host used to open that
/// window by chance). It must get `503` — retry — not `410`, which tells the server to give up;
/// once the route is back, the retry lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_retry_that_beats_the_resubscribe_after_a_restart_is_told_to_retry_not_to_stop() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_LIST_DELAY_MS", "3000"),
    ]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url, "follow_up"));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let held = HeldPort::bind();
    let mut first = daemon(home.path(), &base, &held, &[]);
    eventually(Duration::from_secs(30), "the subscription", || {
        (state(&fixture)["hooks"].as_array().unwrap().len() == 1).then_some(())
    });
    eventually(Duration::from_secs(10), "the persisted callback", || {
        let saved: Value =
            serde_json::from_slice(&std::fs::read(state_file(home.path())).ok()?).ok()?;
        saved["subscriptions"]
            .as_object()?
            .values()
            .any(|s| s["webhook"]["token"].is_string())
            .then_some(())
    });
    first.kill().unwrap();
    let _ = first.wait();
    // Down, as a crashed daemon is — but still held, so the restart gets the same port.
    let down = held.down();
    let r = emit(
        &fixture,
        json!({ "event_id": "early-1", "data": { "summary": "retried early" } }),
    );
    assert_ne!(r["deliveries"][0]["status"], 200, "{r:#}");

    drop(down);
    let _second = daemon(
        home.path(),
        &base,
        &held,
        &[("BEYOND_AI_AGENT_TEST_SLOW_EVENTS_RESTORE_MS", "1500")],
    );
    // At once: the state is not read for 1.5 s, and discovery takes 3 s after that, so neither
    // the token's reservation nor its route exists yet. The port is held, so the delivery waits
    // in its backlog until the daemon accepts it.
    let r = control(
        &fixture,
        "POST",
        "/control/redeliver",
        Some(&json!({ "event_id": "early-1" })),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 503,
        "resuming: retry, not stop: {r:#}"
    );
    let landed = eventually(Duration::from_secs(30), "the retry to land", || {
        let r = control(
            &fixture,
            "POST",
            "/control/redeliver",
            Some(&json!({ "event_id": "early-1" })),
        );
        (r["deliveries"][0]["status"] == 200).then_some(r)
    });
    assert_eq!(landed["deliveries"][0]["status"], 200);
    eventually(Duration::from_secs(20), "the model run", || {
        (runs_for_event(&bodies, "retried early") >= 1).then_some(())
    });
    // Restored: the window is closed, and a token nothing holds is told to stop, at once.
    assert_eq!(dead_token_status(held.port()), 410);
}

/// A delivery to a callback token no subscription holds: the status the daemon answers.
fn dead_token_status(port: u16) -> u16 {
    raw_request(
        port,
        "POST /_beyond/mcp-events/0000000000000000000000000000dead HTTP/1.1\r\nHost: x\r\n\
         Content-Length: 2\r\nConnection: close\r\n\r\n",
        b"{}",
        Duration::from_secs(10),
    )
}

/// A daemon whose only subscriptions are runtime ones (`mcp_events_subscribe`) holds the restart
/// window too: a retried delivery that lands before the session holding the subscription has read
/// its state (held off by a debug seam) gets `503`, not `410`; once restored, the retry lands, and
/// a token nothing holds is `410` again at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn with_only_runtime_subscriptions_an_early_retry_is_told_to_retry_not_to_stop() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    let mut servers = hooks(&mcp_url, "notify");
    servers[0]["events"] = json!([]);
    write_settings(home.path(), servers);
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let held = HeldPort::bind();
    let port = held.port();
    let mut first = daemon(home.path(), &base, &held, &[]);
    let mut ws = ws_connect(port, Some("runtime-owner")).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks", "name": "ticket.updated",
                "delivery": "webhook", "action": "notify" }),
    )
    .await;
    let r = ws_next(&mut ws, Duration::from_secs(20), "the subscribe", |f| {
        f["type"] == "response" && f["id"] == "s"
    })
    .await;
    assert_eq!(r["success"], true, "{r:#}");
    eventually(
        Duration::from_secs(10),
        "the runtime spec and its callback on disk",
        || {
            std::fs::read_dir(home.path().join("sessions"))
                .ok()?
                .map(|e| e.unwrap().path())
                .filter(|p| p.to_string_lossy().ends_with(".mcp-events.json"))
                .any(|p| {
                    let text = std::fs::read_to_string(p).unwrap_or_default();
                    text.contains("\"runtime\"") && text.contains("\"token\"")
                })
                .then_some(())
        },
    );
    drop(ws);
    first.kill().unwrap();
    let _ = first.wait();
    let down = held.down();
    let r = emit(
        &fixture,
        json!({ "event_id": "early-1", "data": { "summary": "retried early" } }),
    );
    assert_ne!(r["deliveries"][0]["status"], 200, "{r:#}");
    drop(down);
    let _second = daemon(
        home.path(),
        &base,
        &held,
        &[("BEYOND_AI_AGENT_TEST_SLOW_EVENTS_RESTORE_MS", "1500")],
    );
    let r = control(
        &fixture,
        "POST",
        "/control/redeliver",
        Some(&json!({ "event_id": "early-1" })),
    );
    assert_eq!(
        r["deliveries"][0]["status"], 503,
        "resuming: retry, not stop: {r:#}"
    );
    eventually(Duration::from_secs(30), "the retry to land", || {
        let r = control(
            &fixture,
            "POST",
            "/control/redeliver",
            Some(&json!({ "event_id": "early-1" })),
        );
        (r["deliveries"][0]["status"] == 200).then_some(())
    });
    assert_eq!(dead_token_status(port), 410);
}

/// A runtime subscription (`mcp_events_subscribe`) survives a restart like a configured one: the
/// daemon starts the session that holds it at boot — no client has to come back — and subscribes
/// it again on its old callback, so events reach that session's model as before.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_runtime_subscription_is_restored_after_a_restart_without_its_client() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    // The server is configured, its events are not: nothing here but the runtime subscription.
    let mut servers = hooks(&mcp_url, "notify");
    servers[0]["events"] = json!([]);
    write_settings(home.path(), servers);
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let held = HeldPort::bind();
    let port = held.port();
    let mut first = daemon(home.path(), &base, &held, &[]);
    let mut ws = ws_connect(port, Some("runtime-owner")).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks", "name": "ticket.updated",
                "delivery": "webhook", "action": "follow_up" }),
    )
    .await;
    let r = ws_next(&mut ws, Duration::from_secs(20), "the subscribe", |f| {
        f["type"] == "response" && f["id"] == "s"
    })
    .await;
    assert_eq!(r["success"], true, "{r:#}");
    let url = eventually(Duration::from_secs(10), "the hook", || {
        state(&fixture)["hooks"][0]["url"]
            .as_str()
            .map(str::to_owned)
    });
    let subscribes = || {
        state(&fixture)["methods"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/subscribe")
            .count()
    };
    eventually(Duration::from_secs(10), "the runtime spec on disk", || {
        std::fs::read_dir(home.path().join("sessions"))
            .ok()?
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".mcp-events.json"))
            .any(|p| {
                std::fs::read_to_string(p)
                    .unwrap_or_default()
                    .contains("\"runtime\"")
            })
            .then_some(())
    });
    drop(ws);
    first.kill().unwrap();
    let _ = first.wait();
    let before = subscribes();
    // Down, as a crashed daemon is — but still held, so the restart gets the same port.
    let down = held.down();
    drop(down);
    let _second = daemon(home.path(), &base, &held, &[]);
    eventually(
        Duration::from_secs(20),
        "the subscription, restored with no client",
        || (subscribes() > before).then_some(()),
    );
    assert_eq!(
        state(&fixture)["hooks"][0]["url"].as_str().unwrap(),
        url,
        "on its old callback"
    );
    let r = emit(
        &fixture,
        json!({ "event_id": "after-restart-1", "data": { "summary": "restored runtime" } }),
    );
    assert_eq!(r["deliveries"][0]["status"], 200, "{r:#}");
    eventually(
        Duration::from_secs(20),
        "the owning session's model run",
        || (runs_for_event(&bodies, "restored runtime") >= 1).then_some(()),
    );
}

/// A runtime subscription whose server is gone from settings does not pin its session forever: the
/// restore after a restart is refused for good (an unknown server), the subscription is forgotten,
/// the session is reaped like any idle one, and the next restart does not bring it back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_runtime_subscription_to_a_removed_server_is_forgotten_not_resurrected() {
    let (_fx, mcp_url, _fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    let mut servers = hooks(&mcp_url, "notify");
    servers[0]["events"] = json!([]);
    write_settings(home.path(), servers.clone());
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let held = HeldPort::bind();
    let port = held.port();
    let mut first = daemon(home.path(), &base, &held, &[]);
    let mut ws = ws_connect(port, Some("orphaned")).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks", "name": "ticket.updated",
                "delivery": "webhook", "action": "notify" }),
    )
    .await;
    let r = ws_next(&mut ws, Duration::from_secs(20), "the subscribe", |f| {
        f["type"] == "response" && f["id"] == "s"
    })
    .await;
    assert_eq!(r["success"], true, "{r:#}");
    let has_runtime = |home: &std::path::Path| {
        std::fs::read_dir(home.join("sessions"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".mcp-events.json"))
            .any(|p| {
                std::fs::read_to_string(p)
                    .unwrap_or_default()
                    .contains("\"runtime\"")
            })
    };
    eventually(Duration::from_secs(10), "the runtime spec on disk", || {
        has_runtime(home.path()).then_some(())
    });
    drop(ws);
    common::mcp_events_fixture::sigterm_and_wait(&mut first);

    // The server is renamed: nothing called `hooks` exists any more.
    servers[0]["name"] = json!("hooks-renamed");
    write_settings(home.path(), servers);
    let (mut second, port) = common::mcp_events_fixture::spawn_daemon(
        home.path(),
        &base,
        &["--session-idle-timeout", "1"],
    );
    let mut reaped = false;
    for _ in 0..100 {
        if daemon_sessions(port).await.get("orphaned") == Some(&false) {
            reaped = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(reaped, "nothing keeps the session alive: it is reaped");
    assert!(
        !has_runtime(home.path()),
        "the dead subscription is forgotten"
    );
    // …and its webhook callback with it: no token or secret is left in the snapshot.
    let leftover = std::fs::read_dir(home.path().join("sessions"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".mcp-events.json"))
        .any(|p| {
            let s = std::fs::read_to_string(p).unwrap_or_default();
            s.contains("\"webhook\"") || s.contains("whsec_")
        });
    assert!(
        !leftover,
        "the forgotten subscription's token and secret are purged"
    );
    common::mcp_events_fixture::sigterm_and_wait(&mut second);

    // And a further restart does not resurrect it.
    let (_third, port) = common::mcp_events_fixture::spawn_daemon(
        home.path(),
        &base,
        &["--session-idle-timeout", "1"],
    );
    for _ in 0..15 {
        assert_ne!(
            daemon_sessions(port).await.get("orphaned"),
            Some(&true),
            "not started again at boot"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// A runtime subscription a single client command created (nothing else from that client) still
/// expires: its creation starts the time to live, so once that has passed with no client it is
/// not restored after a restart.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_runtime_subscription_made_by_one_command_expires_with_no_client() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    let mut servers = hooks(&mcp_url, "notify");
    servers[0]["events"] = json!([]);
    write_settings(home.path(), servers);
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let held = HeldPort::bind();
    let port = held.port();
    let mut first = daemon(home.path(), &base, &held, &[]);
    let mut ws = ws_connect(port, Some("one-shot")).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks", "name": "ticket.updated",
                "delivery": "webhook", "action": "notify" }),
    )
    .await;
    let r = ws_next(&mut ws, Duration::from_secs(20), "the subscribe", |f| {
        f["type"] == "response" && f["id"] == "s"
    })
    .await;
    assert_eq!(r["success"], true, "{r:#}");
    let subscribes = || {
        state(&fixture)["methods"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/subscribe")
            .count()
    };
    eventually(Duration::from_secs(10), "the runtime spec on disk", || {
        std::fs::read_dir(home.path().join("sessions"))
            .ok()?
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".mcp-events.json"))
            .any(|p| {
                std::fs::read_to_string(p)
                    .unwrap_or_default()
                    .contains("\"runtime\"")
            })
            .then_some(())
    });
    drop(ws);
    first.kill().unwrap();
    let _ = first.wait();
    let before = subscribes();
    tokio::time::sleep(Duration::from_millis(50)).await;
    let down = held.down();
    drop(down);
    let _second = daemon(
        home.path(),
        &base,
        &held,
        &[("BEYOND_AI_AGENT_MCP_EVENTS_RUNTIME_TTL_MS", "1")],
    );
    tokio::time::sleep(Duration::from_millis(3000)).await;
    assert_eq!(subscribes(), before, "expired: not restored at boot");
    assert_ne!(
        daemon_sessions(port).await.get("one-shot"),
        Some(&true),
        "and its session is not started"
    );
}

/// A session that holds runtime subscriptions and ends in a panic is started again, so they keep
/// running with no client attached.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_with_runtime_subscriptions_is_restarted_after_a_panic() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    let mut servers = hooks(&mcp_url, "notify");
    servers[0]["events"] = json!([]);
    write_settings(home.path(), servers);
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let held = HeldPort::bind();
    let port = held.port();
    let _d = daemon(
        home.path(),
        &base,
        &held,
        &[("BEYOND_AI_AGENT_TEST_PANICS", "1")],
    );
    let mut ws = ws_connect(port, Some("watcher")).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks", "name": "ticket.updated",
                "delivery": "webhook", "action": "notify" }),
    )
    .await;
    let r = ws_next(&mut ws, Duration::from_secs(20), "the subscribe", |f| {
        f["type"] == "response" && f["id"] == "s"
    })
    .await;
    assert_eq!(r["success"], true, "{r:#}");
    let subscribes = || {
        state(&fixture)["methods"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/subscribe")
            .count()
    };
    eventually(Duration::from_secs(10), "the runtime spec on disk", || {
        std::fs::read_dir(home.path().join("sessions"))
            .ok()?
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".mcp-events.json"))
            .any(|p| {
                std::fs::read_to_string(p)
                    .unwrap_or_default()
                    .contains("\"runtime\"")
            })
            .then_some(())
    });
    let before = subscribes();
    ws_send(&mut ws, json!({ "type": "__test_panic", "id": "boom" })).await;
    ws_next(&mut ws, Duration::from_secs(20), "the error frame", |f| {
        f["type"] == "error"
    })
    .await;
    drop(ws);
    eventually(
        Duration::from_secs(20),
        "the session back, resubscribed",
        || (subscribes() > before).then_some(()),
    );
    assert_eq!(daemon_sessions(port).await.get("watcher"), Some(&true));
}
