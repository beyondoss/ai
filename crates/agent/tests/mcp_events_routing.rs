//! MCP Events (draft) in a multi-session daemon: subscriptions configured in
//! `mcp_servers[].events` are owned by **one** session — the events session, started at boot — so
//! one event is one subscription and one model run however many clients are connected, and
//! ordinary sessions stay reapable. A session whose only subscription has terminated becomes
//! reapable too. Real `serve --listen`, a real streamable-HTTP fixture, a mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::mcp_events_fixture::{
    EVENTS_SESSION, control, daemon_sessions, emit, eventually, runs_for_event, spawn_daemon,
    spawn_http_fixture, state, write_settings, ws_list,
};
use common::{free_port, spawn_model_server_routed, turn_text, ws_connect};
use serde_json::json;

fn hooks(mcp_url: &str) -> serde_json::Value {
    json!([{
        "name": "hooks",
        "transport": "http",
        "url": mcp_url,
        "headers": { "Authorization": "Bearer principal-1" },
        "events": [{ "name": "ticket.updated", "delivery": "webhook", "action": "follow_up" }],
    }])
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn configured_subscriptions_are_owned_by_one_session_and_one_event_runs_the_model_once() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url));
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("handled"));
    let port = free_port();
    let _d = spawn_daemon(home.path(), &base, port, &[]);

    // Two ordinary clients, connected before the event.
    let mut a = ws_connect(port, Some("plain-a")).await;
    let mut b = ws_connect(port, Some("plain-b")).await;
    for (ws, id) in [(&mut a, "a"), (&mut b, "b")] {
        let l = ws_list(ws, id).await;
        assert_eq!(l["data"]["owns_configured"], false, "{l:#}");
        assert!(
            l["data"]["subscriptions"].as_array().unwrap().is_empty(),
            "{l:#}"
        );
    }
    // The events session subscribed on its own — no client attached to it.
    let st = eventually(
        Duration::from_secs(20),
        "the events session's subscription",
        || {
            let st = state(&fixture);
            (st["hooks"].as_array().unwrap().len() == 1).then_some(st)
        },
    );
    assert_eq!(
        st["hooks"].as_array().unwrap().len(),
        1,
        "one subscription, not one per session"
    );

    let r = emit(
        &fixture,
        json!({ "event_id": "solo-1", "data": { "summary": "routed once" } }),
    );
    assert_eq!(r["deliveries"].as_array().unwrap().len(), 1, "{r:#}");
    assert_eq!(r["deliveries"][0]["status"], 200, "{r:#}");
    eventually(Duration::from_secs(20), "the model run", || {
        (runs_for_event(&bodies, "routed once") >= 1).then_some(())
    });
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(
        runs_for_event(&bodies, "routed once"),
        1,
        "exactly one model run for one event"
    );

    // And it ran in the events session: a client attaching there sees it.
    let mut ev = ws_connect(port, Some(EVENTS_SESSION)).await;
    let catchup =
        common::mcp_events_fixture::ws_next(&mut ev, Duration::from_secs(20), "catchup", |f| {
            f["type"] == "catchup"
        })
        .await;
    assert!(
        catchup["data"]["messages"]
            .to_string()
            .contains("routed once")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plain_sessions_stay_reapable_while_the_events_session_is_kept_alive() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let _d = spawn_daemon(home.path(), &base, port, &["--session-idle-timeout", "1"]);
    eventually(
        Duration::from_secs(20),
        "the events session's subscription",
        || (state(&fixture)["hooks"].as_array().unwrap().len() == 1).then_some(()),
    );
    let mut plain = ws_connect(port, Some("plain-reap")).await;
    ws_list(&mut plain, "p").await;
    drop(plain);

    let mut live = daemon_sessions(port).await;
    for _ in 0..100 {
        if live.get("plain-reap") == Some(&false) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        live = daemon_sessions(port).await;
    }
    assert_eq!(
        live.get("plain-reap"),
        Some(&false),
        "the plain session is reaped: {live:?}"
    );
    assert_eq!(
        live.get(EVENTS_SESSION),
        Some(&true),
        "the events session is not reaped: {live:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_whose_only_subscription_terminated_becomes_reapable() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let _d = spawn_daemon(home.path(), &base, port, &["--session-idle-timeout", "1"]);
    eventually(
        Duration::from_secs(20),
        "the events session's subscription",
        || (state(&fixture)["hooks"].as_array().unwrap().len() == 1).then_some(()),
    );
    // Alive while the subscription is.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert_eq!(daemon_sessions(port).await.get(EVENTS_SESSION), Some(&true));

    let r = control(&fixture, "POST", "/control/terminate", Some(&json!({})));
    assert_eq!(r["webhook_statuses"][0], 200, "{r:#}");
    let mut reaped = false;
    for _ in 0..100 {
        if daemon_sessions(port).await.get(EVENTS_SESSION) == Some(&false) {
            reaped = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(reaped, "nothing left to keep it alive: the reaper takes it");
}
