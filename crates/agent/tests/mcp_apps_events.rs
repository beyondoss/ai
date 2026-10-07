//! MCP Apps beside MCP Events: Events subscriptions stay on the **plain** connections, owned once
//! per daemon; an apps-flavored connection never subscribes and never routes `notifications/events/*`.
//! So a session that declared it renders apps changes nothing about events: one event is still one
//! model run. Real `serve --listen`, the real streamable-HTTP events fixture, a mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::mcp_events_fixture::{
    emit, eventually, runs_for_event, spawn_daemon, spawn_http_fixture, state, write_settings,
};
use common::{spawn_model_server_routed, turn_text, ws_connect, ws_read_until_response, ws_send};
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_declared_apps_session_leaves_events_on_the_plain_connection_one_run_per_event() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(
        home.path(),
        json!([{
            "name": "hooks",
            "transport": "http",
            "url": mcp_url,
            "headers": { "Authorization": "Bearer principal-1" },
            "events": [{ "name": "ticket.updated", "delivery": "webhook", "action": "follow_up" }],
        }]),
    );
    let (base, bodies) = spawn_model_server_routed(vec![], turn_text("handled"));
    let (_d, port) = spawn_daemon(home.path(), &base, &[]);

    // The events session subscribes on its own, over the plain connection.
    eventually(
        Duration::from_secs(20),
        "the events session's subscription",
        || (state(&fixture)["hooks"].as_array().unwrap().len() == 1).then_some(()),
    );

    // A renderer declares apps: the same server is dialed again, with the extension advertised.
    let mut renderer = ws_connect(port, Some("renderer")).await;
    ws_send(
        &mut renderer,
        json!({ "type": "set_mcp_apps", "id": "d", "enabled": true }),
    )
    .await;
    let frames = tokio::time::timeout(
        Duration::from_secs(30),
        ws_read_until_response(&mut renderer, "set_mcp_apps"),
    )
    .await
    .unwrap();
    let r = frames.last().unwrap();
    assert_eq!(r["success"], true, "{r}");
    assert_eq!(r["data"]["servers"], json!(["hooks"]), "{r}");
    let st = state(&fixture);
    let ui_methods: Vec<String> = st["ui_methods"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m.as_str().unwrap().to_owned())
        .collect();
    assert!(!ui_methods.is_empty(), "the apps connection exists: {st:#}");

    // One event: still exactly one model run.
    let r = emit(
        &fixture,
        json!({ "event_id": "apps-1", "data": { "summary": "with apps declared" } }),
    );
    assert_eq!(r["deliveries"].as_array().unwrap().len(), 1, "{r:#}");
    eventually(Duration::from_secs(20), "the model run", || {
        (runs_for_event(&bodies, "with apps declared") >= 1).then_some(())
    });
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(runs_for_event(&bodies, "with apps declared"), 1);

    // And the apps connection never sent an `events/*` request; the plain one did.
    let st = state(&fixture);
    let ui_methods: Vec<&str> = st["ui_methods"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m.as_str())
        .collect();
    assert!(
        ui_methods.iter().all(|m| !m.starts_with("events/")),
        "the apps connection must not touch events: {ui_methods:?}"
    );
    assert!(
        st["methods"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m.as_str().is_some_and(|m| m.starts_with("events/"))),
        "{st:#}"
    );
    assert_eq!(
        st["hooks"].as_array().unwrap().len(),
        1,
        "one subscription: {st:#}"
    );
}
