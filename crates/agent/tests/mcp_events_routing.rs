//! MCP Events (draft) in a multi-session daemon: subscriptions configured in
//! `mcp_servers[].events` are owned by **one** session — the events session, started at boot — so
//! one event is one subscription and one model run however many clients are connected, and
//! ordinary sessions stay reapable. Configured subscriptions are kept up — retried after a failure
//! or a termination, the events session kept alive meanwhile — while a session whose only
//! (runtime) subscription has terminated becomes reapable. Real `serve --listen`, a real streamable-HTTP fixture, a mock model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::time::Duration;

use common::mcp_events_fixture::{
    EVENTS_SESSION, control, daemon_sessions, emit, eventually, runs_for_event, spawn_daemon,
    spawn_http_fixture, state, write_settings, ws_list, ws_next,
};
use common::{spawn_model_server_routed, turn_text, ws_connect, ws_send};
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
    let (_d, port) = spawn_daemon(home.path(), &base, &[]);

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
    let (_d, port) = spawn_daemon(home.path(), &base, &["--session-idle-timeout", "1"]);
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

/// A runtime subscription is the session's own: once the server ends it there is nothing left to
/// keep the session alive, and the reaper takes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_session_whose_only_subscription_terminated_becomes_reapable() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    // The server is configured, its events are not: nothing is owned by the events session.
    let mut servers = hooks(&mcp_url);
    servers[0]["events"] = json!([]);
    write_settings(home.path(), servers);
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon(home.path(), &base, &["--session-idle-timeout", "1"]);
    let mut ws = ws_connect(port, Some("runtime")).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_subscribe", "id": "s", "server": "hooks",
                "name": "ticket.updated", "delivery": "webhook", "action": "notify" }),
    )
    .await;
    let r = ws_next(&mut ws, Duration::from_secs(20), "the subscribe", |f| {
        f["type"] == "response" && f["id"] == "s"
    })
    .await;
    assert_eq!(r["success"], true, "{r:#}");
    drop(ws);
    // Alive while the subscription is, detached or not.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert_eq!(daemon_sessions(port).await.get("runtime"), Some(&true));

    let r = control(&fixture, "POST", "/control/terminate", Some(&json!({})));
    assert_eq!(r["webhook_statuses"][0], 200, "{r:#}");
    let mut reaped = false;
    for _ in 0..100 {
        if daemon_sessions(port).await.get("runtime") == Some(&false) {
            reaped = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(reaped, "nothing left to keep it alive: the reaper takes it");
}

/// A configured subscription the server ends is not given up on: the events session subscribes
/// again (with backoff) and stays alive meanwhile.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_configured_subscription_the_server_ends_is_resubscribed() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon(home.path(), &base, &["--session-idle-timeout", "1"]);
    let subscribes = || {
        state(&fixture)["methods"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/subscribe")
            .count()
    };
    eventually(
        Duration::from_secs(20),
        "the events session's subscription",
        || (state(&fixture)["hooks"].as_array().unwrap().len() == 1).then_some(()),
    );
    let before = subscribes();
    let r = control(&fixture, "POST", "/control/terminate", Some(&json!({})));
    assert_eq!(r["webhook_statuses"][0], 200, "{r:#}");
    assert!(state(&fixture)["hooks"].as_array().unwrap().is_empty());
    eventually(Duration::from_secs(20), "a fresh subscription", || {
        (subscribes() > before && state(&fixture)["hooks"].as_array().unwrap().len() == 1)
            .then_some(())
    });
    assert_eq!(
        daemon_sessions(port).await.get(EVENTS_SESSION),
        Some(&true),
        "kept alive throughout"
    );
}

/// Every configured server briefly broken at boot: the subscriptions are retried until they come
/// up — not tried once and forgotten — and the events session is not reaped meanwhile.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn configured_subscriptions_that_fail_at_boot_are_retried_and_kept_alive() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_EVENTS_DOWN", "1"),
    ]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon(home.path(), &base, &["--session-idle-timeout", "1"]);
    tokio::time::sleep(Duration::from_millis(4000)).await;
    assert!(state(&fixture)["hooks"].as_array().unwrap().is_empty());
    assert_eq!(
        daemon_sessions(port).await.get(EVENTS_SESSION),
        Some(&true),
        "a session whose subscriptions are all down for now is not reaped"
    );
    control(
        &fixture,
        "POST",
        "/control/events_down",
        Some(&json!({ "down": false })),
    );
    eventually(
        Duration::from_secs(30),
        "the subscription, once the server is back",
        || (state(&fixture)["hooks"].as_array().unwrap().len() == 1).then_some(()),
    );
}

/// An explicit unsubscribe of a configured subscription that is down — being retried — ends the
/// retrying: when the server comes back, nothing subscribes behind the operator's back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unsubscribing_a_configured_subscription_while_its_server_is_down_stops_the_retries() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[
        ("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1"),
        ("MCP_FIXTURE_EVENTS_DOWN", "1"),
    ]);
    let home = tempfile::tempdir().unwrap();
    write_settings(home.path(), hooks(&mcp_url));
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let (_d, port) = spawn_daemon(home.path(), &base, &[]);
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    // Let the first attempt fail.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    ws_send(
        &mut ws,
        json!({ "type": "mcp_events_unsubscribe", "id": "u", "server": "hooks", "name": "ticket.updated" }),
    )
    .await;
    let r = ws_next(&mut ws, Duration::from_secs(20), "the unsubscribe", |f| {
        f["type"] == "response" && f["id"] == "u"
    })
    .await;
    assert_eq!(r["success"], true, "{r:#}");
    control(
        &fixture,
        "POST",
        "/control/events_down",
        Some(&json!({ "down": false })),
    );
    // Longer than the retry backoff at this point (2–4 s).
    tokio::time::sleep(Duration::from_millis(7000)).await;
    assert!(
        state(&fixture)["hooks"].as_array().unwrap().is_empty(),
        "unsubscribed, so not resubscribed: {:#}",
        state(&fixture)
    );
}

/// A configured subscription the server refuses for good (here: an event it does not offer) is
/// not hammered every minute forever: it is reported as `refused` — in `mcp_events_list` and as an
/// `mcp_event_status` frame — retried only rarely, and it does not keep the events session alive.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_permanently_refused_configured_subscription_is_reported_and_does_not_keep_the_session() {
    let (_fx, mcp_url, fixture) = spawn_http_fixture(&[("MCP_FIXTURE_ALLOW_HTTP_CALLBACK", "1")]);
    let home = tempfile::tempdir().unwrap();
    let mut servers = hooks(&mcp_url);
    servers[0]["events"] = json!([{ "name": "no.such.event", "delivery": "webhook" }]);
    write_settings(home.path(), servers);
    let (base, _bodies) = spawn_model_server_routed(vec![], turn_text("noted"));
    let port = free_port();
    let _d = spawn_daemon(home.path(), &base, port, &["--session-idle-timeout", "1"]);
    let mut ws = ws_connect(port, Some(EVENTS_SESSION)).await;
    let status = ws_next(&mut ws, Duration::from_secs(20), "the refusal", |f| {
        f["type"] == "mcp_event_status" && f["kind"] == "refused"
    })
    .await;
    assert!(
        status["error"]
            .as_str()
            .unwrap()
            .contains("offers no event"),
        "{status:#}"
    );
    let l = ws_list(&mut ws, "l").await;
    assert_eq!(l["data"]["unestablished"][0]["state"], "refused", "{l:#}");
    assert_eq!(l["data"]["unestablished"][0]["name"], "no.such.event");
    let lists = || {
        state(&fixture)["methods"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| *m == "events/list")
            .count()
    };
    let before = lists();
    tokio::time::sleep(Duration::from_millis(5000)).await;
    assert!(
        lists() <= before + 1,
        "a permanent refusal is not retried on the transient backoff"
    );
    // Nothing else keeps the events session: once detached, the reaper takes it.
    drop(ws);
    let mut reaped = false;
    for _ in 0..100 {
        if daemon_sessions(port).await.get(EVENTS_SESSION) == Some(&false) {
            reaped = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    assert!(
        reaped,
        "a refused subscription does not keep its session alive"
    );
}
