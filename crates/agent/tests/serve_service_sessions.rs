//! Session listings, cross-shard routing, and command gating under load.
//!
//! One replica serves many tenants from several mounts. Two properties have to hold together: a
//! tenant sees **all** of its own sessions, wherever they live, and **only** its own. And a command
//! service mode refuses has to be refused *before* the busy-loop arm that would self-abort a running
//! prompt to make room for it — otherwise a client could cancel its own run by asking for something
//! it was never going to get.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::service::Service;
use common::{
    spawn_model_server, spawn_model_server_with_stalled_response, ws_connect_with_headers,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

/// Open `id` for `tenant`, run one `get_state` so the session is genuinely started and persisted,
/// and hand back the still-attached socket.
async fn open(svc: &Service, tenant: &str, id: &str) -> common::TestWs {
    let token = svc.token(tenant, id);
    let mut ws = ws_connect_with_headers(svc.port, Some(id), &svc.header(&token))
        .await
        .unwrap_or_else(|status| panic!("{tenant}/{id} refused with {status}"));
    ws_send(&mut ws, json!({"type":"get_state","id":"warm"})).await;
    let frames = ws_read_until_response(&mut ws, "get_state").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    ws
}

fn ids(response: &Value, key: &str) -> Vec<String> {
    response["data"][key]
        .as_array()
        .unwrap_or_else(|| panic!("no {key} in {response}"))
        .iter()
        .filter_map(|s| s["id"].as_str().map(str::to_string))
        .collect()
}

async fn listing(ws: &mut common::TestWs, command: &str, key: &str) -> Vec<String> {
    ws_send(ws, json!({"type": command, "id": command})).await;
    let frames = ws_read_until_response(ws, command).await;
    let response = frames.last().unwrap().clone();
    assert_eq!(response["success"], true, "{response}");
    ids(&response, key)
}

/// A tenant's sessions can be spread across mounts; a listing has to find them all and nobody
/// else's. `list_sessions` stays scoped to this session's own shard, `list_all_sessions` merges
/// every mounted shard, and `list_daemon_sessions` — answered by the supervisor, which sees the
/// whole replica — is scoped the same way.
#[tokio::test]
async fn listings_are_tenant_scoped_across_every_mounted_shard() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1", "s2"]).await;

    // Tenant A on both shards, tenant B on the first.
    let _a1 = open(&svc, "tenant-a", "s1.a-one").await;
    let _a2 = open(&svc, "tenant-a", "s2.a-two").await;
    let mut b1 = open(&svc, "tenant-b", "s1.b-one").await;
    // A fresh connection for A, so the listing isn't answered by a session that happens to be first.
    let mut a = open(&svc, "tenant-a", "s1.a-one").await;

    let mut all = listing(&mut a, "list_all_sessions", "sessions").await;
    all.sort();
    assert_eq!(all, vec!["s1.a-one", "s2.a-two"], "A sees both of its own");

    let own = listing(&mut a, "list_sessions", "sessions").await;
    assert_eq!(
        own,
        vec!["s1.a-one"],
        "list_sessions stays on this session's shard"
    );

    let mut daemon = listing(&mut a, "list_daemon_sessions", "sessions").await;
    daemon.sort();
    assert_eq!(daemon, vec!["s1.a-one", "s2.a-two"]);

    // And B sees only its own, from the same replica and the same shard as A's first session.
    let b_all = listing(&mut b1, "list_all_sessions", "sessions").await;
    assert_eq!(b_all, vec!["s1.b-one"]);
    let b_daemon = listing(&mut b1, "list_daemon_sessions", "sessions").await;
    assert_eq!(b_daemon, vec!["s1.b-one"]);
}

/// A missing id is an error, not a silent success. The repo's unique-prefix fallback is a
/// convenience for an id a human typed; a tenant's ids are minted, so "no such session" has to be
/// reported rather than resolved to a neighbour — or, for `delete`, swallowed as a no-op.
#[tokio::test]
async fn deleting_a_session_that_is_not_there_is_an_error() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1", "s2"]).await;
    let _neighbour = open(&svc, "tenant-a", "s2.keep-me").await;
    let mut a = open(&svc, "tenant-a", "s1.alpha").await;

    // On another (mounted) shard, but no such session.
    ws_send(
        &mut a,
        json!({"type":"delete_session","id":"d1","session_id":"s2.keep-me-not"}),
    )
    .await;
    let frames = ws_read_until_response(&mut a, "delete_session").await;
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false, "{response}");

    // On a shard this replica doesn't mount at all.
    ws_send(
        &mut a,
        json!({"type":"delete_session","id":"d2","session_id":"s9.elsewhere"}),
    )
    .await;
    let frames = ws_read_until_response(&mut a, "delete_session").await;
    let response = frames.last().unwrap();
    assert_eq!(response["success"], false, "{response}");
    assert!(
        response["error"]
            .as_str()
            .unwrap_or_default()
            .contains("shard"),
        "{response}"
    );

    // A prefix of a real id resolves to nothing rather than to the session it prefixes.
    ws_send(
        &mut a,
        json!({"type":"delete_session","id":"d3","session_id":"s2.keep"}),
    )
    .await;
    let frames = ws_read_until_response(&mut a, "delete_session").await;
    assert_eq!(frames.last().unwrap()["success"], false);
    assert!(svc.sessions_dir("s2", "tenant-a").is_dir());
    assert!(
        std::fs::read_dir(svc.sessions_dir("s2", "tenant-a"))
            .unwrap()
            .any(|e| e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("s2.keep-me")),
        "the neighbouring session must still be there"
    );
}

/// A session on another shard *does* delete, and lands in that shard's own trash.
#[tokio::test]
async fn a_session_on_another_shard_deletes_and_restores_by_its_own_shard() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1", "s2"]).await;
    // Open and close it, so nothing holds it while it is deleted.
    drop(open(&svc, "tenant-a", "s2.doomed").await);
    let mut a = open(&svc, "tenant-a", "s1.alpha").await;

    ws_send(
        &mut a,
        json!({"type":"delete_session","id":"d1","session_id":"s2.doomed"}),
    )
    .await;
    let frames = ws_read_until_response(&mut a, "delete_session").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    assert!(svc.sessions_dir("s2", "tenant-a").join(".trash").is_dir());

    // `list_trash` merges every mounted shard's trash, so the entry is visible from this session…
    ws_send(&mut a, json!({"type":"list_trash","id":"t1"})).await;
    let frames = ws_read_until_response(&mut a, "list_trash").await;
    let response = frames.last().unwrap();
    assert_eq!(response["success"], true, "{response}");
    let trash = response["data"]["trash"].as_array().unwrap();
    assert_eq!(trash.len(), 1, "{response}");
    assert_eq!(trash[0]["id"], "s2.doomed");
    // …with no replica mount path in it.
    let original = trash[0]["original_path"].as_str().unwrap();
    assert!(
        !original.contains(&svc.shard("s2").to_string_lossy().into_owned()),
        "the mount path leaked: {original}"
    );

    // …and restoring routes back to that shard.
    ws_send(
        &mut a,
        json!({"type":"restore_session","id":"r1","session_id":"s2.doomed"}),
    )
    .await;
    let frames = ws_read_until_response(&mut a, "restore_session").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");
    let restored = listing(&mut a, "list_all_sessions", "sessions").await;
    assert!(restored.contains(&"s2.doomed".to_string()), "{restored:?}");
}

/// `fork`/`clone`/`new_session`/`switch_session` self-abort a running prompt to make room for
/// themselves. In service mode they are refused — so the refusal has to happen *before* that arm, or
/// a client could kill its own run by asking for something it was never going to get.
#[tokio::test]
async fn a_refused_command_during_a_run_does_not_abort_it() {
    let base = spawn_model_server_with_stalled_response(
        vec![],
        std::time::Duration::from_millis(900),
        vec![],
    );
    let svc = Service::start(&base, &["s1"]).await;
    let token = svc.token("tenant-a", "s1.alpha");
    let mut ws = ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&token))
        .await
        .unwrap();

    ws_send(&mut ws, json!({"type":"prompt","id":"p1","message":"go"})).await;
    // Wait for the ack, so the run is provably in flight.
    loop {
        let frame = common::ws_next_frame(&mut ws).await.expect("an ack");
        if frame["type"] == "ack" && frame["command"] == "prompt" {
            break;
        }
    }

    for command in ["fork", "clone", "new_session", "switch_session", "reload"] {
        ws_send(&mut ws, json!({"type": command, "id": command})).await;
    }

    // Every refusal arrives, and the run still completes on its own.
    let mut refused = Vec::new();
    let mut prompt = None;
    while prompt.is_none() {
        let frame = common::ws_next_frame(&mut ws).await.expect("frames");
        if frame["type"] != "response" {
            continue;
        }
        if frame["command"] == "prompt" {
            prompt = Some(frame);
        } else {
            assert_eq!(frame["success"], false, "{frame}");
            refused.push(frame["command"].as_str().unwrap_or_default().to_string());
        }
    }
    refused.sort();
    assert_eq!(
        refused,
        vec!["clone", "fork", "new_session", "reload", "switch_session"]
    );
    let prompt = prompt.unwrap();
    assert_eq!(
        prompt["success"], true,
        "a refused command must not abort the run: {prompt}"
    );
    assert_eq!(
        prompt["data"]["assistant_messages"], 1,
        "the stalled turn completed normally rather than being cancelled: {prompt}"
    );
}

/// Re-attaching to a live session needs its own valid grant for the *same* tenant and session —
/// and, given one, lands on the same conversation rather than a fresh one.
#[tokio::test]
async fn a_second_connection_with_its_own_grant_re_attaches_to_the_same_session() {
    let (base, _requests) = spawn_model_server(vec![]);
    let svc = Service::start(&base, &["s1"]).await;
    let mut first = open(&svc, "tenant-a", "s1.alpha").await;
    ws_send(
        &mut first,
        json!({"type":"set_session_name","id":"n1","title":"the same session"}),
    )
    .await;
    let frames = ws_read_until_response(&mut first, "set_session_name").await;
    assert_eq!(frames.last().unwrap()["success"], true, "{frames:#?}");

    // A different grant (fresh ephemeral key and nonce), same tenant and id.
    let second_token = svc.token("tenant-a", "s1.alpha");
    let mut second =
        ws_connect_with_headers(svc.port, Some("s1.alpha"), &svc.header(&second_token))
            .await
            .expect("a second grant for the same session attaches");
    ws_send(&mut second, json!({"type":"get_state","id":"g1"})).await;
    let frames = ws_read_until_response(&mut second, "get_state").await;
    let state = &frames.last().unwrap()["data"];
    assert_eq!(state["title"], "the same session", "{state}");
}
