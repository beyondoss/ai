//! What a service session leaves on the mount, and what a derived one is called.
//!
//! A shard is shared storage: every tenant on it writes into the same filesystem, and an operator,
//! a backup, or a neighbouring task can read the bytes. So the rule is that nothing a tenant said —
//! its transcript, the listing cache built from it, or either memory mount — is readable without
//! that tenant's own key, while everything stays readable *with* it. And an id a session derives
//! carries its shard, because an id is the only address a client ever holds.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use beyond_ai_agent::memory::{MemPath, MemoryBackend, View};
use common::service::{Service, files_under};
use common::{
    TestWs, spawn_model_server, turn_text, turn_tool_use, ws_connect_with_headers,
    ws_read_until_response, ws_send,
};
use serde_json::{Value, json};

/// Anything a tenant typed or was told. None of it may appear in a file on the mount.
const TRANSCRIPT_MARKER: &str = "PLAINTEXT-TRANSCRIPT-MARKER";
const REPLY_MARKER: &str = "PLAINTEXT-REPLY-MARKER";
const TITLE_MARKER: &str = "PLAINTEXT-TITLE-MARKER";
const MEMORY_MARKER: &str = "PLAINTEXT-MEMORY-MARKER";

async fn open(svc: &Service, tenant: &str, id: &str) -> TestWs {
    let token = svc.token(tenant, id);
    ws_connect_with_headers(svc.port, Some(id), &svc.header(&token))
        .await
        .unwrap_or_else(|status| panic!("{tenant}/{id} refused with {status}"))
}

async fn command(ws: &mut TestWs, command: &str, body: Value) -> Value {
    let mut frame = json!({"type": command, "id": command});
    if let (Value::Object(f), Value::Object(b)) = (&mut frame, body) {
        f.extend(b);
    }
    ws_send(ws, frame).await;
    let frames = ws_read_until_response(ws, command).await;
    let response = frames.last().unwrap().clone();
    assert_eq!(response["success"], true, "{response}");
    response
}

/// Everything on the mount is sealed: the segments, the `.listings.json` cache beside them, and the
/// session's own `memory/` directory. The plaintext that remains is deliberate — the segment headers,
/// which a reader has to be able to parse to bound a file it holds no key for.
#[tokio::test]
async fn nothing_a_tenant_wrote_is_readable_on_the_mount_without_its_key() {
    let (base, _requests) = spawn_model_server(vec![
        turn_tool_use(
            "m1",
            "memory",
            &json!({
                "command": "create",
                "path": "/session/scratch.md",
                "file_text": format!("{MEMORY_MARKER}\n"),
            })
            .to_string(),
        ),
        turn_text(REPLY_MARKER),
    ]);
    let svc = Service::start(&base, &["s1"]).await;
    let mut ws = open(&svc, "00tenant1", "s1.alpha").await;

    command(&mut ws, "prompt", json!({ "message": TRANSCRIPT_MARKER })).await;
    command(
        &mut ws,
        "set_session_name",
        json!({ "title": TITLE_MARKER }),
    )
    .await;
    // Builds `.listings.json`, which caches the title, a preview and the session's search text.
    command(&mut ws, "list_sessions", json!({})).await;

    let sessions = svc.sessions_dir("s1", "00tenant1");
    let session = sessions.join("s1.alpha");
    assert!(
        session.is_dir(),
        "{} must be a directory",
        session.display()
    );
    assert!(
        sessions.join(".listings.json").is_file(),
        "the listing cache must have been written"
    );
    assert!(
        session.join("memory").is_dir(),
        "the /session mount lives inside the session directory"
    );

    let files = files_under(&sessions);
    assert!(files.len() >= 3, "{files:?}");
    for file in &files {
        let raw = std::fs::read(file).unwrap();
        for marker in [TRANSCRIPT_MARKER, REPLY_MARKER, TITLE_MARKER, MEMORY_MARKER] {
            assert!(
                !raw.windows(marker.len()).any(|w| w == marker.as_bytes()),
                "{} holds {marker} in plaintext",
                file.display()
            );
        }
    }

    // And all of it opens with the tenant's own key — sealed, not lost.
    let (store, session_state) = svc
        .tenant_repo("s1", "00tenant1")
        .open_id_read_only("s1.alpha")
        .unwrap();
    assert_eq!(store.meta().title.as_deref(), Some(TITLE_MARKER));
    let transcript = serde_json::to_string(&*session_state.messages).unwrap();
    assert!(transcript.contains(TRANSCRIPT_MARKER), "{transcript}");
    assert!(transcript.contains(REPLY_MARKER), "{transcript}");

    let View::Document(text) =
        beyond_ai_agent::memory::file::FileBackend::session_at(store.memory_dir())
            .sealed(svc.codec("00tenant1"))
            .view(
                &MemPath::parse_in("/session/scratch.md", "/session").unwrap(),
                None,
            )
            .await
            .unwrap()
    else {
        panic!("expected a /session document");
    };
    assert_eq!(text, format!("{MEMORY_MARKER}\n"));
}

/// A `fork` mints `<shard>.<opaque>`, so the id a client is handed back routes on its own — and the
/// session that minted it keeps running, because its id is what its slot, its lock and its grant are
/// all keyed on.
#[tokio::test]
async fn a_forked_id_carries_the_shard_and_opens_with_a_grant_of_its_own() {
    let (base, _requests) = spawn_model_server(vec![turn_text("answered once")]);
    let svc = Service::start(&base, &["s1", "s2"]).await;
    let mut ws = open(&svc, "00tenant1", "s2.parent").await;
    command(&mut ws, "prompt", json!({ "message": "remember this" })).await;

    let forked = command(&mut ws, "fork", json!({})).await;
    let new_id = forked["data"]["session_id"].as_str().unwrap().to_string();
    assert_eq!(forked["data"]["switched"], false, "{forked}");
    assert!(
        new_id.starts_with("s2."),
        "a derived id must stay on its parent's shard: {new_id}"
    );
    assert_ne!(new_id, "s2.parent");

    // The live view did not move.
    let state = command(&mut ws, "get_state", json!({})).await;
    assert_eq!(state["data"]["session_id"], "s2.parent", "{state}");

    // …and the fork landed on that shard, addressable with a grant of its own.
    assert!(svc.sessions_dir("s2", "00tenant1").join(&new_id).is_dir());
    let mut child = open(&svc, "00tenant1", &new_id).await;
    let child_state = command(&mut child, "get_state", json!({})).await;
    assert_eq!(child_state["data"]["session_id"], new_id.as_str());
    ws_send(&mut child, json!({"type":"get_messages","id":"m1"})).await;
    let frames = ws_read_until_response(&mut child, "get_messages").await;
    let messages = serde_json::to_string(&frames.last().unwrap()["data"]).unwrap();
    assert!(messages.contains("answered once"), "{messages}");

    // A grant for the parent's *other* shard says nothing about this id: the tenant's listing still
    // finds both, because an id names its own mount.
    let listed = command(&mut ws, "list_all_sessions", json!({})).await;
    let ids: Vec<&str> = listed["data"]["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s["id"].as_str())
        .collect();
    assert!(ids.contains(&"s2.parent"), "{ids:?}");
    assert!(ids.contains(&new_id.as_str()), "{ids:?}");
}

/// `new_session` on an addressed session archives the outgoing conversation and keeps the id — the
/// address a client is routing on must not move underneath it. The archive is the derived one, and
/// so carries the shard.
#[tokio::test]
async fn new_session_archives_onto_the_shard_and_keeps_this_sessions_id() {
    let (base, _requests) = spawn_model_server(vec![turn_text("the old conversation")]);
    let svc = Service::start(&base, &["s1"]).await;
    let mut ws = open(&svc, "00tenant1", "s1.alpha").await;
    command(&mut ws, "prompt", json!({ "message": "hello" })).await;

    let fresh = command(&mut ws, "new_session", json!({})).await;
    assert_eq!(
        fresh["data"]["session_id"], "s1.alpha",
        "an addressed session keeps its id: {fresh}"
    );

    let listed = command(&mut ws, "list_sessions", json!({})).await;
    let ids: Vec<String> = listed["data"]["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s["id"].as_str().map(str::to_string))
        .collect();
    assert_eq!(ids.len(), 2, "the archive is a session of its own: {ids:?}");
    let archive = ids.iter().find(|id| *id != "s1.alpha").unwrap();
    assert!(
        archive.starts_with("s1."),
        "the archive must carry the shard: {archive}"
    );

    // The archive holds the outgoing conversation; this session is blank again.
    let mut archived = open(&svc, "00tenant1", archive).await;
    ws_send(&mut archived, json!({"type":"get_messages","id":"m1"})).await;
    let frames = ws_read_until_response(&mut archived, "get_messages").await;
    let messages = serde_json::to_string(&frames.last().unwrap()["data"]).unwrap();
    assert!(messages.contains("the old conversation"), "{messages}");
}
